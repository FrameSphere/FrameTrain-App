#!/usr/bin/env python3
"""
FrameTrain - Persistent Model Server
=====================================
Bleibt als Hintergrundprozess am Leben und beantwortet Inferenz-Anfragen
via stdin/stdout JSON-Protokoll.

Protokoll:
  Rust -> Python (stdin):   {"text": "..."}\n                (Text / Seq2Seq)
                            {"file_path": "/pfad/bild.png"}\n (Bild / Audio)
  Python -> Rust (stdout):  {"predicted": "...", "confidence": 0.95, ...}\n

Startup:
  Python -> Rust:  {"type": "ready", "modality": "text|image|audio|seq2seq",
                    "input_kind": "text|image|audio"}\n
  Python -> Rust:  {"type": "error", "message": "..."}\n  (bei Fehler)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Unbuffered line-by-line stdout (kritisch fuer IPC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def emit(obj: dict):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def emit_error(message: str):
    emit({"type": "error", "message": message})


# Modell-Typen, die eine Audio-Wellenform statt Text erwarten
AUDIO_MODEL_TYPES = {
    "wav2vec2", "wav2vec2-conformer", "hubert", "wavlm", "unispeech",
    "unispeech-sat", "sew", "sew-d", "whisper", "audio-spectrogram-transformer",
    "ast", "data2vec-audio", "speech-to-text", "clap",
}

# Modell-Typen, die ein Bild erwarten
IMAGE_MODEL_TYPES = {
    "resnet", "vit", "deit", "beit", "convnext", "convnextv2", "swin", "swinv2",
    "efficientnet", "mobilenet_v1", "mobilenet_v2", "mobilevit", "regnet",
    "levit", "poolformer", "segformer", "dinov2", "cvt", "van", "bit",
}


def detect_modality(model_cfg: dict) -> str:
    """Bestimmt aus config.json, welche Auto-Klasse und welche Eingabe passt.

    Rueckgabe: "text" | "image" | "audio" | "seq2seq"
    """
    archs = [a for a in (model_cfg.get("architectures") or []) if isinstance(a, str)]
    arch = archs[0] if archs else ""
    model_type = str(model_cfg.get("model_type", "")).lower()

    if arch.endswith("ForImageClassification"):
        return "image"
    if arch.endswith("ForAudioClassification") or arch.endswith("ForAudioFrameClassification"):
        return "audio"
    if arch.endswith("ForConditionalGeneration") or arch.endswith("ForSeq2SeqLM"):
        return "seq2seq"
    if arch.endswith("ForSequenceClassification"):
        # wav2vec2 & Co. melden ForSequenceClassification, erwarten aber Audio
        if model_type in AUDIO_MODEL_TYPES:
            return "audio"
        if model_type in IMAGE_MODEL_TYPES:
            return "image"
        return "text"

    # Kein (bekannter) Architektur-Eintrag: ueber model_type entscheiden
    if model_type in AUDIO_MODEL_TYPES:
        return "audio"
    if model_type in IMAGE_MODEL_TYPES:
        return "image"
    if model_cfg.get("is_encoder_decoder"):
        return "seq2seq"
    return "text"


# Wie im Audio-Test-Plugin: laengere Aufnahmen werden gekappt
MAX_AUDIO_SECONDS = 10.0

INPUT_KIND = {
    "text":    "text",
    "seq2seq": "text",
    "image":   "image",
    "audio":   "audio",
}


class ModelServer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer  = None   # Text / Seq2Seq
        self.processor  = None   # Bild / Audio
        self.model      = None
        self.id2label   = {}
        self.device     = None
        self.modality   = "text"
        self.sampling_rate = 16000
        self._torch     = None
        self._np        = None

    # ── Laden ────────────────────────────────────────────────────────────

    def load(self):
        try:
            import torch
            import numpy as np
        except ImportError as e:
            raise ImportError(
                f"Fehlende Pakete: {e}. Installiere: pip install torch transformers"
            )

        self._torch = torch
        self._np    = np

        model_cfg = self._read_config()

        self.id2label = self._load_labels(model_cfg)

        self.modality = detect_modality(model_cfg)

        # Geraet waehlen (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        loader = {
            "image":   self._load_image,
            "audio":   self._load_audio,
            "seq2seq": self._load_seq2seq,
            "text":    self._load_text,
        }[self.modality]
        loader()

        self.model.to(self.device)
        self.model.eval()

    def _load_labels(self, model_cfg: dict) -> dict:
        """Klassennamen aus label_mapping.json (id2label ODER classes) bzw. config.json.

        Bild- und Audio-Training schreiben nur {"classes": [...]}, Text-Training
        schreibt id2label — beide Formen muessen echte Namen liefern.
        """
        label_map_file = self.model_path / "label_mapping.json"
        if label_map_file.exists():
            try:
                with open(label_map_file, "r", encoding="utf-8") as f:
                    lm = json.load(f)
            except (json.JSONDecodeError, OSError):
                lm = {}
            raw = lm.get("id2label")
            if isinstance(raw, dict) and raw:
                return {int(k): v for k, v in raw.items()}
            classes = lm.get("classes")
            if isinstance(classes, list) and classes:
                return {i: str(c) for i, c in enumerate(classes)}

        raw = model_cfg.get("id2label") or {}
        return {int(k): v for k, v in raw.items()}

    def _read_config(self) -> dict:
        cfg_file = self.model_path / "config.json"
        if not cfg_file.exists():
            # Klare Diagnose statt nackter Meldung: Canvas-Modelle sind kein
            # HF-Format und können hier grundsätzlich nicht geladen werden.
            if (self.model_path / "graph_metadata.json").exists() or \
               (self.model_path / "canvas_model.py").exists() or \
               (self.model_path / "model.pt").exists():
                raise FileNotFoundError(
                    "Canvas-Modell: Lab-Inferenz unterstützt nur HuggingFace-"
                    "Modelle. Canvas-Modelle im Synapse Builder "
                    "→ Inference-Tab testen."
                )
            found = ", ".join(sorted(p.name for p in self.model_path.iterdir())[:8]) or "(leer)"
            raise FileNotFoundError(
                f"Keine config.json in: {self.model_path}\n"
                f"Vorhandene Dateien: {found}\n"
                "Erwartet wird ein HuggingFace-Modellordner (config.json + Gewichte + Tokenizer)."
            )

        with open(cfg_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_text(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    def _load_seq2seq(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    def _load_image(self):
        from transformers import AutoModelForImageClassification
        try:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
        except Exception:
            from transformers import AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
        self.model = AutoModelForImageClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    def _load_audio(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.processor = AutoFeatureExtractor.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.sampling_rate = int(getattr(self.processor, "sampling_rate", 16000) or 16000)
        self.model = AutoModelForAudioClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )

    # ── Inferenz ─────────────────────────────────────────────────────────

    def infer(self, req: dict) -> dict:
        if self.modality == "image":
            return self._infer_image(self._require_file(req, "Bild"))
        if self.modality == "audio":
            return self._infer_audio(self._require_file(req, "Audio"))
        if self.modality == "seq2seq":
            return self._infer_seq2seq(self._require_text(req))
        return self._infer_text(self._require_text(req))

    def _require_text(self, req: dict) -> str:
        text = req.get("text") or ""
        if not str(text).strip():
            raise ValueError("Kein Text in der Anfrage")
        return str(text)

    def _require_file(self, req: dict, kind: str) -> Path:
        raw = req.get("file_path") or req.get("path") or ""
        if not str(raw).strip():
            raise ValueError(
                f"Dieses Modell erwartet eine {kind}-Datei, es kam aber nur Text an. "
                f"Lade im Labor {kind}-Samples aus einem Dataset."
            )
        p = Path(str(raw))
        if not p.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {p}")
        return p

    def _classify(self, inputs: dict, t0: float) -> dict:
        torch = self._torch
        np    = self._np

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        inference_time = time.time() - t0

        if isinstance(probs, float):
            probs = [probs]

        pred_id    = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        predicted  = self.id2label.get(pred_id, str(pred_id))

        top_n = min(5, len(probs))
        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
        top_predictions = [
            {"label": self.id2label.get(i, str(i)), "score": float(probs[i])}
            for i in sorted_ids
        ]

        return {
            "predicted":       predicted,
            "confidence":      confidence,
            "top_predictions": top_predictions,
            "inference_time":  inference_time,
        }

    def _infer_text(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._classify(inputs, time.time())

    def _infer_image(self, path: Path) -> dict:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow fehlt. Installiere: pip install pillow")

        with Image.open(path) as img:
            img = img.convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._classify(inputs, time.time())

    def _infer_audio(self, path: Path) -> dict:
        waveform = self._read_audio(path)
        inputs = self.processor(
            waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._classify(inputs, time.time())

    def _read_audio(self, path: Path):
        """Laedt eine Audiodatei als Mono-Wellenform in der Modell-Samplerate.

        Gleiche Vorverarbeitung wie das Test-Plugin (librosa, 10s-Kappung),
        damit Labor und Testlauf beim selben Sample dasselbe Ergebnis liefern.
        """
        np = self._np
        data = None

        try:
            import librosa
            data, _ = librosa.load(str(path), sr=self.sampling_rate, mono=True)
        except ImportError:
            pass

        if data is None:
            try:
                import soundfile as sf
                raw, sr = sf.read(str(path), dtype="float32", always_2d=True)
                data = raw.mean(axis=1)
            except ImportError:
                try:
                    import torchaudio
                except ImportError:
                    raise ImportError(
                        "Zum Laden von Audio fehlt librosa, soundfile oder torchaudio. "
                        "Installiere: pip install librosa"
                    )
                tensor, sr = torchaudio.load(str(path))
                data = tensor.mean(dim=0).numpy()

            if sr != self.sampling_rate:
                # Lineare Interpolation — ausreichend fuer Einzel-Inferenz
                duration = data.shape[0] / float(sr)
                target_len = max(1, int(round(duration * self.sampling_rate)))
                data = np.interp(
                    np.linspace(0.0, data.shape[0] - 1, target_len),
                    np.arange(data.shape[0]),
                    data,
                )

        max_len = int(MAX_AUDIO_SECONDS * self.sampling_rate)
        if data.shape[0] > max_len:
            data = data[:max_len]

        return data.astype("float32")

    def _infer_seq2seq(self, text: str) -> dict:
        torch = self._torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=128)
        inference_time = time.time() - t0

        output = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

        # Bewusst ohne confidence/top_predictions: bei generiertem Text gaebe es
        # keine ehrliche Klassen-Wahrscheinlichkeit.
        return {
            "predicted":      output,
            "inference_time": inference_time,
        }

    # ── Loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            self.load()
        except Exception as e:
            emit_error(str(e))
            sys.exit(1)

        # Bereit-Signal an Rust (mit Modalitaet, damit das Labor die richtige
        # Eingabeart verlangt)
        emit({
            "type":       "ready",
            "modality":   self.modality,
            "input_kind": INPUT_KIND[self.modality],
        })

        # Request-Loop: eine JSON-Zeile rein, eine JSON-Zeile raus
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError as e:
                emit_error(f"JSON parse error: {e}")
                continue

            if req.get("cmd") == "shutdown":
                break

            try:
                emit(self.infer(req))
            except Exception as e:
                emit_error(f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FrameTrain Persistent Model Server")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        emit_error(f"Modell-Pfad nicht gefunden: {args.model_path}")
        sys.exit(1)

    ModelServer(args.model_path).run()


if __name__ == "__main__":
    main()
