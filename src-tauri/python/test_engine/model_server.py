#!/usr/bin/env python3
"""
FrameTrain - Persistent Model Server
=====================================
Bleibt als Hintergrundprozess am Leben und beantwortet Inferenz-Anfragen
via stdin/stdout JSON-Protokoll.

Protokoll:
  Rust -> Python (stdin):   {"text": "..."}\n                (Text / Seq2Seq / LLM / Token / Embedding / Text-to-Image)
                            {"file_path": "/pfad/bild.png"}\n (Bild / Audio / ASR)
                            {"file_path": "/v.mp4", "start": 2.0, "end": 6.0}\n (Video)
                            {"file_path": "/bild.png", "question": "..."}\n (VLM)
  Python -> Rust (stdout):  {"predicted": "...", "confidence": 0.95, ...}\n

Startup:
  Python -> Rust:  {"type": "ready", "modality": "text|image|audio|seq2seq|asr|video|causal_lm|token|embedding|vlm|text_to_image",
                    "input_kind": "text|image|audio"}\n
  Python -> Rust:  {"type": "error", "message": "..."}\n  (bei Fehler)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ft_data (gemeinsame Helfer fuer Video, VLM, Diffusion) liegt eine Ebene hoeher.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Unbuffered line-by-line stdout (kritisch fuer IPC) + UTF-8 erzwingen.
# Auf Windows ist stdout/stderr per Default cp1252 (charmap); ein Emoji oder
# tqdm-Unicode-Balken laesst den print sonst mit UnicodeEncodeError sterben.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


# Videoklassifikatoren (VideoMAE, TimeSformer, ViViT)
VIDEO_MODEL_TYPES = {"videomae", "timesformer", "vivit"}

# Spracherkennung: Audio rein, Text raus. Whisper meldet sich als
# ...ForConditionalGeneration und wurde deshalb frueher als Text-Seq2Seq
# geladen — und scheiterte dann an der ersten Anfrage ohne Text.
ASR_MODEL_TYPES = {"whisper", "speech_to_text", "speech-encoder-decoder", "moonshine"}


def detect_modality(model_cfg: dict) -> str:
    """Bestimmt aus config.json, welche Auto-Klasse und welche Eingabe passt.

    Rueckgabe: "text" | "image" | "audio" | "seq2seq" | "asr" | "video" | "causal_lm" | "token" | "vlm"
    ("embedding" erkennt detect_modality_for_dir an modules.json — die
    config.json eines Sentence-Transformers sagt nur "BertModel". "text_to_image"
    erkennt is_diffusion_model an model_index.json.)
    """
    archs = [a for a in (model_cfg.get("architectures") or []) if isinstance(a, str)]
    arch = archs[0] if archs else ""
    model_type = str(model_cfg.get("model_type", "")).lower()

    # Vor dem ForConditionalGeneration-Zweig: SmolVLM, BLIP & Co. melden sich
    # genauso wie T5 und wurden sonst als Text-Seq2Seq ohne Bild geladen.
    if model_type not in ASR_MODEL_TYPES and _is_vlm_config(model_cfg):
        return "vlm"
    if arch.endswith("ForImageClassification"):
        return "image"
    if arch.endswith("ForAudioClassification") or arch.endswith("ForAudioFrameClassification"):
        return "audio"
    if arch.endswith("ForVideoClassification"):
        return "video"
    if arch.endswith("ForTokenClassification"):
        # Vorher landete ein NER-Modell ueber model_type "bert" bei der
        # Sequenzklassifikation — mit zufaellig initialisiertem Kopf.
        return "token"
    if arch.endswith("ForCTC") or arch.endswith("ForSpeechSeq2Seq"):
        return "asr"
    if model_type in ASR_MODEL_TYPES and (not arch or arch.endswith("ForConditionalGeneration")):
        return "asr"
    if arch.endswith("ForConditionalGeneration") or arch.endswith("ForSeq2SeqLM"):
        return "seq2seq"
    if arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel"):
        # Decoder-LLMs (Llama, Qwen, GPT-2 …). Frueher fielen sie ans Ende
        # auf "text" und wurden als Klassifikator mit Zufallskopf geladen.
        return "causal_lm"
    if arch.endswith("ForSequenceClassification"):
        # wav2vec2 & Co. melden ForSequenceClassification, erwarten aber Audio
        if model_type in AUDIO_MODEL_TYPES:
            return "audio"
        if model_type in IMAGE_MODEL_TYPES:
            return "image"
        return "text"

    # Kein (bekannter) Architektur-Eintrag: ueber model_type entscheiden
    if model_type in VIDEO_MODEL_TYPES:
        return "video"
    if model_type in AUDIO_MODEL_TYPES:
        return "audio"
    if model_type in IMAGE_MODEL_TYPES:
        return "image"
    if model_cfg.get("is_encoder_decoder"):
        return "seq2seq"
    return "text"


def detect_modality_for_dir(model_path: Path, model_cfg: dict) -> str:
    """Wie detect_modality, prueft aber zusaetzlich die Dateien im Ordner.

    Ein Sentence-Transformers-Modell (modules.json / sentence_bert_config.json)
    ist ein Embedding-Modell — sein config.json nennt nur das Basismodell.
    """
    if (model_path / "modules.json").exists() or (model_path / "sentence_bert_config.json").exists():
        return "embedding"
    return detect_modality(model_cfg)
def _is_vlm_config(model_cfg: dict) -> bool:
    try:
        from ft_data.vlm import is_vlm_config
    except ImportError:
        return False
    return is_vlm_config(model_cfg)


def is_diffusion_model(model_path: Path) -> bool:
    """Diffusers-Pipeline (model_index.json) oder LoRA-Export von text_to_image_lora."""
    p = Path(model_path)
    if (p / "config.json").exists():
        return False
    return (p / "model_index.json").exists() or (p / "text_to_image_lora.json").exists()


# Wie im Audio-Test-Plugin: laengere Aufnahmen werden gekappt
MAX_AUDIO_SECONDS = 10.0

INPUT_KIND = {
    "text":    "text",
    "seq2seq": "text",
    "causal_lm": "text",
    "image":   "image",
    "audio":   "audio",
    "asr":     "audio",
    "video":   "video",
    "token":   "text",
    "embedding": "text",
    # VLM: Bild plus optionale Frage ("question"/"text"); ohne Frage gilt der
    # Standardprompt aus dem Training.
    "vlm":     "image",
    # Prompt rein, Pfad des erzeugten PNG raus.
    "text_to_image": "text",
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
            from ft_data.deps import install_hint
            raise ImportError(f"Fehlende Pakete: {e}.\n\n" + install_hint("torch", "transformers"))

        self._torch = torch
        self._np    = np

        # Geraet waehlen (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Diffusion-Pipelines haben keine config.json im Wurzelordner, sondern
        # model_index.json + Unterordner — sie laufen ganz ueber ft_data.diffusion.
        if is_diffusion_model(self.model_path):
            self.modality = "text_to_image"
            self._load_text_to_image()
            return

        model_cfg = self._read_config()

        self.id2label = self._load_labels(model_cfg)

        self.modality = detect_modality_for_dir(self.model_path, model_cfg)

        loader = {
            "image":   self._load_image,
            "audio":   self._load_audio,
            "asr":     self._load_asr,
            "video":   self._load_video,
            "token":   self._load_token,
            "embedding": self._load_embedding,
            "vlm":     self._load_vlm,
            "seq2seq": self._load_seq2seq,
            "causal_lm": self._load_causal_lm,
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

    def _load_causal_lm(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ft_data import llm as llm_data
        self._llm = llm_data
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Basismodelle ohne Template bekommen dasselbe Ersatzformat wie im Training.
        llm_data.ensure_chat_template(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True)
        self.system_prompt = ""
        meta = self.model_path / "frametrain_llm.json"
        if meta.exists():
            try:
                self.system_prompt = json.loads(meta.read_text(encoding="utf-8")).get("system_prompt", "")
            except Exception:
                pass

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

    def _load_asr(self):
        """Spracherkennung ueber die pipeline — sie zerlegt lange Aufnahmen in
        30-Sekunden-Stuecke, sonst schneidet Whisper nach 30 s einfach ab."""
        from transformers import pipeline
        dev = self.device
        device_arg = 0 if dev.type == "cuda" else ("mps" if dev.type == "mps" else -1)
        self.pipe = pipeline(
            "automatic-speech-recognition", model=str(self.model_path), device=device_arg,
        )
        self.model = self.pipe.model
        fe = getattr(self.pipe, "feature_extractor", None)
        self.sampling_rate = int(getattr(fe, "sampling_rate", 16000) or 16000)

    def _load_video(self):
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model = AutoModelForVideoClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.num_frames = int(getattr(self.model.config, "num_frames", 16) or 16)

    def _load_token(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        model_type = str(getattr(self.model.config, "model_type", "") or "")
        # BPE-Tokenizer brauchen fuer vorzerlegte Woerter add_prefix_space.
        kwargs = {"add_prefix_space": True} if model_type in {
            "roberta", "longformer", "deberta", "bart", "gpt2", "mvp", "led"} else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, **kwargs
        )
        self._tokens = self._ft_data("tokens")
        self._bio = self._tokens.is_bio_scheme(self.id2label.values())

    def _load_embedding(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            from ft_data.deps import missing
            raise missing("sentence-transformers", what="Das Embedding-Modell")
        dev = self.device.type if self.device is not None else "cpu"
        self.model = SentenceTransformer(str(self.model_path), device=dev, local_files_only=True)

    @staticmethod
    def _ft_data(name: str):
        """Gemeinsame Dataset-Logik (python/ft_data) — dieselbe wie im Test-Plugin."""
        import importlib
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module(f"ft_data.{name}")
    def _load_vlm(self):
        from ft_data.vlm import load_model
        self.model, self.processor, self.vlm_family, self.vlm_info = load_model(self.model_path, self.device)

    def _load_text_to_image(self):
        from ft_data.diffusion import default_size, load_pipeline
        self.pipe, self.t2i_info = load_pipeline(self.model_path, self.device)
        self.t2i_size = default_size(self.pipe, self.t2i_info)
        # Erzeugte Bilder liegen beim Modell, damit das Labor sie spaeter noch findet.
        self.t2i_out = self.model_path / "generated"
        self.model = self.pipe.unet

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
        if self.modality == "vlm":
            return self._infer_vlm(self._require_file(req, "Bild"), req)
        if self.modality == "text_to_image":
            if not str(req.get("text") or "").strip():
                raise ValueError("Dieses Modell erzeugt Bilder aus einem Prompt — bitte einen Text eingeben "
                                 "oder Text-Samples (Captions) laden.")
            return self._infer_text_to_image(self._require_text(req), req)
        if self.modality == "image":
            return self._infer_image(self._require_file(req, "Bild"))
        if self.modality == "audio":
            return self._infer_audio(self._require_file(req, "Audio"))
        if self.modality == "asr":
            return self._infer_asr(self._require_file(req, "Audio"))
        if self.modality == "video":
            return self._infer_video(self._require_file(req, "Video"), req.get("start"), req.get("end"))
        if self.modality == "seq2seq":
            return self._infer_seq2seq(self._require_text(req))
        if self.modality == "causal_lm":
            return self._infer_causal_lm(self._require_text(req), req)
        if self.modality == "token":
            return self._infer_token(self._require_text(req))
        if self.modality == "embedding":
            return self._infer_embedding(self._require_text(req))
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
        problem_type = getattr(self.model.config, "problem_type", None)
        if problem_type in ("multi_label_classification", "regression"):
            return self._special_result(outputs.logits, problem_type, t0)
        with torch.no_grad():
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

    def _special_result(self, logits, problem_type: str, t0: float) -> dict:
        """Multi-Label: alle Labels ueber der Schwelle (Sigmoid statt Softmax).
        Regression: der Zahlwert selbst, ohne erfundene Konfidenz."""
        torch = self._torch
        if problem_type == "regression":
            value = float(logits.reshape(-1)[0].cpu())
            return {"predicted": f"{value:.3f}", "value": value, "inference_time": time.time() - t0}
        scores = torch.sigmoid(logits).reshape(-1).cpu().tolist()
        threshold = self._threshold()
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chosen = [i for i in order if scores[i] >= threshold]
        return {
            "predicted": ", ".join(self.id2label.get(i, str(i)) for i in chosen),
            "labels": [self.id2label.get(i, str(i)) for i in chosen],
            "confidence": float(scores[order[0]]) if order else None,
            "top_predictions": [{"label": self.id2label.get(i, str(i)), "score": float(scores[i])} for i in order[:5]],
            "threshold": threshold,
            "inference_time": time.time() - t0,
        }

    def _threshold(self) -> float:
        try:
            lm = json.loads((self.model_path / "label_mapping.json").read_text(encoding="utf-8"))
            return float(lm.get("threshold", 0.5))
        except (OSError, ValueError, TypeError):
            return 0.5

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
            from ft_data.deps import missing
            raise missing("pillow", what="Die Bildeingabe")

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

    def _infer_asr(self, path: Path) -> dict:
        t0 = time.time()
        wave = self._read_audio(path, cap=False)
        out = self.pipe(
            {"raw": wave, "sampling_rate": self.sampling_rate},
            chunk_length_s=30, batch_size=4,
        )
        text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
        # Keine confidence: eine ehrliche Wahrscheinlichkeit fuer einen ganzen
        # Satz gibt es nicht, und eine erfundene wuerde "Unsicherste zuerst" verfaelschen.
        return {"predicted": text.strip(), "inference_time": time.time() - t0}

    def _infer_video(self, path: Path, start=None, end=None) -> dict:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from ft_data.video import read_clip_frames
        except ImportError as e:
            raise ImportError(f"Video-Hilfen fehlen: {e}")
        frames = read_clip_frames(path, self.num_frames, start, end)
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._classify(inputs, time.time())

    def _infer_token(self, text: str) -> dict:
        tk = self._tokens
        t0 = time.time()
        words = tk.split_words(text)
        preds = tk.predict_words(self.model, self.tokenizer, [w for w, _, _ in words],
                                 self.id2label, self.device)
        ents = tk.entities_from_tags(words, [p[0] for p in preds], [p[1] for p in preds],
                                     text=text, bio=self._bio)
        return {
            "predicted": tk.format_entities(ents) or "Keine Entitaeten gefunden",
            # Unsicherste Entitaet: danach sortiert "Unsicherste zuerst" sinnvoll.
            "confidence": min((e["score"] for e in ents), default=None),
            "top_predictions": [
                {"label": f"{e['text']} [{e['label']}]", "score": e["score"],
                 "entity": e["label"], "start": e["start"], "end": e["end"]}
                for e in ents
            ],
            "entities": ents,
            "inference_time": time.time() - t0,
        }

    def _infer_embedding(self, text: str) -> dict:
        np = self._np
        t0 = time.time()
        if "|||" in text:
            a, b = (part.strip() for part in text.split("|||", 1))
            if not a or not b:
                raise ValueError('Fuer einen Vergleich beide Saetze angeben: "Satz A ||| Satz B"')
            emb = self.model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
            sim = float(np.dot(emb[0], emb[1]))
            return {
                "predicted": f"Kosinus-Aehnlichkeit: {sim:.3f}",
                "similarity": sim,
                "inference_time": time.time() - t0,
            }
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        norm = float(np.linalg.norm(vec))
        head = ", ".join(f"{v:.3f}" for v in vec[:5])
        # Keine confidence: ein Vektor ist keine Entscheidung mit Wahrscheinlichkeit.
        return {
            "predicted": f"Vektor mit {vec.shape[0]} Dimensionen (Norm {norm:.3f}): [{head}, …]",
            "embedding_dim": int(vec.shape[0]),
            "norm": norm,
            "inference_time": time.time() - t0,
        }
    def _infer_vlm(self, path: Path, req: dict) -> dict:
        from PIL import Image
        from ft_data.vlm import DEFAULT_PROMPT, generate
        question = str(req.get("question") or req.get("prompt") or "").strip()
        if not question:
            question = str(self.vlm_info.get("default_prompt", DEFAULT_PROMPT) or "")
        with Image.open(path) as img:
            img = img.convert("RGB")
        t0 = time.time()
        max_new = int(req.get("max_new_tokens") or self.vlm_info.get("max_new_tokens") or 64)
        answer = generate(self.model, self.processor, self.vlm_family, [img], [question], max_new)[0]
        # Freier Text: keine Konfidenz (wie Seq2Seq).
        return {"predicted": answer, "question": question, "inference_time": time.time() - t0}

    def _infer_text_to_image(self, prompt: str, req: dict) -> dict:
        from ft_data.diffusion import generate_image, safe_filename
        t0 = time.time()
        seed = int(req.get("seed", 42))
        img = generate_image(
            self.pipe, prompt, steps=int(req.get("num_inference_steps") or 25),
            guidance=float(req.get("guidance_scale") or 7.5), seed=seed,
            negative_prompt=str(req.get("negative_prompt") or ""), size=self.t2i_size)
        self.t2i_out.mkdir(parents=True, exist_ok=True)
        path = self.t2i_out / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(prompt)}_{seed}.png"
        img.save(path)
        return {"predicted": str(path), "image_path": str(path), "output_kind": "image",
                "inference_time": time.time() - t0}

    def _read_audio(self, path: Path, cap: bool = True):
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
                    from ft_data.deps import install_hint
                    raise ImportError("Zum Laden von Audio fehlt librosa, soundfile oder torchaudio.\n\n"
                                      + install_hint("librosa", "soundfile"))
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
        if cap and data.shape[0] > max_len:
            data = data[:max_len]

        return data.astype("float32")

    def _infer_causal_lm(self, text: str, req: dict) -> dict:
        torch = self._torch
        msgs = [{"role": "user", "content": text}]
        if self.system_prompt:
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        prompt = self._llm.render_chat(self.tokenizer, msgs, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(req.get("max_new_tokens", 256)),
                                      do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {"predicted": answer, "inference_time": time.time() - t0}

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
