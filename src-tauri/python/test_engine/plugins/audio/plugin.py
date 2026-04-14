"""
plugins/audio/plugin.py
=======================
Test-Plugin für Audio-Modelle (Whisper ASR, wav2vec2, Audio-Klassifikation).

Dataset-Struktur:
  dataset_path/test/<audio>.wav  + transcripts.txt | labels.csv
  dataset_path/test/<klasse>/<audio>.wav  (für Audio-Klassifikation)

Single-Modus:
  Erwartet absoluten Pfad zu einer Audio-Datei (.wav, .mp3, .flac, .ogg)

Metriken ASR:      WER (Word Error Rate), CER (Character Error Rate)
Metriken Klassif.: Accuracy, Avg. Confidence
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


class Plugin(TestPlugin):

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.processor = None
        self.pipeline = None
        self.audio_mode: str = "asr"  # "asr" | "classification"
        self.class_names: List[str] = []

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.device = self.get_device()
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.proto.status("init", f"Audio-Plugin | device={self.device}")

    # ── Modell laden ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        import torch
        from transformers import pipeline, AutoConfig

        model_path = self.config.model_path
        self.proto.status("loading", f"Lade Audio-Modell: {Path(model_path).name} …")

        # Architektur bestimmen
        try:
            hf_cfg = AutoConfig.from_pretrained(model_path)
            archs = getattr(hf_cfg, "architectures", []) or []
            arch_str = " ".join(archs).lower()

            if "forsequenceclassification" in arch_str or "foraudioxvector" in arch_str:
                self.audio_mode = "classification"
            else:
                self.audio_mode = "asr"
        except Exception:
            task = self.config.task_type.lower()
            self.audio_mode = "classification" if "classification" in task else "asr"

        # task_type aus config kann auch überschreiben
        if self.config.task_type.lower() == "audio_classification":
            self.audio_mode = "classification"
        elif self.config.task_type.lower() in ("asr", "speech_recognition"):
            self.audio_mode = "asr"

        device_id = 0 if str(self.device) == "cuda" else -1

        if self.audio_mode == "asr":
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                device=device_id,
            )
            self.proto.status("loaded", "ASR-Pipeline geladen")
        else:
            self.pipeline = pipeline(
                "audio-classification",
                model=model_path,
                device=device_id,
            )
            self.proto.status("loaded", "Audio-Klassifikations-Pipeline geladen")

    # ── Audio laden ────────────────────────────────────────────────────────

    def _load_audio(self, path: str):
        """Lädt Audio als numpy-Array mit 16kHz."""
        try:
            import librosa
            audio, _ = librosa.load(path, sr=16000, mono=True)
            return audio
        except ImportError:
            pass

        try:
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                # Einfaches Resampling via scipy
                from scipy.signal import resample
                target_len = int(len(audio) * 16000 / sr)
                audio = resample(audio, target_len)
            return audio.astype("float32")
        except Exception:
            pass

        raise ImportError(
            "Bitte librosa oder soundfile installieren:\n"
            "  pip install librosa  oder  pip install soundfile"
        )

    # ── Einzelne Datei inferieren ──────────────────────────────────────────

    def _infer_audio(self, audio_path: str, expected: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()

        audio = self._load_audio(audio_path)
        result = self.pipeline(audio)
        inference_time = time.time() - t0

        if self.audio_mode == "asr":
            transcript = result.get("text", "") if isinstance(result, dict) else str(result)
            is_correct = False
            wer = None
            if expected:
                wer = self._compute_wer(expected, transcript)
                is_correct = wer < 0.05  # < 5% WER als "korrekt"

            return {
                "input_path": audio_path,
                "predicted_output": transcript.strip(),
                "expected_output": expected,
                "is_correct": is_correct,
                "wer": wer,
                "cer": self._compute_cer(expected, transcript) if expected else None,
                "confidence": None,
                "inference_time": inference_time,
                "error_type": None,
            }

        else:  # classification
            if isinstance(result, list):
                top = sorted(result, key=lambda x: x.get("score", 0), reverse=True)
            else:
                top = [result]

            predicted_label = top[0].get("label", "unknown") if top else "unknown"
            confidence = top[0].get("score") if top else None
            is_correct = (
                predicted_label.lower() == (expected or "").lower()
                if expected else False
            )

            return {
                "input_path": audio_path,
                "predicted_output": predicted_label,
                "expected_output": expected,
                "top_predictions": top[:5],
                "is_correct": is_correct,
                "confidence": confidence,
                "inference_time": inference_time,
                "error_type": None,
            }

    # ── WER / CER ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_wer(reference: str, hypothesis: str) -> float:
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref:
            return 0.0 if not hyp else 1.0
        # Levenshtein auf Wort-Ebene
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[len(ref)][len(hyp)] / len(ref)

    @staticmethod
    def _compute_cer(reference: str, hypothesis: str) -> float:
        ref = reference.lower().replace(" ", "")
        hyp = hypothesis.lower().replace(" ", "")
        if not ref:
            return 0.0 if not hyp else 1.0
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[len(ref)][len(hyp)] / len(ref)

    # ── Dataset sammeln ────────────────────────────────────────────────────

    def _collect_samples(self) -> List[Dict[str, Any]]:
        root = Path(self.config.dataset_path)
        test_dir = root / "test"
        if not test_dir.exists():
            test_dir = root / "val"
        if not test_dir.exists():
            test_dir = root

        samples: List[Dict[str, Any]] = []

        # Variante A: Klassen-Ordner (Klassifikation)
        class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if class_dirs:
            for cls_dir in sorted(class_dirs):
                for af in sorted(cls_dir.rglob("*")):
                    if af.suffix.lower() in AUDIO_EXTENSIONS:
                        samples.append({"input": str(af), "expected": cls_dir.name})
        else:
            # Variante B: Alle Dateien + transcripts.txt oder labels.csv
            label_map: Dict[str, str] = {}

            t_file = test_dir / "transcripts.txt"
            if t_file.exists():
                for line in t_file.read_text().splitlines():
                    if "\t" in line:
                        parts = line.split("\t", 1)
                        label_map[parts[0].strip()] = parts[1].strip()

            c_file = test_dir / "labels.csv"
            if c_file.exists():
                with open(c_file, newline="") as f:
                    for row in csv.DictReader(f):
                        fn = row.get("filename") or row.get("file") or ""
                        lb = row.get("label") or row.get("transcript") or ""
                        if fn:
                            label_map[fn] = lb

            for af in sorted(test_dir.rglob("*")):
                if af.suffix.lower() in AUDIO_EXTENSIONS:
                    label = label_map.get(af.name) or label_map.get(af.stem)
                    samples.append({"input": str(af), "expected": label})

        if not samples:
            raise ValueError(f"Keine Audio-Dateien gefunden in: {test_dir}")

        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        self.proto.status("loaded", f"{len(samples)} Audio-Dateien geladen")
        return samples

    # ── Dataset-Modus ─────────────────────────────────────────────────────

    def run_dataset(self) -> Dict[str, Any]:
        samples = self._collect_samples()
        self.proto.status("testing", f"Teste {len(samples)} Audio-Dateien …")

        results: List[Dict[str, Any]] = []
        t_start = time.time()

        for i, s in enumerate(samples):
            if self.is_stopped:
                break
            try:
                r = self._infer_audio(s["input"], s.get("expected"))
            except Exception as e:
                r = {
                    "input_path": s["input"],
                    "predicted_output": "error",
                    "expected_output": s.get("expected"),
                    "is_correct": False,
                    "inference_time": 0.0,
                    "error_type": str(e),
                }
            r["sample_id"] = i
            results.append(r)

            elapsed = time.time() - t_start
            sps = (i + 1) / elapsed if elapsed > 0 else 0.0
            self.proto.progress(i + 1, len(samples), sps)

        total = len(results)
        has_labels = any(r.get("expected_output") for r in results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        times = [r["inference_time"] for r in results]
        elapsed_total = time.time() - t_start

        metrics: Dict[str, Any] = {
            "total_samples": total,
            "average_inference_time": (sum(times) / len(times)) if times else 0.0,
            "samples_per_second": total / elapsed_total if elapsed_total > 0 else 0.0,
            "total_time": elapsed_total,
            "average_loss": None,
        }

        if self.audio_mode == "asr":
            wers = [r["wer"] for r in results if r.get("wer") is not None]
            cers = [r["cer"] for r in results if r.get("cer") is not None]
            metrics["average_wer"] = (sum(wers) / len(wers)) if wers else None
            metrics["average_cer"] = (sum(cers) / len(cers)) if cers else None
            metrics["accuracy"] = (1 - metrics["average_wer"]) * 100 if metrics["average_wer"] is not None else None
        else:
            metrics["accuracy"] = (correct / total * 100) if (total > 0 and has_labels) else None
            metrics["correct_predictions"] = correct if has_labels else None
            metrics["incorrect_predictions"] = (total - correct) if has_labels else None

        out = Path(self.config.output_path)
        results_file = out / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": results}, f, indent=2, default=str)

        hard = [r for r in results if not r.get("is_correct", True)]
        hard_file = None
        if hard:
            hard_file = str(out / "hard_examples.jsonl")
            with open(hard_file, "w", encoding="utf-8") as f:
                for ex in hard:
                    f.write(json.dumps(ex, default=str) + "\n")

        return {
            "metrics": metrics,
            "predictions": results,
            "results_file": str(results_file),
            "hard_examples_file": hard_file,
            "total_samples": total,
            "task_type": "audio",
        }

    # ── Single-Modus ───────────────────────────────────────────────────────

    def run_single(self, input_data: str) -> Dict[str, Any]:
        self.proto.status("inferring", f"Verarbeite Audio: {Path(input_data).name} …")
        r = self._infer_audio(input_data, expected=None)

        if self.audio_mode == "asr":
            return {
                "transcript": r["predicted_output"],
                "confidence": r.get("confidence"),
                "inference_time": r["inference_time"],
                "mode": "asr",
            }
        else:
            return {
                "predicted_label": r["predicted_output"],
                "top_predictions": r.get("top_predictions", []),
                "confidence": r.get("confidence"),
                "inference_time": r["inference_time"],
                "mode": "classification",
            }
