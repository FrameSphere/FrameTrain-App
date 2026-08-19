"""Test-Plugin: Audioklassifikation mit einem trainierten HuggingFace-Modell."""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import (
    load_label_names, resolve_device, run_dataset_classification,
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.extractor = None
        self.model = None
        self.device = None
        self.sampling_rate = 16000
        self.max_seconds = float(config.plugin_config.get("max_seconds", 10.0))
        self.id2label: Dict[int, str] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")

        TestProtocol.status("loading", "Lade Audiomodell...")
        self.extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
        self.model = AutoModelForAudioClassification.from_pretrained(str(model_path))
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)
        self.sampling_rate = int(getattr(self.extractor, "sampling_rate", 16000) or 16000)
        self.id2label = load_label_names(model_path, self.model.config)
        TestProtocol.status(
            "loading",
            f"Modell geladen | Klassen: {len(self.id2label) or '?'} | "
            f"{self.sampling_rate} Hz | Gerät: {self.device}",
        )

    def _predict(self, path: Path) -> Tuple[str, float, List[Dict[str, Any]]]:
        import librosa
        import torch

        wave, _ = librosa.load(str(path), sr=self.sampling_rate, mono=True)
        max_len = int(self.max_seconds * self.sampling_rate)
        if len(wave) > max_len:
            wave = wave[:max_len]
        inputs = self.extractor(
            [wave], sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        k = min(3, probs.numel())
        top = torch.topk(probs, k)
        preds = [
            {"label": self.id2label.get(int(i), str(int(i))), "score": float(s)}
            for s, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        return preds[0]["label"], preds[0]["score"], preds

    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Audiodatei nicht gefunden: {path}")
        t0 = time.time()
        predicted, confidence, top = self._predict(path)
        TestProtocol.complete_single(
            predicted=predicted, confidence=confidence,
            top_predictions=top, inference_time=time.time() - t0,
        )

    def run_dataset(self):
        run_dataset_classification(self, AUDIO_EXTS, self._predict, "Audiodateien")
