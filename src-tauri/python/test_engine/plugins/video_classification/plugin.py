"""Test-Plugin: Videoklassifikation mit einem trainierten HuggingFace-Modell."""
import json
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


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.device = None
        self.num_frames = 16
        self.id2label: Dict[int, str] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoImageProcessor, AutoModelForVideoClassification

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")

        TestProtocol.status("loading", "Lade Videomodell...")
        self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForVideoClassification.from_pretrained(str(model_path))
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)
        self.num_frames = int(getattr(self.model.config, "num_frames", 16) or 16)
        mapping = model_path / "label_mapping.json"
        if mapping.exists():
            try:
                self.num_frames = int(json.loads(mapping.read_text(encoding="utf-8")).get("num_frames") or self.num_frames)
            except Exception:
                pass
        self.id2label = load_label_names(model_path, self.model.config)
        TestProtocol.status(
            "loading",
            f"Modell geladen | Klassen: {len(self.id2label) or '?'} | "
            f"{self.num_frames} Bilder je Clip | Gerät: {self.device}",
        )

    def _predict(self, path: Path) -> Tuple[str, float, List[Dict[str, Any]]]:
        import torch
        from ft_data.video import read_clip_frames

        frames = read_clip_frames(path, self.num_frames)
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
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
            raise ValueError(f"Videodatei nicht gefunden: {path}")
        t0 = time.time()
        predicted, confidence, top = self._predict(path)
        TestProtocol.complete_single(
            predicted=predicted, confidence=confidence,
            top_predictions=top, inference_time=time.time() - t0,
        )

    def run_dataset(self):
        run_dataset_classification(self, "video", self._predict, "Videodateien")
