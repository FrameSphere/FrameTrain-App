"""Test-Plugin: Bildklassifikation mit einem torchvision-Modell aus FrameTrain.

Das Train-Plugin image_classification speichert model.pt als Checkpoint
({model_state_dict, arch, classes, image_size}). Bisher gab es dazu kein
Test-Plugin — Tests dieser Versionen endeten mit "Kein Test-Plugin".
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device, run_dataset_classification

# Muss mit dem Train-Manifest uebereinstimmen (test_plugin_consistency.py).
SUPPORTED_ARCHITECTURES = {
    "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b4", "vit_b_16",
    "mobilenet_v3_small", "mobilenet_v3_large",
}


def build_backbone(arch: str, num_classes: int):
    """Gleiche Koepfe wie im Training, ohne vortrainierte Gewichte (die kommen aus model.pt)."""
    import torch.nn as nn
    try:
        from torchvision import models
    except ImportError:
        from ft_data.deps import missing
        raise missing("torchvision", what="Bildklassifikation (torchvision)")
    a = arch.lower()
    if a not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architektur '{arch}' wird nicht unterstützt: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}")
    m = getattr(models, a)(weights=None)
    if a.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif a.startswith("efficientnet"):
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif a == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


def find_checkpoint(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    for name in ("model.pt", "best_model.pt"):
        if (model_path / name).exists():
            return model_path / name
    pts = sorted(model_path.glob("*.pt"))
    if pts:
        return pts[0]
    raise FileNotFoundError(f"Kein model.pt in {model_path}")


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.device = None
        self.transform = None
        self.classes: List[str] = []
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        import torch
        from torchvision import transforms

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Bildmodell (torchvision)...")
        ckpt = torch.load(str(find_checkpoint(model_path)), map_location="cpu", weights_only=False)
        self.classes = [str(c) for c in ckpt.get("classes") or []]
        arch = str(ckpt.get("arch") or "resnet18")
        size = int(ckpt.get("image_size") or 224)
        if not self.classes:
            raise ValueError("model.pt enthält keine Klassenliste (classes).")
        self.model = build_backbone(arch, len(self.classes))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        TestProtocol.status("loading", f"Modell geladen | {arch} | Klassen: {len(self.classes)} | Gerät: {self.device}")

    def _predict(self, path: Path) -> Tuple[str, float, List[Dict[str, Any]]]:
        import torch
        from PIL import Image

        with Image.open(path) as im:
            x = self.transform(im.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x)[0], dim=-1)
        top = torch.topk(probs, min(5, probs.numel()))
        # "confidence" zusaetzlich zu "score": die Test-Ansicht dieses Plugins liest confidence.
        preds = [{"label": self.classes[int(i)], "score": float(s), "confidence": float(s)}
                 for s, i in zip(top.values.tolist(), top.indices.tolist())]
        return preds[0]["label"], preds[0]["score"], preds

    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Bilddatei nicht gefunden: {path}")
        t0 = time.time()
        predicted, confidence, top = self._predict(path)
        TestProtocol.complete_single(predicted=predicted, confidence=confidence,
                                     top_predictions=top, inference_time=time.time() - t0)

    def run_dataset(self):
        run_dataset_classification(self, "image", self._predict, "Bilddateien")
