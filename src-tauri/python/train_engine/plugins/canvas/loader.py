"""
loader.py — Canvas Model Loader
================================
Zentrale API um trainierte Synapse-Modelle wiederzuverwenden.
Basiert ausschließlich auf gespeicherten Dateien (model.pt + graph_metadata.json).
Benötigt keine Canvas-Session, kein localStorage, keine laufende App.

Öffentliche API:
    load_canvas_model(model_dir)  →  LoadedCanvasModel
    LoadedCanvasModel.predict(x)  →  dict
    LoadedCanvasModel.predict_batch(xs) → list[dict]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)


@dataclass
class LoadedCanvasModel:
    """Wiederverwendbares Canvas-Modell – geladen aus Dateisystem."""
    model: Any                              # DynamicGraphModule (nn.Module)
    ir: Any                                 # CanvasGraphIR
    device: str
    training_history: Dict[str, Any] = field(default_factory=dict)
    num_classes: int = 10
    task_type: str = "classification"

    def predict(self, x: Any) -> Dict[str, Any]:
        """
        Einzelne Vorhersage.
        x: torch.Tensor (bereits auf richtigem Device) ODER Python-Liste/float/int
        Gibt zurück: { predicted_class, confidence, top_predictions, inference_ms }
        """
        import torch
        import torch.nn.functional as F

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # Batch-Dimension hinzufügen

        x = x.to(self.device)
        t0 = time.time()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        inference_ms = (time.time() - t0) * 1000

        if self.task_type == "regression":
            val = logits.squeeze().cpu().tolist()
            return {
                "predicted_value": val,
                "inference_ms": inference_ms,
                "task_type": "regression",
            }

        # Klassifikation
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]

        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        confidence = float(probs[pred_idx])

        top_n = min(5, len(probs))
        top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]

        return {
            "predicted_class": pred_idx,
            "confidence": confidence,
            # "label" ergänzt — der Lab-Server/UI zeigt top_predictions über 'label'.
            # Canvas-Modelle kennen keine Klassennamen → Klassen-Index als Label.
            "top_predictions": [
                {"class_idx": i, "label": f"Klasse {i}", "score": round(float(probs[i]), 6)}
                for i in top
            ],
            "all_probs": [round(float(p), 6) for p in probs],
            "inference_ms": round(inference_ms, 2),
            "task_type": "classification",
        }

    def predict_batch(self, xs: List[Any]) -> List[Dict[str, Any]]:
        """Mehrere Vorhersagen auf einmal."""
        return [self.predict(x) for x in xs]

    # ── Bild-Inferenz ─────────────────────────────────────────────────────────
    def _image_spec(self) -> Dict[str, Any]:
        """Bildgröße/Kanäle/Normalisierung aus dem IR ableiten (image_loader-Node).
        Fallback: Kanäle aus erstem conv2d-Layer, sonst 3×224×224 ohne Normalisierung."""
        size, channels, normalize = 224, 3, False
        data = getattr(self.ir, "data", None)
        if data is not None and getattr(data, "type", "") == "image_loader":
            p = data.params or {}
            try:
                size = int(p.get("imageSize", size))
            except Exception:
                pass
            try:
                channels = int(p.get("channels", channels))
            except Exception:
                pass
            normalize = bool(p.get("normalize", False))
        else:
            # Kanäle aus erstem Conv2D-Layer schätzen
            for n in getattr(self.ir, "nodes", []):
                if getattr(n, "type", "") == "conv2d":
                    try:
                        channels = int((n.params or {}).get("inChannels", channels))
                    except Exception:
                        pass
                    break
        return {"size": size, "channels": channels, "normalize": normalize}

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Lädt eine Bilddatei, bereitet sie nach IR-Spezifikation auf und sagt vorher."""
        import torch
        from PIL import Image

        spec = self._image_spec()
        size, channels, normalize = spec["size"], spec["channels"], spec["normalize"]

        img = Image.open(image_path)
        img = img.convert("L" if channels == 1 else "RGB")
        img = img.resize((size, size))

        # PIL → Tensor [C, H, W] in [0, 1]
        import numpy as np
        arr = np.asarray(img, dtype="float32") / 255.0
        if arr.ndim == 2:  # Graustufen
            arr = arr[:, :, None]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C, H, W]

        if normalize and channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        tensor = tensor.unsqueeze(0)  # Batch-Dim [1, C, H, W]
        return self.predict(tensor)


def load_canvas_model(
    model_dir: Union[str, Path],
    device: Optional[str] = None,
) -> LoadedCanvasModel:
    """
    Lädt ein trainiertes Canvas-Modell vollständig aus dem Dateisystem.

    Reihenfolge:
      1. graph_metadata.json lesen → graphIR extrahieren
      2. parse_ir() → CanvasGraphIR
      3. build_model_from_graph() → DynamicGraphModule
      4. model.pt lesen → model_state_dict laden
      5. LoadedCanvasModel zurückgeben

    Falls kein IR in graph_metadata.json: versucht graph_ir aus model.pt.
    Falls beides fehlt: klarer Fehler mit Hinweis.

    Args:
        model_dir:  Pfad zum Modell-Ordner (enthält graph_metadata.json + model.pt)
        device:     Optional "cpu" | "cuda" | "mps" – sonst automatisch erkannt

    Returns:
        LoadedCanvasModel mit .predict() und .predict_batch()
    """
    import torch
    from ir import parse_ir
    from model_builder import build_model_from_graph

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Modell-Ordner nicht gefunden: {model_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )

    # ── 1. IR laden ──────────────────────────────────────────────────────────
    # Primäre Quelle: graph_metadata.json (immer vorhanden nach create_canvas_network_model)
    raw_ir = _load_ir_from_metadata(model_dir)

    # Fallback: IR direkt aus model.pt (seit Fix 1.3 gespeichert)
    if raw_ir is None:
        raw_ir = _load_ir_from_checkpoint(model_dir)

    if raw_ir is None:
        raise ValueError(
            f"Kein Graph-IR gefunden in '{model_dir}'.\n"
            f"Erwartet: graph_metadata.json (Feld 'graphIR') oder model.pt (Feld 'graph_ir').\n"
            f"Lösung: Modell einmal neu aus dem Synapse-Builder exportieren."
        )

    # ── 2. Parse + Build ──────────────────────────────────────────────────────
    ir = parse_ir(raw_ir)
    model = build_model_from_graph(ir)

    # ── 3. state_dict laden ───────────────────────────────────────────────────
    pt_path = _find_model_pt(model_dir)
    if pt_path is None:
        raise FileNotFoundError(
            f"Keine model.pt in '{model_dir}' gefunden.\n"
            f"Bitte zuerst ein Training durchführen."
        )

    checkpoint = torch.load(str(pt_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        mismatch = str(e)
        raise RuntimeError(
            f"Gewichte passen nicht zur Modell-Architektur.\n"
            f"Das deutet darauf hin, dass graph_metadata.json und model.pt "
            f"nicht zusammengehören (z.B. nach einer Architektur-Änderung ohne erneutes Training).\n"
            f"Lösung: Modell erneut aus dem Synapse-Builder exportieren und neu trainieren.\n"
            f"PyTorch-Detail: {mismatch}"
        ) from e
    model.to(device)
    model.eval()

    training_history = checkpoint.get("training_history", {})
    num_classes = int(raw_ir.get("training", {}).get("numClasses", 10))
    task_type = str(raw_ir.get("training", {}).get("taskType", "classification"))

    return LoadedCanvasModel(
        model=model,
        ir=ir,
        device=device,
        training_history=training_history,
        num_classes=num_classes,
        task_type=task_type,
    )


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _load_ir_from_metadata(model_dir: Path) -> Optional[Dict]:
    """Liest graphIR aus graph_metadata.json."""
    meta_path = model_dir / "graph_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Feld kann "graphIR" oder "graph_ir" heißen (verschiedene Versionen)
        ir = meta.get("graphIR") or meta.get("graph_ir")
        if ir and isinstance(ir, dict) and ir.get("nodes"):
            return ir
    except Exception:
        pass
    return None


def _load_ir_from_checkpoint(model_dir: Path) -> Optional[Dict]:
    """Liest graph_ir aus model.pt (seit Fix 1.3 mit vollständigen Nodes/Edges)."""
    import torch
    pt_path = _find_model_pt(model_dir)
    if pt_path is None:
        return None
    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        ir = ckpt.get("graph_ir")
        # Nur verwenden wenn Nodes vorhanden (nicht die alte Metadaten-Only-Version)
        if ir and isinstance(ir, dict) and ir.get("nodes"):
            return ir
    except Exception:
        pass
    return None


def _find_model_pt(model_dir: Path) -> Optional[Path]:
    """Findet model.pt oder beste verfügbare Checkpoint-Datei."""
    # Direkter Treffer
    for name in ["model.pt", "model_best.pt", "checkpoint.pt"]:
        p = model_dir / name
        if p.exists():
            return p
    # In Unterordnern suchen (Versions-Struktur)
    for p in sorted(model_dir.glob("**/*.pt")):
        return p  # Ersten Treffer nehmen
    return None
