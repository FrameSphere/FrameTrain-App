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
    preprocess: Dict[str, Any] = field(default_factory=dict)

    def _tabular(self, x: Any) -> Any:
        """Normalisierung aus dem Training anwenden."""
        mean, std = self.preprocess.get("mean"), self.preprocess.get("std")
        if mean is None or x.dim() != 2 or x.size(-1) != len(mean):
            return x
        import torch
        return (x - torch.tensor(mean, device=x.device)) / torch.tensor(std, device=x.device)

    def _label(self, idx: int) -> str:
        names = self.preprocess.get("class_names") or []
        return names[idx] if idx < len(names) else f"Klasse {idx}"

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
        x = x if x.is_floating_point() or x.dtype == torch.long else x.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)  # Batch-Dimension hinzufügen

        x = self._tabular(x.to(self.device))
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
        # Hat das Modell mehr Ausgaenge als das Dataset Klassen, zaehlen nur echte Klassen.
        known = len(self.preprocess.get("class_names") or [])
        if known and logits.size(-1) > known:
            logits = logits[..., :known]
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]

        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        confidence = float(probs[pred_idx])

        top_n = min(5, len(probs))
        top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]

        return {
            "predicted_class": pred_idx,
            "predicted_label": self._label(pred_idx),
            "confidence": confidence,
            # "label" ergänzt — der Lab-Server/UI zeigt top_predictions über 'label'.
            "top_predictions": [
                {"class_idx": i, "label": self._label(i), "score": round(float(probs[i]), 6)}
                for i in top
            ],
            "all_probs": [round(float(p), 6) for p in probs],
            "inference_ms": round(inference_ms, 2),
            "task_type": "classification",
        }

    def generate_images(self, count: int = 4) -> Any:
        """GAN und Diffusion: neue Beispiele erzeugen."""
        import torch
        from synapse.generative import NoisePlan
        self.model.eval()
        if self.task_type == "gan":
            with torch.no_grad():
                return self.model(torch.randn(count, self.model.latent, device=self.device))
        return NoisePlan().sample(self.model, (count, *self.preprocess["sample_shape"]), self.device)

    def predict_boxes(self, path: str, threshold: float = 0.3) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image
        from synapse.generative import decode_boxes
        spec = self.preprocess["pipeline"]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB").resize((spec["size"], spec["size"])), dtype="float32") / 255.0
        x = ((torch.from_numpy(arr).permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).to(self.device)
        self.model.eval()
        t0 = time.time()
        with torch.no_grad():
            boxes = decode_boxes(self.model(x), threshold)
        return {"predicted": f"{len(boxes)} Objekte", "boxes": boxes,
                "inference_ms": (time.time() - t0) * 1000, "task_type": "detection"}

    def predict_file(self, path: str) -> Dict[str, Any]:
        """Bild, Audio oder Video mit derselben Vorverarbeitung wie im Training."""
        spec = self.preprocess.get("pipeline") or {}
        if spec.get("kind") == "boxes":
            return self.predict_boxes(path)
        if spec.get("kind") in ("audio", "video"):
            from synapse.data import load_file
            return self.predict(load_file(path, spec).unsqueeze(0))
        return self.predict_image(path)

    def predict_text(self, text: str) -> Dict[str, Any]:
        import torch
        from synapse.data import encode
        tok = self.preprocess.get("tokenizer")
        if not tok:
            raise ValueError("Dieses Modell wurde nicht mit Text trainiert. Es erwartet Zahlen oder eine Datei.")
        if self.task_type == "seq2seq":
            from synapse.data import translate
            self.model.eval()
            t0 = time.time()
            text = translate(self.model, tok, text, self.device)
            return {"predicted": text, "inference_ms": (time.time() - t0) * 1000, "task_type": "seq2seq"}
        if self.task_type in ("language_model", "generation"):
            from synapse.data import generate
            self.model.eval()
            t0 = time.time()
            text = generate(self.model, tok, text, self.device)
            return {"predicted": text, "inference_ms": (time.time() - t0) * 1000, "task_type": "language_model"}
        return self.predict(torch.tensor([encode(text, tok)], dtype=torch.long))

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
            normalize = p.get("normalize", False) is True or str(p.get("normalize")).lower() == "true"
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

        if normalize and channels == 1:
            tensor = (tensor - 0.5) / 0.5
        elif normalize:
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

    # ── 3. state_dict laden ───────────────────────────────────────────────────
    pt_path = _find_model_pt(model_dir)
    if pt_path is None:
        raise FileNotFoundError(
            f"Keine model.pt in '{model_dir}' gefunden.\n"
            f"Bitte zuerst ein Training durchführen."
        )

    checkpoint = torch.load(str(pt_path), map_location=device, weights_only=False)
    # auto-Groessen stammen aus der Datenform des Trainings.
    input_shape = (checkpoint.get("preprocess") or {}).get("input_shape")
    input_dtype = getattr(torch, (checkpoint.get("preprocess") or {}).get("input_dtype", "float32"))
    if isinstance(input_shape, dict):
        input_shape = {k: tuple(v) for k, v in input_shape.items()}
    elif input_shape:
        input_shape = tuple(input_shape)
    model = build_model_from_graph(ir, input_shape or None, input_dtype)
    # Das beste Modell des Laufs sagt vorher, nicht die letzte Epoche.
    state_dict = checkpoint.get("best_state_dict") or checkpoint.get("model_state_dict", checkpoint)
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
        preprocess=checkpoint.get("preprocess") or {},
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
