"""Test-Plugin: Objekterkennung mit Ultralytics-YOLO.

Bis 1.8.0 gab es fuer task_type "detect" kein Test-Plugin: Tests von YOLO-
und Synapse-Detektoren endeten ohne Ergebnis (Benchmark-Lauf 09.11.).

Kennzahlen kommen aus der Ultralytics-Validierung auf dem Test-Split
(sonst val): mAP50, mAP50-95, Precision, Recall. Je Bild wird ausserdem
vorhergesagt; ein Bild gilt als "richtig", wenn die erkannten Klassen
(mit Anzahl) den Labels entsprechen.
"""
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"test": ["test", "testing"], "val": ["val", "valid", "validation"]}


def find_weights(model_dir: Path) -> Path:
    """Gewichte im Versions- oder Modellordner (model.pt, best.pt, yolov8n.pt ...)."""
    if model_dir.is_file() and model_dir.suffix in (".pt", ".pth"):
        return model_dir
    for rel in ["model.pt", "best.pt", "weights/best.pt", "train/weights/best.pt", "last.pt"]:
        if (model_dir / rel).exists():
            return model_dir / rel
    meta = model_dir / "model.json"
    if meta.exists():
        try:
            base = json.loads(meta.read_text(encoding="utf-8")).get("base_model")
            if base and Path(base).exists():
                return Path(base)
        except Exception:
            pass
    found = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_size) or sorted(model_dir.rglob("*.pt"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Keine .pt-Gewichte in {model_dir} gefunden.")


def split_dir(root: Path, kind: str) -> Optional[Path]:
    for name in SPLIT_NAMES[kind]:
        d = root / name
        if (d / "images").is_dir():
            return d
        if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
            return d
    for name in SPLIT_NAMES[kind]:
        if (root / "images" / name).is_dir():
            return root / "images" / name
    return None


def images_in(d: Path) -> List[Path]:
    base = d / "images" if (d / "images").is_dir() else d
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_for(image: Path) -> Optional[Path]:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = image.with_suffix(".txt")
    return cand if cand.exists() else None


def describe(counter: Counter) -> str:
    if not counter:
        return "(nichts)"
    return ", ".join(f"{n}x {name}" for name, n in sorted(counter.items()))


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.names: Dict[int, str] = {}
        self.is_stopped = False
        self.conf = float(config.get_plugin_value("conf", 0.25))

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from ultralytics import YOLO

        weights = find_weights(Path(self.config.model_path))
        TestProtocol.status("loading", f"Lade YOLO-Gewichte {weights.name}...")
        self.model = YOLO(str(weights))
        names = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        TestProtocol.status("loading", f"Modell geladen | Klassen: {len(self.names)}")

    # ── Einzelbild ────────────────────────────────────────────────────────────
    def _detect(self, path: Path) -> List[Dict[str, Any]]:
        result = self.model.predict(str(path), conf=self.conf, verbose=False)[0]
        out = []
        for box in getattr(result, "boxes", None) or []:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
            out.append({"label": self.names.get(cls_id, str(cls_id)), "confidence": round(float(box.conf), 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return out

    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Bilddatei nicht gefunden: {path}")
        t0 = time.time()
        dets = self._detect(path)
        counts = Counter(d["label"] for d in dets)
        best = max(dets, key=lambda d: d["confidence"]) if dets else None
        TestProtocol.complete_single(
            predicted=describe(counts),
            confidence=best["confidence"] if best else None,
            top_predictions=[{"label": k, "score": max(d["confidence"] for d in dets if d["label"] == k)} for k in counts],
            inference_time=time.time() - t0,
            extra={"detections": dets},
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    def _data_yaml(self, root: Path, split: Path, split_key: str) -> Path:
        """Vorhandene data.yaml nutzen, sonst eine kleine fuer den Split schreiben."""
        import yaml

        for name in ["data.yaml", "dataset.yaml"]:
            p = root / name
            if p.exists():
                try:
                    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    if d.get(split_key):
                        d["path"] = str(root)
                        # Label-IDs beziehen sich auf die Klassen des Modells. Die
                        # Namen in der yaml koennen Platzhalter sein (coco128 ohne
                        # Klassenliste: 'KlasseA'/'KlasseB', nc 2) - dann verwirft
                        # Ultralytics jedes Label mit hoeherer ID.
                        d["names"] = {i: n for i, n in self.names.items()}
                        d.pop("nc", None)
                        tmp = Path(tempfile.mkdtemp()) / "data.yaml"
                        tmp.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
                        return tmp
                except Exception:
                    pass
        rel = split / "images" if (split / "images").is_dir() else split
        d = {"path": str(root), "train": str(rel), "val": str(rel), "test": str(rel),
             "names": {i: n for i, n in self.names.items()}}
        tmp = Path(tempfile.mkdtemp()) / "data.yaml"
        tmp.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
        return tmp

    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split = split_dir(root, "test")
        split_key = "test"
        if split is None:
            split, split_key = split_dir(root, "val"), "val"
            if split is not None:
                TestProtocol.warning("Kein Test-Split gefunden; ausgewertet wird der Validierungs-Split.")
        if split is None:
            raise ValueError(
                f"Kein test/ oder val/ Ordner mit Bildern in '{root}' gefunden. "
                "Teile das Dataset zuerst auf (datasets_split) oder lege test/images und test/labels an."
            )
        images = images_in(split)
        if self.config.max_samples:
            images = images[: int(self.config.max_samples)]
        if not images:
            raise ValueError(f"Keine Bilder in '{split}'.")

        TestProtocol.status("running", f"{len(images)} Bilder werden ausgewertet ({split_key})...")
        rows: List[Dict[str, Any]] = []
        correct = 0
        total_time = 0.0
        started = time.time()
        for idx, img in enumerate(images, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            dets = self._detect(img)
            dt = time.time() - t0
            total_time += dt
            expected = Counter()
            lbl = label_for(img)
            if lbl is not None:
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            expected[self.names.get(int(float(parts[0])), parts[0])] += 1
                        except ValueError:
                            pass
            predicted = Counter(d["label"] for d in dets)
            ok = predicted == expected
            correct += int(ok)
            rows.append({
                "sample_id": idx, "input_text": img.name, "input_path": str(img),
                "expected_output": describe(expected), "predicted_output": describe(predicted),
                "is_correct": ok, "confidence": max((d["confidence"] for d in dets), default=None),
                "inference_time": dt, "detections": dets,
            })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(images), sps=idx / elapsed)

        # Tempo nur fuer die Vorhersagen; die mAP-Berechnung danach zaehlt nicht mit.
        elapsed = max(time.time() - started, 1e-6)

        # mAP ueber die Ultralytics-Validierung (nur ohne Stichprobe exakt auf denselben Bildern)
        metrics: Dict[str, Any] = {}
        try:
            TestProtocol.status("running", "Berechne mAP...")
            data = self._data_yaml(root, split, split_key)
            out = Path(self.config.output_path)
            res = self.model.val(data=str(data), split=split_key, conf=0.001, plots=False, verbose=False,
                                 project=str(out), name="val", exist_ok=True, batch=max(1, int(self.config.batch_size or 16)))
            rd = getattr(res, "results_dict", {}) or {}
            # Reihenfolge zaehlt: die Startseite zeigt die erste Kennzahl (mAP50).
            metrics = {
                "mAP50": float(rd.get("metrics/mAP50(B)", 0.0)),
                "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(rd.get("metrics/precision(B)", 0.0)),
                "recall": float(rd.get("metrics/recall(B)", 0.0)),
                "images_exact": correct / len(rows),
            }
        except Exception as e:  # mAP ist ein Zusatz; die Vorhersagen je Bild bleiben gueltig
            TestProtocol.warning(f"mAP konnte nicht berechnet werden: {e}")
            metrics = {"images_exact": correct / len(rows)}

        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": rows, "metrics": metrics}, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        hard = [r for r in rows if not r["is_correct"]]
        hard_file = out_dir / "hard_examples.json"
        hard_file.write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")

        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(rows),
            # Keine "Accuracy" fuer Detektoren: mAP50 ist die Kennzahl, die zaehlt.
            accuracy=None, correct=correct, average_loss=None,
            average_inference_time=total_time / len(rows), samples_per_second=len(rows) / elapsed,
            hard_examples_file=str(hard_file), metrics=metrics,
        )
