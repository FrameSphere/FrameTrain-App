"""Test-Plugin: Objekterkennung mit HuggingFace-Detektoren (task_type 'hf_object_detection').

Ergebnisaufbau wie beim YOLO-Test-Plugin, damit Benchmark-Vergleich und
Anzeige dieselben Felder lesen: je Bild "detections" mit label, confidence,
x1..y2 in Pixeln; Kennzahlen mAP50 (zuerst), mAP50-95, precision, recall,
images_exact. Die mAP rechnet ft_data.od_metrics (COCO-Definition), weil fuer
HF-Modelle keine Ultralytics-Validierung zur Verfuegung steht.

Daten: YOLO-Labels oder COCO-JSON; ausgewertet wird test/, sonst val/.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from ft_data import hf_detection as hfd
from ft_data.od_formats import load_detection_dataset
from ft_data.od_metrics import detection_metrics


def describe(counter: Counter) -> str:
    if not counter:
        return "(nichts)"
    return ", ".join(f"{n}x {name}" for name, n in sorted(counter.items()))


def model_dir_of(path: Path) -> Path:
    """Versionsordner oder ein Unterordner mit config.json (final_model/)."""
    if (path / "config.json").is_file():
        return path
    for sub in ("final_model", "model"):
        if (path / sub / "config.json").is_file():
            return path / sub
    hits = sorted(path.rglob("config.json"))
    return hits[0].parent if hits else path


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.names: Dict[int, str] = {}
        self.device = "cpu"
        self.is_stopped = False
        self.conf = float(config.get_plugin_value("conf", 0.25))
        self.imgsz = int(config.get_plugin_value("imgsz", 0) or 0)

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        mdir = model_dir_of(Path(self.config.model_path))
        problem = hfd.check_model_dir(mdir)
        if problem:
            raise ValueError(problem)
        TestProtocol.status("loading", f"Lade {hfd.SUPPORTED_MODEL_TYPES[hfd.read_model_type(mdir)]} aus {mdir.name}...")
        self.processor = AutoImageProcessor.from_pretrained(str(mdir))
        imgsz = self.imgsz
        if not imgsz:
            try:
                imgsz = int(json.loads((mdir / "model.json").read_text(encoding="utf-8")).get("imgsz") or 0)
            except (OSError, ValueError):
                imgsz = 0
        hfd.configure_processor_size(self.processor, imgsz)
        self.model = AutoModelForObjectDetection.from_pretrained(str(mdir))
        self.device = hfd.pick_device(str(self.config.get_plugin_value("device", "") or ""))
        self.model.to(self.device).eval()
        self.names = {int(k): str(v) for k, v in (self.model.config.id2label or {}).items()}
        TestProtocol.status("loading", f"Modell geladen | Klassen: {len(self.names)} | Geraet: {self.device}")

    def _detect(self, paths: List[Path]) -> List[List[Dict[str, Any]]]:
        images = [hfd.load_image(str(p)) for p in paths]
        try:
            raw = hfd.predict(self.model, self.processor, images, self.device)
        except NotImplementedError:
            if self.device != "mps":
                raise
            TestProtocol.warning("Operation auf MPS nicht verfuegbar; weiter auf CPU.")
            self.device = "cpu"
            self.model.to("cpu")
            raw = hfd.predict(self.model, self.processor, images, self.device)
        return [hfd.top_k(d) for d in raw]

    def _as_rows(self, dets: List[Dict[str, Any]], min_conf: float) -> List[Dict[str, Any]]:
        out = []
        for d in dets:
            # Hoechster Wert ueberhaupt: erklaert "nichts gefunden" bei einem
            # kaum trainierten Modell (mAP > 0, aber alles unter der Schwelle).
            self.best_score = max(getattr(self, "best_score", 0.0), float(d["score"]))
            if d["score"] < min_conf:
                continue
            x1, y1, x2, y2 = (round(v, 1) for v in d["box"])
            out.append({"label": self.names.get(d["label"], str(d["label"])), "confidence": round(d["score"], 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return out

    # ── Einzelbild ────────────────────────────────────────────────────────────
    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Bilddatei nicht gefunden: {path}")
        t0 = time.time()
        self.best_score = 0.0
        dets = self._as_rows(self._detect([path])[0], self.conf)
        counts = Counter(d["label"] for d in dets)
        best = max(dets, key=lambda d: d["confidence"]) if dets else None
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        TestProtocol.complete_single(
            predicted=describe(counts) if dets else
            f"(nichts ab {self.conf:.2f} - hoechster Wert {self.best_score:.2f})",
            confidence=best["confidence"] if best else None,
            top_predictions=[{"label": k, "score": max(d["confidence"] for d in dets if d["label"] == k)} for k in counts],
            inference_time=time.time() - t0,
            extra={"detections": dets, "image_path": str(path), "image_width": w, "image_height": h,
                   "conf_threshold": self.conf, "best_score": round(self.best_score, 4)},
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        classes = [self.names[i] for i in sorted(self.names)]
        ds = load_detection_dataset(root, class_names=classes)
        split_key = "test" if ds["splits"].get("test") else "val"
        records = ds["splits"].get(split_key) or []
        if split_key == "val" and records:
            TestProtocol.warning("Kein Test-Split gefunden; ausgewertet wird der Validierungs-Split.")
        if not records:
            raise ValueError(
                f"Kein test/ oder val/ Split mit Bildern in '{root}' gefunden. "
                "Teile das Dataset zuerst auf (datasets_split) oder lege test/images und test/labels an.")
        if self.config.max_samples:
            records = records[: int(self.config.max_samples)]

        TestProtocol.status("running", f"{len(records)} Bilder werden ausgewertet ({split_key})...")
        bs = max(1, int(self.config.batch_size or 4))
        rows: List[Dict[str, Any]] = []
        all_preds, all_gts = [], []
        correct, total_time, started = 0, 0.0, time.time()
        for i in range(0, len(records), bs):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            chunk = records[i:i + bs]
            t0 = time.time()
            batch_dets = self._detect([Path(r["path"]) for r in chunk])
            dt = (time.time() - t0) / len(chunk)
            total_time += dt * len(chunk)
            for rec, dets in zip(chunk, batch_dets):
                all_preds.append(dets)
                all_gts.append(rec["boxes"])
                shown = self._as_rows(dets, self.conf)
                expected = Counter(self.names.get(b["label"], str(b["label"])) for b in rec["boxes"])
                predicted = Counter(d["label"] for d in shown)
                ok = predicted == expected
                correct += int(ok)
                rows.append({
                    "sample_id": len(rows) + 1, "input_text": Path(rec["path"]).name, "input_path": rec["path"],
                    "expected_output": describe(expected), "predicted_output": describe(predicted),
                    "is_correct": ok, "confidence": max((d["confidence"] for d in shown), default=None),
                    "inference_time": dt, "detections": shown,
                })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=len(rows), total=len(records), sps=len(rows) / elapsed)
        elapsed = max(time.time() - started, 1e-6)

        TestProtocol.status("running", "Berechne mAP...")
        m = detection_metrics(all_preds, all_gts, conf=self.conf)
        # Reihenfolge zaehlt: die Startseite zeigt die erste Kennzahl (mAP50).
        metrics = {"mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"], "precision": m["precision"],
                   "recall": m["recall"], "images_exact": correct / len(rows),
                   # Precision/Recall zaehlen nur Boxen ab der Schwelle; liegt der
                   # hoechste Wert darunter, sind beide 0, obwohl mAP > 0 ist.
                   "conf_threshold": self.conf, "best_score": round(getattr(self, "best_score", 0.0), 4)}

        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": rows, "metrics": metrics}, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        hard_file = out_dir / "hard_examples.json"
        hard_file.write_text(json.dumps([r for r in rows if not r["is_correct"]], indent=2, ensure_ascii=False),
                             encoding="utf-8")
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(rows),
            accuracy=None, correct=correct, average_loss=None,
            average_inference_time=total_time / len(rows), samples_per_second=len(rows) / elapsed,
            hard_examples_file=str(hard_file), metrics=metrics,
        )
