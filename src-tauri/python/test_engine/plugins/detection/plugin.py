"""
plugins/detection/plugin.py
===========================
Test-Plugin für Objekterkennungs-Modelle (YOLO, Faster-RCNN, DETR).

Dataset-Struktur (YOLO-Standard):
  dataset_path/test/images/<img>.jpg
  dataset_path/test/labels/<img>.txt   (YOLO-Format: class cx cy w h)

  Alternativ COCO-Format:
  dataset_path/test/images/
  dataset_path/test/annotations.json

Single-Modus:
  Erwartet absoluten Bildpfad → gibt Bounding Boxes zurück.

Metriken:
  mAP@0.5, mAP@0.5:0.95, Precision, Recall
  Fallback (ohne GT): Avg. Detections/Image, Avg. Confidence
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Plugin(TestPlugin):

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.class_names: List[str] = []
        self.backend: str = "yolo"  # "yolo" | "hf"
        self.conf_threshold: float = 0.25
        self.iou_threshold: float = 0.45

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.device = self.get_device()
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.proto.status("init", f"Detection-Plugin | device={self.device}")

    # ── Modell laden ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        model_path = self.config.model_path
        self.proto.status("loading", f"Lade Detection-Modell: {Path(model_path).name} …")

        pt_files = list(Path(model_path).glob("*.pt")) + list(Path(model_path).glob("*.pth"))
        yaml_files = list(Path(model_path).glob("*.yaml")) + list(Path(model_path).glob("*.yml"))
        hf_config = Path(model_path) / "config.json"

        if hf_config.exists():
            self._load_hf_detection(model_path)
        elif pt_files or yaml_files:
            self._load_yolo(model_path, pt_files, yaml_files)
        else:
            raise FileNotFoundError(
                f"Kein unterstütztes Detection-Modell gefunden in: {model_path}\n"
                f"Erwartet: .pt/.yaml (YOLO) oder config.json (HuggingFace DETR/DETA)"
            )

    def _load_yolo(self, model_path: str, pt_files: list, yaml_files: list) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics nicht installiert:\n  pip install ultralytics"
            )

        model_file = pt_files[0] if pt_files else yaml_files[0]
        self.model = YOLO(str(model_file))
        self.backend = "yolo"

        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names.keys())]
        elif isinstance(names, list):
            self.class_names = names

        self.proto.status("loaded", f"YOLO-Modell geladen ({len(self.class_names)} Klassen)")

    def _load_hf_detection(self, model_path: str) -> None:
        from transformers import pipeline
        device_id = 0 if str(self.device) == "cuda" else -1
        self.model = pipeline("object-detection", model=model_path, device=device_id)
        self.backend = "hf"
        self.proto.status("loaded", "HF Detection-Pipeline geladen")

    # ── Einzelnes Bild inferieren ──────────────────────────────────────────

    def _infer_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        detections: List[Dict[str, Any]] = []

        if self.backend == "yolo":
            results = self.model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "label": label,
                        "class_id": cls_id,
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x1": xyxy[0], "y1": xyxy[1],
                            "x2": xyxy[2], "y2": xyxy[3],
                        },
                    })

        else:  # hf
            results = self.model(image_path)
            for det in results:
                detections.append({
                    "label": det.get("label", ""),
                    "confidence": det.get("score", 0.0),
                    "bbox": det.get("box", {}),
                })

        inference_time = time.time() - t0
        return {
            "input_path": image_path,
            "detections": detections,
            "num_detections": len(detections),
            "inference_time": inference_time,
        }

    # ── Ground-Truth laden ─────────────────────────────────────────────────

    def _load_yolo_labels(self, labels_dir: Path, image_path: Path) -> Optional[List[Dict]]:
        """Lädt YOLO-Format Labels für ein Bild."""
        label_file = labels_dir / (image_path.stem + ".txt")
        if not label_file.exists():
            return None
        boxes = []
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                boxes.append({"class_id": cls_id, "label": label, "cx": cx, "cy": cy, "w": w, "h": h})
        return boxes

    # ── IoU berechnen ──────────────────────────────────────────────────────

    @staticmethod
    def _iou(boxA: Dict, boxB: Dict) -> float:
        """IoU zwischen zwei {x1,y1,x2,y2} Boxen."""
        xA = max(boxA.get("x1", 0), boxB.get("x1", 0))
        yA = max(boxA.get("y1", 0), boxB.get("y1", 0))
        xB = min(boxA.get("x2", 1), boxB.get("x2", 1))
        yB = min(boxA.get("y2", 1), boxB.get("y2", 1))
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA.get("x2", 1) - boxA.get("x1", 0)) * (boxA.get("y2", 1) - boxA.get("y1", 0))
        areaB = (boxB.get("x2", 1) - boxB.get("x1", 0)) * (boxB.get("y2", 1) - boxB.get("y1", 0))
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    # ── Dataset sammeln ────────────────────────────────────────────────────

    def _collect_samples(self) -> List[Dict[str, Any]]:
        root = Path(self.config.dataset_path)
        test_dir = root / "test"
        if not test_dir.exists():
            test_dir = root / "val"
        if not test_dir.exists():
            test_dir = root

        images_dir = test_dir / "images"
        if not images_dir.exists():
            images_dir = test_dir

        labels_dir = test_dir / "labels"

        images = [
            f for f in sorted(images_dir.rglob("*"))
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            raise ValueError(f"Keine Bilder gefunden in: {images_dir}")

        if self.config.max_samples:
            images = images[: self.config.max_samples]

        samples = []
        for img in images:
            gt = self._load_yolo_labels(labels_dir, img) if labels_dir.exists() else None
            samples.append({"input": str(img), "ground_truth": gt})

        self.proto.status("loaded", f"{len(samples)} Bilder geladen")
        return samples

    # ── Dataset-Modus ─────────────────────────────────────────────────────

    def run_dataset(self) -> Dict[str, Any]:
        samples = self._collect_samples()
        self.proto.status("testing", f"Teste {len(samples)} Bilder …")

        results: List[Dict[str, Any]] = []
        t_start = time.time()
        total_tp, total_fp, total_fn = 0, 0, 0
        conf_all: List[float] = []

        for i, s in enumerate(samples):
            if self.is_stopped:
                break
            try:
                r = self._infer_image(s["input"])
                gt = s.get("ground_truth")

                # Precision / Recall Schätzung mit IoU=0.5 (wenn GT vorhanden)
                tp, fp, fn = 0, 0, 0
                if gt is not None:
                    matched_gt = set()
                    for det in r["detections"]:
                        matched = False
                        for j, g in enumerate(gt):
                            if j in matched_gt:
                                continue
                            if det["label"] == g["label"]:
                                tp += 1
                                matched_gt.add(j)
                                matched = True
                                break
                        if not matched:
                            fp += 1
                    fn = len(gt) - len(matched_gt)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                for det in r["detections"]:
                    if det.get("confidence"):
                        conf_all.append(det["confidence"])

                r["sample_id"] = i
                r["ground_truth"] = gt
                results.append(r)

            except Exception as e:
                results.append({
                    "input_path": s["input"],
                    "detections": [],
                    "num_detections": 0,
                    "inference_time": 0.0,
                    "sample_id": i,
                    "error_type": str(e),
                })

            elapsed = time.time() - t_start
            sps = (i + 1) / elapsed if elapsed > 0 else 0.0
            self.proto.progress(i + 1, len(samples), sps)

        total = len(results)
        times = [r["inference_time"] for r in results]
        elapsed_total = time.time() - t_start
        has_gt = total_tp + total_fp + total_fn > 0

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

        metrics: Dict[str, Any] = {
            "total_samples": total,
            "average_detections": sum(r["num_detections"] for r in results) / max(total, 1),
            "average_confidence": (sum(conf_all) / len(conf_all)) if conf_all else None,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": (precision * 100) if precision else None,  # Precision als Proxy
            "average_inference_time": (sum(times) / len(times)) if times else 0.0,
            "samples_per_second": total / elapsed_total if elapsed_total > 0 else 0.0,
            "total_time": elapsed_total,
            "average_loss": None,
            "correct_predictions": total_tp if has_gt else None,
            "incorrect_predictions": (total_fp + total_fn) if has_gt else None,
        }

        out = Path(self.config.output_path)
        results_file = out / "test_results.json"
        # Detections können groß werden – nur Zusammenfassung pro Bild
        slim = [
            {
                "sample_id": r["sample_id"],
                "input_path": r["input_path"],
                "num_detections": r["num_detections"],
                "inference_time": r["inference_time"],
                "error_type": r.get("error_type"),
                "detections": r["detections"][:20],  # max 20 Boxen pro Bild
            }
            for r in results
        ]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": slim}, f, indent=2, default=str)

        # Hard-Examples: Bilder mit 0 Detektionen oder Fehler
        hard = [r for r in results if r["num_detections"] == 0 or r.get("error_type")]
        hard_file = None
        if hard:
            hard_file = str(out / "hard_examples.jsonl")
            with open(hard_file, "w", encoding="utf-8") as f:
                for ex in hard:
                    f.write(json.dumps({
                        "input_path": ex["input_path"],
                        "num_detections": ex["num_detections"],
                        "error_type": ex.get("error_type"),
                    }) + "\n")

        return {
            "metrics": metrics,
            "predictions": slim,
            "results_file": str(results_file),
            "hard_examples_file": hard_file,
            "total_samples": total,
            "task_type": "detection",
        }

    # ── Single-Modus ───────────────────────────────────────────────────────

    def run_single(self, input_data: str) -> Dict[str, Any]:
        self.proto.status("inferring", f"Erkenne Objekte in: {Path(input_data).name} …")
        r = self._infer_image(input_data)
        return {
            "detections": r["detections"],
            "num_detections": r["num_detections"],
            "inference_time": r["inference_time"],
        }
