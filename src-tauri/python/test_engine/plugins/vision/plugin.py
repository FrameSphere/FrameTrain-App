"""
plugins/vision/plugin.py
========================
Test-Plugin für Bildklassifikations-Modelle (timm, torchvision, HuggingFace ViT).

Dataset-Struktur (ImageNet-Standard):
  dataset_path/test/<klasse>/<bild>.jpg
  dataset_path/val/<klasse>/<bild>.jpg

  Alternativ: dataset_path/test/images/ + labels.csv (filename, label)

Single-Modus:
  Erwartet einen absoluten Bildpfad → gibt Top-5-Vorhersagen zurück.

Metriken:
  Top-1 Accuracy, Top-5 Accuracy, Avg. Confidence, Inference-Zeit
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Plugin(TestPlugin):

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.processor = None
        self.class_names: List[str] = []
        self.backend: str = "unknown"  # "timm" | "hf" | "torchvision"

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.device = self.get_device()
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.proto.status("init", f"Vision-Plugin | device={self.device}")

    # ── Modell laden ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        model_path = self.config.model_path
        self.proto.status("loading", f"Lade Vision-Modell: {Path(model_path).name} …")

        # Klassen-Mapping laden (falls vorhanden)
        self._load_class_names(model_path)

        # Versuche HuggingFace ViT / Swin zuerst
        if (Path(model_path) / "config.json").exists():
            try:
                self._load_hf_model(model_path)
                return
            except Exception as e:
                self.proto.warning(f"HF-Loader fehlgeschlagen: {e} → versuche timm …")

        # Fallback: timm
        self._load_timm_model(model_path)

    def _load_class_names(self, model_path: str) -> None:
        p = Path(model_path)
        for name in ["class_names.txt", "labels.txt", "classes.txt"]:
            f = p / name
            if f.exists():
                self.class_names = [l.strip() for l in f.read_text().splitlines() if l.strip()]
                self.proto.status("init", f"{len(self.class_names)} Klassen geladen")
                return

        cfg = p / "config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                id2label = data.get("id2label") or {}
                if id2label:
                    self.class_names = [id2label[str(i)] for i in range(len(id2label))]
            except Exception:
                pass

    def _load_hf_model(self, model_path: str) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.backend = "hf"
        if not self.class_names and hasattr(self.model.config, "id2label"):
            self.class_names = [self.model.config.id2label[i] for i in sorted(self.model.config.id2label.keys())]
        self.proto.status("loaded", f"HF Vision-Modell geladen ({len(self.class_names)} Klassen)")

    def _load_timm_model(self, model_path: str) -> None:
        import timm, torch
        # Versuche gespeichertes timm-Modell zu laden
        pt_files = list(Path(model_path).glob("*.pt")) + list(Path(model_path).glob("*.pth"))
        if not pt_files:
            raise FileNotFoundError(f"Keine .pt/.pth Datei gefunden in {model_path}")

        # Modell-Name aus model_info.json
        model_name = "resnet50"
        info_file = Path(model_path) / "model_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                model_name = info.get("model_name", model_name)
            except Exception:
                pass

        num_classes = max(len(self.class_names), 2)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state = torch.load(pt_files[0], map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.backend = "timm"

        # Preprocessing-Config aus timm
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_cfg, is_training=False)
        self.proto.status("loaded", f"timm-Modell geladen ({model_name})")

    # ── Bild vorverarbeiten ────────────────────────────────────────────────

    def _preprocess(self, image_path: str):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")

        if self.backend == "hf":
            return self.processor(images=img, return_tensors="pt")
        else:  # timm / torchvision
            import torch
            tensor = self.transforms(img).unsqueeze(0)
            return {"pixel_values": tensor}

    # ── Einzelnes Bild inferieren ──────────────────────────────────────────

    def _infer_image(
        self, image_path: str, expected_class: Optional[str] = None
    ) -> Dict[str, Any]:
        import torch

        t0 = time.time()
        inputs = self._preprocess(image_path)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.backend == "hf":
                logits = self.model(**inputs).logits
            else:
                logits = self.model(inputs["pixel_values"])

            probs = torch.softmax(logits, dim=-1)[0]
            top_k = min(5, probs.shape[0])
            top_probs, top_idxs = torch.topk(probs, top_k)

        inference_time = time.time() - t0

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            top_predictions.append({"label": label, "confidence": round(prob, 4), "class_id": idx})

        predicted_label = top_predictions[0]["label"] if top_predictions else "unknown"
        is_correct = False
        if expected_class is not None:
            is_correct = predicted_label.lower() == expected_class.lower()

        return {
            "input_path": image_path,
            "predicted_output": predicted_label,
            "expected_output": expected_class,
            "top_predictions": top_predictions,
            "is_correct": is_correct,
            "confidence": top_predictions[0]["confidence"] if top_predictions else None,
            "inference_time": inference_time,
            "error_type": None,
        }

    # ── Dataset-Struktur erkennen ──────────────────────────────────────────

    def _collect_samples(self) -> List[Dict[str, Any]]:
        """Sammelt (image_path, class_label) Paare aus dem Dataset."""
        root = Path(self.config.dataset_path)
        test_dir = root / "test"
        if not test_dir.exists():
            test_dir = root / "val"
        if not test_dir.exists():
            test_dir = root

        samples: List[Dict[str, Any]] = []

        # Variante 1: ImageNet-Struktur (test/<klasse>/<bild>)
        class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if class_dirs:
            for class_dir in sorted(class_dirs):
                for img_file in sorted(class_dir.rglob("*")):
                    if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append({
                            "input": str(img_file),
                            "expected": class_dir.name,
                        })
        else:
            # Variante 2: Alle Bilder + labels.csv
            csv_file = test_dir / "labels.csv"
            label_map: Dict[str, str] = {}
            if csv_file.exists():
                with open(csv_file, newline="") as f:
                    for row in csv.DictReader(f):
                        fn = row.get("filename") or row.get("file") or ""
                        lb = row.get("label") or row.get("class") or ""
                        if fn and lb:
                            label_map[fn] = lb

            for img_file in sorted(test_dir.rglob("*")):
                if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    label = label_map.get(img_file.name)
                    samples.append({"input": str(img_file), "expected": label})

        if not samples:
            raise ValueError(
                f"Keine Bilder gefunden in: {test_dir}\n"
                f"Erwartet: test/<klasse>/<bild.jpg> oder test/<bild.jpg> + labels.csv"
            )

        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        self.proto.status("loaded", f"{len(samples)} Bilder geladen")
        return samples

    # ── Dataset-Modus ─────────────────────────────────────────────────────

    def run_dataset(self) -> Dict[str, Any]:
        samples = self._collect_samples()
        self.proto.status("testing", f"Teste {len(samples)} Bilder …")

        results: List[Dict[str, Any]] = []
        t_start = time.time()

        for i, s in enumerate(samples):
            if self.is_stopped:
                break
            try:
                r = self._infer_image(s["input"], s.get("expected"))
            except Exception as e:
                r = {
                    "input_path": s["input"],
                    "predicted_output": "error",
                    "expected_output": s.get("expected"),
                    "top_predictions": [],
                    "is_correct": False,
                    "confidence": None,
                    "inference_time": 0.0,
                    "error_type": str(e),
                }
            r["sample_id"] = i
            results.append(r)

            elapsed = time.time() - t_start
            sps = (i + 1) / elapsed if elapsed > 0 else 0.0
            self.proto.progress(i + 1, len(samples), sps)

        total = len(results)
        correct_top1 = sum(1 for r in results if r["is_correct"])
        # Top-5 Accuracy: expected innerhalb der top-5
        correct_top5 = 0
        for r in results:
            if r.get("expected_output"):
                top5_labels = [p["label"].lower() for p in r.get("top_predictions", [])]
                if r["expected_output"].lower() in top5_labels:
                    correct_top5 += 1

        conf_vals = [r["confidence"] for r in results if r.get("confidence")]
        times = [r["inference_time"] for r in results]
        elapsed_total = time.time() - t_start

        has_labels = any(r.get("expected_output") for r in results)
        metrics = {
            "accuracy": (correct_top1 / total * 100) if (total > 0 and has_labels) else None,
            "top5_accuracy": (correct_top5 / total * 100) if (total > 0 and has_labels) else None,
            "total_samples": total,
            "correct_predictions": correct_top1 if has_labels else None,
            "incorrect_predictions": (total - correct_top1) if has_labels else None,
            "average_confidence": (sum(conf_vals) / len(conf_vals)) if conf_vals else None,
            "average_inference_time": (sum(times) / len(times)) if times else 0.0,
            "samples_per_second": total / elapsed_total if elapsed_total > 0 else 0.0,
            "total_time": elapsed_total,
            "average_loss": None,
        }

        out = Path(self.config.output_path)
        results_file = out / "test_results.json"
        # Predictions ohne full top_predictions Liste für kleinere Datei
        slim_results = [
            {k: v for k, v in r.items() if k != "top_predictions"}
            for r in results
        ]
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": slim_results}, f, indent=2, default=str)

        # Hard-Examples: falsche + niedrige Confidence
        hard = [r for r in results if not r["is_correct"]]
        hard_file = None
        if hard:
            hard_file = str(out / "hard_examples.jsonl")
            with open(hard_file, "w", encoding="utf-8") as f:
                for ex in hard:
                    f.write(json.dumps({
                        "input_path": ex["input_path"],
                        "expected": ex.get("expected_output"),
                        "predicted": ex["predicted_output"],
                        "confidence": ex.get("confidence"),
                        "top5": ex.get("top_predictions", []),
                    }, default=str) + "\n")

        return {
            "metrics": metrics,
            "predictions": slim_results,
            "results_file": str(results_file),
            "hard_examples_file": hard_file,
            "total_samples": total,
            "task_type": "vision",
        }

    # ── Single-Modus ───────────────────────────────────────────────────────

    def run_single(self, input_data: str) -> Dict[str, Any]:
        self.proto.status("inferring", f"Klassifiziere Bild: {Path(input_data).name} …")
        r = self._infer_image(input_data, expected_class=None)
        return {
            "top_predictions": r["top_predictions"],
            "predicted_label": r["predicted_output"],
            "confidence": r["confidence"],
            "inference_time": r["inference_time"],
        }
