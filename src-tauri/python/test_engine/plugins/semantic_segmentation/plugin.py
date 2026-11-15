"""Test-Plugin: semantische Segmentierung mit einem trainierten HuggingFace-Modell.

single:  Bild -> Klassenmaske; gespeichert werden Klassen-ID-Maske, Farbmaske
         und Ueberlagerung. Die Antwort enthaelt die Pfade und eine Legende
         (Klasse, Farbe, Flaechenanteil).
dataset: Test-Split (test vor val vor train) aus images/ + masks/. Kennzahlen:
         mIoU, IoU je Klasse, Pixel-Accuracy, mittlere Klassen-Accuracy.
         Masken werden wie im Training gelesen (ft_data/segmentation.py);
         Klassen/Farben kommen aus label_mapping.json des Modells, wenn das
         Dataset keine eigene Klassenliste hat.

plugin_config: alpha (0.5, Deckkraft der Ueberlagerung), save_overlays (12,
Anzahl gespeicherter Ueberlagerungen im Dataset-Lauf).
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data import segmentation as seg

SUPPORTED_ARCHITECTURES = [
    "segformer", "upernet", "dpt", "beit", "data2vec-vision",
    "mobilevit", "mobilevitv2", "mobilenet_v2",
]


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        pc = config.plugin_config or {}
        self.alpha = float(pc.get("alpha", 0.5) or 0.5)
        self.save_overlays = int(pc.get("save_overlays", 12) or 0)
        self.processor = None
        self.model = None
        self.device = None
        self.mapping: Dict[str, Any] = {}
        self.names: List[str] = []
        self.colors: List[tuple] = []
        self.prep: Dict[str, Any] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Segmentierungsmodell...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        except Exception:
            from transformers import SegformerImageProcessor
            self.processor = SegformerImageProcessor()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(str(model_path))
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)

        mp = model_path / "label_mapping.json"
        if mp.is_file():
            try:
                self.mapping = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                self.mapping = {}
        n = int(self.model.config.num_labels)
        id2label = getattr(self.model.config, "id2label", {}) or {}
        names = self.mapping.get("classes") or [str(id2label.get(i, id2label.get(str(i), f"class_{i}"))) for i in range(n)]
        self.names = list(names)[:n] + [f"class_{i}" for i in range(len(names), n)]
        cols = [tuple(c) for c in (self.mapping.get("colors") or [])]
        self.colors = cols if len(cols) == n else seg.default_palette(n)
        self.prep = seg.processor_settings(self.processor)
        if self.mapping.get("image_size"):
            h, w = self.mapping["image_size"]
            self.prep["size"] = (int(h), int(w))
        TestProtocol.status("loading", f"Modell geladen | Klassen: {n} | Eingabe {self.prep['size'][0]}px | Geraet: {self.device}")

    def _predict(self, image) -> np.ndarray:
        """Klassenmaske in Originalgroesse (H, W)."""
        import torch
        x, _ = seg.prepare_pair(image, None, self.prep["size"], self.prep["mean"], self.prep["std"],
                                rescale=self.prep["rescale"])
        with torch.no_grad():
            out = self.model(pixel_values=torch.from_numpy(x)[None].to(self.device))
            logits = out.logits if hasattr(out, "logits") else out[0]
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(image.height, image.width), mode="bilinear", align_corners=False)
            return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)

    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Bilddatei nicht gefunden: {path}")
        image = seg.load_rgb(path)
        t0 = time.time()
        pred = self._predict(image)
        dt = time.time() - t0
        out_dir = Path(self.config.output_path or path.parent)
        files = seg.save_prediction(image, pred, self.colors, out_dir, path.stem, alpha=self.alpha)
        rows = seg.legend(pred, self.names, self.colors)
        top = rows[0] if rows else {"label": "—", "share": None}
        TestProtocol.complete_single(
            predicted=top["label"], confidence=top["share"],
            top_predictions=[{"label": r["label"], "score": r["share"]} for r in rows],
            inference_time=dt,
            extra={"output_kind": "segmentation", "image_path": str(path), **files,
                   "legend": rows, "classes": [
                       {"id": i, "label": n, "color": "#%02x%02x%02x" % tuple(self.colors[i])}
                       for i, n in enumerate(self.names)]},
        )

    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split, pairs = seg.evaluation_pairs(root)
        if not pairs:
            raise ValueError(f"Keine Bild/Masken-Paare in {root} gefunden (erwartet images/ + masks/).")
        n_classes = len(self.names)
        info = seg.read_class_file(root)
        decoder = seg.build_decoder([m for _, m in pairs], info, fallback=self.mapping, num_classes=n_classes)
        if self.config.max_samples and len(pairs) > self.config.max_samples:
            import random
            pairs = sorted(random.Random(42).sample(pairs, int(self.config.max_samples)))
        TestProtocol.status("running", f"Teste {len(pairs)} Bilder aus Split '{split}' "
                                       f"({'Farbmasken' if decoder.mode == 'color' else 'Klassen-ID-Masken'})...")

        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        overlay_dir = out / "overlays"
        cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
        results: List[Dict[str, Any]] = []
        overlays: List[str] = []
        total_time = 0.0
        started = time.time()
        for idx, (img_path, mask_path) in enumerate(pairs, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            image = seg.load_rgb(img_path)
            label = decoder.decode(mask_path)
            t0 = time.time()
            pred = self._predict(image)
            dt = time.time() - t0
            total_time += dt
            if label.shape != pred.shape:
                from PIL import Image
                label = np.asarray(Image.fromarray(label.astype(np.int32), mode="I")
                                   .resize((pred.shape[1], pred.shape[0]), Image.NEAREST), dtype=np.int64)
            cm = seg.confusion(pred, label, n_classes)
            cm_total += cm
            s = seg.scores(cm, self.names)
            if len(overlays) < self.save_overlays:
                overlays.append(seg.save_prediction(image, pred, self.colors, overlay_dir,
                                                    f"{idx:04d}_{img_path.stem}", alpha=self.alpha)["overlay_path"])
            results.append({
                "sample_id": idx, "input_path": str(img_path), "expected_output": str(mask_path),
                "predicted_output": f"mIoU {s['mean_iou']:.3f}",
                "mean_iou": s["mean_iou"], "pixel_accuracy": s["pixel_accuracy"],
                "is_correct": None, "confidence": s["pixel_accuracy"], "inference_time": dt,
            })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(pairs), sps=idx / elapsed,
                                  eta=(len(pairs) - idx) * elapsed / idx)
        if decoder.unknown_pixels:
            TestProtocol.warning(f"{decoder.unknown_pixels} Maskenpixel mit unbekannter Farbe wurden ignoriert.")

        total = seg.scores(cm_total, self.names)
        per_class = [{"id": i, "label": n, "color": "#%02x%02x%02x" % tuple(self.colors[i]),
                      "iou": total["per_class_iou"][n], "accuracy": total["per_class_accuracy"][n],
                      "pixels": int(cm_total[i].sum())}
                     for i, n in enumerate(self.names)]
        hard = sorted(results, key=lambda r: r["mean_iou"])[: min(20, len(results))]
        hard_file = out / "hard_examples.json"
        hard_file.write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")
        results_file = out / "results.json"
        metrics = {"mean_iou": total["mean_iou"], "pixel_accuracy": total["pixel_accuracy"],
                   "mean_accuracy": total["mean_accuracy"]}
        results_file.write_text(json.dumps({
            "metrics": {**metrics, "per_class_iou": total["per_class_iou"], "split": split},
            "per_class": per_class, "predictions": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results),
            accuracy=total["pixel_accuracy"], correct=None, average_loss=None,
            average_inference_time=total_time / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
            hard_examples_file=str(hard_file), metrics=metrics,
            extra={"output_kind": "segmentation", "split": split, "per_class": per_class,
                   "images": overlays},
        )
