"""
Semantic Segmentation (HuggingFace)
===================================
Trainiert das heruntergeladene Segmentierungsmodell selbst
(AutoModelForSemanticSegmentation: SegFormer, UperNet, DPT, BEiT,
Data2Vec-Vision, MobileViT/-v2, MobileNetV2-DeepLab).

Dataset: Bildordner + Maskenordner, Details in ft_data/segmentation.py.
    <dataset>/images/*.jpg + <dataset>/masks/*.png        (Val wird abgetrennt)
    <dataset>/train/images + train/masks [+ val/ test/]   (fertig aufgeteilt)
Masken: PNG mit Klassen-IDs (255 = ignorieren) oder Farbmasken mit
Farbtabelle (classes.json / labelmap.txt / classes.txt) bzw. aus den Masken
abgeleiteten Farben.

Kennzahlen im Training: Loss, mIoU, Pixel-Accuracy, mittlere Klassen-Accuracy.
Das Modell mit dem besten Val-mIoU wird gespeichert (nicht das letzte).

plugin_config:
  image_size   Kantenlaenge fuer Training (Standard: Groesse des Bild-Prozessors, sonst 512)
  augment      zufaelliges horizontales Spiegeln im Training (Standard: true)
  val_fraction Anteil fuer die Validierung, wenn kein val/ existiert (0.2)
"""
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft
from ft_data import segmentation as seg

SUPPORTED_ARCHITECTURES = [
    "segformer", "upernet", "dpt", "beit", "data2vec-vision",
    "mobilevit", "mobilevitv2", "mobilenet_v2",
]


def _truthy(v: Any, default: bool = True) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "no", "nein", "off")


class SegDataset:
    """Torch-Dataset: liest Bild + Maske erst beim Zugriff."""

    def __init__(self, pairs, decoder, prep, augment: bool, seed: int = 42):
        self.pairs = list(pairs)
        self.decoder = decoder
        self.prep = prep
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import torch
        img_path, mask_path = self.pairs[i]
        image = seg.load_rgb(img_path)
        mask = self.decoder.decode(mask_path)
        if mask.shape != (image.height, image.width):
            # Maske in anderer Aufloesung: auf Bildgroesse bringen (nearest)
            from PIL import Image
            mask = np.asarray(Image.fromarray(mask.astype(np.int32), mode="I")
                              .resize((image.width, image.height), Image.NEAREST), dtype=np.int64)
        flip = bool(self.augment and self.rng.random() < 0.5)
        x, y = seg.prepare_pair(image, mask, self.prep["size"], self.prep["mean"], self.prep["std"],
                                flip=flip, rescale=self.prep["rescale"])
        return {"pixel_values": torch.from_numpy(x), "labels": torch.from_numpy(y)}


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.decoder = None
        self.prep: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.model_type = ""
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate
        self._best_state = None
        self._best = {"mean_iou": -1.0, "epoch": 0, "step": 0}
        self._last_eval: Dict[str, Any] = {}

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        from transformers import AutoImageProcessor

        cfg_path = Path(self.config.model_path) / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", "")
            except Exception:
                self.model_type = ""
        if self.model_type and self.model_type not in SUPPORTED_ARCHITECTURES:
            MessageProtocol.warning(
                f"Architektur '{self.model_type}' ist fuer Segmentierung nicht geprueft "
                f"(unterstuetzt: {', '.join(SUPPORTED_ARCHITECTURES)}). Versuche es trotzdem.")
        MessageProtocol.status("init", f"Architektur: {self.model_type or 'unbekannt'} | Lade Bild-Prozessor...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        except Exception as exc:
            # Reine Encoder-Checkpoints (z. B. aeltere mit-bX) bringen manchmal keinen
            # Prozessor mit — ImageNet-Werte sind fuer diese Modelle richtig.
            from transformers import SegformerImageProcessor
            MessageProtocol.status("init", f"Kein Bild-Prozessor im Modell ({exc}) - nutze ImageNet-Normalisierung.")
            self.processor = SegformerImageProcessor()
        # Klassen-IDs der Masken unveraendert lassen (ADE-Checkpoints haben reduce_labels=True).
        for attr in ("do_reduce_labels", "reduce_labels"):
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, False)
        pc = self.config.plugin_config or {}
        self.prep = seg.processor_settings(self.processor)
        size = int(pc.get("image_size") or 0)
        if size > 0:
            self.prep["size"] = (size, size)
        MessageProtocol.status("init", f"Bild-Prozessor geladen | Eingabegroesse {self.prep['size'][0]}x{self.prep['size'][1]}")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        pc = self.config.plugin_config or {}
        layout = seg.resolve_layout(root, seed=self.config.seed,
                                    val_fraction=float(pc.get("val_fraction", 0.2) or 0.2))
        for note in layout.notes:
            MessageProtocol.status("loading_data", note)
        info = seg.read_class_file(root)
        if info:
            MessageProtocol.status("loading_data", f"Klassenliste aus {info.source}: {len(info.names)} Klassen")
        self.decoder = seg.build_decoder([m for _, m in layout.train + layout.val], info)
        self.classes = self.decoder.names
        if len(self.classes) < 2:
            raise ValueError(
                "Die Masken enthalten nur eine Klasse - Segmentierung braucht mindestens zwei "
                "(z. B. Hintergrund und Objekt). Pruefe die Masken oder lege eine classes.txt an.")
        kind = "Farbmasken" if self.decoder.mode == "color" else "Klassen-ID-Masken"
        MessageProtocol.status(
            "loading_data",
            f"{kind} | Klassen ({len(self.classes)}): {', '.join(self.classes[:20])}"
            f"{' ...' if len(self.classes) > 20 else ''}")

        augment = _truthy(pc.get("augment"), True)
        val_pairs = layout.val
        cap = int(getattr(self.config, "max_eval_samples", 0) or 0)
        if cap > 0 and len(val_pairs) > cap:
            val_pairs = seg_sample(val_pairs, cap, self.config.seed)
            MessageProtocol.status("loading_data", f"Eval-Split auf {cap} Paare begrenzt (Max Eval Samples).")
        self.train_dataset = SegDataset(layout.train, self.decoder, self.prep, augment, self.config.seed)
        self.eval_dataset = SegDataset(val_pairs, self.decoder, self.prep, False)
        MessageProtocol.status("loading_data", f"Train: {len(self.train_dataset)} | Eval: {len(self.eval_dataset)}"
                               f"{f' | Test (fuer den Test-Lauf): {len(layout.test)}' if layout.test else ''}")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForSemanticSegmentation

        MessageProtocol.status("building_model", "Lade Modell fuer semantische Segmentierung...")
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            self.config.model_path,
            num_labels=len(self.classes),
            id2label={i: c for i, c in enumerate(self.classes)},
            label2id={c: i for i, c in enumerate(self.classes)},
            ignore_mismatched_sizes=True,   # Segmentierungskopf wird ersetzt
        )
        for attr in ("semantic_loss_ignore_index", "loss_ignore_index"):
            if hasattr(self.model.config, attr):
                setattr(self.model.config, attr, seg.IGNORE_INDEX)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status(
            "building_model",
            f"Modell geladen ({type(self.model).__name__}) | Parameter: {params/1e6:.1f}M | Klassen: {len(self.classes)}")

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _preprocess_logits(self, logits, labels):
        import torch
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)

    def _compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        cm = seg.confusion(preds, labels, len(self.classes))
        s = seg.scores(cm, self.classes)
        self._last_eval = s
        return {"mean_iou": s["mean_iou"], "pixel_accuracy": s["pixel_accuracy"],
                "mean_accuracy": s["mean_accuracy"]}

    def train(self) -> None:
        from transformers import Trainer, TrainerCallback, TrainingArguments

        self.device_used = hft.device_name()
        MessageProtocol.status("training", f"Training gestartet | Geraet: {self.device_used.upper()}")

        total_steps = max((len(self.train_dataset) // max(self.config.batch_size, 1))
                          * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), TrainingArguments,
            remove_unused_columns=False,
        )
        plugin = self

        class _Best(TrainerCallback):
            """Merkt sich die Gewichte mit dem besten Val-mIoU."""
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                miou = (metrics or {}).get("eval_mean_iou")
                if miou is None or miou <= plugin._best["mean_iou"]:
                    return
                plugin._best = {"mean_iou": float(miou), "step": int(state.global_step),
                                "epoch": hft._epoch_number(state, plugin.config),
                                "pixel_accuracy": float((metrics or {}).get("eval_pixel_accuracy", 0.0))}
                plugin._best_state = {k: v.detach().to("cpu").clone()
                                      for k, v in plugin.model.state_dict().items()}
                MessageProtocol.status("training", f"Neues bestes Modell: mIoU {miou:.4f} (Schritt {state.global_step})")

        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits,
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps), _Best()],
        )
        self._start_time = time.time()
        self._trainer.train(resume_from_checkpoint=hft.resume_checkpoint(self.config))
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate()
        last_miou = float(result.get("eval_mean_iou", 0.0) or 0.0)
        if self._best_state is not None and self._best["mean_iou"] > last_miou + 1e-9:
            MessageProtocol.status(
                "validating",
                f"Letzter Stand mIoU {last_miou:.4f} < bester {self._best['mean_iou']:.4f} "
                f"(Schritt {self._best['step']}) - gespeichert wird das beste Modell.")
            self.model.load_state_dict(self._best_state)
            result = self._trainer.evaluate()
        metrics = hft.final_metrics(self, self._trainer, result, self._start_time,
                                    architecture=self.model_type, num_labels=len(self.classes))
        metrics.update({
            "accuracy": float(result.get("eval_pixel_accuracy", 0.0) or 0.0),
            "mean_iou": float(result.get("eval_mean_iou", 0.0) or 0.0),
            "pixel_accuracy": float(result.get("eval_pixel_accuracy", 0.0) or 0.0),
            "mean_accuracy": float(result.get("eval_mean_accuracy", 0.0) or 0.0),
            "best_epoch": int(self._best.get("epoch", 0) or 0),
            "best_step": int(self._best.get("step", 0) or 0),
            "per_class_iou": self._last_eval.get("per_class_iou", {}),
        })
        # Klassifikations-F1/Precision/Recall gibt es hier nicht — keine Nullen melden.
        for k in ("f1", "precision", "recall"):
            metrics.pop(k, None)
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        try:
            self.processor.save_pretrained(str(out))
        except Exception as exc:
            MessageProtocol.warning(f"Bild-Prozessor nicht gespeichert: {exc}")
        mapping = self.decoder.to_dict()
        mapping.update({"image_size": list(self.prep["size"]), "task": "semantic_segmentation",
                        "best": self._best if self._best_state is not None else None})
        (out / "label_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)


def seg_sample(items, n: int, seed: int):
    import random
    items = list(items)
    random.Random(seed).shuffle(items)
    return sorted(items[:n])
