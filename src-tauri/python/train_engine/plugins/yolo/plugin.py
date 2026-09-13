"""YOLO Object Detection Plugin — task_type: 'detect'"""
import json, os, shutil
from pathlib import Path
from typing import Any, Dict, Optional
from core.config import TrainingConfig
from core.protocol import MessageProtocol


class YOLOPlugin:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.is_stopped = False
        self.results = None
        self._yaml_path: Optional[str] = None
        self._output_dir: Optional[Path] = None
        self._device_used: str = "cpu"
        pc = config.plugin_config or {}
        self.yolo_model   = pc.get("yolo_model") or ""
        self.task         = pc.get("task",          "detect")
        self.imgsz        = int(pc.get("imgsz",    640))
        self.patience     = int(pc.get("patience",  50))
        self.augment      = bool(pc.get("augment",  True))
        self.optimizer_name = pc.get("optimizer",  "SGD")
        self.lr0          = float(pc.get("lr0",     0.01))
        self.lrf          = float(pc.get("lrf",     0.01))
        self.momentum     = float(pc.get("momentum", 0.937))
        self.wd           = float(pc.get("weight_decay", 0.0005))
        self.device_arg   = pc.get("device", "")


    def stop(self) -> None:
        """Abbruch aus der Oberflaeche.

        Diese Klasse erbt nicht von TrainPlugin, wo stop() definiert ist.
        Ohne die Methode lief der Signal-Handler der Engine in einen
        AttributeError: "Stoppen" blieb wirkungslos und das Training lief
        bis zur letzten Epoche weiter, obwohl is_stopped ueberall geprueft wird.
        """
        self.is_stopped = True

    def setup(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa
        except ImportError:
            MessageProtocol.error("Ultralytics nicht installiert",
                "pip install ultralytics>=8.0.0")
            return False
        dsp = self.config.dataset_path
        if not dsp or not Path(dsp).exists():
            MessageProtocol.error("Dataset nicht gefunden", f"Pfad: {dsp!r}")
            return False
        yaml_path = self._find_or_build_yaml(Path(dsp))
        if yaml_path is None:
            return False
        # Pascal VOC (XML) in YOLO-Labels umrechnen und Platzhalter-Klassen aus
        # dem Import ersetzen — beides liest Ultralytics sonst nicht.
        try:
            from ft_data.detection import convert_voc_to_yolo, fill_placeholder_names
            note = lambda m: MessageProtocol.status("setup", m)
            convert_voc_to_yolo(Path(dsp), yaml_path, status=note)
            fill_placeholder_names(Path(dsp), yaml_path, status=note)
        except Exception as e:
            MessageProtocol.status("setup", f"Label-Vorbereitung uebersprungen: {e}")
        if not self._verify_labels(yaml_path):
            return False
        self._yaml_path = str(yaml_path)
        self.yolo_model = self._resolve_weights()
        self._output_dir = Path(self.config.output_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        MessageProtocol.status("setup",
            f"YOLO Setup OK\n  Model: {self.yolo_model}\n  Task: {self.task}\n  YAML: {self._yaml_path}")
        return True

    # Suffixe, an denen Ultralytics die Aufgabe einer Gewichtsdatei erkennt.
    _TASK_SUFFIX = {"segment": "-seg", "pose": "-pose", "classify": "-cls", "obb": "-obb"}
    # Groessenreihenfolge: klein zuerst, damit ein Fine-Tuning auf einem Laptop nicht ausufert.
    _SIZE_ORDER = ["n", "s", "m", "l", "x"]

    def _weights_from_related(self, model_dir: Path) -> Optional[Path]:
        """Gewichte, wenn die gewaehlte Version selbst keine model.pt hat.

        Ein Lauf, der vor 1.2.63 scheiterte, legte trotzdem eine Version an —
        nur mit train/args.yaml. Als "neueste" Version vorausgewaehlt, liess sie
        das naechste Training yolov8n.pt aus dem Netz laden (offline: Abbruch).
        Reihenfolge: eigene train/weights, dann die neueste Vorgaenger-Version mit
        model.pt, dann das importierte Originalmodell.
        """
        for name in ("best.pt", "last.pt"):
            p = model_dir / "train" / "weights" / name
            if p.is_file():
                MessageProtocol.status("setup", f"Startgewichte: train/weights/{name} dieser Version")
                return p
        if model_dir.parent.name != "versions":
            return None
        siblings = sorted(
            (d for d in model_dir.parent.iterdir() if d.is_dir() and d != model_dir and (d / "model.pt").is_file()),
            key=lambda d: (d / "model.pt").stat().st_mtime, reverse=True)
        if siblings:
            MessageProtocol.status("setup",
                f"Diese Version enthaelt keine Gewichte (vermutlich ein abgebrochener Lauf) — "
                f"nutze {siblings[0].name}/model.pt")
            return siblings[0] / "model.pt"
        root = model_dir.parent.parent
        originals = sorted(root.glob("*.pt"))
        if originals:
            MessageProtocol.status("setup",
                f"Diese Version enthaelt keine Gewichte — nutze das importierte Modell {originals[0].name}")
            return min(originals, key=lambda p: p.stat().st_size)
        return None

    def _resolve_weights(self) -> str:
        """Waehlt die Startgewichte.

        Ohne diesen Schritt wurde immer 'yolov8n.pt' geladen – Ultralytics holte
        das Modell aus dem Netz, und das vom Nutzer importierte YOLO11 lag
        ungenutzt daneben.
        """
        explicit = str(self.yolo_model or "").strip()
        if explicit:
            # Ein konkreter Pfad hat Vorrang; ein blosser Name geht an Ultralytics.
            if Path(explicit).exists() or not explicit.endswith(".pt"):
                return explicit
            local = Path(self.config.model_path or "") / explicit
            if local.exists():
                return str(local)
            return explicit

        model_dir = Path(self.config.model_path or "")
        candidates = sorted(model_dir.glob("*.pt")) if model_dir.is_dir() else []
        if not candidates:
            fallback = self._weights_from_related(model_dir)
            if fallback is not None:
                return str(fallback)
            MessageProtocol.status("setup",
                "Keine .pt-Gewichte im Modellordner – Ultralytics laedt yolov8n.pt aus dem Netz.")
            return "yolov8n.pt"

        wanted = self._TASK_SUFFIX.get(self.task, "")
        other  = [s for t, s in self._TASK_SUFFIX.items() if s != wanted]
        def matches_task(p: Path) -> bool:
            stem = p.stem.lower()
            if wanted:
                return stem.endswith(wanted)
            # detect: alles ohne Aufgaben-Suffix
            return not any(stem.endswith(s) for s in other)

        pool = [p for p in candidates if matches_task(p)] or candidates

        def rank(p: Path):
            stem = p.stem.lower()
            base = stem[:-len(wanted)] if wanted and stem.endswith(wanted) else stem
            size = base[-1] if base and base[-1] in self._SIZE_ORDER else ""
            return (self._SIZE_ORDER.index(size) if size else len(self._SIZE_ORDER),
                    p.stat().st_size if p.exists() else 0)

        chosen = sorted(pool, key=rank)[0]
        MessageProtocol.status("setup", f"Startgewichte: {chosen.name} (aus dem importierten Modell)")
        return str(chosen)

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    @staticmethod
    def _label_candidates(img: Path) -> list:
        """Pfade, an denen ein Label zu diesem Bild liegen kann.

        Ultralytics ersetzt im Bildpfad das Segment 'images' durch 'labels'.
        Zusaetzlich werden die in der Praxis ueblichen Varianten geprueft:
        Geschwisterordner 'labels' und die .txt-Datei direkt neben dem Bild.
        """
        parts = list(img.parts)
        cands = []
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
                break
        cands.append(img.parent.parent / "labels" / (img.stem + ".txt"))
        cands.append(img.parent / "labels" / (img.stem + ".txt"))
        cands.append(img.with_suffix(".txt"))
        return cands

    def _sample_labels(self, images_dir: Path, sample: int = 60) -> tuple:
        """(gefundene Labels, geprüfte Bilder) einer Stichprobe."""
        imgs = [f for f in sorted(images_dir.rglob("*"))
                if f.is_file() and f.suffix.lower() in self._IMG_EXTS]
        probe = imgs[:sample]
        found = sum(1 for im in probe if any(c.exists() for c in self._label_candidates(im)))
        return found, len(probe)

    def _yaml_image_dirs(self, yaml_path: Path) -> list:
        """Die in der dataset.yaml genannten Bildordner (train/val), die existieren."""
        return [found for found, _ in self._yaml_split_dirs(yaml_path) if found]

    def _yaml_split_dirs(self, yaml_path: Path) -> list:
        """(gefundener Ordner oder None, erwarteter Pfad) je train/val-Eintrag.

        Loest wie Ultralytics auf: relativ zu `path:`, sonst zum yaml-Ordner;
        Roboflow-Exporte schreiben '../train/images', das Ultralytics ebenfalls
        relativ zum yaml-Ordner findet. Listen-Eintraege ([a, b]) werden nicht
        geprueft, statt faelschlich als fehlend zu gelten.
        """
        root = yaml_path.parent
        try:
            lines = yaml_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        base = None
        entries = []
        for line in lines:
            line = line.split(" #")[0].strip()
            if line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if not value or value.startswith("["):
                continue
            if key == "path":
                base = Path(value) if Path(value).is_absolute() else root / value
            elif key in ("train", "val"):
                entries.append(value)
        result = []
        for value in entries:
            p = Path(value)
            if p.is_absolute():
                candidates = [p]
            else:
                candidates = [(base or root) / value, root / value]
                if value.startswith("../"):
                    candidates.append(root / value[3:])
            found = next((c for c in candidates if c.is_dir()), None)
            result.append((found, candidates[0]))
        return result

    def _verify_labels(self, yaml_path: Path) -> bool:
        """Bricht ab, wenn zu den Bildern keine Labels existieren.

        Ohne diese Pruefung lief ein Training ueber alle Epochen durch, obwohl
        Ultralytics jedes Bild als Hintergrund behandelte: box- und dfl-Loss
        sind dann konstant 0, mAP ebenfalls — die Oberflaeche zeigte
        'loss=0.0000 mAP50=0.0000' und niemand konnte sehen, woran es lag.
        """
        missing = [expected for found, expected in self._yaml_split_dirs(yaml_path) if not found]
        if missing:
            # Sonst bricht erst Ultralytics mit "images not found" ab — und die
            # Meldung nennt weder die yaml-Zeile noch den Ordner, der wirklich da ist.
            MessageProtocol.error(
                "dataset.yaml verweist auf fehlende Ordner",
                f"In {yaml_path} stehen Bildordner, die es nicht gibt:\n  "
                + "\n  ".join(str(d) for d in missing)
                + "\nPruefe train:/val: im Tab 'dataset.yaml' des Datasets.")
            return False
        dirs = self._yaml_image_dirs(yaml_path)
        found = probed = 0
        for d in dirs:
            f, n = self._sample_labels(d)
            found += f
            probed += n
        if probed == 0:
            return True  # Keine Bilder gefunden — daran scheitert Ultralytics selbst.
        if found == 0:
            MessageProtocol.error(
                "Keine Labels zum Dataset gefunden",
                "Zu den Bildern existiert keine einzige .txt-Annotation. Ein "
                "Objekterkennungs-Training waere wirkungslos: Loss und mAP "
                "blieben ueber alle Epochen 0.\n"
                "Erwartet wird neben dem Bildordner ein gleich aufgebauter "
                "Label-Ordner, z.B. images/train/foto.jpg + labels/train/foto.txt.\n"
                f"Geprueft: {', '.join(str(d) for d in dirs) or yaml_path.parent}")
            return False
        if found < probed / 2:
            MessageProtocol.status("setup",
                f"Warnung: nur {found} von {probed} geprueften Bildern haben ein Label. "
                "Bilder ohne Label zaehlen als Hintergrund.")
        return True

    def _find_or_build_yaml(self, root: Path) -> Optional[Path]:
        # Die App traegt die bereits gepruefte (und ggf. reparierte) yaml ein.
        given = (self.config.plugin_config or {}).get("dataset_yaml_path")
        if given and Path(given).is_file():
            return Path(given)
        for c in [root/"dataset.yaml", root/"data.yaml"]:
            if c.exists():
                return c
        MessageProtocol.status("setup", "Kein dataset.yaml — generiere automatisch...")
        return self._generate_yaml(root)

    def _generate_yaml(self, root: Path) -> Optional[Path]:
        def find_images(base: Path) -> Optional[Path]:
            for c in [base/"images", base]:
                if c.is_dir() and any(f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")
                    for f in c.rglob("*") if f.is_file()):
                    return c
            return None
        train_imgs = find_images(root/"train") or find_images(root/"images"/"train")
        val_imgs = (find_images(root/"val") or find_images(root/"images"/"val")
                    or find_images(root/"valid") or find_images(root/"images"/"valid"))
        if not train_imgs:
            MessageProtocol.error("YAML-Generierung fehlgeschlagen",
                f"Keine Trainings-Bilder in {root}.\nErwartet: train/images/*.jpg + train/labels/*.txt")
            return None
        classes = self._detect_classes(root) or ["object"]
        yaml_path = root / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {root.resolve()}\n")
            f.write(f"train: {train_imgs.resolve()}\n")
            f.write(f"val: {val_imgs.resolve() if val_imgs else train_imgs.resolve()}\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {classes}\n")
        MessageProtocol.status("setup", f"dataset.yaml generiert: {len(classes)} Klassen")
        return yaml_path

    def _detect_classes(self, root: Path) -> list:
        class_ids = set()
        for ld in [root/"labels", root/"train"/"labels", root/"val"/"labels"]:
            if not ld.is_dir(): continue
            for txt in ld.rglob("*.txt"):
                try:
                    for line in txt.read_text(encoding="utf-8").strip().splitlines():
                        parts = line.strip().split()
                        if parts: class_ids.add(int(parts[0]))
                except Exception: continue
        for nf in [root/"classes.txt", root/"obj.names", root/"labels.txt"]:
            if nf.exists():
                names = [l.strip() for l in nf.read_text(encoding="utf-8").splitlines() if l.strip()]
                if names: return names
        return [f"class_{i}" for i in sorted(class_ids)] if class_ids else []

    @staticmethod
    def _sum_prefixed(metrics: Dict[str, Any], prefix: str) -> Optional[float]:
        """Summiert box/cls/dfl-Loss eines Praefixes ('train/' oder 'val/')."""
        vals = [float(v) for k, v in metrics.items()
                if k.startswith(prefix) and k.endswith("_loss") and v is not None]
        return sum(vals) if vals else None

    def _running_loss(self, trainer) -> float:
        """Laufender Trainings-Loss (box + cls + dfl) der aktuellen Epoche."""
        try:
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                items = trainer.label_loss_items(tloss)
                total = self._sum_prefixed(items, "train/")
                if total is None:
                    # label_loss_items kann je nach Task ohne Praefix liefern.
                    total = sum(float(v) for v in items.values()) if isinstance(items, dict) else None
                if total is not None:
                    return float(total)
        except Exception:
            pass
        total = self._sum_prefixed(dict(getattr(trainer, "metrics", None) or {}), "train/")
        return float(total) if total is not None else 0.0

    @staticmethod
    def _current_lr(trainer) -> float:
        # MessageProtocol.progress erwartet ein float, kein None.
        try:
            lrs = [float(v) for v in (getattr(trainer, "lr", None) or {}).values()]
            if lrs:
                return lrs[0]
        except Exception:
            pass
        return 0.0

    def load_data(self) -> None: pass
    def build_model(self) -> None: pass

    def train(self) -> bool:
        if not self._yaml_path:
            MessageProtocol.error("Training", "setup() nicht aufgerufen.")
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            MessageProtocol.error("Ultralytics", "Import fehlgeschlagen.")
            return False
        try:
            import torch
            MessageProtocol.status("train", f"Lade {self.yolo_model}...")
            self.model = YOLO(self.yolo_model)
            if self.device_arg:
                device = self.device_arg
            elif torch.cuda.is_available():
                device = "0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            # Die Analyse-Seite liest das Geraet aus den final_metrics; ohne das
            # stand dort immer "cpu", auch wenn auf MPS trainiert wurde.
            self._device_used = "cuda" if device.isdigit() else device

            # Schritt-Fortschritt ueber den ganzen Lauf: Ultralytics zaehlt in
            # Batches, die App in Steps. Ohne das stand der Balken je Epoche
            # minutenlang still.
            def batches_per_epoch(trainer) -> int:
                try:
                    return max(1, len(trainer.train_loader))
                except Exception:
                    return 1

            def on_batch_end(trainer):
                if self.is_stopped:
                    trainer.stop = True
                    return
                bpe = batches_per_epoch(trainer)
                done = getattr(trainer, "_ft_batch", 0) + 1
                trainer._ft_batch = done % bpe
                ep = trainer.epoch + 1
                tot = trainer.epochs
                step = trainer.epoch * bpe + done
                MessageProtocol.progress(
                    epoch=ep, total_epochs=tot, step=step, total_steps=tot * bpe,
                    train_loss=self._running_loss(trainer),
                    learning_rate=self._current_lr(trainer))

            def on_fit_epoch_end(trainer):
                """Feuert nach Training UND Validierung einer Epoche.

                Vorher hing der Callback an 'on_train_epoch_end' – der laeuft vor
                der Validierung, also enthielt trainer.metrics noch die Werte der
                vorherigen Epoche (Epoche 1 meldete 0.0, Epoche 2 den Wert von 1).
                Die Losses stehen ausserdem nicht in trainer.metrics, sondern in
                label_loss_items(trainer.tloss) – deshalb war der Loss immer 0.0000.
                """
                if self.is_stopped:
                    trainer.stop = True
                    return
                m = dict(trainer.metrics or {})
                ep = trainer.epoch + 1
                tot = trainer.epochs
                # Ultralytics feuert diesen Callback auch nach der finalen
                # Validierung, mit bereits hochgezaehltem trainer.epoch. Das
                # ergab "Epoch 3 / 2 · Step 174 / 116" und ueberschrieb die
                # Val-Loss-Kachel mit einem leeren Wert.
                if ep > tot:
                    return
                bpe = batches_per_epoch(trainer)
                loss = self._running_loss(trainer)
                val_loss = self._sum_prefixed(m, "val/")
                map50 = float(m.get("metrics/mAP50(B)", 0.0) or 0.0)
                map5095 = float(m.get("metrics/mAP50-95(B)", 0.0) or 0.0)
                metrics = {
                    "mAP50": map50,
                    "mAP50-95": map5095,
                    "precision": float(m.get("metrics/precision(B)", 0.0) or 0.0),
                    "recall":    float(m.get("metrics/recall(B)",    0.0) or 0.0),
                }
                MessageProtocol.progress(
                    epoch=ep, total_epochs=tot, step=ep * bpe, total_steps=tot * bpe,
                    train_loss=loss, val_loss=val_loss,
                    learning_rate=self._current_lr(trainer), metrics=metrics)
                MessageProtocol.status("train",
                    f"[Metric] epoch={ep}/{tot} loss={loss:.4f} mAP50={map50:.4f} mAP50-95={map5095:.4f}")

            self.model.add_callback("on_train_batch_end", on_batch_end)
            self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            self.results = self.model.train(
                data=self._yaml_path,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.imgsz,
                device=device,
                patience=self.patience,
                augment=self.augment,
                optimizer=self.optimizer_name,
                lr0=self.lr0, lrf=self.lrf,
                momentum=self.momentum,
                weight_decay=self.wd,
                project=str(self._output_dir),
                name="train",
                exist_ok=True,
                verbose=False, plots=False, save=True,
            )
            return True
        except Exception as e:
            import traceback
            MessageProtocol.error("YOLO Training Fehler", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            return False

    def validate(self) -> Dict[str, float]:
        if not self.results: return {}
        try:
            m = self.results.results_dict or {}
            return {
                "mAP50":    float(m.get("metrics/mAP50(B)",    0)),
                "mAP50-95": float(m.get("metrics/mAP50-95(B)", 0)),
                "precision": float(m.get("metrics/precision(B)", 0)),
                "recall":    float(m.get("metrics/recall(B)",    0)),
            }
        except Exception: return {}

    def save_model(self, output_path: str, **_) -> bool:
        try:
            run_dir = self._output_dir / "train"
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            src = best if best.exists() else last if last.exists() else None
            if not src:
                MessageProtocol.error("Save", f"Kein Checkpoint in {run_dir/'weights'}"); return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), output_path)
            meta = {"framework":"ultralytics","base_model":self.yolo_model,
                    "task":self.task,"imgsz":self.imgsz,"yaml_path":self._yaml_path,
                    "metrics":self.validate()}
            with open(Path(output_path).with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)
            MessageProtocol.status("save", f"YOLO-Modell gespeichert: {output_path}")
            return True
        except Exception as e:
            MessageProtocol.error("Save Fehler", str(e)); return False

    def _split_sizes(self) -> Dict[str, int]:
        """Zaehlt die Bilder je Split anhand der dataset.yaml.

        Die Analyse-Seite zeigte sonst "n_train 0 / n_val 0", obwohl 463 bzw.
        116 Bilder trainiert wurden.
        """
        counts = {"n_train": 0, "n_val": 0}
        if not self._yaml_path:
            return counts
        yaml_path = Path(self._yaml_path)
        root = yaml_path.parent
        entries: Dict[str, str] = {}
        try:
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].strip()
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k.strip() in ("path", "train", "val"):
                    entries[k.strip()] = v.strip()
        except Exception:
            return counts
        base = Path(entries.get("path", str(root)))
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for key, out in (("train", "n_train"), ("val", "n_val")):
            rel = entries.get(key)
            if not rel:
                continue
            d = Path(rel) if Path(rel).is_absolute() else base / rel
            if d.is_dir():
                counts[out] = sum(1 for f in d.rglob("*")
                                  if f.is_file() and f.suffix.lower() in exts)
        return counts

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "framework":    "ultralytics",
            "base_model":   self.yolo_model,
            "task":         self.task,
            # Von der Analyse-Seite ausgewertet:
            "architecture": Path(self.yolo_model).stem or "yolo",
            "device":       self._device_used,
            "imgsz":        self.imgsz,
            **self._split_sizes(),
            **self.validate(),
        }

    def export(self) -> str:
        try:
            out = Path(self.config.output_path) / "model.pt"
            self.save_model(str(out))
            return str(self.config.output_path)
        except Exception as e:
            MessageProtocol.error("Export Fehler", str(e))
            return str(self.config.output_path)
