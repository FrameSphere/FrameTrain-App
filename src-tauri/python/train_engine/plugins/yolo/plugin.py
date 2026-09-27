"""YOLO Plugin (Ultralytics) — task_type: 'detect'

Ein Plugin fuer alle Ultralytics-Aufgaben: Objekterkennung (detect),
Instanz-Segmentierung (segment), Keypoints (pose), orientierte Boxen (obb) und
Bildklassifikation (classify). Der Orchestrator waehlt genau ein Plugin pro
task_type; ein eigener task_type je YOLO-Aufgabe haette fuenf fast gleiche
Plugins bedeutet. Welche Aufgabe gemeint ist, steht in den Gewichten
(yolo11n-seg.pt, ein trainiertes model.pt) — deshalb bestimmt das Plugin sie
selbst (plugin_config.task = "auto") und laesst sie nur auf Wunsch festlegen.
"""
import json, os, random, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


# ── Aufgaben ────────────────────────────────────────────────────────────────
YOLO_TASKS = ("detect", "segment", "pose", "obb", "classify")

# Was Nutzer in das Textfeld "task" schreiben — alles auf die Ultralytics-Namen.
_TASK_ALIASES = {
    "": "auto", "auto": "auto",
    "detect": "detect", "detection": "detect", "det": "detect", "bbox": "detect",
    "segment": "segment", "seg": "segment", "segmentation": "segment",
    "pose": "pose", "keypoints": "pose", "keypoint": "pose", "kpt": "pose",
    "obb": "obb", "oriented": "obb",
    "classify": "classify", "cls": "classify", "classification": "classify",
}

# Suffixe, an denen Ultralytics die Aufgabe einer Gewichtsdatei erkennt.
TASK_SUFFIX = {"segment": "-seg", "pose": "-pose", "classify": "-cls", "obb": "-obb"}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def normalize_task(value: Any) -> str:
    """'auto' oder einer der Ultralytics-Aufgabennamen; Unbekanntes gilt als 'auto'."""
    return _TASK_ALIASES.get(str(value or "").strip().lower(), "auto")


def task_from_weights_name(name: str) -> Optional[str]:
    """Aufgabe aus dem Dateinamen (yolo11n-seg.pt -> segment).

    None, wenn der Name nichts verraet (model.pt, best.pt) — dann entscheidet
    der Checkpoint selbst.
    """
    stem = Path(str(name or "")).stem.lower()
    if not stem:
        return None
    for task, suffix in TASK_SUFFIX.items():
        if stem.endswith(suffix):
            return task
    if stem.startswith("yolo"):
        return "detect"
    return None


def _has_images(d: Path) -> bool:
    try:
        return any(f.is_file() and f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir())
    except OSError:
        return False


# Ordnernamen, die zu Detektions-/Segmentierungs-Layouts gehoeren und nie eine Klasse sind.
_NOT_A_CLASS = {"images", "labels", "annotations", "train", "val", "valid", "validation", "test"}


def class_dirs(d: Path) -> List[Path]:
    """Unterordner, die direkt Bilder enthalten — bei Ordner-pro-Klasse die Klassen."""
    if not d.is_dir():
        return []
    return sorted(c for c in d.iterdir()
                  if c.is_dir() and c.name.lower() not in _NOT_A_CLASS
                  and not c.name.startswith(".") and _has_images(c))


def is_class_folder_dataset(root: Path) -> bool:
    """Ordner pro Klasse (direkt oder unter train/)? Das liest YOLO-cls."""
    if (root / "labels").is_dir() or any((root / s / "labels").is_dir() for s in ("train", "val", "valid")):
        return False
    return len(class_dirs(root / "train")) >= 2 or len(class_dirs(root)) >= 2


def label_value_counts(label_files: List[Path], max_lines: int = 400) -> List[int]:
    """Anzahl der Zahlen je Label-Zeile — daran unterscheiden sich die Formate."""
    counts: List[int] = []
    for f in label_files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    counts.append(len(parts))
                    if len(counts) >= max_lines:
                        return counts
        except (OSError, UnicodeDecodeError):
            continue
    return counts


def infer_kpt_shape(counts: List[int]) -> Optional[List[int]]:
    """kpt_shape [Punkte, Dimension] aus der Breite der Pose-Labels.

    Eine Pose-Zeile ist 'klasse cx cy w h' plus Punkte*Dimension Werte. Ohne
    kpt_shape in der yaml bricht Ultralytics ab; die App erzeugt ihre yaml aber
    ohne dieses Feld. Dimension 3 (x, y, sichtbar) hat Vorrang, weil COCO und
    Ultralytics sie verwenden.
    """
    widths = {c for c in counts if c > 5}
    if len(widths) != 1:
        return None
    extra = widths.pop() - 5
    if extra % 3 == 0:
        return [extra // 3, 3]
    if extra % 2 == 0:
        return [extra // 2, 2]
    return None


def guess_task_from_labels(counts: List[int]) -> Optional[str]:
    """Aufgabe aus der Form der Label-Zeilen (nur, wenn die Gewichte nichts sagen).

    5 Werte = Box. Genau 9 Werte = orientierte Box (4 Eckpunkte). Gleich breite
    Zeilen mit 5 + n*3 (oder n*2) Werten = Keypoints. Unterschiedlich lange
    Zeilen mit ungerader Breite = Polygone (Segmentierung).
    """
    if not counts:
        return None
    if all(c == 5 for c in counts):
        return "detect"
    if all(c == 9 for c in counts):
        return "obb"
    if infer_kpt_shape(counts) and len(set(counts)) == 1:
        return "pose"
    if all(c >= 7 and (c - 1) % 2 == 0 for c in counts if c != 5):
        return "segment"
    return None


def task_metrics(task: str, m: Dict[str, Any]) -> Dict[str, float]:
    """Ultralytics-Ergebnisse unter den Schluesseln, die die App liest.

    Detect-Schluessel (mAP50, mAP50-95, precision, recall) bleiben wie sie
    waren — auch fuer segment/pose/obb, dort als Box-Werte. Die Masken- bzw.
    Keypoint-Werte kommen als eigene Schluessel dazu. Klassifikation hat keine
    mAP; 'accuracy' ist dort top-1, damit die Analyse-Seite sie anzeigt.
    """
    def g(key: str) -> float:
        try:
            return float(m.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    if task == "classify":
        top1 = g("metrics/accuracy_top1")
        return {"accuracy": top1, "top1_accuracy": top1,
                "top5_accuracy": g("metrics/accuracy_top5")}
    out = {
        "mAP50":     g("metrics/mAP50(B)"),
        "mAP50-95":  g("metrics/mAP50-95(B)"),
        "precision": g("metrics/precision(B)"),
        "recall":    g("metrics/recall(B)"),
    }
    extra = {"segment": ("mask", "M"), "pose": ("pose", "P")}.get(task)
    if extra:
        name, tag = extra
        out[f"{name}_mAP50"] = g(f"metrics/mAP50({tag})")
        out[f"{name}_mAP50-95"] = g(f"metrics/mAP50-95({tag})")
        out[f"{name}_precision"] = g(f"metrics/precision({tag})")
        out[f"{name}_recall"] = g(f"metrics/recall({tag})")
    return out


def resume_requested(value: Any) -> bool:
    """plugin_config.resume: true, "auto" oder ein Pfad heisst fortsetzen."""
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    return bool(s) and s not in ("false", "0", "no", "nein", "off")


class YOLOPlugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        # Bewusst ohne super().__init__: das Verhalten bleibt, wie es war;
        # TrainPlugin dient hier als Vertrag (Pflichtmethoden, stop()).
        self.config = config
        self.model = None
        self.is_stopped = False
        self.results = None
        self._yaml_path: Optional[str] = None
        self._data_arg: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._device_used: str = "cpu"
        pc = config.plugin_config or {}
        self.yolo_model   = pc.get("yolo_model") or ""
        # "auto" (Standard): die Gewichte bestimmen die Aufgabe. Bis setup()
        # sie kennt, gilt detect — so verhaelt sich _resolve_weights wie bisher.
        self.task_setting = normalize_task(pc.get("task"))
        self.task         = self.task_setting if self.task_setting != "auto" else "detect"
        self.resume       = pc.get("resume", False)
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

        Ohne die Methode lief der Signal-Handler der Engine frueher in einen
        AttributeError: "Stoppen" blieb wirkungslos und das Training lief
        bis zur letzten Epoche weiter, obwohl is_stopped ueberall geprueft wird.
        (TrainPlugin.stop() taete dasselbe; die Methode bleibt ausdruecklich.)
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
        root = Path(dsp)
        self._output_dir = Path(self.config.output_path)

        # Erst die Aufgabe, dann die Daten: Klassifikation liest Ordner statt
        # einer dataset.yaml, Pose braucht kpt_shape.
        self.task = self._initial_task(root)
        self.yolo_model = self._resolve_weights()
        if not self._settle_task():
            return False

        if self.task == "classify":
            data_dir = self._prepare_classify_data(root)
            if data_dir is None:
                return False
            self._data_arg = str(data_dir)
        else:
            yaml_path = self._find_or_build_yaml(root)
            if yaml_path is None:
                return False
            # Pascal VOC (XML) in YOLO-Labels umrechnen und Platzhalter-Klassen aus
            # dem Import ersetzen — beides liest Ultralytics sonst nicht.
            try:
                from ft_data.detection import convert_voc_to_yolo, fill_placeholder_names
                note = lambda m: MessageProtocol.status("setup", m)
                convert_voc_to_yolo(root, yaml_path, status=note)
                fill_placeholder_names(root, yaml_path, status=note)
            except Exception as e:
                MessageProtocol.status("setup", f"Label-Vorbereitung uebersprungen: {e}")
            if not self._verify_labels(yaml_path):
                return False
            if self.task == "pose":
                yaml_path = self._ensure_kpt_shape(yaml_path)
                if yaml_path is None:
                    return False
            self._yaml_path = str(yaml_path)
            self._data_arg = self._yaml_path
        self._output_dir.mkdir(parents=True, exist_ok=True)
        MessageProtocol.status("setup",
            f"YOLO Setup OK\n  Model: {self.yolo_model}\n  Task: {self.task}\n  Daten: {self._data_arg}")
        return True

    # ── Aufgabe bestimmen ────────────────────────────────────────────────────
    def _candidate_weights(self) -> List[Path]:
        model_dir = Path(self.config.model_path or "")
        return sorted(model_dir.glob("*.pt")) if model_dir.is_dir() else []

    def _initial_task(self, root: Path) -> str:
        """Vorlaeufige Aufgabe, nach der _resolve_weights die Gewichte waehlt.

        Reihenfolge bei "auto": ausdrueckliche Gewichte (yolo_model), dann die
        Gewichte im Modellordner, wenn sie alle dieselbe Aufgabe haben, dann
        die Form des Datasets. Ein Basisordner wie Ultralytics/YOLO11 enthaelt
        Detect-, Seg- und Pose-Gewichte — dort entscheidet das Dataset.
        """
        if self.task_setting != "auto":
            return self.task_setting
        named = task_from_weights_name(self.yolo_model) if self.yolo_model else None
        if named:
            return named
        tasks = {task_from_weights_name(p.name) for p in self._candidate_weights()}
        tasks.discard(None)
        if len(tasks) == 1:
            return tasks.pop()
        guessed = self._guess_task_from_dataset(root)
        if guessed and (not tasks or guessed in tasks):
            MessageProtocol.status("setup", f"Aufgabe aus dem Dataset erkannt: {guessed}")
            return guessed
        return "detect"

    def _guess_task_from_dataset(self, root: Path) -> Optional[str]:
        if is_class_folder_dataset(root):
            return "classify"
        yaml_path = self._existing_yaml(root)
        if yaml_path is not None:
            try:
                if "kpt_shape" in yaml_path.read_text(encoding="utf-8"):
                    return "pose"
            except OSError:
                pass
        label_files = sorted(root.rglob("*.txt"))[:80]
        label_files = [f for f in label_files
                       if f.name.lower() not in ("classes.txt", "labels.txt", "readme.txt")]
        return guess_task_from_labels(label_value_counts(label_files))

    def _weights_task(self, weights: str) -> Optional[str]:
        """Die Aufgabe, die die Gewichte wirklich haben.

        Ein trainiertes model.pt verraet sie nicht im Namen; der Checkpoint
        selbst (bzw. die daneben geschriebene model.json) schon.
        """
        p = Path(weights)
        meta = p.with_suffix(".json")
        if p.is_file() and meta.is_file():
            try:
                t = json.loads(meta.read_text(encoding="utf-8")).get("task")
                if t in YOLO_TASKS:
                    return t
            except (OSError, ValueError):
                pass
        if p.is_file():
            try:
                import contextlib, io
                from ultralytics import YOLO
                with contextlib.redirect_stdout(io.StringIO()):
                    t = getattr(YOLO(str(p)), "task", None)
                if t in YOLO_TASKS:
                    return t
            except Exception:
                pass
        return task_from_weights_name(p.name)

    def _settle_task(self) -> bool:
        """Gleicht die vorlaeufige Aufgabe mit den gewaehlten Gewichten ab."""
        actual = self._weights_task(self.yolo_model)
        if not actual or actual == self.task:
            return True
        if self.task_setting == "auto":
            MessageProtocol.status("setup",
                f"Die Gewichte {Path(self.yolo_model).name} sind ein {actual}-Modell — Aufgabe: {actual}")
            self.task = actual
            return True
        suffix = TASK_SUFFIX.get(self.task_setting, "")
        MessageProtocol.error(
            "YOLO-Aufgabe passt nicht zu den Gewichten",
            f"Eingestellt ist task={self.task_setting}, die Gewichte "
            f"{Path(self.yolo_model).name} sind aber ein {actual}-Modell.\n"
            + (f"Fuer {self.task_setting} werden Gewichte wie yolo11n{suffix}.pt gebraucht "
               "(im Modellordner ablegen oder per yolo_model waehlen), "
               if suffix else "Fuer detect werden Gewichte ohne Aufgaben-Suffix gebraucht (z.B. yolo11n.pt), ")
            + "oder task auf 'auto' stellen.")
        return False

    # Rueckwaertskompatibel: frueher Klassenattribut.
    _TASK_SUFFIX = TASK_SUFFIX
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
        if self.task in ("segment", "pose", "obb"):
            return self._verify_label_format(yaml_path)
        return True

    # Was jede Aufgabe in einer Label-Zeile erwartet — fuer die Fehlermeldung.
    _FORMAT_HINT = {
        "segment": "klasse x1 y1 x2 y2 x3 y3 ... (Polygon mit mindestens 3 Punkten, normiert 0-1)",
        "pose":    "klasse cx cy w h px1 py1 [v1] px2 py2 [v2] ... (gleich viele Punkte je Zeile)",
        "obb":     "klasse x1 y1 x2 y2 x3 y3 x4 y4 (die vier Ecken der gedrehten Box)",
    }

    def _verify_label_format(self, yaml_path: Path) -> bool:
        """Prueft, ob die Labels zur Aufgabe passen.

        Ein Segmentierungs-Training auf reinen Box-Labels laeuft sonst durch,
        aber der Masken-Loss bleibt 0 und die Masken-mAP ebenso — dasselbe
        Bild wie frueher bei Detektion ohne Labels.
        """
        counts = label_value_counts(self._label_files_for(yaml_path))
        if not counts:
            return True
        if self.task == "segment":
            valid = [c for c in counts if c >= 7 and (c - 1) % 2 == 0]
        elif self.task == "obb":
            valid = [c for c in counts if c == 9]
        else:  # pose
            shape = self._read_yaml(yaml_path).get("kpt_shape")
            try:
                want = 5 + int(shape[0]) * int(shape[1]) if shape else None
            except (TypeError, ValueError, IndexError):
                want = None
            if want is None:
                inferred = infer_kpt_shape(counts)
                want = 5 + inferred[0] * inferred[1] if inferred else None
            valid = [c for c in counts if want is not None and c == want]
        if not valid:
            widths = sorted(set(counts))[:6]
            MessageProtocol.error(
                f"Labels passen nicht zur Aufgabe '{self.task}'",
                f"Erwartet je Zeile: {self._FORMAT_HINT[self.task]}\n"
                f"Gefunden: Zeilen mit {', '.join(map(str, widths))} Werten"
                + (" — das sind Box-Labels fuer Objekterkennung.\nFuer diese Labels passt "
                   "ein Detect-Modell (z.B. yolo11n.pt) oder task='detect'."
                   if widths == [5] else "."))
            return False
        if len(valid) < len(counts):
            MessageProtocol.status("setup",
                f"Warnung: {len(counts) - len(valid)} von {len(counts)} geprueften Label-Zeilen "
                f"passen nicht zum Format fuer {self.task}.")
        return True

    def _existing_yaml(self, root: Path) -> Optional[Path]:
        # Die App traegt die bereits gepruefte (und ggf. reparierte) yaml ein.
        given = (self.config.plugin_config or {}).get("dataset_yaml_path")
        if given and Path(given).is_file():
            return Path(given)
        for c in [root/"dataset.yaml", root/"data.yaml"]:
            if c.exists():
                return c
        return None

    def _find_or_build_yaml(self, root: Path) -> Optional[Path]:
        found = self._existing_yaml(root)
        if found is not None:
            return found
        MessageProtocol.status("setup", "Kein dataset.yaml — generiere automatisch...")
        return self._generate_yaml(root)

    # ── Aufgabenspezifische Daten ────────────────────────────────────────────
    def _label_files_for(self, yaml_path: Path, sample: int = 60) -> List[Path]:
        files: List[Path] = []
        for d in self._yaml_image_dirs(yaml_path):
            imgs = [f for f in sorted(d.rglob("*"))
                    if f.is_file() and f.suffix.lower() in self._IMG_EXTS][:sample]
            for im in imgs:
                lab = next((c for c in self._label_candidates(im) if c.exists()), None)
                if lab is not None:
                    files.append(lab)
        return files

    @staticmethod
    def _read_yaml(yaml_path: Path) -> Dict[str, Any]:
        try:
            import yaml  # kommt mit ultralytics
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _ensure_kpt_shape(self, yaml_path: Path) -> Optional[Path]:
        """Pose braucht kpt_shape in der yaml — fehlt es, aus den Labels ableiten.

        Die dataset.yaml der App (oder eine selbst geschriebene) kennt das Feld
        nicht; Ultralytics bricht dann mit einem KeyError ab. Die Ableitung
        landet in einer Kopie im Ausgabeordner, das Dataset bleibt unberuehrt.
        """
        data = self._read_yaml(yaml_path)
        if data.get("kpt_shape"):
            return yaml_path
        shape = infer_kpt_shape(label_value_counts(self._label_files_for(yaml_path)))
        if not shape:
            MessageProtocol.error(
                "Keypoints: kpt_shape fehlt",
                f"In {yaml_path} steht kein kpt_shape, und aus den Labels laesst es sich "
                "nicht ableiten (alle Zeilen muessen gleich breit sein: klasse cx cy w h "
                "+ Punkte x Dimension).\nTrage es in die dataset.yaml ein, z.B. "
                "'kpt_shape: [17, 3]' fuer COCO-Keypoints.")
            return None
        base = data.get("path")
        base_path = Path(base) if base else yaml_path.parent
        if not base_path.is_absolute():
            base_path = (yaml_path.parent / base_path).resolve()
        data["path"] = str(base_path)
        data["kpt_shape"] = shape
        try:
            import yaml
            self._output_dir.mkdir(parents=True, exist_ok=True)
            derived = self._output_dir / "dataset_pose.yaml"
            derived.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                               encoding="utf-8")
        except Exception as e:
            MessageProtocol.error("Keypoints: yaml nicht schreibbar", str(e))
            return None
        MessageProtocol.status("setup",
            f"kpt_shape {shape} aus den Labels abgeleitet (Kopie: {derived.name})")
        return derived

    def _prepare_classify_data(self, root: Path) -> Optional[Path]:
        """Ordner-pro-Klasse fuer YOLO-cls.

        Ultralytics erwartet <root>/train/<klasse>/bild.jpg und val/ (oder test/).
        Liegt nur <root>/<klasse>/ vor oder fehlt ein Validierungsteil, entsteht
        im Arbeitsordner des Jobs eine 80/20-Aufteilung aus Verknuepfungen — das
        Dataset selbst wird nicht umgebaut.
        """
        train = class_dirs(root / "train")
        val_name = next((s for s in ("val", "validation", "test") if class_dirs(root / s)), None)
        if len(train) >= 2 and val_name:
            return root
        if len(train) >= 2:
            source = {c.name: c for c in train}
            extra_val = {c.name: c for c in class_dirs(root / "valid")}
        else:
            source = {c.name: c for c in class_dirs(root)}
            extra_val = {}
        if len(source) < 2:
            MessageProtocol.error(
                "Klassifikation: keine Klassenordner gefunden",
                f"YOLO-cls erwartet einen Ordner pro Klasse (mindestens zwei), z.B.\n"
                f"  {root}/train/katze/bild.jpg\n  {root}/train/hund/bild.jpg\n"
                f"oder direkt {root}/katze/, {root}/hund/.")
            return None
        # Nicht in den Modellordner: der wird als Version kopiert, und die
        # Verknuepfungen kaemen dort als volle Bildkopien an.
        work = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else self._output_dir
        out = work / "cls_data"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        rng = random.Random(int(getattr(self.config, "seed", 42) or 42))
        counts = {"train": 0, "val": 0}
        for name, d in source.items():
            imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
            if extra_val:
                parts = {"train": imgs,
                         "val": sorted(f for f in extra_val.get(name, d).iterdir()
                                       if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
                         if name in extra_val else []}
            else:
                rng.shuffle(imgs)
                n_val = max(1, len(imgs) // 5) if len(imgs) >= 2 else 0
                parts = {"train": imgs[n_val:], "val": imgs[:n_val]}
            for split, files in parts.items():
                target = out / split / name
                target.mkdir(parents=True, exist_ok=True)
                for f in files:
                    link = target / f.name
                    try:
                        os.symlink(f.resolve(), link)
                    except OSError:
                        shutil.copy2(f, link)  # Windows ohne Symlink-Recht
                    counts[split] += 1
        if counts["val"] == 0:
            MessageProtocol.error("Klassifikation: zu wenige Bilder",
                                  "Fuer eine Validierung braucht jede Klasse mindestens zwei Bilder.")
            return None
        MessageProtocol.status("setup",
            f"Klassifikation: {len(source)} Klassen, {counts['train']} Trainings- und "
            f"{counts['val']} Validierungsbilder (Aufteilung in {out})")
        return out

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
            if self.task == "pose":
                # Ultralytics bricht ohne kpt_shape ab; die Breite der Labels verraet es.
                shape = infer_kpt_shape(label_value_counts(sorted(root.rglob("*.txt"))[:80]))
                if shape:
                    f.write(f"kpt_shape: {shape}\n")
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
        # Durchgehend bis zur hoechsten ID: Ultralytics prueft 'Klasse < nc'.
        # Mit nur den vorkommenden IDs (z.B. 0, 22, 45 -> nc=3) brach es ab.
        return [f"class_{i}" for i in range(max(class_ids) + 1)] if class_ids else []

    @staticmethod
    def _sum_prefixed(metrics: Dict[str, Any], prefix: str) -> Optional[float]:
        """Summiert die Loss-Anteile eines Praefixes ('train/' oder 'val/').

        box/cls/dfl bei Detektion, dazu seg bzw. pose/kobj; die Klassifikation
        hat nur einen Wert namens 'loss' ('train/loss').
        """
        vals = [float(v) for k, v in metrics.items()
                if k.startswith(prefix) and (k.endswith("_loss") or k == f"{prefix}loss")
                and v is not None]
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
        if not self._data_arg:
            MessageProtocol.error("Training", "setup() nicht aufgerufen.")
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            MessageProtocol.error("Ultralytics", "Import fehlgeschlagen.")
            return False
        try:
            import torch
            resume_ckpt = self._resume_checkpoint() if resume_requested(self.resume) else None
            start_weights = str(resume_ckpt) if resume_ckpt else self.yolo_model
            MessageProtocol.status("train", f"Lade {start_weights}...")
            self.model = YOLO(start_weights)
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
                metrics = task_metrics(self.task, m)
                # Fuer die Analyse-Seite: dort stand bei YOLO immer "Final Train Loss 0".
                self._last_losses = (loss, val_loss)
                MessageProtocol.progress(
                    epoch=ep, total_epochs=tot, step=ep * bpe, total_steps=tot * bpe,
                    train_loss=loss, val_loss=val_loss,
                    learning_rate=self._current_lr(trainer), metrics=metrics)
                MessageProtocol.status("train", f"[Metric] epoch={ep}/{tot} loss={loss:.4f} "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                               if k not in ("precision", "recall", "top1_accuracy")))

            self._callbacks = {"on_train_batch_end": [on_batch_end],
                               "on_fit_epoch_end": [on_fit_epoch_end]}
            for event, fns in self._callbacks.items():
                for fn in fns:
                    self.model.add_callback(event, fn)
            if resume_ckpt:
                if self._train_resumed(resume_ckpt, device):
                    return True
                if self.is_stopped:
                    return True
                # Nichts mehr fortzusetzen: normales Training ab diesen Gewichten.
            self.results = self.model.train(
                data=self._data_arg,
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
            self._remember_run_dir()
            return True
        except Exception as e:
            import traceback
            MessageProtocol.error("YOLO Training Fehler", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            return False

    # ── Fortsetzen ───────────────────────────────────────────────────────────
    def _resume_checkpoint(self) -> Optional[Path]:
        """last.pt eines unterbrochenen Laufs (plugin_config.resume).

        Ein Pfad zeigt auf last.pt oder einen Ordner, in dem es liegt (z.B. den
        Job-Ordner des abgebrochenen Laufs). true/"auto" sucht in der
        gewaehlten Version und im eigenen Ausgabeordner. Die App legt fuer jeden
        Start einen neuen Job-Ordner an — ein Lauf von gestern ist deshalb nur
        ueber seinen Pfad erreichbar.
        """
        value = self.resume
        roots: List[Path] = []
        if isinstance(value, str) and value.strip().lower() not in ("true", "auto", "1", "yes", "ja"):
            roots.append(Path(value.strip()).expanduser())
        else:
            roots += [Path(self.config.model_path or ""), self._output_dir or Path(self.config.output_path)]
        for r in roots:
            if r.is_file() and r.suffix == ".pt":
                return r
            for rel in ("train/weights/last.pt", "weights/last.pt", "last.pt",
                        "final_model/train/weights/last.pt"):
                if (r / rel).is_file():
                    return r / rel
        MessageProtocol.status("setup",
            f"Fortsetzen: kein last.pt gefunden ({', '.join(str(r) for r in roots)}) — "
            "Training startet neu.")
        return None

    def _train_resumed(self, ckpt: Path, device: str) -> bool:
        """Setzt einen Ultralytics-Lauf ab last.pt fort. False = nichts fortzusetzen."""
        MessageProtocol.status("train", f"Setze Training fort ab {ckpt}")
        try:
            # Ultralytics uebernimmt Daten, Epochen und Ausgabeordner aus dem
            # Checkpoint; nur Geraet und Batch duerfen sich aendern.
            self.results = self.model.train(resume=True, device=device,
                                            batch=self.config.batch_size)
        except AssertionError as e:
            # "... training to N epochs is finished, nothing to resume"
            MessageProtocol.status("train",
                f"Fortsetzen nicht moeglich ({e}). Neues Training ab diesen Gewichten.")
            from ultralytics import YOLO
            self.model = YOLO(self.yolo_model)
            for event, fns in getattr(self, "_callbacks", {}).items():
                for fn in fns:
                    self.model.add_callback(event, fn)
            return False
        self._remember_run_dir()
        return True

    def _remember_run_dir(self) -> None:
        """Wohin Ultralytics wirklich geschrieben hat (beim Fortsetzen der alte Ordner)."""
        try:
            save_dir = getattr(getattr(self.model, "trainer", None), "save_dir", None)
            if save_dir:
                self._run_dir = Path(save_dir)
        except Exception:
            pass

    def validate(self) -> Dict[str, float]:
        if not self.results: return {}
        try:
            return task_metrics(self.task, self.results.results_dict or {})
        except Exception: return {}

    def save_model(self, output_path: str, **_) -> bool:
        try:
            run_dir = self._run_dir or (self._output_dir / "train")
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            src = best if best.exists() else last if last.exists() else None
            if not src:
                MessageProtocol.error("Save", f"Kein Checkpoint in {run_dir/'weights'}"); return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), output_path)
            meta = {"framework":"ultralytics","base_model":self.yolo_model,
                    "task":self.task,"imgsz":self.imgsz,"yaml_path":self._yaml_path,
                    "data":self._data_arg,
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
        if self.task == "classify" and self._data_arg:
            # Klassifikation: <daten>/train/<klasse>/*.jpg und val/ bzw. test/.
            base = Path(self._data_arg)
            val = next((s for s in ("val", "validation", "test") if (base / s).is_dir()), None)
            for split, out in (("train", "n_train"), (val, "n_val")):
                if split and (base / split).is_dir():
                    counts[out] = sum(1 for f in (base / split).rglob("*")
                                      if f.suffix.lower() in _IMAGE_EXTS)
            return counts
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
            **self._final_losses(),
        }

    def _final_losses(self) -> Dict[str, float]:
        train_loss, val_loss = getattr(self, "_last_losses", (None, None))
        out: Dict[str, float] = {}
        if train_loss is not None:
            out["final_train_loss"] = float(train_loss)
        if val_loss is not None:
            out["final_val_loss"] = float(val_loss)
        return out

    def export(self) -> str:
        try:
            out = Path(self.config.output_path) / "model.pt"
            self.save_model(str(out))
            return str(self.config.output_path)
        except Exception as e:
            MessageProtocol.error("Export Fehler", str(e))
            return str(self.config.output_path)
