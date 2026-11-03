"""DataLoaders for Canvas IR training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)

_PY_DIR = str(Path(__file__).resolve().parents[3])
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

from ir import CanvasGraphIR, IRDataSpec  # noqa: E402
from synapse.data import (  # noqa: E402
    MEDIA_TYPES, lm_loaders, mask_loaders, media_loaders, seq2seq_loaders, split_settings, split_subset, step,
    text_loaders,
)


# Erzeugende Modelle und Detektion brauchen viele Schritte. Bei kleinen
# Datensaetzen waeren das nur zwei, drei je Epoche, deshalb wird der
# Trainingsteil wiederholt, bis eine Epoche mindestens so viele Schritte hat.
MIN_STEPS = {"gan": 100, "diffusion": 100, "detection": 50}


class _Repeat(torch.utils.data.Dataset):
    def __init__(self, base, times: int):
        self.base, self.times = base, times

    def __len__(self):
        return len(self.base) * self.times

    def __getitem__(self, i):
        return self.base[i % len(self.base)]


def get_dataloaders(
    ir: CanvasGraphIR,
    dataset_path: str,
    batch_size: int,
    preprocess: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
    train, val = _get_dataloaders(ir, dataset_path, batch_size, preprocess)
    task = getattr(ir.training, "task_type", "")
    wanted = MIN_STEPS.get(task)
    if wanted and 0 < len(train) < wanted:
        times = -(-wanted * batch_size // len(train.dataset))
        _status(f"Kleiner Datensatz: {len(train)} Schritte je Epoche, fuer {task} je Epoche {times}x wiederholt.")
        train = DataLoader(_Repeat(train.dataset, times), batch_size=batch_size, shuffle=True,
                           collate_fn=train.collate_fn, generator=torch.Generator().manual_seed(42))
    return train, val


def _get_dataloaders(
    ir: CanvasGraphIR,
    dataset_path: str,
    batch_size: int,
    preprocess: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
    # preprocess sammelt, was die Vorhersage spaeter wissen muss (Klassennamen,
    # Normalisierung). Es wird mit den Gewichten gespeichert.
    preprocess = preprocess if preprocess is not None else {}
    num_classes    = ir.training.num_classes
    data           = ir.data
    data_type      = data.type if data else "default"
    data_params    = data.params if data else {}
    pipeline       = getattr(data, "pipeline", None) or []
    dsp            = dataset_path or os.environ.get("DATASET_PATH", "")

    if data_type in MEDIA_TYPES or data_type in ("text_loader", "mask_loader", "seq2seq_loader", "detection_loader"):
        if not dsp:
            raise ValueError(f"{data_type}: kein Dataset-Pfad angegeben.")
        if data_type == "detection_loader":
            from synapse.generative import box_loaders
            return box_loaders(dsp, data_params, pipeline, batch_size, num_classes, preprocess, _status)
        if data_type == "seq2seq_loader":
            return seq2seq_loaders(dsp, data_params, pipeline, batch_size, preprocess, _status, _read_text)
        if data_type == "mask_loader":
            return mask_loaders(dsp, data_params, pipeline, batch_size, num_classes, preprocess, _status)
        if data_type == "text_loader" and ir.training.task_type in ("language_model", "generation"):
            return lm_loaders(dsp, pipeline, data_params, batch_size, preprocess, _status, _read_text)
        if data_type == "text_loader":
            return text_loaders(dsp, data_params, pipeline, batch_size, num_classes, preprocess, _status, _read_text)
        return media_loaders(data_type, dsp, data_params, pipeline, batch_size, num_classes, preprocess, _status)

    # ── image_loader ─────────────────────────────────────────────
    if data_type == "image_loader":
        if not dsp:
            raise ValueError(
                "image_loader: kein Dataset-Pfad angegeben.\n"
                "Bitte ein Bild-Dataset (ImageFolder-Struktur) ausw\u00e4hlen: "
                "ein Ordner mit Unterordnern pro Klasse."
            )

        root = Path(dsp)

        # H\u00e4ufige Fehlauswahl 1: YOLO-/Objekterkennungs-Dataset
        # (images/ + labels/ bzw. data.yaml) \u2014 das ist KEIN Klassifikations-Dataset.
        def _has_yolo_pair(d: Path) -> bool:
            return (d / "images").is_dir() and (d / "labels").is_dir()

        if (
            _has_yolo_pair(root) or _has_yolo_pair(root / "train")
            or (root / "dataset.yaml").exists() or (root / "data.yaml").exists()
        ):
            raise ValueError(
                "image_loader: Dieses Dataset hat YOLO-Struktur (images/ + labels/ "
                "bzw. dataset.yaml) \u2014 das ist ein OBJEKTERKENNUNGS-Dataset.\n"
                "Der Canvas-image_loader trainiert Bild-KLASSIFIKATION und erwartet:\n"
                "  dataset/<klasse1>/*.jpg  dataset/<klasse2>/*.jpg\n"
                "  (oder train/<klasse>/... + val/<klasse>/...)\n\n"
                "L\u00f6sungen:\n"
                "  - F\u00fcr Objekterkennung: YOLO-Training im Training-Panel nutzen\n"
                "  - F\u00fcr Klassifikation: Bilder in einen Ordner pro Klasse sortieren "
                "und als neues Dataset importieren"
            )

        # Unterstuetzte Strukturen (gemeinsame Regeln aus ft_data.media):
        #   (A) Ordner pro Klasse, (B) train/ [+ val/ test/] mit Klassenordnern,
        #   (C) HF-Parquet mit Bildspalte + Label (wird einmalig entpackt).
        from torchvision import transforms
        from ft_data.media import resolve_class_layout

        sz  = int(data_params.get("imageSize", 224))
        nrm = data_params.get("normalize", True)
        # channels-Param: 1 = Graustufen; 3/4 = RGB (RGBA wird als RGB geladen)
        try:
            channels = int(str(data_params.get("channels", "3")))
        except (TypeError, ValueError):
            channels = 3
        steps = [transforms.Resize((sz, sz))]
        if channels == 1:
            steps.append(transforms.Grayscale(num_output_channels=1))
        steps.append(transforms.ToTensor())
        if nrm:
            steps.append(
                transforms.Normalize(mean=[0.5], std=[0.5]) if channels == 1
                else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        tfm = transforms.Compose(steps)
        aug = step(pipeline, "augmentation")
        split = split_settings(pipeline)
        train_steps = list(steps)
        if aug is not None:
            extra = []
            if aug.get("flip", True) is True:
                extra.append(transforms.RandomHorizontalFlip())
            if float(aug.get("rotate", 0) or 0) > 0:
                extra.append(transforms.RandomRotation(float(aug["rotate"])))
            if float(aug.get("brightness", 0) or 0) > 0:
                extra.append(transforms.ColorJitter(brightness=float(aug["brightness"])))
            train_steps = extra + train_steps
            if aug.get("noise") is True:
                train_steps.append(transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.02))
        train_tfm = transforms.Compose(train_steps)
        cache_ram = data_params.get("cacheRam") is True or str(data_params.get("cacheRam")).lower() == "true"

        if not root.is_dir():
            raise ValueError(
                f"image_loader: Pfad existiert nicht: {dsp}\n"
                "Bitte einen Ordner mit Klassen-Unterordnern ausw\u00e4hlen."
            )
        try:
            if ir.training.task_type in ("gan", "diffusion", "autoencoder"):
                # Erzeugende Modelle brauchen keine Klassen: ein Ordner reicht.
                layout = _all_images(root, split)
            else:
                layout = resolve_class_layout(root, "image", seed=split["seed"], val_fraction=split["val"])
        except ValueError as e:
            has_parquet = list(root.glob("*.parquet")) or list((root / "train").glob("*.parquet"))
            if has_parquet:
                raise ValueError(
                    "image_loader: Dieses Dataset enth\u00e4lt Parquet-Dateien ohne Bildspalte (Tabellendaten).\n"
                    "Nutze im Canvas den parquet_loader-Node statt des image_loaders \u2014 "
                    "oder w\u00e4hle ein Bild-Dataset mit einem Ordner pro Klasse."
                )
            found = ", ".join(d.name for d in sorted(root.iterdir()) if d.is_dir()) or "(keine)"
            raise ValueError(
                f"image_loader: {e}\n"
                f"Gefundene Unterordner in {dsp}: {found}\n"
                "Erwartet: ein Unterordner pro Klasse mit Bildern (z.B. hund/*.jpg, katze/*.jpg), "
                "optional unter train/ und val/."
            )

        for note in layout.notes:
            _status(f"image_loader: {note}")
        _status(f"image_loader: Klassen ({len(layout.classes)}): {layout.classes[:10]} | "
                f"Train: {len(layout.train)} | Val: {len(layout.val)}")

        rgb = channels != 1
        preprocess["class_names"] = [str(c) for c in layout.classes]

        class _Files(torch.utils.data.Dataset):
            def __init__(self, items, transform):
                self.items = list(items)
                self.transform = transform
                self.cache = {}

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                from PIL import Image
                path, label = self.items[i]
                image = self.cache.get(i)
                if image is None:
                    with Image.open(path) as im:
                        image = im.convert("RGB") if rgb else im.convert("L")
                    if cache_ram:
                        self.cache[i] = image
                return self.transform(image), label

        if num_classes and len(layout.classes) > int(num_classes):
            raise ValueError(
                f"image_loader: Das Dataset hat {len(layout.classes)} Klassen ({layout.classes[:10]}), "
                f"das Modell ist auf {num_classes} Klassen eingestellt.\n"
                "Setze die Klassenanzahl im Synapse Builder auf die Anzahl der Klassenordner."
            )
        gen = torch.Generator().manual_seed(42)
        return (
            DataLoader(_Files(layout.train, train_tfm), batch_size=batch_size, shuffle=True, num_workers=0, generator=gen),
            DataLoader(_Files(layout.val, tfm), batch_size=batch_size, shuffle=False, num_workers=0),
        )

    # ── csv_loader / parquet_loader ─────────────────────────────
    if data_type in ("csv_loader", "parquet_loader"):
        regression = getattr(ir.training, "task_type", "") == "regression"
        return _tabular_loaders(data_type, dsp, data_params, num_classes, batch_size, regression, preprocess, pipeline)

    # ── Unbekannter data_type ────────────────────────────────────────
    # KEIN stiller Dummy-Fallback mehr -- expliziter Fehler damit der User
    # weiss was er falsch konfiguriert hat.
    supported = ["image_loader", "audio_loader", "video_loader", "text_loader", "csv_loader", "parquet_loader"]
    raise ValueError(
        f"Canvas data_type '{data_type}' wird nicht unterst\u00fctzt.\n"
        f"Unterst\u00fctzte Typen: {supported}\n"
        f"Dataset-Pfad: {dsp or '(nicht gesetzt)'}\n\n"
        "L\u00f6sungen:\n"
        "  - Im Synapse Builder den Daten-Node auf einen g\u00fcltigen Typ setzen\n"
        "  - Ein kompatibles Dataset ausw\u00e4hlen (Bilder f\u00fcr image_loader, CSV f\u00fcr csv_loader)"
    )


# ── Tabellen (CSV / Parquet) ───────────────────────────────────────────────────
#
# Frueher gemachte Fehler, die hier ausgeschlossen werden:
#   * Bei train.csv / val.csv / test.csv im Root wurde irgendeine Datei genommen
#     (Reihenfolge von glob) — eventuell test.csv, die Val-Datei wurde ignoriert.
#   * Text-Labels ("cat") und Labels ab 1 statt 0 stuerzten in CrossEntropy ab.
#   * Nicht-numerische Feature-Spalten liessen die CSV-Umwandlung abstuerzen.
#   * Zeilen ohne Label (-1, leer) wurden als Klasse mittrainiert.

_SPLIT_WORDS = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing"),
}


def _split_of(name: str) -> Optional[str]:
    import re
    low = name.lower()
    for canon, words in _SPLIT_WORDS.items():
        if any(re.search(rf"(^|[^a-z]){w}([^a-z]|$)", low) for w in words):
            return canon
    return None


def _tabular_files(root: Path, exts: Tuple[str, ...]):
    """(train-Dateien, val-Dateien) — test wird nie zum Training genutzt."""
    if root.is_file():
        return [root], []

    def files(d: Path):
        return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts) if d.is_dir() else []

    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    train = next((files(subdirs[w]) for w in _SPLIT_WORDS["train"] if w in subdirs and files(subdirs[w])), [])
    val = next((files(subdirs[w]) for w in _SPLIT_WORDS["val"] if w in subdirs and files(subdirs[w])), [])
    if train:
        return train, val

    loose = files(root)
    named_train = [f for f in loose if _split_of(f.stem) == "train"]
    named_val = [f for f in loose if _split_of(f.stem) == "val"]
    if named_train:
        return named_train, named_val
    return [f for f in loose if _split_of(f.stem) != "test"] or loose, []


def _all_images(root: Path, split):
    from types import SimpleNamespace
    from ft_data.media import IMAGE_EXTS
    from synapse.data import split_indices
    files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"Keine Bilder in {root}")
    tr, va = split_indices(len(files), split)
    return SimpleNamespace(classes=["image"], train=[(files[i], 0) for i in tr], val=[(files[i], 0) for i in va], notes=[])


def _read_text(root: Path):
    """Text-Tabellen als (train, val oder None); test wird nie trainiert."""
    import pandas as pd

    def read(f: Path):
        if f.suffix.lower() == ".parquet":
            return pd.read_parquet(str(f))
        if f.suffix.lower() in (".jsonl", ".json"):
            return pd.read_json(str(f), lines=f.suffix.lower() == ".jsonl")
        return pd.read_csv(str(f), sep="\t" if f.suffix.lower() == ".tsv" else ",", encoding="utf-8-sig")

    train, val = _tabular_files(root, (".csv", ".tsv", ".jsonl", ".json", ".parquet"))
    if not train:
        raise ValueError(f"text_loader: Keine Textdateien in {root}. Erwartet: csv, tsv, jsonl oder parquet.")
    df = pd.concat([read(f) for f in train], ignore_index=True)
    return df, (pd.concat([read(f) for f in val], ignore_index=True) if val else None)


def _is_unlabeled(v) -> bool:
    if v is None:
        return True
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v < 0
    return str(v).strip() in ("", "-1", "nan", "None")


def _tabular_loaders(data_type: str, dsp: str, data_params: Dict[str, Any], num_classes, batch_size: int,
                     regression: bool = False, preprocess: Optional[Dict[str, Any]] = None, pipeline=()):
    import numpy as np
    import pandas as pd

    name = data_type
    if not dsp:
        raise ValueError(f"{name}: kein Dataset-Pfad angegeben.")
    root = Path(dsp)
    if not root.exists():
        raise ValueError(f"{name}: Pfad nicht gefunden: {dsp}")

    if data_type == "csv_loader":
        exts = (".csv", ".tsv", ".txt")
        sep = str(data_params.get("separator", ",")) or ","
        if sep in ("\\t", "\\\\t"):
            sep = "\t"
        has_header = data_params.get("hasHeader", True)
        has_header = has_header is True or str(has_header).lower() in ("true", "1", "yes")

        def read(f: Path):
            return pd.read_csv(str(f), sep="\t" if f.suffix.lower() == ".tsv" else sep,
                               header=0 if has_header else None, encoding="utf-8-sig")
    else:
        exts = (".parquet",)

        def read(f: Path):
            return pd.read_parquet(str(f))

    train_files, val_files = _tabular_files(root, exts)
    if not train_files:
        raise ValueError(
            f"{name}: Keine {'/'.join(exts)}-Dateien gefunden in {dsp}.\n"
            "Erwartet: Dateien im Root (train.* / val.*) oder in train/ und val/."
        )
    try:
        df_train = pd.concat([read(f) for f in train_files], ignore_index=True)
        df_val = pd.concat([read(f) for f in val_files], ignore_index=True) if val_files else None
    except Exception as e:
        raise ValueError(f"{name}: Fehler beim Lesen von {[f.name for f in train_files + val_files]}: {e}")
    if df_train.empty:
        raise ValueError(f"{name}: {train_files[0].name} ist leer.")
    if len(df_train.columns) < 2:
        raise ValueError(
            f"{name}: Nur {len(df_train.columns)} Spalte(n) — benoetigt werden Features + Label. "
            f"Spalten: {list(df_train.columns)}"
        )

    tgt = data_params.get("targetCol", "label")
    if tgt not in df_train.columns:
        tgt = df_train.columns[-1]

    def clean(df):
        # Regression: negative Zielwerte sind gueltig, nur leere Zellen fallen raus.
        mask = df[tgt].notna() if regression else ~df[tgt].map(_is_unlabeled)
        return df[mask].reset_index(drop=True), int((~mask).sum())

    df_train, dropped = clean(df_train)
    if df_val is not None:
        df_val, dropped_val = clean(df_val)
        dropped += dropped_val
    if dropped:
        _status(f"{name}: {dropped} Zeilen ohne Label (leer/-1) ignoriert.")

    feat_cols = [c for c in df_train.columns
                 if c != tgt and not str(c).startswith("__") and pd.api.types.is_numeric_dtype(df_train[c])]
    skipped = [c for c in df_train.columns if c != tgt and c not in feat_cols and not str(c).startswith("__")]
    if not feat_cols:
        raise ValueError(
            f"{name}: Keine numerischen Feature-Spalten gefunden.\n"
            f"Spalten: {skipped} | Label: '{tgt}'\n\n"
            "Der Canvas-Loader erwartet TABELLARISCHE Daten (Zahlen-Features + Label-Spalte).\n"
            "Text-Datasets brauchen Tokenizer + Embedding — nutze dafuer das normale "
            "Training-Panel (z.B. Sequenzklassifikation)."
        )
    if skipped:
        _status(f"{name}: nicht-numerische Spalten ignoriert: {skipped[:8]}")

    preprocess = preprocess if preprocess is not None else {}
    preprocess["feature_columns"] = [str(c) for c in feat_cols]
    do_norm = data_params.get("normalize", False)
    do_norm = do_norm is True or str(do_norm).lower() in ("true", "1", "yes")

    def fit_norm(X):
        """Mittelwert und Streuung aus Train, gespeichert fuer die Vorhersage."""
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True)
        std[std == 0] = 1.0
        preprocess["mean"] = mean.squeeze(0).tolist()
        preprocess["std"] = std.squeeze(0).tolist()
        return mean, std

    if regression:
        def to_reg(df):
            feats = df[feat_cols].astype("float32").fillna(0.0)
            return torch.tensor(feats.values), torch.tensor(df[tgt].astype("float32").values)
        X_train, y_train = to_reg(df_train)
        reg_mean = reg_std = None
        if do_norm:
            reg_mean, reg_std = fit_norm(X_train)
            X_train = (X_train - reg_mean) / reg_std
        gen = torch.Generator().manual_seed(42)
        if df_val is not None and len(df_val):
            X_val, y_val = to_reg(df_val)
            if reg_mean is not None:
                X_val = (X_val - reg_mean) / reg_std
            return (DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=gen),
                    DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size))
        tr, va = split_subset(TensorDataset(X_train, y_train), split_settings(pipeline))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=gen), DataLoader(va, batch_size=batch_size))

    # Labels immer auf 0..K-1 abbilden: Text-Labels und Labels ab 1 funktionieren so auch.
    values = list(df_train[tgt]) + (list(df_val[tgt]) if df_val is not None and tgt in df_val.columns else [])
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)
    keys = sorted({float(v) for v in values}) if numeric else sorted({str(v) for v in values})
    label_map = {k: i for i, k in enumerate(keys)}
    preprocess["class_names"] = [str(int(k)) if numeric and float(k).is_integer() else str(k) for k in keys]
    if len(keys) < 2:
        raise ValueError(f"{name}: Label-Spalte '{tgt}' hat nur einen Wert ({keys[:1]}) — keine Klassifikation moeglich.")
    if num_classes and len(keys) > int(num_classes):
        raise ValueError(
            f"{name}: Label-Spalte '{tgt}' hat {len(keys)} Klassen, das Modell ist auf {num_classes} eingestellt.\n"
            "Setze die Klassenanzahl im Synapse Builder passend (oder waehle die richtige Label-Spalte)."
        )
    if not numeric or keys != [float(i) for i in range(len(keys))]:
        _status(f"{name}: Labels auf 0..{len(keys) - 1} abgebildet: {dict(list(zip(keys, range(len(keys))))[:10])}")

    def to_tensors(df):
        feats = df[feat_cols].astype("float32")
        if feats.isna().any().any():
            feats = feats.fillna(0.0)
        X = torch.tensor(feats.values)
        y = torch.tensor([label_map[float(v) if numeric else str(v)] for v in df[tgt]], dtype=torch.long)
        return X, y

    X_train, y_train = to_tensors(df_train)
    mean = std = None
    if do_norm:
        mean, std = fit_norm(X_train)
        X_train = (X_train - mean) / std
    ds_train = TensorDataset(X_train, y_train)

    gen = torch.Generator().manual_seed(42)
    if df_val is not None and len(df_val):
        X_val, y_val = to_tensors(df_val)
        if mean is not None:
            X_val = (X_val - mean) / std
        return (DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=gen),
                DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size))

    tr, va = split_subset(ds_train, split_settings(pipeline))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(va, batch_size=batch_size))


def _status(message: str) -> None:
    try:
        from core.protocol import MessageProtocol
        MessageProtocol.status("loading_data", message)
    except Exception:
        print(f"[canvas] {message}", file=sys.stderr)
