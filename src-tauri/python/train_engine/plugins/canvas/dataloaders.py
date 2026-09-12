"""DataLoaders for Canvas IR training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys
from pathlib import Path

_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)

from ir import CanvasGraphIR, IRDataSpec  # noqa: E402


def _infer_input_features(ir: CanvasGraphIR) -> int:
    for nid in ir.execution_order:
        node = ir.node_by_id().get(nid)
        if not node:
            continue
        if node.type == "dense":
            return int(node.params.get("inputSize", 128))
        if node.type == "lstm":
            return int(node.params.get("inputSize", 256))
        if node.type == "conv2d":
            return int(node.params.get("inChannels", 3))
        if node.type == "embedding":
            return 32
    return 128


def get_dataloaders(
    ir: CanvasGraphIR,
    dataset_path: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    num_classes    = ir.training.num_classes
    input_features = _infer_input_features(ir)
    data           = ir.data
    data_type      = data.type if data else "default"
    data_params    = data.params if data else {}
    dsp            = dataset_path or os.environ.get("DATASET_PATH", "")

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

        if not root.is_dir():
            raise ValueError(
                f"image_loader: Pfad existiert nicht: {dsp}\n"
                "Bitte einen Ordner mit Klassen-Unterordnern ausw\u00e4hlen."
            )
        try:
            layout = resolve_class_layout(root, "image", val_fraction=0.2)
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

        rgb = channels != 1

        class _Files(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = list(items)

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                from PIL import Image
                path, label = self.items[i]
                with Image.open(path) as im:
                    return tfm(im.convert("RGB") if rgb else im.convert("L")), label

        if num_classes and len(layout.classes) > int(num_classes):
            raise ValueError(
                f"image_loader: Das Dataset hat {len(layout.classes)} Klassen ({layout.classes[:10]}), "
                f"das Modell ist auf {num_classes} Klassen eingestellt.\n"
                "Setze die Klassenanzahl im Synapse Builder auf die Anzahl der Klassenordner."
            )
        gen = torch.Generator().manual_seed(42)
        return (
            DataLoader(_Files(layout.train), batch_size=batch_size, shuffle=True, num_workers=0, generator=gen),
            DataLoader(_Files(layout.val),   batch_size=batch_size, shuffle=False, num_workers=0),
        )

    # ── csv_loader / parquet_loader ─────────────────────────────
    if data_type in ("csv_loader", "parquet_loader"):
        regression = getattr(ir.training, "task_type", "") == "regression"
        return _tabular_loaders(data_type, dsp, data_params, num_classes, batch_size, regression)

    # ── Unbekannter data_type ────────────────────────────────────────
    # KEIN stiller Dummy-Fallback mehr -- expliziter Fehler damit der User
    # weiss was er falsch konfiguriert hat.
    supported = ["image_loader", "csv_loader", "parquet_loader"]
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
                     regression: bool = False):
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
                 if c != tgt and not str(c).startswith("__") and np.issubdtype(df_train[c].dtype, np.number)]
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

    if regression:
        def to_reg(df):
            feats = df[feat_cols].astype("float32").fillna(0.0)
            return torch.tensor(feats.values), torch.tensor(df[tgt].astype("float32").values)
        X_train, y_train = to_reg(df_train)
        gen = torch.Generator().manual_seed(42)
        if df_val is not None and len(df_val):
            X_val, y_val = to_reg(df_val)
            return (DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=gen),
                    DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size))
        ds = TensorDataset(X_train, y_train)
        n = max(1, int(len(ds) * 0.8))
        tr, va = random_split(ds, [n, len(ds) - n], generator=torch.Generator().manual_seed(42))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=gen), DataLoader(va, batch_size=batch_size))

    # Labels immer auf 0..K-1 abbilden: Text-Labels und Labels ab 1 funktionieren so auch.
    values = list(df_train[tgt]) + (list(df_val[tgt]) if df_val is not None and tgt in df_val.columns else [])
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)
    keys = sorted({float(v) for v in values}) if numeric else sorted({str(v) for v in values})
    label_map = {k: i for i, k in enumerate(keys)}
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
    do_norm = data_params.get("normalize", False)
    mean = std = None
    if do_norm is True or str(do_norm).lower() in ("true", "1", "yes"):
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True)
        std[std == 0] = 1.0
        X_train = (X_train - mean) / std
    ds_train = TensorDataset(X_train, y_train)

    gen = torch.Generator().manual_seed(42)
    if df_val is not None and len(df_val):
        X_val, y_val = to_tensors(df_val)
        if mean is not None:
            X_val = (X_val - mean) / std
        return (DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=gen),
                DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size))

    n = max(1, int(len(ds_train) * 0.8))
    tr, va = random_split(ds_train, [n, len(ds_train) - n], generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(va, batch_size=batch_size))


def _status(message: str) -> None:
    try:
        from core.protocol import MessageProtocol
        MessageProtocol.status("loading_data", message)
    except Exception:
        print(f"[canvas] {message}", file=sys.stderr)
