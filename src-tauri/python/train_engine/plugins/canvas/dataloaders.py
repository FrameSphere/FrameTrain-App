"""DataLoaders for Canvas IR training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

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

        # H\u00e4ufige Fehlauswahl 2: Parquet-/Tabellen-Dataset
        if list(root.glob("*.parquet")) or list((root / "train").glob("*.parquet")):
            raise ValueError(
                "image_loader: Dieses Dataset enth\u00e4lt Parquet-Dateien (Tabellendaten).\n"
                "Nutze im Canvas den parquet_loader-Node statt des image_loaders \u2014 "
                "oder w\u00e4hle ein Bild-Dataset mit einem Ordner pro Klasse."
            )

        # Unterst\u00fctzte Strukturen:
        # (A) ImageFolder-Root: Unterordner = Klassen (train/hund/*, train/katze/*)
        # (B) Split-Root: train/ + val/ Unterordner enthalten selbst Klassen-Unterordner
        from torchvision import datasets, transforms

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

        train_dir = Path(dsp) / "train"
        val_dir   = Path(dsp) / "val"
        valid_dir = Path(dsp) / "valid"  # Roboflow-Standard

        if train_dir.is_dir() and (val_dir.is_dir() or valid_dir.is_dir()):
            # Struktur (B): vorgefertigte Splits
            effective_val = val_dir if val_dir.is_dir() else valid_dir
            try:
                ds_train = datasets.ImageFolder(str(train_dir), transform=tfm)
                ds_val   = datasets.ImageFolder(str(effective_val), transform=tfm)
            except FileNotFoundError as e:
                found = ", ".join(sorted(d.name for d in train_dir.iterdir() if d.is_dir())) or "(keine)"
                raise ValueError(
                    f"image_loader: Konnte train/{effective_val.name}/ nicht als ImageFolder lesen.\n"
                    f"Gefundene Unterordner in train/: {found}\n"
                    f"Erwartet: Unterordner pro Klasse mit Bildern (z.B. train/hund/*.jpg, train/katze/*.jpg).\n"
                    f"Detail: {e}"
                )
            return (
                DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0),
                DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0),
            )

        # Struktur (A): kein vorgefertigter Split
        if not Path(dsp).is_dir():
            raise ValueError(
                f"image_loader: Pfad existiert nicht: {dsp}\n"
                "Bitte einen Ordner mit Klassen-Unterordnern ausw\u00e4hlen."
            )
        subdirs = [d for d in Path(dsp).iterdir() if d.is_dir()]
        if not subdirs:
            raise ValueError(
                f"image_loader: Keine Unterordner in {dsp}.\n"
                "Erwartet wird ein Ordner pro Klasse (z.B. hund/, katze/)."
            )
        try:
            ds = datasets.ImageFolder(dsp, transform=tfm)
        except Exception as e:
            found = ", ".join(sorted(d.name for d in subdirs)) or "(keine)"
            raise ValueError(
                f"image_loader: Fehler beim Laden des Bild-Datasets aus {dsp}.\n"
                f"Gefundene Unterordner: {found}\n"
                f"Erwartet: ein Unterordner pro Klasse mit Bildern (z.B. hund/*.jpg, katze/*.jpg).\n"
                f"Detail: {e}"
            )
        n    = int(len(ds) * 0.8)
        tr, va = random_split(ds, [n, len(ds) - n])
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0),
        )

    # ── csv_loader ───────────────────────────────────────────────
    if data_type == "csv_loader":
        if not dsp:
            raise ValueError(
                "csv_loader: kein Dataset-Pfad angegeben.\n"
                "Bitte eine CSV-Datei ausw\u00e4hlen."
            )

        csv_path = Path(dsp)
        # Unterst\u00fctze: direkte CSV-Datei oder Ordner mit train.csv
        if csv_path.is_dir():
            candidates = list(csv_path.glob("*.csv"))
            train_csv  = csv_path / "train" / (next(iter((csv_path / "train").glob("*.csv")), None) or Path("")).name if (csv_path / "train").is_dir() else None
            if train_csv and train_csv.exists():
                csv_path = train_csv
            elif candidates:
                csv_path = candidates[0]
            else:
                raise ValueError(
                    f"csv_loader: Keine CSV-Datei gefunden in {dsp}.\n"
                    "Erwartet: eine .csv-Datei oder einen Ordner mit train/*.csv."
                )

        if not csv_path.exists():
            raise ValueError(f"csv_loader: Datei nicht gefunden: {csv_path}")

        import pandas as pd
        try:
            tgt = str(data_params.get("targetCol", "label"))
            sep = str(data_params.get("separator", ",")) or ","
            if sep in ("\\t", "\\\\t"):
                sep = "\t"
            has_header = data_params.get("hasHeader", True)
            has_header = has_header is True or str(has_header).lower() in ("true", "1", "yes")
            df = pd.read_csv(str(csv_path), sep=sep, header=0 if has_header else None)
        except Exception as e:
            raise ValueError(f"csv_loader: Fehler beim Lesen von {csv_path}: {e}")

        if df.empty:
            raise ValueError(f"csv_loader: CSV-Datei {csv_path} ist leer.")

        if tgt in df.columns:
            X = torch.tensor(df.drop(columns=[tgt]).values.astype("float32"))
            y = torch.tensor(df[tgt].values, dtype=torch.long)
        elif len(df.columns) >= 2:
            # Fallback: letzte Spalte als Label (auch bei Header-losen CSVs)
            X = torch.tensor(df.values[:, :-1].astype("float32"))
            y = torch.tensor(df.values[:, -1].astype("int64"), dtype=torch.long)
        else:
            raise ValueError(
                f"csv_loader: CSV hat nur {len(df.columns)} Spalte(n). "
                f"Ben\u00f6tigt mindestens 2 (Features + Label). "
                f"Spalten: {list(df.columns)}"
            )

        # normalize-Param: Features spaltenweise auf Mittelwert 0 / Std 1 bringen
        do_norm = data_params.get("normalize", False)
        if do_norm is True or str(do_norm).lower() in ("true", "1", "yes"):
            std = X.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            X = (X - X.mean(dim=0, keepdim=True)) / std

        ds   = TensorDataset(X, y)
        n    = int(len(ds) * 0.8)
        tr, va = random_split(ds, [n, len(ds) - n])
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size),
        )

    # ── parquet_loader ───────────────────────────────────────────
    if data_type == "parquet_loader":
        if not dsp:
            raise ValueError("parquet_loader: kein Dataset-Pfad angegeben.")

        import pandas as pd
        pq_path = Path(dsp)

        # Sammle alle .parquet-Dateien (train-Split bevorzugt)
        def collect_parquets(root: Path):
            train_dir = root / "train"
            if train_dir.is_dir():
                files = sorted(train_dir.glob("*.parquet"))
                if files: return files, root / "val"
            return sorted(root.glob("*.parquet")), None

        train_files, val_dir = collect_parquets(pq_path)
        if not train_files:
            raise ValueError(
                f"parquet_loader: Keine .parquet-Dateien gefunden in {dsp}.\n"
                "Erwartet: .parquet-Dateien im Root oder in einem train/-Unterordner."
            )

        try:
            df_train = pd.concat([pd.read_parquet(str(f)) for f in train_files], ignore_index=True)
        except Exception as e:
            raise ValueError(f"parquet_loader: Fehler beim Lesen der Parquet-Dateien: {e}")

        import numpy as np

        # Val-Split VOR dem Label-Mapping laden, damit die Klassen-IDs aus
        # train UND val gebildet werden (sonst KeyError bei nur-in-val-Labels).
        df_val = None
        if val_dir and val_dir.is_dir():
            val_files = sorted(val_dir.glob("*.parquet"))
            if val_files:
                df_val = pd.concat([pd.read_parquet(str(f)) for f in val_files], ignore_index=True)

        tgt = str(data_params.get("targetCol", "label"))
        if tgt not in df_train.columns:
            # Fallback: letzte Spalte
            tgt = df_train.columns[-1]

        # Nur NUMERISCHE Spalten sind als Features nutzbar — Text-Spalten
        # (z.B. title/abstract) können nicht in Float-Tensoren umgewandelt werden.
        feat_cols = [
            c for c in df_train.columns
            if c != tgt and not c.startswith("__")
            and np.issubdtype(df_train[c].dtype, np.number)
        ]
        skipped = [c for c in df_train.columns if c != tgt and c not in feat_cols and not c.startswith("__")]
        if not feat_cols:
            raise ValueError(
                "parquet_loader: Keine numerischen Feature-Spalten gefunden.\n"
                f"Spalten: {skipped} | Label: '{tgt}'\n\n"
                "Der Canvas-parquet_loader erwartet TABELLARISCHE Daten "
                "(Zahlen-Features + Label-Spalte).\n"
                "Text-Datasets (Titel, Abstracts, Keyphrases …) brauchen "
                "Tokenizer + Embedding — nutze dafür das normale Training-Panel "
                "(z.B. Sequenzklassifikation) statt des Canvas-parquet_loaders."
            )

        # Label: numerisch direkt übernehmen, Strings auf IDs mappen ('prmu' → 0…K).
        # Klassen aus train+val vereinigen, damit kein Split unbekannte Labels hat.
        def label_series(df):
            return df[tgt].astype(str) if not np.issubdtype(df[tgt].dtype, np.number) else df[tgt]

        numeric_label = np.issubdtype(df_train[tgt].dtype, np.number)
        label_map = None
        if not numeric_label:
            classes = set(label_series(df_train))
            if df_val is not None and tgt in df_val.columns:
                classes |= set(label_series(df_val))
            label_map = {c: i for i, c in enumerate(sorted(classes))}

        def to_tensors(df):
            X = torch.tensor(df[feat_cols].values.astype("float32"))
            if numeric_label:
                y = torch.tensor(df[tgt].values.astype("int64"), dtype=torch.long)
            else:
                y = torch.tensor([label_map[v] for v in df[tgt].astype(str)], dtype=torch.long)
            return X, y

        try:
            X_train, y_train = to_tensors(df_train)
        except Exception as e:
            raise ValueError(
                f"parquet_loader: Konnte Spalten nicht in Tensoren umwandeln.\n"
                f"Numerische Feature-Spalten: {feat_cols[:5]} | Label: '{tgt}'"
                + (f" | Ignorierte Text-Spalten: {skipped[:5]}" if skipped else "") + "\n"
                f"Detail: {e}"
            )

        if skipped:
            try:
                from core.protocol import MessageProtocol
                MessageProtocol.status(
                    "loading_data",
                    f"parquet_loader: Text-Spalten ignoriert (nicht numerisch): {skipped[:8]}"
                )
            except Exception:
                print(f"[parquet_loader] Text-Spalten ignoriert: {skipped[:8]}", file=sys.stderr)

        ds_train = TensorDataset(X_train, y_train)

        # Val-Split
        if df_val is not None:
            X_val, y_val = to_tensors(df_val)
            ds_val = TensorDataset(X_val, y_val)
            return (
                DataLoader(ds_train, batch_size=batch_size, shuffle=True),
                DataLoader(ds_val,   batch_size=batch_size),
            )

        n    = int(len(ds_train) * 0.8)
        tr, va = random_split(ds_train, [n, len(ds_train) - n])
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size),
        )

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
