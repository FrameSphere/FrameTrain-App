"""
core/dataset_manager.py
=======================
Generische Dataset-Lade-Hilfsfunktionen.

Verwendet von NLP-, Vision- und anderen Plugins.
Unterstützt: Parquet, Arrow, JSONL, JSON, CSV, TSV, TXT, Bildordner.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .protocol import MessageProtocol

# Dateien die beim Scannen übersprungen werden
SKIP_NAMES = {
    "README.md", "readme.md", "dataset_infos.json", "dataset_dict.json",
    "state.json", "datasetdict.json", ".gitattributes", ".gitignore",
}
SKIP_SUFFIXES = {".md", ".gitattributes", ".gitignore", ".yaml", ".yml", ".lock"}

# Split-Aliase (Ordner-Namen die als train/val/test erkannt werden)
SPLIT_ALIASES = {
    "train": ["train", "training", "unused"],
    "val":   ["val", "validation", "valid", "dev"],
    "test":  ["test", "testing"],
}

# Bekannte Text-Spalten (NLP)
TEXT_COLS = [
    "text", "content", "document", "abstract", "body", "passage",
    "sentence", "article", "context", "question", "input", "src",
    "source_text", "input_text", "premise", "hypothesis",
]
TARGET_COLS = [
    "keyphrases", "extractive_keyphrases", "abstractive_keyphrases",
    "keywords", "tags", "summary", "target", "output", "tgt",
    "target_text", "answer", "answers", "label", "labels", "response",
]

# Bild-Endungen
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def list_data_files(path: Path) -> List[Path]:
    """Alle relevanten Datendateien in einem Verzeichnis."""
    return [
        f for f in path.rglob("*")
        if f.is_file()
        and f.name not in SKIP_NAMES
        and f.suffix.lower() not in SKIP_SUFFIXES
    ]


def load_hf_dataset_from_dir(path: Path):
    """
    Lädt ein HuggingFace Dataset aus einem Verzeichnis.
    Probiert Formate in dieser Reihenfolge: parquet, arrow, jsonl, json, csv, tsv, txt.
    """
    from datasets import Dataset, concatenate_datasets

    files = list_data_files(path)
    if not files:
        return None

    by_ext: Dict[str, List[str]] = {}
    for f in files:
        by_ext.setdefault(f.suffix.lower(), []).append(str(f))

    for ext in [".parquet", ".arrow", ".jsonl", ".json", ".csv", ".tsv", ".txt"]:
        group = sorted(by_ext.get(ext, []))
        if not group:
            continue
        try:
            if ext == ".parquet":
                ds = Dataset.from_parquet(group)
            elif ext == ".arrow":
                parts = [Dataset.from_file(f) for f in group]
                ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            elif ext in (".jsonl", ".json"):
                ds = Dataset.from_json(group)
            elif ext in (".csv", ".tsv"):
                ds = Dataset.from_csv(group)
            elif ext == ".txt":
                rows = []
                for fp in group:
                    with open(fp, encoding="utf-8", errors="replace") as fh:
                        rows.extend({"text": ln.strip()} for ln in fh if ln.strip())
                ds = Dataset.from_list(rows)
            else:
                continue
            MessageProtocol.status(
                "loading",
                f"Geladen: {len(ds):,} Zeilen aus {len(group)} {ext}-Datei(en) [{path.name}/]"
            )
            return ds
        except Exception as e:
            MessageProtocol.warning(f"{ext}-Dateien aus {path} nicht lesbar: {e}")

    return None


def load_hf_splits(dataset_path: str) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Lädt train/val/test-Splits als HuggingFace Datasets.
    Gibt (train_ds, val_ds, test_ds) zurück — train_ds ist immer nicht-None.

    Strategien:
    1. HuggingFace DatasetDict (save_to_disk-Format)
    2. Benannte Unterordner (train/, val/, test/)
    3. Gesamtes Root-Verzeichnis als Train-Split
    """
    from datasets import load_from_disk, DatasetDict

    root = Path(dataset_path)
    MessageProtocol.status("loading", f"Lade Datensatz: {root}")

    # Strategie 1: HF DatasetDict
    try:
        ds = load_from_disk(str(root))
        if isinstance(ds, DatasetDict):
            train = ds.get("train")
            val = ds.get("validation") or ds.get("val") or ds.get("dev")
            test = ds.get("test")
            if train is not None:
                MessageProtocol.status("loading", f"DatasetDict-Splits erkannt: {list(ds.keys())}")
                return train, val, test
    except Exception:
        pass

    # Strategie 2: Unterordner
    def _find_split(key):
        for alias in SPLIT_ALIASES[key]:
            p = root / alias
            if p.exists():
                d = load_hf_dataset_from_dir(p)
                if d is not None:
                    return d
        return None

    train_ds = _find_split("train")
    val_ds = _find_split("val")
    test_ds = _find_split("test")

    # Strategie 3: Root als Train
    if train_ds is None:
        MessageProtocol.status("loading", "Keine Split-Ordner — scanne Root-Verzeichnis...")
        train_ds = load_hf_dataset_from_dir(root)

    if train_ds is None:
        exts = {f.suffix.lower() for f in list_data_files(root)}
        raise ValueError(
            f"Keine Trainingsdaten gefunden in: {root}\n"
            f"Gefundene Dateiendungen: {exts or 'keine'}\n"
            f"Unterstützt: .parquet .arrow .jsonl .json .csv .tsv .txt"
        )

    return train_ds, val_ds, test_ds


def find_text_and_target_cols(dataset) -> Tuple[str, Optional[str]]:
    """Erkennt automatisch Text- und Zielspalte aus einem HuggingFace Dataset."""
    cols = dataset.column_names
    text_col = next((c for c in TEXT_COLS if c in cols), None)
    if text_col is None:
        for c in cols:
            sample = dataset[0].get(c)
            if isinstance(sample, str) and len(sample) > 5:
                text_col = c
                break
    if text_col is None:
        raise ValueError(
            f"Keine Text-Spalte gefunden.\n"
            f"Verfügbare Spalten: {cols}\n"
            f"Erwartete Spalten (Beispiele): {TEXT_COLS[:6]}..."
        )
    target_col = next((c for c in TARGET_COLS if c in cols), None)
    return text_col, target_col


def detect_image_dataset_structure(dataset_path: str) -> str:
    """
    Erkennt die Struktur eines Bild-Datensatzes.
    Rückgabe: "imagefolder" | "annotated" | "yolo" | "coco" | "unknown"
    """
    root = Path(dataset_path)
    train_path = root / "train"
    check_path = train_path if train_path.exists() else root

    # Unterordner mit Bildern → ImageFolder-Stil
    subdirs = [d for d in check_path.iterdir() if d.is_dir()]
    if subdirs:
        has_images = any(
            any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file())
            for d in subdirs[:3]
        )
        if has_images:
            return "imagefolder"

    # YOLO labels/-Ordner
    if (check_path / "labels").exists() or (root / "labels").exists():
        return "yolo"

    # COCO annotations.json
    if (check_path / "annotations.json").exists() or (root / "annotations.json").exists():
        return "coco"

    # Flach mit annotations.json
    if (check_path / "annotations.json").exists():
        return "annotated"

    # Bilder direkt im Ordner → versuche ImageFolder
    flat_images = [f for f in check_path.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    if flat_images:
        return "imagefolder"

    return "unknown"
