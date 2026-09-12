"""Bild- und Audio-Datasets: Klassenordner, Splits und HuggingFace-Parquet.

Unterstuetzte Layouts:
    <root>/<klasse>/*.jpg                         (ungeteilt)
    <root>/train/<klasse>/*.jpg [+ val/ test/]    (geteilt; auch training/, valid/, validation/, dev/, testing/)
    <root>/*.parquet | <root>/<split>/*.parquet   (HF-Download: Bild/Audio als Bytes + Label)

Frueher gemachte Fehler, die hier ausgeschlossen werden:
  * train/ + test/ ohne val/ wurde als ungeteilter Ordner gelesen — die Klassen
    hiessen dann "test" und "train".
  * torchvision.ImageFolder vergibt Klassen-IDs pro Ordner. Fehlte in val/ eine
    Klasse, verschoben sich alle IDs danach und die Val-Accuracy war falsch.
  * Leere Split-Ordner (der App-Split legt test/<klasse>/ auch bei 0 % an)
    wurden bevorzugt, obwohl nebenan val/ die Daten hatte.
  * HF-Bild-/Audio-Datasets liegen als Parquet vor; die Plugins suchten nur
    Klassenordner und meldeten "0 Klassenordner".
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing"),
}
_ALL_SPLIT_NAMES = {n for names in SPLIT_ALIASES.values() for n in names}

MEDIA_CACHE_DIR = ".frametrain_media"

Item = Tuple[Path, int]


def exts_for(kind: str) -> set:
    return IMAGE_EXTS if kind == "image" else AUDIO_EXTS


def _visible_dirs(d: Path) -> List[Path]:
    try:
        return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "__")))
    except OSError:
        return []


def _media_files(d: Path, exts: set) -> List[Path]:
    return sorted(f for f in d.rglob("*")
                  if f.is_file() and f.suffix.lower() in exts and not f.name.startswith("."))


def class_files(base: Path, exts: set) -> Dict[str, List[Path]]:
    """Klassenordner mit mindestens einer Datei (leere Ordner zaehlen nicht)."""
    out: Dict[str, List[Path]] = {}
    for d in _visible_dirs(base):
        if d.name.lower() in _ALL_SPLIT_NAMES:
            continue
        files = _media_files(d, exts)
        if files:
            out[d.name] = files
    return out


def split_dirs(root: Path, exts: set) -> Dict[str, Path]:
    """Split-Ordner, die wirklich Dateien enthalten (kanonischer Name -> Ordner)."""
    found: Dict[str, Path] = {}
    by_name = {d.name.lower(): d for d in _visible_dirs(root)}
    for canon, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            d = by_name.get(alias)
            if d is not None and class_files(d, exts):
                found[canon] = d
                break
    return found


@dataclass
class ClassLayout:
    classes: List[str]
    train: List[Item]
    val: List[Item]
    test: List[Item]
    root: Path
    val_from_train: bool = False
    notes: List[str] = field(default_factory=list)


def _items(files_by_class: Dict[str, List[Path]], label2id: Dict[str, int]) -> List[Item]:
    return [(f, label2id[c]) for c, files in files_by_class.items() if c in label2id for f in files]


def resolve_class_layout(root: Path, kind: str, seed: int = 42, val_fraction: float = 0.1,
                          status=None) -> ClassLayout:
    """Liest ein Klassifikations-Dataset mit festen Klassen-IDs fuer alle Splits.

    Die Klassen kommen aus dem Trainings-Split. Validierung ist val/; fehlt sie,
    wird ein Anteil von train abgetrennt — NICHT test/, damit die Testdaten
    unberuehrt bleiben.
    """
    root = ensure_media_folders(Path(root), kind, status)
    exts = exts_for(kind)
    notes: List[str] = []
    splits = split_dirs(root, exts)

    if splits:
        if "train" not in splits:
            raise ValueError(
                f"In '{root}' gibt es Split-Ordner ({', '.join(sorted(splits))}), aber keinen "
                "train/-Ordner mit Dateien. Erwartet: train/<klasse>/..."
            )
        train_by_class = class_files(splits["train"], exts)
    else:
        train_by_class = class_files(root, exts)

    classes = sorted(train_by_class)
    if len(classes) < 2:
        base = splits.get("train", root)
        raise ValueError(
            f"In '{base}' wurden {len(classes)} Klassenordner mit Dateien gefunden "
            f"({classes or 'keine'}). Fuer eine Klassifikation braucht es mindestens zwei "
            "Unterordner mit Dateien — einen pro Klasse."
        )
    label2id = {c: i for i, c in enumerate(classes)}
    train = _items(train_by_class, label2id)

    def split_items(name: str) -> List[Item]:
        if name not in splits:
            return []
        by_class = class_files(splits[name], exts)
        unknown = sorted(set(by_class) - set(label2id))
        if unknown:
            notes.append(f"{name}/ enthaelt Klassen, die im Training fehlen und ignoriert werden: {unknown}")
        return _items(by_class, label2id)

    val = split_items("val")
    test = split_items("test")
    val_from_train = False
    if not val:
        rng = random.Random(seed)
        shuffled = train[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        val, train = shuffled[:n_val], shuffled[n_val:]
        val_from_train = True
        notes.append(f"Kein val/-Ordner mit Dateien — {len(val)} Dateien ({val_fraction:.0%}) aus train abgetrennt.")

    return ClassLayout(classes, train, val, test, root, val_from_train, notes)


def evaluation_files(root: Path, kind: str, status=None) -> Tuple[List[Tuple[Path, Optional[str]]], Optional[str]]:
    """Dateien fuer einen Test: test/ vor val/ vor train/, leere Ordner zaehlen nicht.

    Liefert (Datei, erwartete Klasse oder None) und den verwendeten Split-Namen.
    """
    root = ensure_media_folders(Path(root), kind, status)
    exts = exts_for(kind)
    splits = split_dirs(root, exts)
    for name in ("test", "val", "train"):
        if name in splits:
            by_class = class_files(splits[name], exts)
            return [(f, c) for c, files in sorted(by_class.items()) for f in files], name
    by_class = class_files(root, exts)
    if by_class:
        return [(f, c) for c, files in sorted(by_class.items()) for f in files], None
    return [(f, None) for f in _media_files(root, exts)], None


def sample(items: Sequence, n: Optional[int], seed: int = 42) -> list:
    """Zufaellige, reproduzierbare Stichprobe statt der ersten N.

    Die ersten N einer nach Klasse sortierten Liste enthalten oft nur eine Klasse
    — die Accuracy misst dann nichts.
    """
    items = list(items)
    if not n or n >= len(items):
        return items
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(items)), int(n)))
    return [items[i] for i in picked]


# ── HuggingFace-Parquet mit Bild-/Audio-Bytes ────────────────────────────────

_MAGIC = [
    (b"\x89PNG", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"GIF8", ".gif"), (b"BM", ".bmp"),
    (b"fLaC", ".flac"), (b"OggS", ".ogg"), (b"ID3", ".mp3"), (b"\xff\xfb", ".mp3"), (b"\xff\xf3", ".mp3"),
]


def _guess_ext(data: bytes, path_hint: Optional[str], kind: str) -> str:
    if path_hint:
        suffix = Path(str(path_hint)).suffix.lower()
        if suffix in exts_for(kind):
            return suffix
    head = data[:12]
    if head[:4] == b"RIFF":
        return ".wav" if head[8:12] == b"WAVE" else ".webp"
    for magic, ext in _MAGIC:
        if head.startswith(magic):
            return ext
    return ".png" if kind == "image" else ".wav"


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(name)).strip().strip(".")
    return cleaned or "leer"


def _parquet_files(root: Path) -> Dict[str, List[Path]]:
    """Parquet-Dateien je Split: <split>/*.parquet oder train.parquet/test-00000.parquet im Root."""
    by_split: Dict[str, List[Path]] = {}
    for d in _visible_dirs(root):
        canon = next((c for c, al in SPLIT_ALIASES.items() if d.name.lower() in al), None)
        if canon:
            files = sorted(d.glob("*.parquet"))
            if files:
                by_split.setdefault(canon, []).extend(files)
    for f in sorted(root.glob("*.parquet")):
        stem = f.stem.lower()
        canon = next((c for c, al in SPLIT_ALIASES.items() if any(re.search(rf"(^|[^a-z]){a}([^a-z]|$)", stem) for a in al)), "train")
        by_split.setdefault(canon, []).append(f)
    return by_split


def _hf_features(schema) -> dict:
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("info", {}).get("features", {}) or {}
    except ValueError:
        return {}


def _find_columns(schema, kind: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """(Medien-Spalte, Label-Spalte, Klassennamen aus ClassLabel)."""
    import pyarrow as pa

    feats = _hf_features(schema)
    wanted = "Image" if kind == "image" else "Audio"
    media = next((n for n, f in feats.items() if isinstance(f, dict) and f.get("_type") == wanted), None)
    if media is None:
        names = ("image", "img", "picture", "pixel_values") if kind == "image" else ("audio", "sound", "file", "speech")
        for field_ in schema:
            if field_.name.lower() in names and (pa.types.is_struct(field_.type) or pa.types.is_binary(field_.type)):
                media = field_.name
                break
    label = next((n for n, f in feats.items() if isinstance(f, dict) and f.get("_type") == "ClassLabel"), None)
    names: List[str] = list(feats[label].get("names") or []) if label else []
    if label is None:
        cols = [f.name for f in schema]
        label = next((c for c in ("label", "labels", "class", "category", "target", "fine_label", "coarse_label") if c in cols), None)
    return media, label, names


def _signature(files: Dict[str, List[Path]]) -> str:
    h = hashlib.sha1()
    for split in sorted(files):
        for f in files[split]:
            st = f.stat()
            h.update(f"{split}|{f.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def ensure_media_folders(root: Path, kind: str, status=None) -> Path:
    """Gibt einen Ordner mit Klassenordnern zurueck.

    Hat das Dataset schon Klassenordner, ist das der Dataset-Ordner selbst. Liegt
    es als HF-Parquet vor (Bild-/Audiobytes + Label), werden die Dateien einmalig
    nach <root>/.frametrain_media/<split>/<klasse>/ geschrieben und dieser Ordner
    zurueckgegeben. Aendern sich die Parquet-Dateien, wird neu entpackt.
    """
    root = Path(root)
    exts = exts_for(kind)
    if split_dirs(root, exts) or class_files(root, exts):
        return root

    parquet = _parquet_files(root)
    if not parquet:
        return root

    import pyarrow.parquet as pq

    first = next(iter(parquet.values()))[0]
    schema = pq.read_schema(first)
    media_col, label_col, names = _find_columns(schema, kind)
    if media_col is None:
        return root  # Tabellen-/Text-Parquet: kein Medien-Dataset
    if label_col is None:
        raise ValueError(
            f"Parquet-Dataset mit {'Bildern' if kind == 'image' else 'Audiodateien'} in Spalte "
            f"'{media_col}', aber ohne Label-Spalte. Vorhandene Spalten: {[f.name for f in schema]}"
        )

    cache = root / MEDIA_CACHE_DIR
    marker = cache / ".complete.json"
    sig = _signature(parquet)
    try:
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("signature") == sig:
            return cache
    except ValueError:
        pass

    import shutil
    if cache.exists():
        shutil.rmtree(cache)
    if status:
        status(f"Entpacke {'Bilder' if kind == 'image' else 'Audiodateien'} aus Parquet (einmalig)...")

    counts: Dict[str, int] = {}
    for split, files in parquet.items():
        n = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=[media_col, label_col], batch_size=256):
                media_vals = batch.column(0).to_pylist()
                label_vals = batch.column(1).to_pylist()
                for value, label in zip(media_vals, label_vals):
                    if label is None or (isinstance(label, int) and label < 0):
                        continue  # ungelabelte Zeile (HF-Testsplits haben oft -1)
                    if isinstance(value, dict):
                        data, hint = value.get("bytes"), value.get("path")
                    else:
                        data, hint = value, None
                    if not data:
                        continue
                    cls = names[label] if names and isinstance(label, int) and label < len(names) else label
                    target = cache / split / _safe_name(cls)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"{n:07d}{_guess_ext(data, hint, kind)}").write_bytes(data)
                    n += 1
        counts[split] = n

    cache.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"signature": sig, "counts": counts}), encoding="utf-8")
    if status:
        status(f"Parquet entpackt: " + ", ".join(f"{s}: {c}" for s, c in counts.items()))
    return cache
