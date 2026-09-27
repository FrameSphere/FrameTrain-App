"""Split-Erkennung fuer Datendateien (Tabellen, CoNLL).

Dieselben Regeln wie im Seq2Seq-Plugin: zuerst Unterordner train/ val/ test/,
sonst Dateinamen im Root (train.jsonl, validation-00000-of-00001.parquet).
Alles ohne erkennbaren Split zaehlt als Training — den Validierungsanteil
trennt dann das Plugin selbst ab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .seq2seq import file_split_name

SPLIT_DIRS = (("train", ("train", "training")),
              ("val", ("val", "valid", "validation", "dev")),
              ("test", ("test", "testing")))


def _files(d: Path, exts: Iterable[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    return [f for f in sorted(d.rglob("*"))
            if f.is_file() and f.suffix.lower() in exts
            and ".frametrain_media" not in f.parts
            and not f.name.startswith(".")]


def split_files(root: Path, exts: Iterable[str]) -> Dict[str, List[Path]]:
    """{"train": [...], "val": [...], "test": [...]} — nur Splits mit Dateien."""
    root = Path(root)
    exts = list(exts)
    out: Dict[str, List[Path]] = {}
    for split, names in SPLIT_DIRS:
        for name in names:
            d = root / name
            if d.is_dir():
                found = _files(d, exts)
                if found:
                    out[split] = found
                    break
    if out:
        return out
    # Kein Split-Ordner: nach Dateinamen zuordnen. Metadateien der
    # Datensatz-Werkstatt (PROVENANCE.csv, EXPORT_REPORT.md) haben keine
    # passende Endung oder heissen nicht wie ein Split — sie landen sonst im
    # Training und sprengen die Spaltenerkennung.
    for f in _files(root, exts):
        if f.stem.upper() in ("PROVENANCE", "EXPORT_REPORT", "README"):
            continue
        split = file_split_name(f.stem) or "train"
        out.setdefault(split, []).append(f)
    if out and "train" not in out:
        # Nur test.jsonl vorhanden: zum Trainieren trotzdem nutzbar.
        first = next(iter(out))
        out["train"] = out.pop(first)
    return out


def evaluation_files(root: Path, exts: Iterable[str]) -> tuple:
    """Dateien fuer einen Testlauf: test vor val vor train. -> (split, files)"""
    found = split_files(root, exts)
    for split in ("test", "val", "train"):
        if found.get(split):
            return split, found[split]
    return None, []
