"""Seq2Seq: Quell-/Zielspalten, Uebersetzungs-Dicts, Listen als Ziel.

Gemeinsam fuer Training und Test. Die im Training gewaehlten Spalten werden
neben dem Modell gespeichert (SPEC_FILE) — der Test brach vorher mit
"Eingabespalte nicht erkannt" ab, sobald eigene Spaltennamen gesetzt waren.

Weitere frueher gemachte Fehler:
  * HF-Uebersetzungs-Datasets (wmt, opus) haben eine Spalte
    translation = {"de": ..., "en": ...} — sie wurde nicht erkannt.
  * Listen als Ziel (z.B. keyphrases) liessen die Tokenisierung abstuerzen.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SOURCE_CANDIDATES = ["source", "input", "text", "article", "document", "de", "src", "question", "abstract"]
TARGET_CANDIDATES = ["target", "output", "summary", "highlights", "translation", "en", "tgt", "answer", "keyphrases"]
SPEC_FILE = "frametrain_seq2seq.json"


def _as_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_as_text(v) for v in value if _as_text(v))
    return str(value)


def _dict_languages(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [k for k, v in value.items() if isinstance(v, str) or v is None]
    return []


def resolve_spec(columns: Sequence[str], first_row: Dict[str, Any], plugin_config: Dict[str, Any]) -> Dict[str, Any]:
    """Waehlt die Spalten. Wirft ValueError mit Hinweis, wenn nichts passt."""
    cfg = plugin_config or {}
    cols = [c for c in columns if not str(c).startswith("__")]

    # Uebersetzung als Dict-Spalte
    dict_col = next((c for c in cols if len(_dict_languages(first_row.get(c))) >= 2), None)
    if dict_col and not (cfg.get("source_column") and cfg.get("target_column")):
        langs = _dict_languages(first_row[dict_col])
        src_lang = cfg.get("source_lang") or langs[0]
        tgt_lang = cfg.get("target_lang") or next(l for l in langs if l != src_lang)
        if src_lang not in langs or tgt_lang not in langs:
            raise ValueError(f"Sprachen {src_lang!r}/{tgt_lang!r} nicht in Spalte '{dict_col}' ({langs}).")
        return {"translation_column": dict_col, "source_lang": src_lang, "target_lang": tgt_lang,
                "source_column": None, "target_column": None}

    source = cfg.get("source_column") or next((c for c in SOURCE_CANDIDATES if c in cols), None)
    target = cfg.get("target_column") or next((c for c in TARGET_CANDIDATES if c in cols and c != source), None)
    if not source or not target or source not in cols or target not in cols:
        raise ValueError(
            f"Eingabe- und Zielspalte nicht erkannt. Vorhandene Spalten: {cols}.\n"
            "Setze sie in der Plugin-Konfiguration, z.B.:\n"
            '  {"source_column": "artikel", "target_column": "kurzfassung"}\n'
            'Bei Uebersetzungen mit einer Dict-Spalte: {"source_lang": "de", "target_lang": "en"}'
        )
    return {"translation_column": None, "source_lang": None, "target_lang": None,
            "source_column": source, "target_column": target}


def describe(spec: Dict[str, Any]) -> str:
    if spec.get("translation_column"):
        return f"'{spec['translation_column']}': {spec['source_lang']} → {spec['target_lang']}"
    return f"'{spec['source_column']}' → '{spec['target_column']}'"


def row_texts(row: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """(Eingabetext, Zieltext oder None) einer Zeile."""
    if spec.get("translation_column"):
        value = row.get(spec["translation_column"]) or {}
        target = value.get(spec["target_lang"]) if isinstance(value, dict) else None
        return _as_text(value.get(spec["source_lang"]) if isinstance(value, dict) else None), \
            (_as_text(target) if target is not None else None)
    target = row.get(spec["target_column"]) if spec.get("target_column") else None
    return _as_text(row.get(spec["source_column"])), (_as_text(target) if target is not None else None)


def batch_texts(batch: Dict[str, List[Any]], spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    keys = list(batch.keys())
    n = len(batch[keys[0]]) if keys else 0
    sources, targets = [], []
    for i in range(n):
        s, t = row_texts({k: batch[k][i] for k in keys}, spec)
        sources.append(s)
        targets.append(t or "")
    return sources, targets


def save_spec(model_dir: Path, spec: Dict[str, Any], prefix: str) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / SPEC_FILE).write_text(
        json.dumps({**spec, "task_prefix": prefix}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spec(model_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
