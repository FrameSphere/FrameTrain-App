"""Text-Klassifikation: Spalten, Satzpaare, ungelabelte Zeilen, Label-Namen.

Gemeinsam fuer Training und Test, damit beide dieselben Spalten waehlen.

Frueher gemachte Fehler, die hier ausgeschlossen werden:
  * HF-Testsplits (GLUE u.a.) haben label = -1. Das wurde als eigene Klasse
    gezaehlt — SST-2 bekam 3 statt 2 Klassen.
  * Satzpaar-Aufgaben (MRPC, MNLI, QQP, RTE) wurden nur mit dem ersten Satz
    trainiert.
  * Listen-Labels mit mehreren Eintraegen (Multi-Label) wurden stumm auf den
    ersten Eintrag gekuerzt.
  * ClassLabel-Namen ('neg'/'pos') gingen verloren, das Modell hiess '0'/'1'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

LABEL_COLUMN_NAMES = ["label", "labels", "category", "class", "target", "sentiment"]
TEXT_COLUMN_NAMES = [
    "text", "sentence", "content", "review_body", "input", "document", "title", "body",
    "description", "abstract", "question", "passage", "premise", "hypothesis",
]
# Bekannte Satzpaare (erste Spalte, zweite Spalte) — Reihenfolge = Prioritaet.
TEXT_PAIRS = [
    ("sentence1", "sentence2"), ("premise", "hypothesis"), ("question1", "question2"),
    ("text_a", "text_b"), ("sentence_a", "sentence_b"), ("question", "sentence"),
    ("question", "passage"), ("query", "passage"), ("query", "document"),
]
ID_COLUMN_NAMES = {
    "id", "idx", "index", "row_id", "sample_id", "uid", "uuid", "key",
    "review_id", "product_id", "user_id", "item_id", "doc_id", "article_id",
    "tweet_id", "post_id", "comment_id", "message_id", "conversation_id",
    "passage_id", "question_id", "answer_id", "sentence_id", "token_id",
    "source_id", "target_id", "pair_id", "example_id", "data_id",
}


class MultiLabelError(ValueError):
    pass


def detect_columns(columns: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(Textspalte, zweite Textspalte bei Satzpaaren oder None, Labelspalte)."""
    cols = [c for c in columns if not str(c).startswith("__")]
    label_col = next((n for n in LABEL_COLUMN_NAMES if n in cols), None)

    for first, second in TEXT_PAIRS:
        if first in cols and second in cols:
            return first, second, label_col or _fallback_label(cols, {first, second})

    text_col = next((n for n in TEXT_COLUMN_NAMES if n in cols), None)
    non_id = [c for c in cols if c.lower() not in ID_COLUMN_NAMES]
    if text_col is None and non_id:
        text_col = non_id[0]
    if label_col is None:
        label_col = _fallback_label(cols, {text_col} if text_col else set())
    return text_col, None, label_col


def _fallback_label(cols: List[str], used: set) -> Optional[str]:
    non_id = [c for c in cols if c.lower() not in ID_COLUMN_NAMES and c not in used]
    if non_id:
        return non_id[-1]
    rest = [c for c in cols if c not in used]
    return rest[-1] if rest else None


def is_unlabeled(value: Any) -> bool:
    """None, leere Werte und negative Zahlen (HF: -1 = kein Label) zaehlen nicht."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value != value or value < 0  # NaN oder negativ
    if isinstance(value, str):
        return value.strip() in ("", "-1", "nan", "None")
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def label_value(value: Any, column: str) -> Any:
    """Einzelwert eines Labels. Listen mit genau einem Eintrag werden entpackt."""
    if isinstance(value, (list, tuple)):
        if len(value) > 1:
            raise MultiLabelError(
                f"Die Label-Spalte '{column}' enthaelt mehrere Werte pro Zeile (z.B. {list(value)[:5]}). "
                "Das ist eine Multi-Label-Aufgabe; die Sequenzklassifikation ordnet jedem Text "
                "genau eine Klasse zu.\n"
                "Loesungen: eine Spalte mit genau einem Label pro Zeile verwenden — oder falls "
                "die Spalte gar kein Label ist (z.B. Keyphrases), passt der Aufgabentyp nicht."
            )
        return value[0]
    return value


def class_label_names(features: Any, column: str) -> List[str]:
    """Namen eines HF-ClassLabel ('neg', 'pos'), sonst leer."""
    try:
        feat = features[column]
    except (KeyError, TypeError):
        return []
    names = getattr(feat, "names", None)
    return list(names) if names else []


def display_label(value: Any, names: Sequence[str]) -> str:
    """Anzeigename eines Labels: ClassLabel-Name fuer Index-Werte, sonst der Wert."""
    if names and isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
        return str(names[value])
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def safe_text(value: Any) -> str:
    """Leere Zellen (None/NaN) als leerer Text statt Tokenizer-Absturz."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def expected_label(raw: Any, value_to_label: Dict[str, str]) -> Optional[str]:
    """Erwartetes Label im Test, in den Namen, die das Modell kennt."""
    try:
        raw = label_value(raw, "label")
    except MultiLabelError:
        return None
    if is_unlabeled(raw):
        return None
    key = display_label(raw, [])
    return value_to_label.get(key, key)
