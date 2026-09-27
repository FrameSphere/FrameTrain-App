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


# ── Multi-Label und Regression ───────────────────────────────────────────────

SINGLE_LABEL = "single_label_classification"
MULTI_LABEL = "multi_label_classification"
REGRESSION = "regression"
MULTI_LABEL_SEPARATORS = (";", "|")


def _setting(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value if value is not None else "auto").strip().lower() or "auto"


def split_multi_labels(value: Any, names: Sequence[str] = ()) -> List[str]:
    """Labels einer Zeile: Liste oder 'sport;politik' bzw. 'sport|politik'."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value)
        parts = [text]
        for sep in MULTI_LABEL_SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                break
    out: List[str] = []
    for p in parts:
        if isinstance(p, str):
            p = p.strip()
            if not p:
                continue
        label = display_label(p, names)
        if label not in out:
            out.append(label)
    return out


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def _looks_multi_label(values: Sequence[Any]) -> bool:
    for v in values:
        if isinstance(v, (list, tuple)) and len(v) > 1:
            return True
        if isinstance(v, str):
            for sep in MULTI_LABEL_SEPARATORS:
                if sep in v and sum(1 for p in v.split(sep) if p.strip()) > 1:
                    return True
    return False


def _looks_regression(values: Sequence[Any]) -> bool:
    """Nur echte Kommazahlen mit vielen verschiedenen Werten.

    Sterne 1-5 oder Klassen 0/1 bleiben Klassifikation wie bisher — eine
    Regression muss man dort ausdruecklich waehlen (problem_type).
    """
    nums = [as_float(v) for v in values]
    if not nums or any(n is None for n in nums):
        return False
    if all(float(n).is_integer() for n in nums):
        return False
    return len(set(nums)) >= 10


def resolve_problem_type(values: Sequence[Any], multi_label: Any = "auto", problem_type: Any = "auto") -> str:
    """single_label_classification | multi_label_classification | regression.

    plugin_config problem_type hat Vorrang, dann multi_label (auto/true/false),
    dann die Erkennung an den Werten. Ohne Treffer bleibt alles wie bisher.
    """
    pt = _setting(problem_type)
    if pt in ("regression", "regress"):
        return REGRESSION
    if pt in ("multi_label_classification", "multi_label", "multilabel", "multi"):
        return MULTI_LABEL
    if pt in ("single_label_classification", "single_label", "single"):
        return SINGLE_LABEL
    ml = _setting(multi_label)
    if ml in ("true", "1", "yes", "ja"):
        return MULTI_LABEL
    if ml not in ("false", "0", "no", "nein") and _looks_multi_label(values):
        return MULTI_LABEL
    if _looks_regression(values):
        return REGRESSION
    return SINGLE_LABEL


def multi_label_scores(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    """Micro/Macro-F1 und Subset-Accuracy (alle Labels einer Zeile richtig).

    accuracy/f1/precision/recall werden zusaetzlich befuellt (Subset-Accuracy,
    Micro-Werte), damit die bestehende Karte der Analyse-Seite etwas zeigt.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    y = np.asarray(labels) >= 0.5
    pred = np.asarray(probs) >= float(threshold)
    p, r, micro, _ = precision_recall_fscore_support(y, pred, average="micro", zero_division=0)
    subset = float(accuracy_score(y, pred))
    return {
        "accuracy": subset, "f1": float(micro), "precision": float(p), "recall": float(r),
        "subset_accuracy": subset, "micro_f1": float(micro),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def regression_scores(labels, preds) -> Dict[str, float]:
    """MSE, RMSE, MAE, R², Pearson und Spearman."""
    import numpy as np

    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(preds, dtype=float).reshape(-1)
    if y.size == 0:
        return {}
    mse = float(np.mean((p - y) ** 2))
    out = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(p - y)))}
    var = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = float(1 - np.sum((p - y) ** 2) / var) if var > 0 else 0.0
    # Bei konstanten Vorhersagen ist die Korrelation undefiniert — 0 statt NaN,
    # NaN wuerde das JSON fuer die Analyse-Seite ungueltig machen.
    if y.size > 1 and np.std(p) > 0 and np.std(y) > 0:
        try:
            from scipy.stats import pearsonr, spearmanr
            out["pearson"] = float(pearsonr(y, p)[0])
            out["spearman"] = float(spearmanr(y, p)[0])
        except ImportError:
            out["pearson"] = float(np.corrcoef(y, p)[0, 1])
    else:
        out["pearson"] = 0.0
        out["spearman"] = 0.0
    return out
