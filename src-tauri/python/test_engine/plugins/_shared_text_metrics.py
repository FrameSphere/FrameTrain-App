"""
Kennzahlen fuer freie Text-Antworten (Vision-Language, Texterkennung).

Exact Match ist bei freien Antworten fast immer 0: "Eine rote Katze." und
"rote katze" sind dieselbe Antwort. Deshalb zusaetzlich:
  * tolerant_match  normalisiert gleich ODER die Antwort enthaelt die
                    Referenz als ganze Wortfolge ("it is a red cat" ~ "red cat")
  * rougeL          laengste gemeinsame Wortfolge (F1), aus ft_data.image_text
  * cer             Zeichenfehlerrate (Levenshtein / Referenzlaenge, ueber das
                    ganze Dataset summiert wie jiwer) — nur bei Texterkennung

Rein in Python, ohne Zusatzpakete.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Fuehrende Artikel spielen fuer die Uebereinstimmung keine Rolle.
ARTICLES = {"a", "an", "the", "der", "die", "das", "ein", "eine", "einen", "einem", "einer"}

# Fragen/Prompts, die nach dem Ablesen von Text fragen — dann ist CER sinnvoll.
OCR_PROMPT = re.compile(
    r"\b(ocr|transcribe|transcription|read (the )?text|what (does|is) (the )?text|"
    r"text (in|on|im|auf)|lies|ablesen|abschreiben|texterkennung|schrift)\b",
    re.IGNORECASE,
)


def normalise(text: Any) -> str:
    """Klein, ohne Satzzeichen, ohne fuehrenden Artikel, Leerraum zusammengefasst."""
    t = re.sub(r"[^\w\s]", " ", str(text or "").lower())
    words = t.split()
    while len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    return " ".join(words)


def tolerant_match(prediction: Any, reference: Any) -> bool:
    p, r = normalise(prediction), normalise(reference)
    if not r:
        return not p
    return p == r or f" {r} " in f" {p} "


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _ocr_form(text: Any) -> str:
    # Gross/Klein und Satzzeichen zaehlen bei Texterkennung; nur Leerraum vereinheitlichen.
    return " ".join(str(text or "").split())


def character_error_rate(predictions: Iterable[Any], references: Iterable[Any]) -> Optional[float]:
    """Summe der Zeichen-Editierschritte / Summe der Referenzzeichen."""
    edits = chars = 0
    for p, r in zip(predictions, references):
        p, r = _ocr_form(p), _ocr_form(r)
        edits += levenshtein(p, r)
        chars += len(r)
    return edits / chars if chars else None


def looks_like_ocr(prompts: Iterable[Any]) -> bool:
    """Mehrheit der Prompts fragt nach Text im Bild."""
    ps = [str(p or "") for p in prompts]
    return bool(ps) and sum(1 for p in ps if OCR_PROMPT.search(p)) * 2 > len(ps)


def free_text_scores(predictions: Sequence[Any], references: Sequence[Any],
                     prompts: Optional[Sequence[Any]] = None,
                     ocr: Optional[bool] = None) -> Dict[str, Any]:
    """exact_match, tolerant_match, rougeL und (bei Texterkennung) cer.

    ocr=None: aus den Prompts ableiten; True/False erzwingt es (plugin_config).
    """
    # exact_match und rougeL wie bisher (ft_data.text_scores) — bestehende
    # Werte in der Test-Historie bleiben vergleichbar.
    from ft_data.image_text import text_scores

    preds, refs = list(predictions), list(references)
    if not preds or len(preds) != len(refs):
        return {}
    n = len(preds)
    out: Dict[str, Any] = dict(text_scores([str(p or "") for p in preds], [str(r or "") for r in refs]))
    out["tolerant_match"] = sum(1 for p, r in zip(preds, refs) if tolerant_match(p, r)) / n
    if ocr is None:
        ocr = looks_like_ocr(prompts or [])
    if ocr:
        cer = character_error_rate(preds, refs)
        if cer is not None:
            out["cer"] = cer
    return out


def row_flags(prediction: Any, reference: Any) -> Dict[str, bool]:
    return {"tolerant_correct": tolerant_match(prediction, reference)}


__all__: List[str] = [
    "normalise", "tolerant_match", "levenshtein", "character_error_rate",
    "looks_like_ocr", "free_text_scores", "row_flags",
]
