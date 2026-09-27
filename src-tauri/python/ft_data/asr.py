"""Spracherkennung (ASR): Audio + Transkript finden, teilen und bewerten.

Gemeinsam fuer Training (train_engine/plugins/speech_recognition) und Test
(test_engine/plugins/speech_recognition), damit beide dieselben Paare lesen.

Unterstuetzte Layouts (jeweils auch in train/ val/ test/):
    <root>/aufnahme_1.wav + aufnahme_1.txt        (audio_transcript — so exportiert
                                                   die Datensatz-Werkstatt der App)
    <root>/metadata.csv|jsonl + Audiodateien      (HF-audiofolder: file_name + transcription/text/sentence)
    <root>/clips/*.mp3 + train.tsv/dev.tsv/test.tsv|validated.tsv   (Common Voice: path + sentence)
    <root>/*.parquet | <root>/<split>/*.parquet   (HF-Download: Audio-Bytes + Textspalte)

Frueher gemachte Fehler, die hier ausgeschlossen werden (aus media.py uebernommen):
  * Leere Split-Ordner zaehlen nicht — der App-Split legt test/ auch bei 0 % an.
  * Validierung wird von train abgetrennt, NIE von test/: die Testdaten bleiben
    fuer den Test unberuehrt.
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .media import (
    AUDIO_EXTS, MEDIA_CACHE_DIR, SPLIT_ALIASES, _ALL_SPLIT_NAMES, _guess_ext,
    _hf_features, _parquet_files, _signature, _visible_dirs,
)

# (Audiodatei, Transkript)
AsrItem = Tuple[str, str]

# Reihenfolge = Prioritaet. "transcription" (HF-Konvention) vor "text", weil
# manche Datasets beides haben und "text" dort die normalisierte Fassung ist.
TEXT_COLUMNS = (
    "transcription", "transcript", "sentence", "text", "normalized_text",
    "raw_transcription", "raw_text", "caption", "label_text",
)
FILE_COLUMNS = ("file_name", "path", "file", "filename", "audio_path", "audio_filepath", "audio")
METADATA_NAMES = ("metadata.csv", "metadata.jsonl", "metadata.tsv", "metadata.json")
# Common Voice: Dateiname -> kanonischer Split. validated.tsv/metadata.tsv sind
# der Gesamtbestand und nur dann Trainingsdaten, wenn train.tsv fehlt.
CV_SPLIT_FILES = {"train.tsv": "train", "dev.tsv": "val", "test.tsv": "test"}
CV_POOL_FILES = ("validated.tsv", "metadata.tsv")

ASR_CACHE = "asr"


@dataclass
class AsrLayout:
    train: List[AsrItem]
    val: List[AsrItem]
    test: List[AsrItem]
    source: str                      # audio_transcript | audiofolder | common_voice | parquet
    val_from_train: bool = False
    notes: List[str] = field(default_factory=list)


# ── Tabellen lesen ───────────────────────────────────────────────────────────

def _read_table(path: Path) -> List[dict]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return next((v for v in data.values() if isinstance(v, list)), [])
        return []
    # Common-Voice-TSVs setzen Anfuehrungszeichen woertlich in den Satz — mit
    # dem Standard-Quoting wuerden Saetze mit " verschluckt oder zusammengezogen.
    delimiter = "\t" if ext == ".tsv" else ","
    quoting = csv.QUOTE_NONE if ext == ".tsv" else csv.QUOTE_MINIMAL
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter, quoting=quoting))


def _pick(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in columns}
    return next((lower[c] for c in candidates if c in lower), None)


def text_column(columns: Iterable[str], override: Optional[str] = None) -> Optional[str]:
    cols = list(columns)
    if override and override in cols:
        return override
    return _pick(cols, TEXT_COLUMNS)


def file_column(columns: Iterable[str], override: Optional[str] = None) -> Optional[str]:
    cols = list(columns)
    if override and override in cols:
        return override
    return _pick(cols, FILE_COLUMNS)


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    return " ".join(str(text).split())


# ── Einzelne Layouts ─────────────────────────────────────────────────────────

def transcript_for(audio: Path) -> Optional[str]:
    """Wie studio_manager.rs::transcript_for: gleichnamige .txt, getrimmt, nicht leer."""
    for ext in (".txt", ".TXT"):
        p = audio.with_suffix(ext)
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError:
                text = p.read_text(encoding="latin-1").strip()
            if text:
                return _clean(text)
    return None


def _audio_files(d: Path, skip_split_dirs: bool) -> List[Path]:
    out: List[Path] = []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS or f.name.startswith("."):
            continue
        rel = f.relative_to(d).parts[:-1]
        if any(p.startswith(".") or p.startswith("__") for p in rel):
            continue  # u.a. .frametrain_media
        if skip_split_dirs and rel and rel[0].lower() in _ALL_SPLIT_NAMES:
            continue
        out.append(f)
    return out


def pair_items(d: Path, skip_split_dirs: bool = False) -> Tuple[List[AsrItem], int]:
    """Audio + gleichnamige .txt. Zweiter Wert: Audiodateien ohne Transkript."""
    items: List[AsrItem] = []
    missing = 0
    for f in _audio_files(d, skip_split_dirs):
        text = transcript_for(f)
        if text is None:
            missing += 1
            continue
        items.append((str(f), text))
    return items, missing


def metadata_items(d: Path, overrides: Optional[dict] = None) -> Optional[Tuple[List[AsrItem], int]]:
    """HF-audiofolder: metadata.csv/jsonl mit Dateiname und Transkript.

    None, wenn es keine passende Metadatei gibt. Zweiter Wert: Zeilen, deren
    Audiodatei fehlt oder deren Text leer ist.
    """
    overrides = overrides or {}
    for name in METADATA_NAMES:
        meta = d / name
        if not meta.is_file():
            continue
        rows = _read_table(meta)
        if not rows:
            continue
        cols = list(rows[0].keys())
        fcol = file_column(cols, overrides.get("audio_column"))
        tcol = text_column(cols, overrides.get("text_column"))
        if fcol is None or tcol is None:
            raise ValueError(
                f"{meta.name} in '{d}' braucht eine Spalte mit dem Dateinamen "
                f"({', '.join(FILE_COLUMNS[:3])}) und eine mit dem Transkript "
                f"({', '.join(TEXT_COLUMNS[:4])}). Vorhanden: {cols}. "
                "Eigene Namen: plugin_config audio_column / text_column."
            )
        items: List[AsrItem] = []
        skipped = 0
        for row in rows:
            rel, text = row.get(fcol), _clean(row.get(tcol))
            if isinstance(rel, dict):  # HF-Export: {"path": ...}
                rel = rel.get("path")
            if not rel or not text:
                skipped += 1
                continue
            p = Path(str(rel))
            p = p if p.is_absolute() else d / p
            if not p.is_file():
                skipped += 1
                continue
            items.append((str(p), text))
        return items, skipped
    return None


def _is_common_voice(root: Path) -> bool:
    if not (root / "clips").is_dir():
        return False
    names = {f.name.lower() for f in root.iterdir() if f.is_file()}
    return any(n in names for n in (*CV_SPLIT_FILES, *CV_POOL_FILES))


def common_voice_splits(root: Path, notes: List[str]) -> Dict[str, List[AsrItem]]:
    """Common Voice: clips/ + TSVs mit path und sentence."""
    clips = root / "clips"

    def read(tsv: Path) -> List[AsrItem]:
        rows = _read_table(tsv)
        if not rows:
            return []
        cols = list(rows[0].keys())
        pcol = _pick(cols, ("path", "file_name", "filename", "file"))
        tcol = text_column(cols)
        if pcol is None or tcol is None:
            raise ValueError(f"{tsv.name}: Spalten 'path' und 'sentence' erwartet, vorhanden: {cols}")
        items, missing = [], 0
        for row in rows:
            rel, text = row.get(pcol), _clean(row.get(tcol))
            if not rel or not text:
                continue
            p = clips / str(rel)
            if not p.is_file() and not p.suffix:
                p = p.with_suffix(".mp3")  # aeltere CV-Versionen ohne Endung
            if not p.is_file():
                missing += 1
                continue
            items.append((str(p), text))
        if missing:
            notes.append(f"{tsv.name}: {missing} Eintraege ohne Datei in clips/ ignoriert.")
        return items

    files = {f.name.lower(): f for f in root.iterdir() if f.is_file()}
    splits: Dict[str, List[AsrItem]] = {}
    for fname, canon in CV_SPLIT_FILES.items():
        if fname in files:
            items = read(files[fname])
            if items:
                splits[canon] = items
    if "train" not in splits:
        pool_name = next((n for n in CV_POOL_FILES if n in files), None)
        if pool_name:
            used = {p for items in splits.values() for p, _ in items}
            pool = [it for it in read(files[pool_name]) if it[0] not in used]
            if pool:
                splits["train"] = pool
                notes.append(f"Common Voice ohne train.tsv — {pool_name} ist der Trainingsbestand "
                             "(ohne die Clips aus dev/test).")
    return splits


def _collect_dir(d: Path, overrides: dict, notes: List[str], skip_split_dirs: bool) -> Tuple[List[AsrItem], str]:
    meta = metadata_items(d, overrides)
    if meta is not None:
        items, skipped = meta
        if skipped:
            notes.append(f"{d.name}/: {skipped} Zeilen der Metadatei ohne Audiodatei oder Text ignoriert.")
        return items, "audiofolder"
    items, missing = pair_items(d, skip_split_dirs=skip_split_dirs)
    if missing:
        notes.append(f"{d.name}/: {missing} Audiodatei(en) ohne gleichnamige .txt ignoriert.")
    return items, "audio_transcript"


def _split_dirs(root: Path) -> Dict[str, Path]:
    by_name = {d.name.lower(): d for d in _visible_dirs(root)}
    found: Dict[str, Path] = {}
    for canon, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in by_name:
                found[canon] = by_name[alias]
                break
    return found


# ── Parquet ──────────────────────────────────────────────────────────────────

def _parquet_columns(schema, overrides: dict) -> Tuple[Optional[str], Optional[str]]:
    import pyarrow as pa

    feats = _hf_features(schema)
    names = [f.name for f in schema]
    audio = overrides.get("audio_column") if overrides.get("audio_column") in names else None
    if audio is None:
        audio = next((n for n, f in feats.items() if isinstance(f, dict) and f.get("_type") == "Audio"), None)
    if audio is None:
        for fld in schema:
            if fld.name.lower() in ("audio", "speech", "sound", "file") and (
                    pa.types.is_struct(fld.type) or pa.types.is_binary(fld.type)):
                audio = fld.name
                break
    return audio, text_column(names, overrides.get("text_column"))


def ensure_parquet_pairs(root: Path, overrides: Optional[dict] = None, status=None) -> Optional[Path]:
    """Entpackt ein HF-Parquet mit Audio + Text einmalig zu audio_transcript-Paaren.

    Ziel: <root>/.frametrain_media/asr/<split>/<n>.<ext> + <n>.txt. Aendern sich
    die Parquet-Dateien, wird neu entpackt. None, wenn kein Audio-Parquet da ist.
    """
    overrides = overrides or {}
    parquet = _parquet_files(root)
    if not parquet:
        return None
    import pyarrow.parquet as pq

    schema = pq.read_schema(next(iter(parquet.values()))[0])
    audio_col, text_col = _parquet_columns(schema, overrides)
    if audio_col is None:
        return None
    if text_col is None:
        raise ValueError(
            f"Parquet mit Audio in Spalte '{audio_col}', aber ohne Transkript-Spalte "
            f"({', '.join(TEXT_COLUMNS[:4])}). Vorhanden: {[f.name for f in schema]}. "
            "Eigene Spalte: plugin_config text_column."
        )

    cache = root / MEDIA_CACHE_DIR / ASR_CACHE
    marker = cache / ".complete.json"
    sig = _signature(parquet) + f"|{audio_col}|{text_col}"
    try:
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("signature") == sig:
            return cache
    except ValueError:
        pass

    import shutil
    if cache.exists():
        shutil.rmtree(cache)
    if status:
        status("Entpacke Audio und Transkripte aus Parquet (einmalig)...")
    counts: Dict[str, int] = {}
    for split, files in parquet.items():
        target = cache / split
        target.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(columns=[audio_col, text_col], batch_size=128):
                for value, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    text = _clean(text)
                    if isinstance(value, dict):
                        data, hint = value.get("bytes"), value.get("path")
                    else:
                        data, hint = value, None
                    if not data or not text:
                        continue
                    stem = target / f"{n:06d}"
                    stem.with_suffix(_guess_ext(data, hint, "audio")).write_bytes(data)
                    stem.with_suffix(".txt").write_text(text, encoding="utf-8")
                    n += 1
        counts[split] = n
    marker.write_text(json.dumps({"signature": sig, "counts": counts}), encoding="utf-8")
    return cache


# ── Gesamtlayout ─────────────────────────────────────────────────────────────

def _collect(root: Path, overrides: dict, notes: List[str], status=None) -> Tuple[Dict[str, List[AsrItem]], str, bool]:
    """(Splits, Quelle, ob der Datensatz schon geteilt war)."""
    if _is_common_voice(root):
        return common_voice_splits(root, notes), "common_voice", True

    split_dirs = _split_dirs(root)
    if split_dirs:
        splits: Dict[str, List[AsrItem]] = {}
        source = "audio_transcript"
        for canon, d in split_dirs.items():
            if _is_common_voice(d):
                cv = common_voice_splits(d, notes)
                items = [it for v in cv.values() for it in v]
                source = "common_voice"
            else:
                items, source = _collect_dir(d, overrides, notes, skip_split_dirs=False)
            if items:
                splits[canon] = items
        if splits:
            return splits, source, True

    items, source = _collect_dir(root, overrides, notes, skip_split_dirs=True)
    if items:
        return {"train": items}, source, False

    cache = ensure_parquet_pairs(root, overrides, status)
    if cache is not None:
        splits = {}
        for d in _visible_dirs(cache):
            got, _ = pair_items(d)
            if got:
                canon = next((c for c, al in SPLIT_ALIASES.items() if d.name.lower() in al), "train")
                splits[canon] = got
        if splits:
            return splits, "parquet", len(splits) > 1 or "train" not in splits
    return {}, source, False


def resolve_asr_layout(root: Path, seed: int = 42, val_fraction: float = 0.1,
                       overrides: Optional[dict] = None, status=None) -> AsrLayout:
    """Liest ein ASR-Dataset. Validierung ist val/; fehlt sie, gehen 10 % von train ab."""
    root = Path(root)
    notes: List[str] = []
    splits, source, _ = _collect(root, overrides or {}, notes, status)
    if not splits:
        raise ValueError(
            f"In '{root}' wurden keine Paare aus Audio und Transkript gefunden.\n"
            "Erwartet wird eines dieser Formate (auch in train/ val/ test/):\n"
            "  - aufnahme.wav + aufnahme.txt (gleicher Name, so exportiert die Datensatz-Werkstatt)\n"
            "  - metadata.csv / metadata.jsonl mit file_name und transcription (HF-audiofolder)\n"
            "  - Common Voice: clips/ + train.tsv/dev.tsv/test.tsv oder validated.tsv (path, sentence)\n"
            "  - Parquet mit Audio-Spalte und Textspalte (transcription/text/sentence)"
        )
    if "train" not in splits:
        raise ValueError(
            f"In '{root}' gibt es Splits ({', '.join(sorted(splits))}), aber keinen train-Split "
            "mit Audio und Transkript."
        )
    train, val, test = splits["train"], splits.get("val", []), splits.get("test", [])
    val_from_train = False
    if not val:
        rng = random.Random(seed)
        shuffled = train[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        val, train = shuffled[:n_val], shuffled[n_val:]
        val_from_train = True
        notes.append(f"Kein Validierungs-Split — {len(val)} Aufnahmen ({val_fraction:.0%}) aus train abgetrennt.")
    return AsrLayout(train, val, test, source, val_from_train, notes)


def evaluation_items(root: Path, overrides: Optional[dict] = None, status=None) -> Tuple[List[AsrItem], Optional[str], List[str]]:
    """Paare fuer einen Test: test vor val vor train. (Paare, Split-Name, Hinweise)."""
    notes: List[str] = []
    splits, _, was_split = _collect(Path(root), overrides or {}, notes, status)
    for name in ("test", "val", "train"):
        if splits.get(name):
            return splits[name], (name if was_split else None), notes
    return [], None, notes


# ── Text und Metriken ────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_transcript(text: str) -> str:
    """Fuer WER/CER: klein, ohne Satzzeichen, einfache Leerzeichen.

    Whisper schreibt "Licht einschalten." mit Punkt und Grossbuchstaben, das
    Transkript oft ohne — ohne Normalisierung zaehlte jedes Satzzeichen als Fehler.
    """
    text = _PUNCT.sub(" ", str(text or "").lower()).replace("_", " ")
    return " ".join(text.split())


def _levenshtein(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(predictions: Sequence[str], references: Sequence[str],
                normalize: bool = True) -> Dict[str, float]:
    """WER und CER ueber den ganzen Split (Korpus, nicht Mittel der Einzelwerte).

    Leere Referenzen werden uebersprungen — jiwer lehnt sie ab und eine WER
    gegen "nichts" ist nicht definiert. jiwer rechnet; fehlt es, dieselbe
    Formel per Levenshtein (Ergebnis identisch, nur langsamer).
    """
    pairs = []
    for p, r in zip(predictions, references):
        p, r = (normalize_transcript(p), normalize_transcript(r)) if normalize else (str(p or ""), str(r or ""))
        if r.strip():
            pairs.append((p, r))
    if not pairs:
        return {}
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    try:
        import jiwer
        return {"wer": float(jiwer.wer(refs, preds)), "cer": float(jiwer.cer(refs, preds))}
    except ImportError:
        pass
    w_err = sum(_levenshtein(r.split(), p.split()) for p, r in pairs)
    w_tot = sum(len(r.split()) for r in refs)
    c_err = sum(_levenshtein(r, p) for p, r in pairs)
    c_tot = sum(len(r) for r in refs)
    return {"wer": w_err / max(w_tot, 1), "cer": c_err / max(c_tot, 1)}


def sample_wer(prediction: str, reference: str) -> Optional[float]:
    rates = error_rates([prediction], [reference])
    return rates.get("wer")


# ── Modellart ────────────────────────────────────────────────────────────────

# Encoder-Decoder: Audio rein, Text wird Token fuer Token erzeugt.
SEQ2SEQ_MODEL_TYPES = {
    "whisper", "moonshine", "speech_to_text", "speech-to-text",
    "speech-encoder-decoder", "speech_encoder_decoder",
}
# Reine Encoder mit CTC-Kopf: ein Zeichen pro Audio-Frame.
CTC_MODEL_TYPES = {
    "wav2vec2", "wav2vec2-bert", "wav2vec2_bert", "wav2vec2-conformer", "wav2vec2_conformer",
    "hubert", "wavlm", "data2vec-audio", "data2vec_audio", "unispeech", "unispeech-sat",
    "unispeech_sat", "sew", "sew-d", "sew_d", "mctct",
}

# Deutsche Sprachnamen, die Whisper nicht kennt — im Formular tippt man "deutsch".
_LANGUAGE_ALIASES = {
    "deutsch": "german", "englisch": "english", "franzoesisch": "french", "französisch": "french",
    "spanisch": "spanish", "italienisch": "italian", "niederlaendisch": "dutch",
    "niederländisch": "dutch", "polnisch": "polish", "tuerkisch": "turkish", "türkisch": "turkish",
    "russisch": "russian", "portugiesisch": "portuguese", "auto": "",
}


def normalize_language(value) -> str:
    lang = str(value or "").strip().lower()
    return _LANGUAGE_ALIASES.get(lang, lang)


def asr_mode(model_cfg: dict) -> Optional[str]:
    """'seq2seq' | 'ctc' | None aus config.json (Architektur vor model_type)."""
    archs = [a for a in (model_cfg.get("architectures") or []) if isinstance(a, str)]
    if any(a.endswith("ForCTC") for a in archs):
        return "ctc"
    if any(a.endswith("ForSpeechSeq2Seq") for a in archs):
        return "seq2seq"
    model_type = str(model_cfg.get("model_type", "")).lower()
    if model_type in SEQ2SEQ_MODEL_TYPES:
        return "seq2seq"
    if model_type in CTC_MODEL_TYPES:
        return "ctc"
    return None


def has_ctc_head(model_cfg: dict) -> bool:
    return any(isinstance(a, str) and a.endswith("ForCTC") for a in (model_cfg.get("architectures") or []))


# ── CTC-Vokabular ────────────────────────────────────────────────────────────

CTC_PAD, CTC_UNK, CTC_WORD = "[PAD]", "[UNK]", "|"


def ctc_text(text: str, lowercase: bool = True) -> str:
    """Transkript fuer ein selbst gebautes CTC-Vokabular: ohne Satzzeichen."""
    text = _PUNCT.sub(" ", str(text or "")).replace("_", " ")
    text = " ".join(text.split())
    return text.lower() if lowercase else text


def build_ctc_vocab(texts: Iterable[str]) -> Dict[str, int]:
    """Zeichen-Vokabular: [PAD] (zugleich CTC-Blank) = 0, [UNK] = 1, | = Wortgrenze."""
    chars = sorted({c for t in texts for c in t if c != " "})
    vocab = {CTC_PAD: 0, CTC_UNK: 1, CTC_WORD: 2}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def audio_seconds(path: str) -> Optional[float]:
    """Dauer aus dem Datei-Kopf, ohne die Aufnahme zu dekodieren."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        pass
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except Exception:
        return None
