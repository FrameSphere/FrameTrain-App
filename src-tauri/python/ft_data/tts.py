"""Sprachausgabe (Text-to-Speech): Text aufbereiten, Sprecher-Embedding, Audio erzeugen, bewerten.

Gemeinsam fuer Training (train_engine/plugins/text_to_speech), Test
(test_engine/plugins/text_to_speech) und den Modell-Server (Labor, Hosting),
damit alle drei aus demselben Text dasselbe Audio erzeugen.

Zwei Modellfamilien:
  speecht5  microsoft/speecht5_tts — trainierbar. Erzeugt ein Mel-Spektrogramm,
            der Vocoder (microsoft/speecht5_hifigan) macht daraus die Wellenform.
            Die Stimme steuert ein Sprecher-Embedding mit 512 Werten (x-vector).
  vits      facebook/mms-tts-* und andere VITS-Modelle — NUR Inferenz.
            transformers bricht das Training ab ("Training of VITS is not
            supported yet"): es fehlen Diskriminator, Posterior-Encoder-Loss und
            die Ausrichtungssuche. Testen, Labor und Hosting gehen.

Optionale Abhaengigkeit: speechbrain (x-vector aus den Aufnahmen). Fehlt es,
gilt fuer alle Aufnahmen derselbe Standard-Sprechervektor (tts_default_speaker)
— das Modell lernt die Stimme dann allein ueber seine Gewichte, was nur bei
Daten mit EINER Stimme sinnvoll ist.

Kennzahl: Verstaendlichkeit per Ruecktranskription. Ein kleines, lokal
vorhandenes ASR-Modell hoert das erzeugte Audio ab; WER/CER gegen den
Eingabetext. Ohne lokales ASR-Modell gibt es keine Kennzahl (kein Download).
"""
from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TTS_TRAINABLE = ("speecht5",)
TTS_INFERENCE_ONLY = ("vits",)

SPEAKER_DIM = 512
XVECTOR_SOURCE = "speechbrain/spkrec-xvect-voxceleb"
DEFAULT_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_FILE = "speaker_embedding.npy"
TTS_META_FILE = "tts_config.json"
VOCODER_DIR = "vocoder"

# Kleine Modelle zuerst: die Kennzahl soll den Test nicht dominieren.
ASR_CANDIDATES = (
    "openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small",
    "openai/whisper-tiny.en", "openai/whisper-base.en", "distil-whisper/distil-small.en",
    "facebook/wav2vec2-base-960h",
)

VITS_NOT_TRAINABLE = (
    "VITS / MMS-TTS laesst sich in FrameTrain nicht trainieren: transformers enthaelt fuer diese "
    "Architektur nur die Inferenz (kein Diskriminator, kein Trainings-Loss). "
    "Testen, Labor und Hosting funktionieren mit dem Modell. "
    "Zum Trainieren einer eigenen Stimme: microsoft/speecht5_tts."
)


# ── Modellart ────────────────────────────────────────────────────────────────

def tts_kind(model_cfg: dict) -> Optional[str]:
    """'speecht5' | 'vits' | None aus config.json.

    SpeechT5 gibt es auch als Spracherkenner (SpeechT5ForSpeechToText) und als
    Stimmwandler (…ForSpeechToSpeech) — gleicher model_type, andere Aufgabe.
    """
    archs = [a for a in (model_cfg.get("architectures") or []) if isinstance(a, str)]
    model_type = str(model_cfg.get("model_type", "")).lower()
    if any(a == "VitsModel" for a in archs) or model_type == "vits":
        return "vits"
    if any(a.endswith("ForTextToSpeech") for a in archs):
        return "speecht5" if model_type in ("speecht5", "") else None
    if model_type == "speecht5":
        if any(a.endswith("ForSpeechToText") or a.endswith("ForSpeechToSpeech") for a in archs):
            return None
        return "speecht5"
    return None


def read_model_config(model_path) -> dict:
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Keine config.json im Modellordner: {model_path}")
    return json.loads(cfg.read_text(encoding="utf-8"))


def read_tts_meta(model_path) -> dict:
    f = Path(model_path) / TTS_META_FILE
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return {}


# ── Text ─────────────────────────────────────────────────────────────────────

# SpeechT5 kennt nur lateinische Buchstaben und wenige Satzzeichen. Ohne diese
# Ersetzungen wuerde aus "Grüße" ein "Gr<unk><unk>e" — das Modell lernte, an
# diesen Stellen irgendetwas zu sprechen.
_TRANSLIT = {
    "ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss", "ẞ": "SS",
    "ø": "oe", "Ø": "Oe", "å": "aa", "Å": "Aa", "ł": "l", "Ł": "L", "đ": "d", "þ": "th", "ð": "d",
    "–": "-", "‐": "-", "‑": "-", "−": "-", "…": "...", "„": '"', "“": '"', "”": '"', "«": '"', "»": '"',
    "‚": "'", "‘": "'", "’": "'", "´": "'", "`": "'", "&": " and ", " ": " ", "\t": " ", "\n": " ",
}


def vocab_chars(tokenizer) -> Set[str]:
    """Einzelzeichen, die der Tokenizer kennt (ohne Spezial-Token)."""
    return {t for t in tokenizer.get_vocab() if len(t) == 1}


def prepare_text(text: str, known: Optional[Set[str]] = None) -> Tuple[str, Set[str]]:
    """Text fuer einen Zeichen-Tokenizer. Rueckgabe: (Text, nicht darstellbare Zeichen).

    known=None laesst jeden Buchstaben durch (VITS bringt sein eigenes
    Vokabular mit) und raeumt nur Leerraum und typografische Zeichen auf.
    Reihenfolge je Zeichen: bekannt -> Ersetzungstabelle -> Akzent abtrennen
    (é -> e) -> weglassen und melden.
    """
    out: List[str] = []
    unknown: Set[str] = set()
    for ch in str(text or ""):
        if ch.isspace():
            out.append(" ")
        elif known is None:
            out.append(_TRANSLIT.get(ch, ch) if not ch.isalnum() else ch)
        elif ch in known:
            out.append(ch)
        elif ch in _TRANSLIT:
            rep = _TRANSLIT[ch]
            out.append("".join(c for c in rep if c in known or c == " "))
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            if base and all(c in known for c in base):
                out.append(base)
            else:
                unknown.add(ch)
                out.append(" ")
    return " ".join("".join(out).split()), unknown


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def split_text(text: str, max_chars: int = 220) -> List[str]:
    """Zerlegt langen Text an Satzgrenzen (notfalls an Kommas/Woertern).

    SpeechT5 wurde auf einzelne Saetze trainiert; bei langen Eingaben verliert
    die Aufmerksamkeit den Faden und das Audio bricht ab oder wiederholt sich.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return []
    chunks: List[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(", ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= max_chars // 2:
            chunks[-1] = f"{chunks[-1]} {sentence}"   # sehr kurze Saetze zusammenlegen
        else:
            chunks.append(sentence)
    return chunks


# ── Sprecher-Embedding ───────────────────────────────────────────────────────

def speechbrain_available() -> bool:
    return importlib.util.find_spec("speechbrain") is not None


SPEECHBRAIN_HINT = (
    "Fuer Sprecher-Embeddings aus den Aufnahmen (x-vector) wird das Paket 'speechbrain' gebraucht: "
    "pip install speechbrain. Beim ersten Lauf laedt es das Modell "
    f"{XVECTOR_SOURCE} (ca. 20 MB)."
)


def default_speaker(dim: int = SPEAKER_DIM):
    """Standard-Sprechervektor (echter x-vector, siehe tts_default_speaker).

    Kein Null-Vektor: damit bricht das Basismodell jeden Satz nach gut einer
    Sekunde ab.
    """
    import numpy as np
    from .tts_default_speaker import DEFAULT_SPEAKER
    vec = np.asarray(DEFAULT_SPEAKER, dtype=np.float32)
    if vec.shape[0] != dim:
        raise ValueError(f"Standard-Sprechervektor hat {vec.shape[0]} Werte, das Modell erwartet {dim}.")
    return vec


def _unit(vec):
    import numpy as np
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def mean_speaker(vectors: Sequence) -> Any:
    """Mittel ueber alle Aufnahmen, wieder auf Laenge 1 gebracht (wie die Einzelvektoren)."""
    import numpy as np
    if not len(vectors):
        return default_speaker()
    return _unit(np.mean(np.stack([np.asarray(v, dtype=np.float32) for v in vectors]), axis=0))


def load_speaker_file(path, dim: int = SPEAKER_DIM):
    import numpy as np
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Sprecher-Embedding nicht gefunden: {p}")
    if p.suffix.lower() == ".json":
        vec = np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=np.float32)
    else:
        vec = np.load(str(p)).astype(np.float32)
    vec = vec.reshape(-1)
    if vec.shape[0] != dim:
        raise ValueError(f"Sprecher-Embedding in {p.name} hat {vec.shape[0]} Werte, erwartet werden {dim}.")
    return vec


class XVectorEncoder:
    """x-vector (512 Werte) je Aufnahme ueber speechbrain. Wirft ImportError ohne das Paket."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if not speechbrain_available():
            raise ImportError(SPEECHBRAIN_HINT)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        # Ohne savedir legt speechbrain "pretrained_models/" im Arbeitsverzeichnis an.
        save = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "frametrain" / "spkrec-xvect-voxceleb"
        save.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = EncoderClassifier.from_hparams(
            source=XVECTOR_SOURCE, savedir=str(save), run_opts={"device": "cpu"})

    def embed(self, wave, sampling_rate: int = 16000):
        import numpy as np
        import torch
        wave = np.asarray(wave, dtype=np.float32)
        if sampling_rate != 16000:
            import librosa
            wave = librosa.resample(wave, orig_sr=sampling_rate, target_sr=16000)
        with torch.no_grad():
            emb = self.encoder.encode_batch(torch.from_numpy(wave).unsqueeze(0))
        return _unit(emb.squeeze().cpu().numpy())


def make_speaker_encoder(choice: str = "auto", warn: Optional[Callable[[str], None]] = None) -> Optional[XVectorEncoder]:
    """choice: auto | xvector | default. None = Standard-Sprechervektor.

    'auto' weicht mit einer Meldung aus, wenn speechbrain fehlt oder das
    x-vector-Modell nicht geladen werden kann (offline); 'xvector' bricht ab.
    """
    choice = str(choice or "auto").strip().lower()
    if choice in ("neutral", "none", "default", "off"):
        return None
    try:
        return XVectorEncoder()
    except ImportError as exc:
        if choice == "xvector":
            raise
        reason = str(exc)
    except Exception as exc:  # Download fehlgeschlagen, defekter Cache …
        if choice == "xvector":
            raise RuntimeError(f"x-vector-Modell {XVECTOR_SOURCE} nicht ladbar: {exc}") from exc
        reason = f"x-vector-Modell {XVECTOR_SOURCE} nicht ladbar ({type(exc).__name__}: {exc})."
    if warn:
        warn(
            "Sprecher-Embeddings werden NICHT aus den Aufnahmen berechnet — fuer alle Aufnahmen gilt "
            "derselbe Standard-Sprechervektor. Das Modell lernt die Stimme dann nur ueber seine Gewichte; sinnvoll ist "
            f"das bei Daten mit einer einzigen Stimme. Grund: {reason}")
    return None


# ── Mel-Spektrogramm ─────────────────────────────────────────────────────────

def mel_target(feature_extractor, wave, sampling_rate: int = 16000):
    """Log-Mel-Ziel (Frames x 80) so, wie SpeechT5 es als Label erwartet."""
    import numpy as np
    out = feature_extractor(audio_target=wave, sampling_rate=sampling_rate, return_attention_mask=False)
    values = out.get("input_values", out.get("labels"))
    return np.asarray(values[0], dtype=np.float32)


def collate_speecht5(features: List[dict], pad_token_id: int, reduction_factor: int = 2) -> Dict[str, Any]:
    """Batch aus {input_ids, labels (T x Mel), speaker}.

    Die Ziel-Laenge muss ein Vielfaches des reduction_factor sein: der Decoder
    sagt je Schritt so viele Frames voraus. Aufgefuellt wird mit -100 (zaehlt
    nicht in den Loss).
    """
    import numpy as np
    import torch

    r = max(int(reduction_factor), 1)
    ids = [list(f["input_ids"]) for f in features]
    width = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), width), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        attention[i, : len(x)] = 1

    mels = []
    for f in features:
        mel = np.asarray(f["labels"], dtype=np.float32)
        keep = (mel.shape[0] // r) * r
        mels.append(mel[: max(keep, r)] if mel.shape[0] >= r else np.pad(mel, ((0, r - mel.shape[0]), (0, 0))))
    frames = max(m.shape[0] for m in mels)
    labels = torch.full((len(mels), frames, mels[0].shape[1]), -100.0, dtype=torch.float32)
    for i, m in enumerate(mels):
        labels[i, : m.shape[0]] = torch.from_numpy(m)

    speakers = torch.tensor(np.stack([np.asarray(f["speaker"], dtype=np.float32) for f in features]))
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels,
            "speaker_embeddings": speakers}


def griffin_lim(log_mel, sampling_rate: int = 16000, n_iter: int = 48):
    """Notloesung ohne Vocoder: Wellenform aus dem Log-Mel-Spektrogramm schaetzen.

    Klingt blechern (die Phase wird geraten), ist aber verstaendlich und braucht
    kein zweites Modell. Parameter wie SpeechT5FeatureExtractor: 1024er FFT,
    Schritt 256, 80 Mel-Baender von 80 bis 7600 Hz, log10 der Amplitude.
    """
    import librosa
    import numpy as np
    mel = np.power(10.0, np.asarray(log_mel, dtype=np.float32)).T   # (Mel, Frames)
    wave = librosa.feature.inverse.mel_to_audio(
        mel, sr=sampling_rate, n_fft=1024, hop_length=256, win_length=1024,
        fmin=80, fmax=7600, power=1.0, n_iter=n_iter)
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    return (wave / peak * 0.9).astype(np.float32) if peak > 0 else wave.astype(np.float32)


# ── Vocoder ──────────────────────────────────────────────────────────────────

def _cached_snapshot(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None


def load_vocoder(model_path, choice: Any = None, allow_download: bool = True,
                 status: Optional[Callable[[str], None]] = None):
    """(Vocoder oder None, Quelle). None = Griffin-Lim.

    Reihenfolge: ausdruecklich genannt (Pfad/HF-ID) -> <modell>/vocoder (legt
    das Training an) -> HF-Cache -> Download. choice 'griffin_lim' erzwingt den
    Weg ohne Vocoder.
    """
    from transformers import SpeechT5HifiGan

    choice = str(choice or "").strip()
    if choice.lower() in ("griffin_lim", "griffin-lim", "none", "off"):
        return None, "griffin_lim"
    candidates: List[Tuple[str, str]] = []
    if choice:
        candidates.append((choice, choice))
    bundled = Path(model_path) / VOCODER_DIR
    if (bundled / "config.json").is_file():
        candidates.append((str(bundled), "modellordner"))
    cached = _cached_snapshot(DEFAULT_VOCODER)
    if cached:
        candidates.append((cached, DEFAULT_VOCODER))
    for src, label in candidates:
        try:
            local = Path(src).exists()
            return SpeechT5HifiGan.from_pretrained(src, local_files_only=local or not allow_download), label
        except Exception as exc:
            if status:
                status(f"Vocoder {label} nicht ladbar ({type(exc).__name__}).")
    if allow_download and not cached:
        try:
            if status:
                status(f"Lade Vocoder {DEFAULT_VOCODER} (ca. 50 MB, einmalig)...")
            return SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER), DEFAULT_VOCODER
        except Exception as exc:
            if status:
                status(f"Vocoder-Download fehlgeschlagen ({type(exc).__name__}).")
    return None, "griffin_lim"


# ── Erzeugen ─────────────────────────────────────────────────────────────────

class TtsEngine:
    """Geladenes TTS-Modell: synthesize(text) -> (Wellenform float32, Abtastrate)."""

    def __init__(self):
        self.kind = ""
        self.model = None
        self.tokenizer = None
        self.vocoder = None
        self.vocoder_source = ""
        self.speaker = None            # torch-Tensor (1, 512) — nur SpeechT5
        self.speaker_source = ""
        self.sampling_rate = 16000
        self.device = None
        self.known_chars: Optional[Set[str]] = None
        self.meta: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.seed = 0
        self.last_dropped: Set[str] = set()

    def synthesize(self, text: str):
        import numpy as np
        import torch

        chunks = split_text(text)
        if not chunks:
            raise ValueError("Kein Text zum Sprechen.")
        pause = np.zeros(int(0.18 * self.sampling_rate), dtype=np.float32)
        parts: List[Any] = []
        dropped: Set[str] = set()
        for chunk in chunks:
            prepared, unknown = prepare_text(chunk, self.known_chars)
            dropped |= unknown
            if not prepared.strip():
                continue
            torch.manual_seed(self.seed)   # VITS wuerfelt Dauer und Klang — gleiche Eingabe, gleiches Audio
            wave = self._speak(prepared)
            if parts:
                parts.append(pause)
            parts.append(np.asarray(wave, dtype=np.float32).reshape(-1))
        if not parts:
            raise ValueError(
                "Der Text enthaelt keine Zeichen, die das Modell sprechen kann"
                + (f" (nicht darstellbar: {' '.join(sorted(dropped))})." if dropped else "."))
        self.last_dropped = dropped
        return np.concatenate(parts), self.sampling_rate

    def _speak(self, text: str):
        import torch

        if self.kind == "vits":
            enc = self.tokenizer(text, return_tensors="pt")
            if enc["input_ids"].shape[-1] == 0:
                raise ValueError("Der Tokenizer dieses Modells kennt keines der Zeichen im Text.")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                return self.model(**enc).waveform[0].float().cpu().numpy()

        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        limit = int(getattr(self.model.config, "max_text_positions", 600) or 600)
        ids = ids[:, :limit].to(self.device)
        with torch.no_grad():
            out = self.model.generate_speech(ids, self.speaker.to(self.device), vocoder=self.vocoder)
        out = out.float().cpu().numpy()
        return out if self.vocoder is not None else griffin_lim(out, self.sampling_rate)


def _resolve_speaker(model_path: Path, options: dict, engine: TtsEngine) -> None:
    import torch

    explicit = str(options.get("speaker_embedding") or "").strip()
    audio = str(options.get("speaker_audio") or "").strip()
    saved = model_path / SPEAKER_FILE
    fixed = explicit.lower() in ("default", "neutral")
    if explicit and not fixed and explicit.lower() not in ("auto", "xvector"):
        vec, src = load_speaker_file(explicit), f"Datei {Path(explicit).name}"
    elif audio:
        import librosa
        wave, _ = librosa.load(audio, sr=16000, mono=True)
        vec, src = XVectorEncoder().embed(wave), f"x-vector aus {Path(audio).name}"
    elif saved.is_file() and not fixed:
        vec, src = load_speaker_file(saved), "Stimme aus dem Training"
    else:
        vec, src = default_speaker(), "Standardstimme"
        if not fixed:
            engine.notes.append(
                "Kein Sprecher-Embedding beim Modell — es spricht mit der Standardstimme. "
                "Eigene Stimme: plugin_config speaker_embedding (.npy mit 512 Werten) oder "
                "speaker_audio (Aufnahme, braucht speechbrain).")
    engine.speaker = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    engine.speaker_source = src


def load_tts(model_path, device=None, options: Optional[dict] = None,
             status: Optional[Callable[[str], None]] = None) -> TtsEngine:
    """Laedt SpeechT5 (+ Vocoder, Sprecher) oder ein VITS-Modell fuer die Inferenz."""
    import torch

    model_path = Path(model_path)
    options = options or {}
    cfg = read_model_config(model_path)
    kind = tts_kind(cfg)
    if kind is None:
        raise ValueError(
            f"Modell-Architektur '{cfg.get('model_type') or 'unbekannt'}' ist kein Sprachausgabe-Modell "
            "(nicht unterstützt: erwartet SpeechT5 fuer Text-to-Speech oder VITS / MMS-TTS).")
    engine = TtsEngine()
    engine.kind = kind
    engine.meta = read_tts_meta(model_path)
    engine.device = device if device is not None else torch.device("cpu")
    engine.seed = int(options.get("seed", 0) or 0)

    if kind == "vits":
        from transformers import AutoTokenizer, VitsModel
        engine.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if getattr(engine.tokenizer, "is_uroman", False) and importlib.util.find_spec("uroman") is None:
            raise ImportError(
                "Dieses MMS-TTS-Modell erwartet romanisierten Text und braucht dafuer das Paket 'uroman' "
                "(pip install uroman).")
        engine.model = VitsModel.from_pretrained(str(model_path))
        engine.sampling_rate = int(getattr(engine.model.config, "sampling_rate", 16000) or 16000)
        engine.vocoder_source = "im Modell enthalten"
    else:
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
        engine.tokenizer = SpeechT5Tokenizer.from_pretrained(str(model_path))
        engine.known_chars = vocab_chars(engine.tokenizer)
        engine.model = SpeechT5ForTextToSpeech.from_pretrained(str(model_path))
        engine.vocoder, engine.vocoder_source = load_vocoder(
            model_path, options.get("vocoder"), allow_download=True, status=status)
        if engine.vocoder is None:
            engine.notes.append(
                f"Kein Vocoder ({DEFAULT_VOCODER}) verfuegbar — das Audio wird mit Griffin-Lim "
                "geschaetzt und klingt blechern. Mit Vocoder klingt dasselbe Modell deutlich besser.")
        _resolve_speaker(model_path, options, engine)

    engine.model.to(engine.device).eval()
    if engine.vocoder is not None:
        engine.vocoder.to(engine.device).eval()
    return engine


def write_wav(path, wave, sampling_rate: int) -> Path:
    """16-Bit-PCM-WAV — spielt jeder Browser und der Audio-Player der App."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.clip(np.asarray(wave, dtype=np.float32).reshape(-1), -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), wave, int(sampling_rate), subtype="PCM_16")
    except ImportError:
        import wave as wav_mod
        with wav_mod.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sampling_rate))
            w.writeframes((wave * 32767.0).astype("<i2").tobytes())
    return path


def safe_filename(text: str, max_len: int = 40) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", unicodedata.normalize("NFKD", str(text or ""))).strip("_")
    return (stem[:max_len].strip("_") or "audio").lower()


# ── Kennzahl: Ruecktranskription ─────────────────────────────────────────────

def _usable_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / n).is_file() for n in ("model.safetensors", "pytorch_model.bin"))


def find_local_asr(choice: Any = None, candidates: Iterable[str] = ASR_CANDIDATES) -> Optional[str]:
    """Pfad eines lokal vorhandenen ASR-Modells — ohne Download. None = keine Kennzahl.

    choice: Ordner, HF-ID (nur aus dem lokalen Cache) oder 'none'. Leer/auto
    sucht die ueblichen kleinen Modelle im HF-Cache.
    """
    choice = str(choice or "").strip()
    if choice.lower() in ("none", "off", "aus", "false"):
        return None
    wanted = [choice] if choice and choice.lower() != "auto" else list(candidates)
    for name in wanted:
        p = Path(name).expanduser()
        if p.is_dir():
            if _usable_model_dir(p):
                return str(p)
            continue
        snap = _cached_snapshot(name)
        if snap and _usable_model_dir(Path(snap)):
            return snap
    return None


class BackTranscriber:
    """Hoert erzeugtes Audio mit einem ASR-Modell ab."""

    def __init__(self, model_dir: str, device=None, language: str = ""):
        from transformers import pipeline
        dev = getattr(device, "type", "cpu")
        self.pipe = pipeline(
            "automatic-speech-recognition", model=str(model_dir),
            device=0 if dev == "cuda" else ("mps" if dev == "mps" else -1))
        self.model_dir = str(model_dir)
        self.sampling_rate = int(getattr(self.pipe.feature_extractor, "sampling_rate", 16000) or 16000)
        cfg = self.pipe.model.config
        multilingual = str(getattr(cfg, "model_type", "")) == "whisper" and int(getattr(cfg, "vocab_size", 0)) >= 51865
        self.generate_kwargs = {"language": language, "task": "transcribe"} if (language and multilingual) else {}
        self.name = Path(model_dir).parent.parent.name.replace("models--", "").replace("--", "/") \
            if "snapshots" in Path(model_dir).parts else Path(model_dir).name

    def transcribe(self, wave, sampling_rate: int) -> str:
        import numpy as np
        wave = np.asarray(wave, dtype=np.float32)
        if sampling_rate != self.sampling_rate:
            import librosa
            wave = librosa.resample(wave, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        kwargs: Dict[str, Any] = {}
        if len(wave) > 30 * self.sampling_rate:
            kwargs["chunk_length_s"] = 30
        if self.generate_kwargs:
            kwargs["generate_kwargs"] = dict(self.generate_kwargs)
        try:
            out = self.pipe({"raw": wave, "sampling_rate": self.sampling_rate}, **kwargs)
        except (ValueError, TypeError):
            # z. B. englisches Whisper, das keine Sprachangabe annimmt
            kwargs.pop("generate_kwargs", None)
            self.generate_kwargs = {}
            out = self.pipe({"raw": wave, "sampling_rate": self.sampling_rate}, **kwargs)
        return str((out.get("text") if isinstance(out, dict) else out) or "").strip()


def comparable_text(text: str) -> str:
    """Eingabetext so, wie ihn ein ASR-Modell schreiben wuerde (ae -> ä bleibt ae).

    Beide Seiten laufen durch dieselbe Umschrift: SpeechT5 spricht "Gruesse",
    Whisper schreibt "Grüße" — ohne Angleichung zaehlte das als Fehler, obwohl
    das Audio stimmt.
    """
    out = []
    for ch in str(text or ""):
        out.append(_TRANSLIT.get(ch, ch) if ch.isalpha() else ch)
    return "".join(out)


def intelligibility(transcripts: Sequence[str], texts: Sequence[str]) -> Dict[str, float]:
    """WER/CER der Ruecktranskription gegen den Eingabetext (0 = jedes Wort verstanden)."""
    from .asr import error_rates
    return error_rates([comparable_text(t) for t in transcripts], [comparable_text(t) for t in texts])
