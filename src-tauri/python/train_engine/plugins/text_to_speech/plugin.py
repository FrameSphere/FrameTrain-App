"""
Text-to-Speech / Sprachausgabe (SpeechT5)
=========================================
Text rein, Sprache raus. Fine-Tuning von microsoft/speecht5_tts auf Aufnahmen
mit Transkript — das Gegenstueck zur Spracherkennung, mit denselben Daten.

Was trainiert wird: SpeechT5 sagt aus Zeichen ein Mel-Spektrogramm voraus; der
Loss vergleicht es mit dem Spektrogramm der Aufnahme. Die Wellenform erzeugt
erst der Vocoder (microsoft/speecht5_hifigan) — er wird NICHT mittrainiert,
sondern beim Export neben das Modell gelegt, damit Test, Labor und Hosting ohne
Netz Audio erzeugen koennen.

Sprecher: je Aufnahme ein x-vector (512 Werte) ueber speechbrain. Das Paket ist
optional. Fehlt es, gilt fuer alle Aufnahmen ein Standardvektor und die Stimme wird allein ueber
die Gewichte gelernt — sinnvoll nur bei Daten mit einer Stimme (ft_data.tts).

VITS / MMS-TTS: nicht trainierbar (transformers: "Training of VITS is not
supported yet") — das Plugin lehnt solche Modelle mit Begruendung ab; das
Test-Plugin kann sie abspielen.

Daten: wie die Spracherkennung (ft_data.asr) — Audio + gleichnamige .txt,
HF-audiofolder (metadata.csv/jsonl), Common Voice, Parquet; train/ val/ test/
werden respektiert, sonst gehen 10 % von train in die Validierung.

plugin_config:
    speaker_embedding   auto (Standard) | xvector | default | Pfad zu .npy (512 Werte)
    language            Sprache der Daten ("de", "en") — fuer die Ruecktranskription im Test
    max_seconds         laengere Aufnahmen werden ausgelassen, nicht gekappt (Standard 15)
    min_seconds         kuerzere Aufnahmen werden ausgelassen (Standard 0.5)
    vocoder             Pfad/HF-ID eines HiFi-GAN oder "griffin_lim"
    sample_text         Text fuer die Hoerprobe nach dem Training
    text_column / audio_column   eigene Spaltennamen in metadata.csv/Parquet
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft
from ft_data.asr import audio_seconds, resolve_asr_layout
from ft_data.tts import (
    SPEAKER_FILE, TTS_META_FILE, VITS_NOT_TRAINABLE, VOCODER_DIR, collate_speecht5,
    load_speaker_file, load_vocoder, make_speaker_encoder, mean_speaker, mel_target,
    default_speaker, prepare_text, tts_kind, vocab_chars, write_wav,
)

SUPPORTED_ARCHITECTURES = ["speecht5"]

DEFAULT_MAX_SECONDS = 15.0
DEFAULT_MIN_SECONDS = 0.5


def filter_items(items, tokenizer, known, min_seconds: float, max_seconds: float, max_tokens: int,
                 duration=audio_seconds):
    """Paare, die sich fuers Training eignen: (Liste aus (Pfad, Text, input_ids), Statistik).

    Zu lange Aufnahmen werden ausgelassen statt gekappt: ein abgeschnittenes
    Audio zu einem vollstaendigen Text braechte dem Modell bei, Saetze mitten
    im Wort zu beenden.
    """
    kept = []
    stats = {"too_long": 0, "too_short": 0, "no_text": 0, "text_too_long": 0, "unknown_chars": set()}
    for path, text in items:
        prepared, unknown = prepare_text(text, known)
        stats["unknown_chars"] |= unknown
        if not prepared:
            stats["no_text"] += 1
            continue
        seconds = duration(path)
        if seconds is not None and seconds > max_seconds:
            stats["too_long"] += 1
            continue
        if seconds is not None and seconds < min_seconds:
            stats["too_short"] += 1
            continue
        ids = tokenizer(prepared)["input_ids"]
        if len(ids) > max_tokens:
            stats["text_too_long"] += 1
            continue
        kept.append((path, prepared, list(ids)))
    return kept, stats


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.model_cfg: Dict[str, Any] = {}
        self.model_type = ""
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.sampling_rate = 16000
        self.speaker_choice = str(pc.get("speaker_embedding", "auto") or "auto").strip()
        self.language = str(pc.get("language", "") or "").strip().lower()
        self.max_seconds = float(pc.get("max_seconds", DEFAULT_MAX_SECONDS) or DEFAULT_MAX_SECONDS)
        self.min_seconds = float(pc.get("min_seconds", DEFAULT_MIN_SECONDS) or 0.0)
        self.sample_text = str(pc.get("sample_text", "") or "").strip()
        self.overrides = {k: pc.get(k) for k in ("text_column", "audio_column") if pc.get(k)}
        self.speakers: Dict[str, np.ndarray] = {}
        self.speaker_source = "default"
        self.voice: Optional[np.ndarray] = None
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        cfg_path = Path(self.config.model_path) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Keine config.json im Modellordner: {self.config.model_path}")
        self.model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.model_type = str(self.model_cfg.get("model_type", "")).lower()
        kind = tts_kind(self.model_cfg)
        if kind == "vits":
            raise ValueError(f"Modell-Architektur 'vits' wird fuer das Training nicht unterstützt. {VITS_NOT_TRAINABLE}")
        if kind != "speecht5":
            raise ValueError(
                f"Modell-Architektur '{self.model_type or 'unbekannt'}' wird fuer Sprachausgabe nicht "
                "unterstützt.\nTrainierbar: SpeechT5 (microsoft/speecht5_tts).")
        from transformers import SpeechT5Processor

        self.processor = SpeechT5Processor.from_pretrained(self.config.model_path)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000) or 16000)
        MessageProtocol.status(
            "init", f"✓ Architektur erkannt: speecht5 (Text-to-Speech) | Abtastrate: {self.sampling_rate} Hz")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        layout = resolve_asr_layout(
            root, seed=self.config.seed, overrides=self.overrides,
            status=lambda m: MessageProtocol.status("loading_data", m))
        for note in layout.notes:
            MessageProtocol.status("loading_data", note)

        known = vocab_chars(self.tokenizer)
        max_tokens = int(self.model_cfg.get("max_text_positions", 600) or 600) - 2
        train, st_train = filter_items(layout.train, self.tokenizer, known,
                                       self.min_seconds, self.max_seconds, max_tokens)
        val, st_val = filter_items(layout.val, self.tokenizer, known,
                                   self.min_seconds, self.max_seconds, max_tokens)
        self._report_filter(st_train, st_val)
        if not train:
            raise ValueError(
                f"Nach dem Filtern bleibt keine Trainingsaufnahme uebrig (erlaubt: {self.min_seconds:g} bis "
                f"{self.max_seconds:g} s mit Text). Aufnahmen in einzelne Saetze schneiden oder "
                "plugin_config max_seconds erhoehen.")
        if not val:
            # Ohne Validierung gaebe es keinen Val-Loss — lieber ein Trainingsbeispiel doppelt nutzen.
            val = train[:1]
            MessageProtocol.warning("Keine Validierungsaufnahme uebrig — der Val-Loss wird auf einer "
                                    "Trainingsaufnahme gemessen und sagt nichts ueber neue Saetze.")

        def as_dict(rows):
            return {"path": [p for p, _, _ in rows], "text": [t for _, t, _ in rows],
                    "input_ids": [ids for _, _, ids in rows]}

        self.train_dataset = Dataset.from_dict(as_dict(train))
        self.eval_dataset = hft.cap_eval_dataset(
            Dataset.from_dict(as_dict(val)), getattr(self.config, "max_eval_samples", 0), self.config.seed)
        total = sum(d for d in (audio_seconds(p) for p, _, _ in train) if d)
        MessageProtocol.status(
            "loading_data",
            f"Format: {layout.source} | Train: {len(self.train_dataset)} ({total / 60:.1f} min) | "
            f"Val: {len(self.eval_dataset)} | Test (bleibt fuer den Test): {len(layout.test)}")
        if total < 10 * 60:
            MessageProtocol.warning(
                f"Nur {total / 60:.1f} Minuten Trainingsmaterial. Fuer eine erkennbar eigene Stimme sind "
                "etwa 30 Minuten bis einige Stunden saubere Aufnahmen einer Person ueblich; mit weniger "
                "passt sich vor allem die Klangfarbe an, die Aussprache bleibt die des Basismodells.")
        self._prepare_speakers()

    def _report_filter(self, *stats) -> None:
        total = {k: sum(s[k] for s in stats) for k in ("too_long", "too_short", "no_text", "text_too_long")}
        unknown = set().union(*(s["unknown_chars"] for s in stats))
        if total["too_long"]:
            MessageProtocol.warning(
                f"{total['too_long']} Aufnahme(n) laenger als {self.max_seconds:g} s ausgelassen "
                "(nicht gekappt: ein abgeschnittenes Audio passt nicht mehr zum Text). "
                "Besser vorher in einzelne Saetze schneiden.")
        if total["too_short"]:
            MessageProtocol.status("loading_data", f"{total['too_short']} Aufnahme(n) kuerzer als "
                                                   f"{self.min_seconds:g} s ausgelassen.")
        if total["no_text"] or total["text_too_long"]:
            MessageProtocol.status(
                "loading_data",
                f"{total['no_text']} Aufnahme(n) ohne sprechbaren Text, {total['text_too_long']} mit zu "
                "langem Text ausgelassen.")
        if unknown:
            digits = sorted(c for c in unknown if c.isdigit())
            MessageProtocol.warning(
                "Diese Zeichen kennt SpeechT5 nicht, sie fehlen im Trainingstext: "
                f"{' '.join(sorted(unknown)[:40])}."
                + (" Zahlen bitte ausschreiben (\"drei\" statt \"3\") — sonst hoert das Modell ein Wort, "
                   "zu dem es keinen Text sieht." if digits else ""))

    def _prepare_speakers(self) -> None:
        """Ein Sprecher-Vektor je Aufnahme — einmal vorab, nicht in jedem Schritt."""
        choice = self.speaker_choice
        paths = list(self.train_dataset["path"]) + list(self.eval_dataset["path"])
        if choice.lower() not in ("auto", "xvector", "neutral", "none", "default", "off"):
            vec = load_speaker_file(choice)
            self.speakers = {p: vec for p in paths}
            self.voice, self.speaker_source = vec, "file"
            MessageProtocol.status("loading_data", f"Sprecher-Embedding aus Datei: {Path(choice).name}")
            return
        encoder = make_speaker_encoder(choice, warn=MessageProtocol.warning)
        if encoder is None:
            vec = default_speaker()
            self.speakers = {p: vec for p in paths}
            self.voice, self.speaker_source = vec, "default"
            return
        import librosa
        MessageProtocol.status("loading_data", f"Berechne Sprecher-Embeddings (x-vector) fuer {len(paths)} Aufnahmen...")
        for i, p in enumerate(paths, start=1):
            if self.is_stopped:
                return
            wave, _ = librosa.load(p, sr=16000, mono=True)
            self.speakers[p] = encoder.embed(wave)
            if i % 50 == 0:
                MessageProtocol.status("loading_data", f"Sprecher-Embeddings: {i}/{len(paths)}")
        # Die Stimme fuer spaeter: Mittel ueber die Trainingsaufnahmen.
        self.voice = mean_speaker([self.speakers[p] for p in self.train_dataset["path"]])
        self.speaker_source = "xvector"
        MessageProtocol.status("loading_data", "✓ Sprecher-Embeddings berechnet (x-vector, speechbrain)")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import SpeechT5ForTextToSpeech

        MessageProtocol.status("building_model", "Lade SpeechT5 fuer Sprachausgabe...")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.config.model_path)
        # Der Cache beschleunigt nur die Erzeugung und stoert das Training.
        self.model.config.use_cache = False
        hft.repair_uninitialized_params(self.model)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"✓ Modell geladen | Parameter: {params / 1e6:.1f}M")

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _load_wave(self, path: str) -> np.ndarray:
        import librosa
        wave, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return wave

    def _collate(self, batch):
        features = [{
            "input_ids": row["input_ids"],
            "labels": mel_target(self.feature_extractor, self._load_wave(row["path"]), self.sampling_rate),
            "speaker": self.speakers[row["path"]],
        } for row in batch]
        return collate_speecht5(
            features, self.tokenizer.pad_token_id,
            int(getattr(self.model.config, "reduction_factor", 2) or 2))

    def train(self) -> None:
        from transformers import Trainer, TrainerCallback, TrainingArguments

        self.device_used = hft.device_name()
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")
        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1) // max(self.config.gradient_accumulation_steps, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)
        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), TrainingArguments,
            remove_unused_columns=False,
            # Nur der Loss: die vorhergesagten Spektrogramme aller Aufnahmen zu
            # sammeln, sprengt den Speicher und es gibt keine Kennzahl darauf.
            prediction_loss_only=True,
            label_names=["labels"],
        )
        self._trainer = hft.nan_safe_trainer(Trainer)(
            model=self.model, args=args,
            train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
            data_collator=self._collate,
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps)],
        )
        self._start_time = time.time()
        self._trainer.train(resume_from_checkpoint=hft.resume_checkpoint(self.config))
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, Any]:
        MessageProtocol.status("validating", "Finale Validierung (Spektrogramm-Loss)...")
        result = self._trainer.evaluate()
        metrics = hft.final_metrics(
            self, self._trainer, result, self._start_time, architecture=self.model_type, num_labels=0)
        # Keine Klassen — accuracy/f1 waeren irrefuehrende Nullen. Wie gut das
        # Audio ist, misst erst der Test (Ruecktranskription).
        for key in ("accuracy", "f1", "precision", "recall", "num_labels"):
            metrics.pop(key, None)
        metrics["speaker_embedding"] = self.speaker_source
        if self.language:
            metrics["language"] = self.language
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.config.use_cache = True
        self.model.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))
        voice = self.voice if self.voice is not None else default_speaker()
        np.save(str(out / SPEAKER_FILE), np.asarray(voice, dtype=np.float32))

        vocoder, vocoder_source = load_vocoder(
            self.config.model_path, (self.config.plugin_config or {}).get("vocoder"),
            status=lambda m: MessageProtocol.status("export", m))
        if vocoder is not None:
            # Neben das Modell: Test, Labor und Hosting brauchen dann kein Netz.
            vocoder.save_pretrained(str(out / VOCODER_DIR))
        else:
            MessageProtocol.warning(
                "Kein Vocoder verfuegbar — das Modell ist gespeichert, Audio wird bis dahin mit "
                "Griffin-Lim geschaetzt (blechern). microsoft/speecht5_hifigan laden behebt das.")
        meta = {
            "kind": "speecht5",
            "model_type": self.model_type,
            "language": self.language,
            "sampling_rate": self.sampling_rate,
            "speaker_embedding": self.speaker_source,
            "speaker_file": SPEAKER_FILE,
            "vocoder": VOCODER_DIR if vocoder is not None else "griffin_lim",
            "vocoder_source": vocoder_source,
            "n_train": len(self.train_dataset) if self.train_dataset is not None else 0,
        }
        sample = self._write_sample(out, vocoder)
        if sample:
            meta["sample"] = sample
        (out / TTS_META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)

    def _write_sample(self, out: Path, vocoder) -> Optional[str]:
        """Hoerprobe mit dem frisch trainierten Modell — der schnellste Eindruck vom Ergebnis."""
        try:
            import torch
            from ft_data.tts import griffin_lim

            text = self.sample_text or (self.eval_dataset["text"][0] if len(self.eval_dataset) else "")
            prepared, _ = prepare_text(text, vocab_chars(self.tokenizer))
            if not prepared:
                return None
            self.model.eval()
            device = next(self.model.parameters()).device
            ids = self.tokenizer(prepared, return_tensors="pt")["input_ids"].to(device)
            voice = self.voice if self.voice is not None else default_speaker()
            speaker = torch.tensor(np.asarray(voice), dtype=torch.float32).unsqueeze(0).to(device)
            if vocoder is not None:
                vocoder = vocoder.to(device).eval()
            with torch.no_grad():
                wave = self.model.generate_speech(ids, speaker, vocoder=vocoder).float().cpu().numpy()
            if vocoder is None:
                wave = griffin_lim(wave, self.sampling_rate)
            path = write_wav(out / "samples" / "sample.wav", wave, self.sampling_rate)
            (path.with_suffix(".txt")).write_text(text, encoding="utf-8")
            MessageProtocol.status("export", f"Hoerprobe gespeichert: {path} ({len(wave) / self.sampling_rate:.1f} s)")
            return "samples/sample.wav"
        except Exception as exc:  # die Hoerprobe darf den Export nicht verhindern
            MessageProtocol.warning(f"Hoerprobe nicht erzeugt: {type(exc).__name__}: {exc}")
            return None
