"""
Speech Recognition / ASR (HuggingFace)
======================================
Audio rein, Text raus. Zwei Wege in einem Plugin, entschieden ueber die
config.json des Modells (ft_data.asr.asr_mode):

  seq2seq  Whisper, Moonshine, Speech2Text — ein Decoder erzeugt den Text
           Token fuer Token. Trainiert mit Seq2SeqTrainer; die Validierung
           generiert echten Text (predict_with_generate), sonst gaebe es keine WER.
  ctc      wav2vec2, HuBERT, WavLM, data2vec-audio — ein Zeichen pro Audio-Frame.
           Fehlt dem Modell der CTC-Kopf (z.B. facebook/wav2vec2-base), wird das
           Zeichen-Vokabular aus den Transkripten gebaut und mit exportiert.

Daten: siehe ft_data.asr — Audio + gleichnamige .txt (Datensatz-Werkstatt),
HF-audiofolder (metadata.csv/jsonl), Common Voice, Parquet. train/ val/ test/
werden respektiert, sonst gehen 10 % von train in die Validierung.

plugin_config:
    language      Whisper-Sprache ("de", "german", "deutsch"); leer = am
                  Trainingsmaterial erkennen
    task          "transcribe" (Standard) oder "translate"
    max_seconds   laengere Aufnahmen werden gekappt (Whisper: hoechstens 30)
    freeze_feature_encoder   CTC: CNN-Frontend einfrieren (Standard an)
    eval_before_training     Basis-WER vor dem Training messen (Seq2Seq)
    text_column / audio_column   eigene Spaltennamen in metadata.csv/Parquet
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft
from ft_data.asr import (
    CTC_PAD, CTC_UNK, CTC_WORD, asr_mode, audio_seconds, build_ctc_vocab, ctc_text,
    error_rates, has_ctc_head, normalize_language, resolve_asr_layout,
)

# Whisper verarbeitet Fenster von genau 30 Sekunden; alles dahinter faellt weg.
WHISPER_MAX_SECONDS = 30.0


def ctc_loss_on_cpu(model, logits, labels, attention_mask=None):
    """CTC-Loss wie in Wav2Vec2ForCTC, aber immer auf der CPU gerechnet.

    Apple Silicon (MPS) kennt aten::_ctc_loss nicht. Der Ausweich per
    PYTORCH_ENABLE_MPS_FALLBACK wirkt nur, wenn er vor dem torch-Import gesetzt
    ist — und im Test lief der Lauf damit nach ~15 Schritten in NaN. Hier bleibt
    das Modell auf MPS, nur die kleine Loss-Rechnung wandert auf die CPU; der
    Gradient fliesst ueber .cpu() zurueck.
    """
    import torch
    import torch.nn.functional as F

    if attention_mask is not None:
        input_lengths = model._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    else:
        input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    mask = labels >= 0
    targets = labels.masked_select(mask).cpu()
    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1).cpu()
    cfg = model.config
    return F.ctc_loss(
        log_probs, targets, input_lengths.cpu(), mask.sum(-1).cpu(),
        blank=cfg.pad_token_id, reduction=getattr(cfg, "ctc_loss_reduction", "mean"),
        zero_infinity=bool(getattr(cfg, "ctc_zero_infinity", True)),
    ).to(logits.device)


def _ctc_trainer_class():
    from transformers import Trainer

    class _CtcTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = ctc_loss_on_cpu(model, outputs.logits, labels, inputs.get("attention_mask"))
            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch
            loss = super().training_step(model, inputs, num_items_in_batch)
            # Auf MPS lieferte der Rueckweg durch wav2vec2 bei einzelnen Batches
            # NaN-Gradienten, obwohl Loss und Logit-Gradient endlich waren. Ein
            # einziger solcher Schritt macht alle Gewichte NaN — danach meldete
            # der Lauf Loss 0 und leere Transkripte. Solche Schritte verwerfen.
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads and not all(torch.isfinite(g).all() for g in grads):
                model.zero_grad(set_to_none=True)
                self._skipped_nan = getattr(self, "_skipped_nan", 0) + 1
                MessageProtocol.warning(
                    f"Schritt mit ungueltigen Gradienten (NaN) verworfen ({self._skipped_nan}x).")
                return loss.detach()
            return loss

    return _CtcTrainer


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "ja", "an", "on")


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.model_cfg: Dict[str, Any] = {}
        self.model_type = ""
        self.mode = ""                      # "seq2seq" | "ctc"
        self.is_whisper = False
        self.processor = None
        self.feature_extractor = None
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.sampling_rate = 16000
        self.language = normalize_language(pc.get("language", ""))
        self.task = str(pc.get("task", "transcribe") or "transcribe").strip().lower()
        self.max_seconds = float(pc.get("max_seconds", WHISPER_MAX_SECONDS) or WHISPER_MAX_SECONDS)
        self.freeze_encoder = _as_bool(pc.get("freeze_feature_encoder"), True)
        self.eval_before = _as_bool(pc.get("eval_before_training"), True)
        self.generation_max_length = int(pc.get("generation_max_length", 128) or 128)
        self.overrides = {k: pc.get(k) for k in ("text_column", "audio_column") if pc.get(k)}
        self.vocab_built = False
        self.text_case = "keep"             # CTC: lower | upper | keep
        self.baseline: Dict[str, float] = {}
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
        forced = str((self.config.plugin_config or {}).get("mode", "") or "").strip().lower()
        self.mode = forced if forced in ("seq2seq", "ctc") else (asr_mode(self.model_cfg) or "")
        if not self.mode:
            raise ValueError(
                f"Modell-Architektur '{self.model_type or 'unbekannt'}' wird fuer Spracherkennung "
                "nicht unterstuetzt.\nUnterstuetzt: Whisper, Moonshine, Speech2Text (Seq2Seq) sowie "
                "wav2vec2, HuBERT, WavLM, data2vec-audio (CTC)."
            )
        self.is_whisper = self.model_type == "whisper"
        if self.is_whisper:
            # Whisper sieht nie mehr als 30 s — ein hoeherer Wert waere eine Luege.
            self.max_seconds = min(self.max_seconds, WHISPER_MAX_SECONDS)

        MessageProtocol.status(
            "init",
            f"✓ Architektur erkannt: {self.model_type or 'unbekannt'} | Weg: "
            f"{'Seq2Seq (Decoder erzeugt Text)' if self.mode == 'seq2seq' else 'CTC (Zeichen je Frame)'}",
        )
        from transformers import AutoFeatureExtractor, AutoProcessor

        if self.mode == "seq2seq":
            self.processor = AutoProcessor.from_pretrained(self.config.model_path)
            self.feature_extractor = self.processor.feature_extractor
            self.tokenizer = self.processor.tokenizer
        else:
            # Der Tokenizer entsteht erst in load_data — ohne CTC-Kopf aus den Transkripten.
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_path)
        self.sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000) or 16000)
        MessageProtocol.status("init", f"Feature-Extractor geladen ✓ | Abtastrate: {self.sampling_rate} Hz")

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
        MessageProtocol.status(
            "loading_data",
            f"Format: {layout.source} | Train: {len(layout.train)} | Val: {len(layout.val)} | "
            f"Test (bleibt fuer den Test): {len(layout.test)}",
        )
        self._warn_long(layout.train + layout.val)

        if self.mode == "ctc":
            self._prepare_ctc_tokenizer([t for _, t in layout.train + layout.val])

        def as_dict(items):
            return {"path": [p for p, _ in items], "text": [self._target_text(t) for _, t in items]}

        self.train_dataset = Dataset.from_dict(as_dict(layout.train))
        eval_ds = Dataset.from_dict(as_dict(layout.val))
        self.eval_dataset = hft.cap_eval_dataset(
            eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        MessageProtocol.status(
            "loading_data", f"✓ Train: {len(self.train_dataset)} | Eval: {len(self.eval_dataset)}")

    def _warn_long(self, items) -> None:
        durations = [audio_seconds(p) for p, _ in items]
        long = [d for d in durations if d is not None and d > self.max_seconds]
        if long:
            MessageProtocol.warning(
                f"{len(long)} Aufnahme(n) sind laenger als {self.max_seconds:.0f} s (laengste: "
                f"{max(long):.0f} s) und werden gekappt"
                + (" — Whisper verarbeitet hoechstens 30 s am Stueck. " if self.is_whisper else ". ")
                + "Das Transkript beschreibt dann mehr, als das Modell hoert; "
                "besser vorher in kurze Abschnitte schneiden."
            )

    def _target_text(self, text: str) -> str:
        if self.mode != "ctc":
            return text
        if self.text_case == "upper":
            return ctc_text(text, lowercase=False).upper()
        if self.text_case == "lower":
            return ctc_text(text, lowercase=True)
        return ctc_text(text, lowercase=False)

    def _prepare_ctc_tokenizer(self, texts: List[str]) -> None:
        """Vorhandenes CTC-Vokabular nutzen oder eines aus den Transkripten bauen."""
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor

        choice = str((self.config.plugin_config or {}).get("vocab", "auto") or "auto").lower()
        model_dir = Path(self.config.model_path)
        use_model_vocab = (choice == "model") or (
            choice == "auto" and has_ctc_head(self.model_cfg) and (model_dir / "vocab.json").exists())

        if use_model_vocab:
            tok = Wav2Vec2CTCTokenizer.from_pretrained(str(model_dir))
            letters = [c for c in tok.get_vocab() if len(c) == 1 and c.isalpha()]
            # Die englischen 960h-Modelle kennen nur GROSSBUCHSTABEN — ein
            # kleingeschriebenes Transkript bestuende sonst nur aus [UNK].
            if letters and all(c.isupper() for c in letters):
                self.text_case = "upper"
            elif letters and all(c.islower() for c in letters):
                self.text_case = "lower"
            known = set(tok.get_vocab())
            missing = sorted({c for t in texts for c in self._target_text(t)
                              if c != " " and c not in known})
            if missing:
                MessageProtocol.warning(
                    f"Diese Zeichen kennt das Vokabular des Modells nicht und werden als "
                    f"Unbekannt gelernt: {' '.join(missing[:30])}. Fuer eine andere Sprache "
                    "besser ein Modell ohne CTC-Kopf nehmen (plugin_config vocab='build')."
                )
            MessageProtocol.status("loading_data", f"CTC: Vokabular des Modells ({len(known)} Zeichen)")
        else:
            self.text_case = "lower"
            vocab = build_ctc_vocab(self._target_text(t) for t in texts)
            vocab_dir = Path(self.config.effective_output_dir()) / "ctc_vocab"
            vocab_dir.mkdir(parents=True, exist_ok=True)
            vocab_file = vocab_dir / "vocab.json"
            vocab_file.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
            tok = Wav2Vec2CTCTokenizer(
                str(vocab_file), unk_token=CTC_UNK, pad_token=CTC_PAD, word_delimiter_token=CTC_WORD,
                bos_token=None, eos_token=None,
            )
            self.vocab_built = True
            chars = "".join(k for k in vocab if len(k) == 1 and k != CTC_WORD)
            MessageProtocol.status(
                "loading_data",
                f"CTC: Vokabular aus den Transkripten gebaut ({len(vocab)} Eintraege: {chars[:60]})",
            )
        self.tokenizer = tok
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=tok)

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        MessageProtocol.status("building_model", "Lade Modell fuer Spracherkennung...")
        if self.mode == "seq2seq":
            from transformers import AutoModelForSpeechSeq2Seq
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.config.model_path)
            if self.is_whisper:
                self._configure_whisper()
        else:
            from transformers import AutoModelForCTC
            self.model = AutoModelForCTC.from_pretrained(
                self.config.model_path,
                vocab_size=len(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id,
                ctc_loss_reduction="mean",
                # Eine Aufnahme, die kuerzer ist als ihr Transkript, liefert sonst
                # einen unendlichen Loss und reisst den ganzen Lauf auf NaN.
                ctc_zero_infinity=True,
                ignore_mismatched_sizes=True,
            )
            if self.freeze_encoder and hasattr(self.model, "freeze_feature_encoder"):
                self.model.freeze_feature_encoder()
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"✓ Modell geladen | Parameter: {params/1e6:.1f}M")

    def _configure_whisper(self) -> None:
        """Sprache und Aufgabe in Labels UND Generierung festschreiben.

        Die Labels beginnen mit <|startoftranscript|><|de|><|transcribe|><|notimestamps|>.
        Weicht die Generierung davon ab (andere Sprache, alte forced_decoder_ids),
        misst die WER etwas anderes als das Training gelernt hat.
        """
        gc = self.model.generation_config
        multilingual = bool(getattr(gc, "is_multilingual", True))
        if multilingual:
            if not self.language:
                self.language = self._detect_language()
            self.tokenizer.set_prefix_tokens(language=self.language or None, task=self.task)
            gc.language = self.language or None
            gc.task = self.task
            MessageProtocol.status(
                "building_model", f"Whisper: Sprache '{self.language or 'automatisch'}', Aufgabe '{self.task}'")
        elif self.language:
            MessageProtocol.warning(
                "Dieses Whisper-Modell ist nur englisch (.en) — die Einstellung language wird ignoriert.")
            self.language = ""
        # forced_decoder_ids sind der alte Weg und wuerden language/task uebersteuern.
        gc.forced_decoder_ids = None
        if hasattr(self.model.config, "forced_decoder_ids"):
            self.model.config.forced_decoder_ids = None

    def _detect_language(self) -> str:
        """Sprache per Mehrheitsentscheid ueber bis zu 8 Trainingsaufnahmen."""
        try:
            import torch
            from collections import Counter
            paths = list(self.train_dataset["path"])[:8]
            if not paths:
                return ""
            waves = [self._load_wave(p) for p in paths]
            feats = self.feature_extractor(waves, sampling_rate=self.sampling_rate, return_tensors="pt")
            with torch.no_grad():
                ids = self.model.detect_language(feats["input_features"])
            codes = [self.tokenizer.decode([int(i)]).strip("<|>") for i in ids.view(-1).tolist()]
            lang, count = Counter(codes).most_common(1)[0]
            MessageProtocol.status(
                "building_model", f"Sprache am Trainingsmaterial erkannt: {lang} ({count}/{len(codes)})")
            return lang
        except Exception as exc:
            MessageProtocol.warning(f"Sprache nicht erkannt ({exc}) — Whisper waehlt sie je Aufnahme selbst. "
                                    "Besser language in den Plugin-Parametern setzen.")
            return ""

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _load_wave(self, path: str) -> np.ndarray:
        import librosa
        wave, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        max_len = int(self.max_seconds * self.sampling_rate)
        return wave[:max_len] if len(wave) > max_len else wave

    def _features(self, waves):
        if self.is_whisper:
            # Whisper erwartet immer 3000 Frames (30 s) — kein "longest"-Padding.
            return self.feature_extractor(waves, sampling_rate=self.sampling_rate, return_tensors="pt")
        return self.feature_extractor(
            waves, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)

    def _seq2seq_labels(self, texts: List[str]):
        import torch
        tok = self.tokenizer
        rows = tok(texts, add_special_tokens=True)["input_ids"]
        eos = tok.eos_token_id
        start = getattr(self.model.config, "decoder_start_token_id", None)
        fixed = []
        for ids in rows:
            ids = list(ids)
            # Das Modell setzt den Start-Token beim Verschieben selbst davor;
            # stuende er auch im Label, lernte es ihn doppelt.
            if start is not None and ids and ids[0] == start:
                ids = ids[1:]
            if eos is not None and (not ids or ids[-1] != eos):
                ids.append(eos)
            fixed.append(ids[: self.generation_max_length])
        width = max(len(r) for r in fixed)
        labels = torch.full((len(fixed), width), -100, dtype=torch.long)
        for i, r in enumerate(fixed):
            labels[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return labels

    def _ctc_labels(self, texts: List[str]):
        enc = self.tokenizer(texts, padding=True, return_tensors="pt")
        return enc["input_ids"].masked_fill(enc["attention_mask"].ne(1), -100)

    def _collate(self, batch):
        waves = [self._load_wave(row["path"]) for row in batch]
        texts = [row["text"] for row in batch]
        enc = self._features(waves)
        enc["labels"] = self._seq2seq_labels(texts) if self.mode == "seq2seq" else self._ctc_labels(texts)
        return enc

    def _decode(self, ids) -> List[str]:
        ids = np.asarray(ids)
        ids = np.where(ids < 0, self.tokenizer.pad_token_id or 0, ids)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _compute_metrics(self, eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        if self.mode == "ctc":
            pred_text = self.tokenizer.batch_decode(np.asarray(preds))
            labels = np.where(np.asarray(labels) < 0, self.tokenizer.pad_token_id, labels)
            # group_tokens=False: im Label sind doppelte Buchstaben ("ll") echt.
            ref_text = self.tokenizer.batch_decode(labels, group_tokens=False)
        else:
            pred_text = self._decode(preds)
            ref_text = self._decode(labels)
        rates = error_rates(pred_text, ref_text)
        if getattr(self, "_log_examples", True) and pred_text:
            MessageProtocol.status("evaluating", f"Beispiel: '{ref_text[0]}' -> '{pred_text[0]}'")
        return rates

    def train(self) -> None:
        from transformers import Trainer, TrainerCallback

        self.device_used = hft.device_name()
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1) // max(self.config.gradient_accumulation_steps, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)
        callbacks = [hft.progress_callback(TrainerCallback, self, total_steps)]

        if self.mode == "seq2seq":
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            args = hft.build_training_arguments(
                self.config, self.config.effective_output_dir(), Seq2SeqTrainingArguments,
                remove_unused_columns=False,
                predict_with_generate=True,
                generation_max_length=self.generation_max_length,
                generation_num_beams=1,
            )
            self._trainer = Seq2SeqTrainer(
                model=self.model, args=args,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                data_collator=self._collate, compute_metrics=self._compute_metrics,
                callbacks=callbacks,
            )
        else:
            import torch
            args = hft.build_training_arguments(
                self.config, self.config.effective_output_dir(),
                __import__("transformers").TrainingArguments,
                remove_unused_columns=False,
            )
            self._trainer = _ctc_trainer_class()(
                model=self.model, args=args,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                data_collator=self._collate, compute_metrics=self._compute_metrics,
                # Nur die Argmax-IDs aufheben: die vollen Logits (Frames x Vokabular)
                # je Aufnahme sprengen bei laengeren Splits sonst den Speicher.
                preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(
                    logits[0] if isinstance(logits, tuple) else logits, dim=-1),
                callbacks=callbacks,
            )

        if self.mode == "seq2seq" and self.eval_before and len(self.eval_dataset):
            self._measure_baseline()

        self._start_time = time.time()
        self._trainer.train()
        MessageProtocol.status("training", "Training abgeschlossen")

    def _measure_baseline(self) -> None:
        """WER des unveraenderten Modells — sonst weiss niemand, ob das Training half."""
        subset = self.eval_dataset
        if len(subset) > 100:
            subset = subset.select(range(100))
        MessageProtocol.status("evaluating", f"Basis-WER des unveraenderten Modells ueber {len(subset)} Aufnahmen...")
        try:
            res = self._trainer.evaluate(eval_dataset=subset, metric_key_prefix="baseline")
        except Exception as exc:
            MessageProtocol.warning(f"Basis-WER nicht messbar: {exc}")
            return
        if "baseline_wer" in res:
            self.baseline = {"wer_before": float(res["baseline_wer"]),
                             "cer_before": float(res.get("baseline_cer", 0.0))}
            MessageProtocol.status(
                "evaluating",
                f"Basis vor dem Training: WER {self.baseline['wer_before']:.3f} | "
                f"CER {self.baseline['cer_before']:.3f}")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Validierung (Transkripte werden erzeugt)...")
        result = self._trainer.evaluate()
        metrics = hft.final_metrics(
            self, self._trainer, result, self._start_time,
            architecture=self.model_type, num_labels=0)
        # Keine Klassen — accuracy/f1 waeren irrefuehrende Nullen (wie bei seq2seq).
        for key in ("accuracy", "f1", "precision", "recall", "num_labels"):
            metrics.pop(key, None)
        if "eval_wer" in result:
            metrics["wer"] = float(result["eval_wer"])
            metrics["cer"] = float(result.get("eval_cer", 0.0))
        metrics.update(self.baseline)
        metrics["asr_mode"] = self.mode
        if self.language:
            metrics["language"] = self.language
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        # Processor = Feature-Extractor + Tokenizer. Beim CTC-Weg mit neuem
        # Vokabular ist das die einzige Stelle, an der das Vokabular ueberlebt —
        # ohne vocab.json kann weder der Test noch das Labor Text erzeugen.
        self.processor.save_pretrained(str(out))
        meta = {
            "mode": self.mode,
            "model_type": self.model_type,
            "language": self.language,
            "task": self.task if self.is_whisper else None,
            "vocab_built": self.vocab_built,
            "text_case": self.text_case,
            "sampling_rate": self.sampling_rate,
            "max_seconds": self.max_seconds,
        }
        (out / "asr_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
