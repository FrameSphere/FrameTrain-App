"""Test-Plugin: Sprachausgabe (SpeechT5 trainiert oder im Original, VITS / MMS-TTS).

single:  Text -> WAV-Datei zum Anhoeren
dataset: spricht die Transkripte des Testsplits und misst die Verstaendlichkeit
         per Ruecktranskription — ein kleines ASR-Modell hoert das erzeugte
         Audio ab, WER/CER gegen den Eingabetext.

Die Kennzahl gibt es nur, wenn ein ASR-Modell LOKAL vorhanden ist (HF-Cache
oder plugin_config asr_model). Es wird nichts heruntergeladen: ohne ASR-Modell
entstehen die Audiodateien trotzdem, nur ohne WER/CER und ohne Trefferquote.

Was die Kennzahl sagt und was nicht: WER misst, ob die Woerter zu verstehen
sind — nicht, ob die Stimme natuerlich klingt oder der Zielstimme aehnelt.
Dafuer bleibt nur das Anhoeren.

plugin_config:
    asr_model           Ordner oder HF-ID (nur lokaler Cache) | "none"; leer = suchen
    language            Sprache fuer die Ruecktranskription ("de", "en"); leer = aus dem Training
    speaker_embedding   .npy mit 512 Werten | "default"; leer = Stimme aus dem Training
    speaker_audio       Aufnahme als Stimmvorlage (braucht speechbrain)
    vocoder             Pfad/HF-ID eines HiFi-GAN oder "griffin_lim"
    max_audios          Obergrenze im Datensatz-Lauf (Standard 25)
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data.asr import evaluation_items, normalize_transcript, sample_wer
from ft_data.media import sample as random_sample
from ft_data.tts import (
    BackTranscriber, comparable_text, find_local_asr, intelligibility, load_tts,
    safe_filename, write_wav,
)

DEFAULT_MAX_AUDIOS = 25


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.engine = None
        self.asr: Optional[BackTranscriber] = None
        self.device = None
        self.language = ""
        self.is_stopped = False
        self.max_audios = int(config.plugin_config.get("max_audios", DEFAULT_MAX_AUDIOS) or DEFAULT_MAX_AUDIOS)

    def stop(self):
        self.is_stopped = True

    def setup(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Sprachausgabe-Modell...")
        self.device = resolve_device()
        try:
            self.engine = load_tts(model_path, self.device, self.config.plugin_config,
                                   status=lambda m: TestProtocol.status("loading", m))
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type == "cpu":
                raise
            # Einzelne Operationen fehlen auf MPS — dann rechnet die CPU.
            import torch
            TestProtocol.status("loading", f"{self.device} nicht nutzbar ({type(exc).__name__}) — rechne auf der CPU.")
            self.device = torch.device("cpu")
            self.engine = load_tts(model_path, self.device, self.config.plugin_config)
        for note in self.engine.notes:
            TestProtocol.warning(note)
        self.language = str(self.config.plugin_config.get("language", "")
                            or self.engine.meta.get("language", "") or "").strip().lower()
        TestProtocol.status(
            "loading",
            f"Modell geladen ({self.engine.kind}) | Gerät: {self.device} | "
            f"Vocoder: {self.engine.vocoder_source or 'Griffin-Lim'}"
            + (f" | Stimme: {self.engine.speaker_source}" if self.engine.speaker_source else ""))

    def _speak(self, text: str):
        """Erzeugt Audio; faellt bei einem Geraetefehler einmalig auf die CPU zurueck."""
        try:
            return self.engine.synthesize(text)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type == "cpu":
                raise
            import torch
            TestProtocol.status("running", f"{self.device} nicht nutzbar ({type(exc).__name__}) — rechne auf der CPU.")
            self.device = torch.device("cpu")
            self.engine.device = self.device
            self.engine.model.to(self.device)
            if self.engine.vocoder is not None:
                self.engine.vocoder.to(self.device)
            return self.engine.synthesize(text)

    def _load_asr(self) -> None:
        """Ruecktranskription vorbereiten — oder sauber ohne Kennzahl weitermachen."""
        found = find_local_asr(self.config.plugin_config.get("asr_model"))
        if not found:
            TestProtocol.warning(
                "Kein ASR-Modell lokal vorhanden — die Audiodateien werden erzeugt, aber ohne WER/CER. "
                "Fuer die Kennzahl ein kleines Spracherkennungsmodell laden (z. B. openai/whisper-tiny) "
                "oder seinen Ordner in plugin_config asr_model angeben.")
            return
        try:
            import torch
            # Das ASR-Modell laeuft auf der CPU: es ist klein, und so teilt es
            # sich den Grafikspeicher nicht mit dem TTS-Modell.
            self.asr = BackTranscriber(found, torch.device("cpu"), self.language)
            TestProtocol.status("loading", f"Ruecktranskription mit {self.asr.name}"
                                           + (f" (Sprache: {self.language})" if self.language else ""))
        except Exception as exc:
            self.asr = None
            TestProtocol.warning(f"ASR-Modell nicht ladbar ({type(exc).__name__}: {exc}) — Test laeuft ohne WER/CER.")

    def run_single(self):
        text = (self.config.single_input or "").strip()
        if not text:
            raise ValueError("Text ist leer.")
        t0 = time.time()
        wave, sr = self._speak(text)
        out = write_wav(Path(self.config.output_path) / f"{safe_filename(text)}.wav", wave, sr)
        seconds = len(wave) / sr
        extra: Dict[str, Any] = {
            "output_kind": "audio", "audio_path": str(out), "text": text,
            "audio_seconds": seconds, "sampling_rate": sr,
        }
        if self.engine.last_dropped:
            extra["dropped_characters"] = sorted(self.engine.last_dropped)
            TestProtocol.warning("Nicht sprechbare Zeichen ausgelassen: " + " ".join(sorted(self.engine.last_dropped)))
        TestProtocol.complete_single(
            predicted=str(out), confidence=None, top_predictions=[],
            inference_time=time.time() - t0, extra=extra)

    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        overrides = {k: self.config.plugin_config.get(k) for k in ("text_column", "audio_column")
                     if self.config.plugin_config.get(k)}
        items, split, notes = evaluation_items(root, overrides,
                                               status=lambda m: TestProtocol.status("loading", m))
        for note in notes:
            TestProtocol.status("loading", note)
        if not items:
            raise ValueError(
                f"Keine Paare aus Audio und Transkript in '{root}' gefunden "
                "(erwartet: aufnahme.wav + aufnahme.txt, metadata.csv, Common Voice oder Parquet).")
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}/")
        n = self.config.max_samples or self.max_audios
        if n > self.max_audios:
            TestProtocol.status("loading", f"Sprachausgabe ist langsam — begrenzt auf {self.max_audios} "
                                           "Saetze (plugin_config max_audios).")
            n = self.max_audios
        items = random_sample(items, n)
        self._load_asr()
        TestProtocol.status("running", f"{len(items)} Saetze werden gesprochen...")

        out_dir = Path(self.config.output_path)
        audio_dir = out_dir / "audio"
        predictions: List[Dict[str, Any]] = []
        transcripts: List[str] = []
        texts: List[str] = []
        exact = 0
        total_time = 0.0
        audio_total = 0.0
        started = time.time()
        for idx, (ref_audio, text) in enumerate(items, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            row: Dict[str, Any] = {
                "sample_id": idx, "input_text": text, "input_path": str(ref_audio),
                "expected_output": text, "predicted_output": "", "is_correct": None,
                "loss": None, "confidence": None,
            }
            try:
                wave, sr = self._speak(text)
                path = write_wav(audio_dir / f"{idx:04d}_{safe_filename(text, 24)}.wav", wave, sr)
                row["audio_path"] = str(path)
                row["audio_seconds"] = len(wave) / sr
                audio_total += row["audio_seconds"]
                if self.asr is not None:
                    heard = self.asr.transcribe(wave, sr)
                    row["predicted_output"] = heard
                    row["wer"] = sample_wer(comparable_text(heard), comparable_text(text))
                    row["is_correct"] = (normalize_transcript(comparable_text(heard))
                                         == normalize_transcript(comparable_text(text)))
                    exact += int(row["is_correct"])
                    transcripts.append(heard)
                    texts.append(text)
                else:
                    row["predicted_output"] = str(path)
            except Exception as exc:  # ein Satz, den das Modell nicht sprechen kann, beendet den Lauf nicht
                row["error_type"] = type(exc).__name__
                row["predicted_output"] = ""
                if self.asr is not None:
                    row["is_correct"] = False
                    row["wer"] = 1.0
                    transcripts.append("")
                    texts.append(text)
            dt = time.time() - t0
            total_time += dt
            row["inference_time"] = dt
            predictions.append(row)
            TestProtocol.progress(current=idx, total=len(items), sps=idx / max(time.time() - started, 1e-6))

        elapsed = max(time.time() - started, 1e-6)
        rates = intelligibility(transcripts, texts) if self.asr is not None else {}
        measured = self.asr is not None and bool(rates)
        produced = [p for p in predictions if p.get("audio_path")]
        metrics: Dict[str, Any] = {
            **rates,
            "total_samples": len(predictions),
            "generated_audios": len(produced),
            "audio_seconds": audio_total,
            # < 1 = schneller als Echtzeit (Rechenzeit je Sekunde Audio, inkl. Ruecktranskription)
            "real_time_factor": total_time / audio_total if audio_total > 0 else None,
            "average_inference_time": total_time / max(len(predictions), 1),
            "samples_per_second": len(predictions) / elapsed,
            "total_time": elapsed,
            "asr_model": self.asr.name if self.asr is not None else None,
        }
        if measured:
            # "Accuracy" wie bei der Spracherkennung: der Satz wurde Wort fuer Wort verstanden.
            metrics["exact_match"] = exact / max(len(predictions), 1)
            metrics["accuracy"] = metrics["exact_match"]
            metrics["correct_predictions"] = exact
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": predictions, "metrics": metrics},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        hard_file: Optional[str] = None
        hard = sorted((p for p in predictions if p.get("is_correct") is False),
                      key=lambda p: -(p.get("wer") or 0.0))
        if hard:
            hard_file = str(out_dir / "hard_examples.json")
            Path(hard_file).write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")
        if measured:
            TestProtocol.status("running", f"Verstaendlichkeit: WER {rates['wer']:.3f} | CER {rates['cer']:.3f}")

        shown = {k: rates[k] for k in ("wer", "cer") if k in rates}
        if metrics["real_time_factor"] is not None:
            shown["real_time_factor"] = metrics["real_time_factor"]
        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=len(predictions),
            accuracy=metrics.get("exact_match") if measured else None,
            correct=exact if measured else None,
            average_loss=None,
            average_inference_time=metrics["average_inference_time"],
            samples_per_second=metrics["samples_per_second"],
            hard_examples_file=hard_file,
            metrics=shown,
            extra={"output_kind": "audio", "audios": [p["audio_path"] for p in produced[:12]],
                   "asr_model": metrics["asr_model"]},
        )
