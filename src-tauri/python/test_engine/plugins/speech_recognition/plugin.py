"""Test-Plugin: Spracherkennung (Whisper & Co. als Seq2Seq, wav2vec2 & Co. mit CTC).

single:  Audiodatei -> Transkript
dataset: Transkripte fuer den Testsplit, WER/CER gegen die Referenz.

Transkribiert wird ueber die transformers-pipeline — dieselbe, die der
Modell-Server im Labor nutzt. So liefern Test und Labor fuer dieselbe Datei
denselben Text.
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
from ft_data.asr import (
    asr_mode, error_rates, evaluation_items, normalize_language,
    normalize_transcript, sample_wer,
)
from ft_data.media import sample as random_sample

WHISPER_WINDOW = 30.0


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.pipe = None
        self.device = None
        self.mode = ""
        self.model_type = ""
        self.meta: Dict[str, Any] = {}
        self.generate_kwargs: Dict[str, Any] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import pipeline

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        cfg_file = model_path / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Keine config.json in: {model_path}")
        model_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.model_type = str(model_cfg.get("model_type", "")).lower()
        self.mode = asr_mode(model_cfg) or ""
        if not self.mode:
            raise ValueError(
                f"Modell-Architektur '{self.model_type or 'unbekannt'}' ist kein Spracherkennungsmodell "
                "(nicht unterstützt: erwartet Whisper/Moonshine/Speech2Text oder wav2vec2 & Co. mit CTC-Kopf)."
            )
        meta_file = model_path / "asr_config.json"
        if meta_file.exists():
            try:
                self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except ValueError:
                self.meta = {}

        # Sprache: Test-Einstellung vor Training. Ein trainiertes Whisper traegt
        # sie schon in generation_config — dann ist hier nichts zu tun.
        lang = normalize_language(self.config.plugin_config.get("language", ""))
        task = str(self.config.plugin_config.get("task", "") or "").strip().lower()
        if self.model_type == "whisper":
            if lang:
                self.generate_kwargs["language"] = lang
            if task:
                self.generate_kwargs["task"] = task
            elif lang:
                self.generate_kwargs["task"] = "transcribe"

        TestProtocol.status("loading", f"Lade Spracherkennung ({self.mode}, {self.model_type})...")
        self.device = resolve_device()
        dev = self.device.type
        self.pipe = pipeline(
            "automatic-speech-recognition", model=str(model_path),
            device=0 if dev == "cuda" else ("mps" if dev == "mps" else -1),
        )
        TestProtocol.status("loading", f"Modell geladen | Gerät: {self.device}")

    def _transcribe(self, path: Path) -> str:
        import librosa

        sr = int(getattr(self.pipe.feature_extractor, "sampling_rate", 16000) or 16000)
        wave, _ = librosa.load(str(path), sr=sr, mono=True)
        kwargs: Dict[str, Any] = {}
        # Lange Aufnahmen in 30-s-Stuecke zerlegen (wie der Modell-Server) —
        # Whisper schneidet sonst nach 30 s einfach ab.
        if len(wave) > WHISPER_WINDOW * sr:
            kwargs["chunk_length_s"] = WHISPER_WINDOW
        if self.generate_kwargs:
            kwargs["generate_kwargs"] = dict(self.generate_kwargs)
        out = self.pipe({"raw": wave, "sampling_rate": sr}, **kwargs)
        text = out.get("text") if isinstance(out, dict) else str(out)
        return str(text or "").strip()

    def run_single(self):
        path = Path(self.config.single_input)
        if not path.exists():
            raise ValueError(f"Audiodatei nicht gefunden: {path}")
        t0 = time.time()
        text = self._transcribe(path)
        # Keine Konfidenz: eine ehrliche Wahrscheinlichkeit fuer einen ganzen
        # Satz gibt es nicht (wie beim Modell-Server).
        TestProtocol.complete_single(predicted=text, confidence=None, top_predictions=[],
                                     inference_time=time.time() - t0)

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
                "(erwartet: aufnahme.wav + aufnahme.txt, metadata.csv, Common Voice oder Parquet)."
            )
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}/")
        items = random_sample(items, self.config.max_samples)
        TestProtocol.status("running", f"{len(items)} Aufnahmen werden transkribiert...")

        predictions: List[Dict[str, Any]] = []
        preds: List[str] = []
        refs: List[str] = []
        exact = 0
        total_time = 0.0
        started = time.time()
        for idx, (path, reference) in enumerate(items, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            try:
                text = self._transcribe(Path(path))
                error = None
            except Exception as exc:  # eine kaputte Datei soll nicht den ganzen Lauf beenden
                text, error = "", type(exc).__name__
            dt = time.time() - t0
            total_time += dt
            is_correct = normalize_transcript(text) == normalize_transcript(reference)
            exact += int(is_correct)
            preds.append(text)
            refs.append(reference)
            row: Dict[str, Any] = {
                "sample_id": idx,
                "input_text": Path(path).name,
                "input_path": str(path),
                "expected_output": reference,
                "predicted_output": text,
                "is_correct": is_correct,
                "loss": None,
                "confidence": None,
                "wer": sample_wer(text, reference),
                "inference_time": dt,
            }
            if error:
                row["error_type"] = error
            predictions.append(row)
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(items), sps=idx / elapsed)

        elapsed = max(time.time() - started, 1e-6)
        rates = error_rates(preds, refs)
        metrics: Dict[str, Any] = {
            **rates,
            # "Accuracy" heisst hier wie bei Seq2Seq: Transkript Wort fuer Wort
            # identisch (nach Kleinschreibung und ohne Satzzeichen).
            "exact_match": exact / max(len(predictions), 1),
            "accuracy": exact / max(len(predictions), 1),
            "correct_predictions": exact,
            "total_samples": len(predictions),
            "average_inference_time": total_time / max(len(predictions), 1),
            "samples_per_second": len(predictions) / elapsed,
            "total_time": elapsed,
        }
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": predictions, "metrics": metrics},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        hard = sorted((p for p in predictions if not p["is_correct"]),
                      key=lambda p: -(p.get("wer") or 0.0))
        hard_file: Optional[str] = None
        if hard:
            hard_file = str(out_dir / "hard_examples.json")
            Path(hard_file).write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")
        if rates:
            TestProtocol.status("running", f"WER {rates['wer']:.3f} | CER {rates['cer']:.3f}")

        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=len(predictions),
            accuracy=metrics["exact_match"],
            correct=exact,
            average_loss=None,
            average_inference_time=metrics["average_inference_time"],
            samples_per_second=metrics["samples_per_second"],
            hard_examples_file=hard_file,
            metrics={k: rates[k] for k in ("wer", "cer") if k in rates},
        )
