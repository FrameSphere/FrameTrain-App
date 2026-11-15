"""Tests fuer das TTS-Test-Plugin — ohne Modell: Erzeugung und ASR sind Attrappen.

Geprueft wird, was das Plugin aus den Ergebnissen macht: WAV-Dateien,
results.json, Kennzahlen mit ASR-Modell und der saubere Lauf ohne.

Aufruf:  python3.11 plugins/text_to_speech/test_tts_test_plugin.py   (aus test_engine/)
"""
import io
import json
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # test_engine/
sys.path.insert(0, str(HERE.parents[2]))          # python/ (ft_data)
sys.path.insert(0, str(HERE.parent))              # plugins/ (_shared_classify)

import numpy as np  # noqa: E402

from core.config import TestConfig  # noqa: E402
from plugins.text_to_speech import plugin as tts_plugin  # noqa: E402


class FakeEngine:
    kind = "speecht5"
    vocoder = None
    vocoder_source = "attrappe"
    speaker_source = "attrappe"
    notes = []
    meta = {"language": "de"}

    def __init__(self):
        self.last_dropped = set()
        self.spoken = []

    def synthesize(self, text):
        if "kaputt" in text:
            raise ValueError("nicht sprechbar")
        self.spoken.append(text)
        self.last_dropped = {c for c in text if c.isdigit()}
        return np.zeros(8000, dtype=np.float32), 16000


class FakeAsr:
    """Hoert genau den zuletzt gesprochenen Text — nur aus 'neun' wird 'nein'."""
    name = "attrappe/asr"
    engine = None

    def transcribe(self, wave_, sr):
        return self.engine.spoken[-1].replace("neun", "nein")


def _dataset(root: Path, texts):
    for i, t in enumerate(texts):
        d = root / "test"
        d.mkdir(parents=True, exist_ok=True)
        with wave.open(str(d / f"a{i}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 1600)
        (d / f"a{i}.txt").write_text(t, encoding="utf-8")


class TtsTestPluginTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _plugin(self, asr=None, **cfg):
        import torch
        config = TestConfig.from_dict({"task_type": "text_to_speech", "output_path": str(self.root / "out"), **cfg})
        p = tts_plugin.Plugin(config)
        p.engine = FakeEngine()
        p.device = torch.device("cpu")
        if asr is not None:
            asr.engine = p.engine
        p._load_asr = lambda: setattr(p, "asr", asr)
        return p

    def _run(self, fn):
        log = io.StringIO()
        with redirect_stdout(log):
            fn()
        return [json.loads(line) for line in log.getvalue().splitlines() if line.startswith("{")]

    def test_einzeltest_schreibt_abspielbare_wav(self):
        p = self._plugin(mode="single", single_input="Guten Morgen 7")
        msgs = self._run(p.run_single)
        done = next(m["data"] for m in msgs if m["type"] == "complete")
        self.assertEqual(done["output_kind"], "audio")
        self.assertEqual(done["predicted_output"], done["audio_path"])
        self.assertIsNone(done["confidence"])
        self.assertAlmostEqual(done["audio_seconds"], 0.5)
        self.assertEqual(done["dropped_characters"], ["7"])
        with wave.open(done["audio_path"], "rb") as w:
            self.assertEqual((w.getframerate(), w.getsampwidth(), w.getnframes()), (16000, 2, 8000))

    def test_einzeltest_ohne_text(self):
        with self.assertRaises(ValueError):
            self._plugin(mode="single", single_input="  ").run_single()

    def test_datensatz_mit_asr_misst_verstaendlichkeit(self):
        texts = ["Guten Morgen", "Es ist neun Uhr", "Grüße aus Köln"]
        _dataset(self.root / "ds", texts)
        p = self._plugin(asr=FakeAsr(), dataset_path=str(self.root / "ds"))
        msgs = self._run(p.run_dataset)
        done = next(m["data"] for m in msgs if m["type"] == "complete")
        self.assertEqual(done["total_samples"], 3)
        self.assertEqual(done["correct_predictions"], 2)
        self.assertAlmostEqual(done["accuracy"], 2 / 3)
        self.assertAlmostEqual(done["metrics"]["wer"], 1 / 9)        # 1 falsches von 9 Woertern
        self.assertGreater(done["metrics"]["cer"], 0)
        self.assertEqual(len(done["audios"]), 3)
        self.assertTrue(all(Path(a).is_file() for a in done["audios"]))
        results = json.loads(Path(done["results_file"]).read_text(encoding="utf-8"))
        self.assertEqual(results["metrics"]["asr_model"], "attrappe/asr")
        wrong = [r for r in results["predictions"] if not r["is_correct"]]
        self.assertEqual(len(wrong), 1)
        self.assertIn("nein", wrong[0]["predicted_output"])
        hard = json.loads(Path(done["hard_examples_file"]).read_text(encoding="utf-8"))
        self.assertEqual(len(hard), 1)

    def test_datensatz_ohne_asr_keine_kennzahl_aber_audio(self):
        _dataset(self.root / "ds", ["Guten Morgen", "Guten Abend"])
        p = self._plugin(asr=None, dataset_path=str(self.root / "ds"))
        msgs = self._run(p.run_dataset)
        done = next(m["data"] for m in msgs if m["type"] == "complete")
        self.assertIsNone(done["accuracy"])
        self.assertIsNone(done["correct_predictions"])
        self.assertNotIn("wer", done.get("metrics", {}))
        self.assertIsNone(done["hard_examples_file"])
        self.assertEqual(len(done["audios"]), 2)
        results = json.loads(Path(done["results_file"]).read_text(encoding="utf-8"))
        self.assertNotIn("accuracy", results["metrics"])
        self.assertIsNone(results["metrics"]["asr_model"])
        self.assertTrue(all(r["is_correct"] is None for r in results["predictions"]))

    def test_meldung_ohne_lokales_asr_modell(self):
        p = tts_plugin.Plugin(TestConfig.from_dict({"plugin_config": {"asr_model": "none"}}))
        msgs = self._run(p._load_asr)
        self.assertIsNone(p.asr)
        self.assertIn("ohne WER/CER", msgs[0]["data"]["message"])

    def test_nicht_sprechbarer_satz_beendet_den_lauf_nicht(self):
        _dataset(self.root / "ds", ["Guten Morgen", "kaputt"])
        p = self._plugin(asr=FakeAsr(), dataset_path=str(self.root / "ds"))
        done = next(m["data"] for m in self._run(p.run_dataset) if m["type"] == "complete")
        self.assertEqual(done["total_samples"], 2)
        self.assertEqual(done["correct_predictions"], 1)
        self.assertEqual(len(done["audios"]), 1)

    def test_obergrenze_fuer_erzeugte_audios(self):
        _dataset(self.root / "ds", [f"Satz {chr(97 + i)}" for i in range(6)])
        p = self._plugin(asr=None, dataset_path=str(self.root / "ds"), max_samples=50,
                         plugin_config={"max_audios": 4})
        done = next(m["data"] for m in self._run(p.run_dataset) if m["type"] == "complete")
        self.assertEqual(done["total_samples"], 4)

    def test_leeres_dataset_klare_meldung(self):
        (self.root / "leer").mkdir()
        with self.assertRaises(ValueError) as ctx:
            self._plugin(dataset_path=str(self.root / "leer")).run_dataset()
        self.assertIn("Audio und Transkript", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
