"""Tests fuer das TTS-Plugin und ft_data.tts — ohne Netz, ohne Modell-Download.

Datenaufbereitung, Kennzahl und ein Mini-Trainingslauf. Der Lauf nutzt ein
winziges, zufaellig initialisiertes SpeechT5 (wenige tausend Parameter), das
der Test selbst anlegt: geprueft wird der Weg durch das Plugin (laden, filtern,
Batch bauen, Schritte rechnen, exportieren, wieder laden, Audio erzeugen),
nicht die Klangqualitaet.

Aufruf:  python3.11 plugins/text_to_speech/test_text_to_speech.py   (aus train_engine/)
"""
import functools
import io
import json
import math
import struct
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # train_engine/
sys.path.insert(0, str(HERE.parents[2]))          # python/ (ft_data)

from ft_data import tts  # noqa: E402

CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'-")


def _wav(path: Path, seconds: float = 0.6, freq: float = 220.0) -> None:
    """Sinuston als 16-kHz-WAV — genug Signal fuer ein Mel-Spektrogramm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(16000 * seconds)
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / 16000))) for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)


def _pair(d: Path, name: str, text: str, seconds: float = 0.6) -> None:
    _wav(d / f"{name}.wav", seconds)
    (d / f"{name}.txt").write_text(text, encoding="utf-8")


class FakeTokenizer:
    """Zeichen-Tokenizer wie SpeechT5: ein Token je Zeichen, </s> am Ende."""
    pad_token_id = 1

    def __init__(self):
        self.vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "▁": 4}
        for c in CHARS:
            self.vocab[c] = len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def __call__(self, text, **_):
        ids = [self.vocab.get("▁" if c == " " else c, 3) for c in text] + [2]
        return {"input_ids": ids}


# ── Modellart ────────────────────────────────────────────────────────────────

class KindTest(unittest.TestCase):
    def test_speecht5_tts_und_vits(self):
        self.assertEqual(tts.tts_kind({"model_type": "speecht5", "architectures": ["SpeechT5ForTextToSpeech"]}), "speecht5")
        self.assertEqual(tts.tts_kind({"model_type": "speecht5"}), "speecht5")
        self.assertEqual(tts.tts_kind({"model_type": "vits", "architectures": ["VitsModel"]}), "vits")
        self.assertEqual(tts.tts_kind({"model_type": "vits"}), "vits")

    def test_andere_speecht5_aufgaben_und_fremde_modelle(self):
        # Gleicher model_type, andere Aufgabe: Spracherkennung und Stimmwandlung.
        self.assertIsNone(tts.tts_kind({"model_type": "speecht5", "architectures": ["SpeechT5ForSpeechToText"]}))
        self.assertIsNone(tts.tts_kind({"model_type": "speecht5", "architectures": ["SpeechT5ForSpeechToSpeech"]}))
        self.assertIsNone(tts.tts_kind({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}))
        self.assertIsNone(tts.tts_kind({"model_type": "bert"}))
        self.assertIsNone(tts.tts_kind({}))

    def test_vits_ist_nur_inferenz(self):
        self.assertIn("vits", tts.TTS_INFERENCE_ONLY)
        self.assertNotIn("vits", tts.TTS_TRAINABLE)
        self.assertIn("nicht trainieren", tts.VITS_NOT_TRAINABLE)


# ── Datenaufbereitung ────────────────────────────────────────────────────────

class TextTest(unittest.TestCase):
    known = set(CHARS)

    def test_umlaute_und_typografie_werden_umgeschrieben(self):
        text, unknown = tts.prepare_text("Grüße aus Köln – „schön“!", self.known)
        self.assertEqual(text, "Gruesse aus Koeln - schoen!")
        self.assertEqual(unknown, set())

    def test_akzente_werden_abgetrennt(self):
        text, unknown = tts.prepare_text("Café à la carte", self.known)
        self.assertEqual(text, "Cafe a la carte")
        self.assertEqual(unknown, set())

    def test_unbekannte_zeichen_werden_gemeldet_nicht_verschluckt(self):
        text, unknown = tts.prepare_text("Zimmer 42 kostet 80 €", self.known)
        self.assertEqual(text, "Zimmer kostet")
        self.assertEqual(unknown, {"4", "2", "8", "0", "€"})

    def test_ohne_vokabular_bleiben_buchstaben_erhalten(self):
        # VITS bringt sein eigenes Vokabular mit — Umlaute nicht anfassen.
        text, unknown = tts.prepare_text("Grüße  aus\tKöln", None)
        self.assertEqual(text, "Grüße aus Köln")
        self.assertEqual(unknown, set())

    def test_leerer_text(self):
        self.assertEqual(tts.prepare_text("", self.known), ("", set()))
        self.assertEqual(tts.prepare_text(None, self.known), ("", set()))

    def test_lange_texte_werden_an_satzgrenzen_geteilt(self):
        text = "Das ist der erste Satz mit einigen Woertern darin. " * 12
        chunks = tts.split_text(text, max_chars=120)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertTrue(all(c.endswith(".") for c in chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

    def test_satz_ohne_punkt_wird_an_woertern_geteilt(self):
        chunks = tts.split_text("wort " * 100, max_chars=60)
        self.assertTrue(all(len(c) <= 60 for c in chunks))
        self.assertEqual(sum(c.count("wort") for c in chunks), 100)

    def test_kurzer_text_bleibt_ganz(self):
        self.assertEqual(tts.split_text("Hallo Welt."), ["Hallo Welt."])
        self.assertEqual(tts.split_text("   "), [])

    def test_dateiname(self):
        self.assertEqual(tts.safe_filename("Grüße, Welt! 123"), "gru_e_welt_123")
        self.assertEqual(tts.safe_filename("???"), "audio")
        self.assertLessEqual(len(tts.safe_filename("a" * 200)), 40)


class SpeakerTest(unittest.TestCase):
    def test_standardvektor_ist_ein_echter_xvector(self):
        import numpy as np
        vec = tts.default_speaker()
        self.assertEqual(vec.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=3)
        # Kein Null-Vektor: damit bricht SpeechT5 jeden Satz nach gut einer Sekunde ab.
        self.assertGreater(float(np.abs(vec).max()), 0.01)

    def test_mittel_ueber_aufnahmen_hat_laenge_eins(self):
        import numpy as np
        a = np.zeros(512, dtype=np.float32); a[0] = 1.0
        b = np.zeros(512, dtype=np.float32); b[1] = 1.0
        m = tts.mean_speaker([a, b])
        self.assertAlmostEqual(float(np.linalg.norm(m)), 1.0, places=5)
        self.assertAlmostEqual(float(m[0]), float(m[1]), places=6)
        self.assertEqual(tts.mean_speaker([]).shape, (512,))

    def test_sprecher_datei_wird_geprueft(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "stimme.npy"
            np.save(good, np.ones(512, dtype=np.float32))
            self.assertEqual(tts.load_speaker_file(good).shape, (512,))
            bad = Path(d) / "falsch.npy"
            np.save(bad, np.ones(100, dtype=np.float32))
            with self.assertRaises(ValueError) as ctx:
                tts.load_speaker_file(bad)
            self.assertIn("512", str(ctx.exception))
            with self.assertRaises(FileNotFoundError):
                tts.load_speaker_file(Path(d) / "fehlt.npy")

    def test_ohne_speechbrain_ausweichen_mit_meldung(self):
        warnings = []
        original = tts.speechbrain_available
        tts.speechbrain_available = lambda: False
        try:
            self.assertIsNone(tts.make_speaker_encoder("auto", warn=warnings.append))
            self.assertEqual(len(warnings), 1)
            self.assertIn("speechbrain", warnings[0])
            self.assertIn("Standard-Sprechervektor", warnings[0])
            # Ausdruecklich verlangt: nicht still ausweichen.
            with self.assertRaises(ImportError) as ctx:
                tts.make_speaker_encoder("xvector")
            self.assertIn("pip install speechbrain", str(ctx.exception))
            # Ausdruecklich abgewaehlt: keine Meldung.
            warnings.clear()
            self.assertIsNone(tts.make_speaker_encoder("default", warn=warnings.append))
            self.assertEqual(warnings, [])
        finally:
            tts.speechbrain_available = original


class CollateTest(unittest.TestCase):
    def test_batch_formen_und_auffuellen(self):
        import numpy as np
        feats = [
            {"input_ids": [5, 6, 7, 2], "labels": np.ones((11, 80), dtype=np.float32), "speaker": np.zeros(512, dtype=np.float32)},
            {"input_ids": [5, 2], "labels": np.ones((6, 80), dtype=np.float32) * 2, "speaker": np.ones(512, dtype=np.float32)},
        ]
        batch = tts.collate_speecht5(feats, pad_token_id=1, reduction_factor=2)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["input_ids"][1].tolist(), [5, 2, 1, 1])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        # 11 Frames -> 10 (Vielfaches des reduction_factor), 6 bleiben 6
        self.assertEqual(tuple(batch["labels"].shape), (2, 10, 80))
        self.assertEqual(float(batch["labels"][0, 9, 0]), 1.0)
        self.assertEqual(float(batch["labels"][1, 5, 0]), 2.0)
        self.assertEqual(float(batch["labels"][1, 6, 0]), -100.0)   # aufgefuellt, zaehlt nicht im Loss
        self.assertEqual(tuple(batch["speaker_embeddings"].shape), (2, 512))

    def test_sehr_kurze_aufnahme_bleibt_gueltig(self):
        import numpy as np
        feats = [{"input_ids": [5, 2], "labels": np.ones((1, 80), dtype=np.float32), "speaker": np.zeros(512, dtype=np.float32)}]
        batch = tts.collate_speecht5(feats, pad_token_id=1, reduction_factor=2)
        self.assertEqual(batch["labels"].shape[1] % 2, 0)
        self.assertGreaterEqual(batch["labels"].shape[1], 2)


class FilterTest(unittest.TestCase):
    def test_lange_aufnahmen_werden_ausgelassen_nicht_gekappt(self):
        from plugins.text_to_speech.plugin import filter_items
        durations = {"kurz.wav": 0.2, "gut.wav": 3.0, "lang.wav": 40.0, "leer.wav": 3.0, "zahl.wav": 3.0, "roman.wav": 3.0}
        items = [("kurz.wav", "Hallo"), ("gut.wav", "Grüße aus Köln"), ("lang.wav", "Ein langer Text"),
                 ("leer.wav", "123"), ("zahl.wav", "Zimmer 7 ist frei"), ("roman.wav", "wort " * 200)]
        kept, stats = filter_items(items, FakeTokenizer(), set(CHARS), 0.5, 15.0, 100,
                                   duration=lambda p: durations[p])
        self.assertEqual([p for p, _, _ in kept], ["gut.wav", "zahl.wav"])
        self.assertEqual(kept[0][1], "Gruesse aus Koeln")
        self.assertEqual(kept[1][1], "Zimmer ist frei")
        self.assertEqual(len(kept[0][2]), len("Gruesse aus Koeln") + 1)       # + </s>
        self.assertEqual((stats["too_short"], stats["too_long"], stats["no_text"], stats["text_too_long"]), (1, 1, 1, 1))
        self.assertEqual(stats["unknown_chars"], {"1", "2", "3", "7"})

    def test_unbekannte_dauer_wird_nicht_gefiltert(self):
        from plugins.text_to_speech.plugin import filter_items
        kept, _ = filter_items([("x.mp3", "Hallo")], FakeTokenizer(), set(CHARS), 0.5, 15.0, 100,
                               duration=lambda p: None)
        self.assertEqual(len(kept), 1)


class AudioTest(unittest.TestCase):
    def test_wav_ist_16_bit_pcm_und_abspielbar(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            wave_in = np.sin(np.linspace(0, 200, 8000)).astype(np.float32) * 1.5   # uebersteuert -> wird begrenzt
            path = tts.write_wav(Path(d) / "unter" / "probe.wav", wave_in, 16000)
            with wave.open(str(path), "rb") as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()),
                                 (1, 2, 16000, 8000))
            self.assertEqual(path.read_bytes()[:4], b"RIFF")

    def test_griffin_lim_liefert_wellenform(self):
        import numpy as np
        try:
            import librosa  # noqa: F401
        except ImportError:
            self.skipTest("librosa fehlt")
        mel = np.log10(np.clip(np.random.RandomState(0).rand(40, 80).astype(np.float32), 1e-5, None))
        out = tts.griffin_lim(mel, 16000, n_iter=4)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(len(out), 39 * 256 - 1024)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)
        self.assertTrue(np.isfinite(out).all())


# ── Kennzahl ─────────────────────────────────────────────────────────────────

class MetricTest(unittest.TestCase):
    def test_perfekt_verstanden_ist_null(self):
        r = tts.intelligibility(["the library opens at nine."], ["The library opens at nine"])
        self.assertEqual(r["wer"], 0.0)
        self.assertEqual(r["cer"], 0.0)

    def test_ein_falsches_wort(self):
        r = tts.intelligibility(["the library opens at night"], ["The library opens at nine."])
        self.assertAlmostEqual(r["wer"], 1 / 5)
        self.assertGreater(r["cer"], 0)

    def test_umschrift_zaehlt_nicht_als_fehler(self):
        # SpeechT5 bekommt "Gruesse" zu sprechen, das ASR-Modell schreibt "Grüße".
        self.assertEqual(tts.intelligibility(["Grüße aus Köln"], ["Gruesse aus Koeln"])["wer"], 0.0)
        self.assertEqual(tts.intelligibility(["Gruesse aus Koeln"], ["Grüße aus Köln"])["wer"], 0.0)

    def test_nichts_gehoert_ist_volle_fehlerrate(self):
        self.assertEqual(tts.intelligibility([""], ["Guten Morgen"])["wer"], 1.0)

    def test_korpus_statt_mittel_der_einzelwerte(self):
        r = tts.intelligibility(["a b c d", "x"], ["a b c d", "y"])
        self.assertAlmostEqual(r["wer"], 1 / 5)      # 1 Fehler auf 5 Woerter, nicht (0 + 1) / 2

    def test_ohne_referenz_keine_kennzahl(self):
        self.assertEqual(tts.intelligibility([], []), {})

    def test_asr_suche_ohne_download(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(tts.find_local_asr("none"))
            self.assertIsNone(tts.find_local_asr(str(root / "gibt_es_nicht"), candidates=()))
            halb = root / "halb"
            halb.mkdir()
            (halb / "config.json").write_text("{}")
            self.assertIsNone(tts.find_local_asr(str(halb)))            # config ohne Gewichte
            ganz = root / "ganz"
            ganz.mkdir()
            (ganz / "config.json").write_text("{}")
            (ganz / "model.safetensors").write_bytes(b"x")
            self.assertEqual(tts.find_local_asr(str(ganz)), str(ganz))
            # Nichts im Cache -> None statt Download
            self.assertIsNone(tts.find_local_asr("auto", candidates=("frametrain-test/gibt-es-nicht",)))


# ── Mini-Trainingslauf ───────────────────────────────────────────────────────

def _tiny_speecht5(model_dir: Path) -> None:
    """Winziges SpeechT5 mit zufaelligen Gewichten samt Tokenizer und Feature-Extractor."""
    import sentencepiece as spm
    from transformers import (
        SpeechT5Config, SpeechT5FeatureExtractor, SpeechT5ForTextToSpeech, SpeechT5Processor,
        SpeechT5Tokenizer,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    corpus = model_dir / "corpus.txt"
    corpus.write_text("\n".join(
        ["the quick brown fox jumps over the lazy dog", "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
         "hello, world! how are you? it's fine - really."] * 40), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(model_dir / "spm_char"), model_type="char",
        vocab_size=70, character_coverage=1.0, hard_vocab_limit=False,
        bos_id=0, pad_id=1, eos_id=2, unk_id=3, minloglevel=2)
    corpus.unlink()
    (model_dir / "spm_char.vocab").unlink()
    tokenizer = SpeechT5Tokenizer(str(model_dir / "spm_char.model"))
    config = SpeechT5Config(
        vocab_size=len(tokenizer), hidden_size=24,
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=32,
        speech_decoder_prenet_layers=1, speech_decoder_prenet_units=16,
        speech_decoder_postnet_layers=1, speech_decoder_postnet_units=16,
        speech_decoder_postnet_kernel=3, positional_dropout=0.0, hidden_dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, speech_decoder_prenet_dropout=0.0,
        speech_decoder_postnet_dropout=0.0, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
        max_text_positions=120, max_speech_positions=400,
        num_mel_bins=80, speaker_embedding_dim=512, reduction_factor=2,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
    )
    model = SpeechT5ForTextToSpeech(config)
    model.save_pretrained(str(model_dir))
    SpeechT5Processor(SpeechT5FeatureExtractor(), tokenizer).save_pretrained(str(model_dir))


class MiniTrainingTest(unittest.TestCase):
    """Ein kompletter Lauf durch das Plugin mit einem Winzmodell (CPU, Sekunden)."""

    @classmethod
    def setUpClass(cls):
        try:
            import datasets  # noqa: F401
            import librosa  # noqa: F401
            import sentencepiece  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"Paket fehlt: {exc}")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.model_dir = cls.root / "tiny_speecht5"
        _tiny_speecht5(cls.model_dir)
        cls.data = cls.root / "data"
        texts = ["the quick brown fox", "hello, world!", "how are you?", "pack my box", "five dozen jugs",
                 "it's fine", "over the lazy dog", "really fine"]
        for i, t in enumerate(texts):
            _pair(cls.data / "train", f"t{i}", t, seconds=0.5 + 0.05 * i)
        _pair(cls.data / "val", "v0", "the lazy dog")
        _pair(cls.data / "val", "v1", "hello fox")
        _pair(cls.data / "test", "x0", "quick jugs")
        _pair(cls.data / "train", "zu_lang", "the fox", seconds=3.0)       # > max_seconds
        _pair(cls.data / "train", "ziffern", "box 7 9")                    # Ziffern kennt das Vokabular nicht

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, name: str, **plugin_config):
        from core.config import TrainingConfig
        from plugins.text_to_speech.plugin import Plugin

        out = self.root / name
        config = TrainingConfig.from_dict({
            "model_path": str(self.model_dir), "dataset_path": str(self.data),
            "output_path": str(out / "model"), "checkpoint_dir": str(out / "ckpt"),
            "task_type": "text_to_speech", "epochs": 1, "batch_size": 4, "max_steps": 3,
            "learning_rate": 1e-3, "warmup_ratio": 0.0, "logging_steps": 1,
            "eval_strategy": "steps", "eval_steps": 2, "save_steps": 500,
            # Der Test prueft den Weg, nicht das Geraet: CPU haelt ihn von laufenden Trainings fern.
            "plugin_config": {"vocoder": "griffin_lim", "max_seconds": 2.0, "speaker_embedding": "default",
                              "language": "en", **plugin_config},
        })
        plugin = Plugin(config)
        log = io.StringIO()
        import transformers
        original = transformers.TrainingArguments.__init__

        # wraps: build_training_arguments liest die Signatur, um unbekannte
        # Argumente zu verwerfen — ohne sie fielen alle weg.
        @functools.wraps(original)
        def cpu_only(self_, *a, **kw):
            kw["use_cpu"] = True
            original(self_, *a, **kw)

        transformers.TrainingArguments.__init__ = cpu_only
        try:
            with redirect_stdout(log):
                plugin.setup()
                plugin.load_data()
                plugin.build_model()
                plugin.train()
                metrics = plugin.validate()
                path = plugin.export()
        finally:
            transformers.TrainingArguments.__init__ = original
        messages = []
        for line in log.getvalue().splitlines():
            try:
                messages.append(json.loads(line))
            except ValueError:
                pass
        return plugin, metrics, Path(path), messages

    def test_lauf_export_und_wiedergabe(self):
        import numpy as np
        import torch

        plugin, metrics, out, messages = self._run("lauf")

        # Daten: 8 gute Trainingspaare + "ziffern" (Text bleibt "box"), "zu_lang" faellt weg.
        self.assertEqual(len(plugin.train_dataset), 9)
        self.assertEqual(len(plugin.eval_dataset), 2)
        self.assertNotIn(str(self.data / "train" / "zu_lang.wav"), list(plugin.train_dataset["path"]))
        self.assertFalse(any("test" in Path(p).parts for p in plugin.train_dataset["path"]))
        warnings = " ".join(m["data"].get("message", "") for m in messages if m["type"] == "warning")
        self.assertIn("laenger als 2 s ausgelassen", warnings)
        self.assertIn("Zahlen bitte ausschreiben", warnings)
        self.assertIn("Minuten Trainingsmaterial", warnings)

        # Training: Fortschritt gemeldet, Loss endlich, Val-Loss gemessen.
        progress = [m["data"] for m in messages if m["type"] == "progress"]
        self.assertTrue(progress)
        self.assertEqual(max(p["step"] for p in progress), 3)
        self.assertTrue(all(math.isfinite(p["train_loss"]) for p in progress))
        self.assertTrue(any(p["val_loss"] is not None for p in progress))
        self.assertEqual(metrics["total_steps"], 3)
        self.assertTrue(math.isfinite(metrics["final_val_loss"]) and metrics["final_val_loss"] > 0)
        self.assertEqual(metrics["speaker_embedding"], "default")
        self.assertEqual(metrics["language"], "en")
        for key in ("accuracy", "f1", "precision", "recall"):
            self.assertNotIn(key, metrics)

        # Export: alles da, was Test, Labor und Hosting brauchen.
        for name in ("config.json", "model.safetensors", "spm_char.model", tts.SPEAKER_FILE, tts.TTS_META_FILE):
            self.assertTrue((out / name).is_file(), name)
        meta = json.loads((out / tts.TTS_META_FILE).read_text(encoding="utf-8"))
        self.assertEqual((meta["kind"], meta["vocoder"], meta["language"]), ("speecht5", "griffin_lim", "en"))
        self.assertEqual(np.load(out / tts.SPEAKER_FILE).shape, (512,))
        self.assertTrue((out / "samples" / "sample.wav").is_file())

        # Wieder laden und sprechen: eine abspielbare Wellenform.
        engine = tts.load_tts(out, torch.device("cpu"), {"vocoder": "griffin_lim"})
        self.assertEqual(engine.kind, "speecht5")
        self.assertEqual(engine.speaker_source, "Stimme aus dem Training")
        self.assertTrue(any("Griffin-Lim" in n for n in engine.notes))
        audio, sr = engine.synthesize("the quick fox 7")
        self.assertEqual(sr, 16000)
        # Laenge nicht pruefen: ein zufaellig initialisiertes Modell setzt das Stopp-Signal sofort.
        self.assertGreater(len(audio), 0)
        self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(engine.last_dropped, {"7"})
        with self.assertRaises(ValueError):
            engine.synthesize("123")            # nichts Sprechbares
        with self.assertRaises(ValueError):
            engine.synthesize("   ")

    def test_training_veraendert_die_gewichte(self):
        import torch
        from transformers import SpeechT5ForTextToSpeech

        _, _, out, _ = self._run("gewichte")
        before = SpeechT5ForTextToSpeech.from_pretrained(str(self.model_dir)).state_dict()
        after = SpeechT5ForTextToSpeech.from_pretrained(str(out)).state_dict()
        changed = [k for k in before if before[k].dtype.is_floating_point and not torch.equal(before[k], after[k])]
        self.assertGreater(len(changed), 10)

    def test_sprecher_aus_datei(self):
        import numpy as np
        voice = self.root / "stimme.npy"
        vec = np.zeros(512, dtype=np.float32)
        vec[3] = 1.0
        np.save(voice, vec)
        plugin, metrics, out, _ = self._run("datei", speaker_embedding=str(voice))
        self.assertEqual(metrics["speaker_embedding"], "file")
        self.assertTrue(np.allclose(np.load(out / tts.SPEAKER_FILE), vec))

    def test_vits_wird_mit_begruendung_abgelehnt(self):
        from core.config import TrainingConfig
        from plugins.text_to_speech.plugin import Plugin

        d = self.root / "vits"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"model_type": "vits", "architectures": ["VitsModel"]}))
        plugin = Plugin(TrainingConfig.from_dict({"model_path": str(d), "task_type": "text_to_speech"}))
        with self.assertRaises(ValueError) as ctx:
            plugin.setup()
        self.assertIn("nicht unterstützt", str(ctx.exception))     # -> "Konfigurationsfehler" in der App
        self.assertIn("nur die Inferenz", str(ctx.exception))

    def test_fremdes_modell_wird_abgelehnt(self):
        from core.config import TrainingConfig
        from plugins.text_to_speech.plugin import Plugin

        d = self.root / "bert"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
        with self.assertRaises(ValueError) as ctx:
            Plugin(TrainingConfig.from_dict({"model_path": str(d)})).setup()
        self.assertIn("speecht5_tts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
