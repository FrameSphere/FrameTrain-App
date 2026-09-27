"""Tests fuer das ASR-Plugin und ft_data.asr — ohne Netz, ohne Modell-Download.

Aufruf:  python3.11 plugins/speech_recognition/test_speech_recognition.py   (aus train_engine/)
"""
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # train_engine/
sys.path.insert(0, str(HERE.parents[2]))          # python/ (ft_data)

from ft_data.asr import (  # noqa: E402
    asr_mode, build_ctc_vocab, ctc_text, error_rates, evaluation_items, has_ctc_head,
    normalize_language, normalize_transcript, resolve_asr_layout,
)


def _wav(path: Path, seconds: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


def _pair(d: Path, name: str, text: str) -> None:
    _wav(d / f"{name}.wav")
    (d / f"{name}.txt").write_text(text, encoding="utf-8")


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_werkstatt_export_audio_mit_txt(self):
        for i in range(20):
            _pair(self.root, f"s_{i}", f"satz nummer {i}")
        _wav(self.root / "ohne_text.wav")
        layout = resolve_asr_layout(self.root, seed=1)
        self.assertEqual(layout.source, "audio_transcript")
        self.assertEqual(len(layout.train) + len(layout.val), 20)
        self.assertEqual(len(layout.val), 2)           # 10 % abgetrennt
        self.assertTrue(layout.val_from_train)
        self.assertEqual(layout.test, [])
        self.assertTrue(any("ohne gleichnamige .txt" in n for n in layout.notes))

    def test_splits_werden_respektiert_und_test_bleibt_unberuehrt(self):
        for split, n in (("train", 4), ("val", 2), ("test", 3)):
            for i in range(n):
                _pair(self.root / split, f"{split}_{i}", f"{split} {i}")
        layout = resolve_asr_layout(self.root)
        self.assertEqual((len(layout.train), len(layout.val), len(layout.test)), (4, 2, 3))
        self.assertFalse(layout.val_from_train)
        items, split, _ = evaluation_items(self.root)
        self.assertEqual(split, "test")
        self.assertEqual(len(items), 3)

    def test_leerer_test_ordner_zaehlt_nicht(self):
        for i in range(3):
            _pair(self.root / "train", f"t{i}", "hallo")
        (self.root / "test").mkdir()
        items, split, _ = evaluation_items(self.root)
        self.assertEqual(split, "train")
        self.assertEqual(len(items), 3)

    def test_audiofolder_metadata(self):
        _wav(self.root / "a.wav")
        _wav(self.root / "sub" / "b.wav")
        (self.root / "metadata.csv").write_text(
            "file_name,transcription\na.wav,Guten Morgen\nsub/b.wav,\"Hallo, Welt\"\nfehlt.wav,x\n",
            encoding="utf-8")
        layout = resolve_asr_layout(self.root)
        self.assertEqual(layout.source, "audiofolder")
        texts = sorted(t for _, t in layout.train + layout.val)
        self.assertEqual(texts, ["Guten Morgen", "Hallo, Welt"])

    def test_metadata_jsonl_mit_sentence(self):
        _wav(self.root / "x.wav")
        (self.root / "metadata.jsonl").write_text(
            json.dumps({"file_name": "x.wav", "sentence": "Eins zwei"}) + "\n", encoding="utf-8")
        items, _, _ = evaluation_items(self.root)
        self.assertEqual(items[0][1], "Eins zwei")

    def test_metadata_ohne_textspalte_meldet_klar(self):
        _wav(self.root / "x.wav")
        (self.root / "metadata.csv").write_text("file_name,label\nx.wav,ja\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            resolve_asr_layout(self.root)
        self.assertIn("text_column", str(ctx.exception))
        # Eigener Spaltenname ueber plugin_config
        layout = resolve_asr_layout(self.root, overrides={"text_column": "label"})
        self.assertEqual((layout.train + layout.val)[0][1], "ja")

    def test_common_voice(self):
        clips = self.root / "clips"
        for name in ("c1.mp3", "c2.mp3", "c3.mp3", "c4.mp3"):
            _wav(clips / name)
        head = "client_id\tpath\tsentence\tup_votes\n"
        (self.root / "validated.tsv").write_text(
            head + "".join(f"u\tc{i}.mp3\tSatz \"{i}\"\t2\n" for i in range(1, 5)), encoding="utf-8")
        (self.root / "test.tsv").write_text(head + "u\tc4.mp3\tSatz \"4\"\t2\n", encoding="utf-8")
        layout = resolve_asr_layout(self.root)
        self.assertEqual(layout.source, "common_voice")
        self.assertEqual(len(layout.test), 1)
        # validated.tsv ohne die Test-Clips — sonst saesse der Test im Training
        pool = {Path(p).name for p, _ in layout.train + layout.val}
        self.assertEqual(pool, {"c1.mp3", "c2.mp3", "c3.mp3"})
        self.assertEqual(layout.test[0][1], 'Satz "4"')

    def test_parquet_mit_audio_bytes(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow fehlt")
        _wav(self.root / "tmp.wav")
        data = (self.root / "tmp.wav").read_bytes()
        (self.root / "tmp.wav").unlink()
        table = pa.table({
            "audio": [{"bytes": data, "path": f"{i}.wav"} for i in range(3)],
            "text": ["eins", "zwei", "drei"],
        })
        pq.write_table(table, self.root / "train-00000.parquet")
        layout = resolve_asr_layout(self.root)
        self.assertEqual(layout.source, "parquet")
        self.assertEqual(sorted(t for _, t in layout.train + layout.val), ["drei", "eins", "zwei"])
        self.assertTrue(all(Path(p).exists() for p, _ in layout.train))

    def test_ohne_paare_klare_meldung(self):
        _wav(self.root / "nur_audio.wav")
        with self.assertRaises(ValueError) as ctx:
            resolve_asr_layout(self.root)
        self.assertIn("Audio und Transkript", str(ctx.exception))


class MetricsAndTextTest(unittest.TestCase):
    def test_normalisierung_ignoriert_satzzeichen_und_gross(self):
        self.assertEqual(normalize_transcript("Licht  einschalten."), "licht einschalten")
        self.assertEqual(normalize_transcript("Grüße, Straße!"), "grüße straße")

    def test_wer_cer(self):
        r = error_rates(["Licht an.", "heizung aus"], ["licht an", "Heizung an"])
        self.assertAlmostEqual(r["wer"], 1 / 4)
        self.assertGreater(r["cer"], 0)
        self.assertEqual(error_rates(["x"], [""]), {})   # leere Referenz zaehlt nicht
        self.assertEqual(error_rates([""], ["a b"])["wer"], 1.0)

    def test_ctc_vokabular(self):
        texts = [ctc_text("Grüße, Welt!"), ctc_text("Straße 7")]
        self.assertEqual(texts, ["grüße welt", "straße 7"])
        vocab = build_ctc_vocab(texts)
        self.assertEqual((vocab["[PAD]"], vocab["[UNK]"], vocab["|"]), (0, 1, 2))
        for c in "grüßeweltsa7":
            self.assertIn(c, vocab)
        self.assertNotIn(" ", vocab)
        self.assertEqual(len(set(vocab.values())), len(vocab))

    def test_modellart(self):
        self.assertEqual(asr_mode({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}), "seq2seq")
        self.assertEqual(asr_mode({"model_type": "moonshine"}), "seq2seq")
        self.assertEqual(asr_mode({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForPreTraining"]}), "ctc")
        self.assertEqual(asr_mode({"model_type": "hubert", "architectures": ["HubertForCTC"]}), "ctc")
        self.assertEqual(asr_mode({"model_type": "bert"}), None)
        self.assertTrue(has_ctc_head({"architectures": ["Wav2Vec2ForCTC"]}))
        self.assertFalse(has_ctc_head({"architectures": ["Wav2Vec2ForPreTraining"]}))

    def test_sprache(self):
        self.assertEqual(normalize_language("Deutsch"), "german")
        self.assertEqual(normalize_language("de"), "de")
        self.assertEqual(normalize_language("auto"), "")


class CtcLossTest(unittest.TestCase):
    def test_cpu_loss_wie_torch(self):
        import torch
        from plugins.speech_recognition.plugin import ctc_loss_on_cpu

        class Cfg:
            pad_token_id = 0
            ctc_loss_reduction = "mean"
            ctc_zero_infinity = True

        class Fake:
            config = Cfg()

            @staticmethod
            def _get_feat_extract_output_lengths(n):
                return n // 10

        torch.manual_seed(0)
        logits = torch.randn(2, 20, 6, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, 2, 1]])
        loss = ctc_loss_on_cpu(Fake(), logits, labels)
        ref = torch.nn.functional.ctc_loss(
            logits.log_softmax(-1).transpose(0, 1), torch.tensor([1, 2, 3, 4, 5, 2, 1]),
            torch.tensor([20, 20]), torch.tensor([3, 4]), blank=0, zero_infinity=True)
        self.assertAlmostEqual(loss.item(), ref.item(), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Mit attention_mask: Laenge je Aufnahme aus den Samples
        mask = torch.ones(2, 200, dtype=torch.long)
        mask[1, 150:] = 0
        self.assertTrue(torch.isfinite(ctc_loss_on_cpu(Fake(), logits.detach(), labels, mask)))


if __name__ == "__main__":
    unittest.main()
