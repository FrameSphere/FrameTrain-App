"""Tests fuer causal_lm: volles Fine-Tuning mit Speichersperre, Rohtext, Perplexitaet.

Aufruf (aus src-tauri/python/):
  python3 -m unittest train_engine/plugins/causal_lm/test_full_finetune_rawtext.py
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _tiny_models as T  # noqa: E402
from ft_data import llm as D  # noqa: E402

GB = 1024 ** 3


class RohtextTest(unittest.TestCase):
    def test_textspalte_per_override(self):
        self.assertEqual(D.detect_format(["zeile", "id"], {"text_column": "zeile"}), ("text", ("zeile",)))
        with self.assertRaises(ValueError):
            D.detect_format(["a"], {"text_column": "fehlt"})

    def test_eine_spalte_ist_rohtext(self):
        self.assertEqual(D.detect_format(["absatz"]), ("text", ("absatz",)))

    def test_bestehende_erkennung_bleibt(self):
        self.assertEqual(D.detect_format(["prompt", "completion"])[0], "pair")
        self.assertEqual(D.detect_format(["prompt", "chosen", "rejected"])[0], "preference")
        self.assertEqual(D.detect_format(["text"]), ("text", ("text",)))
        self.assertEqual(D.detect_format(["messages"])[0], "messages")
        with self.assertRaises(ValueError):
            D.detect_format(["a", "b"])

    def test_csv_mit_einer_spalte_wird_geladen(self):
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "train.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["zeile"])
                for i in range(12):
                    w.writerow([f"satz nummer {i}"])
            data = D.load_examples(d)
        self.assertEqual(data.kind, "text")
        self.assertEqual(len(data.train) + len(data.val), 12)


class SpeichersperreTest(unittest.TestCase):
    def test_kleines_modell_frei(self):
        self.assertIsNone(D.full_finetune_block(135e6, 16 * GB))     # SmolLM-135M
        self.assertIsNone(D.full_finetune_block(124e6, 8 * GB))      # GPT-2

    def test_grosses_modell_gesperrt_mit_erklaerung(self):
        msg = D.full_finetune_block(7e9, 32 * GB)
        self.assertIsNotNone(msg)
        self.assertIn("LoRA", msg)
        self.assertIn("GB", msg)

    def test_grenze_und_unbekannt(self):
        n = 1e9
        need = D.full_finetune_bytes(n)
        self.assertIsNone(D.full_finetune_block(n, need / D.MEMORY_BUDGET + 1))
        self.assertIsNotNone(D.full_finetune_block(n, need / D.MEMORY_BUDGET - GB))
        self.assertIsNone(D.full_finetune_block(0, 8 * GB))
        self.assertIsNone(D.full_finetune_block(7e9, None))

    def test_engine_sperrt_vor_dem_laden(self):
        """Nur config.json eines 70B-Llama: setup() bricht ab, ohne Gewichte zu brauchen."""
        mod = T.load_plugin("train", "causal_lm")
        from core.config import TrainingConfig  # nach load_plugin: core der Train-Engine
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({
                "model_type": "llama", "hidden_size": 8192, "num_hidden_layers": 80,
                "intermediate_size": 28672, "vocab_size": 128256}))
            plugin = mod.Plugin(TrainingConfig.from_dict({
                "model_path": d, "use_lora": False, "plugin_config": {"backend": "torch"}}))
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as ctx:
                plugin.setup()
            self.assertIn("gesperrt", str(ctx.exception))
            # Mit LoRA greift die Sperre nicht (Pruefung laeuft dann gar nicht).
            plugin = mod.Plugin(TrainingConfig.from_dict({
                "model_path": d, "use_lora": True, "plugin_config": {"backend": "torch"}}))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    plugin.setup()
                except ValueError as e:
                    self.assertNotIn("gesperrt", str(e))
                except Exception:
                    pass   # kein Tokenizer im Ordner — hier geht es nur um die Sperre

    def test_parameter_schaetzung_gpt2_namen(self):
        mod = T.load_plugin("train", "causal_lm")
        from core.config import TrainingConfig  # nach load_plugin: core der Train-Engine
        p = mod.Plugin(TrainingConfig.from_dict({"model_path": "/nicht/da"}))
        p.model_cfg = {"n_embd": 768, "n_layer": 12, "vocab_size": 50257}
        self.assertTrue(80e6 < p._estimate_params() < 200e6)


class MiniTrainingTest(unittest.TestCase):
    def test_volles_finetuning_auf_rohtext_mit_perplexitaet(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = T.gpt2_dir(d / "gpt2")
            (d / "data").mkdir()
            (d / "data" / "korpus.txt").write_text("\n\n".join(
                f"anna wohnt in berlin und die stadt ist schoen {i % 3}" for i in range(40)))
            metrics, out, msgs = T.run_training("causal_lm", T.base_cfg(
                model, d / "data", d / "out", "causal_lm", use_lora=False, epochs=3, learning_rate=5e-3,
                plugin_config={"backend": "torch", "eval_samples": 0}))
            self.assertFalse(metrics["lora"])
            self.assertEqual(metrics["data_kind"], "text")
            self.assertEqual(metrics["trainable_params"], metrics["total_params"])
            self.assertIn("baseline_perplexity", metrics)
            self.assertIn("perplexity", metrics)
            self.assertLess(metrics["perplexity"], metrics["baseline_perplexity"])
            texts = [m["data"].get("message", "") for m in msgs if m["type"] == "status"]
            self.assertTrue(any("Perplexitaet: vorher" in t for t in texts))
            # Vorher-Wert darf nicht als Val-Loss der ersten Epoche erscheinen.
            first = next(m["data"] for m in msgs if m["type"] == "progress")
            self.assertIsNone(first["val_loss"])
            self.assertTrue((Path(out) / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
