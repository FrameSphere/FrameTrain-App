"""Tests fuer ft_data.seq2seq.

Aufruf:  python3 -m ft_data.test_seq2seq   (aus src-tauri/python/)
"""
import tempfile
import unittest
from pathlib import Path

from ft_data.seq2seq import batch_texts, file_split_name, load_spec, resolve_spec, row_texts, save_spec


class Seq2SeqSpecTest(unittest.TestCase):
    def test_uebersetzungs_dict(self):
        row = {"id": 1, "translation": {"de": "Hallo", "en": "Hello"}}
        spec = resolve_spec(list(row), row, {})
        self.assertEqual(row_texts(row, spec), ("Hallo", "Hello"))
        rev = resolve_spec(list(row), row, {"source_lang": "en", "target_lang": "de"})
        self.assertEqual(row_texts(row, rev), ("Hello", "Hallo"))

    def test_listen_ziel_wird_text(self):
        row = {"abstract": "Text", "keyphrases": ["a", "b"]}
        spec = resolve_spec(list(row), row, {})
        self.assertEqual(row_texts(row, spec), ("Text", "a; b"))

    def test_eigene_spalten_und_fehlermeldung(self):
        row = {"artikel": "x", "kurz": "y"}
        with self.assertRaises(ValueError):
            resolve_spec(list(row), row, {})
        spec = resolve_spec(list(row), row, {"source_column": "artikel", "target_column": "kurz"})
        self.assertEqual(batch_texts({"artikel": ["x", None], "kurz": ["y", "z"]}, spec), (["x", ""], ["y", "z"]))

    def test_dateinamen_ergeben_splits(self):
        self.assertEqual(file_split_name("train-00000-of-00001"), "train")
        self.assertEqual(file_split_name("validation"), "val")
        self.assertEqual(file_split_name("test_0"), "test")
        self.assertIsNone(file_split_name("train_labels"))
        self.assertIsNone(file_split_name("daten"))

    def test_spec_wird_neben_dem_modell_gespeichert(self):
        with tempfile.TemporaryDirectory() as d:
            save_spec(Path(d), {"source_column": "a", "target_column": "b"}, "summarize: ")
            self.assertEqual(load_spec(Path(d))["task_prefix"], "summarize: ")
            self.assertEqual(load_spec(Path(d) / "fehlt"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
