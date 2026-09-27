"""Schnelle Tests fuer die Token-Klassifikation (ohne Modell, ohne Netz).

Aufruf (aus src-tauri/python/):  python3 -m unittest train_engine/plugins/token_classification/test_token_classification.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PY = Path(__file__).resolve().parents[3]          # src-tauri/python
sys.path.insert(0, str(_PY))
sys.path.insert(0, str(_PY / "test_engine"))

from ft_data import tokens as tk                     # noqa: E402
from ft_data.splits import evaluation_files, split_files  # noqa: E402


class FormateTest(unittest.TestCase):
    def test_jsonl_mit_string_tags(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.jsonl"
            p.write_text(json.dumps({"tokens": ["Karol", "in", "Berlin"], "ner_tags": ["B-PER", "O", "B-LOC"]}) + "\n")
            rows, spec = tk.load_files([p])
            self.assertEqual(spec["tags_column"], "ner_tags")
            self.assertEqual(rows, [{"tokens": ["Karol", "in", "Berlin"], "tags": ["B-PER", "O", "B-LOC"]}])

    def test_int_tags_mit_label_list(self):
        rows = [{"words": ["a", "b"], "labels": [0, 1]}]
        spec = tk.resolve_columns(["words", "labels"])
        out = tk.normalize_rows(rows, spec, {}, {"label_list": ["O", "B-X"]})
        self.assertEqual(out[0]["tags"], ["O", "B-X"])
        # Ohne Namen bleiben es Zahlen als Text — trainierbar, nur nicht lesbar.
        self.assertEqual(tk.normalize_rows(rows, spec)[0]["tags"], ["0", "1"])

    def test_classlabel_namen_aus_parquet(self):
        try:
            from datasets import ClassLabel, Dataset, Features, List, Value
        except ImportError:
            self.skipTest("datasets fehlt")
        feats = Features({"tokens": List(Value("string")),
                          "ner_tags": List(ClassLabel(names=["O", "B-PER", "I-PER"]))})
        ds = Dataset.from_dict({"tokens": [["Anna", "Weber", "lacht"]], "ner_tags": [[1, 2, 0]]}, features=feats)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train-00000-of-00001.parquet"
            ds.to_parquet(str(p))
            rows, _ = tk.load_files([p])
        self.assertEqual(rows[0]["tags"], ["B-PER", "I-PER", "O"])

    def test_conll_mit_docstart_und_mehreren_spalten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.conll"
            p.write_text("-DOCSTART- -X- O O\n\nEU NNP B-NP B-ORG\nrejects VBZ B-VP O\n\nPeter NNP B-NP B-PER\n")
            rows = tk.read_conll(p)
        self.assertEqual(rows, [{"tokens": ["EU", "rejects"], "tags": ["B-ORG", "O"]},
                                {"tokens": ["Peter"], "tags": ["B-PER"]}])

    def test_spans_werden_bio(self):
        toks, tags = tk.spans_to_bio("Karol Paschek wohnt in Zornheim.",
                                     [{"start": 0, "end": 13, "label": "PER"},
                                      {"start": 23, "end": 31, "label": "LOC"}])
        self.assertEqual(toks, ["Karol", "Paschek", "wohnt", "in", "Zornheim", "."])
        self.assertEqual(tags, ["B-PER", "I-PER", "O", "O", "B-LOC", "O"])

    def test_unbekannte_spalten_geben_klare_meldung(self):
        with self.assertRaises(ValueError) as ctx:
            tk.resolve_columns(["foo", "bar"])
        self.assertIn("tokens_column", str(ctx.exception))

    def test_ungleich_lange_zeilen_werden_verworfen(self):
        spec = tk.resolve_columns(["tokens", "tags"])
        self.assertEqual(tk.normalize_rows([{"tokens": ["a", "b"], "tags": ["O"]}], spec), [])


class LabelsTest(unittest.TestCase):
    def test_label_liste_o_zuerst_und_fehlendes_b_ergaenzt(self):
        labels = tk.build_label_list([{"tokens": [], "tags": ["I-LOC", "B-PER", "O"]}])
        self.assertEqual(labels[0], "O")
        self.assertIn("B-LOC", labels)
        self.assertLess(labels.index("B-LOC"), labels.index("I-LOC"))

    def test_nur_erstes_subtoken_traegt_das_label(self):
        # [CLS] Karo ##lina wohnt [SEP]
        word_ids = [None, 0, 0, 1, None]
        self.assertEqual(tk.align_labels(word_ids, [3, 0]), [-100, 3, -100, 0, -100])
        # label_all_tokens: Folge-Subtoken von B-PER wird I-PER
        self.assertEqual(tk.align_labels(word_ids, [3, 0], True, {3: 4}), [-100, 3, 4, 0, -100])

    def test_bio_schema_erkennung(self):
        self.assertTrue(tk.is_bio_scheme(["O", "B-PER", "I-PER"]))
        self.assertFalse(tk.is_bio_scheme(["NOUN", "VERB", "DET"]))


class MetrikTest(unittest.TestCase):
    def test_entitaeten_f1(self):
        true = [["B-PER", "I-PER", "O", "B-LOC"]]
        pred = [["B-PER", "I-PER", "O", "O"]]
        s = tk.score_sequences(true, pred)
        self.assertEqual(s["scheme"], "entity")
        self.assertAlmostEqual(s["precision"], 1.0)
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["accuracy"], 0.75)

    def test_pos_wird_pro_token_gezaehlt(self):
        s = tk.score_sequences([["NOUN", "VERB"]], [["NOUN", "NOUN"]])
        self.assertEqual(s["scheme"], "token")
        self.assertAlmostEqual(s["accuracy"], 0.5)


class EntitaetenTest(unittest.TestCase):
    def test_zusammenfassen_mit_zeichenpositionen(self):
        text = "Heute fliegt Karol Paschek nach Berlin."
        words = tk.split_words(text)
        tags = ["O", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]
        ents = tk.entities_from_tags(words, tags, [1, 1, 0.9, 0.7, 1, 0.8, 1], text=text)
        self.assertEqual(tk.format_entities(ents), "Karol Paschek [PER], Berlin [LOC]")
        self.assertAlmostEqual(ents[0]["score"], 0.8)
        self.assertEqual(text[ents[1]["start"]:ents[1]["end"]], "Berlin")

    def test_i_ohne_b_beginnt_neue_entitaet(self):
        ents = tk.entities_from_tags(["a", "b", "c"], ["I-ORG", "I-ORG", "B-ORG"])
        self.assertEqual([e["text"] for e in ents], ["a b", "c"])

    def test_pos_ohne_schema_einzeln(self):
        ents = tk.entities_from_tags(["Hunde", "bellen"], ["NOUN", "VERB"], bio=False)
        self.assertEqual(tk.format_entities(ents), "Hunde [NOUN], bellen [VERB]")


class SplitTest(unittest.TestCase):
    def test_ordner_und_dateinamen(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "train.conll").write_text("a O\n")
            (root / "validation.jsonl").write_text("{}\n")
            (root / "test.txt").write_text("b O\n")
            (root / "PROVENANCE.csv").write_text("x\n")
            found = split_files(root, tk.DATA_EXTS)
            self.assertEqual(sorted(found), ["test", "train", "val"])
            self.assertEqual(evaluation_files(root, tk.DATA_EXTS)[0], "test")
            sub = root / "sub"
            (sub / "train").mkdir(parents=True)
            (sub / "train" / "a.jsonl").write_text("{}\n")
            self.assertEqual(list(split_files(sub, tk.DATA_EXTS)), ["train"])


class ModellServerTest(unittest.TestCase):
    def test_token_modalitaet(self):
        import model_server
        self.assertEqual(model_server.detect_modality(
            {"architectures": ["BertForTokenClassification"], "model_type": "bert"}), "token")
        self.assertEqual(model_server.INPUT_KIND["token"], "text")
        # Sequenzklassifikation bleibt, wie sie war
        self.assertEqual(model_server.detect_modality(
            {"architectures": ["BertForSequenceClassification"], "model_type": "bert"}), "text")


if __name__ == "__main__":
    unittest.main()
