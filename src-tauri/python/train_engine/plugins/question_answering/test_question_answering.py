"""Tests fuer extraktives Frage-Antwort: Datenformate, Fenster, SQuAD-Kennzahlen, Mini-Training.

Aufruf (aus src-tauri/python/):
  python3 -m unittest train_engine/plugins/question_answering/test_question_answering.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _tiny_models as T  # noqa: E402
from ft_data import qa as Q  # noqa: E402


def rows(*r):
    return Q.normalize_rows(list(r))


class DatenformatTest(unittest.TestCase):
    SQUAD = {"version": "v2.0", "data": [{"title": "t", "paragraphs": [{
        "context": "Anna wohnt in Berlin.",
        "qas": [{"id": "a", "question": "Wo wohnt Anna?", "answers": [{"text": "Berlin", "answer_start": 14}]},
                {"id": "b", "question": "Wie alt ist Anna?", "answers": [], "is_impossible": True}]}]}]}

    def test_squad_json_wird_flach(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "train.json"
            f.write_text(json.dumps(self.SQUAD))
            out, st = Q.normalize_rows(Q.load_rows(f))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["answers"], {"text": ["Berlin"], "answer_start": [14]})
        self.assertEqual(out[1]["answers"], {"text": [], "answer_start": []})
        self.assertEqual(st["no_answer"], 1)

    def test_flach_ohne_start_sucht_im_kontext(self):
        out, st = rows({"context": "Ben lebt in Paris.", "question": "Wo?", "answer": "Paris"})
        self.assertEqual(out[0]["answers"]["answer_start"], [12])
        self.assertEqual(st["start_searched"], 1)

    def test_falscher_start_wird_korrigiert_und_grossschreibung(self):
        out, _ = rows({"context": "Ben lebt in Paris.", "question": "Wo?", "answer": "paris", "answer_start": 3})
        self.assertEqual(out[0]["answers"], {"text": ["Paris"], "answer_start": [12]})

    def test_naechstes_vorkommen_zum_hinweis(self):
        ctx = "Rom ist alt. Wien ist schoen. Rom ist gross."
        self.assertEqual(Q.find_answer(ctx, "Rom", 30), 30)
        self.assertEqual(Q.find_answer(ctx, "Rom", 28), 30)
        self.assertEqual(Q.find_answer(ctx, "Rom"), 0)

    def test_hf_answers_dict_und_liste(self):
        a, _ = rows({"context": "x in Rom", "question": "q", "answers": {"text": ["Rom"], "answer_start": [5]}})
        b, _ = rows({"context": "x in Rom", "question": "q", "answers": json.dumps([{"text": "Rom", "answer_start": 5}])})
        c, _ = rows({"context": "x in Rom", "question": "q", "answers": ["Rom"]})
        for r in (a, b, c):
            self.assertEqual(r[0]["answers"], {"text": ["Rom"], "answer_start": [5]})

    def test_nicht_gefunden_wird_verworfen_leer_ist_keine_antwort(self):
        out, st = rows({"context": "x", "question": "q", "answer": "fehlt"},
                       {"context": "y", "question": "q", "answer": ""})
        self.assertEqual(st["not_found"], 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["answers"]["text"], [])

    def test_eigene_spalten_und_hinweis(self):
        out, _ = Q.normalize_rows([{"absatz_x": "A in Rom", "f": "wo", "loesung": "Rom"}],
                                  {"context_column": "absatz_x", "question_column": "f", "answer_column": "loesung"})
        self.assertEqual(out[0]["answers"]["text"], ["Rom"])
        with self.assertRaises(ValueError) as ctx:
            Q.resolve_columns(["foo", "bar"])
        self.assertIn("context_column", str(ctx.exception))

    def test_ids_werden_eindeutig(self):
        out, _ = rows({"id": 1, "context": "a Rom", "question": "q", "answer": "Rom"},
                      {"id": 1, "context": "b Rom", "question": "q", "answer": "Rom"})
        self.assertEqual(len({r["id"] for r in out}), 2)

    def test_einzeleingabe(self):
        self.assertEqual(Q.parse_single_input("Kontext", "Frage?"), ("Kontext", "Frage?"))
        self.assertEqual(Q.parse_single_input("Frage? ||| Kontext"), ("Kontext", "Frage?"))
        self.assertEqual(Q.parse_single_input('{"context": "K", "question": "F"}'), ("K", "F"))
        with self.assertRaises(ValueError):
            Q.parse_single_input("nur Kontext")

    def test_markierung(self):
        self.assertEqual(Q.mark_answer("Anna wohnt in Berlin.", 14, 20), "Anna wohnt in »Berlin«.")


class KennzahlenTest(unittest.TestCase):
    def test_normalisierung_wie_squad(self):
        self.assertEqual(Q.normalize_answer("The  Eiffel-Tower!"), "eiffeltower")
        self.assertEqual(Q.normalize_answer("a cat, an owl"), "cat owl")

    def test_em_f1(self):
        self.assertEqual(Q.score_one("the Berlin", ["Berlin"]), (1.0, 1.0))
        em, f1 = Q.score_one("in Berlin city", ["Berlin"])
        self.assertEqual(em, 0.0)
        self.assertAlmostEqual(f1, 2 * (1 / 3) * 1 / (1 / 3 + 1))   # P=1/3, R=1
        self.assertEqual(Q.score_one("Paris", ["Rom", "Paris"]), (1.0, 1.0))

    def test_keine_antwort(self):
        self.assertEqual(Q.score_one("", []), (1.0, 1.0))
        self.assertEqual(Q.score_one("Berlin", []), (0.0, 0.0))
        self.assertEqual(Q.score_one("", ["Berlin"]), (0.0, 0.0))

    def test_aggregat_mit_v2_aufteilung(self):
        ex = [{"id": "1", "answers": {"text": ["Rom"]}}, {"id": "2", "answers": {"text": []}}]
        m = Q.squad_metrics({"1": "Rom", "2": "Wien"}, ex)
        self.assertEqual(m["exact_match"], 0.5)
        self.assertEqual(m["has_answer_exact_match"], 1.0)
        self.assertEqual(m["no_answer_accuracy"], 0.0)


class FensterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tok = AutoTokenizer.from_pretrained(str(T.bert_dir(Path(cls.tmp.name) / "bert")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_langer_kontext_mehrere_fenster_antwort_stimmt(self):
        ctx = " ".join(["die stadt ist gross ."] * 30) + " anna wohnt in oslo ."
        ex = [{"id": "x", "context": ctx, "question": "wo wohnt anna ?",
               "answers": {"text": ["oslo"], "answer_start": [ctx.index("oslo")]}}]
        f = Q.prepare_train_features(self.tok, ex, max_length=48, stride=16)
        self.assertGreater(len(f["input_ids"]), 2)
        hits = [i for i, (s, e) in enumerate(zip(f["start_positions"], f["end_positions"])) if s > 0]
        self.assertTrue(hits, "Antwort in keinem Fenster")
        for i in hits:
            span = self.tok.decode(f["input_ids"][i][f["start_positions"][i]:f["end_positions"][i] + 1])
            self.assertEqual(span.strip(), "oslo")
        # Fenster ohne Antwort zeigen auf CLS (Index 0)
        self.assertIn(0, f["start_positions"])

    def test_postprocess_ueber_fenster_und_keine_antwort(self):
        import numpy as np
        ctx = "anna wohnt in oslo ."
        ex = [{"id": "x", "context": ctx, "question": "wo ?", "answers": {"text": [], "answer_start": []}}]
        feats = Q.prepare_eval_features(self.tok, ex, max_length=32, stride=8)
        n = len(feats["input_ids"][0])
        tok_oslo = next(i for i, o in enumerate(feats["offset_mapping"][0]) if o and ctx[o[0]:o[1]] == "oslo")
        s, e = np.zeros(n), np.zeros(n)
        s[tok_oslo] = e[tok_oslo] = 5.0
        pred = Q.postprocess(ex, feats, [s], [e])["x"]
        self.assertEqual((pred["text"], pred["start"]), ("oslo", 14))
        s[0] = e[0] = 9.0                      # CLS gewinnt -> keine Antwort (v2)
        self.assertEqual(Q.postprocess(ex, feats, [s], [e], allow_no_answer=True)["x"]["text"], "")
        self.assertEqual(Q.postprocess(ex, feats, [s], [e], allow_no_answer=False)["x"]["text"], "oslo")


def _dataset(root: Path):
    names, cities = ["anna", "ben", "clara", "david", "eva", "felix"], ["berlin", "paris", "rom", "wien", "oslo", "prag"]
    paras = []
    for i in range(24):
        n, c = names[i % 6], cities[(i * 5) % 6]
        ctx = f"{n} wohnt in {c} . die stadt ist schoen ."
        qas = [{"id": f"q{i}", "question": f"wo wohnt {n} ?", "answers": [{"text": c, "answer_start": ctx.index(c)}]}]
        if i % 4 == 0:
            qas.append({"id": f"n{i}", "question": f"wie alt ist {n} ?", "answers": [], "is_impossible": True})
        paras.append({"context": ctx, "qas": qas})
    (root / "train").mkdir(parents=True)
    (root / "train" / "train.json").write_text(json.dumps({"data": [{"paragraphs": paras}]}))
    (root / "test").mkdir()
    with open(root / "test" / "test.jsonl", "w") as fh:
        for i in range(6):
            fh.write(json.dumps({"context": f"{names[i]} wohnt in {cities[i]} .", "question": f"wo wohnt {names[i]} ?",
                                 "answer": cities[i]}) + "\n")


class MiniTrainingTest(unittest.TestCase):
    def test_train_und_test_engine(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = T.bert_dir(d / "bert")
            _dataset(d / "data")
            metrics, out, msgs = T.run_training("question_answering", T.base_cfg(
                model, d / "data", d / "out", "question_answering", epochs=3, learning_rate=3e-3))
            for k in ("exact_match", "f1", "exact_match_before", "f1_before", "final_train_loss"):
                self.assertIn(k, metrics)
            self.assertTrue(metrics["allow_no_answer"])
            self.assertTrue((Path(out) / Q.SPEC_FILE).is_file())
            self.assertTrue((Path(out) / "config.json").is_file())
            self.assertTrue(any(m["type"] == "progress" for m in msgs))

            res, _ = T.run_test("question_answering", {
                "model_path": out, "dataset_path": str(d / "data"), "output_path": str(d / "t"),
                "task_type": "question_answering", "mode": "dataset"})
            self.assertEqual(res["total_samples"], 6)
            self.assertIn("f1", res["metrics"])
            single, _ = T.run_test("question_answering", {
                "model_path": out, "output_path": str(d / "t"), "task_type": "question_answering",
                "mode": "single", "single_input": "eva wohnt in oslo .", "plugin_config": {"question": "wo wohnt eva ?"}})
            self.assertEqual(single["output_kind"], "qa_span")
            self.assertIn("answer_start", single)


if __name__ == "__main__":
    unittest.main()
