"""Tests fuer den Reranker: Formate, MRR/nDCG, Paar-Kennzahlen, Mini-Training.

Aufruf (aus src-tauri/python/):
  python3 -m unittest train_engine/plugins/reranker/test_reranker.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _tiny_models as T  # noqa: E402
from ft_data import rerank as R  # noqa: E402


class FormatTest(unittest.TestCase):
    def spec(self, rows, cfg=None):
        return R.resolve_spec(list(rows[0].keys()), rows, cfg or {})

    def test_tripel(self):
        s = self.spec([{"query": "q", "positive": "p", "negative": "n"}])
        self.assertEqual((s["format"], s["negative_column"]), ("triplets", "negative"))

    def test_query_positive_ohne_negative(self):
        s = self.spec([{"question": "q", "pos": "p"}])
        self.assertEqual((s["format"], s["negative_column"]), ("triplets", None))

    def test_paare_mit_label_und_skala(self):
        rows = [{"query": "q", "passage": "p", "label": 1}, {"query": "q", "passage": "x", "label": 0}]
        s = self.spec(rows)
        self.assertEqual((s["format"], s["score_scale"]), ("scored", 1.0))
        s5 = self.spec([{"sentence1": "a", "sentence2": "b", "score": 4.0}])
        self.assertEqual((s5["format"], s5["score_scale"]), ("scored", 5.0))
        g = R.groups([{"sentence1": "a", "sentence2": "b", "score": 4.0}], s5)
        self.assertAlmostEqual(g["a"]["b"], 0.8)

    def test_negatives_als_liste(self):
        rows = [{"query": "q", "positive": "p", "negatives": json.dumps(["n1", "n2"])}]
        g = R.groups(rows, self.spec(rows))
        self.assertEqual(g, {"q": {"p": 1.0, "n1": 0.0, "n2": 0.0}})

    def test_eigene_spalten_und_fehler(self):
        rows = [{"suche": "q", "text_a": "p", "note": 2}]
        s = self.spec(rows, {"query_column": "suche", "passage_column": "text_a", "score_column": "note"})
        self.assertEqual(s["format"], "scored")
        with self.assertRaises(ValueError):
            self.spec([{"foo": 1}])
        with self.assertRaises(ValueError):
            self.spec([{"query": "q", "passage": "p", "label": "ja"}])

    def test_gezogene_negative_und_split_nach_fragen(self):
        grp = {f"q{i}": {f"p{i}": 1.0} for i in range(12)}
        neg = R.add_sampled_negatives(grp, 3, seed=1)
        for q, g in neg.items():
            self.assertEqual(sum(1 for v in g.values() if v == 0.0), 3)
            self.assertEqual(g[f"p{q[1:]}"], 1.0)
        train, val = R.split_groups(grp, 0.2, seed=1)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(val), 2)

    def test_einzeleingabe(self):
        self.assertEqual(R.parse_single_input("f ||| a ||| b"), ("f", ["a", "b"]))
        self.assertEqual(R.parse_single_input("nur frage"), ("nur frage", []))


class KennzahlenTest(unittest.TestCase):
    def test_mrr(self):
        self.assertEqual(R.mrr_at_k([0, 0, 1]), 1 / 3)
        self.assertEqual(R.mrr_at_k([0] * 10 + [1]), 0.0)

    def test_ndcg(self):
        self.assertEqual(R.ndcg_at_k([1, 0, 0]), 1.0)
        self.assertAlmostEqual(R.ndcg_at_k([0, 1]), (1 / math.log2(3)) / 1.0)
        # abgestufte Relevanz: 0.5 vor 1.0 ist schlechter als umgekehrt
        self.assertLess(R.ndcg_at_k([0.5, 1.0]), 1.0)

    def test_ranking_ueber_fragen(self):
        ranked = R.rank({"a": [("x", 0.9, 1.0), ("y", 0.1, 0.0)],
                         "b": [("x", 0.9, 0.0), ("y", 0.2, 1.0)],
                         "c": [("z", 0.5, 0.0)]})           # ohne Relevantes: zaehlt nicht
        m = R.ranking_metrics(ranked)
        self.assertEqual(m["ranked_queries"], 2)
        self.assertAlmostEqual(m["mrr_at_10"], 0.75)
        self.assertEqual(m["hit_at_1"], 0.5)

    def test_paare(self):
        m = R.pair_metrics([0.9, 0.2, 0.7, 0.4], [1, 0, 0, 1])
        self.assertEqual(m["accuracy"], 0.5)
        self.assertIn("spearman", m)

    def test_evaluate_mit_bewertungsfunktion(self):
        grp = {"katze": {"katzen fakten": 1.0, "hunde fakten": 0.0}}
        m, ranked = R.evaluate(lambda pairs: [1.0 if p.split()[0][:4] == q[:4] else 0.0 for q, p in pairs],
                               grp, "triplets")
        self.assertEqual(m["mrr_at_10"], 1.0)
        self.assertEqual(ranked["katze"][0][0], "katzen fakten")


class MiniTrainingTest(unittest.TestCase):
    def test_train_und_test_engine(self):
        topics = ["cats", "dogs", "rain", "tea", "coffee", "berlin", "paris", "rom", "wien", "oslo", "prag", "anna"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = T.bert_dir(d / "bert")
            (d / "data").mkdir()
            with open(d / "data" / "train.jsonl", "w") as fh:
                for i, t in enumerate(topics):
                    fh.write(json.dumps({"query": f"tell me about {t}",
                                         "positive": f"this passage explains facts about {t}",
                                         "negative": f"this passage explains facts about {topics[(i + 3) % 12]}"}) + "\n")
            metrics, out, msgs = T.run_training("reranker", T.base_cfg(
                model, d / "data", d / "out", "reranker", epochs=3, learning_rate=2e-3))
            for k in ("mrr_at_10", "ndcg_at_10", "mrr_at_10_before", "final_train_loss"):
                self.assertIn(k, metrics)
            self.assertEqual(metrics["rerank_format"], "triplets")
            self.assertTrue((Path(out) / R.SPEC_FILE).is_file())
            st = json.loads((Path(out) / "config_sentence_transformers.json").read_text())
            self.assertEqual(st.get("model_type"), "CrossEncoder")
            self.assertTrue(R.single_head(Path(out)))

            res, _ = T.run_test("reranker", {"model_path": out, "dataset_path": str(d / "data"),
                                             "output_path": str(d / "t"), "task_type": "reranker", "mode": "dataset"})
            self.assertIn("ndcg_at_10", res["metrics"])
            single, _ = T.run_test("reranker", {"model_path": out, "output_path": str(d / "t"), "task_type": "reranker",
                                                "mode": "single", "single_input": "tell me about tea ||| a ||| b"})
            self.assertEqual(len(single["top_predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
