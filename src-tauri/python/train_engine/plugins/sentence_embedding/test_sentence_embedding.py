"""Schnelle Tests fuer das Embedding-Plugin (ohne Modell, ohne Netz).

Aufruf (aus src-tauri/python/):  python3 -m unittest train_engine/plugins/sentence_embedding/test_sentence_embedding.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PY = Path(__file__).resolve().parents[3]          # src-tauri/python
sys.path.insert(0, str(_PY))
sys.path.insert(0, str(_PY / "test_engine"))

from ft_data import embedding as emb               # noqa: E402


class FormatErkennungTest(unittest.TestCase):
    def spec(self, rows, cfg=None):
        return emb.resolve_spec(list(rows[0].keys()), rows, cfg or {})

    def test_paare_query_document(self):
        s = self.spec([{"query": "q", "document": "d"}])
        self.assertEqual((s["format"], s["anchor_column"], s["positive_column"]), ("pairs", "query", "document"))

    def test_frage_antwort(self):
        self.assertEqual(self.spec([{"id": 1, "question": "q", "answer": "a"}])["format"], "pairs")

    def test_tripel(self):
        s = self.spec([{"anchor": "a", "positive": "p", "negative": "n"}])
        self.assertEqual(s["format"], "triplets")
        self.assertEqual(s["negative_column"], "negative")

    def test_score_paare_werden_normiert(self):
        rows = [{"sentence1": "a", "sentence2": "b", "score": 5.0},
                {"sentence1": "c", "sentence2": "d", "score": 2.5}]
        s = self.spec(rows)
        self.assertEqual((s["format"], s["score_scale"]), ("scored", 5.0))
        cols = emb.to_columns(rows, s)
        self.assertEqual(cols["score"], [1.0, 0.5])
        self.assertEqual(list(cols), ["sentence1", "sentence2", "score"])

    def test_label_als_score_bei_paaren(self):
        # MRPC-artig: sentence1/sentence2/label mit 0/1
        s = self.spec([{"sentence1": "a", "sentence2": "b", "label": 1}])
        self.assertEqual((s["format"], s["score_column"], s["score_scale"]), ("scored", "label", 1.0))

    def test_paare_ohne_score(self):
        self.assertEqual(self.spec([{"sentence1": "a", "sentence2": "b"}])["format"], "pairs")

    def test_texte_mit_klasse(self):
        rows = [{"text": "x", "label": "sport"}, {"text": "y", "label": "wetter"}]
        s = self.spec(rows)
        self.assertEqual(s["format"], "labeled")
        ids = {}
        cols = emb.to_columns(rows, s, ids)
        self.assertEqual(cols["label"], [0, 1])
        self.assertEqual(ids, {"sport": 0, "wetter": 1})

    def test_eigene_spalten(self):
        s = self.spec([{"frage": "q", "antwort": "a"}], {"anchor_column": "frage", "positive_column": "antwort"})
        self.assertEqual(s["format"], "pairs")
        with self.assertRaises(ValueError):
            self.spec([{"frage": "q"}], {"anchor_column": "fehlt", "positive_column": "frage"})

    def test_unbekannt_gibt_hinweis(self):
        with self.assertRaises(ValueError) as ctx:
            self.spec([{"foo": 1}])
        self.assertIn("anchor_column", str(ctx.exception))

    def test_leere_zeilen_fallen_weg(self):
        s = self.spec([{"anchor": "a", "positive": "p"}])
        cols = emb.to_columns([{"anchor": "a", "positive": ""}, {"anchor": "b", "positive": "q"}], s)
        self.assertEqual(cols, {"anchor": ["b"], "positive": ["q"]})

    def test_score_skala(self):
        self.assertEqual(emb.score_scale([0.2, 0.9]), 1.0)
        self.assertEqual(emb.score_scale([0, 3.8]), 5.0)
        self.assertEqual(emb.score_scale([0, 80]), 80.0)


class RetrievalTest(unittest.TestCase):
    def test_retrieval_set_fasst_dubletten_zusammen(self):
        q, c, rel = emb.retrieval_set({"anchor": ["a", "b", "a"], "positive": ["x", "x", "y"],
                                       "negative": ["n", "n", "n"]})
        self.assertEqual(len(q), 2)
        self.assertEqual(sorted(c.values()), ["n", "x", "y"])
        self.assertEqual(len(rel["q0"]), 2)

    def test_metriken(self):
        m = emb.retrieval_metrics([["d1", "d2"], ["d3", "d1"], ["d9"]], [{"d1"}, {"d1"}, {"d2"}])
        self.assertAlmostEqual(m["recall_at_1"], 1 / 3)
        self.assertAlmostEqual(m["recall_at_5"], 2 / 3)
        self.assertAlmostEqual(m["mrr_at_10"], (1 + 0.5) / 3)

    def test_rangfolge_und_knn(self):
        import numpy as np
        corpus = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype="float32")
        idx, sims = emb.rank_corpus(np.array([[1, 0.1]]), corpus, top_k=2)
        self.assertEqual(list(idx[0]), [0, 2])
        self.assertEqual(emb.knn_label_accuracy(np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]),
                                                ["a", "a", "b", "b"]), 1.0)


class DateienTest(unittest.TestCase):
    def test_zeilen_aus_csv_und_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.csv").write_text("query,document\nq1,d1\n")
            (Path(d) / "b.jsonl").write_text(json.dumps({"query": "q2", "document": "d2"}) + "\n")
            self.assertEqual(emb.load_rows(Path(d) / "a.csv"), [{"query": "q1", "document": "d1"}])
            self.assertEqual(emb.load_rows(Path(d) / "b.jsonl"), [{"query": "q2", "document": "d2"}])

    def test_spec_neben_dem_modell(self):
        with tempfile.TemporaryDirectory() as d:
            emb.save_spec(Path(d), {"format": "pairs"})
            self.assertEqual(emb.load_spec(Path(d))["format"], "pairs")
            self.assertEqual(emb.load_spec(Path(d) / "fehlt"), {})


class ModellServerTest(unittest.TestCase):
    def test_modules_json_macht_embedding(self):
        import model_server
        with tempfile.TemporaryDirectory() as d:
            cfg = {"architectures": ["BertModel"], "model_type": "bert"}
            self.assertEqual(model_server.detect_modality_for_dir(Path(d), cfg), "text")
            (Path(d) / "modules.json").write_text("[]")
            self.assertEqual(model_server.detect_modality_for_dir(Path(d), cfg), "embedding")
        self.assertEqual(model_server.INPUT_KIND["embedding"], "text")


if __name__ == "__main__":
    unittest.main()
