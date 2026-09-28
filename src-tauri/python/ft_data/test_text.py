"""Tests fuer ft_data.text.

Aufruf:  python3 -m ft_data.test_text   (aus src-tauri/python/)
"""
import unittest

from ft_data.text import (
    MultiLabelError, detect_columns, display_label, expected_label, is_unlabeled, label_value, safe_text,
)


class TextColumnsTest(unittest.TestCase):
    def test_satzpaare_werden_erkannt(self):
        self.assertEqual(detect_columns(["idx", "sentence1", "sentence2", "label"]), ("sentence1", "sentence2", "label"))
        self.assertEqual(detect_columns(["premise", "hypothesis", "label"]), ("premise", "hypothesis", "label"))
        self.assertEqual(detect_columns(["question1", "question2", "is_duplicate"]), ("question1", "question2", "is_duplicate"))

    def test_einzeltext_bleibt_wie_bisher(self):
        self.assertEqual(detect_columns(["text", "label"]), ("text", None, "label"))
        self.assertEqual(detect_columns(["review", "stars", "__index_level_0__"]), ("review", None, "stars"))

    def test_ungelabelt(self):
        for v in (None, -1, -1.0, "-1", "", float("nan"), []):
            self.assertTrue(is_unlabeled(v), v)
        for v in (0, 1, "pos", False, [3]):
            self.assertFalse(is_unlabeled(v), v)

    def test_multilabel_gibt_klaren_fehler(self):
        self.assertEqual(label_value(["P"], "prmu"), "P")
        with self.assertRaises(MultiLabelError):
            label_value(["P", "R"], "prmu")

    def test_classlabel_namen(self):
        self.assertEqual(display_label(1, ["neg", "pos"]), "pos")
        self.assertEqual(display_label(1.0, []), "1")
        self.assertEqual(display_label("x", ["neg"]), "x")

    def test_erwartetes_label_im_test(self):
        mapping = {"0": "neg", "1": "pos"}
        self.assertEqual(expected_label(1, mapping), "pos")
        self.assertEqual(expected_label("0", mapping), "neg")
        self.assertIsNone(expected_label(-1, mapping))
        self.assertEqual(expected_label(2, {}), "2")  # alte Modelle ohne Mapping

    def test_leere_zellen(self):
        self.assertEqual(safe_text(None), "")
        self.assertEqual(safe_text(float("nan")), "")


class ProblemTypeTest(unittest.TestCase):
    def test_single_label_bleibt_standard(self):
        from ft_data.text import SINGLE_LABEL, resolve_problem_type
        # Genau die Faelle, die bisher liefen, muessen Single-Label bleiben.
        self.assertEqual(resolve_problem_type(["pos", "neg", "pos"]), SINGLE_LABEL)
        self.assertEqual(resolve_problem_type([1, 2, 3, 4, 5]), SINGLE_LABEL)        # Sterne = Klassen
        self.assertEqual(resolve_problem_type([0.0, 1.0, 0.0]), SINGLE_LABEL)
        self.assertEqual(resolve_problem_type([["a"], ["b"]]), SINGLE_LABEL)         # Einer-Listen
        self.assertEqual(resolve_problem_type([0.5, 1.5, 2.5]), SINGLE_LABEL)        # zu wenige Werte

    def test_multi_label_erkennung_und_schalter(self):
        from ft_data.text import MULTI_LABEL, SINGLE_LABEL, resolve_problem_type
        self.assertEqual(resolve_problem_type(["sport;politik", "sport"]), MULTI_LABEL)
        self.assertEqual(resolve_problem_type(["a|b"]), MULTI_LABEL)
        self.assertEqual(resolve_problem_type([["a", "b"], ["a"]]), MULTI_LABEL)
        self.assertEqual(resolve_problem_type(["a;b"], multi_label=False), SINGLE_LABEL)
        self.assertEqual(resolve_problem_type(["a;b"], multi_label="false"), SINGLE_LABEL)
        self.assertEqual(resolve_problem_type(["a"], multi_label=True), MULTI_LABEL)

    def test_regression(self):
        from ft_data.text import REGRESSION, resolve_problem_type
        self.assertEqual(resolve_problem_type([i / 7 for i in range(20)]), REGRESSION)
        self.assertEqual(resolve_problem_type([1, 2, 3], problem_type="regression"), REGRESSION)
        self.assertEqual(resolve_problem_type(["3,5", "4,25"] * 1, problem_type="regression"), REGRESSION)

    def test_split_multi_labels(self):
        from ft_data.text import split_multi_labels
        self.assertEqual(split_multi_labels("sport; politik ;"), ["sport", "politik"])
        self.assertEqual(split_multi_labels([0, 2], ["a", "b", "c"]), ["a", "c"])
        self.assertEqual(split_multi_labels(None), [])

    def test_kennzahlen(self):
        from ft_data.text import multi_label_scores, regression_scores
        m = multi_label_scores([[1, 0, 1], [0, 1, 0]], [[0.9, 0.2, 0.7], [0.1, 0.4, 0.2]])
        self.assertAlmostEqual(m["subset_accuracy"], 0.5)
        self.assertAlmostEqual(m["micro_f1"], 0.8)
        r = regression_scores([1, 2, 3, 4], [1, 2, 3, 5])
        self.assertAlmostEqual(r["mse"], 0.25)
        self.assertAlmostEqual(r["mae"], 0.25)
        self.assertGreater(r["pearson"], 0.9)
        flat = regression_scores([1, 2, 3], [2, 2, 2])      # konstante Vorhersage: kein NaN
        self.assertEqual(flat["pearson"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
