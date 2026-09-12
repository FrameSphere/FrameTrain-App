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


if __name__ == "__main__":
    unittest.main(verbosity=2)
