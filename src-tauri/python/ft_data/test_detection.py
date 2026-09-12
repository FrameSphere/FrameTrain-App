"""Tests fuer ft_data.detection.

Aufruf:  python3 -m ft_data.test_detection   (aus src-tauri/python/)
"""
import tempfile
import unittest
from pathlib import Path

from ft_data.detection import convert_voc_to_yolo, fill_placeholder_names, label_path_for

VOC = """<annotation><filename>{name}.jpg</filename><size><width>200</width><height>100</height></size>
<object><name>{cls}</name><bndbox><xmin>50</xmin><ymin>25</ymin><xmax>150</xmax><ymax>75</ymax></bndbox></object>
</annotation>"""

YAML = "path: {root}\ntrain: train/images\nval: val/images\n\nnc: 0\nnames:\n  # Klassen hier eintragen:\n  - 'KlasseA'\n  - 'KlasseB'\n"


class VocTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_voc_split_wird_umgerechnet_und_klassen_eingetragen(self):
        for split, name, cls in (("train", "a", "hund"), ("train", "b", "katze"), ("val", "c", "hund")):
            (self.root / split / "images").mkdir(parents=True, exist_ok=True)
            (self.root / split / "annotations").mkdir(parents=True, exist_ok=True)
            (self.root / split / "images" / f"{name}.jpg").write_bytes(b"x")
            (self.root / split / "annotations" / f"{name}.xml").write_text(VOC.format(name=name, cls=cls))
        y = self.root / "dataset.yaml"
        y.write_text(YAML.format(root=self.root))

        self.assertEqual(convert_voc_to_yolo(self.root, y), 3)
        label = (self.root / "train" / "labels" / "a.txt").read_text().split()
        self.assertEqual(label, ["0", "0.500000", "0.500000", "0.500000", "0.500000"])
        text = y.read_text()
        self.assertIn("nc: 2", text)
        self.assertIn("- 'hund'", text)
        self.assertNotIn("KlasseA", text)
        self.assertIn("train: train/images", text)
        # Zweiter Lauf schreibt nichts neu.
        self.assertEqual(convert_voc_to_yolo(self.root, y), 0)

    def test_label_pfad_wie_ultralytics(self):
        self.assertEqual(label_path_for(Path("/d/train/images/x.jpg")), Path("/d/train/labels/x.txt"))
        self.assertEqual(label_path_for(Path("/d/bilder/x.png")), Path("/d/bilder/x.txt"))

    def test_platzhalter_klassen_aus_txt_labels(self):
        (self.root / "labels" / "train").mkdir(parents=True)
        (self.root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n")
        y = self.root / "dataset.yaml"
        y.write_text(YAML.format(root=self.root))
        self.assertTrue(fill_placeholder_names(self.root, y))
        self.assertIn("nc: 4", y.read_text())
        self.assertIn("- 'class_3'", y.read_text())
        self.assertFalse(fill_placeholder_names(self.root, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
