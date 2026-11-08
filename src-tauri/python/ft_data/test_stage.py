"""Tests fuer ft_data.stage.

Aufruf:  python3 -m ft_data.test_stage   (aus src-tauri/python/)
"""
import json
import tempfile
import unittest
from pathlib import Path

from ft_data import stage
from ft_data.test_fetch import _png


class StageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _yolo(self):
        src = self.root / "ds"
        for split in ("train", "val"):
            (src / split / "images").mkdir(parents=True)
            (src / split / "labels").mkdir(parents=True)
        (src / "data.yaml").write_text("names:\n  - person\n  - car\n  - truck\n  - dog\n")
        rows = {"a": "0 .5 .5 .1 .1\n2 .3 .3 .1 .1\n", "b": "3 .5 .5 .2 .2\n", "c": "1 .5 .5 .2 .2\n", "d": ""}
        for stem, lab in rows.items():
            (src / "train" / "images" / f"{stem}.jpg").write_bytes(_png())
            (src / "train" / "labels" / f"{stem}.txt").write_text(lab)
        return src

    def test_boxen_umbenennen_zusammenlegen_weglassen(self):
        res = stage.stage({"source": str(self._yolo()), "target": str(self.root / "w"), "project_kind": "image_boxes",
                           "class_map": {"car": "vehicle", "Truck": "vehicle", "person": "person"}, "keep_unmapped": False})
        self.assertEqual(sorted(res["label_classes"]), ["person", "vehicle"])
        self.assertEqual(res["samples"], 3)  # Bild mit nur "dog" faellt raus, leeres Bild bleibt
        self.assertEqual(res["dropped_classes"], {"dog": 1})
        labels = "".join(p.read_text() for p in Path(res["path"], "labels").glob("*.txt"))
        self.assertEqual(sorted(l.split()[0] for l in labels.splitlines()),
                         sorted(str(res["label_classes"].index(n)) for n in ["person", "vehicle", "vehicle"]))

    def test_klassenordner_mit_stichprobe_je_klasse(self):
        src = self.root / "imgs"
        for cls, n in (("Cat", 10), ("dog", 2)):
            (src / "train" / cls).mkdir(parents=True)
            for i in range(n):
                (src / "train" / cls / f"{i}.png").write_bytes(_png())
        res = stage.stage({"source": str(src), "target": str(self.root / "w"), "project_kind": "image_classify",
                           "class_map": {"cat": "katze"}, "max_samples": 4, "per_class": True})
        self.assertEqual(res["per_class"], {"katze": 2, "dog": 2})
        self.assertTrue((Path(res["path"]) / "katze").is_dir())

    def test_text_tabellen_aus_csv(self):
        src = self.root / "txt"
        src.mkdir()
        (src / "train.csv").write_text("review,sentiment\ngut,pos\nschlecht,neg\nok,neutral\n")
        res = stage.stage({"source": str(src), "target": str(self.root / "w"), "project_kind": "text_classify",
                           "class_map": {"neutral": None}})
        rows = [json.loads(l) for l in Path(res["path"]).read_text().splitlines()]
        self.assertEqual({r["label"] for r in rows}, {"pos", "neg"})
        self.assertEqual(res["source_columns"], {"text": "review", "label": "sentiment"})

    def test_text_paare(self):
        src = self.root / "pairs"
        src.mkdir()
        (src / "data.jsonl").write_text('{"question": "2+2?", "answer": "4"}\n')
        res = stage.stage({"source": str(src), "target": str(self.root / "w"), "project_kind": "text_pairs"})
        self.assertEqual(json.loads(Path(res["path"]).read_text()), {"text": "2+2?", "target": "4"})

    def test_audio_mit_transkript(self):
        src = self.root / "asr"
        src.mkdir()
        for i in range(3):
            (src / f"r{i}.wav").write_bytes(b"RIFF")
            (src / f"r{i}.txt").write_text(f"satz {i}")
        res = stage.stage({"source": str(src), "target": str(self.root / "w"), "project_kind": "audio_transcript", "max_samples": 2})
        self.assertEqual(res["samples"], 2)
        self.assertEqual(len(list(Path(res["path"]).glob("*.txt"))), 2)


if __name__ == "__main__":
    unittest.main()
