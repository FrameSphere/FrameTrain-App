"""Formatumwandlung (YOLO/COCO -> Pixelboxen -> COCO-Annotation) und mAP.

Aufruf:  python3 plugins/hf_object_detection/test_formats_metrics.py   (aus train_engine/)
"""
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import core  # noqa: E402,F401  (haengt ft_data an sys.path)
from ft_data import od_formats as F            # noqa: E402
from ft_data.od_metrics import box_iou, detection_metrics  # noqa: E402
from plugins.hf_object_detection.plugin import split_records  # noqa: E402


def make_image(path: Path, w=100, h=50):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (120, 30, 30)).save(path)


class YoloLineTest(unittest.TestCase):
    def test_box(self):
        cls, box = F.yolo_line_to_box("2 0.5 0.5 0.2 0.4", 100, 50)
        self.assertEqual(cls, 2)
        for a, b in zip(box, [40, 15, 60, 35]):
            self.assertAlmostEqual(a, b)

    def test_polygon_wird_umschliessende_box(self):
        cls, box = F.yolo_line_to_box("0 0.1 0.2 0.3 0.2 0.3 0.6", 100, 100)
        for a, b in zip(box, [10, 20, 30, 60]):
            self.assertAlmostEqual(a, b)

    def test_ungueltig_und_abgeschnitten(self):
        self.assertIsNone(F.yolo_line_to_box("0 0.5 0.5", 10, 10))
        self.assertIsNone(F.yolo_line_to_box("x 0.5 0.5 0.1 0.1", 10, 10))
        _, box = F.yolo_line_to_box("0 0.05 0.5 0.2 0.2", 100, 100)  # ragt links heraus
        self.assertEqual(box[0], 0.0)


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _yolo(self):
        for split in ("train", "val"):
            make_image(self.root / split / "images" / f"{split}1.jpg")
            (self.root / split / "labels").mkdir(parents=True, exist_ok=True)
            (self.root / split / "labels" / f"{split}1.txt").write_text("1 0.5 0.5 0.2 0.4\n0 0.25 0.25 0.1 0.1\n")
        (self.root / "data.yaml").write_text("names:\n  0: cat\n  1: dog\n")

    def test_yolo_layout(self):
        self._yolo()
        ds = F.load_detection_dataset(self.root)
        self.assertEqual(ds["format"], "yolo_bbox")
        self.assertEqual(ds["classes"], ["cat", "dog"])
        rec = ds["splits"]["train"][0]
        self.assertEqual((rec["width"], rec["height"]), (100, 50))
        self.assertEqual([b["label"] for b in rec["boxes"]], [1, 0])

    def test_yolo_ids_ueber_namen_auf_modellklassen(self):
        """coco128 (0=person) gegen ein COCO-DETR (0=N/A, 1=person) — frueher mAP 0."""
        self._yolo()
        ds = F.load_detection_dataset(self.root, class_names=["N/A", "dog", "cat"])
        self.assertEqual([b["label"] for b in ds["splits"]["val"][0]["boxes"]], [1, 2])

    def test_remap_ohne_treffer_bleibt_identisch(self):
        self.assertEqual(F.remap_by_name(["class_0", "class_1"], ["a", "b"]), {0: 0, 1: 1})

    def test_coco_mit_lueckenhaften_ids(self):
        make_image(self.root / "train" / "a.jpg", 200, 100)
        make_image(self.root / "valid" / "b.jpg", 200, 100)
        cats = [{"id": 3, "name": "car"}, {"id": 17, "name": "cat"}]
        for split, fname in (("train", "a.jpg"), ("valid", "b.jpg")):
            (self.root / split / "_annotations.coco.json").write_text(json.dumps({
                "images": [{"id": 7, "file_name": fname, "width": 200, "height": 100}],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 17, "bbox": [10, 20, 30, 40]},
                                {"id": 2, "image_id": 7, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 1}],
                "categories": cats}))
        ds = F.load_detection_dataset(self.root)
        self.assertEqual(ds["format"], "coco_json")
        self.assertEqual(ds["classes"], ["car", "cat"])
        self.assertEqual(set(ds["splits"]), {"train", "val"})
        boxes = ds["splits"]["train"][0]["boxes"]
        self.assertEqual(boxes, [{"label": 1, "box": [10.0, 20.0, 40.0, 60.0]}])  # crowd faellt raus
        # Beim Testen: Modellklassen in anderer Reihenfolge
        ds2 = F.load_detection_dataset(self.root, class_names=["cat", "car"])
        self.assertEqual(ds2["splits"]["val"][0]["boxes"][0]["label"], 0)

    def test_leeres_dataset_meldet_formate(self):
        with self.assertRaises(ValueError) as ctx:
            F.load_detection_dataset(self.root)
        self.assertIn("COCO", str(ctx.exception))

    def test_coco_target_und_spiegeln(self):
        rec = {"path": "x", "width": 100, "height": 50, "boxes": [{"label": 3, "box": [10, 5, 30, 25]}]}
        t = F.to_coco_target(rec, 4)
        self.assertEqual(t["image_id"], 4)
        self.assertEqual(t["annotations"][0]["bbox"], [10, 5, 20, 20])
        self.assertEqual(t["annotations"][0]["area"], 400)
        self.assertEqual(F.hflip_record(rec)["boxes"][0]["box"], [70, 5, 90, 25])

    def test_split_records_zweigt_val_ab(self):
        recs = [{"path": str(i), "boxes": []} for i in range(10)]
        train, val = split_records({"train": recs})
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({r["path"] for r in train} & {r["path"] for r in val})
        train, val = split_records({"train": recs[:3], "val": recs[3:4]})
        self.assertEqual((len(train), len(val)), (3, 1))


def det(label, box, score=0.9):
    return {"label": label, "score": score, "box": box}


def gt(label, box):
    return {"label": label, "box": box}


class MetricsTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_perfekt(self):
        m = detection_metrics([[det("a", [0, 0, 10, 10])], [det("b", [5, 5, 20, 20])]],
                              [[gt("a", [0, 0, 10, 10])], [gt("b", [5, 5, 20, 20])]])
        self.assertAlmostEqual(m["mAP50"], 1.0)
        self.assertAlmostEqual(m["mAP50-95"], 1.0)
        self.assertEqual((m["precision"], m["recall"]), (1.0, 1.0))

    def test_ohne_vorhersage_und_falsche_klasse(self):
        self.assertEqual(detection_metrics([[]], [[gt("a", [0, 0, 10, 10])]])["mAP50"], 0.0)
        m = detection_metrics([[det("b", [0, 0, 10, 10])]], [[gt("a", [0, 0, 10, 10])]])
        self.assertEqual((m["mAP50"], m["precision"], m["recall"]), (0.0, 0.0, 0.0))

    def test_verschobene_box_zaehlt_nur_bis_iou_06(self):
        # IoU = 0.6 genau: Treffer bei 0.50, 0.55, 0.60 -> mAP50-95 = 3/10
        g = [0, 0, 10, 10]
        p = [0, 0, 10, 6]  # Schnitt 60, Vereinigung 100
        self.assertAlmostEqual(box_iou(p, g), 0.6)
        m = detection_metrics([[det("a", p)]], [[gt("a", g)]])
        self.assertAlmostEqual(m["mAP50"], 1.0)
        self.assertAlmostEqual(m["mAP50-95"], 0.3)

    def test_doppelte_box_ist_fehlalarm(self):
        g = [[gt("a", [0, 0, 10, 10])]]
        m = detection_metrics([[det("a", [0, 0, 10, 10], 0.9), det("a", [0, 0, 10, 10], 0.8)]], g)
        self.assertAlmostEqual(m["mAP50"], 1.0)  # Treffer kommt zuerst: AP bleibt 1
        self.assertAlmostEqual(m["precision"], 0.5)
        # Fehlalarm mit hoeherer Sicherheit vor dem Treffer senkt die AP
        m2 = detection_metrics([[det("a", [50, 50, 60, 60], 0.95), det("a", [0, 0, 10, 10], 0.5)]], g)
        self.assertAlmostEqual(m2["mAP50"], 0.5, places=2)

    def test_halber_recall(self):
        m = detection_metrics([[det("a", [0, 0, 10, 10])]],
                              [[gt("a", [0, 0, 10, 10]), gt("a", [20, 20, 30, 30])]])
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["mAP50"], 51 / 101, places=4)  # Precision 1 bis Recall 0.5

    def test_mittel_ueber_klassen_mit_gt(self):
        m = detection_metrics([[det("a", [0, 0, 10, 10]), det("z", [40, 40, 50, 50])]],
                              [[gt("a", [0, 0, 10, 10]), gt("b", [20, 20, 30, 30])]])
        self.assertAlmostEqual(m["mAP50"], 0.5)  # a=1, b=0; z ohne GT zaehlt nicht

    def test_schwelle_fuer_precision(self):
        m = detection_metrics([[det("a", [0, 0, 10, 10], 0.1)]], [[gt("a", [0, 0, 10, 10])]], conf=0.25)
        self.assertAlmostEqual(m["mAP50"], 1.0)
        self.assertEqual(m["recall"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
