"""Tests fuer ft_data.segmentation (Layout, beide Maskenformate, Klassenlisten, mIoU).

Aufruf:  python3.11 -m unittest ft_data.test_segmentation   (aus src-tauri/python/)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ft_data import segmentation as seg

PALETTE = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]


def write_pair(img_dir: Path, mask_dir: Path, name: str, ids: np.ndarray, color: bool = False,
               palette_png: bool = False, mask_name: str = None):
    from PIL import Image
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    h, w = ids.shape
    Image.fromarray((np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)).save(img_dir / f"{name}.jpg")
    target = mask_dir / f"{mask_name or name}.png"
    if color:
        Image.fromarray(np.array(PALETTE, dtype=np.uint8)[ids]).save(target)
    elif palette_png:
        im = Image.fromarray(ids.astype(np.uint8), mode="P")
        im.putpalette([v for c in PALETTE for v in c] + [0] * (768 - 9))
        im.save(target)
    else:
        Image.fromarray(ids.astype(np.uint8), mode="L").save(target)


def ids_map(seed: int, h=16, w=16, n=3) -> np.ndarray:
    a = np.zeros((h, w), dtype=np.int64)
    a[: h // 2, :] = 1
    a[:, : w // 4] = 2 if n > 2 else 1
    return np.roll(a, seed, axis=1)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)


class LayoutTest(Base):
    def test_ungeteilt_trennt_val_ab_und_paart_mit_suffix(self):
        for i in range(5):
            write_pair(self.root / "images", self.root / "masks", f"b{i}", ids_map(i), mask_name=f"b{i}_mask")
        write_pair(self.root / "images", self.root / "masks_other", "verwaist", ids_map(0))
        lay = seg.resolve_layout(self.root, seed=1)
        self.assertEqual(len(lay.train) + len(lay.val), 5)
        self.assertEqual(len(lay.val), 1)
        self.assertTrue(any("ohne passende Maske" in n for n in lay.notes))
        for img, m in lay.train:
            self.assertEqual(seg.mask_key(m.stem), img.stem.lower())

    def test_geteilt_split_ordner_und_test_bleibt_unberuehrt(self):
        for s, n in (("train", 3), ("val", 2), ("test", 2)):
            for i in range(n):
                write_pair(self.root / s / "images", self.root / s / "masks", f"{s}{i}", ids_map(i))
        lay = seg.resolve_layout(self.root)
        self.assertEqual((len(lay.train), len(lay.val), len(lay.test)), (3, 2, 2))
        split, pairs = seg.evaluation_pairs(self.root)
        self.assertEqual(split, "test")
        self.assertEqual(len(pairs), 2)

    def test_ade_stil_images_split_und_annotations_split(self):
        for s in ("training", "validation"):
            for i in range(2):
                write_pair(self.root / "images" / s, self.root / "annotations" / s, f"{s}{i}", ids_map(i))
        lay = seg.resolve_layout(self.root)
        self.assertEqual((len(lay.train), len(lay.val)), (2, 2))

    def test_ohne_paare_klare_meldung(self):
        (self.root / "images").mkdir()
        with self.assertRaises(ValueError) as cm:
            seg.resolve_layout(self.root)
        self.assertIn("images/ + masks/", str(cm.exception))


class MaskFormatTest(Base):
    def test_klassen_id_masken_klassen_aus_masken(self):
        for i in range(3):
            write_pair(self.root / "images", self.root / "masks", f"b{i}", ids_map(i))
        lay = seg.resolve_layout(self.root)
        dec = seg.build_decoder(lay.all_masks)
        self.assertEqual(dec.mode, "index")
        self.assertEqual(dec.num_classes, 3)
        self.assertEqual(dec.names, ["class_0", "class_1", "class_2"])
        np.testing.assert_array_equal(dec.decode(lay.train[0][1]), ids_map(int(lay.train[0][0].stem[1:])))

    def test_paletten_png_liefert_indizes_und_255_ignoriert(self):
        ids = ids_map(0)
        ids[0, 0] = 255
        write_pair(self.root / "images", self.root / "masks", "voc", ids, palette_png=True)
        dec = seg.build_decoder([self.root / "masks" / "voc.png"])
        out = dec.decode(self.root / "masks" / "voc.png")
        self.assertEqual(out[0, 0], seg.IGNORE_INDEX)
        self.assertEqual(dec.num_classes, 3)  # 255 zaehlt nicht als Klasse

    def test_farbmasken_mit_farbtabelle_classes_json(self):
        (self.root / "classes.json").write_text(json.dumps([
            {"id": 0, "name": "hintergrund", "color": [0, 0, 0]},
            {"id": 1, "name": "auto", "color": "#ff0000"},
            {"id": 2, "name": "baum", "color": [0, 255, 0]},
        ]), encoding="utf-8")
        write_pair(self.root / "images", self.root / "masks", "c0", ids_map(2), color=True)
        info = seg.read_class_file(self.root)
        self.assertEqual(info.names, ["hintergrund", "auto", "baum"])
        dec = seg.build_decoder([self.root / "masks" / "c0.png"], info)
        self.assertEqual(dec.mode, "color")
        np.testing.assert_array_equal(dec.decode(self.root / "masks" / "c0.png"), ids_map(2))

    def test_farbmasken_ohne_tabelle_farben_abgeleitet_schwarz_zuerst(self):
        write_pair(self.root / "images", self.root / "masks", "c0", ids_map(1), color=True)
        dec = seg.build_decoder([self.root / "masks" / "c0.png"])
        self.assertEqual(dec.colors[0], (0, 0, 0))
        self.assertEqual(sorted(dec.colors), sorted(PALETTE))
        out = dec.decode(self.root / "masks" / "c0.png")
        # gleiche Pixel -> gleiche Klasse, schwarz -> 0
        self.assertTrue((out[ids_map(1) == 0] == 0).all())
        self.assertEqual(len(np.unique(out)), 3)

    def test_unbekannte_farbe_wird_ignoriert(self):
        write_pair(self.root / "images", self.root / "masks", "c0", ids_map(0), color=True)
        dec = seg.MaskDecoder("color", 2, ["a", "b"], [(0, 0, 0), (255, 0, 0)])
        out = dec.decode(self.root / "masks" / "c0.png")
        self.assertTrue((out[ids_map(0) == 2] == seg.IGNORE_INDEX).all())
        self.assertGreater(dec.unknown_pixels, 0)

    def test_labelmap_txt_cvat_und_classes_txt_mit_rgb(self):
        (self.root / "labelmap.txt").write_text("# label:color_rgb:parts:actions\nbackground:0,0,0::\ncar:255,0,0::\n")
        info = seg.read_class_file(self.root)
        self.assertEqual(info.names, ["background", "car"])
        self.assertEqual(info.colors, [(0, 0, 0), (255, 0, 0)])
        (self.root / "labelmap.txt").unlink()
        (self.root / "classes.txt").write_text("background\nauto 255 0 0\n")
        info = seg.read_class_file(self.root)
        self.assertEqual(info.names, ["background", "auto"])
        (self.root / "classes.txt").write_text("0 hintergrund\n1 strasse\n")
        self.assertEqual(seg.read_class_file(self.root).names, ["hintergrund", "strasse"])

    def test_klassenliste_bestimmt_klassenzahl_ids_darueber_ignoriert(self):
        (self.root / "classes.txt").write_text("hintergrund\nobjekt\n")
        write_pair(self.root / "images", self.root / "masks", "b0", ids_map(0))  # IDs 0,1,2
        info = seg.read_class_file(self.root)
        dec = seg.build_decoder([self.root / "masks" / "b0.png"], info)
        self.assertEqual(dec.num_classes, 2)
        out = dec.decode(self.root / "masks" / "b0.png")
        self.assertTrue((out[ids_map(0) == 2] == seg.IGNORE_INDEX).all())

    def test_test_engine_nutzt_farbtabelle_des_modells(self):
        write_pair(self.root / "images", self.root / "masks", "c0", ids_map(0), color=True)
        fallback = {"mask_mode": "color", "classes": ["bg", "rot", "gruen"],
                    "colors": [[0, 0, 0], [255, 0, 0], [0, 255, 0]]}
        dec = seg.build_decoder([self.root / "masks" / "c0.png"], None, fallback=fallback, num_classes=3)
        np.testing.assert_array_equal(dec.decode(self.root / "masks" / "c0.png"), ids_map(0))
        self.assertEqual(dec.names, ["bg", "rot", "gruen"])


class MetricTest(unittest.TestCase):
    def test_perfekte_vorhersage(self):
        y = ids_map(0)
        s = seg.scores(seg.confusion(y, y, 3))
        self.assertAlmostEqual(s["mean_iou"], 1.0)
        self.assertAlmostEqual(s["pixel_accuracy"], 1.0)

    def test_bekannte_werte_und_ignore(self):
        label = np.array([[0, 0, 1, 1], [0, 0, 1, 255]])
        pred = np.array([[0, 1, 1, 1], [0, 0, 0, 0]])
        cm = seg.confusion(pred, label, 2)
        np.testing.assert_array_equal(cm, [[3, 1], [1, 2]])   # 255 nicht gezaehlt
        s = seg.scores(cm, ["a", "b"])
        self.assertAlmostEqual(s["pixel_accuracy"], 5 / 7)
        self.assertAlmostEqual(s["per_class_iou"]["a"], 3 / 5)
        self.assertAlmostEqual(s["per_class_iou"]["b"], 2 / 4)
        self.assertAlmostEqual(s["mean_iou"], (3 / 5 + 2 / 4) / 2)

    def test_abwesende_klasse_zaehlt_nicht_im_miou(self):
        y = np.zeros((4, 4), dtype=np.int64)
        s = seg.scores(seg.confusion(y, y, 3), ["bg", "x", "y"])
        self.assertAlmostEqual(s["mean_iou"], 1.0)
        self.assertIsNone(s["per_class_iou"]["x"])

    def test_vorbereitung_maske_nearest_und_flip(self):
        from PIL import Image
        img = Image.new("RGB", (16, 8), (255, 255, 255))
        m = np.zeros((8, 16), dtype=np.int64)
        m[:, 8:] = 2
        x, y = seg.prepare_pair(img, m, (4, 4), [0.5] * 3, [0.5] * 3, flip=True)
        self.assertEqual(x.shape, (3, 4, 4))
        self.assertEqual(set(np.unique(y).tolist()), {0, 2})   # keine Zwischenwerte
        self.assertEqual(y[0, 0], 2)                           # gespiegelt
        self.assertAlmostEqual(float(x.max()), 1.0, places=5)

    def test_legende_und_ueberlagerung(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            pred = ids_map(0)
            files = seg.save_prediction(Image.new("RGB", (16, 16)), pred, seg.default_palette(3), Path(d), "x")
            for p in files.values():
                self.assertTrue(Path(p).is_file())
            rows = seg.legend(pred, ["a", "b", "c"], seg.default_palette(3))
            self.assertAlmostEqual(sum(r["share"] for r in rows), 1.0)
            self.assertTrue(rows[0]["color"].startswith("#"))


if __name__ == "__main__":
    unittest.main()
