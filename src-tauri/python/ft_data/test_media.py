"""Tests fuer ft_data.media.

Aufruf:  python3 -m ft_data.test_media   (aus src-tauri/python/)
"""
import io
import tempfile
import unittest
from pathlib import Path

from ft_data.media import (
    MEDIA_CACHE_DIR, ensure_media_folders, evaluation_files, resolve_class_layout, sample,
)


def png_bytes(color=(255, 0, 0)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def imgs(self, d: Path, n: int):
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"b{i}.png").write_bytes(png_bytes())


class LayoutTest(Base):
    def test_train_und_test_ohne_val_ergibt_keine_klassen_train_test(self):
        self.imgs(self.root / "train" / "hund", 5)
        self.imgs(self.root / "train" / "katze", 5)
        self.imgs(self.root / "test" / "hund", 2)
        self.imgs(self.root / "test" / "katze", 2)
        lay = resolve_class_layout(self.root, "image")
        self.assertEqual(lay.classes, ["hund", "katze"])
        self.assertTrue(lay.val_from_train)
        self.assertEqual(len(lay.test), 4)
        self.assertEqual(len(lay.train) + len(lay.val), 10)

    def test_fehlende_klasse_in_val_verschiebt_keine_ids(self):
        for c in ("a", "b", "c"):
            self.imgs(self.root / "train" / c, 3)
        self.imgs(self.root / "val" / "c", 2)  # a und b fehlen in val
        lay = resolve_class_layout(self.root, "image")
        self.assertEqual({label for _, label in lay.val}, {2})

    def test_leere_split_ordner_zaehlen_nicht(self):
        self.imgs(self.root / "train" / "a", 4)
        self.imgs(self.root / "train" / "b", 4)
        (self.root / "val" / "a").mkdir(parents=True)   # vom App-Split, leer
        (self.root / "test" / "a").mkdir(parents=True)
        lay = resolve_class_layout(self.root, "image")
        self.assertTrue(lay.val_from_train)
        self.assertTrue(lay.val)

    def test_ungeteilter_ordner(self):
        self.imgs(self.root / "a", 3)
        self.imgs(self.root / "b", 3)
        (self.root / "leer").mkdir()
        lay = resolve_class_layout(self.root, "image")
        self.assertEqual(lay.classes, ["a", "b"])

    def test_nur_testordner_gibt_klaren_fehler(self):
        self.imgs(self.root / "test" / "a", 2)
        self.imgs(self.root / "test" / "b", 2)
        with self.assertRaises(ValueError) as cm:
            resolve_class_layout(self.root, "image")
        self.assertIn("train/", str(cm.exception))

    def test_auswertung_nimmt_ersten_split_mit_dateien(self):
        self.imgs(self.root / "train" / "a", 3)
        (self.root / "test" / "a").mkdir(parents=True)  # leer
        self.imgs(self.root / "val" / "a", 2)
        files, split = evaluation_files(self.root, "image")
        self.assertEqual(split, "val")
        self.assertEqual(len(files), 2)

    def test_stichprobe_ist_zufaellig_und_reproduzierbar(self):
        items = [("a", i) for i in range(50)] + [("b", i) for i in range(50)]
        picked = sample(items, 10)
        self.assertEqual(picked, sample(items, 10))
        self.assertEqual({c for c, _ in picked}, {"a", "b"})


class ParquetTest(Base):
    def write_hf_image_parquet(self, path: Path, labels, names):
        from datasets import ClassLabel, Dataset, Features, Image
        feats = Features({"image": Image(), "label": ClassLabel(names=names)})
        colors = [(255, 0, 0), (0, 0, 255)]
        ds = Dataset.from_dict(
            {"image": [{"bytes": png_bytes(colors[l % 2]), "path": None} for l in labels], "label": labels},
            features=feats,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(path))

    def test_hf_bild_parquet_wird_zu_klassenordnern(self):
        self.write_hf_image_parquet(self.root / "train.parquet", [0, 1, 0, 1, 0, 1], ["bohne", "blatt"])
        self.write_hf_image_parquet(self.root / "test.parquet", [0, 1], ["bohne", "blatt"])
        lay = resolve_class_layout(self.root, "image")
        self.assertEqual(lay.classes, ["blatt", "bohne"])
        self.assertEqual(len(lay.test), 2)
        self.assertEqual(lay.root, self.root / MEDIA_CACHE_DIR)
        # Zweiter Aufruf entpackt nicht erneut.
        marker = self.root / MEDIA_CACHE_DIR / ".complete.json"
        mtime = marker.stat().st_mtime_ns
        ensure_media_folders(self.root, "image")
        self.assertEqual(marker.stat().st_mtime_ns, mtime)

    def test_split_unterordner_und_ungelabelte_zeilen(self):
        from datasets import Dataset, Features, Image, Value
        self.write_hf_image_parquet(self.root / "train" / "0000.parquet", [0, 1, 1], ["x", "y"])
        # Testsplit wie bei vielen HF-Datasets: label -1
        feats = Features({"image": Image(), "label": Value("int64")})
        Dataset.from_dict({"image": [{"bytes": png_bytes(), "path": None}], "label": [-1]},
                          features=feats).to_parquet(str(self.root / "test" / "0000.parquet"))
        files, split = evaluation_files(self.root, "image")
        self.assertEqual(split, "train")  # test hat keine gelabelten Zeilen
        self.assertEqual(len(files), 3)

    def test_hf_audio_parquet_ohne_torchcodec(self):
        # datasets 5 braucht torchcodec zum Dekodieren — die Bytes werden hier
        # direkt aus Arrow gelesen und als .wav geschrieben.
        import json, struct, wave
        import pyarrow as pa, pyarrow.parquet as pq
        buf = io.BytesIO()
        w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 160, *([500, -500] * 80))); w.close()
        typ = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        table = pa.table({"audio": pa.array([{"bytes": buf.getvalue(), "path": "c.wav"}] * 4, type=typ),
                          "label": pa.array([0, 1, 0, 1], pa.int64())})
        meta = {"info": {"features": {"audio": {"_type": "Audio"},
                                      "label": {"_type": "ClassLabel", "names": ["ja", "nein"]}}}}
        pq.write_table(table.replace_schema_metadata({b"huggingface": json.dumps(meta).encode()}),
                       self.root / "train-00000-of-00001.parquet")
        lay = resolve_class_layout(self.root, "audio")
        self.assertEqual(lay.classes, ["ja", "nein"])
        self.assertEqual(lay.train[0][0].suffix, ".wav")

    def test_text_parquet_bleibt_unberuehrt(self):
        from datasets import Dataset
        Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}).to_parquet(str(self.root / "train.parquet"))
        self.assertEqual(ensure_media_folders(self.root, "image"), self.root)
        self.assertFalse((self.root / MEDIA_CACHE_DIR).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
