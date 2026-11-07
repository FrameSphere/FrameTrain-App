"""Tests fuer ft_data.fetch.

Aufruf:  python3 -m ft_data.test_fetch   (aus src-tauri/python/)
"""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from ft_data import fetch


def _png(w=40, h=20) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_select_ordner_muster_und_datei(self):
        files = [("sample/images/a.jpg", 10), ("sample/labels/a.txt", 1), ("videos/x.mp4", 9999), ("README.md", 3)]
        self.assertEqual([p for p, _ in fetch.select(files, ["sample/"])], ["sample/images/a.jpg", "sample/labels/a.txt"])
        self.assertEqual([p for p, _ in fetch.select(files, ["*.txt"])], ["sample/labels/a.txt"])
        self.assertEqual(len(fetch.select(files, [])), 4)

    def test_summarize_ordner_und_bilder(self):
        s = fetch.summarize([("sample/a.jpg", 2_000_000), ("videos/x.mp4", 9_000_000)])
        self.assertEqual(s["folders"][0]["path"], "videos")
        self.assertEqual(s["image_files"], 1)

    def test_bbox_umrechnung(self):
        self.assertEqual(fetch._bbox_to_yolo([10, 5, 20, 10], 40, 20, "coco"), (0.5, 0.5, 0.5, 0.5))
        self.assertEqual(fetch._bbox_to_yolo([10, 5, 30, 15], 40, 20, "xyxy"), (0.5, 0.5, 0.5, 0.5))
        self.assertIsNone(fetch._bbox_to_yolo([10, 5, 0, 10], 40, 20, "coco"))

    def test_parquet_mit_bildern_wird_yolo(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        src = self.root / "raw" / "data"
        src.mkdir(parents=True)
        rows = [{"image": {"bytes": _png(), "path": None},
                 "objects": {"bbox": [[10, 5, 20, 10]], "category": [1]}} for _ in range(10)]
        table = pa.Table.from_pylist(rows)
        meta = {b"huggingface": json.dumps({"info": {"features": {"objects": {"feature": {"category": {"names": ["person", "car"]}}}}}}).encode()}
        pq.write_table(table.replace_schema_metadata(meta), src / "train-00000.parquet")
        self.assertTrue(fetch.needs_yolo_conversion(self.root / "raw"))
        out = self.root / "yolo"
        res = fetch.parquet_to_yolo(self.root / "raw", out, max_rows=8)
        self.assertEqual(res["images"], 8)
        self.assertEqual(res["classes"], ["person", "car"])
        rep = fetch.inspect(out)
        self.assertEqual(rep["format"], "yolo")
        self.assertEqual(rep["images"], 8)
        self.assertIn("val", rep["splits"])
        self.assertEqual(rep["classes"], ["person", "car"])
        label = next((out / "train" / "labels").glob("*.txt")).read_text().split()
        self.assertEqual(label[0], "1")

    def test_inspect_meldet_fehlende_bilder(self):
        (self.root / "a.parquet").write_bytes(b"x")
        rep = fetch.inspect(self.root)
        self.assertEqual(rep["images"], 0)
        self.assertTrue(any("No image files" in p for p in rep["problems"]))

    def test_zip_wird_entpackt_und_ordner_abgestiegen(self):
        z = self.root / "d.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("ds/train/images/a.jpg", _png())
            zf.writestr("ds/train/labels/a.txt", "0 0.5 0.5 0.1 0.1\n")
            zf.writestr("ds/data.yaml", "names:\n  - person\n")
        root = fetch.extract_if_archive(z, self.root / "x")
        self.assertEqual(root.name, "ds")
        rep = fetch.inspect(root)
        self.assertEqual((rep["format"], rep["classes"]), ("yolo", ["person"]))

    def test_zip_mit_pfad_nach_aussen_wird_abgelehnt(self):
        z = self.root / "evil.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../../etc/x.txt", "boom")
        with self.assertRaises(fetch.FetchError):
            fetch.extract_if_archive(z, self.root / "x")

    def test_url_nur_https(self):
        with self.assertRaises(fetch.FetchError):
            fetch.url_download({"url": "http://example.com/a.zip"}, self.root)

    def test_names_formate(self):
        self.assertEqual(fetch._parse_names("names: [a, 'b']\n"), ["a", "b"])
        self.assertEqual(fetch._parse_names("names:\n  0: person\n  1: car\nnc: 2\n"), ["person", "car"])


if __name__ == "__main__":
    unittest.main()
