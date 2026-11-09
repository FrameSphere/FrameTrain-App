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
        self.assertEqual(rep["format"], "parquet")

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

class FakeResp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}


class FakeSession:
    """Range-faehiger Server im Speicher; zaehlt, wie viele Bytes geholt wurden."""
    def __init__(self, files):
        self.files, self.sent = files, 0

    def head(self, url, **kw):
        data = self.files[url]
        return FakeResp(200, headers={"content-length": str(len(data)), "accept-ranges": "bytes"})

    def get(self, url, headers=None, **kw):
        data = self.files[url]
        a, b = headers["Range"].split("=")[1].split("-")
        chunk = data[int(a):int(b) + 1]
        self.sent += len(chunk)
        return FakeResp(206, chunk)


class MehrFormateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _zip(self, entries):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for name, data in entries:
                zf.writestr(name, data)
        return buf.getvalue()

    def test_teil_download_holt_nur_gewaehlte_bilder_und_passende_labels(self):
        big = b"x" * 200_000
        imgs = self._zip([(f"cam1/img_{i:03d}.jpg", big) for i in range(30)])
        lbls = self._zip([(f"labels/img_{i:03d}.txt", "0 0.5 0.5 0.2 0.2\n") for i in range(30)] + [("labels/classes.txt", "person\ncar\n")])
        sess = FakeSession({"https://z/img.zip": imgs, "https://z/lbl.zip": lbls})
        out = self.root / "x"
        info = fetch.partial_zip("https://z/img.zip", out, [], 5, None, sess, 100)
        self.assertEqual(info["media"], 5)
        self.assertLess(sess.sent, len(imgs) / 3)
        stems = {p.stem for p in out.rglob("*.jpg")}
        fetch.partial_zip("https://z/lbl.zip", out, [], 0, stems, sess, 100)
        self.assertEqual(len(list(out.rglob("img_*.txt"))), 5)
        paired = fetch.pair_images_and_labels(out)
        rep = fetch.inspect(paired)
        self.assertEqual((rep["format"], rep["images"], rep["classes"]), ("yolo", 5, ["person", "car"]))

    def test_coco_json_wird_yolo(self):
        (self.root / "imgs").mkdir()
        for n in ("a.jpg", "b.jpg"):
            (self.root / "imgs" / n).write_bytes(_png())
        coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 40, "height": 20}, {"id": 2, "file_name": "b.jpg", "width": 40, "height": 20}],
                "annotations": [{"image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 10]}],
                "categories": [{"id": 3, "name": "person"}, {"id": 7, "name": "car"}]}
        (self.root / "train.json").write_text(json.dumps(coco))
        out = fetch.coco_to_yolo(self.root)
        rep = fetch.inspect(out)
        self.assertEqual(rep["classes"], ["person", "car"])
        self.assertEqual(rep["images"], 2)
        txt = [p.read_text() for p in out.rglob("a.txt")][0]
        self.assertTrue(txt.startswith("1 0.500000 0.500000 0.500000 0.500000"))

    def test_audio_formate(self):
        for cls in ("hund", "katze"):
            (self.root / cls).mkdir()
            (self.root / cls / "a.wav").write_bytes(b"RIFF")
        rep = fetch.inspect(self.root)
        self.assertEqual((rep["format"], rep["classes"]), ("audio_folders", ["hund", "katze"]))
        t = self.root / "asr"
        t.mkdir()
        (t / "x.wav").write_bytes(b"RIFF")
        (t / "x.txt").write_text("hallo")
        self.assertEqual(fetch.inspect(t)["format"], "audio_transcript")

    def test_text_tabelle(self):
        (self.root / "train.csv").write_text("text,label\nhi,a\nho,a\nhu,b\n")
        rep = fetch.inspect(self.root)
        self.assertEqual(rep["format"], "table")
        t = rep["table"]
        self.assertEqual((t["rows"], t["task"], t["labels"]), (3, "classification", {"a": 2, "b": 1}))
        self.assertEqual(len(t["examples"]), 3)

    def test_llm_tabelle_wird_erkannt(self):
        (self.root / "chat.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ho"}]}) + "\n")
        t = fetch.inspect(self.root)["table"]
        self.assertEqual((t["task"], t["detected"]["llm_format"]), ("chat", "messages"))

    def test_zeilengrenze_laedt_nur_noetige_shards(self):
        """COCO-Fall: Repo viel groesser als max_mb, aber 100 Zeilen passen in einen Shard."""
        import sys
        import types
        import pyarrow as pa
        import pyarrow.parquet as pq
        src = self.root / "src"
        src.mkdir()
        for name in ("data/train-00000.parquet", "data/train-00001.parquet", "data/val-00000.parquet"):
            (src / name).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"x": list(range(150))}), src / name)
        files = [("data/val-00000.parquet", 400_000_000), ("data/train-00001.parquet", 400_000_000),
                 ("data/train-00000.parquet", 400_000_000)]
        geholt = []

        def fake_download(repo, path, repo_type=None, revision=None, local_dir=None):
            geholt.append(path)
            dst = Path(local_dir) / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((src / path).read_bytes())
            return str(dst)

        mod = types.ModuleType("huggingface_hub")
        mod.hf_hub_download = fake_download
        alt_mod = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = mod
        alt_list = fetch.hf_list
        fetch.hf_list = lambda repo, revision=None: files
        try:
            out = self.root / "out"
            fetch.hf_download({"repo_id": "x/coco", "max_rows": 100, "max_mb": 500}, out)
        finally:
            fetch.hf_list = alt_list
            if alt_mod is not None:
                sys.modules["huggingface_hub"] = alt_mod
            else:
                del sys.modules["huggingface_hub"]
        self.assertEqual(geholt, ["data/train-00000.parquet"])

    def test_klassennamen_aus_hf_features(self):
        feats = {"image": {"_type": "Image"}, "objects": {"feature": {
            "category": {"names": ["person", "bicycle", "car"], "_type": "ClassLabel"},
            "bbox": {"feature": {"dtype": "float64"}, "_type": "Sequence"}}, "_type": "Sequence"},
            "source": {"names": ["a", "b"], "_type": "ClassLabel"}}
        self.assertEqual(fetch._classlabel_names(feats), ["person", "bicycle", "car"])
        self.assertEqual(fetch._classlabel_names({"x": {"dtype": "int64"}}), [])

    def test_paralleler_download_holt_alle_dateien_und_wiederholt_fehler(self):
        import sys
        import threading
        import types
        geholt, gleichzeitig, maximal, lock = [], [0], [0], threading.Lock()
        fehler = {"b/2.jpg": 1}

        def fake_download(repo, path, repo_type=None, revision=None, local_dir=None):
            import time
            with lock:
                gleichzeitig[0] += 1
                maximal[0] = max(maximal[0], gleichzeitig[0])
            time.sleep(0.05)
            with lock:
                gleichzeitig[0] -= 1
                if fehler.get(path):
                    fehler[path] -= 1
                    raise OSError("kurz weg")
                geholt.append(path)
            return str(Path(local_dir) / path)

        mod = types.ModuleType("huggingface_hub")
        mod.hf_hub_download = fake_download
        alt = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = mod
        try:
            files = [(f"b/{i}.jpg", 10) for i in range(20)]
            fetch._download_parallel("x/y", files, None, self.root, 200)
        finally:
            if alt is not None:
                sys.modules["huggingface_hub"] = alt
            else:
                del sys.modules["huggingface_hub"]
        self.assertEqual(sorted(geholt), sorted(p for p, _ in files))
        self.assertGreater(maximal[0], 1)


if __name__ == "__main__":
    unittest.main()
