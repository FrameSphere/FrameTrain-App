"""Tests fuer ft_data.gguf_export — ohne Netz. Aufruf: python3 -m ft_data.test_gguf_export"""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from ft_data import gguf_export as G


def _tar(names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for n in names:
            data = b"x"
            info = tarfile.TarInfo(n)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


class GgufExportTest(unittest.TestCase):
    def test_nur_benoetigte_dateien_und_keine_pfad_tricks(self):
        tar = _tar(["llama.cpp-v0/convert_hf_to_gguf.py", "llama.cpp-v0/conversion/llama.py",
                    "llama.cpp-v0/gguf-py/gguf/__init__.py", "llama.cpp-v0/src/llama.cpp",
                    "llama.cpp-v0/../../boese.py"])
        names = sorted(m.name for m in G._safe_members(tar))
        self.assertEqual(names, ["conversion/llama.py", "convert_hf_to_gguf.py", "gguf-py/gguf/__init__.py"])

    def test_modelfile(self):
        with tempfile.TemporaryDirectory() as d:
            p = G.write_modelfile(Path(d), "model.gguf", 'Sei "kurz".')
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("FROM ./model.gguf"))
            self.assertIn('SYSTEM """Sei "kurz"."""', text)

    def test_cache_liegt_beim_nutzer_und_traegt_die_version(self):
        self.assertIn(G.LLAMA_CPP_TAG, str(G.cache_dir()))
        self.assertTrue(str(G.cache_dir()).startswith(str(Path.home())) or "LOCALAPPDATA" in str(G.cache_dir()))

    def test_offline_wird_zur_warnung_nicht_zum_absturz(self):
        orig = G.ensure_converter
        G.ensure_converter = lambda status=None: (_ for _ in ()).throw(OSError("offline"))
        try:
            ok, msg = G.convert(Path("/gibt/es/nicht"), Path("/tmp/x.gguf"))
        finally:
            G.ensure_converter = orig
        self.assertFalse(ok)
        self.assertIn("HF-Format", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
