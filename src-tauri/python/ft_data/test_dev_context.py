"""Tests fuer ft_data.dev_context.

Aufruf:  python3 -m ft_data.test_dev_context   (aus src-tauri/python/)
"""
import json
import tempfile
import unittest
from pathlib import Path

from ft_data.dev_context import MAX_CHARS, describe, describe_dataset, describe_model


class DevContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_chronos_wird_als_solches_erkannt(self):
        # Regression 27.10.2026: Chronos-2 (model_type "t5") bekam ein Raten-Skript.
        m = self.root / "chronos"
        m.mkdir()
        (m / "config.json").write_text(json.dumps({
            "model_type": "t5", "architectures": ["Chronos2Model"],
            "chronos_pipeline_class": "Chronos2Pipeline",
            "chronos_config": {"context_length": 8192, "quantiles": [0.1, 0.5]},
        }))
        (m / "model.safetensors").write_bytes(b"x")
        text = "\n".join(describe_model(m))
        self.assertIn("Chronos2Model", text)
        self.assertIn("Chronos2Pipeline", text)
        self.assertIn("context_length", text)
        self.assertNotIn("quantiles", text)  # Listen bleiben draussen (Laenge)

    def test_zeitreihen_tabelle_zeigt_spalten_und_reihen(self):
        d = self.root / "ds"
        (d / "train").mkdir(parents=True)
        (d / "val").mkdir()
        rows = [{"id": s, "timestamp": f"2024-01-0{i + 1}", "target": float(i)} for s in ("AAPL", "MSFT") for i in range(3)]
        (d / "train" / "data.json").write_text(json.dumps(rows))
        text = "\n".join(describe_dataset(d))
        self.assertIn("train/: 1 Dateien", text)
        self.assertIn("val/: leer", text)
        self.assertIn("Zeilen: 6", text)
        self.assertIn("id: 2 Werte", text)
        self.assertIn("target", text)

    def test_bildklassen_werden_genannt(self):
        d = self.root / "img"
        for cls in ("kreis", "quadrat"):
            (d / "train" / cls).mkdir(parents=True)
            (d / "train" / cls / "a.png").write_bytes(b"x")
        text = "\n".join(describe_dataset(d))
        self.assertIn("image-Klassenordner: kreis, quadrat", text)

    def test_fehlende_pfade_und_laenge(self):
        text = describe(str(self.root / "fehlt"), str(self.root / "auch_nicht"))
        self.assertIn("Pfad existiert nicht", text)
        self.assertIn("PYTHON:", text)
        self.assertLessEqual(len(text), MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
