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

    def test_gemischte_frequenzen_werden_gemeldet(self):
        # Regression 28.10.2026: Krypto taeglich + Aktien an Boersentagen —
        # Chronos predict_df brach mit "same frequency" ab.
        import pandas as pd
        from ft_data.dev_context import _frequency_summary, _series_frequency
        daily = pd.date_range("2024-01-01", periods=30, freq="D")
        bdays = pd.bdate_range("2024-01-01", periods=30).delete(5)  # Feiertag-Luecke
        df = pd.concat([
            pd.DataFrame({"id": "BTC", "timestamp": daily, "target": 1.0}),
            pd.DataFrame({"id": "AAPL", "timestamp": bdays, "target": 1.0}),
        ])
        self.assertEqual(_series_frequency(daily), "taeglich inkl. Wochenende")
        self.assertEqual(_series_frequency(bdays), "Boersen-/Werktage (ohne Wochenende)")
        text = "\n".join(_frequency_summary(df))
        self.assertIn("ACHTUNG: unterschiedliche Frequenzen", text)
        self.assertIn("AAPL", text)
        self.assertIn("BTC", text)

    def test_ohne_id_und_zeit_keine_frequenz(self):
        import pandas as pd
        from ft_data.dev_context import _frequency_summary
        self.assertEqual(_frequency_summary(pd.DataFrame({"text": ["a"], "label": [1]})), [])

    def test_bildklassen_werden_genannt(self):
        d = self.root / "img"
        for cls in ("kreis", "quadrat"):
            (d / "train" / cls).mkdir(parents=True)
            (d / "train" / cls / "a.png").write_bytes(b"x")
        text = "\n".join(describe_dataset(d))
        self.assertIn("image-Klassenordner: kreis, quadrat", text)

    def test_bibliotheks_api_kommt_aus_dem_installierten_paket(self):
        # Regression 27.10.2026: ohne die API von chronos-forecasting baute die
        # KI dreimal einen eigenen Loop um das Modell-Innere und scheiterte.
        from ft_data import dev_context
        m = self.root / "lib"
        m.mkdir()
        (m / "config.json").write_text(json.dumps({"model_type": "t5", "demo_pipeline_class": "DemoPipeline"}))
        dev_context.LIBRARY_APIS["DemoPipeline"] = {
            "pip": "demo-lib",
            "objects": [("json", None, ("dumps",)), ("fehlt_wirklich", None, ("x",))],
            "notes": ["Hinweis"],
        }
        try:
            text = "\n".join(dev_context.describe_library_api(m))
        finally:
            del dev_context.LIBRARY_APIS["DemoPipeline"]
        self.assertIn("BIBLIOTHEK FUER DIESES MODELL (demo-lib)", text)
        self.assertIn("nicht importierbar", text)  # zweites Modul fehlt → klarer Hinweis
        self.assertIn("erst demo-lib installieren", text)

    def test_ohne_pipeline_klasse_keine_bibliothek(self):
        from ft_data.dev_context import describe_library_api
        m = self.root / "bert"
        m.mkdir()
        (m / "config.json").write_text(json.dumps({"model_type": "bert"}))
        self.assertEqual(describe_library_api(m), [])

    def test_fehlende_pfade_und_laenge(self):
        text = describe(str(self.root / "fehlt"), str(self.root / "auch_nicht"))
        self.assertIn("Pfad existiert nicht", text)
        self.assertIn("PYTHON:", text)
        self.assertLessEqual(len(text), MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
