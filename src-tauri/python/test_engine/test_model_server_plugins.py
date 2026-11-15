"""Modell-Server: Aufgaben mit eigenem Test-Plugin (Labor und Hosting).

Aufruf:  python3 test_model_server_plugins.py   (aus test_engine/)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_server as ms


class Erkennung(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_architektur_bestimmt_die_aufgabe(self):
        for arch, want in [("SegformerForSemanticSegmentation", "segmentation"),
                           ("BertForQuestionAnswering", "qa"),
                           ("YolosForObjectDetection", "hf_detect")]:
            self.assertEqual(ms.detect_plugin_modality(self.dir, {"architectures": [arch]}), want)

    def test_leere_modelle_und_patchtst(self):
        self.assertEqual(ms.detect_plugin_modality(self.dir, {"model_type": "ft_tabular"}), "tabular")
        self.assertEqual(ms.detect_plugin_modality(self.dir, {"model_type": "patchtst"}), "time_series")
        self.assertEqual(ms.detect_plugin_modality(self.dir, {"model_type": "ft_time_series"}), "time_series")

    def test_metadaten_gehen_vor_architektur(self):
        # Ein Reranker meldet sich als BertForSequenceClassification.
        (self.dir / "frametrain_rerank.json").write_text(json.dumps({"task": "reranker"}))
        self.assertEqual(ms.detect_plugin_modality(
            self.dir, {"architectures": ["BertForSequenceClassification"]}), "reranker")

    def test_yolo_und_klassifikatoren_bleiben_beim_alten_weg(self):
        (self.dir / "model.json").write_text(json.dumps({"framework": "ultralytics", "task": "detect"}))
        self.assertIsNone(ms.detect_plugin_modality(self.dir, {}))
        self.assertIsNone(ms.detect_plugin_modality(
            self.dir, {"architectures": ["DistilBertForSequenceClassification"]}))

    def test_jede_aufgabe_hat_eingabeart_und_plugin(self):
        plugins = Path(__file__).resolve().parent / "plugins"
        tasks = {json.loads((d / "manifest.json").read_text())["task_type"]
                 for d in plugins.iterdir() if (d / "manifest.json").exists()}
        for modality, (task, kind) in ms.PLUGIN_TASKS.items():
            self.assertIn(task, tasks, modality)
            self.assertEqual(ms.INPUT_KIND[modality], kind)


class Anfrage(unittest.TestCase):
    """_infer_plugin baut die Eingabe des Einzeltests und reicht das Ergebnis durch."""

    class FakePlugin:
        def __init__(self):
            from core.config import TestConfig
            self.config = TestConfig()

        def run_single(self):
            from core.protocol import TestProtocol
            TestProtocol.status("running", "wird nicht durchgereicht")
            TestProtocol.complete_single(predicted=self.config.single_input, confidence=0.5,
                                         top_predictions=[], inference_time=0.01,
                                         extra={"detections": [{"label": "a", "confidence": 0.9}]})

    def server(self, modality):
        s = ms.ModelServer("/tmp")
        s.modality = modality
        s.plugin = self.FakePlugin()
        return s

    def test_frage_und_kontext_werden_zusammengesetzt(self):
        r = self.server("qa").infer({"text": "Mainz liegt am Rhein.", "question": "Wo liegt Mainz?"})
        self.assertEqual(r["predicted"], "Wo liegt Mainz? ||| Mainz liegt am Rhein.")
        self.assertNotIn("predicted_output", r)
        self.assertNotIn("mode", r)

    def test_reranker_nimmt_eine_passage_je_zeile(self):
        r = self.server("reranker").infer({"text": "A\n\nB", "question": "F"})
        self.assertEqual(r["predicted"], "F ||| A ||| B")

    def test_erkennung_liefert_boxen_wie_yolo(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            r = self.server("hf_detect").infer({"file_path": f.name})
        self.assertEqual(r["boxes"], [{"label": "a", "confidence": 0.9}])

    def test_ohne_text_klare_meldung(self):
        with self.assertRaises(ValueError):
            self.server("tabular").infer({})


if __name__ == "__main__":
    unittest.main()
