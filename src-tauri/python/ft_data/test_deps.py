"""Tests fuer ft_data.deps. Aufruf: python3 -m ft_data.test_deps (aus src-tauri/python/)."""
import sys
import unittest

from ft_data.deps import hint_for_exception, install_hint, missing, pip_name


class DepsTest(unittest.TestCase):
    def test_hinweis_nennt_app_und_genau_diesen_interpreter(self):
        h = install_hint("peft")
        self.assertIn("Einstellungen → Python-Pakete", h)
        self.assertIn("LLM Fine-Tuning", h)
        self.assertIn(f'"{sys.executable}" -m pip install "peft>=0.17.0"', h)
        self.assertNotIn("\n  pip install", h)

    def test_import_name_wird_zum_pip_namen(self):
        self.assertEqual(pip_name("sklearn.metrics"), "scikit-learn")
        self.assertEqual(pip_name("cv2"), "opencv-python")
        self.assertEqual(pip_name("sentence_transformers"), "sentence-transformers")

    def test_hinweis_aus_importerror(self):
        try:
            import gibt_es_nicht_123  # noqa: F401
        except ImportError as e:
            self.assertIn("gibt-es-nicht-123", hint_for_exception(e))

    def test_missing_ist_importerror_mit_hinweis(self):
        e = missing("diffusers", what="Diffusion")
        self.assertIsInstance(e, ImportError)
        self.assertIn("Generative Modelle", str(e))


if __name__ == "__main__":
    unittest.main(verbosity=2)
