"""Prueft hf_training.resume_checkpoint (Fortsetzen ab checkpoint-N).

Hintergrund: Nach "Stoppen" registriert die App den letzten HF-Checkpoint als
Version, aber jedes Training begann trotzdem bei Schritt 0 — Optimizer- und
Scheduler-Zustand im Checkpoint blieben ungenutzt.

Aufruf:  python3.11 test_resume.py   (aus train_engine/)
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import TrainingConfig            # noqa: E402
from core.hf_training import resume_checkpoint    # noqa: E402


def checkpoint(folder: Path, step: int) -> Path:
    c = folder / f"checkpoint-{step}"
    c.mkdir(parents=True)
    (c / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return c


def cfg(tmp: Path, value, model_path=None) -> TrainingConfig:
    c = TrainingConfig()
    c.output_path = str(tmp / "job" / "final_model")
    c.checkpoint_dir = str(tmp / "job" / "checkpoints")
    c.model_path = str(model_path or tmp / "model")
    c.plugin_config = {"resume_from_checkpoint": value}
    return c


def run(c):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = resume_checkpoint(c)
    return result, buf.getvalue()


class ResumeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_aus_ist_standard(self):
        checkpoint(self.root / "job" / "checkpoints", 50)
        for v in (None, False, "", "false"):
            self.assertIsNone(run(cfg(self.root, v))[0], v)

    def test_auto_nimmt_den_neuesten_im_eigenen_ordner(self):
        checkpoint(self.root / "job" / "checkpoints", 50)
        newest = checkpoint(self.root / "job" / "checkpoints", 100)
        result, log = run(cfg(self.root, "auto"))
        self.assertEqual(result, str(newest))
        self.assertIn("ab Schritt 100", log)

    def test_auto_nimmt_die_stopp_version(self):
        # Nach "Stoppen" ist die Version selbst ein checkpoint-N-Ordner.
        version = checkpoint(self.root / "versions", 40)
        self.assertEqual(run(cfg(self.root, True, model_path=version))[0], str(version))

    def test_expliziter_pfad_auf_alten_job(self):
        old = self.root / "alt" / "checkpoints"
        checkpoint(old, 10)
        want = checkpoint(old, 20)
        self.assertEqual(run(cfg(self.root, str(old)))[0], str(want))
        self.assertEqual(run(cfg(self.root, str(want)))[0], str(want))

    def test_ohne_checkpoint_freundlich_weiter(self):
        result, log = run(cfg(self.root, True))
        self.assertIsNone(result)
        self.assertIn("von vorn", log)


if __name__ == "__main__":
    unittest.main()
