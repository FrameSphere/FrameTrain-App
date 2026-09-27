"""Konsistenz der Plugin-Beschreibungen (manifest.json <-> plugin.py <-> Test-Engine).

Architekturlisten standen bisher an drei Stellen (Manifest, plugin.py,
Frontend) und wurden von Hand gepflegt. Dieser Test prueft die Python-Seite
ohne ein einziges Plugin zu importieren (per AST, also ohne torch/transformers):
  - jedes manifest.json ist gueltig, 'entry' existiert, 'class' ist dort definiert
  - task_types sind eindeutig
  - SUPPORTED_ARCHITECTURES im plugin.py == supported_architectures im Manifest
    (auch im Test-Engine-Plugin, gegen das Train-Manifest desselben task_type)
  - jedes Test-Engine-Plugin hat ein Gegenstueck in der Train-Engine
Die Frontend-Seite prueft src/plugins/__tests__/pluginManifests.test.ts.

Aufruf:  python3.11 test_plugin_consistency.py   (aus train_engine/)
"""
import ast
import json
import unittest
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "plugins"
TEST = Path(__file__).resolve().parent.parent / "test_engine" / "plugins"

# Train-Plugins ohne Test-Engine-Plugin, mit Grund. Die Gegenrichtung (Test ohne
# Train) ist nie erlaubt: was man nicht trainieren kann, braucht keinen Test.
TRAIN_ONLY = {
    "canvas": "Synapse-Netze testet canvas_inference_server.py",
    "detect": "YOLO testet run_yolo_inference (Rust) und yolo_inference_server.py",
    "image_classification": "torchvision-Plugin: Test ueber die Canvas-/Bild-Pfade der App",
}
REQUIRED_KEYS = ("task_type", "entry", "class", "name")


def manifests(root: Path):
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "manifest.json").is_file():
            yield d, json.loads((d / "manifest.json").read_text(encoding="utf-8"))


def module_info(path: Path):
    """(Klassennamen, SUPPORTED_ARCHITECTURES oder None) aus dem Quelltext."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    archs = None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "SUPPORTED_ARCHITECTURES" for t in targets):
            archs = set(ast.literal_eval(node.value))
    return classes, archs


class ManifestTest(unittest.TestCase):
    def check_root(self, root: Path, reference=None):
        """reference: task_type -> Architekturen (fuer die Test-Engine: die Train-Manifeste)."""
        seen = {}
        for d, m in manifests(root):
            with self.subTest(plugin=str(d.relative_to(root.parent.parent))):
                for key in REQUIRED_KEYS:
                    self.assertTrue(m.get(key), f"manifest.json ohne '{key}'")
                entry = d / m["entry"]
                self.assertTrue(entry.is_file(), f"entry {m['entry']} fehlt")
                classes, archs = module_info(entry)
                self.assertIn(m["class"], classes, f"Klasse {m['class']} nicht in {m['entry']}")
                self.assertNotIn(m["task_type"], seen,
                                 f"task_type doppelt: {d.name} und {seen.get(m['task_type'])}")
                seen[m["task_type"]] = d.name
                if archs is not None:
                    want = (reference or {}).get(m["task_type"], m.get("supported_architectures"))
                    self.assertEqual(archs, set(want or []),
                                     "SUPPORTED_ARCHITECTURES und Manifest weichen ab")
        return seen

    def test_train_engine(self):
        self.assertGreaterEqual(len(self.check_root(TRAIN)), 8)

    def test_test_engine(self):
        # Das Test-Plugin muss genau das laden koennen, was das Train-Plugin trainiert.
        train = {m["task_type"]: m.get("supported_architectures") for _, m in manifests(TRAIN)}
        self.check_root(TEST, reference=train)

    def test_jedes_test_plugin_hat_ein_train_plugin(self):
        train = {m["task_type"]: d.name for d, m in manifests(TRAIN)}
        test = {m["task_type"]: d.name for d, m in manifests(TEST)}
        for task_type, name in test.items():
            self.assertIn(task_type, train, f"Test-Plugin {name} ohne Train-Plugin")
            self.assertEqual(train[task_type], name, "gleicher task_type, anderer Ordnername")
        untested = set(train) - set(test)
        self.assertEqual(untested, set(TRAIN_ONLY),
                         "Train-Plugin ohne Test-Plugin: in TRAIN_ONLY begruenden oder Test-Plugin anlegen")

    def test_manifest_architekturen_ohne_dubletten(self):
        for d, m in manifests(TRAIN):
            archs = m.get("supported_architectures") or []
            self.assertEqual(len(archs), len(set(archs)), d.name)


if __name__ == "__main__":
    unittest.main()
