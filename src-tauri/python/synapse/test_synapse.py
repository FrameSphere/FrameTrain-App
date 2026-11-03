"""Katalog, Migration und Formen-Pruefung.

Aufruf:  python3.11 -m unittest synapse.test_synapse   (aus src-tauri/python)
"""
import json
import re
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from synapse.check import check_graph, check_raw
from synapse.graph import ENTRY, Graph, GEdge, GNode, load_graph, migrate_v1
from synapse.runtime import SynapseModule
from synapse.spec import REGISTRY, catalog

ROOT = Path(__file__).resolve().parents[3]
NODE_TYPES = ROOT / "src" / "components" / "synapse" / "nodeTypes.ts"
GENERATED = ROOT / "src" / "components" / "synapse" / "catalog.generated.json"


def chain(*items):
    nodes = [GNode(f"n{i}", t, p) for i, (t, p) in enumerate(items)]
    edges = [GEdge(ENTRY, "out", "n0", nodes[0].spec.inputs[0].id)]
    for a, b in zip(nodes, nodes[1:]):
        edges.append(GEdge(a.id, a.spec.outputs[0].id, b.id, b.spec.inputs[0].id))
    return Graph(nodes, edges, {"out": (nodes[-1].id, nodes[-1].spec.outputs[0].id)})


class CatalogTest(unittest.TestCase):
    def test_jeder_editor_node_ist_im_katalog(self):
        ui = re.findall(r"^\s+type: '(\w+)', label:", NODE_TYPES.read_text(encoding="utf-8"), re.M)
        self.assertEqual(sorted(ui), sorted(n["type"] for n in catalog()["nodes"]))

    def test_erzeugter_katalog_ist_aktuell(self):
        self.assertEqual(json.loads(GENERATED.read_text(encoding="utf-8")), catalog())

    def test_rechnende_nodes_haben_eine_laufzeit(self):
        for spec in REGISTRY.values():
            self.assertEqual(spec.run is None, spec.structural, spec.type)


class MigrationTest(unittest.TestCase):
    def test_gewichte_und_werte_alter_modelle_bleiben(self):
        raw = {"version": 1, "execution_order": ["i", "c", "d"], "edges": [
            {"source": "i", "target": "c"}, {"source": "c", "target": "d"}], "nodes": [
            {"id": "i", "type": "input", "params": {}},
            {"id": "c", "type": "conv2d", "params": {"inChannels": 3, "outChannels": 8, "kernelSize": "3", "padding": "1"}},
            {"id": "d", "type": "dense", "params": {"inputSize": 8, "outputSize": 2}}]}
        m = SynapseModule(load_graph(raw)).eval()
        self.assertEqual(sorted(m.state_dict())[:2], ["layers.c.bias", "layers.c.weight"])
        x = torch.ones(1, 3, 8, 8)
        conv, dense = nn.Conv2d(3, 8, 3, padding=1), nn.Linear(8, 2)
        conv.load_state_dict(m.layers["c"].state_dict())
        dense.load_state_dict(m.layers["d"].state_dict())
        self.assertTrue(torch.equal(m(x), dense(conv(x).mean(dim=(2, 3)))))

    def test_layer_ohne_eingang_bekommt_die_modell_eingabe(self):
        g = migrate_v1([{"id": "d", "type": "dense", "params": {"inputSize": 4, "outputSize": 2}}], [])
        self.assertEqual(SynapseModule(g)(torch.ones(1, 4)).shape, (1, 2))

    def test_kreis_ist_ein_klarer_fehler(self):
        g = migrate_v1([{"id": "a", "type": "relu"}, {"id": "b", "type": "relu"}],
                       [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}])
        with self.assertRaisesRegex(ValueError, "Kreis"):
            SynapseModule(g)


class CheckTest(unittest.TestCase):
    def test_auto_groessen_folgen_der_vorstufe(self):
        g = chain(("conv2d", {"outChannels": 8, "padding": "1"}), ("batchnorm", {}), ("relu", {}),
                  ("pool", {"type": "global_avg"}), ("dense", {"outputSize": 3}))
        r = check_graph(g, (1, 3, 32, 32))
        self.assertTrue(r["ok"], r["errors"])
        self.assertEqual(r["nodes"]["n4"]["outputs"]["out"], [1, 3])
        self.assertEqual(r["params"], (3 * 8 * 9 + 8) + 16 + (8 * 3 + 3))

    def test_auto_modell_trainiert_echt(self):
        g = chain(("dense", {"outputSize": 8}), ("relu", {}), ("dense", {"outputSize": 2}))
        m = SynapseModule(g, input_shape=(1, 4))
        self.assertFalse(next(m.parameters()).is_meta)
        m(torch.ones(2, 4)).sum().backward()
        self.assertIsNotNone(m.layers["n0"].weight.grad)

    def test_fehler_steht_am_richtigen_node(self):
        g = chain(("conv2d", {"inChannels": 3, "outChannels": 8}), ("transpose", {}),
                  ("conv2d", {"inChannels": 99, "outChannels": 8}))
        r = check_graph(g, (1, 3, 32, 32))
        self.assertFalse(r["ok"])
        self.assertEqual(r["errors"][0]["node"], "n2")

    def test_alle_rechnenden_nodes_laufen_auf_meta(self):
        shapes = {"embedding": (1, 8), "conv2d": (1, 3, 16, 16), "batchnorm": (1, 3, 16, 16), "pool": (1, 3, 16, 16),
                  "conv3d": (1, 3, 4, 16, 16), "pool3d": (1, 3, 4, 16, 16),
                  "conv_transpose2d": (1, 3, 8, 8), "upsample": (1, 3, 8, 8)}
        sized = {"attention": {"numHeads": 2}, "transformer_block": {"numHeads": 2, "ffnDim": 8},
                 "embedding": {"vocabSize": 100, "embeddingDim": 8}, "matmul": {}, "dropout": {"p": 0.1},
                 "transpose": {"dim0": 1, "dim1": 2}, "reshape": {"shape": "-1, 32"}}
        for spec in REGISTRY.values():
            if spec.structural or spec.type in ("reparameterize", "pretrained", "block", "transformer_decoder"):
                continue
            with self.subTest(node=spec.type):
                g = chain((spec.type, sized.get(spec.type, {})))
                if spec.type == "matmul":
                    shape = (2, 4, 4)
                else:
                    shape = shapes.get(spec.type, (2, 4, 8))
                r = check_graph(g, shape, torch.long if spec.type == "embedding" else torch.float32)
                self.assertTrue(r["ok"], r["errors"])

    def test_vit_groesse_wird_in_kurzer_zeit_geprueft(self):
        import time
        items = [("dense", {"outputSize": 192})]
        items += [("transformer_block", {"numHeads": 3, "ffnDim": 768, "dropout": 0})] * 12
        g = chain(*items)
        check_graph(g, (1, 196, 768))
        t = time.time()
        r = check_graph(g, (1, 196, 768))
        self.assertTrue(r["ok"], r["errors"])
        self.assertLess(time.time() - t, 0.5)


class StepTest(unittest.TestCase):
    def test_ein_schritt_zeigt_forward_gradient_und_update(self):
        from synapse.step import one_step
        raw = {"version": 1, "execution_order": ["i", "a", "r", "b"], "nodes": [
            {"id": "i", "type": "input", "params": {"shape": "-1, 6"}},
            {"id": "a", "type": "dense", "params": {"inputSize": "auto", "outputSize": 8, "implicitPool": False}},
            {"id": "r", "type": "relu", "params": {}},
            {"id": "b", "type": "dense", "params": {"inputSize": "auto", "outputSize": 3, "implicitPool": False}}],
            "edges": [{"source": "i", "target": "a"}, {"source": "a", "target": "r"}, {"source": "r", "target": "b"}]}
        r = one_step(raw, lr=0.1)
        self.assertEqual(r["order"], ["a", "r", "b"])
        self.assertEqual(r["nodes"]["b"]["shape"], [4, 3])
        self.assertGreater(r["nodes"]["a"]["grad"], 0)
        self.assertAlmostEqual(r["nodes"]["a"]["update"], 0.1 * r["nodes"]["a"]["grad"], places=4)
        self.assertEqual(r["nodes"]["a"]["params"], 6 * 8 + 8)
        self.assertNotIn("grad", r["nodes"]["r"])


class GuessInputTest(unittest.TestCase):
    def test_feste_dense_groesse_schlaegt_standard_des_input_nodes(self):
        raw = {"version": 1, "execution_order": ["i", "d"], "edges": [{"source": "i", "target": "d"}], "nodes": [
            {"id": "i", "type": "input", "params": {"shape": "-1, 3, 224, 224"}},
            {"id": "d", "type": "dense", "params": {"inputSize": 128, "outputSize": 64}}]}
        self.assertTrue(check_raw(raw)["ok"])


class ServiceTest(unittest.TestCase):
    def test_dienst_prueft_ueber_json_zeilen(self):
        import subprocess
        import sys
        raw = {"version": 1, "execution_order": ["i", "c", "d"], "edges": [
            {"source": "i", "target": "c"}, {"source": "c", "target": "d"}], "nodes": [
            {"id": "i", "type": "image_loader", "params": {"imageSize": 32, "channels": "3"}},
            {"id": "c", "type": "conv2d", "params": {"inChannels": "auto", "outChannels": 8, "padding": "1"}},
            {"id": "d", "type": "dense", "params": {"inputSize": "auto", "outputSize": 2, "implicitPool": True}}]}
        p = subprocess.Popen([sys.executable, str(Path(__file__).parent / "service.py")], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(json.loads(p.stdout.readline())["type"], "ready")
            p.stdin.write("kein json\n" + json.dumps({"id": 7, "cmd": "check", "graph": raw}) + "\n")
            p.stdin.flush()
            self.assertIn("error", json.loads(p.stdout.readline()))
            r = json.loads(p.stdout.readline())
            self.assertEqual(r["id"], 7)
            self.assertEqual(r["result"]["nodes"]["c"]["outputs"]["out"], [1, 8, 32, 32])
            self.assertEqual(r["result"]["errors"], [])
        finally:
            p.stdin.close()
            p.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
