"""Code-Node, Block, vortrainierter Baustein und Export.

Aufruf:  python3.11 -m unittest synapse.test_export   (aus src-tauri/python)
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from synapse.check import check_graph
from synapse.export import export_onnx, export_pytorch, export_safetensors, export_torchscript, model_source
from synapse.graph import ENTRY, GEdge, GNode, Graph, migrate_v1
from synapse.runtime import SynapseModule
from synapse.test_synapse import chain


def run_export(module, shape, dtype=torch.float32):
    out = Path(tempfile.mkdtemp())
    export_pytorch(module, {}, out, shape)
    spec = importlib.util.spec_from_file_location(f"m{abs(hash(str(out)))}", out / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exported = mod.Model()
    exported.load_state_dict(torch.load(out / "model.pt"))
    return exported.eval(), module.eval()


CHAINS = {
    "bild": ((1, 3, 16, 16), [("conv2d", {"outChannels": 4, "padding": "1"}), ("batchnorm", {}), ("relu", {}),
                              ("pool", {"type": "max_2d"}), ("conv_transpose2d", {"outChannels": 4}), ("upsample", {}),
                              ("flatten", {}), ("dropout", {"p": 0.5}), ("dense", {"outputSize": 3}), ("softmax", {})]),
    "audio": ((1, 1, 64), [("conv1d", {"outChannels": 4}), ("gelu", {}), ("pool1d", {"type": "avg"}),
                           ("pool1d", {"type": "global_max"}), ("dense", {"outputSize": 2})]),
    "video": ((1, 3, 4, 16, 16), [("conv3d", {"outChannels": 4}), ("tanh", {}), ("pool3d", {"type": "global_avg"}),
                                  ("dense", {"outputSize": 2})]),
    "text": ((1, 12), [("embedding", {"vocabSize": 50, "embeddingDim": 8}), ("positional", {"maxLength": 12}),
                       ("transformer_block", {"numHeads": 2, "ffnDim": 16, "dropout": 0, "causal": True}),
                       ("layernorm", {}), ("lstm", {"hiddenSize": 8, "numLayers": 1, "returnSequence": True}),
                       ("gru", {"hiddenSize": 8, "bidirectional": True}), ("dense", {"outputSize": 2})]),
    "attention": ((1, 6, 8), [("attention", {"numHeads": 2, "dropout": 0}), ("transpose", {}), ("reshape", {"shape": "-1, 48"}),
                              ("normalize", {}), ("leaky_relu", {}), ("silu", {}), ("sigmoid", {})]),
}


class CodeAndBlocks(unittest.TestCase):
    def test_code_node_lernt_und_wird_geprueft(self):
        code = """class Node(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.w = nn.Parameter(torch.ones(shape[-1]))

    def forward(self, x):
        return x * self.w + 1
"""
        g = chain(("code", {"code": code}), ("dense", {"outputSize": 2}))
        self.assertEqual(check_graph(g, (1, 4))["params"], 4 + 10)
        m = SynapseModule(g, input_shape=(1, 4))
        m(torch.ones(2, 4)).sum().backward()
        self.assertIsNotNone(m.layers["n0"].w.grad)

    def test_fehler_im_code_steht_am_node(self):
        r = check_graph(chain(("code", {"code": "class Falsch: pass"})), (1, 4))
        self.assertIn("Node(nn.Module)", r["errors"][0]["message"])

    def test_neuron_ebene_mit_eigener_formel(self):
        code = """class Node(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.w = nn.Parameter(torch.randn(shape[-1]))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid((x * self.w).sum(-1, keepdim=True) + self.b)
"""
        m = SynapseModule(chain(("code", {"code": code})), input_shape=(1, 2))
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [0.], [0.], [1.]])
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        for _ in range(800):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy(m(x), y).backward()
            opt.step()
        self.assertTrue(((m(x) > 0.5).float() == y).all())

    def test_block_wiederholt_mit_eigenen_gewichten(self):
        block = {"nodes": [{"id": "a", "type": "dense", "params": {"outputSize": 8, "implicitPool": False}},
                           {"id": "r", "type": "relu", "params": {}}], "edges": [{"source": "a", "target": "r"}]}
        g = chain(("block", {"graph": block, "repeat": 3}), ("dense", {"outputSize": 2}))
        r = check_graph(g, (1, 4))
        self.assertTrue(r["ok"], r["errors"])
        self.assertEqual(r["params"], (4 * 8 + 8) + 2 * (8 * 8 + 8) + (8 * 2 + 2))

    def test_vortrainierter_baustein_aus_lokalem_ordner(self):
        from transformers import BertConfig, BertModel
        path = tempfile.mkdtemp()
        BertModel(BertConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                             vocab_size=50)).save_pretrained(path)
        g = chain(("pretrained", {"path": path}), ("pool", {}), ("dense", {"outputSize": 2}))
        m = SynapseModule(g, input_shape=(1, 6), input_dtype=torch.long)
        self.assertFalse(any(p.requires_grad for p in m.layers["n0"].parameters()))
        self.assertTrue(any(p.requires_grad for p in m.layers["n2"].parameters()))
        self.assertEqual(m(torch.ones(2, 6, dtype=torch.long)).shape, (2, 2))


class Export(unittest.TestCase):
    def test_vortrainiert_im_pytorch_export(self):
        from transformers import BertConfig, BertModel
        path = tempfile.mkdtemp()
        BertModel(BertConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                             vocab_size=50)).save_pretrained(path)
        m = SynapseModule(chain(("pretrained", {"path": path}), ("pool", {}), ("dense", {"outputSize": 2})),
                          input_shape=(1, 6), input_dtype=torch.long)
        exported, runtime = run_export(m, (1, 6))
        x = torch.ones(2, 6, dtype=torch.long)
        self.assertLess((exported(x) - runtime(x)).abs().max().item(), 1e-5)

    def test_export_rechnet_wie_die_laufzeit(self):
        for name, (shape, items) in CHAINS.items():
            with self.subTest(chain=name):
                dtype = torch.long if name == "text" else torch.float32
                m = SynapseModule(chain(*items), input_shape=shape, input_dtype=dtype)
                exported, runtime = run_export(m, shape)
                x = torch.randint(0, 50, (2, *shape[1:])) if name == "text" else torch.randn(2, *shape[1:])
                self.assertLess((exported(x) - runtime(x)).abs().max().item(), 1e-5)

    def test_skip_split_und_ports_im_export(self):
        nodes = [{"id": "x", "type": "input"}, {"id": "s", "type": "split_node", "params": {"chunks": 2}},
                 {"id": "d", "type": "dense", "params": {"inputSize": "auto", "outputSize": 2, "implicitPool": False}},
                 {"id": "a", "type": "add_node"}, {"id": "m", "type": "merge", "params": {"dim": -1}}]
        edges = [{"source": "x", "target": "s"}, {"source": "s", "sourceHandle": "a", "target": "d"},
                 {"source": "d", "target": "a", "targetHandle": "a"}, {"source": "s", "sourceHandle": "b", "target": "a", "targetHandle": "b"},
                 {"source": "a", "target": "m", "targetHandle": "a"}, {"source": "x", "target": "m", "targetHandle": "b"}]
        m = SynapseModule(migrate_v1(nodes, edges), input_shape=(1, 4))
        exported, runtime = run_export(m, (1, 4))
        x = torch.randn(3, 4)
        self.assertLess((exported(x) - runtime(x)).abs().max().item(), 1e-5)

    def test_block_im_export(self):
        block = {"nodes": [{"id": "a", "type": "dense", "params": {"outputSize": 8, "implicitPool": False}},
                           {"id": "r", "type": "relu", "params": {}}], "edges": [{"source": "a", "target": "r"}]}
        m = SynapseModule(chain(("block", {"graph": block, "repeat": 3}), ("dense", {"outputSize": 2})), input_shape=(1, 4))
        exported, runtime = run_export(m, (1, 4))
        x = torch.randn(2, 4)
        self.assertLess((exported(x) - runtime(x)).abs().max().item(), 1e-5)

    def test_code_node_im_export(self):
        code = "class Node(nn.Module):\n    def __init__(self, shape):\n        super().__init__()\n        self.s = nn.Parameter(torch.ones(1) * 2)\n\n    def forward(self, x):\n        return x * self.s\n"
        m = SynapseModule(chain(("code", {"code": code.replace("torch.ones(1)", "torch.ones(shape[-1])")})), input_shape=(1, 4))
        exported, runtime = run_export(m, (1, 4))
        self.assertTrue(torch.equal(exported(torch.ones(1, 4)), runtime(torch.ones(1, 4))))

    def test_export_laeuft_ohne_frametrain(self):
        m = SynapseModule(chain(("dense", {"outputSize": 8}), ("relu", {}), ("dense", {"outputSize": 2})), input_shape=(1, 4))
        out = tempfile.mkdtemp()
        export_pytorch(m, {"class_names": ["a", "b"]}, out, (1, 4))
        run = subprocess.run([sys.executable, "-I", "-c",
                              "import sys;sys.path.insert(0,'.');import synapse" if False else
                              "import sys;sys.path.insert(0,'.');import train;import importlib.util as u;"
                              "assert u.find_spec('synapse') is None;print('ok')"],
                             cwd=out, capture_output=True, text=True)
        self.assertEqual(run.stdout.strip(), "ok", run.stderr[-400:])

    def test_torchscript_und_safetensors(self):
        m = SynapseModule(chain(("dense", {"outputSize": 2})), input_shape=(1, 4)).eval()
        d = Path(tempfile.mkdtemp())
        ts = torch.jit.load(str(export_torchscript(m, {}, d / "m.pt", (1, 4))))
        x = torch.randn(1, 4)
        self.assertTrue(torch.allclose(ts(x), m(x)))
        self.assertTrue(export_safetensors(m, d / "m.safetensors").exists())

    def test_onnx_export_ist_gueltig(self):
        if not importlib.util.find_spec("onnx"):
            self.skipTest("onnx fehlt")
        import onnx
        m = SynapseModule(chain(("conv2d", {"outChannels": 4, "padding": "1"}), ("relu", {}), ("pool", {}),
                                ("dense", {"outputSize": 2})), input_shape=(1, 3, 16, 16)).eval()
        path = Path(tempfile.mkdtemp()) / "m.onnx"
        export_onnx(m, {"input_shape": [1, 3, 16, 16]}, path)
        onnx.checker.check_model(onnx.load(str(path)))

    def test_onnx_ohne_paket_sagt_was_fehlt(self):
        if importlib.util.find_spec("onnx"):
            self.skipTest("onnx ist installiert")
        m = SynapseModule(chain(("dense", {"outputSize": 2})), input_shape=(1, 4))
        with self.assertRaisesRegex(ImportError, "onnx"):
            export_onnx(m, {}, "x.onnx", (1, 4))


class NeuronExport(unittest.TestCase):
    def test_export_rechnet_gleich_und_braucht_kein_torch(self):
        import json as js
        items = [("conv2d", {"outChannels": 4, "padding": "1"}), ("relu", {}), ("flatten", {})]
        m = SynapseModule(chain(("dense", {"outputSize": 8}), ("layernorm", {}), ("gelu", {}), ("dense", {"outputSize": 2})),
                          input_shape=(1, 4)).eval()
        from synapse.export import export_neuron
        out = export_neuron(m, {}, tempfile.mkdtemp())
        x = torch.randn(3, 4, dtype=torch.float64)
        code = ("import sys,json;sys.path.insert(0,'.');import numpy as np;from model import model,ne;"
                f"x=np.array({x.tolist()});print(json.dumps(model(ne.Tensor(x)).data.tolist()));"
                "assert 'torch' not in sys.modules")
        r = subprocess.run([sys.executable, "-I", "-c", code], cwd=out, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        got = torch.tensor(js.loads(r.stdout), dtype=torch.float32)
        self.assertLess((got - m(x.float())).abs().max().item(), 1e-4)

    def test_verzweigung_wird_klar_abgelehnt(self):
        from synapse.export import export_neuron
        nodes = [{"id": "s", "type": "split_node"}, {"id": "a", "type": "add_node"}]
        edges = [{"source": "s", "sourceHandle": "a", "target": "a", "targetHandle": "a"},
                 {"source": "s", "sourceHandle": "b", "target": "a", "targetHandle": "b"}]
        with self.assertRaisesRegex(ValueError, "Node"):
            export_neuron(SynapseModule(migrate_v1(nodes, edges), input_shape=(1, 4)), {}, tempfile.mkdtemp())


class Blueprints(unittest.TestCase):
    def test_jede_vorlage_baut_und_ist_differenzierbar(self):
        from synapse.check import check_raw, guess_input
        import json as js
        data = js.loads((Path(__file__).resolve().parents[3] / "src/components/synapse/blueprints.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data), 12)
        for bp in data:
            with self.subTest(vorlage=bp["id"]):
                raw = {"version": 1, "nodes": bp["nodes"], "execution_order": [n["id"] for n in bp["nodes"]],
                       "edges": [{"source": e[0], "target": e[1], "targetHandle": e[3] if len(e) > 3 else None} for e in bp["edges"]]}
                for n in raw["nodes"]:
                    n["params"] = {**n["params"], **({"implicitPool": False} if n["type"] == "dense" else {})}
                r = check_raw(raw)
                self.assertTrue(r["ok"], (r["errors"], r["input_shape"]))
                self.assertGreater(r["params"], 0)
                if bp["id"] == "gan":
                    continue
                shape, dtype = guess_input(raw)
                m = SynapseModule(migrate_v1(raw["nodes"], raw["edges"], raw["execution_order"]), input_shape=shape,
                                  input_dtype=getattr(torch, dtype))
                if isinstance(shape, dict):
                    kind = getattr(torch, dtype)
                    x = (torch.ones(2, *shape["out"][1:], dtype=kind), torch.ones(2, *shape["decoder"][1:], dtype=kind))
                elif dtype == "int64":
                    x = torch.ones(2, *shape[1:], dtype=torch.long)
                else:
                    x = torch.randn(2, *shape[1:])
                m(x).sum().backward()


if __name__ == "__main__":
    unittest.main()
