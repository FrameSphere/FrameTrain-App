"""Befunde aus der Synapse-Pruefung als echte Laeufe gegen das Canvas-Plugin.

Aufruf:  python3.11 test_regressions.py   (aus diesem Ordner)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent.parent
sys.path[:0] = [str(HERE), str(ENGINE), str(ENGINE.parent)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import plugin as P  # noqa: E402
from ir import parse_ir  # noqa: E402
from loader import load_canvas_model  # noqa: E402
from model_builder import build_model_from_graph  # noqa: E402

P.torch.backends.mps.is_available = lambda: False


def n(id, t, cat, **p):
    return {"id": id, "type": t, "category": cat, "params": p}


def e(a, b, sh=None, th=None):
    d = {"source": a, "target": b}
    if sh:
        d["sourceHandle"] = sh
    if th:
        d["targetHandle"] = th
    return d


def graph(nodes, edges, data=None, **tr):
    t = {"epochs": 3, "batchSize": 16, "learningRate": 0.01, "numClasses": 2,
         "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "cosine"}
    t.update(tr)
    return {"version": 1, "nodes": nodes, "edges": edges,
            "execution_order": [x["id"] for x in nodes], "training": t, "data": data}


def train(g, ds, out):
    cfg = types.SimpleNamespace(
        canvas_graph=g, canvas_model_code="", plugin_config={}, learning_rate=.01, weight_decay=0,
        optimizer="adamw", epochs=3, batch_size=16, fp16=False, bf16=False, dataset_path=ds,
        max_grad_norm=1.0, output_path=out)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        p = P.CanvasPlugin(cfg)
        ok = bool(p.setup() and p.train())
        if ok:
            p.export()
    if ok:
        (Path(out) / "graph_metadata.json").write_text(json.dumps({"graphIR": g}))
    return ok, p, buf.getvalue()


def model(g):
    return build_model_from_graph(parse_ir(g)).eval()


MLP_NODES = [
    n("c", "csv_loader", "data", normalize=True, targetCol="label"),
    n("d1", "dense", "layer", inputSize=4, outputSize=16),
    n("r", "relu", "activation"),
    n("d2", "dense", "layer", inputSize=16, outputSize=2),
    n("o", "output_node", "training", numClasses=2),
]
MLP_EDGES = [e("c", "d1"), e("d1", "r"), e("r", "d2"), e("d2", "o")]
CSV_DATA = {"type": "csv_loader", "params": {"normalize": True, "targetCol": "label"}}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        rs = np.random.RandomState(0)
        X = rs.normal(1000, 50, (400, 4))
        y = (X[:, 0] > 1000).astype(int)
        df = pd.DataFrame(X, columns=list("abcd"))
        df["label"] = np.where(y, "ja", "nein")
        cls.csv = os.path.join(cls.tmp, "csv")
        os.makedirs(cls.csv)
        df[:320].to_csv(cls.csv + "/train.csv", index=False)
        df[320:].to_csv(cls.csv + "/val.csv", index=False)
        cls.df = df
        reg = df.copy()
        reg["label"] = X[:, 0] / 1000.0
        cls.reg = os.path.join(cls.tmp, "reg")
        os.makedirs(cls.reg)
        reg.to_csv(cls.reg + "/d.csv", index=False)
        cls.img = os.path.join(cls.tmp, "img")
        for name, val in (("dunkel", 30), ("hell", 220)):
            os.makedirs(f"{cls.img}/{name}")
            for i in range(24):
                arr = rs.normal(val, 15, (32, 32, 3)).clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(f"{cls.img}/{name}/{i}.png")


class GraphSemantics(Base):
    def test_skip_vom_input_nutzt_die_eingabe(self):
        g = graph([n("x", "input", "data"), n("d", "dense", "layer", inputSize=4, outputSize=4),
                   n("a", "add_node", "math")],
                  [e("x", "d"), e("d", "a", th="a"), e("x", "a", th="b")])
        m = model(g)
        x = torch.ones(2, 4)
        self.assertTrue(torch.allclose(m(x), m.layers["d"](x) + x))

    def test_split_liefert_je_ausgang_eigene_teile(self):
        g = graph([n("x", "input", "data"), n("s", "split_node", "logic", chunks=2),
                   n("m", "add_node", "math")],
                  [e("x", "s"), e("s", "m", "a", "a"), e("s", "m", "b", "b")])
        x = torch.tensor([[1., 2., 10., 20.]])
        self.assertEqual(model(g)(x).tolist(), [[11., 22.]])

    def test_ports_entscheiden_nicht_die_kantenreihenfolge(self):
        g = graph([n("x", "input", "data"), n("s", "split_node", "logic", chunks=2),
                   n("m", "merge", "logic", dim=-1)],
                  [e("x", "s"), e("s", "m", "b", "b"), e("s", "m", "a", "a")])
        self.assertEqual(model(g)(torch.tensor([[1., 2.]])).tolist(), [[1., 2.]])

    def test_output_node_bestimmt_die_ausgabe(self):
        g = graph([n("x", "input", "data"), n("main", "dense", "layer", inputSize=4, outputSize=2),
                   n("o", "output_node", "training", numClasses=2),
                   n("side", "dense", "layer", inputSize=4, outputSize=7)],
                  [e("x", "main"), e("main", "o"), e("x", "side")])
        self.assertEqual(model(g)(torch.ones(2, 4)).shape[-1], 2)

    def test_attention_nutzt_query_key_value(self):
        g = graph([n("q", "input", "data"),
                   n("a", "transformer_block", "layer", embedDim=4, numHeads=1, ffnDim=8, dropout=0),
                   n("at", "attention", "layer", embedDim=4, numHeads=1, dropout=0)],
                  [e("q", "a"), e("q", "at", th="query"), e("a", "at", th="key"), e("a", "at", th="value")])
        m = model(g)
        x = torch.ones(2, 3, 4)
        before = m(x).clone()
        with torch.no_grad():
            for w in m.layers["a"].parameters():
                w.add_(1.0)
        self.assertFalse(torch.allclose(before, m(x)))


class Training(Base):
    def test_loss_arten_trainieren_oder_sagen_klar_warum_nicht(self):
        for loss in ("focal", "kl_div", "mse", "mae", "huber", "bce", "nll", "cross_entropy"):
            with self.subTest(loss=loss):
                g = graph(MLP_NODES, MLP_EDGES, CSV_DATA, loss=loss)
                ok, p, log = train(g, self.csv, f"{self.tmp}/loss_{loss}")
                self.assertTrue(ok, log[-400:])
                self.assertEqual(p.ir.training.loss, loss)
                self.assertEqual(type(p._make_loss()).__name__.lower().replace("loss", ""),
                                 {"focal": "focal", "kl_div": "kldiv", "mse": "onehot", "mae": "onehot",
                                  "huber": "onehot", "bce": "onehot", "nll": "nll",
                                  "cross_entropy": "crossentropy"}[loss])

    def test_optimizer_wird_nicht_still_ersetzt(self):
        g = graph(MLP_NODES, MLP_EDGES, CSV_DATA, optimizer="adagrad")
        ok, p, log = train(g, self.csv, f"{self.tmp}/opt")
        self.assertTrue(ok, log[-300:])
        self.assertEqual(type(p._optimizer).__name__, "Adagrad")

    def test_geraet_folgt_der_auswahl(self):
        g = graph(MLP_NODES, MLP_EDGES, CSV_DATA, gpu="cpu", gpuExplicit=True)
        ok, p, _ = train(g, self.csv, f"{self.tmp}/dev")
        self.assertTrue(ok)
        self.assertEqual(p.device, "cpu")

    def test_regression_nutzt_normalize(self):
        nodes = [n("c", "csv_loader", "data", normalize=True, targetCol="label"),
                 n("d", "dense", "layer", inputSize=4, outputSize=1)]
        g = graph(nodes, [e("c", "d")], CSV_DATA, loss="mse", taskType="regression",
                  numClasses=1, learningRate=0.01, epochs=20)
        ok, p, log = train(g, self.reg, f"{self.tmp}/reg_m")
        self.assertTrue(ok, log[-300:])
        self.assertLess(p._final["final_val_loss"], 1.0)

    def test_bestes_modell_wird_gespeichert(self):
        g = graph(MLP_NODES, MLP_EDGES, CSV_DATA, epochs=4)
        ok, p, _ = train(g, self.csv, f"{self.tmp}/best")
        self.assertTrue(ok)
        self.assertTrue((Path(self.tmp) / "best" / "model_best.pt").exists())


class Inference(Base):
    def test_csv_vorhersage_nutzt_die_normalisierung_und_klassennamen(self):
        g = graph(MLP_NODES, MLP_EDGES, CSV_DATA, epochs=30)
        out = f"{self.tmp}/inf"
        ok, p, log = train(g, self.csv, out)
        self.assertTrue(ok, log[-300:])
        lm = load_canvas_model(out, device="cpu")
        rows = self.df[320:]
        names = [lm.predict(list(map(float, r)))["top_predictions"][0]["label"]
                 for r in rows[list("abcd")].values]
        self.assertGreater(np.mean(np.array(names) == rows.label.values), 0.9)

    def test_auto_groessen_trainieren_und_laden_wieder(self):
        nodes = [n("i", "image_loader", "data", imageSize=32, channels="3", normalize=True),
                 n("c1", "conv2d", "layer", inChannels="auto", outChannels=8, kernelSize="3", padding="1"),
                 n("b", "batchnorm", "layer", numFeatures="auto"),
                 n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "image_loader", "params": {"imageSize": 32, "channels": "3", "normalize": True}}
        g = graph(nodes, [e("i", "c1"), e("c1", "b"), e("b", "d")], data, epochs=6)
        out = f"{self.tmp}/auto"
        ok, p, log = train(g, self.img, out)
        self.assertTrue(ok, log[-400:])
        self.assertEqual(p.model.layers["d"].in_features, 8)
        r = load_canvas_model(out, device="cpu").predict_image(f"{self.img}/hell/0.png")
        self.assertEqual(r["top_predictions"][0]["label"], "hell")

    def test_ueberzaehlige_ausgaenge_erscheinen_nicht(self):
        g = graph(MLP_NODES[:3] + [n("d2", "dense", "layer", inputSize=16, outputSize=6), MLP_NODES[4]], MLP_EDGES, CSV_DATA,
                  epochs=5, numClasses=6)
        out = f"{self.tmp}/extra"
        ok, _, log = train(g, self.csv, out)
        self.assertTrue(ok, log[-300:])
        self.assertIn("Das Dataset hat 2 Klassen", log)
        r = load_canvas_model(out, device="cpu").predict([1000.0, 1000.0, 1000.0, 1000.0])
        self.assertEqual({t["label"] for t in r["top_predictions"]}, {"ja", "nein"})

    def test_bild_klassen_haben_namen(self):
        nodes = [n("i", "image_loader", "data", imageSize=32, channels="3", normalize=True),
                 n("c1", "conv2d", "layer", inChannels=3, outChannels=8, kernelSize="3", padding="1"),
                 n("d", "dense", "layer", inputSize=8, outputSize=2)]
        data = {"type": "image_loader", "params": {"imageSize": 32, "channels": "3", "normalize": True}}
        g = graph(nodes, [e("i", "c1"), e("c1", "d")], data, epochs=6)
        out = f"{self.tmp}/imgm"
        ok, p, log = train(g, self.img, out)
        self.assertTrue(ok, log[-300:])
        r = load_canvas_model(out, device="cpu").predict_image(f"{self.img}/hell/0.png")
        self.assertEqual(r["top_predictions"][0]["label"], "hell")

    def test_graustufen_werden_wie_im_training_normalisiert(self):
        nodes = [n("i", "image_loader", "data", imageSize=32, channels="1", normalize=True),
                 n("c1", "conv2d", "layer", inChannels=1, outChannels=4, kernelSize="3", padding="1"),
                 n("d", "dense", "layer", inputSize=4, outputSize=2)]
        data = {"type": "image_loader", "params": {"imageSize": 32, "channels": "1", "normalize": True}}
        g = graph(nodes, [e("i", "c1"), e("c1", "d")], data, epochs=6)
        out = f"{self.tmp}/gray"
        ok, p, log = train(g, self.img, out)
        self.assertTrue(ok, log[-300:])
        lm = load_canvas_model(out, device="cpu")
        seen = {}
        orig = lm.predict
        lm.predict = lambda t: seen.setdefault("x", t) is not None and orig(t)
        lm.predict_image(f"{self.img}/dunkel/0.png")
        self.assertLess(seen["x"].mean().item(), -0.5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
