"""Mini-GPT, Segmentierung und VAE lernen messbar.

Aufruf:  python3.11 test_recipes.py   (aus diesem Ordner)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from test_regressions import e, graph, load_canvas_model, n, train


def fit(case, g, ds, out):
    torch.manual_seed(0)
    ok, p, log = train(g, ds, out)
    case.assertTrue(ok, [l[:300] for l in log.splitlines() if '"error"' in l[:30]])
    return p


class Recipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        rs = np.random.RandomState(0)
        cls.text = f"{cls.tmp}/text"
        os.makedirs(cls.text)
        Path(cls.text, "buch.txt").write_text("die katze sitzt auf der matte. " * 300, encoding="utf-8")
        cls.seg = f"{cls.tmp}/seg"
        os.makedirs(f"{cls.seg}/images")
        os.makedirs(f"{cls.seg}/masks")
        for i in range(40):
            mask = np.zeros((32, 32), dtype="uint8")
            x, y = rs.randint(2, 18, 2)
            mask[y:y + 12, x:x + 12] = 1
            img = rs.normal(40, 10, (32, 32, 3)) + mask[..., None] * 170
            Image.fromarray(img.clip(0, 255).astype("uint8")).save(f"{cls.seg}/images/{i}.png")
            Image.fromarray(mask).save(f"{cls.seg}/masks/{i}.png")
        cls.img = f"{cls.tmp}/img"
        for c in ("a", "b"):
            os.makedirs(f"{cls.img}/{c}")
        for i in range(60):
            v = rs.randint(40, 220)
            Image.fromarray(np.full((16, 16, 3), v, "uint8")).save(f"{cls.img}/{'ab'[i % 2]}/{i}.png")

    def test_mini_gpt_lernt_und_schreibt_weiter(self):
        tok = n("t", "tokenizer", "data", level="char", maxLength=32, vocabSize=64)
        block = dict(embedDim="auto", numHeads=2, ffnDim=64, dropout=0, causal=True)
        nodes = [n("x", "text_loader", "data"), tok, n("em", "embedding", "layer", vocabSize=64, embeddingDim=32),
                 n("pos", "positional", "layer", maxLength=32, dim="auto"),
                 n("b1", "transformer_block", "layer", **block), n("b2", "transformer_block", "layer", **block),
                 n("head", "dense", "layer", inputSize="auto", outputSize=64, implicitPool=False)]
        data = {"type": "text_loader", "params": {}, "pipeline": [{"type": "tokenizer", "params": tok["params"]}]}
        chain = ["x", "t", "em", "pos", "b1", "b2", "head"]
        g = graph(nodes, [e(a, b) for a, b in zip(chain, chain[1:])], data, epochs=12, taskType="language_model",
                  learningRate=0.003, numClasses=64)
        out = f"{self.tmp}/gpt"
        p = fit(self, g, self.text, out)
        self.assertGreater(p._final["accuracy"], 0.9)
        text = load_canvas_model(out, device="cpu").predict_text("die katze ")["predicted"]
        self.assertIn("matte", text)

    def test_segmentierung_mit_unet_skip(self):
        nodes = [n("m", "mask_loader", "data", imageSize=32),
                 n("c1", "conv2d", "layer", inChannels="auto", outChannels=8, padding="1"), n("r1", "relu", "activation"),
                 n("p", "pool", "logic", type="max_2d", stride=2),
                 n("c2", "conv2d", "layer", inChannels="auto", outChannels=16, padding="1"), n("r2", "relu", "activation"),
                 n("up", "conv_transpose2d", "layer", inChannels="auto", outChannels=8),
                 n("cat", "merge", "logic", dim=1),
                 n("out", "conv2d", "layer", inChannels="auto", outChannels=2, kernelSize="1", padding="0")]
        data = {"type": "mask_loader", "params": {"imageSize": 32}}
        edges = [e("m", "c1"), e("c1", "r1"), e("r1", "p"), e("p", "c2"), e("c2", "r2"), e("r2", "up"),
                 e("up", "cat", th="a"), e("r1", "cat", th="b"), e("cat", "out")]
        g = graph(nodes, edges, data, epochs=25, taskType="segmentation", learningRate=0.01, batchSize=8)
        p = fit(self, g, self.seg, f"{self.tmp}/seg_m")
        self.assertGreater(p._final["accuracy"], 0.97)

    def test_vae_rekonstruiert_und_nutzt_kl(self):
        nodes = [n("i", "image_loader", "data", imageSize=16, channels="3", normalize=False),
                 n("f", "flatten", "math"), n("enc", "dense", "layer", inputSize="auto", outputSize=32, implicitPool=False),
                 n("r", "relu", "activation"), n("mu", "dense", "layer", inputSize="auto", outputSize=4, implicitPool=False),
                 n("lv", "dense", "layer", inputSize="auto", outputSize=4, implicitPool=False),
                 n("z", "reparameterize", "layer", beta=0.001),
                 n("dec", "dense", "layer", inputSize="auto", outputSize=768, implicitPool=False),
                 n("s", "sigmoid", "activation"), n("back", "reshape", "math", shape="-1, 3, 16, 16")]
        data = {"type": "image_loader", "params": {"imageSize": 16, "channels": "3", "normalize": False}}
        edges = [e("i", "f"), e("f", "enc"), e("enc", "r"), e("r", "mu"), e("r", "lv"),
                 e("mu", "z", th="mu"), e("lv", "z", th="logvar"), e("z", "dec", "out"), e("dec", "s"), e("s", "back")]
        g = graph(nodes, edges, data, epochs=120, taskType="autoencoder", loss="mse", learningRate=0.005, batchSize=8)
        p = fit(self, g, self.img, f"{self.tmp}/vae")
        self.assertLess(p._final["final_val_loss"], 0.02)
        self.assertIsNotNone(p.model.aux_loss)

    def test_seq2seq_uebersetzt_mit_zwei_eingaengen(self):
        import pandas as pd
        words = {"eins": "one", "zwei": "two", "drei": "three", "vier": "four"}
        rs = np.random.RandomState(1)
        rows = []
        for _ in range(240):
            src = list(rs.choice(list(words), 3))
            rows.append({"source": " ".join(src), "target": " ".join(words[w] for w in src)})
        ds = f"{self.tmp}/s2s"
        os.makedirs(ds)
        pd.DataFrame(rows).to_csv(f"{ds}/data.csv", index=False)
        tok = n("t", "tokenizer", "data", maxLength=6, vocabSize=32)
        emb = dict(vocabSize=32, embeddingDim=32)
        block = dict(embedDim="auto", numHeads=2, ffnDim=64, dropout=0)
        nodes = [n("x", "seq2seq_loader", "data"), tok, n("dec", "decoder_input", "data"),
                 n("e1", "embedding", "layer", **emb), n("p1", "positional", "layer", maxLength=6, dim="auto"),
                 n("enc", "transformer_block", "layer", **block),
                 n("e2", "embedding", "layer", **emb), n("p2", "positional", "layer", maxLength=6, dim="auto"),
                 n("d", "transformer_decoder", "layer", **block),
                 n("head", "dense", "layer", inputSize="auto", outputSize=32, implicitPool=False)]
        edges = [e("x", "t"), e("t", "e1"), e("e1", "p1"), e("p1", "enc"), e("dec", "e2"), e("e2", "p2"),
                 e("p2", "d", th="in"), e("enc", "d", th="memory"), e("d", "head")]
        data = {"type": "seq2seq_loader", "params": {}, "pipeline": [{"type": "tokenizer", "params": tok["params"]}]}
        g = graph(nodes, edges, data, epochs=40, taskType="seq2seq", learningRate=0.003, numClasses=32, batchSize=32)
        out = f"{self.tmp}/s2s_m"
        p = fit(self, g, ds, out)
        self.assertGreater(p._final["accuracy"], 0.9)
        self.assertEqual(load_canvas_model(out, device="cpu").predict_text("zwei vier eins")["predicted"], "two four one")

    def small_images(self):
        ds = f"{self.tmp}/hell"
        os.makedirs(f"{ds}/a", exist_ok=True)
        rs = np.random.RandomState(3)
        for i in range(64):
            Image.fromarray(rs.normal(200, 8, (8, 8, 3)).clip(0, 255).astype("uint8")).save(f"{ds}/a/{i}.png")
        return ds

    def test_gan_erzeugt_bilder_wie_die_daten(self):
        def dense(i, size):
            return {"id": i, "type": "dense", "params": {"outputSize": size, "implicitPool": False}}
        gen = {"nodes": [dense("g1", 64), {"id": "r", "type": "relu"}, dense("g2", 192), {"id": "s", "type": "sigmoid"},
                         {"id": "b", "type": "reshape", "params": {"shape": "-1, 3, 8, 8"}}],
               "edges": [{"source": a, "target": b} for a, b in [("g1", "r"), ("r", "g2"), ("g2", "s"), ("s", "b")]]}
        disc = {"nodes": [{"id": "f", "type": "flatten"}, dense("d1", 64), {"id": "r", "type": "relu"}, dense("d2", 1)],
                "edges": [{"source": a, "target": b} for a, b in [("f", "d1"), ("d1", "r"), ("r", "d2")]]}
        nodes = [n("i", "image_loader", "data", imageSize=8, channels="3", normalize=False),
                 n("z", "latent", "data", dim=8), n("g", "block", "logic", graph=gen, role="generator"),
                 n("d", "block", "logic", graph=disc, role="discriminator")]
        data = {"type": "image_loader", "params": {"imageSize": 8, "channels": "3", "normalize": False}}
        g = graph(nodes, [], data, epochs=500, taskType="gan", optimizer="adam", learningRate=0.0005, numClasses=2, batchSize=16)
        out = f"{self.tmp}/gan"
        fit(self, g, self.small_images(), out)
        fake = load_canvas_model(out, device="cpu").generate_images(16)
        self.assertEqual(tuple(fake.shape), (16, 3, 8, 8))
        self.assertLess(abs(fake.mean().item() - 200 / 255), 0.12)

    def test_diffusion_lernt_das_rauschen_und_erzeugt_bilder(self):
        nodes = [n("i", "image_loader", "data", imageSize=8, channels="3", normalize=False), n("t", "timestep", "data"),
                 n("f", "flatten", "math"), n("tf", "dense", "layer", inputSize="auto", outputSize=512, implicitPool=False),
                 n("d1", "dense", "layer", inputSize="auto", outputSize=512, implicitPool=False),
                 n("m", "add_node", "math"), n("r", "gelu", "activation"),
                 n("d2", "dense", "layer", inputSize="auto", outputSize=512, implicitPool=False), n("r2", "gelu", "activation"),
                 n("d3", "dense", "layer", inputSize="auto", outputSize=192, implicitPool=False),
                 n("b", "reshape", "math", shape="-1, 3, 8, 8")]
        edges = [e("i", "f"), e("f", "d1"), e("t", "tf"), e("d1", "m", th="a"), e("tf", "m", th="b"), e("m", "r"),
                 e("r", "d2"), e("d2", "r2"), e("r2", "d3"), e("d3", "b")]
        data = {"type": "image_loader", "params": {"imageSize": 8, "channels": "3", "normalize": False}}
        g = graph(nodes, edges, data, epochs=500, taskType="diffusion", loss="mse", learningRate=0.002, numClasses=2, batchSize=16)
        out = f"{self.tmp}/dif"
        p = fit(self, g, self.small_images(), out)
        self.assertLess(p._final["final_train_loss"], 0.2)
        made = load_canvas_model(out, device="cpu").generate_images(16)
        self.assertLess(abs(made.mean().item() - 200 / 255), 0.2)

    def test_detektion_findet_die_box(self):
        ds = f"{self.tmp}/det_ds"
        os.makedirs(f"{ds}/images")
        os.makedirs(f"{ds}/labels")
        rs = np.random.RandomState(0)
        truth = {}
        for i in range(120):
            img = rs.normal(30, 8, (64, 64, 3))
            x, y = rs.randint(4, 44, 2)
            img[y:y + 16, x:x + 16] = 230
            Image.fromarray(img.clip(0, 255).astype("uint8")).save(f"{ds}/images/{i}.png")
            truth[i] = [(x + 8) / 64, (y + 8) / 64, 16 / 64, 16 / 64]
            Path(f"{ds}/labels/{i}.txt").write_text("0 " + " ".join(map(str, truth[i])))

        def c(i, ch):
            return n(i, "conv2d", "layer", inChannels="auto", outChannels=ch, padding="1")
        nodes = [n("d", "detection_loader", "data", imageSize=64, gridSize=8), c("c1", 16), n("r1", "relu", "activation"),
                 n("p1", "pool", "logic", type="max_2d", stride=2), c("c2", 32), n("r2", "relu", "activation"),
                 n("p2", "pool", "logic", type="max_2d", stride=2), c("c3", 32), n("r3", "relu", "activation"),
                 n("p3", "pool", "logic", type="max_2d", stride=2),
                 n("h", "conv2d", "layer", inChannels="auto", outChannels=6, kernelSize="1", padding="0")]
        chain = [x["id"] for x in nodes]
        data = {"type": "detection_loader", "params": {"imageSize": 64, "gridSize": 8}}
        g = graph(nodes, [e(a, b) for a, b in zip(chain, chain[1:])], data, epochs=40, taskType="detection",
                  numClasses=1, learningRate=0.003, batchSize=16)
        out = f"{self.tmp}/det"
        fit(self, g, ds, out)
        from synapse.generative import iou
        m = load_canvas_model(out, device="cpu")
        scores = []
        for i in range(10):
            boxes = m.predict_file(f"{ds}/images/{i}.png")["boxes"]
            scores.append(iou(boxes[0]["box"], truth[i]) if boxes else 0)
        self.assertGreater(np.mean(scores), 0.5)

    def test_unbekannte_aufgabe_wird_klar_abgelehnt(self):
        nodes = [n("c", "csv_loader", "data"), n("d", "dense", "layer", inputSize=4, outputSize=2)]
        g = graph(nodes, [e("c", "d")], {"type": "csv_loader", "params": {}}, taskType="tracking")
        ok, p, log = train(g, self.text, f"{self.tmp}/det")
        self.assertFalse(ok)
        self.assertIn("noch nicht", log)


if __name__ == "__main__":
    unittest.main(verbosity=1)
