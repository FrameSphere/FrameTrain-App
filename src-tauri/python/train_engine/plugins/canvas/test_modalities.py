"""Je Datenart ein Lauf von Dateien bis Vorhersage, dazu Pipeline-Wirkung.

Aufruf:  python3.11 test_modalities.py   (aus diesem Ordner)
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf
import cv2
import torch

from test_regressions import e, graph, load_canvas_model, n, train
from dataloaders import get_dataloaders
from ir import parse_ir


def fit(self_case, g, ds, out):
    torch.manual_seed(0)
    ok, p, log = train(g, ds, out)
    self_case.assertTrue(ok, log[-500:])
    return p, load_canvas_model(out, device="cpu")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        rs = np.random.RandomState(0)
        cls.audio = f"{cls.tmp}/audio"
        for name, hz in (("hoch", 3000), ("tief", 300)):
            os.makedirs(f"{cls.audio}/{name}")
            for i in range(24):
                t = np.arange(8000) / 8000
                sf.write(f"{cls.audio}/{name}/{i}.wav", np.sin(2 * np.pi * hz * t + i) * 0.5 + rs.normal(0, .02, 8000), 8000)
        cls.video = f"{cls.tmp}/video"
        for name, val in (("hell", 220), ("dunkel", 30)):
            os.makedirs(f"{cls.video}/{name}")
            for i in range(16):
                w = cv2.VideoWriter(f"{cls.video}/{name}/{i}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
                for _ in range(10):
                    w.write(rs.normal(val, 10, (32, 32, 3)).clip(0, 255).astype("uint8"))
                w.release()
        cls.text = f"{cls.tmp}/text"
        os.makedirs(cls.text)
        good, bad = "super toll genial schoen prima".split(), "schlecht furchtbar mies grausam kaputt".split()
        rows = [{"text": " ".join(rs.choice(w, 6)), "label": lab} for _ in range(100)
                for w, lab in ((good, "gut"), (bad, "boese"))]
        pd.DataFrame(rows).to_csv(f"{cls.text}/data.csv", index=False)


class EndToEnd(Base):
    def test_audio_wellenform_mit_conv1d(self):
        nodes = [n("a", "audio_loader", "data", sampleRate=8000, seconds=1.0),
                 n("c", "conv1d", "layer", inChannels="auto", outChannels=8, kernelSize=9, stride=2, padding=4),
                 n("r", "relu", "activation"), n("p", "pool1d", "logic", type="global_max"),
                 n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "audio_loader", "params": {"sampleRate": 8000, "seconds": 1.0}}
        g = graph(nodes, [e("a", "c"), e("c", "r"), e("r", "p"), e("p", "d")], data, epochs=25)
        _, m = fit(self, g, self.audio, f"{self.tmp}/m_audio")
        self.assertEqual(m.predict_file(f"{self.audio}/hoch/0.wav")["predicted_label"], "hoch")
        self.assertEqual(m.predict_file(f"{self.audio}/tief/0.wav")["predicted_label"], "tief")

    def test_audio_mel_spektrogramm_mit_conv2d(self):
        mel = n("mel", "mel_spectrogram", "data", nMels=32, nFft=256, hopLength=128)
        nodes = [n("a", "audio_loader", "data", sampleRate=8000, seconds=1.0), mel,
                 n("c", "conv2d", "layer", inChannels="auto", outChannels=8, kernelSize="3", padding="1"),
                 n("p", "pool", "logic", type="global_avg"), n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "audio_loader", "params": {"sampleRate": 8000, "seconds": 1.0},
                "pipeline": [{"type": "mel_spectrogram", "params": mel["params"]}]}
        g = graph(nodes, [e("a", "mel"), e("mel", "c"), e("c", "p"), e("p", "d")], data, epochs=25)
        _, m = fit(self, g, self.audio, f"{self.tmp}/m_mel")
        self.assertEqual(m.predict_file(f"{self.audio}/hoch/1.wav")["predicted_label"], "hoch")

    def test_video_mit_conv3d(self):
        nodes = [n("v", "video_loader", "data", numFrames=4, imageSize=32),
                 n("c", "conv3d", "layer", inChannels="auto", outChannels=8),
                 n("p", "pool3d", "logic", type="global_avg"), n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "video_loader", "params": {"numFrames": 4, "imageSize": 32}}
        g = graph(nodes, [e("v", "c"), e("c", "p"), e("p", "d")], data, epochs=12)
        _, m = fit(self, g, self.video, f"{self.tmp}/m_video")
        self.assertEqual(m.predict_file(f"{self.video}/hell/0.mp4")["predicted_label"], "hell")

    def test_text_mit_eigenem_tokenizer_und_lstm(self):
        tok = n("t", "tokenizer", "data", maxLength=12, vocabSize=64)
        nodes = [n("x", "text_loader", "data", targetCol="label"), tok,
                 n("em", "embedding", "layer", vocabSize=64, embeddingDim=16),
                 n("l", "lstm", "layer", inputSize="auto", hiddenSize=16, numLayers=1),
                 n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "text_loader", "params": {"targetCol": "label"},
                "pipeline": [{"type": "tokenizer", "params": tok["params"]}]}
        g = graph(nodes, [e("x", "t"), e("t", "em"), e("em", "l"), e("l", "d")], data, epochs=10)
        p, m = fit(self, g, self.text, f"{self.tmp}/m_text")
        self.assertGreater(p._final["accuracy"], 0.95)
        self.assertEqual(m.predict_text("toll super genial prima schoen toll")["predicted_label"], "gut")

    def test_text_transformer_selbst_gebaut(self):
        tok = n("t", "tokenizer", "data", maxLength=12, vocabSize=64)
        nodes = [n("x", "text_loader", "data", targetCol="label"), tok,
                 n("em", "embedding", "layer", vocabSize=64, embeddingDim=16),
                 n("pos", "positional", "layer", maxLength=12, dim="auto"),
                 n("tb", "transformer_block", "layer", embedDim="auto", numHeads=2, ffnDim=32, dropout=0),
                 n("p", "pool", "logic", type="global_avg"), n("d", "dense", "layer", inputSize="auto", outputSize=2)]
        data = {"type": "text_loader", "params": {"targetCol": "label"},
                "pipeline": [{"type": "tokenizer", "params": tok["params"]}]}
        g = graph(nodes, [e("x", "t"), e("t", "em"), e("em", "pos"), e("pos", "tb"), e("tb", "p"), e("p", "d")],
                  data, epochs=10)
        p, m = fit(self, g, self.text, f"{self.tmp}/m_tf")
        self.assertGreater(p._final["accuracy"], 0.95)
        self.assertEqual(m.predict_text("furchtbar mies grausam kaputt schlecht mies")["predicted_label"], "boese")


class Pipeline(Base):
    def loaders(self, data, ds):
        g = graph([n("a", data["type"], "data")], [], data)
        return get_dataloaders(parse_ir(g), ds, 8, {})

    def test_split_steuert_die_groessen_und_haelt_test_zurueck(self):
        data = {"type": "text_loader", "params": {"targetCol": "label"},
                "pipeline": [{"type": "dataset_split", "params": {"trainRatio": 0.5, "valRatio": 0.25}}]}
        tr, va = self.loaders(data, self.text)
        self.assertEqual((len(tr.dataset), len(va.dataset)), (100, 50))

    def test_tokenizer_begrenzt_vokabular_und_laenge(self):
        p = {}
        data = {"type": "text_loader", "params": {"targetCol": "label"},
                "pipeline": [{"type": "tokenizer", "params": {"maxLength": 3, "vocabSize": 6}}]}
        g = graph([n("a", "text_loader", "data")], [], data)
        tr, _ = get_dataloaders(parse_ir(g), self.text, 8, p)
        self.assertEqual(next(iter(tr))[0].shape[1], 3)
        self.assertEqual(len(p["tokenizer"]["vocab"]), 4)

    def test_augmentation_gilt_nur_im_training(self):
        data = {"type": "video_loader", "params": {"numFrames": 2, "imageSize": 16},
                "pipeline": [{"type": "augmentation", "params": {"noise": True, "flip": False}}]}
        tr, va = self.loaders(data, self.video)
        self.assertFalse(np.array_equal(tr.dataset[0][0], tr.dataset[0][0]))
        self.assertTrue(np.array_equal(va.dataset[0][0], va.dataset[0][0]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
