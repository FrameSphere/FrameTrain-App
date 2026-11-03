"""Gradienten gleich PyTorch, Lernen ganz ohne PyTorch.

Aufruf:  python3.11 -m unittest neuron_engine.test_neuron_engine   (aus src-tauri/python)
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import neuron_engine as ne


def check(case, ne_tensors, torch_tensors):
    for a, b in zip(ne_tensors, torch_tensors):
        case.assertLess(np.abs(a.grad - b.grad.numpy()).max(), 1e-5)


class Gradients(unittest.TestCase):
    def test_mlp(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 4))
        y = np.array([0, 1, 2, 1, 0])
        l1, l2 = ne.Linear(4, 8, rng=rng), ne.Linear(8, 3, rng=rng)
        loss = ne.cross_entropy(l2(l1(ne.Tensor(x)).gelu()), y)
        loss.backward()
        w1, b1, w2, b2 = [torch.tensor(t.data, requires_grad=True) for t in (l1.w, l1.b, l2.w, l2.b)]
        t = F.cross_entropy(F.linear(F.gelu(F.linear(torch.tensor(x), w1, b1)), w2, b2), torch.tensor(y))
        t.backward()
        self.assertLess(abs(loss.data - t.item()), 1e-5)
        check(self, (l1.w, l1.b, l2.w, l2.b), (w1, b1, w2, b2))

    def test_conv2d(self):
        rng = np.random.default_rng(1)
        x = ne.Tensor(rng.normal(size=(2, 3, 6, 6)), requires_grad=True)
        c = ne.Conv2d(3, 4, 3, stride=2, padding=1, rng=rng)
        loss = (c(x) * c(x)).sum()
        loss.backward()
        tx = torch.tensor(x.data, requires_grad=True)
        w, b = torch.tensor(c.w.data, requires_grad=True), torch.tensor(c.b.data, requires_grad=True)
        t = (F.conv2d(tx, w, b, stride=2, padding=1) ** 2).sum()
        t.backward()
        self.assertLess(abs(loss.data - t.item()) / abs(t.item()), 1e-6)
        check(self, (c.w, c.b, x), (w, b, tx))

    def test_attention_und_layernorm(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 4, 8))
        a, n = ne.Attention(8, 2, rng=rng), ne.LayerNorm(8)
        loss = (n(a(ne.Tensor(x), True)) * 1.0).sum()
        loss.backward()
        ps = {k: [torch.tensor(getattr(a, k).w.data, requires_grad=True),
                  torch.tensor(getattr(a, k).b.data, requires_grad=True)] for k in "qkv"}
        pout = [torch.tensor(a.out.w.data, requires_grad=True), torch.tensor(a.out.b.data, requires_grad=True)]
        tx = torch.tensor(x)

        def head(k):
            return F.linear(tx, *ps[k]).reshape(2, 4, 2, 4).transpose(1, 2)

        s = head("q") @ head("k").transpose(-1, -2) / 4 ** 0.5
        s = s + torch.triu(torch.full((4, 4), -1e9, dtype=torch.float64), 1)
        m = (s.softmax(-1) @ head("v")).transpose(1, 2).reshape(2, 4, 8)
        o = F.linear(m, *pout)
        t = F.layer_norm(o, (8,), torch.tensor(n.w.data), torch.tensor(n.b.data)).sum()
        t.backward()
        self.assertLess(abs(loss.data - t.item()), 1e-4)
        check(self, (a.q.w, a.k.w, a.v.b, a.out.w), (ps["q"][0], ps["k"][0], ps["v"][1], pout[0]))

    def test_adam_gleich_pytorch(self):
        w = ne.Tensor(np.array([1.0, -2.0]), requires_grad=True)
        tw = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        a, b = ne.Adam([w], lr=0.1), torch.optim.Adam([tw], lr=0.1)
        for _ in range(5):
            a.zero_grad()
            (w * w).sum().backward()
            a.step()
            b.zero_grad()
            (tw * tw).sum().backward()
            b.step()
        self.assertLess(np.abs(w.data - tw.detach().numpy()).max(), 1e-6)


class WithoutTorch(unittest.TestCase):
    def test_xor_lernt_ohne_pytorch(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import neuron_engine as ne
rng = np.random.default_rng(0)
model = ne.Sequential([ne.Linear(2, 8, rng=rng), ne.Tanh(), ne.Linear(8, 2, rng=rng)])
x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]); y = [0, 1, 1, 0]
opt = ne.Adam(model.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad(); loss = ne.cross_entropy(model(ne.Tensor(x)), y); loss.backward(); opt.step()
assert list(model(ne.Tensor(x)).data.argmax(-1)) == y
assert "torch" not in sys.modules
print("ok")
"""
        r = subprocess.run([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parent.parent)],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "ok", r.stderr[-500:])

    def test_mini_sprachmodell_lernt(self):
        rng = np.random.default_rng(0)
        text = "abcd" * 30
        ids = np.array(["abcd".index(c) for c in text])
        em, att, head = ne.Embedding(4, 16, rng=rng), ne.Attention(16, 2, rng=rng), ne.Linear(16, 4, rng=rng)
        params = list(em.parameters()) + list(att.parameters()) + list(head.parameters())
        opt = ne.Adam(params, lr=0.01)
        x, y = ids[:8][None], ids[1:9]
        for _ in range(150):
            opt.zero_grad()
            logits = head(att(em(x), True))
            loss = ne.cross_entropy(logits.reshape(8, 4), y)
            loss.backward()
            opt.step()
        self.assertTrue((logits.data[0].argmax(-1) == y).all())


if __name__ == "__main__":
    unittest.main()
