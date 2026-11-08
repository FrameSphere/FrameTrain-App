"""Export: normales PyTorch ohne FrameTrain, dazu TorchScript und ONNX."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .codenode import code_kwargs, info_of
from .graph import ENTRY
from .runtime import SynapseModule

HELPERS = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class AnyRankBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training, self.momentum, self.eps)


class LearnedPosition(nn.Module):
    def __init__(self, max_length, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, max_length, dim))

    def forward(self, x):
        return x + self.weight[:, : x.size(1)]


class Repeat(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
'''


def _layer_source(nid: str, mod: nn.Module, node_type: str, params: Dict[str, Any]) -> str:
    """Konstruktor-Aufruf mit aufgeloesten Groessen."""
    if isinstance(mod, nn.MultiheadAttention):
        return f"nn.MultiheadAttention({mod.embed_dim}, {mod.num_heads}, dropout={mod.dropout}, batch_first=True)"
    if isinstance(mod, nn.TransformerDecoderLayer):
        return (f"nn.TransformerDecoderLayer(d_model={mod.self_attn.embed_dim}, nhead={mod.self_attn.num_heads}, "
                f"dim_feedforward={mod.linear1.out_features}, dropout={mod.dropout.p}, batch_first=True, norm_first=True)")
    if isinstance(mod, nn.TransformerEncoderLayer):
        return (f"nn.TransformerEncoderLayer(d_model={mod.self_attn.embed_dim}, nhead={mod.self_attn.num_heads}, "
                f"dim_feedforward={mod.linear1.out_features}, dropout={mod.dropout.p}, batch_first=True, norm_first=True)")
    if type(mod).__name__ == "AnyRankBatchNorm":
        return f"AnyRankBatchNorm({mod.num_features}, eps={mod.eps}, momentum={mod.momentum}, affine={mod.affine})"
    if type(mod).__name__ == "LearnedPosition":
        return f"LearnedPosition({mod.weight.size(1)}, {mod.weight.size(2)})"
    if node_type == "code":
        extra = "".join(f", {k}={v!r}" for k, v in code_kwargs(params, info_of(params)).items())
        return f"Code_{_ident(nid)}({params.get('_shape')!r}{extra})"
    if type(mod).__module__.startswith("torch.nn"):
        return "nn." + " ".join(repr(mod).split())
    raise ValueError(f"Node {nid} ({node_type}) laesst sich nicht als Quelltext exportieren")


def _ident(nid: str) -> str:
    return re.sub(r"\W", "_", nid)


def _forward_lines(module: SynapseModule) -> Tuple[List[str], Dict[str, str]]:
    names: Dict[Tuple[str, str], str] = {(ENTRY, "out"): "x", (ENTRY, "decoder"): "decoder"}
    lines: List[str] = []

    def var(nid: str, port: str) -> str:
        return names.setdefault((nid, port), f"{_ident(nid)}_{port}")

    for nid in module.order:
        node = module.node_map[nid]
        if node.spec.structural:
            continue
        wired = module.incoming.get(nid, {})
        a: Dict[str, str] = {}
        for port in node.inputs:
            if port.id in wired:
                a[port.id] = names[wired[port.id]]
            elif port.fallback and port.fallback in a:
                a[port.id] = a[port.fallback]
        p, t = node.params, node.type
        o = var(nid, node.outputs[0].id) if node.outputs else ""
        layer = f'self.layers["{nid}"]'
        i = a.get("in", "")
        if t == "dense" and str(p.get("implicitPool")).lower() == "true":
            lines.append(f"{o} = {layer}({i}.mean(dim=(2, 3)) if {i}.dim() == 4 else {i}.mean(dim=1) if {i}.dim() == 3 else {i})")
        elif t == "embedding":
            lines.append(f"{o} = {layer}({a['tokens']}.long())")
        elif t in ("lstm", "gru"):
            h = var(nid, "hidden")
            seq = f"{_ident(nid)}_seq"
            state = "state" if t == "gru" else "(state, _)"
            src = f"{i}.unsqueeze(1) if {i}.dim() == 2 else {i}" if t == "lstm" else i
            lines.append(f"{seq}, {state} = {layer}({src})")
            lines.append(f"{h} = torch.cat([state[-2], state[-1]], dim=-1) if {layer}.bidirectional else state[-1]")
            keep = str(p.get("returnSequence")).lower() == "true"
            lines.append(f"{o} = {seq}" if keep else f"{o} = {seq}[:, -1, :]")
        elif t == "attention":
            q, k, v = [f"{_ident(nid)}_{n}" for n in "qkv"]
            for n, src in zip((q, k, v), (a["query"], a["key"], a["value"])):
                lines.append(f"{n} = {src}.unsqueeze(1) if {src}.dim() == 2 else {src}")
            mask = "None"
            if str(p.get("causal")).lower() == "true":
                mask = f"torch.triu(torch.ones({q}.size(1), {k}.size(1), device={q}.device, dtype=torch.bool), diagonal=1)"
            lines.append(f"{o}, _ = {layer}({q}, {k}, {v}, attn_mask={mask})")
            lines.append(f"{o} = {o}.squeeze(1) if {o}.size(1) == 1 else {o}")
        elif t == "transformer_block":
            lines.append(f"{o} = {i}.unsqueeze(1) if {i}.dim() == 2 else {i}")
            if str(p.get("causal")).lower() == "true":
                lines.append(f"{o} = {layer}({o}, src_mask=torch.triu(torch.ones({o}.size(1), {o}.size(1), "
                             f"device={o}.device, dtype=torch.bool), diagonal=1))")
            else:
                lines.append(f"{o} = {layer}({o})")
                lines.append(f"{o} = {o}.squeeze(1) if {o}.size(1) == 1 else {o}")
        elif t == "pretrained":
            arg = f"pixel_values={i}" if str(p.get("inputKind")) == "image" else f"input_ids={i}.long()"
            lines.append(f"{o} = {layer}({arg}).last_hidden_state")
        elif t == "transformer_decoder":
            mask = f"torch.triu(torch.ones({i}.size(1), {i}.size(1), device={i}.device, dtype=torch.bool), diagonal=1)"
            lines.append(f"{o} = {layer}({i}, {a['memory']}, tgt_mask={mask})")
        elif t == "code":
            info = info_of(p)
            call = ", ".join(f"{arg}={a[port.id]}" for port, arg in zip(info.inputs, info.args) if port.id in a)
            res = f"{_ident(nid)}_res"
            lines.append(f"{res} = {layer}({call})")
            for k, port in enumerate(info.outputs):
                pick = f'{res}["{port.id}"] if isinstance({res}, dict) else '
                lines.append(f"{var(nid, port.id)} = {pick}{res if len(info.outputs) == 1 else f'{res}[{k}]'}")
        elif nid in module.layers and t != "reparameterize":
            lines.append(f"{o} = {layer}({i})")
        elif t == "add_node":
            lines.append(f"{o} = {a['a']} + {a['b']}")
        elif t == "multiply_node":
            lines.append(f"{o} = {a['a']} * {a['b']}")
        elif t == "matmul":
            lines.append(f"{o} = torch.matmul({a['a']}, {a['b']})")
        elif t == "normalize":
            lines.append(f"{o} = F.normalize({i}, p={float(p.get('p', 2))}, dim={int(p.get('dim', -1))})")
        elif t == "reshape":
            dims = [int(v) for v in str(p.get("shape", "")).replace(" ", "").split(",") if v]
            dims = dims[1:] if dims and dims[0] == -1 else dims
            lines.append(f"{o} = {i}.reshape({i}.size(0), {', '.join(map(str, dims))})")
        elif t == "transpose":
            lines.append(f"{o} = {i}.transpose({int(p.get('dim0', -2))}, {int(p.get('dim1', -1))})")
        elif t == "flatten":
            lines.append(f"{o} = {i}.flatten(1)")
        elif t == "merge":
            lines.append(f"{o} = torch.cat([{a['a']}, {a['b']}], dim={int(p.get('dim', -1))})")
        elif t == "split_node":
            b = var(nid, "b")
            dim = int(p.get("dim", -1))
            parts = f"{_ident(nid)}_parts"
            lines.append(f"{parts} = torch.chunk({i}, {int(p.get('chunks', 2))}, dim={dim})")
            lines.append(f"{o} = {parts}[0]")
            lines.append(f"{b} = torch.cat({parts}[1:], dim={dim})")
        elif t == "pool":
            kind, stride = str(p.get("type", "global_avg")), int(p.get("stride", 2))
            seq = f"{i}.max(dim=1).values" if "max" in kind else f"{i}.mean(dim=1)"
            img = {"avg_2d": f"F.avg_pool2d({i}, 2, stride={stride})", "max_2d": f"F.max_pool2d({i}, 2, stride={stride})",
                   "global_max": f"F.adaptive_max_pool2d({i}, 1).flatten(1)"}.get(
                kind, f"F.adaptive_avg_pool2d({i}, 1).flatten(1)")
            lines.append(f"{o} = {seq} if {i}.dim() == 3 else {img}")
        elif t in ("pool1d", "pool3d"):
            d = t[-2]
            kind, k = str(p.get("type", "max")), int(p.get("kernel", 2))
            call = {"global_avg": f"F.adaptive_avg_pool{d}d({i}, 1).flatten(1)",
                    "global_max": f"F.adaptive_max_pool{d}d({i}, 1).flatten(1)",
                    "avg": f"F.avg_pool{d}d({i}, {k})"}.get(kind, f"F.max_pool{d}d({i}, {k})")
            lines.append(f"{o} = {call}")
        elif t == "reparameterize":
            lines.append(f"{o} = {a['mu']} + torch.randn_like({a['mu']}) * torch.exp(0.5 * {a['logvar']}) "
                         f"if self.training else {a['mu']}")
        else:
            raise ValueError(f"Node {nid} ({t}) laesst sich nicht als Quelltext exportieren")
    returns = [names[k] for k in module.graph.outputs.values()]
    lines.append(f"return {returns[0]}" if len(returns) == 1 else "return {" + ", ".join(
        f'"{n}": {names[k]}' for n, k in module.graph.outputs.items()) + "}")
    return lines, {}


def _class_source(module: SynapseModule, name: str, extra: List[str], head_imports: Any = None) -> str:
    """Eine nn.Module-Klasse; Bloecke werden zu eigenen Klassen davor."""
    head_imports = head_imports if head_imports is not None else set()
    items = []
    for nid, mod in module.layers.items():
        node = module.node_map[nid]
        if node.type == "reparameterize":
            continue
        if node.type == "pretrained":
            items.append(f'            "{nid}": _pretrained({type(mod.config).__name__}.from_dict({mod.config.to_dict()!r})),')
            head_imports.add(type(mod.config).__name__)
            continue
        if node.type == "block":
            names = []
            for i, inner in enumerate(mod.blocks):
                inner_name = f"{name}_{_ident(nid)}_{i}"
                extra.append(_class_source(inner, inner_name, extra, head_imports))
                names.append(f"{inner_name}()")
            items.append(f'            "{nid}": Repeat([{", ".join(names)}]),')
            continue
        params = {**node.params, "_shape": module.input_shapes.get(nid)}
        items.append(f'            "{nid}": {_layer_source(nid, mod, node.type, params)},')
    lines, _ = _forward_lines(module)
    body = "\n".join(f"        {line}" for line in lines)
    joined = "\n".join(items)
    return f'''class {name}(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({{
{joined}
        }})

    def forward(self, x, decoder=None):
{body}
'''


def model_source(module: SynapseModule) -> str:
    codes = [(nid, module.node_map[nid].params.get("code")) for nid in module.order if module.node_map[nid].type == "code"]
    head = HELPERS + ("\nfrom typing import Literal, Optional\n" if codes else "")
    for nid, code in codes:
        head += "\n\n" + re.sub(r"class\s+Node\b", f"class Code_{_ident(nid)}", str(code or ""))
    extra: List[str] = []
    imports: set = set()
    main = _class_source(module, "Model", extra, imports)
    if imports:
        head = (f"from transformers import AutoModel, {', '.join(sorted(imports))}\n\n" + head
                + "\n\ndef _pretrained(config):\n    return AutoModel.from_config(config)\n")
    return "\n\n".join([head.rstrip(), *extra, main])


def _train_source(input_shape: List[int]) -> str:
    return f'''"""Beispiel-Trainingsschleife. Tausche `batches()` gegen deine Daten."""
import torch
import torch.nn.functional as F

from model import Model


def batches():
    for _ in range(10):
        yield torch.zeros(8, *{list(input_shape[1:])}), torch.zeros(8, dtype=torch.long)


model = Model()
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for x, y in batches():
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), "model.pt")
'''


INFER_SOURCE = '''"""Vorhersage mit dem exportierten Modell."""
import json

import torch

from model import Model

model = Model()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
preprocess = json.load(open("preprocess.json", encoding="utf-8"))


def predict(x):
    with torch.no_grad():
        return model(x)
'''


def export_pytorch(module: SynapseModule, preprocess: Dict[str, Any], out_dir: Any, input_shape: Any) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = list(input_shape or preprocess.get("input_shape") or [1, 16])
    (out / "model.py").write_text(model_source(module), encoding="utf-8")
    (out / "train.py").write_text(_train_source(shape), encoding="utf-8")
    (out / "infer.py").write_text(INFER_SOURCE, encoding="utf-8")
    needs = [f"torch>={torch.__version__.split('+')[0]}"]
    if any(n.type == "pretrained" for n in module.graph.nodes):
        needs.append("transformers")
    (out / "requirements.txt").write_text("\n".join(needs) + "\n", encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps(preprocess, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    state = {k: v for k, v in module.state_dict().items() if ".reparameterize" not in k}
    torch.save(state, out / "model.pt")
    return out


def _example(module: SynapseModule, preprocess: Dict[str, Any], input_shape: Any) -> torch.Tensor:
    if preprocess.get("task") == "seq2seq":
        raise ValueError("Seq2Seq-Modelle haben zwei Eingaenge: nutze das PyTorch-Projekt oder safetensors.")
    shape = tuple(input_shape or preprocess.get("input_shape") or (1, 16))
    dtype = getattr(torch, str(preprocess.get("input_dtype", "float32")))
    return torch.zeros(*shape, dtype=dtype)


def export_torchscript(module: SynapseModule, preprocess: Dict[str, Any], path: Any, input_shape: Any = None) -> Path:
    module.eval()
    traced = torch.jit.trace(module, _example(module, preprocess, input_shape), strict=False)
    traced.save(str(path))
    return Path(path)


def export_onnx(module: SynapseModule, preprocess: Dict[str, Any], path: Any, input_shape: Any = None) -> Path:
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        from ft_data.deps import missing
        raise missing("onnx", "onnxscript", what="Der ONNX-Export") from e
    module.eval()
    # Eine Datei statt .onnx plus .onnx.data: sonst geht beim Weitergeben die
    # Haelfte verloren. Ueber 2 GB erzwingt das Format externe Daten.
    big = sum(p.numel() * p.element_size() for p in module.parameters()) > 1_800_000_000
    torch.onnx.export(module, (_example(module, preprocess, input_shape),), str(path), external_data=big)
    return Path(path)


def export_safetensors(module: SynapseModule, path: Any) -> Path:
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(path))
    return Path(path)


def export_ftmodel(model_dir: Any, path: Any) -> Path:
    """Ganzer Modell-Ordner in einer Datei."""
    base = shutil.make_archive(str(Path(path).with_suffix("")), "zip", str(model_dir))
    target = Path(path)
    Path(base).replace(target)
    return target


NEURON_LAYERS = {"relu": "ne.ReLU()", "gelu": "ne.GELU()", "sigmoid": "ne.Sigmoid()", "tanh": "ne.Tanh()",
                 "softmax": "ne.Softmax()"}


def export_neuron(module: SynapseModule, preprocess: Dict[str, Any], out_dir: Any) -> Path:
    """Kette aus Dense, Conv2D, LayerNorm und Aktivierungen ganz ohne PyTorch."""
    import numpy as np
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).resolve().parent.parent / "neuron_engine", out / "neuron_engine", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "test_*"))
    lines, weights = [], {}
    previous = ENTRY
    for nid in module.order:
        node = module.node_map[nid]
        if node.spec.structural:
            continue
        wired = list(module.incoming.get(nid, {}).values())
        if len(wired) != 1 or wired[0][0] != previous:
            raise ValueError(f"NeuronEngine exportiert nur Ketten, Node {nid} hat eine Verzweigung")
        previous = nid
        mod = module.layers[nid] if nid in module.layers else None
        if node.type == "dense":
            lines.append(f"ne.Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})")
            weights[f"{len(lines) - 1}.w"] = mod.weight.detach().numpy()
            if mod.bias is not None:
                weights[f"{len(lines) - 1}.b"] = mod.bias.detach().numpy()
        elif node.type == "conv2d":
            lines.append(f"ne.Conv2d({mod.in_channels}, {mod.out_channels}, {mod.kernel_size[0]}, "
                         f"stride={mod.stride[0]}, padding={mod.padding[0]})")
            weights[f"{len(lines) - 1}.w"] = mod.weight.detach().numpy()
            weights[f"{len(lines) - 1}.b"] = mod.bias.detach().numpy()
        elif node.type == "layernorm":
            lines.append(f"ne.LayerNorm({mod.normalized_shape[0]}, eps={mod.eps})")
            weights[f"{len(lines) - 1}.w"] = mod.weight.detach().numpy()
            weights[f"{len(lines) - 1}.b"] = mod.bias.detach().numpy()
        elif node.type in NEURON_LAYERS:
            lines.append(NEURON_LAYERS[node.type])
        else:
            raise ValueError(f"Node {nid} ({node.type}) gibt es in NeuronEngine noch nicht")
    np.savez(out / "weights.npz", **weights)
    layers = ",\n    ".join(lines)
    (out / "model.py").write_text(f'''"""Modell ohne PyTorch. Gewichte liegen in weights.npz."""
import numpy as np

import neuron_engine as ne

model = ne.Sequential([
    {layers},
])
weights = np.load("weights.npz")
for key in weights.files:
    index, name = key.split(".")
    getattr(model.layers[int(index)], name).data = weights[key].astype(np.float64)
''', encoding="utf-8")
    (out / "preprocess.json").write_text(json.dumps(preprocess, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    return out


def export_model(model_dir: str, fmt: str, target: str) -> str:
    """Einstieg fuer den Dienst: laedt ein trainiertes Modell und exportiert es."""
    import sys
    canvas = Path(__file__).resolve().parent.parent / "train_engine" / "plugins" / "canvas"
    if str(canvas) not in sys.path:
        sys.path.insert(0, str(canvas))
    from loader import load_canvas_model
    if fmt == "ftmodel":
        return str(export_ftmodel(model_dir, target))
    loaded = load_canvas_model(model_dir, device="cpu")
    module, pre = loaded.model, loaded.preprocess
    shape = None
    if hasattr(module, "generator"):  # GAN: exportiert wird der Generator, Eingang ist der Zufallsvektor
        shape = (1, module.latent)
        pre = {**pre, "input_shape": list(shape), "input_dtype": "float32"}
        module = module.generator
    if fmt == "pytorch":
        return str(export_pytorch(module, pre, target, shape))
    if fmt == "neuron":
        return str(export_neuron(module, pre, target))
    if fmt == "torchscript":
        return str(export_torchscript(module, pre, target, shape))
    if fmt == "onnx":
        return str(export_onnx(module, pre, target, shape))
    if fmt == "safetensors":
        return str(export_safetensors(module, target))
    raise ValueError(f"Unbekanntes Export-Format: {fmt}")
