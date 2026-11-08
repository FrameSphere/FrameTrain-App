"""Alle Nodes des Synapse Builders."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codenode import DEFAULT_CODE, build_code, compile_code, run_code  # noqa: F401
from .spec import AUTO, Param, Port, node


class AnyRankBatchNorm(nn.Module):
    """BatchNorm fuer jede Eingabe-Dimensionalitaet (2D bis 5D).

    Derselbe Node sieht nach Dense [N, C] und nach Conv2D [N, C, H, W].
    Parameter heissen wie bei nn.BatchNorm*, bestehende Checkpoints bleiben ladbar.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features, self.eps, self.momentum, self.affine = num_features, eps, momentum, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"BatchNorm erwartet mindestens 2D (N, C, ...), bekam {x.dim()}D")
        if self.training:
            self.num_batches_tracked += 1
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training, self.momentum, self.eps)


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return default


def _size(p: Dict[str, Any], key: str, default: int, shapes: Dict[str, Any], port: str, dim: int) -> int:
    """Groesse aus dem Parameter oder, bei 'auto', aus der Eingabeform."""
    # Ohne Eingabeform (gespeicherte v1-Graphen) gilt der alte Standardwert.
    value = p.get(key, AUTO if shapes else default)
    if value == AUTO:
        shape = (shapes or {}).get(port)
        if shape is None:
            raise ValueError(f"'{key}' steht auf auto, aber die Eingabeform ist unbekannt")
        return int(shape[dim])
    return _int(value, default)


def _out(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"out": x}


def _plain(m, p, i):
    return _out(m(i["in"]))


# ── Datenquellen und Datenstruktur ──────────────────────────────────────────
# Sie liefern im Modell-Graph die Eingabe. Was davon noch keine Wirkung hat, ist
# als planned markiert und wird mit dem Daten-Graph (Phase 2) wirksam.
node("input", "data", inputs=(), structural=True, shape="-1, 3, 224, 224", dtype="float32")
node("csv_loader", "data", inputs=(), structural=True, separator=",", hasHeader=True, targetCol="label", normalize=False)
node("image_loader", "data", inputs=(), structural=True, imageSize=224, channels="3", normalize=True,
     cacheRam=False)
node("parquet_loader", "data", inputs=(), structural=True, targetCol="label")
node("audio_loader", "data", inputs=(), structural=True, sampleRate=16000, seconds=1.0)
node("video_loader", "data", inputs=(), structural=True, numFrames=8, imageSize=112)
node("text_loader", "data", inputs=(), structural=True, textCol="", targetCol="label")
node("mask_loader", "data", inputs=(), structural=True, imageSize=128)
node("seq2seq_loader", "data", inputs=(), structural=True, sourceCol="source", targetCol="target")
node("decoder_input", "data", inputs=(), structural=True)
node("timestep", "data", inputs=(), structural=True)
node("latent", "data", inputs=(), structural=True, dim=64)
node("detection_loader", "data", inputs=(), structural=True, imageSize=128, gridSize=8)
node("mel_spectrogram", "data", inputs=(Port("in", "Dataset"),), outputs=(Port("out", "Dataset"),), structural=True,
     nMels=64, nFft=400, hopLength=160)
node("tokenizer", "data", inputs=(Port("text", "Dataset"),), outputs=("tokens",), structural=True,
     level="word", maxLength=512, vocabSize=50000, padding=True, truncation=True)
node("dataset_split", "data", inputs=(Port("in", "Dataset"),),
     outputs=(Port("train", "Dataset"), Port("val", "Dataset"), Port("test", "Dataset")), structural=True,
     trainRatio=0.8, valRatio=0.1, shuffle=True, seed=42)
node("augmentation", "data", inputs=(Port("in", "Dataset"),), outputs=(Port("out", "Dataset"),), structural=True,
     flip=True, rotate=15, brightness=0.2, noise=False)


# ── Layer ───────────────────────────────────────────────────────────────────
def _dense_build(p, s):
    # Beim stillen Mitteln bleibt von [B, C, H, W] nur C uebrig.
    dim = 1 if _bool(p.get("implicitPool")) and len((s or {}).get("in", ())) == 4 else -1
    return nn.Linear(_size(p, "inputSize", 128, s, "in", dim), _int(p.get("outputSize"), 256),
                     bias=_bool(p.get("bias"), True))


def _dense_init(mod: nn.Module, p: Dict[str, Any]) -> None:
    fn = {"xavier_uniform": nn.init.xavier_uniform_, "xavier_normal": nn.init.xavier_normal_,
          "kaiming_uniform": nn.init.kaiming_uniform_, "kaiming_normal": nn.init.kaiming_normal_,
          "zeros": nn.init.zeros_, "ones": nn.init.ones_}.get(str(p.get("initializer", "") or "").lower())
    if fn is not None:
        fn(mod.weight)


def _dense_run(m, p, i):
    x = i["in"]
    # Vor Phase 1 gebaute Graphen verlassen sich auf das stille Mitteln.
    if _bool(p.get("implicitPool")):
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        elif x.dim() == 3:
            x = x.mean(dim=1)
    return _out(m(x))


def _dense(p, s):
    mod = _dense_build(p, s)
    _dense_init(mod, p)
    return mod


node("dense", "layer", build=_dense, run=_dense_run, inputSize=AUTO, outputSize=256, bias=True,
     initializer="xavier_uniform", implicitPool=False)


def _conv2d(p, s):
    pad = p.get("padding", 1)
    return nn.Conv2d(_size(p, "inChannels", 3, s, "in", 1), _int(p.get("outChannels"), 64),
                     kernel_size=_int(p.get("kernelSize"), 3), stride=_int(p.get("stride"), 1),
                     padding="same" if pad == "same" else _int(pad, 1), groups=_int(p.get("groups"), 1))


node("conv2d", "layer", build=_conv2d, run=_plain, inChannels=AUTO, outChannels=64, kernelSize="3", stride=1,
     padding="1", groups=1)


def _conv_t2d(p, s):
    return nn.ConvTranspose2d(_size(p, "inChannels", 64, s, "in", 1), _int(p.get("outChannels"), 32),
                              kernel_size=_int(p.get("kernelSize"), 2), stride=_int(p.get("stride"), 2),
                              padding=_int(p.get("padding"), 0))


node("conv_transpose2d", "layer", build=_conv_t2d, run=_plain, inChannels=AUTO, outChannels=32, kernelSize=2,
     stride=2, padding=0)
node("upsample", "layer", build=lambda p, s: nn.Upsample(scale_factor=_float(p.get("scale"), 2),
                                                         mode=str(p.get("mode", "nearest"))),
     run=_plain, scale=2, mode="nearest")


def _reparameterize(m, p, i):
    mu, logvar = i["mu"], i["logvar"]
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if m is None or m.training else mu
    return {"out": z, "kl": kl * _float(p.get("beta"), 1.0)}


class _Mode(nn.Module):
    """Traegt nur train/eval fuer Nodes ohne Gewichte."""


node("reparameterize", "layer", inputs=("mu", "logvar"), outputs=("out", Port("kl", "Scalar")),
     build=lambda p, s: _Mode(), run=_reparameterize, beta=1.0)


node("code", "layer", build=build_code, run=run_code, code=DEFAULT_CODE)


def _pretrained(p, s):
    path = str(p.get("path") or "")
    if not path:
        raise ValueError("Vortrainiert: Bitte einen Modell-Ordner oder Hugging-Face-Namen angeben")
    try:
        from transformers import AutoModel
    except ImportError as e:
        from ft_data.deps import missing
        raise missing("transformers", what="Vortrainierte Bausteine") from e
    model = AutoModel.from_pretrained(path)
    if _bool(p.get("freeze"), True):
        model.requires_grad_(False)
    return model


def _pretrained_run(m, p, i):
    x = i["in"]
    if str(p.get("inputKind", "tokens")) == "image":
        return _out(m(pixel_values=x).last_hidden_state)
    return _out(m(input_ids=x.long()).last_hidden_state)


node("pretrained", "layer", build=_pretrained, run=_pretrained_run, path="", freeze=True, inputKind="tokens")


class Repeat(nn.Module):
    """Unter-Graph, N-mal hintereinander. Jede Wiederholung hat eigene Gewichte."""

    def __init__(self, raw: Dict[str, Any], times: int, shape: Any, dtype: Any = None):
        super().__init__()
        from .graph import parse_block
        from .runtime import SynapseModule
        self.blocks = nn.ModuleList()
        for _ in range(max(times, 1)):
            block = SynapseModule(parse_block(raw), input_shape=shape, input_dtype=dtype or torch.float32)
            self.blocks.append(block)
            if shape is not None:
                with torch.device("meta"):
                    probe = copy_to_meta(block)(torch.empty(*shape, dtype=dtype or torch.float32))
                shape = tuple(probe.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def copy_to_meta(mod: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(mod).to_empty(device="meta")


node("block", "structure", build=lambda p, s: Repeat(p.get("graph") or {}, _int(p.get("repeat"), 1), (s or {}).get("in")),
     run=_plain, graph=None, repeat=1, role="")


def _decoder(p, s):
    return nn.TransformerDecoderLayer(d_model=_size(p, "embedDim", 512, s, "in", -1), nhead=_int(p.get("numHeads"), 8),
                                      dim_feedforward=_int(p.get("ffnDim"), 2048), dropout=_float(p.get("dropout"), 0.1),
                                      batch_first=True, norm_first=True)


def _decoder_run(m, p, i):
    x = i["in"]
    mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1)
    return _out(m(x, i["memory"], tgt_mask=mask))


node("transformer_decoder", "layer", inputs=("in", "memory"), build=_decoder, run=_decoder_run, embedDim=AUTO,
     numHeads=8, ffnDim=2048, dropout=0.1)


def _conv1d(p, s):
    return nn.Conv1d(_size(p, "inChannels", 1, s, "in", 1), _int(p.get("outChannels"), 32),
                     kernel_size=_int(p.get("kernelSize"), 3), stride=_int(p.get("stride"), 1),
                     padding=_int(p.get("padding"), 1))


node("conv1d", "layer", build=_conv1d, run=_plain, inChannels=AUTO, outChannels=32, kernelSize=3, stride=1, padding=1)


def _conv3d(p, s):
    return nn.Conv3d(_size(p, "inChannels", 3, s, "in", 1), _int(p.get("outChannels"), 16),
                     kernel_size=_int(p.get("kernelSize"), 3), stride=_int(p.get("stride"), 1),
                     padding=_int(p.get("padding"), 1))


node("conv3d", "layer", build=_conv3d, run=_plain, inChannels=AUTO, outChannels=16, kernelSize=3, stride=1, padding=1)


class LearnedPosition(nn.Module):
    """Lernbare Position je Stelle einer Sequenz [B, T, D]."""

    def __init__(self, max_length: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, max_length, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.weight.size(1):
            raise ValueError(f"Sequenz hat {x.size(1)} Stellen, Position kennt nur {self.weight.size(1)}")
        return x + self.weight[:, : x.size(1)]


node("positional", "layer", build=lambda p, s: LearnedPosition(_int(p.get("maxLength"), 512),
                                                               _size(p, "dim", 512, s, "in", -1)),
     run=_plain, maxLength=512, dim=AUTO)


def _gru(p, s):
    layers = _int(p.get("numLayers"), 1)
    return nn.GRU(_size(p, "inputSize", 256, s, "in", -1), _int(p.get("hiddenSize"), 256), num_layers=layers,
                  bidirectional=_bool(p.get("bidirectional")),
                  dropout=_float(p.get("dropout"), 0.0) if layers > 1 else 0.0, batch_first=True)


def _gru_run(m, p, i):
    seq, h = m(i["in"])
    hidden = torch.cat([h[-2], h[-1]], dim=-1) if m.bidirectional else h[-1]
    return {"out": seq if _bool(p.get("returnSequence")) else seq[:, -1, :], "hidden": hidden}


node("gru", "layer", outputs=("out", "hidden"), build=_gru, run=_gru_run, inputSize=AUTO, hiddenSize=256, numLayers=1,
     bidirectional=False, dropout=0.0, returnSequence=False)


def _embedding(p, s):
    return nn.Embedding(_int(p.get("vocabSize"), 50000), _int(p.get("embeddingDim"), 512),
                        padding_idx=_int(p.get("paddingIdx"), 0))


node("embedding", "layer", inputs=("tokens",), build=_embedding, run=lambda m, p, i: _out(m(i["tokens"].long())),
     vocabSize=50000, embeddingDim=512, paddingIdx=0)


def _lstm(p, s):
    layers = _int(p.get("numLayers"), 2)
    return nn.LSTM(_size(p, "inputSize", 256, s, "in", -1), _int(p.get("hiddenSize"), 512), num_layers=layers,
                   bidirectional=_bool(p.get("bidirectional")),
                   dropout=_float(p.get("dropout"), 0.1) if layers > 1 else 0.0, batch_first=True)


def _lstm_run(m, p, i):
    x = i["in"]
    if x.dim() == 2:
        x = x.unsqueeze(1)
    seq, (h, _) = m(x)
    hidden = torch.cat([h[-2], h[-1]], dim=-1) if m.bidirectional else h[-1]
    return {"out": seq if _bool(p.get("returnSequence")) else seq[:, -1, :], "hidden": hidden}


node("lstm", "layer", outputs=("out", "hidden"), build=_lstm, run=_lstm_run, inputSize=AUTO, hiddenSize=512,
     numLayers=2, bidirectional=False, dropout=0.1, returnSequence=False)


def _attention(p, s):
    return nn.MultiheadAttention(_size(p, "embedDim", 512, s, "query", -1), _int(p.get("numHeads"), 8),
                                 dropout=_float(p.get("dropout"), 0.1), batch_first=True)


def _attention_run(m, p, i):
    def seq(x):
        return x.unsqueeze(1) if x.dim() == 2 else x
    q, k, v = seq(i["query"]), seq(i["key"]), seq(i["value"])
    mask = None
    if _bool(p.get("causal")) and q.size(1) > 1:
        mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool), diagonal=1)
    out, _ = m(q, k, v, attn_mask=mask)
    return _out(out.squeeze(1) if out.size(1) == 1 else out)


node("attention", "layer", inputs=(Port("query"), Port("key", fallback="query"), Port("value", fallback="key")),
     build=_attention, run=_attention_run, embedDim=AUTO, numHeads=8, dropout=0.1, causal=False)


def _transformer(p, s):
    return nn.TransformerEncoderLayer(d_model=_size(p, "embedDim", 512, s, "in", -1), nhead=_int(p.get("numHeads"), 8),
                                      dim_feedforward=_int(p.get("ffnDim"), 2048), dropout=_float(p.get("dropout"), 0.1),
                                      batch_first=True, norm_first=True)


def _transformer_run(m, p, i):
    x = i["in"]
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if _bool(p.get("causal")):
        # Jede Stelle sieht nur sich und die Vergangenheit (Sprachmodelle).
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1)
        return _out(m(x, src_mask=mask))
    out = m(x)
    return _out(out.squeeze(1) if out.size(1) == 1 else out)


node("transformer_block", "layer", build=_transformer, run=_transformer_run, embedDim=AUTO, numHeads=8, ffnDim=2048,
     dropout=0.1, causal=False)


def _layernorm(p, s):
    return nn.LayerNorm(_size(p, "normalizedShape", 512, s, "in", -1), eps=_float(p.get("eps"), 1e-5),
                        elementwise_affine=_bool(p.get("affine"), True))


node("layernorm", "layer", build=_layernorm, run=_plain, normalizedShape=AUTO, eps=1e-5, affine=True)


def _batchnorm(p, s):
    return AnyRankBatchNorm(_size(p, "numFeatures", 64, s, "in", 1), eps=_float(p.get("eps"), 1e-5),
                            momentum=_float(p.get("momentum"), 0.1), affine=_bool(p.get("affine"), True))


node("batchnorm", "layer", build=_batchnorm, run=_plain, numFeatures=AUTO, eps=1e-5, momentum=0.1, affine=True)
node("dropout", "layer", build=lambda p, s: nn.Dropout(p=_float(p.get("p"), 0.1), inplace=_bool(p.get("inplace"))),
     run=_plain, p=0.1, inplace=False)

# ── Aktivierungen ───────────────────────────────────────────────────────────
node("relu", "activation", build=lambda p, s: nn.ReLU(inplace=_bool(p.get("inplace"))), run=_plain, inplace=False)
node("gelu", "activation", build=lambda p, s: nn.GELU(approximate=str(p.get("approximate", "none"))), run=_plain,
     approximate="none")
node("sigmoid", "activation", build=lambda p, s: nn.Sigmoid(), run=_plain)
node("softmax", "activation", build=lambda p, s: nn.Softmax(dim=_int(p.get("dim"), -1)), run=_plain, dim=-1)
node("tanh", "activation", build=lambda p, s: nn.Tanh(), run=_plain)
node("leaky_relu", "activation", build=lambda p, s: nn.LeakyReLU(negative_slope=_float(p.get("negativeSlope"), 0.01)),
     run=_plain, negativeSlope=0.01)
node("silu", "activation", build=lambda p, s: nn.SiLU(), run=_plain)

# ── Lernen: wirkt ueber die Trainings-Spezifikation, nicht im forward ───────
node("optimizer", "training", inputs=(), outputs=(Port("optimizer", "Any"),), structural=True)
node("loss", "training", inputs=("pred", "targets"), outputs=(Port("loss", "Scalar"),), structural=True)
node("scheduler", "training", inputs=(Port("optimizer", "Any"),), outputs=(Port("scheduler", "Any"),), structural=True)
node("output_node", "training", inputs=("logits",), outputs=(), structural=True, numClasses=10,
     taskType="classification")


# ── Rechnen ─────────────────────────────────────────────────────────────────
def _second(i):
    return i["b"] if "b" in i else i["a"]


node("add_node", "math", inputs=("a", Port("b", fallback="a")), run=lambda m, p, i: _out(i["a"] + i["b"]))
node("multiply_node", "math", inputs=("a", Port("b", fallback="a")), run=lambda m, p, i: _out(i["a"] * i["b"]))
node("matmul", "math", inputs=("a", Port("b", fallback="a")), run=lambda m, p, i: _out(torch.matmul(i["a"], i["b"])))
node("normalize", "math", run=lambda m, p, i: _out(F.normalize(i["in"], p=float(p.get("p", 2)), dim=int(p.get("dim", -1)))),
     p=2, dim=-1)


def _reshape(m, p, i):
    x = i["in"]
    shape = [int(v) for v in str(p.get("shape", "-1, 512")).replace(" ", "").split(",") if v]
    if shape and shape[0] == -1:
        shape = shape[1:]
    return _out(x.reshape(x.size(0), *shape))


node("reshape", "math", run=_reshape, shape="-1, 512")
node("transpose", "math", run=lambda m, p, i: _out(i["in"].transpose(int(p.get("dim0", -2)), int(p.get("dim1", -1)))),
     dim0=-2, dim1=-1)


node("flatten", "math", run=lambda m, p, i: _out(i["in"].flatten(1)))


# ── Logik ───────────────────────────────────────────────────────────────────
node("merge", "logic", inputs=("a", Port("b", fallback="a")),
     run=lambda m, p, i: _out(torch.cat([i["a"], i["b"]], dim=int(p.get("dim", -1)))), dim=-1)


def _split(m, p, i):
    dim = int(p.get("dim", -1))
    parts = torch.chunk(i["in"], int(p.get("chunks", 2)), dim=dim)
    # Ausgang b traegt alles ausser dem ersten Teil, damit bei mehr als zwei
    # Teilen nichts verloren geht.
    return {"a": parts[0], "b": torch.cat(parts[1:], dim=dim) if len(parts) > 1 else parts[0]}


node("split_node", "logic", outputs=("a", "b"), run=_split, chunks=2, dim=-1)


def _pool(m, p, i):
    x, kind = i["in"], str(p.get("type", "global_avg"))
    stride = int(p.get("stride", 2))
    if x.dim() == 3:  # Sequenz [B, T, D]: ueber die Zeit
        return _out(x.max(dim=1).values if "max" in kind else x.mean(dim=1))
    if kind == "avg_2d":
        return _out(F.avg_pool2d(x, 2, stride=stride))
    if kind == "max_2d":
        return _out(F.max_pool2d(x, 2, stride=stride))
    if kind == "global_max":
        return _out(F.adaptive_max_pool2d(x, 1).flatten(1))
    return _out(F.adaptive_avg_pool2d(x, 1).flatten(1))


node("pool", "logic", run=_pool, type="global_avg", stride=2)


def _pool_nd(fn_max, fn_avg, adaptive_max, adaptive_avg):
    def run(m, p, i):
        x, kind = i["in"], str(p.get("type", "max"))
        if kind == "global_avg":
            return _out(adaptive_avg(x, 1).flatten(1))
        if kind == "global_max":
            return _out(adaptive_max(x, 1).flatten(1))
        fn = fn_avg if kind == "avg" else fn_max
        return _out(fn(x, int(p.get("kernel", 2))))
    return run


node("pool1d", "logic", run=_pool_nd(F.max_pool1d, F.avg_pool1d, F.adaptive_max_pool1d, F.adaptive_avg_pool1d),
     type="max", kernel=2)
node("pool3d", "logic", run=_pool_nd(F.max_pool3d, F.avg_pool3d, F.adaptive_max_pool3d, F.adaptive_avg_pool3d),
     type="max", kernel=2)
