"""Datenarten des Synapse Builders: Audio, Video und Text.

Training und Vorhersage laden ein Beispiel ueber dieselbe Funktion. Alles, was
die Vorhersage dafuer braucht, steht im spec-Dict und wird mit den Gewichten
gespeichert.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

MEDIA_TYPES = {"audio_loader": "audio", "video_loader": "video"}
_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAD, UNK, BOS, EOS = 0, 1, 2, 3


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return value is True or str(value).lower() in ("true", "1", "yes")


def step(pipeline: Sequence[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    """Parameter eines Daten-Nodes aus der Pipeline, sonst None."""
    return next((p.get("params") or {} for p in pipeline if p.get("type") == kind), None)


# ── Split ────────────────────────────────────────────────────────────────────
def split_settings(pipeline: Sequence[Dict[str, Any]], default_val: float = 0.2) -> Dict[str, Any]:
    p = step(pipeline, "dataset_split") or {}
    train = float(p.get("trainRatio", 1 - default_val))
    val = float(p.get("valRatio", default_val))
    return {"train": train, "val": val, "shuffle": _flag(p.get("shuffle"), True), "seed": int(p.get("seed", 42))}


def split_indices(n: int, s: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Train und Val; was uebrig bleibt, ist Test und wird nie trainiert."""
    order = np.arange(n)
    if s["shuffle"]:
        order = np.random.RandomState(s["seed"]).permutation(n)
    n_train = max(1, int(n * s["train"]))
    n_val = max(1, int(n * s["val"])) if n > 1 else 0
    n_train = min(n_train, n - n_val)
    return order[:n_train].tolist(), order[n_train:n_train + n_val].tolist()


# ── Audio ────────────────────────────────────────────────────────────────────
def audio_spec(params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mel = step(pipeline, "mel_spectrogram")
    return {
        "kind": "audio", "sample_rate": int(params.get("sampleRate", 16000)),
        "seconds": float(params.get("seconds", 1.0)),
        "mel": None if mel is None else {
            "n_mels": int(mel.get("nMels", 64)), "n_fft": int(mel.get("nFft", 400)),
            "hop": int(mel.get("hopLength", 160))},
    }


def load_audio(path: Any, spec: Dict[str, Any]) -> torch.Tensor:
    try:
        import soundfile as sf
    except ImportError as e:
        from ft_data.deps import missing
        raise missing("soundfile", what="Das Lesen von Audio") from e
    wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(wave.mean(axis=1))
    target = spec["sample_rate"]
    if rate != target:
        # Lineare Interpolation reicht fuers Resampling und braucht kein torchaudio.
        size = int(round(x.numel() * target / rate))
        x = torch.nn.functional.interpolate(x[None, None], size=size, mode="linear", align_corners=False)[0, 0]
    length = int(target * spec["seconds"])
    x = x[:length]
    if x.numel() < length:
        x = torch.nn.functional.pad(x, (0, length - x.numel()))
    x = x.unsqueeze(0)  # [1, S]
    mel = spec.get("mel")
    if mel is None:
        return x
    window = torch.hann_window(mel["n_fft"])
    power = torch.stft(x, mel["n_fft"], mel["hop"], window=window, return_complex=True).abs() ** 2
    return torch.log(_mel_filters(mel["n_fft"], mel["n_mels"], target) @ power + 1e-6)


def _mel_filters(n_fft: int, n_mels: int, rate: int) -> torch.Tensor:
    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    bins = n_fft // 2 + 1
    points = to_hz(np.linspace(to_mel(0.0), to_mel(rate / 2), n_mels + 2))
    freqs = np.linspace(0, rate / 2, bins)
    out = np.zeros((n_mels, bins), dtype="float32")
    for i in range(n_mels):
        left, mid, right = points[i], points[i + 1], points[i + 2]
        up = (freqs - left) / max(mid - left, 1e-6)
        down = (right - freqs) / max(right - mid, 1e-6)
        out[i] = np.clip(np.minimum(up, down), 0, None)
    return torch.from_numpy(out)


# ── Video ────────────────────────────────────────────────────────────────────
def video_spec(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "video", "frames": int(params.get("numFrames", 8)), "size": int(params.get("imageSize", 112))}


def load_video(path: Any, spec: Dict[str, Any]) -> torch.Tensor:
    import cv2
    from ft_data.video import read_clip_frames
    size = spec["size"]
    frames = [cv2.resize(f, (size, size)) for f in read_clip_frames(path, spec["frames"])]
    x = torch.from_numpy(np.stack(frames)).float() / 255.0  # [T, H, W, C]
    return ((x - 0.5) / 0.5).permute(3, 0, 1, 2)  # [C, T, H, W]


def load_file(path: Any, spec: Dict[str, Any]) -> torch.Tensor:
    return load_audio(path, spec) if spec["kind"] == "audio" else load_video(path, spec)


class _Media(Dataset):
    def __init__(self, items: Sequence[Tuple[Any, int]], spec: Dict[str, Any], aug: Optional[Dict[str, Any]]):
        self.items, self.spec, self.aug = list(items), spec, aug

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        x = load_file(path, self.spec)
        if self.aug:
            if self.spec["kind"] == "video" and _flag(self.aug.get("flip")) and torch.rand(1) < 0.5:
                x = x.flip(-1)
            if _flag(self.aug.get("noise")):
                x = x + torch.randn_like(x) * 0.01
        return x, label


def media_loaders(data_type: str, root: str, params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]],
                  batch_size: int, num_classes: int, preprocess: Dict[str, Any], status) -> Tuple[DataLoader, DataLoader]:
    from ft_data.media import resolve_class_layout

    kind = MEDIA_TYPES[data_type]
    s = split_settings(pipeline)
    layout = resolve_class_layout(Path(root), kind, seed=s["seed"], val_fraction=s["val"])
    _check_classes(data_type, len(layout.classes), num_classes)
    spec = audio_spec(params, pipeline) if kind == "audio" else video_spec(params)
    preprocess.update(class_names=[str(c) for c in layout.classes], pipeline=spec)
    status(f"{data_type}: {len(layout.classes)} Klassen | Train: {len(layout.train)} | Val: {len(layout.val)}")
    gen = torch.Generator().manual_seed(s["seed"])
    return (DataLoader(_Media(layout.train, spec, step(pipeline, "augmentation")), batch_size=batch_size,
                       shuffle=True, generator=gen),
            DataLoader(_Media(layout.val, spec, None), batch_size=batch_size))


def _check_classes(name: str, found: int, wanted: int) -> None:
    if wanted and found > int(wanted):
        raise ValueError(f"{name}: Das Dataset hat {found} Klassen, das Modell ist auf {wanted} eingestellt.\n"
                         "Setze die Klassenanzahl im Synapse Builder passend.")


# ── Text ─────────────────────────────────────────────────────────────────────
def tokens_of(text: str, level: str) -> List[str]:
    return list(text) if level == "char" else _WORD.findall(text.lower())


def fit_tokenizer(texts: Sequence[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Eigenes Vokabular aus den Trainingstexten. Id 0 ist Padding, 1 unbekannt."""
    level = str(params.get("level", "word"))
    counts: Counter = Counter()
    for t in texts:
        counts.update(tokens_of(t, level))
    limit = max(int(params.get("vocabSize", 50000)) - 2, 1)
    vocab = {tok: i + 2 for i, (tok, _) in enumerate(counts.most_common(limit))}
    longest = max((len(tokens_of(t, level)) for t in texts), default=1)
    truncate = _flag(params.get("truncation"), True)
    return {"level": level, "vocab": vocab, "pad_batch": not _flag(params.get("padding"), True),
            "max_length": min(int(params.get("maxLength", 512)), longest) if truncate else longest}


def encode(text: str, tok: Dict[str, Any], pad: bool = True) -> List[int]:
    ids = [tok["vocab"].get(t, UNK) for t in tokens_of(text, tok["level"])][:tok["max_length"]] or [PAD]
    return ids + [PAD] * (tok["max_length"] - len(ids)) if pad else ids


def _pad_batch(batch):
    width = max(len(ids) for ids, _ in batch)
    x = torch.tensor([ids + [PAD] * (width - len(ids)) for ids, _ in batch], dtype=torch.long)
    return x, torch.tensor([y for _, y in batch], dtype=torch.long)


def text_loaders(root: str, params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]], batch_size: int,
                 num_classes: int, preprocess: Dict[str, Any], status, read_files) -> Tuple[DataLoader, DataLoader]:
    from ft_data.text import detect_columns

    train_df, val_df = read_files(Path(root))
    text_col, _, label_col = detect_columns(list(train_df.columns))
    text_col = params.get("textCol") or text_col
    label_col = params.get("targetCol") if params.get("targetCol") in train_df.columns else label_col
    if text_col not in train_df.columns or label_col not in train_df.columns:
        raise ValueError(f"text_loader: Text- oder Label-Spalte fehlt. Spalten: {list(train_df.columns)}")
    frames = [f.dropna(subset=[text_col, label_col]) for f in (train_df, val_df) if f is not None]
    names = sorted({str(v) for f in frames for v in f[label_col]})
    _check_classes("text_loader", len(names), num_classes)
    index = {n: i for i, n in enumerate(names)}

    def rows(frame) -> List[Tuple[str, int]]:
        return [(str(t), index[str(y)]) for t, y in zip(frame[text_col], frame[label_col])]

    samples = rows(frames[0])
    if len(frames) > 1:
        train, val = samples, rows(frames[1])
    else:
        tr, va = split_indices(len(samples), split_settings(pipeline))
        train, val = [samples[i] for i in tr], [samples[i] for i in va]
    tok = fit_tokenizer([t for t, _ in train], step(pipeline, "tokenizer") or {})
    preprocess.update(class_names=names, tokenizer=tok, pipeline={"kind": "text"})
    status(f"text_loader: {len(names)} Klassen | Vokabular {len(tok['vocab']) + 2} | Train: {len(train)} | Val: {len(val)}")

    pad = not tok["pad_batch"]

    def dataset(items):
        return [(encode(t, tok, pad), y) for t, y in items]

    gen = torch.Generator().manual_seed(42)
    return (DataLoader(dataset(train), batch_size=batch_size, shuffle=True, generator=gen, collate_fn=_pad_batch),
            DataLoader(dataset(val), batch_size=batch_size, collate_fn=_pad_batch))


def lm_loaders(root: str, pipeline: Sequence[Dict[str, Any]], params: Dict[str, Any], batch_size: int,
               preprocess: Dict[str, Any], status, read_files) -> Tuple[DataLoader, DataLoader]:
    """Sprachmodell: Text wird zu Fenstern, Ziel ist jeweils das naechste Token."""
    from ft_data.text import detect_columns

    root_path = Path(root)
    txt = sorted(root_path.rglob("*.txt")) if root_path.is_dir() else [root_path]
    if txt and txt[0].suffix == ".txt":
        corpus = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in txt)
    else:
        frame, _ = read_files(root_path)
        col = params.get("textCol") or detect_columns(list(frame.columns))[0]
        corpus = "\n".join(str(t) for t in frame[col].dropna())
    t = dict(step(pipeline, "tokenizer") or {})
    t.setdefault("level", "char")
    tok = fit_tokenizer([corpus], {**t, "truncation": False})
    tok["max_length"] = context = int(t.get("maxLength", 64))
    ids = [tok["vocab"].get(c, UNK) for c in tokens_of(corpus, tok["level"])]
    if len(ids) <= context + 1:
        raise ValueError(f"text_loader: Der Text hat nur {len(ids)} Tokens, das Kontextfenster {context}.")
    windows = [(torch.tensor(ids[i:i + context]), torch.tensor(ids[i + 1:i + context + 1]))
               for i in range(0, len(ids) - context - 1, max(context // 2, 1))]
    s = split_settings(pipeline)
    cut = max(1, int(len(windows) * (1 - s["val"])))
    train, val = windows[:cut], windows[cut:] or windows[-1:]
    preprocess.update(tokenizer=tok, pipeline={"kind": "text"}, class_names=[])
    status(f"Sprachmodell: Vokabular {len(tok['vocab']) + 2} | {len(ids)} Tokens | Fenster: {len(train)}/{len(val)}")
    gen = torch.Generator().manual_seed(42)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(val, batch_size=batch_size))


class _Masks(Dataset):
    def __init__(self, pairs, size: int):
        self.pairs, self.size = pairs, size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        from PIL import Image
        img, mask = self.pairs[i]
        size = (self.size, self.size)
        with Image.open(img) as im:
            x = torch.from_numpy(np.asarray(im.convert("RGB").resize(size), dtype="float32") / 255.0).permute(2, 0, 1)
        with Image.open(mask) as m:
            y = torch.from_numpy(np.asarray(m.convert("L").resize(size, Image.NEAREST), dtype="int64").copy())
        return (x - 0.5) / 0.5, y


def mask_loaders(root: str, params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]], batch_size: int,
                 num_classes: int, preprocess: Dict[str, Any], status) -> Tuple[DataLoader, DataLoader]:
    """Segmentierung: images/ und masks/, Dateien gleichen Namens bilden ein Paar."""
    from ft_data.media import IMAGE_EXTS

    base = Path(root)
    base = base / "train" if (base / "train" / "images").is_dir() else base
    images = sorted(f for f in (base / "images").rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    masks = {f.stem: f for f in (base / "masks").rglob("*") if f.suffix.lower() in IMAGE_EXTS}
    pairs = [(f, masks[f.stem]) for f in images if f.stem in masks]
    if not pairs:
        raise ValueError(f"mask_loader: Keine Paare gefunden. Erwartet: {base}/images und {base}/masks mit gleichen Dateinamen.")
    size = int(params.get("imageSize", 128))
    tr, va = split_indices(len(pairs), split_settings(pipeline))
    preprocess.update(pipeline={"kind": "mask", "size": size}, class_names=[])
    status(f"mask_loader: {len(pairs)} Paare | Train: {len(tr)} | Val: {len(va)}")
    gen = torch.Generator().manual_seed(42)
    return (DataLoader(_Masks([pairs[i] for i in tr], size), batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(_Masks([pairs[i] for i in va], size), batch_size=batch_size))


def generate(model: Any, tok: Dict[str, Any], prompt: str, device: str, max_new: int = 80,
             temperature: float = 0.8) -> str:
    """Text fortsetzen, Token fuer Token."""
    back = {i: t for t, i in tok["vocab"].items()}
    ids = [tok["vocab"].get(t, UNK) for t in tokens_of(prompt, tok["level"])] or [UNK]
    context = tok["max_length"]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-context:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            logits[PAD] = float("-inf")
            logits[UNK] = float("-inf")
            if temperature <= 0:
                ids.append(int(logits.argmax()))
            else:
                ids.append(int(torch.multinomial(torch.softmax(logits / temperature, -1), 1)))
    sep = "" if tok["level"] == "char" else " "
    return prompt + sep.join(back.get(i, "") for i in ids[-max_new:])


def seq2seq_loaders(root: str, params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]], batch_size: int,
                    preprocess: Dict[str, Any], status, read_files) -> Tuple[DataLoader, DataLoader]:
    """Text zu Text: Encoder liest die Quelle, Decoder das Ziel um eins versetzt."""
    train_df, val_df = read_files(Path(root))
    src_col, tgt_col = params.get("sourceCol", "source"), params.get("targetCol", "target")
    if src_col not in train_df.columns or tgt_col not in train_df.columns:
        raise ValueError(f"seq2seq_loader: Spalten {src_col}/{tgt_col} fehlen. Vorhanden: {list(train_df.columns)}")
    pairs = [(str(a), str(b)) for a, b in zip(train_df[src_col], train_df[tgt_col])]
    t = dict(step(pipeline, "tokenizer") or {})
    t.setdefault("level", "word")
    tok = fit_tokenizer([a for a, _ in pairs] + [b for _, b in pairs], {**t, "truncation": False})
    tok["vocab"] = {k: v + 2 for k, v in tok["vocab"].items()}  # 2 und 3 sind Start und Ende
    tok["max_length"] = length = int(t.get("maxLength", 32))

    def item(a: str, b: str):
        ids = [tok["vocab"].get(x, UNK) for x in tokens_of(b, tok["level"])][:length - 1]
        dec, gold = [BOS] + ids, ids + [EOS]
        fill = [PAD] * (length - len(dec))
        return (torch.tensor(encode(a, tok)), torch.tensor(dec + fill)), torch.tensor(gold + fill)

    if val_df is not None:
        val_pairs = [(str(a), str(b)) for a, b in zip(val_df[src_col], val_df[tgt_col])]
    else:
        tr, va = split_indices(len(pairs), split_settings(pipeline))
        pairs, val_pairs = [pairs[i] for i in tr], [pairs[i] for i in va]
    preprocess.update(tokenizer=tok, pipeline={"kind": "text"}, class_names=[], task="seq2seq")
    status(f"seq2seq_loader: Vokabular {len(tok['vocab']) + 4} | Train: {len(pairs)} | Val: {len(val_pairs)}")
    gen = torch.Generator().manual_seed(42)
    return (DataLoader([item(a, b) for a, b in pairs], batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader([item(a, b) for a, b in val_pairs], batch_size=batch_size))


def translate(model: Any, tok: Dict[str, Any], text: str, device: str) -> str:
    """Decoder Token fuer Token laufen lassen, bis er das Ende meldet."""
    back = {i: t for t, i in tok["vocab"].items()}
    src = torch.tensor([encode(text, tok)], dtype=torch.long, device=device)
    length = tok["max_length"]
    out = [BOS]
    with torch.no_grad():
        for _ in range(length - 1):
            dec = torch.tensor([out + [PAD] * (length - len(out))], dtype=torch.long, device=device)
            logits = model(src, dec)[0, len(out) - 1]
            next_id = int(logits.argmax())
            if next_id == EOS:
                break
            out.append(next_id)
    sep = "" if tok["level"] == "char" else " "
    return sep.join(back.get(i, "") for i in out[1:])


def split_subset(ds: Dataset, s: Dict[str, Any]) -> Tuple[Subset, Subset]:
    tr, va = split_indices(len(ds), s)
    return Subset(ds, tr), Subset(ds, va)
