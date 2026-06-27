"""DataLoaders for Canvas IR training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

import sys
from pathlib import Path

_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)

from ir import CanvasGraphIR, IRDataSpec  # noqa: E402


def _infer_input_features(ir: CanvasGraphIR) -> int:
    for nid in ir.execution_order:
        node = ir.node_by_id().get(nid)
        if not node:
            continue
        if node.type == "dense":
            return int(node.params.get("inputSize", 128))
        if node.type == "lstm":
            return int(node.params.get("inputSize", 256))
        if node.type == "conv2d":
            return int(node.params.get("inChannels", 3))
        if node.type == "embedding":
            return 32
    return 128


def get_dataloaders(
    ir: CanvasGraphIR,
    dataset_path: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    num_classes = ir.training.num_classes
    input_features = _infer_input_features(ir)
    data = ir.data
    data_type = data.type if data else "default"
    data_params = data.params if data else {}

    dsp = dataset_path or os.environ.get("DATASET_PATH", "")

    if data_type == "image_loader" and dsp and os.path.isdir(dsp):
        try:
            from torchvision import datasets, transforms

            sz = int(data_params.get("imageSize", 224))
            ch = int(str(data_params.get("channels", "3")))
            nrm = data_params.get("normalize", True)
            norm = (
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if nrm
                else transforms.Lambda(lambda x: x)
            )
            tfm = transforms.Compose([
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                norm,
            ])
            ds = datasets.ImageFolder(dsp, transform=tfm)
            n = int(len(ds) * 0.8)
            tr, va = random_split(ds, [n, len(ds) - n])
            return (
                DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0),
                DataLoader(va, batch_size=batch_size, num_workers=0),
            )
        except Exception:
            pass

    if data_type == "csv_loader" and dsp and os.path.isfile(dsp):
        try:
            import pandas as pd

            tgt = str(data_params.get("targetCol", "label"))
            df = pd.read_csv(dsp)
            if tgt in df.columns:
                X = torch.tensor(df.drop(columns=[tgt]).values.astype("float32"))
                y = torch.tensor(df[tgt].values, dtype=torch.long)
            else:
                X = torch.tensor(df.values[:, :-1].astype("float32"))
                y = torch.tensor(df.values[:, -1].astype("int64"), dtype=torch.long)
            ds = TensorDataset(X, y)
            n = int(len(ds) * 0.8)
            tr, va = random_split(ds, [n, len(ds) - n])
            return (
                DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(va, batch_size=batch_size),
            )
        except Exception:
            pass

    # Dummy fallback
    if data_type == "image_loader" or input_features <= 16:
        ch = int(str(data_params.get("channels", "3"))) if data else 3
        sz = int(data_params.get("imageSize", 224)) if data else 224
        X = torch.randn(256, ch, sz, sz)
        y = torch.randint(0, num_classes, (256,))
    else:
        X = torch.randn(512, input_features)
        y = torch.randint(0, num_classes, (512,))

    ds = TensorDataset(X, y)
    return (
        DataLoader(Subset(ds, range(400)), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(ds, range(400, len(ds))), batch_size=batch_size),
    )
