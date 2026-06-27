"""Functional graph ops (no registered parameters)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def run_op(node_type: str, params: Dict[str, Any], inputs: List[torch.Tensor]) -> torch.Tensor:
    p = params or {}
    v0 = inputs[0] if inputs else None
    v1 = inputs[1] if len(inputs) > 1 else v0
    if v0 is None:
        raise ValueError(f"Op {node_type} hat keinen Input")

    if node_type == "add_node":
        return v0 + v1
    if node_type == "multiply_node":
        return v0 * v1
    if node_type == "matmul":
        return torch.matmul(v0, v1)
    if node_type == "normalize":
        return F.normalize(v0, p=float(p.get("p", 2)), dim=int(p.get("dim", -1)))
    if node_type == "reshape":
        shp = str(p.get("shape", "-1, 512")).replace(" ", "").replace("-1,", "")
        parts = [int(x) for x in shp.split(",") if x]
        return v0.reshape(v0.size(0), *parts)
    if node_type == "transpose":
        return v0.transpose(int(p.get("dim0", -2)), int(p.get("dim1", -1)))
    if node_type == "merge":
        return torch.cat([v0, v1], dim=int(p.get("dim", -1)))
    if node_type == "split_node":
        parts = torch.chunk(v0, int(p.get("chunks", 2)), dim=int(p.get("dim", -1)))
        return parts[0]
    if node_type == "pool":
        pool_type = str(p.get("type", "global_avg"))
        if "max" in pool_type and "global" in pool_type:
            return F.adaptive_max_pool2d(v0, 1).flatten(1)
        if pool_type == "avg_2d":
            return F.avg_pool2d(v0, 2, stride=int(p.get("stride", 2)))
        if pool_type == "max_2d":
            return F.max_pool2d(v0, 2, stride=int(p.get("stride", 2)))
        return F.adaptive_avg_pool2d(v0, 1).flatten(1)
    raise ValueError(f"Unbekannter Op-Typ: {node_type}")
