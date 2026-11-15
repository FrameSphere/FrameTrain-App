"""mAP fuer Objekterkennung ohne pycocotools.

Gleiche Definition wie COCO: je Klasse und IoU-Schwelle (0.50, 0.55 … 0.95)
werden die Vorhersagen nach Sicherheit sortiert und gierig der besten noch
freien Box gleicher Klasse zugeordnet; die Precision-Recall-Kurve wird an 101
Recall-Punkten interpoliert. mAP50 ist der Wert bei IoU 0.50, mAP50-95 der
Mittelwert ueber alle zehn Schwellen. Klassen ohne Box im Ground Truth zaehlen
nicht mit (wie bei COCO und Ultralytics).

Boxen sind immer [x1, y1, x2, y2] in Pixeln.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

Box = Sequence[float]


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ap_101(recalls: List[float], precisions: List[float]) -> float:
    """COCO-AP: Precision monoton machen, an 101 Recall-Punkten abtasten."""
    if not recalls:
        return 0.0
    prec = list(precisions)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    total, j = 0.0, 0
    for k in range(101):
        r = k / 100
        while j < len(recalls) and recalls[j] < r:
            j += 1
        total += prec[j] if j < len(recalls) else 0.0
    return total / 101


def _class_ap(preds: List[Tuple[float, int, Box]], gts: Dict[int, List[Box]], thr: float) -> Optional[float]:
    """AP einer Klasse. preds: (score, bild, box); gts: bild -> Boxen."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return None
    used = {img: [False] * len(boxes) for img, boxes in gts.items()}
    tp_cum = fp_cum = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for score, img, box in sorted(preds, key=lambda p: -p[0]):
        best, best_j = thr, -1
        for j, g in enumerate(gts.get(img, [])):
            if used[img][j]:
                continue
            iou = box_iou(box, g)
            if iou >= best:
                best, best_j = iou, j
        if best_j >= 0:
            used[img][best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / n_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))
    return _ap_101(recalls, precisions)


def detection_metrics(predictions: Iterable[Iterable[dict]], ground_truth: Iterable[Iterable[dict]],
                      conf: float = 0.25) -> Dict[str, float]:
    """mAP50, mAP50-95, Precision und Recall ueber alle Bilder.

    predictions[i] / ground_truth[i] gehoeren zum selben Bild. Eine Vorhersage
    ist {"label", "score" (oder "confidence"), "box": [x1,y1,x2,y2]}, ein
    Ground-Truth-Eintrag {"label", "box"}. Precision/Recall gelten bei IoU 0.5
    fuer Vorhersagen ab `conf` (wie die Boxen, die der Nutzer sieht).
    """
    preds_by_cls: Dict[str, List[Tuple[float, int, Box]]] = {}
    gts_by_cls: Dict[str, Dict[int, List[Box]]] = {}
    for img, (preds, gts) in enumerate(zip(predictions, ground_truth)):
        for g in gts:
            gts_by_cls.setdefault(str(g["label"]), {}).setdefault(img, []).append(list(g["box"]))
        for p in preds:
            score = float(p.get("score", p.get("confidence", 0.0)))
            preds_by_cls.setdefault(str(p["label"]), []).append((score, img, list(p["box"])))

    per_thr: Dict[float, List[float]] = {t: [] for t in IOU_THRESHOLDS}
    for cls, gts in gts_by_cls.items():
        for t in IOU_THRESHOLDS:
            ap = _class_ap(preds_by_cls.get(cls, []), gts, t)
            if ap is not None:
                per_thr[t].append(ap)

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    map50 = mean(per_thr[0.5])
    map50_95 = mean([mean(per_thr[t]) for t in IOU_THRESHOLDS]) if gts_by_cls else 0.0

    # Precision/Recall bei IoU 0.5 ab der Anzeigeschwelle
    tp = fp = 0
    n_gt = sum(len(b) for g in gts_by_cls.values() for b in g.values())
    for cls, preds in preds_by_cls.items():
        kept = [p for p in preds if p[0] >= conf]
        gts = gts_by_cls.get(cls, {})
        used = {img: [False] * len(boxes) for img, boxes in gts.items()}
        for score, img, box in sorted(kept, key=lambda p: -p[0]):
            best, best_j = 0.5, -1
            for j, g in enumerate(gts.get(img, [])):
                if not used[img][j]:
                    iou = box_iou(box, g)
                    if iou >= best:
                        best, best_j = iou, j
            if best_j >= 0:
                used[img][best_j] = True
                tp += 1
            else:
                fp += 1
    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
    }
