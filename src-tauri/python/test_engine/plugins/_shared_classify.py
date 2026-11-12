"""
Gemeinsame Bausteine für klassifizierende Test-Plugins (Bild, Audio, Video, Text).

Sie unterscheiden sich nur darin, wie eine Datei zu Modell-Eingaben wird —
der Rest (Label-Zuordnung, Dataset-Durchlauf, Ergebnisdatei, Kennzahlen) ist
identisch und liegt deshalb hier.

Kennzahlen (classification_metrics): Accuracy allein sagt bei ungleich
verteilten Klassen wenig. Dazu kommen Macro-F1/-Precision/-Recall, die Werte
je Klasse und eine Konfusionsmatrix. Rein in Python gerechnet (kein sklearn
noetig), Ergebnis identisch zu sklearn mit labels=<gesehene Klassen> und
zero_division=0.
"""
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.protocol import TestProtocol


# Fehlgeschlagene Vorhersage (Ausnahme im Plugin) — keine Klasse, zaehlt aber
# als verpasster Treffer der erwarteten Klasse.
ERROR_LABEL = "ERROR"
# Groesser wird die Matrix in results.json nicht: bei 1000 Klassen waeren das
# eine Million Zellen in der Test-Historie. Darueber bleiben die Klassen, die
# am haeufigsten verwechselt werden.
MATRIX_MAX_CLASSES = 20
TOP_CONFUSIONS = 10


def _r(x: float) -> float:
    return round(float(x), 4)


def classification_metrics(
    expected: Sequence[Optional[str]],
    predicted: Sequence[Optional[str]],
    max_matrix_classes: int = MATRIX_MAX_CLASSES,
    top_confusions: int = TOP_CONFUSIONS,
) -> Dict[str, Any]:
    """Accuracy, Macro-F1/-Precision/-Recall, Werte je Klasse, Konfusionsmatrix.

    Paare ohne erwartete Klasse fallen weg. Eine Vorhersage "ERROR" ist keine
    eigene Klasse (sie wuerde den Macro-Schnitt mit einer 0 drücken), zaehlt
    aber als falsch negativ fuer die erwartete Klasse — wie sklearn mit
    ``labels=`` ohne diese Marke.

    Rueckgabe (leer, wenn nichts gelabelt ist):
      accuracy, macro_f1, macro_precision, macro_recall, weighted_f1,
      per_class_f1 {klasse: f1}            (Feld gab es schon im Textplugin)
      per_class {klasse: {precision, recall, f1, support}}
      confusion_matrix {labels, matrix[zeile=erwartet][spalte=vorhergesagt],
                        total_classes, truncated}
      top_confusions [{expected, predicted, count}]  (haeufigste Verwechslungen)
    """
    pairs = [(str(e), ERROR_LABEL if p is None else str(p))
             for e, p in zip(expected, predicted) if e is not None]
    if not pairs:
        return {}
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs if p != ERROR_LABEL})
    index = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    missed = [0] * n  # ERROR-Vorhersagen je erwarteter Klasse
    for e, p in pairs:
        if p in index:
            cm[index[e]][index[p]] += 1
        else:
            missed[index[e]] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    raw = []  # (precision, recall, f1, support) ungerundet — gemittelt wird vor dem Runden
    for c, i in index.items():
        tp = cm[i][i]
        support = sum(cm[i]) + missed[i]
        pred_n = sum(cm[r][i] for r in range(n))
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": _r(prec), "recall": _r(rec), "f1": _r(f1), "support": support}
        raw.append((prec, rec, f1, support))

    total = len(pairs)
    correct = sum(cm[i][i] for i in range(n))
    out: Dict[str, Any] = {
        "accuracy": _r(correct / total),
        "macro_f1": _r(sum(x[2] for x in raw) / n),
        "macro_precision": _r(sum(x[0] for x in raw) / n),
        "macro_recall": _r(sum(x[1] for x in raw) / n),
        "weighted_f1": _r(sum(x[2] * x[3] for x in raw) / total),
        "per_class_f1": {c: v["f1"] for c, v in per_class.items()},
        "per_class": per_class,
    }

    confusions = sorted(
        ((cm[i][j], labels[i], labels[j]) for i in range(n) for j in range(n) if i != j and cm[i][j]),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    out["top_confusions"] = [
        {"expected": e, "predicted": p, "count": c} for c, e, p in confusions[:top_confusions]
    ]

    keep = list(range(n))
    if n > max_matrix_classes:
        # Klassen nach Beteiligung an Verwechslungen (als Ziel und als Fehlgriff),
        # dann nach Haeufigkeit — die Diagonale allein ist uninteressant.
        def weight(i: int):
            off = sum(cm[i]) - cm[i][i] + sum(cm[r][i] for r in range(n)) - cm[i][i]
            return (-off, -per_class[labels[i]]["support"], labels[i])
        keep = sorted(sorted(range(n), key=weight)[:max_matrix_classes])
    out["confusion_matrix"] = {
        "labels": [labels[i] for i in keep],
        "matrix": [[cm[i][j] for j in keep] for i in keep],
        "total_classes": n,
        "truncated": len(keep) < n,
    }
    return out


def resolve_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_label_names(model_path: Path, model_config) -> Dict[int, str]:
    """Klassennamen aus label_mapping.json oder id2label des Modells."""
    mapping = model_path / "label_mapping.json"
    if mapping.exists():
        try:
            classes = json.loads(mapping.read_text(encoding="utf-8")).get("classes")
            if classes:
                return {i: str(c) for i, c in enumerate(classes)}
        except Exception:
            pass
    id2label = getattr(model_config, "id2label", None) or {}
    return {int(k): str(v) for k, v in id2label.items()} if id2label else {}


def collect_class_files(root: Path, kind: str) -> List[Tuple[Path, Optional[str]]]:
    """
    Sammelt Dateien samt erwarteter Klasse (gemeinsame Regeln aus ft_data.media).

    test/ vor val/ vor train/ — leere Split-Ordner zaehlen nicht (der App-Split
    legt test/<klasse>/ auch bei 0 % Testanteil an). HF-Parquet mit Bild- oder
    Audiospalte wird einmalig in Klassenordner entpackt.
    """
    from ft_data.media import evaluation_files
    files, split = evaluation_files(root, kind, status=lambda m: TestProtocol.status("loading", m))
    if split:
        TestProtocol.status("loading", f"Verwende Split: {split}/")
    return files


def run_dataset_classification(
    plugin,
    kind: str,
    predict: Callable[[Path], Tuple[str, float, List[Dict[str, Any]]]],
    kind_label: str,
) -> None:
    """Durchläuft ein Dataset und schreibt Ergebnisse + Kennzahlen."""
    root = Path(plugin.config.dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

    from ft_data.media import exts_for, sample
    files = collect_class_files(root, kind)
    if not files:
        raise ValueError(
            f"Keine {kind_label} in '{root}' gefunden. "
            f"Erwartet werden: {', '.join(sorted(exts_for(kind)))}"
        )
    # Zufaellige Stichprobe statt der ersten N: die Liste ist nach Klasse
    # sortiert, die ersten N enthielten sonst oft nur eine Klasse.
    files = sample(files, plugin.config.max_samples)

    TestProtocol.status("running", f"{len(files)} Dateien werden ausgewertet...")

    rows: List[Dict[str, Any]] = []
    correct = 0
    labelled = 0
    total_time = 0.0
    started = time.time()

    for idx, (path, expected) in enumerate(files, start=1):
        if plugin.is_stopped:
            TestProtocol.status("stopped", "Test abgebrochen.")
            return
        t0 = time.time()
        predicted, confidence, _top = predict(path)
        dt = time.time() - t0
        total_time += dt

        is_correct: Optional[bool] = None
        if expected is not None:
            labelled += 1
            is_correct = (str(predicted) == str(expected))
            if is_correct:
                correct += 1

        rows.append({
            "sample_id": idx,
            "input_text": path.name,
            "expected_output": expected,
            "predicted_output": predicted,
            "is_correct": is_correct,
            "confidence": confidence,
            "inference_time": dt,
        })

        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.progress(current=idx, total=len(files), sps=idx / elapsed)

    elapsed = max(time.time() - started, 1e-6)
    metrics = classification_metrics(
        [r["expected_output"] for r in rows], [r["predicted_output"] for r in rows])
    write_classification_results(plugin, rows, metrics, elapsed,
                                 correct=correct if labelled else None,
                                 total_time=total_time)


def write_classification_results(plugin, rows: List[Dict[str, Any]], metrics: Dict[str, Any],
                                 elapsed: float, correct: Optional[int], total_time: float,
                                 average_loss: Optional[float] = None) -> None:
    """results.json ({predictions, metrics}) schreiben und den Abschluss melden.

    Die Kennzahlen stehen in results.json (daraus liest die App Test-Historie,
    Benchmark-Vergleich und das MCP-Werkzeug tests_results) und in der
    Abschlussmeldung (daraus zeigt die Test-Ansicht sie sofort an).
    """
    out_dir = Path(plugin.config.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"
    accuracy = metrics.get("accuracy")
    file_metrics = {
        "accuracy": accuracy,
        "correct_predictions": correct,
        "total_samples": len(rows),
        "average_inference_time": total_time / max(len(rows), 1),
        "samples_per_second": len(rows) / elapsed,
        "total_time": elapsed,
        **metrics,
    }
    results_file.write_text(json.dumps({"predictions": rows, "metrics": file_metrics},
                                       indent=2, ensure_ascii=False), encoding="utf-8")

    hard = [r for r in rows if r.get("is_correct") is False and r.get("expected_output") is not None]
    hard_file = None
    if hard:
        hard_file = str(out_dir / "hard_examples.json")
        Path(hard_file).write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")

    TestProtocol.complete_dataset(
        results_file=str(results_file),
        total_samples=len(rows),
        # Ohne Klassenordner gibt es nichts zu vergleichen — dann lieber keine
        # Accuracy melden als eine erfundene 0.
        accuracy=accuracy,
        correct=correct,
        average_loss=average_loss,
        average_inference_time=total_time / max(len(rows), 1),
        samples_per_second=len(rows) / elapsed,
        hard_examples_file=hard_file,
        metrics=metrics or None,
    )
