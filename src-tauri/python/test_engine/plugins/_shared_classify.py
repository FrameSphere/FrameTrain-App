"""
Gemeinsame Bausteine für klassifizierende Test-Plugins (Bild, Audio).

Beide unterscheiden sich nur darin, wie eine Datei zu Modell-Eingaben wird —
der Rest (Label-Zuordnung, Dataset-Durchlauf, Ergebnisdatei, Kennzahlen) ist
identisch und liegt deshalb hier.
"""
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.protocol import TestProtocol


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


def collect_class_files(root: Path, extensions: set) -> List[Tuple[Path, Optional[str]]]:
    """
    Sammelt Dateien samt erwarteter Klasse.

    Unterstützt das Trainingslayout (Ordner pro Klasse, optional unter
    train/ val/ test/) und einen flachen Ordner ohne Klassenzuordnung.
    """
    for sub in ("test", "val", "validation", "train"):
        if (root / sub).is_dir():
            root = root / sub
            break

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    files: List[Tuple[Path, Optional[str]]] = []
    if class_dirs:
        for d in class_dirs:
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in extensions:
                    files.append((f, d.name))
    else:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in extensions:
                files.append((f, None))
    return files


def run_dataset_classification(
    plugin,
    extensions: set,
    predict: Callable[[Path], Tuple[str, float, List[Dict[str, Any]]]],
    kind_label: str,
) -> None:
    """Durchläuft ein Dataset und schreibt Ergebnisse + Kennzahlen."""
    root = Path(plugin.config.dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

    files = collect_class_files(root, extensions)
    if not files:
        raise ValueError(
            f"Keine {kind_label} in '{root}' gefunden. "
            f"Erwartet werden: {', '.join(sorted(extensions))}"
        )
    if plugin.config.max_samples:
        files = files[: int(plugin.config.max_samples)]

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

    out_dir = Path(plugin.config.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"
    results_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = max(time.time() - started, 1e-6)
    TestProtocol.complete_dataset(
        results_file=str(results_file),
        total_samples=len(rows),
        # Ohne Klassenordner gibt es nichts zu vergleichen — dann lieber keine
        # Accuracy melden als eine erfundene 0.
        accuracy=(correct / labelled) if labelled else None,
        correct=correct if labelled else None,
        average_loss=None,
        average_inference_time=total_time / max(len(rows), 1),
        samples_per_second=len(rows) / elapsed,
    )
