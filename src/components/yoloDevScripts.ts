// Dev-Train- und Dev-Test-Vorlagen fuer YOLO (Ultralytics).
//
// Vorher bekam ein YOLO-Modell in beiden Dev-Bereichen die Text-Vorlage mit
// AutoModelForSequenceClassification und load_dataset — die beim ersten Lauf
// abstuerzte. Ultralytics braucht die dataset.yaml (DATASET_YAML), nicht den
// Dataset-Ordner. FrameTrain prueft und repariert die yaml vor dem Start.

export interface YoloScriptPaths {
  modelPath: string;
  datasetPath: string;
  datasetYaml: string;
  outputPath: string;
}

// Gemeinsamer Kopf: Pfade, Events, yaml- und Gewichtssuche.
function header(kind: 'Train' | 'Test', p: YoloScriptPaths, intro: string): string {
  return `#!/usr/bin/env python3
# FrameTrain - Dev ${kind} Script (YOLO / Ultralytics)
#
${intro}
#
# Ultralytics bekommt die dataset.yaml (DATASET_YAML), nicht den Ordner.
# FrameTrain prueft die yaml vor dem Start: path, train und val stimmen.

import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH   = os.environ.get("MODEL_PATH",   "${p.modelPath}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${p.datasetPath}")
DATASET_YAML = os.environ.get("DATASET_YAML", "${p.datasetYaml}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${p.outputPath}")


def emit(kind: str, **data):
    # Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event).
    print(json.dumps({"type": kind, "data": data}), flush=True)


def find_yaml() -> str:
    if DATASET_YAML and Path(DATASET_YAML).is_file():
        return DATASET_YAML
    for name in ("dataset.yaml", "data.yaml"):
        candidate = Path(DATASET_PATH) / name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Keine dataset.yaml in {DATASET_PATH}. Lege sie im Dataset unter 'dataset.yaml' an."
    )


def find_weights() -> str:
    # MODEL_PATH ist meist ein Ordner mit einer oder mehreren .pt-Dateien.
    path = Path(MODEL_PATH)
    if path.is_file():
        return str(path)
    weights = sorted(path.glob("*.pt")) if path.is_dir() else []
    # Fuer Objekterkennung Gewichte ohne -seg/-pose/-cls/-obb bevorzugen.
    plain = [w for w in weights if not w.stem.endswith(("-seg", "-pose", "-cls", "-obb"))] or weights
    if not plain:
        raise FileNotFoundError(f"Keine .pt-Gewichte in {MODEL_PATH}")
    return str(min(plain, key=lambda w: w.stat().st_size))


def box_metrics(results) -> dict:
    m = dict(getattr(results, "results_dict", None) or {})
    return {
        "mAP50": float(m.get("metrics/mAP50(B)", 0.0) or 0.0),
        "mAP50-95": float(m.get("metrics/mAP50-95(B)", 0.0) or 0.0),
        "precision": float(m.get("metrics/precision(B)", 0.0) or 0.0),
        "recall": float(m.get("metrics/recall(B)", 0.0) or 0.0),
    }


data_yaml = find_yaml()
weights = find_weights()
print(f"dataset.yaml: {data_yaml}", flush=True)
print(f"Gewichte:     {weights}", flush=True)
`;
}

export function generateYoloTrainScript(p: YoloScriptPaths): string {
  return `${header('Train', p,
    '# Trainiert das YOLO-Modell mit Ultralytics. Fortschritt erscheint in FrameTrain,\n' +
    '# sobald eine JSON-Zeile mit {"type": "progress", ...} auf stdout geschrieben wird.')}
# -- Hyperparameter -------------------------------------------------------
EPOCHS = 3       # zum Ausprobieren klein halten
BATCH  = 8
IMGSZ  = 640     # Bildgroesse; RAM waechst quadratisch mit der Kantenlaenge
LR0    = 0.01


def loss_sum(items, prefix: str):
    vals = [float(v) for k, v in (items or {}).items() if k.startswith(prefix) and k.endswith("_loss")]
    return sum(vals) if vals else None


def on_fit_epoch_end(trainer):
    # Laeuft nach Training UND Validierung einer Epoche.
    epoch, total = trainer.epoch + 1, trainer.epochs
    if epoch > total:
        return  # Ultralytics feuert nach der finalen Validierung noch einmal
    train_items = trainer.label_loss_items(trainer.tloss) if trainer.tloss is not None else {}
    lrs = list((getattr(trainer, "lr", None) or {}).values())
    emit(
        "progress",
        epoch=epoch,
        total_epochs=total,
        step=epoch,
        total_steps=total,
        train_loss=loss_sum(train_items, "train/"),
        val_loss=loss_sum(trainer.metrics, "val/"),
        learning_rate=float(lrs[0]) if lrs else LR0,
        metrics={"mAP50": float((trainer.metrics or {}).get("metrics/mAP50(B)", 0.0) or 0.0)},
    )


emit("status", stage="loading", message="Modell wird geladen ...")
model = YOLO(weights)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

emit("status", stage="training", message="Training laeuft ...")
results = model.train(
    data=data_yaml,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    lr0=LR0,
    project=OUTPUT_PATH,
    name="train",
    exist_ok=True,
    plots=False,
    verbose=False,
)

# -- Speichern ------------------------------------------------------------
weights_dir = Path(model.trainer.save_dir) / "weights"
best = next((w for w in (weights_dir / "best.pt", weights_dir / "last.pt") if w.is_file()), None)
if best is None:
    raise RuntimeError(f"Kein Checkpoint in {weights_dir} - das Training hat nichts gespeichert.")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
shutil.copy2(best, Path(OUTPUT_PATH) / "model.pt")
print(f"Modell gespeichert unter {OUTPUT_PATH}/model.pt", flush=True)

emit(
    "complete",
    model_path=OUTPUT_PATH,
    final_metrics={**box_metrics(results), "total_epochs": EPOCHS},
)
`;
}

export function generateYoloTestScript(p: YoloScriptPaths): string {
  return `${header('Test', p,
    '# Wertet das YOLO-Modell auf dem Test-Split aus (sonst Val) und schreibt\n' +
    '# den Bericht nach OUTPUT_PATH/results.json.')}
IMGSZ = 640
BATCH = 8


def has_split(yaml_path: str, name: str) -> bool:
    for line in Path(yaml_path).read_text(encoding="utf-8").splitlines():
        key, _, value = line.split(" #")[0].partition(":")
        if key.strip() == name and value.strip():
            return True
    return False


split = "test" if has_split(data_yaml, "test") else "val"
print(f"Werte Split '{split}' aus", flush=True)

model = YOLO(weights)
results = model.val(
    data=data_yaml,
    split=split,
    imgsz=IMGSZ,
    batch=BATCH,
    project=OUTPUT_PATH,
    name="val",
    exist_ok=True,
    plots=False,
)

report = {"split": split, "weights": weights, "dataset_yaml": data_yaml, **box_metrics(results)}
names = getattr(results, "names", None) or {}
maps = getattr(getattr(results, "box", None), "maps", None)
if maps is not None:
    report["mAP50-95_per_class"] = {names.get(i, str(i)): float(v) for i, v in enumerate(maps)}

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_PATH) / "results.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print(f"Bericht gespeichert unter {OUTPUT_PATH}/results.json", flush=True)
`;
}
