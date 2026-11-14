"""tabular_core – gemeinsamer Kern fuer Tabellen-Klassifikation und -Regression.

Train-Engine, Test-Engine (und spaeter Hosting) importieren genau dieses Modul,
damit Vorverarbeitung, Netz und Kennzahlen an einer Stelle stehen. Die
Vorverarbeitung (Skalierung, Kategorie-Tabellen, Zielklassen) wird als JSON
neben die Gewichte geschrieben; wer das Modell laedt, bekommt dieselbe
Transformation wie beim Training.

Kein sklearn: die Kennzahlen sind kurz genug, um sie selbst zu rechnen, und
das Plugin bleibt mit torch + pandas + numpy lauffaehig.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".parquet", ".jsonl", ".json")
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "dev": "val", "test": "test", "testing": "test"}
TARGET_NAMES = ("label", "target", "y", "class", "klasse", "ziel")
PREPROCESSING_FILE = "preprocessing.json"
WEIGHTS_FILE = "model.pt"
MODEL_TYPE = "ft_tabular"


# ─── Dateien finden und lesen ──────────────────────────────────────────────

def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    # CSV: Trennzeichen erraten (Semikolon ist in deutschen Exporten ueblich)
    return pd.read_csv(path, sep=None, engine="python")


def _split_of(path: Path, root: Path) -> Optional[str]:
    rel = path.relative_to(root)
    for part in list(rel.parts[:-1]) + [path.stem]:
        key = part.lower()
        if key in SPLIT_NAMES:
            return SPLIT_NAMES[key]
        for name, split in SPLIT_NAMES.items():
            if key.startswith(name + "_") or key.endswith("_" + name) or key.endswith("-" + name):
                return split
    return None


def find_table_files(dataset_path: str) -> Dict[str, List[Path]]:
    """{'train': [...], 'val': [...], 'test': [...], 'all': [...]} — 'all' = ohne Split-Hinweis."""
    root = Path(dataset_path)
    if root.is_file():
        return {"all": [root]}
    if not root.exists():
        raise FileNotFoundError(f"Dataset-Pfad nicht gefunden: {dataset_path}")
    out: Dict[str, List[Path]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TABLE_EXTS or p.name.startswith("."):
            continue
        if p.name in ("dataset_info.json", "metadata.json", "preprocessing.json", "config.json"):
            continue
        out.setdefault(_split_of(p, root) or "all", []).append(p)
    if not out:
        raise ValueError(f"Keine Tabellendatei (CSV, TSV, Parquet, JSONL) in {dataset_path} gefunden.")
    return out


def load_frames(dataset_path: str) -> Dict[str, pd.DataFrame]:
    files = find_table_files(dataset_path)
    return {split: pd.concat([read_table(p) for p in paths], ignore_index=True)
            for split, paths in files.items()}


# ─── Ziel und Aufgabe ──────────────────────────────────────────────────────

def pick_target(df: pd.DataFrame, target_column: str = "") -> str:
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Zielspalte '{target_column}' fehlt. Vorhanden: {list(df.columns)}")
        return target_column
    lower = {c.lower(): c for c in df.columns}
    for name in TARGET_NAMES:
        if name in lower:
            return lower[name]
    return df.columns[-1]


def infer_task(y: pd.Series, requested: str = "auto") -> str:
    if requested in ("classification", "regression"):
        return requested
    y = y.dropna()
    if y.empty:
        raise ValueError("Die Zielspalte enthaelt keine Werte.")
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    values = y.to_numpy(dtype=float)
    n_unique = len(np.unique(values))
    integer_like = np.all(np.isclose(values, np.round(values)))
    # Ganze Zahlen mit wenigen Auspraegungen sind Klassen (0/1, 1..5 Sterne).
    if integer_like and n_unique <= max(2, min(20, int(len(values) * 0.05) + 2)):
        return "classification"
    return "regression"


# ─── Vorverarbeitung ───────────────────────────────────────────────────────

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _cat_key(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    s = str(v).strip()
    return s or None


class Preprocessor:
    """Zahlen: Median-Imputation + Standardisierung + Fehlt-Indikator.
    Kategorien: Index-Tabelle, 0 = fehlend/unbekannt. Ziel: Klassenliste bzw. Mittel/Std."""

    MAX_CATEGORIES = 1000

    def __init__(self) -> None:
        self.target: str = ""
        self.task: str = "classification"
        self.numeric: List[Dict[str, Any]] = []      # {name, median, mean, std, has_missing}
        self.categorical: List[Dict[str, Any]] = []  # {name, vocab: [..]}
        self.dropped: List[str] = []
        self.classes: List[str] = []
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    # ── Fit ──
    def fit(self, df: pd.DataFrame, target: str, task: str = "auto") -> "Preprocessor":
        self.target = target
        y = df[target]
        self.task = infer_task(y, task)
        n = len(df)
        for col in df.columns:
            if col == target:
                continue
            s = df[col]
            if s.isna().all():
                self.dropped.append(col)
                continue
            if _is_numeric(s):
                v = _to_float(s)
                med = float(v.median())
                filled = v.fillna(med)
                std = float(filled.std(ddof=0)) or 1.0
                self.numeric.append({"name": col, "median": med, "mean": float(filled.mean()),
                                     "std": std, "has_missing": bool(v.isna().any())})
                continue
            keys = [k for k in (_cat_key(x) for x in s) if k is not None]
            n_unique = len(set(keys))
            # Freitext oder IDs: fast jede Zeile eigen — als Kategorie nutzlos.
            if n > 20 and n_unique > 0.9 * len(keys):
                self.dropped.append(col)
                continue
            counts = pd.Series(keys).value_counts()
            vocab = [str(k) for k in counts.index[: self.MAX_CATEGORIES]]
            self.categorical.append({"name": col, "vocab": vocab})
        if not self.numeric and not self.categorical:
            raise ValueError("Keine nutzbare Eingabespalte neben der Zielspalte gefunden.")
        if self.task == "classification":
            keys = sorted({k for k in (_cat_key(v) for v in y) if k is not None}, key=_natural)
            if len(keys) < 2:
                raise ValueError("Klassifikation braucht mindestens zwei verschiedene Zielwerte.")
            self.classes = keys
        else:
            v = _to_float(y).dropna()
            self.target_mean = float(v.mean())
            self.target_std = float(v.std(ddof=0)) or 1.0
        return self

    # ── Abmessungen ──
    @property
    def num_dim(self) -> int:
        return len(self.numeric) + sum(1 for c in self.numeric if c["has_missing"])

    @property
    def cardinalities(self) -> List[int]:
        return [len(c["vocab"]) + 1 for c in self.categorical]

    @property
    def output_dim(self) -> int:
        return len(self.classes) if self.task == "classification" else 1

    @property
    def feature_columns(self) -> List[str]:
        return [c["name"] for c in self.numeric] + [c["name"] for c in self.categorical]

    # ── Transform ──
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(df)
        num_parts: List[np.ndarray] = []
        miss_parts: List[np.ndarray] = []
        for c in self.numeric:
            v = _to_float(df[c["name"]]) if c["name"] in df.columns else pd.Series([np.nan] * n)
            miss = v.isna().to_numpy()
            filled = v.fillna(c["median"]).to_numpy(dtype=float)
            num_parts.append((filled - c["mean"]) / c["std"])
            if c["has_missing"]:
                miss_parts.append(miss.astype(float))
        x_num = np.stack(num_parts + miss_parts, axis=1).astype(np.float32) if (num_parts or miss_parts) \
            else np.zeros((n, 0), dtype=np.float32)
        # Ausreisser im Test (z. B. 1000-fache Werte) sollen das Netz nicht sprengen.
        x_num = np.clip(np.nan_to_num(x_num), -20, 20)
        cat_parts = []
        for c in self.categorical:
            index = {k: i + 1 for i, k in enumerate(c["vocab"])}
            col = df[c["name"]] if c["name"] in df.columns else pd.Series([None] * n)
            cat_parts.append(np.array([index.get(_cat_key(v), 0) for v in col], dtype=np.int64))
        x_cat = np.stack(cat_parts, axis=1) if cat_parts else np.zeros((n, 0), dtype=np.int64)
        return x_num, x_cat

    def transform_target(self, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """(Zielwerte, Maske gueltiger Zeilen). Klassen als Index, Regression normalisiert."""
        if self.task == "classification":
            index = {k: i for i, k in enumerate(self.classes)}
            idx = np.array([index.get(_cat_key(v), -1) for v in y], dtype=np.int64)
            return idx, idx >= 0
        v = _to_float(y).to_numpy()
        mask = ~np.isnan(v)
        return ((np.nan_to_num(v) - self.target_mean) / self.target_std).astype(np.float32), mask

    def decode_regression(self, z: np.ndarray) -> np.ndarray:
        return z * self.target_std + self.target_mean

    # ── Speichern ──
    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "target": self.target, "task": self.task, "numeric": self.numeric,
                "categorical": self.categorical, "dropped": self.dropped, "classes": self.classes,
                "target_mean": self.target_mean, "target_std": self.target_std}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Preprocessor":
        p = cls()
        for k in ("target", "task", "numeric", "categorical", "dropped", "classes",
                  "target_mean", "target_std"):
            setattr(p, k, d[k])
        return p


def _natural(k: str):
    try:
        return (0, float(k), k)
    except ValueError:
        return (1, 0.0, k)


# ─── Aufteilung ────────────────────────────────────────────────────────────

def random_split(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministisch: dieselbe Datei und derselbe Seed ergeben im Test dieselbe Aufteilung."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    if n - n_test - n_val < 1:
        raise ValueError(f"Zu wenige Zeilen ({n}) fuer die Aufteilung.")
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def resolve_splits(frames: Dict[str, pd.DataFrame], val_ratio: float, test_ratio: float,
                   seed: int) -> Dict[str, pd.DataFrame]:
    """Vorhandene train/val/test-Dateien gelten; sonst seeded Zufallsaufteilung aus 'all'."""
    if "train" in frames:
        out = {"train": frames["train"]}
        if "all" in frames:
            out["train"] = pd.concat([frames["train"], frames["all"]], ignore_index=True)
        if "val" in frames:
            out["val"] = frames["val"]
        if "test" in frames:
            out["test"] = frames["test"]
        if "val" not in out:
            tr, va, _ = random_split(len(out["train"]), val_ratio, 0.0, seed)
            base = out["train"]
            out["train"], out["val"] = base.iloc[tr].reset_index(drop=True), base.iloc[va].reset_index(drop=True)
        return out
    base = pd.concat(list(frames.values()), ignore_index=True)
    tr, va, te = random_split(len(base), val_ratio, test_ratio, seed)
    return {"train": base.iloc[tr].reset_index(drop=True), "val": base.iloc[va].reset_index(drop=True),
            "test": base.iloc[te].reset_index(drop=True)}


# ─── Netz ──────────────────────────────────────────────────────────────────

def build_model(prep: Preprocessor, hidden: List[int], dropout: float):
    import torch
    from torch import nn

    class TabularNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeds = nn.ModuleList(
                nn.Embedding(card, min(32, (card + 1) // 2 + 1)) for card in prep.cardinalities)
            in_dim = prep.num_dim + sum(e.embedding_dim for e in self.embeds)
            layers: List[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                in_dim = h
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(in_dim, prep.output_dim)

        def forward(self, x_num, x_cat):
            parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
            return self.head(self.body(torch.cat(parts, dim=1)))

    return TabularNet()


def save_bundle(out_dir: Path, model, prep: Preprocessor, arch: Dict[str, Any], extra: Dict[str, Any]) -> None:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / WEIGHTS_FILE)
    (out_dir / PREPROCESSING_FILE).write_text(json.dumps(prep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"model_type": MODEL_TYPE, "frametrain_task": "tabular", "task": prep.task, **arch}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # infer_task_type in Rust liest model.json → Test laeuft ohne Angabe des task_type.
    (out_dir / "model.json").write_text(json.dumps({"task": "tabular", **extra}, indent=2), encoding="utf-8")


def load_bundle(model_dir: str):
    import torch
    d = Path(model_dir)
    if not (d / WEIGHTS_FILE).is_file():
        raise FileNotFoundError(f"Kein trainiertes Tabellenmodell in {model_dir} (model.pt fehlt). "
                                "Erst trainieren, dann die trainierte Version testen.")
    prep = Preprocessor.from_dict(json.loads((d / PREPROCESSING_FILE).read_text(encoding="utf-8")))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    model = build_model(prep, cfg.get("hidden", [128, 64]), float(cfg.get("dropout", 0.1)))
    model.load_state_dict(torch.load(d / WEIGHTS_FILE, map_location="cpu", weights_only=True))
    model.eval()
    return model, prep


def predict_frame(model, prep: Preprocessor, df: pd.DataFrame, batch: int = 1024) -> Dict[str, np.ndarray]:
    """Klassifikation: {'pred': Klassenindex, 'proba': [n, k]}; Regression: {'pred': Werte}."""
    import torch
    x_num, x_cat = prep.transform(df)
    outs = []
    with torch.no_grad():
        for i in range(0, len(df), batch):
            outs.append(model(torch.from_numpy(x_num[i:i + batch]), torch.from_numpy(x_cat[i:i + batch])).numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, prep.output_dim), dtype=np.float32)
    if prep.task == "classification":
        e = np.exp(out - out.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
        return {"pred": proba.argmax(axis=1), "proba": proba}
    return {"pred": prep.decode_regression(out[:, 0])}


# ─── Kennzahlen ────────────────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        if tp + fp + fn == 0:
            continue  # Klasse kommt weder vor noch wird sie vorhergesagt
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": float(np.mean(y_true == y_pred)), "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}


def baseline_metrics(prep: Preprocessor, y_train: np.ndarray, y_eval: np.ndarray) -> Dict[str, float]:
    """Vergleichswert ohne Modell: haeufigste Klasse bzw. Mittelwert des Trainings."""
    if prep.task == "classification":
        majority = int(np.bincount(y_train, minlength=len(prep.classes)).argmax()) if len(y_train) else 0
        m = classification_metrics(y_eval, np.full(len(y_eval), majority), len(prep.classes))
        return {"baseline_accuracy": m["accuracy"], "baseline_macro_f1": m["macro_f1"]}
    m = regression_metrics(y_eval, np.full(len(y_eval), float(np.mean(y_train)) if len(y_train) else 0.0))
    return {"baseline_mae": m["mae"], "baseline_rmse": m["rmse"]}


# ─── Einzeltest: eine Zeile parsen ─────────────────────────────────────────

def parse_row(text: str, columns: List[str]) -> Dict[str, Any]:
    """JSON-Objekt, 'spalte=wert; spalte=wert' oder Werte in Spaltenreihenfolge."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Leere Eingabe.")
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("JSON muss ein Objekt {\"spalte\": wert} sein.")
        return obj
    sep = ";" if ";" in text or "\n" in text else ","
    parts = [p.strip() for p in text.replace("\n", ";").split(sep) if p.strip()]
    if parts and all(("=" in p or ":" in p) for p in parts):
        row = {}
        for p in parts:
            k, v = p.split("=", 1) if "=" in p else p.split(":", 1)
            row[k.strip()] = _guess_value(v.strip())
        return row
    if len(parts) != len(columns):
        raise ValueError(f"{len(parts)} Werte, erwartet {len(columns)} in dieser Reihenfolge: {', '.join(columns)}. "
                         "Alternativ spalte=wert; spalte=wert oder ein JSON-Objekt.")
    return {c: _guess_value(v) for c, v in zip(columns, parts)}


def _guess_value(v: str) -> Any:
    if v == "" or v.lower() in ("nan", "na", "null", "none", "?"):
        return None
    try:
        return float(v.replace(",", ".")) if v.count(",") == 1 and "." not in v else float(v)
    except ValueError:
        return v
