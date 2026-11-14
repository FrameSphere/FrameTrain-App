"""ts_core – gemeinsamer Kern der Zeitreihen-Vorhersage (Train- und Test-Engine).

Grundsaetze:
  * Aufteilung immer zeitlich, je Serie: Training = frueher Teil, Validierung
    = mittlerer, Test = letzter Teil. Die zufaellige Zeilen-Aufteilung der App
    wird ignoriert — alle Dateien werden zusammengefuehrt und nach Zeit sortiert.
  * Ein Fenster gehoert zu dem Abschnitt, in dem seine VORHERSAGE liegt. Der
    Kontext darf weiter zurueckreichen (das ist Vergangenheit, kein Leck);
    Trainingsfenster sehen nie einen Wert nach dem Trainingsende.
  * Normalisierung je Serie und Kanal nur aus dem Trainingsteil.
  * Vergleichswert: letzter Wert bzw. saisonal naiv (Wert vor einer Saison).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".parquet", ".jsonl", ".json")
TIME_NAMES = ("timestamp", "datetime", "date", "time", "ds", "datum", "zeit", "zeitpunkt", "period", "month", "day")
ID_NAMES = ("series_id", "unique_id", "item_id", "id", "series", "serie", "store", "sensor")
META_FILE = "frametrain_timeseries.json"
BLANK_MODEL_TYPE = "ft_time_series"


# ─── Laden ─────────────────────────────────────────────────────────────────

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
    return pd.read_csv(path, sep=None, engine="python")


def load_all(dataset_path: str) -> pd.DataFrame:
    root = Path(dataset_path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in TABLE_EXTS
                 and not p.name.startswith(".") and p.name not in ("dataset_info.json", "metadata.json", "config.json")]
    else:
        raise FileNotFoundError(f"Dataset-Pfad nicht gefunden: {dataset_path}")
    if not files:
        raise ValueError(f"Keine Tabellendatei (CSV, TSV, Parquet, JSONL) in {dataset_path} gefunden.")
    return pd.concat([read_table(p) for p in files], ignore_index=True)


def _parses_as_time(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


@dataclass
class Columns:
    time: Optional[str]
    values: List[str]
    id: Optional[str]


def detect_columns(df: pd.DataFrame, time_column: str = "", target_columns: str = "",
                   id_column: str = "") -> Columns:
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}

    def need(name: str) -> str:
        if name not in df.columns:
            raise ValueError(f"Spalte '{name}' fehlt. Vorhanden: {cols}")
        return name

    t = need(time_column) if time_column else None
    if t is None:
        t = next((lower[n] for n in TIME_NAMES if n in lower), None)
        if t is None:
            t = next((c for c in cols if _parses_as_time(df[c])), None)
    i = need(id_column) if id_column else next(
        (lower[n] for n in ID_NAMES if n in lower and lower[n] != t), None)
    if target_columns:
        values = [need(c.strip()) for c in str(target_columns).split(",") if c.strip()]
    else:
        values = [c for c in cols if c not in (t, i) and pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_bool_dtype(df[c])]
    if not values:
        raise ValueError("Keine Wertspalte (Zahlen) gefunden. Bitte 'target_columns' angeben.")
    return Columns(time=t, values=values, id=i)


def build_series(df: pd.DataFrame, cols: Columns) -> Dict[str, np.ndarray]:
    """{serien_id: Array [T, C]} nach Zeit sortiert, Luecken interpoliert."""
    work = df.copy()
    if cols.time:
        work["__t"] = pd.to_datetime(work[cols.time], errors="coerce", format="mixed") \
            if not pd.api.types.is_numeric_dtype(work[cols.time]) else work[cols.time]
        work = work.dropna(subset=["__t"])
    else:
        work["__t"] = np.arange(len(work))
    for c in cols.values:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    groups = work.groupby(work[cols.id].astype(str)) if cols.id else [("series", work)]
    out: Dict[str, np.ndarray] = {}
    for sid, g in groups:
        g = g.sort_values("__t", kind="stable").drop_duplicates("__t", keep="last")
        vals = g[cols.values].astype(float)
        # Fehlende Werte: linear zwischen Nachbarn, an den Raendern fortschreiben.
        vals = vals.interpolate(method="linear", limit_direction="both").ffill().bfill()
        arr = vals.to_numpy(dtype=np.float32)
        if np.isnan(arr).any():
            continue  # Kanal komplett leer
        out[str(sid)] = arr
    if not out:
        raise ValueError("Keine verwertbare Zeitreihe gefunden (alle Werte fehlen).")
    return out


# ─── Zeitliche Aufteilung ──────────────────────────────────────────────────

@dataclass
class SplitPoints:
    train_end: int   # exklusiv
    val_end: int     # exklusiv; Test = [val_end, T)
    length: int


def split_points(length: int, horizon: int, val_ratio: float, test_ratio: float) -> SplitPoints:
    n_test = max(int(round(length * test_ratio)), horizon) if test_ratio > 0 else 0
    n_val = max(int(round(length * val_ratio)), horizon) if val_ratio > 0 else 0
    return SplitPoints(train_end=length - n_val - n_test, val_end=length - n_test, length=length)


def window_starts(sp: SplitPoints, part: str, context: int, horizon: int, stride: Optional[int] = None) -> List[int]:
    """Startindizes s (Kontext = [s, s+L), Ziel = [s+L, s+L+H)) fuer einen Abschnitt."""
    lo, hi = {"train": (0, sp.train_end), "val": (sp.train_end, sp.val_end), "test": (sp.val_end, sp.length)}[part]
    first_target = max(lo, context)  # Kontext braucht L Werte davor
    step = stride or (1 if part == "train" else horizon)
    starts = []
    t = first_target
    while t + horizon <= hi:
        starts.append(t - context)
        t += step
    return starts


@dataclass
class Normalizer:
    mean: Dict[str, List[float]] = field(default_factory=dict)
    std: Dict[str, List[float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, series: Dict[str, np.ndarray], ends: Dict[str, int]) -> "Normalizer":
        n = cls()
        for sid, arr in series.items():
            part = arr[: max(ends[sid], 2)]
            n.mean[sid] = part.mean(axis=0).astype(float).tolist()
            n.std[sid] = [float(s) if s > 1e-8 else 1.0 for s in part.std(axis=0)]
        return n

    def stats(self, sid: str, arr: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if sid in self.mean:
            return np.array(self.mean[sid], dtype=np.float32), np.array(self.std[sid], dtype=np.float32)
        # Unbekannte Serie (Einzeltest): Statistik aus der Eingabe selbst.
        assert arr is not None
        s = arr.std(axis=0)
        return arr.mean(axis=0).astype(np.float32), np.where(s > 1e-8, s, 1.0).astype(np.float32)


def make_windows(series: Dict[str, np.ndarray], splits: Dict[str, SplitPoints], norm: Normalizer,
                 part: str, context: int, horizon: int, stride: Optional[int] = None):
    """(past [N, L, C], future [N, H, C], meta [(sid, start)]) normalisiert."""
    past, fut, meta = [], [], []
    for sid, arr in series.items():
        mu, sd = norm.stats(sid)
        z = (arr - mu) / sd
        for s in window_starts(splits[sid], part, context, horizon, stride):
            past.append(z[s:s + context])
            fut.append(z[s + context:s + context + horizon])
            meta.append((sid, s))
    if not past:
        c = next(iter(series.values())).shape[1]
        return (np.zeros((0, context, c), np.float32), np.zeros((0, horizon, c), np.float32), [])
    return np.stack(past).astype(np.float32), np.stack(fut).astype(np.float32), meta


# ─── Vergleichswerte und Kennzahlen ────────────────────────────────────────

def naive_forecast(history: np.ndarray, horizon: int, season: int = 0) -> np.ndarray:
    """history [L, C] -> [H, C]. season>1: Werte vor einer Saison, sonst letzter Wert."""
    if season and season > 1 and len(history) >= season:
        last_season = history[-season:]
        reps = int(np.ceil(horizon / season))
        return np.tile(last_season, (reps, 1))[:horizon]
    return np.repeat(history[-1:], horizon, axis=0)


def mase_scale(train_part: np.ndarray, season: int = 0) -> np.ndarray:
    """Mittlerer absoluter Fehler der naiven In-Sample-Vorhersage je Kanal."""
    m = season if season and season > 1 and len(train_part) > season else 1
    if len(train_part) <= m:
        return np.ones(train_part.shape[1], dtype=np.float64)
    d = np.abs(train_part[m:] - train_part[:-m]).mean(axis=0)
    return np.where(d > 1e-8, d, 1.0)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: Optional[np.ndarray] = None) -> Dict[str, float]:
    """y_* [N, H, C] in Originaleinheiten; scale [N, C] fuer MASE (je Fenster die Serie)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "smape": 0.0, "mase": 0.0}
    err = y_pred - y_true
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.where(denom > 1e-12, 2 * np.abs(err) / np.maximum(denom, 1e-12), 0.0)
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "smape": float(np.mean(smape) * 100.0)}
    if scale is not None:
        out["mase"] = float(np.mean(np.abs(err) / np.asarray(scale)[:, None, :]))
    return out


# ─── Modell ────────────────────────────────────────────────────────────────

def is_pretrained_patchtst(model_path: str) -> bool:
    d = Path(model_path or "")
    cfg = d / "config.json"
    if not cfg.is_file():
        return False
    try:
        mt = json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except Exception:  # noqa: BLE001
        return False
    has_weights = any((d / f).is_file() for f in ("model.safetensors", "pytorch_model.bin"))
    return mt == "patchtst" and has_weights


def build_patchtst(channels: int, context: int, horizon: int, pc: Dict[str, Any], model_path: str = "",
                   from_scratch: bool = False):
    """Neu initialisiert oder von einem vortrainierten PatchTST (HF) ausgehend.

    Vortrainiert: Kontextlaenge und Patch-Groesse kommen aus dessen config —
    die gelernten Positions-Embeddings haengen davon ab. Nur der Vorhersagekopf
    wird bei anderer Vorhersagelaenge neu angelegt.
    """
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    if not from_scratch and is_pretrained_patchtst(model_path):
        base = PatchTSTConfig.from_pretrained(model_path)
        model = PatchTSTForPrediction.from_pretrained(
            model_path, num_input_channels=channels, prediction_length=horizon,
            ignore_mismatched_sizes=True)
        return model, int(base.context_length), True
    patch = int(pc.get("patch_length", 0) or 0) or max(2, min(16, context // 4))
    stride = int(pc.get("patch_stride", 0) or 0) or max(1, patch // 2)
    cfg = PatchTSTConfig(
        num_input_channels=channels, context_length=context, prediction_length=horizon,
        patch_length=patch, patch_stride=stride,
        d_model=int(pc.get("d_model", 64)), num_hidden_layers=int(pc.get("num_layers", 2)),
        num_attention_heads=int(pc.get("num_heads", 4)), ffn_dim=int(pc.get("ffn_dim", 128)),
        dropout=float(pc.get("model_dropout", 0.1)), head_dropout=float(pc.get("model_dropout", 0.1)),
        scaling="std", loss="mse")
    return PatchTSTForPrediction(cfg), context, False


def predict(model, past: np.ndarray, device, batch: int = 256) -> np.ndarray:
    import torch
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(past), batch):
            x = torch.from_numpy(past[i:i + batch]).to(device)
            outs.append(model(past_values=x).prediction_outputs.float().cpu().numpy())
    if not outs:
        return np.zeros((0,) + past.shape[1:], np.float32)
    return np.concatenate(outs)


# ─── Speichern / Laden ─────────────────────────────────────────────────────

def save_meta(out_dir: Path, meta: Dict[str, Any]) -> None:
    (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "model.json").write_text(json.dumps({"task": "time_series"}, indent=2), encoding="utf-8")


def load_trained(model_dir: str, device="cpu"):
    from transformers import PatchTSTForPrediction
    d = Path(model_dir)
    if not (d / META_FILE).is_file():
        raise FileNotFoundError(f"Kein trainiertes Zeitreihenmodell in {model_dir} ({META_FILE} fehlt). "
                                "Erst trainieren, dann die trainierte Version testen.")
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    model = PatchTSTForPrediction.from_pretrained(str(d)).to(device)
    model.eval()
    return model, meta


def normalizer_from_meta(meta: Dict[str, Any]) -> Normalizer:
    return Normalizer(mean=meta.get("norm_mean", {}), std=meta.get("norm_std", {}))


def evaluate_part(model, series: Dict[str, np.ndarray], meta: Dict[str, Any], part: str, device,
                  max_windows: Optional[int] = None):
    """Vorhersagen + Kennzahlen (Modell und naiv) fuer einen Abschnitt, in Originaleinheiten."""
    L, H = int(meta["context_length"]), int(meta["prediction_length"])
    season = int(meta.get("season_length", 0) or 0)
    norm = normalizer_from_meta(meta)
    # Serien ohne gespeicherte Statistik (anderes Dataset): Statistik aus ihrem Trainingsteil.
    splits = {sid: split_points(len(a), H, meta["val_ratio"], meta["test_ratio"]) for sid, a in series.items()}
    for sid, arr in series.items():
        if sid not in norm.mean:
            part_arr = arr[: max(splits[sid].train_end, 2)]
            mu, sd = norm.stats("__neu__", part_arr)
            norm.mean[sid], norm.std[sid] = mu.tolist(), sd.tolist()
    past, fut, wmeta = make_windows(series, splits, norm, part, L, H)
    if max_windows:
        past, fut, wmeta = past[:max_windows], fut[:max_windows], wmeta[:max_windows]
    pred_z = predict(model, past, device)
    mus = np.stack([np.array(norm.mean[s], np.float32) for s, _ in wmeta]) if wmeta else np.zeros((0, 1))
    sds = np.stack([np.array(norm.std[s], np.float32) for s, _ in wmeta]) if wmeta else np.zeros((0, 1))
    true = fut * sds[:, None, :] + mus[:, None, :] if len(wmeta) else fut
    pred = pred_z * sds[:, None, :] + mus[:, None, :] if len(wmeta) else pred_z
    hist = past * sds[:, None, :] + mus[:, None, :] if len(wmeta) else past
    naive = np.stack([naive_forecast(h, H, season) for h in hist]) if len(wmeta) else pred
    scales = np.stack([mase_scale(series[s][: splits[s].train_end], season) for s, _ in wmeta]) \
        if len(wmeta) else None
    m_model = forecast_metrics(true, pred, scales)
    m_naive = forecast_metrics(true, naive, scales)
    return {"true": true, "pred": pred, "naive": naive, "meta": wmeta, "splits": splits,
            "metrics": m_model, "naive_metrics": m_naive}


def forecast_curve(series: Dict[str, np.ndarray], ev: Dict[str, Any], meta: Dict[str, Any],
                   max_points: int = 600) -> Dict[str, Any]:
    """Kurve fuer die Anzeige: erste Serie, Kanal 0 — Verlauf vor dem Test, echte Werte, Vorhersage, naiv."""
    if not ev["meta"]:
        return {}
    sid = ev["meta"][0][0]
    L = int(meta["context_length"])
    idx = [i for i, (s, _) in enumerate(ev["meta"]) if s == sid]
    arr = series[sid][:, 0]
    start_t = ev["meta"][idx[0]][1] + L
    end_t = ev["meta"][idx[-1]][1] + L + int(meta["prediction_length"])
    hist_from = max(0, start_t - L)
    pred = np.concatenate([ev["pred"][i][:, 0] for i in idx])
    naive = np.concatenate([ev["naive"][i][:, 0] for i in idx])
    actual = arr[hist_from:end_t]
    offset = start_t - hist_from
    n = len(actual)
    step = max(1, int(np.ceil(n / max_points)))

    def thin(xs):
        # NaN ist kein gueltiges JSON (Rust/serde verwirft die Nachricht) -> null.
        return [None if not np.isfinite(x) else round(float(x), 6) for x in xs[::step]]

    return {"series_id": sid, "channel": meta["value_columns"][0], "step": step,
            "forecast_start": offset // step, "actual": thin(actual),
            "forecast": thin(np.concatenate([np.full(offset, np.nan), pred])),
            "naive": thin(np.concatenate([np.full(offset, np.nan), naive]))}


def parse_history(text: str, channels: int) -> np.ndarray:
    """Einzeltest: Werte durch Komma/Leerzeichen (1 Kanal) bzw. Zeilen/';' mit Komma je Kanal."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Bitte die letzten Werte der Reihe eingeben, z. B. 12, 15, 14, 18.")
    if channels == 1:
        tokens = [t for t in text.replace(";", " ").replace("\n", " ").replace(",", " ").split() if t]
        return np.array([float(t) for t in tokens], dtype=np.float32)[:, None]
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    data = [[float(v) for v in r.replace(",", " ").split()] for r in rows]
    if any(len(r) != channels for r in data):
        raise ValueError(f"Jede Zeile braucht {channels} Werte (ein Wert je Spalte), Zeilen mit ';' trennen.")
    return np.array(data, dtype=np.float32)
