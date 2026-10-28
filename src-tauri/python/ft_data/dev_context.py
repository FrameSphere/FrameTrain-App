"""ft_data.dev_context – was die Code-KI in Dev Train / Dev Test ueber Modell und Daten wissen muss.

Vorher bekam die KI nur die Pfade. Fuer Chronos-2 (config: model_type "t5",
Architektur Chronos2Model) und eine Zeitreihen-Tabelle baute sie deshalb ein
Raten-Skript mit YOLO-, Klassifikations- und LLM-Zweig. Dieses Modul liest die
Fakten aus den Dateien und gibt einen kurzen Text fuer den System-Prompt aus.

Aufruf:  python3 -m ft_data.dev_context <MODEL_PATH> [DATASET_PATH]
"""
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

TABULAR = (".parquet", ".csv", ".tsv", ".json", ".jsonl")
MEDIA = {
    "image": (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"),
    "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"),
    "video": (".mp4", ".avi", ".mov", ".mkv", ".webm"),
}
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf")
CONFIG_KEYS = ("model_type", "architectures", "chronos_pipeline_class", "problem_type", "num_labels")

# Import-Name, pip-Name
PACKAGES = (
    ("torch", "torch"), ("transformers", "transformers"), ("datasets", "datasets"),
    ("accelerate", "accelerate"), ("peft", "peft"), ("sklearn", "scikit-learn"),
    ("pandas", "pandas"), ("ultralytics", "ultralytics"), ("librosa", "librosa"),
    ("chronos", "chronos-forecasting"), ("sentence_transformers", "sentence-transformers"),
    ("diffusers", "diffusers"), ("bitsandbytes", "bitsandbytes"),
)
MAX_CHARS = 6000


def _short(v, n: int = 60) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def describe_model(path: Path) -> List[str]:
    out = [f"MODELL ({path.name}):"]
    if path.is_file():
        return out + [f"- Einzeldatei: {path.name}"]
    if not path.is_dir():
        return out + ["- Pfad existiert nicht"]
    cfg_file = path / "config.json"
    if cfg_file.is_file():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            facts = ", ".join(f"{k}={cfg[k]}" for k in CONFIG_KEYS if k in cfg)
            out.append(f"- config.json: {facts or 'ohne model_type/architectures'}")
            if isinstance(cfg.get("id2label"), dict) and len(cfg["id2label"]) <= 30:
                out.append(f"- id2label: {_short(cfg['id2label'], 300)}")
            chronos = cfg.get("chronos_config")
            if isinstance(chronos, dict):
                keep = {k: v for k, v in chronos.items() if not isinstance(v, (list, dict))}
                out.append(f"- chronos_config: {_short(keep, 300)}")
        except (OSError, ValueError) as e:
            out.append(f"- config.json nicht lesbar: {e}")
    files = sorted(p.name for p in path.iterdir() if p.is_file())
    weights = [f for f in files if f.lower().endswith(WEIGHTS)]
    out.append(f"- Dateien: {', '.join(files[:15])}{' …' if len(files) > 15 else ''}")
    if not weights:
        out.append("- Keine Gewichtsdatei direkt im Ordner")
    readme = path / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        # Front-Matter (--- … ---) ueberspringen, dann die ersten Zeilen
        if text.startswith("---"):
            end = text.find("\n---", 3)
            text = text[end + 4:] if end > 0 else text
        lines = [l.strip() for l in text.splitlines() if l.strip()][:6]
        if lines:
            out.append(f"- README (Anfang): {_short(' '.join(lines), 400)}")
    return out


ID_COLUMNS = ("id", "item_id", "series_id", "unique_id", "symbol", "ticker")
TIME_COLUMNS = ("timestamp", "date", "datetime", "time", "ds")


def _series_frequency(ts) -> str:
    """Grobe Frequenz einer Zeitreihe — robust gegen Feiertage (infer_freq ist es nicht)."""
    import pandas as pd
    ts = pd.Series(pd.to_datetime(ts, errors="coerce")).dropna().drop_duplicates().sort_values()
    if len(ts) < 3:
        return "zu kurz"
    step = ts.diff().dropna().median()
    day = pd.Timedelta(days=1)
    if step < day:
        return f"untertaeglich ({step})"
    if step == day:
        return "taeglich inkl. Wochenende" if (ts.dt.dayofweek >= 5).any() else "Boersen-/Werktage (ohne Wochenende)"
    if step == 7 * day:
        return "woechentlich"
    if pd.Timedelta(days=28) <= step <= pd.Timedelta(days=31):
        return "monatlich"
    return f"alle {step}"


def _frequency_summary(df) -> List[str]:
    """Bei Zeitreihen im Long-Format: Frequenz je Reihe, gruppiert.

    Regression 28.10.2026: Krypto (taeglich inkl. Wochenende) und Aktien
    (Boersentage) gemischt — Chronos predict_df brach mit "All time series
    must have the same frequency" ab, weil die KI das nicht wissen konnte.
    """
    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    ts_col = next((c for c in TIME_COLUMNS if c in df.columns), None)
    if not id_col or not ts_col:
        return []
    groups = {}
    for sid, g in df.groupby(id_col, sort=True):
        groups.setdefault(_series_frequency(g[ts_col]), []).append(str(sid))
    out = ["  Frequenz je Reihe (" + id_col + " / " + ts_col + "): " + "; ".join(
        f"{freq}: {len(ids)} Reihen (z. B. {', '.join(ids[:4])})" for freq, ids in sorted(groups.items(), key=lambda x: -len(x[1]))
    )]
    if len(groups) > 1:
        out.append("  ACHTUNG: unterschiedliche Frequenzen — Bibliotheken, die eine gemeinsame Frequenz "
                   "verlangen (z. B. Chronos predict_df), brauchen getrennte Laeufe je Frequenz, "
                   "ein Auffuellen auf eine Frequenz oder Vorhersage ueber Positionen statt Zeitstempel.")
    return out


def _table_summary(f: Path) -> List[str]:
    """Spalten, Typen, Zeilenzahl und drei Beispielzeilen einer Tabellendatei."""
    try:
        import pandas as pd
    except ImportError:
        return ["  (pandas fehlt – Spalten nicht lesbar)"]
    try:
        ext = f.suffix.lower()
        total: Optional[int] = None
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq
                total = pq.ParquetFile(f).metadata.num_rows
            except Exception:
                total = None
            full = pd.read_parquet(f)
            total = total if total is not None else len(full)
            df = full.head(5000)
        elif ext in (".csv", ".tsv"):
            df = pd.read_csv(f, sep="\t" if ext == ".tsv" else ",", nrows=200_000)
            full, df = df, df.head(5000)
        else:
            raw = f.read_text(encoding="utf-8", errors="replace").lstrip()
            if raw.startswith("["):
                rows = json.loads(raw)
                total = len(rows)
                full = pd.DataFrame(rows)
            else:
                full = pd.read_json(f, lines=True, nrows=200_000)
            df = full.head(5000)
    except Exception as e:
        return [f"  (nicht lesbar: {_short(e, 120)})"]

    out = [f"  Zeilen: {total if total is not None else f'≥{len(df)}'} | Spalten: "
           + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:30])]
    # Kategorische Spalten (z. B. id einer Zeitreihe, label) mit Anzahl Werten
    cats = []
    for c in df.columns[:30]:
        if df[c].dtype == object or str(df[c].dtype) in ("string", "str", "category"):
            try:
                n = df[c].nunique()
            except TypeError:
                continue
            if 1 < n <= 200:
                cats.append(f"{c}: {n} Werte (z. B. {', '.join(_short(v, 20) for v in df[c].dropna().unique()[:5])})")
    if cats:
        out.append("  Kategorien (aus den ersten 5000 Zeilen): " + "; ".join(cats))
    try:
        out.extend(_frequency_summary(full))
    except Exception as e:  # Frequenz ist ein Zusatz — nie die ganze Beschreibung verlieren
        out.append(f"  (Frequenz nicht bestimmbar: {_short(e, 80)})")
    for _, row in df.head(3).iterrows():
        out.append("  Beispiel: " + _short({k: _short(v, 40) for k, v in row.items()}, 400))
    return out


def describe_dataset(path: Path) -> List[str]:
    out = [f"DATASET ({path.name}):"]
    if not path.is_dir():
        return out + ["- Pfad existiert nicht"]
    yamls = [p.name for p in path.iterdir() if p.suffix.lower() in (".yaml", ".yml")]
    if yamls:
        out.append(f"- YAML: {', '.join(yamls)}")
    parts = [d for d in sorted(path.iterdir()) if d.is_dir() and not d.name.startswith((".", "_"))]
    sections = [(d.name + "/", d) for d in parts] or [("(Wurzel)", path)]
    shown_table = False
    for label, d in sections:
        files = [f for f in d.rglob("*") if f.is_file() and not f.name.startswith(".")]
        if not files:
            out.append(f"- {label}: leer")
            continue
        exts = {}
        for f in files:
            exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        by_ext = sorted(exts.items(), key=lambda x: -x[1])[:5]
        ext_text = ", ".join("%d× %s" % (n, e or "ohne Endung") for e, n in by_ext)
        out.append(f"- {label}: {len(files)} Dateien ({ext_text})")
        kind = next((k for k, e in MEDIA.items() if any(f.suffix.lower() in e for f in files)), None)
        if kind:
            classes = sorted({f.parent.name for f in files if f.suffix.lower() in MEDIA[kind] and f.parent != d})
            if classes:
                out.append(f"  {kind}-Klassenordner: {', '.join(classes[:20])}{' …' if len(classes) > 20 else ''}")
        tables = sorted(f for f in files if f.suffix.lower() in TABULAR)
        if tables and not shown_table:
            out.append(f"  Tabelle {tables[0].relative_to(path)}:")
            out.extend(_table_summary(tables[0]))
            shown_table = True
    # Root-Dateien neben Split-Ordnern
    root_files = [p.name for p in path.iterdir() if p.is_file() and not p.name.startswith(".")]
    if parts and root_files:
        out.append(f"- Dateien im Wurzelordner: {', '.join(sorted(root_files)[:10])}")
    return out


# Modelle, deren Training ueber eine eigene Bibliothek laeuft. Ohne deren API
# baute die KI fuer Chronos-2 dreimal einen eigenen Loop um das Modell-Innere
# (encoder(inputs_embeds=…), group_ids …) und scheiterte jedes Mal — dabei
# bringt chronos-forecasting mit Chronos2Pipeline.fit das Fine-Tuning mit.
# Signaturen werden zur Laufzeit gelesen, damit sie zur installierten Version passen.
LIBRARY_APIS = {
    "Chronos2Pipeline": {
        "pip": "chronos-forecasting",
        "objects": [
            ("chronos", "Chronos2Pipeline", ("from_pretrained", "fit", "predict_df", "save_pretrained")),
            ("chronos.chronos2.preprocess", None, ("from_data_frame",)),
        ],
        "notes": [
            "Fine-Tuning NICHT selbst bauen, sondern: pipe = Chronos2Pipeline.from_pretrained(MODEL_PATH); "
            "inputs = from_data_frame(df, target_columns=[...], prediction_length=H, id_column=..., timestamp_column=...); "
            "ft = pipe.fit(inputs, prediction_length=H, validation_inputs=..., num_steps=..., batch_size=..., context_length=..., "
            "output_dir=<temp-Ordner, NICHT OUTPUT_PATH>, callbacks=[TrainerCallback fuer progress-Events]); ft.save_pretrained(OUTPUT_PATH).",
            "from_data_frame braucht nur id-, timestamp-, Ziel- und Kovariaten-Spalten (Hilfsspalten wie split vorher entfernen).",
            "context_length klein halten (z. B. 512): mit dem Standard 8192 dauert ein Schritt auf dem Mac ~18 s.",
        ],
    },
}


def _signature(obj) -> str:
    import inspect
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(…)"
    parts = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{name}")
        elif p.default is p.empty:
            parts.append(name)
        else:
            parts.append(f"{name}={_short(repr(p.default), 30)}")
    return "(" + ", ".join(parts) + ")"


def describe_library_api(model_path: Path) -> List[str]:
    """API der Bibliothek, mit der dieses Modell trainiert wird (falls bekannt)."""
    cfg_file = model_path / "config.json" if model_path.is_dir() else None
    if not cfg_file or not cfg_file.is_file():
        return []
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keys = {v for k, v in cfg.items() if k.endswith("pipeline_class") and isinstance(v, str)}
    out: List[str] = []
    for key in sorted(keys & LIBRARY_APIS.keys()):
        spec = LIBRARY_APIS[key]
        out.append(f"BIBLIOTHEK FUER DIESES MODELL ({spec['pip']}):")
        lines: List[str] = []
        try:
            import importlib
            for module_name, cls_name, names in spec["objects"]:
                mod = importlib.import_module(module_name)
                owner = getattr(mod, cls_name) if cls_name else mod
                prefix = f"{cls_name}." if cls_name else f"{module_name}."
                for n in names:
                    fn = getattr(owner, n, None)
                    if fn is not None:
                        lines.append(f"- {prefix}{n}{_signature(fn)}")
        except Exception as e:  # nicht installiert oder Import-Fehler
            out.append(f"- nicht importierbar ({_short(e, 120)}) — erst {spec['pip']} installieren")
            continue
        out.extend(lines)
        out.extend(f"- {n}" for n in spec["notes"])
    return out


def describe_packages() -> List[str]:
    have, missing = [], []
    for module, pip in PACKAGES:
        if importlib.util.find_spec(module) is None:
            missing.append(pip)
            continue
        try:
            version = importlib.metadata.version(pip)
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        note = " (ohne CUDA nicht nutzbar)" if module == "bitsandbytes" and sys.platform == "darwin" else ""
        have.append(f"{pip} {version}{note}")
    out = [f"PYTHON: {sys.executable} ({sys.version.split()[0]})",
           f"- installiert: {', '.join(have) or '–'}"]
    if missing:
        out.append(f"- NICHT installiert: {', '.join(missing)}")
        out.append(f'  Installieren mit: "{sys.executable}" -m pip install <paket>')
    return out


def describe(model_path: str, dataset_path: Optional[str] = None) -> str:
    lines: List[str] = []
    if model_path:
        lines += describe_model(Path(model_path))
    if dataset_path:
        lines += describe_dataset(Path(dataset_path))
    lines += describe_packages()
    if model_path:
        lines += describe_library_api(Path(model_path))
    text = "\n".join(lines)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


if __name__ == "__main__":
    print(describe(sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2] if len(sys.argv) > 2 else None))
