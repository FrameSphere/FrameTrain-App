"""
FrameTrain - Dataset Split Script
===================================
Splits a dataset directory into two equal halves, preserving file format.
Called from Rust via Command::new("python3").

Usage:
    python3 split_dataset.py <source_dir> <out_dir_a> <out_dir_b>

Supports: .parquet, .arrow, .jsonl, .json, .csv, .tsv, .txt
"""

import sys
import json
import math
from pathlib import Path

SKIP_NAMES = {
    "README.md", "readme.md", "dataset_infos.json", "dataset_dict.json",
    "state.json", "datasetdict.json", ".gitattributes", ".gitignore",
}
SKIP_SUFFIXES = {".md", ".gitattributes", ".gitignore", ".yaml", ".yml", ".lock"}


def log(msg: str):
    print(json.dumps({"type": "status", "message": msg}), flush=True)


def err(msg: str):
    print(json.dumps({"type": "error", "message": msg}), flush=True)


def data_files_in(path: Path):
    return [
        f for f in path.rglob("*")
        if f.is_file()
        and f.name not in SKIP_NAMES
        and f.suffix.lower() not in SKIP_SUFFIXES
    ]


def split_file(src: Path, out_a: Path, out_b: Path) -> tuple[int, int]:
    """Split a single file into two halves. Returns (count_a, count_b)."""
    ext = src.suffix.lower()
    out_a.parent.mkdir(parents=True, exist_ok=True)
    out_b.parent.mkdir(parents=True, exist_ok=True)

    # ── Parquet ──────────────────────────────────────────────────────────
    if ext == ".parquet":
        import pyarrow.parquet as pq
        import pyarrow as pa
        table = pq.read_table(str(src))
        n = len(table)
        mid = math.ceil(n / 2)
        pq.write_table(table.slice(0, mid),   str(out_a))
        pq.write_table(table.slice(mid),       str(out_b))
        return mid, n - mid

    # ── Arrow ─────────────────────────────────────────────────────────────
    if ext == ".arrow":
        import pyarrow as pa
        with pa.memory_map(str(src), "r") as source:
            table = pa.ipc.open_file(source).read_all()
        n = len(table)
        mid = math.ceil(n / 2)
        schema = table.schema
        sink_a = pa.OSFile(str(out_a), "wb")
        sink_b = pa.OSFile(str(out_b), "wb")
        with pa.ipc.new_file(sink_a, schema) as w:
            w.write_table(table.slice(0, mid))
        with pa.ipc.new_file(sink_b, schema) as w:
            w.write_table(table.slice(mid))
        return mid, n - mid

    # ── JSONL ─────────────────────────────────────────────────────────────
    if ext == ".jsonl":
        lines = [l for l in src.read_text("utf-8").splitlines() if l.strip()]
        mid = math.ceil(len(lines) / 2)
        out_a.write_text("\n".join(lines[:mid]),  encoding="utf-8")
        out_b.write_text("\n".join(lines[mid:]),  encoding="utf-8")
        return mid, len(lines) - mid

    # ── JSON (array) ─────────────────────────────────────────────────────
    if ext == ".json":
        raw = src.read_text("utf-8")
        # Detect JSONL-in-.json (e.g. kp20k): try JSON array first, fall back to line-by-line
        is_jsonl = False
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            is_jsonl = True
            data = []
            for lineno, line in enumerate(raw.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Zeile {lineno} ist kein gueltiges JSON: {e}")
        mid = math.ceil(len(data) / 2)
        if is_jsonl:
            # Preserve JSONL format in output
            out_a.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data[:mid]),  "utf-8")
            out_b.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data[mid:]), "utf-8")
        else:
            out_a.write_text(json.dumps(data[:mid],  ensure_ascii=False, indent=2), "utf-8")
            out_b.write_text(json.dumps(data[mid:], ensure_ascii=False, indent=2), "utf-8")
        return mid, len(data) - mid

    # ── CSV / TSV ─────────────────────────────────────────────────────────
    if ext in (".csv", ".tsv"):
        import csv
        delim = "\t" if ext == ".tsv" else ","
        with open(src, newline="", encoding="utf-8", errors="replace") as f:
            reader = list(csv.reader(f, delimiter=delim))
        if not reader:
            return 0, 0
        header = reader[0]
        rows   = reader[1:]
        mid    = math.ceil(len(rows) / 2)

        def write_csv(path, rows_out):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f, delimiter=delim).writerows([header] + rows_out)

        write_csv(out_a, rows[:mid])
        write_csv(out_b, rows[mid:])
        return mid, len(rows) - mid

    # ── TXT ───────────────────────────────────────────────────────────────
    if ext == ".txt":
        lines = [l for l in src.read_text("utf-8", errors="replace").splitlines() if l.strip()]
        mid = math.ceil(len(lines) / 2)
        out_a.write_text("\n".join(lines[:mid]), encoding="utf-8")
        out_b.write_text("\n".join(lines[mid:]), encoding="utf-8")
        return mid, len(lines) - mid

    raise ValueError(f"Unbekanntes Dateiformat: {ext}")


def split_directory(src_dir: Path, out_a: Path, out_b: Path):
    """Split all data files in src_dir into two output directories."""
    files = data_files_in(src_dir)
    if not files:
        err(f"Keine Datendateien gefunden in {src_dir}")
        sys.exit(1)

    total_a = total_b = 0

    for src in sorted(files):
        rel = src.relative_to(src_dir)
        dst_a = out_a / rel
        dst_b = out_b / rel
        log(f"Splitting {src.name}...")
        try:
            ca, cb = split_file(src, dst_a, dst_b)
            total_a += ca
            total_b += cb
            log(f"  {src.name}: {ca} + {cb} Rows")
        except Exception as e:
            err(f"Fehler bei {src.name}: {e}")
            sys.exit(1)

    print(json.dumps({
        "type": "done",
        "total_a": total_a,
        "total_b": total_b,
    }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: split_dataset.py <source_dir> <out_dir_a> <out_dir_b>")
        sys.exit(1)

    src  = Path(sys.argv[1])
    a    = Path(sys.argv[2])
    b    = Path(sys.argv[3])

    if not src.exists():
        err(f"Quellverzeichnis nicht gefunden: {src}")
        sys.exit(1)

    split_directory(src, a, b)
