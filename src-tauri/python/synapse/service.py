#!/usr/bin/env python3
"""Formen-Dienst des Synapse Builders.

Laeuft dauerhaft, damit torch nur einmal geladen wird. Protokoll wie bei den
Modell-Servern: eine JSON-Zeile rein, eine JSON-Zeile raus.
  {"id": 1, "cmd": "check", "graph": {...}, "input_shape": null, "dtype": null}
  {"id": 1, "result": {...}}
Beim Start: {"type": "ready"} oder {"type": "error", "message": "..."}
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> None:
    try:
        from synapse.check import check_raw
        from synapse.spec import catalog
    except Exception as e:
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        sys.exit(1)
    emit({"type": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            cmd = req.get("cmd")
            if cmd == "shutdown":
                break
            if cmd == "check":
                emit({"id": rid, "result": check_raw(req["graph"], req.get("input_shape"), req.get("dtype"))})
            elif cmd == "export":
                from synapse.export import export_model
                emit({"id": rid, "result": export_model(req["model_dir"], req["format"], req["target"])})
            elif cmd == "catalog":
                emit({"id": rid, "result": catalog()})
            else:
                emit({"id": rid, "error": f"Unbekannter Befehl: {cmd}"})
        except Exception as e:
            emit({"id": rid, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
