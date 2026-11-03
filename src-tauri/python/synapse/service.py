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


_OUT = sys.stdout


def emit(obj: dict) -> None:
    # Immer ueber das echte stdout: waehrend eines Befehls zeigt sys.stdout auf
    # stderr, damit Bibliotheken (z. B. der ONNX-Exporter) das Protokoll nicht stoeren.
    print(json.dumps(obj, ensure_ascii=False), file=_OUT, flush=True)


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
        sys.stdout = sys.stderr
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
            elif cmd == "infer":
                from synapse.infer import infer
                emit({"id": rid, "result": infer(req["model_dir"], req["kind"], req.get("value"))})
            elif cmd == "step":
                from synapse.step import one_step
                emit({"id": rid, "result": one_step(req["graph"], float(req.get("lr", 0.01)))})
            elif cmd == "noisy":  # fuer Tests: Ausgaben einer Bibliothek
                print("Fortschritt einer Bibliothek")
                emit({"id": rid, "result": "ok"})
            elif cmd == "catalog":
                emit({"id": rid, "result": catalog()})
            else:
                emit({"id": rid, "error": f"Unbekannter Befehl: {cmd}"})
        except Exception as e:
            emit({"id": rid, "error": f"{type(e).__name__}: {e}"})
        finally:
            sys.stdout = _OUT


if __name__ == "__main__":
    main()
