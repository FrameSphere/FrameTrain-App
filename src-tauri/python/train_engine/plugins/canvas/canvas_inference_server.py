#!/usr/bin/env python3
"""
canvas_inference_server.py — Persistenter Inference-Server für Canvas-Modelle
===============================================================================
Gleiche stdin/stdout-Architektur wie model_server.py, aber für DynamicGraphModule
aus graph_metadata.json + model.pt (kein HuggingFace, kein Tokenizer).

Protokoll:
  Rust → Python (stdin):   {"input": [...], "input_type": "tensor"}\n
                        oder {"input": "text", "input_type": "text"}\n
                        oder {"cmd": "shutdown"}\n
  Python → Rust (stdout):  { ...InferResult... }\n

Startup:
  Python → Rust:  {"type": "ready", "num_classes": N, "task_type": "..."}\n
  Python → Rust:  {"type": "error", "message": "..."}\n
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Unbuffered stdout für IPC
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Canvas-Loader importieren
_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)


def emit(obj: dict):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def emit_error(message: str):
    emit({"type": "error", "message": message})


class CanvasInferenceServer:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.loaded = None  # LoadedCanvasModel

    def load(self):
        from loader import load_canvas_model
        self.loaded = load_canvas_model(self.model_dir)

    def infer(self, req: dict) -> dict:
        import torch

        input_type = req.get("input_type", "tensor")
        raw = req.get("input")

        if raw is None:
            return {"type": "error", "message": "Kein 'input' in Anfrage"}

        # Tensor-Input: Liste von Zahlen oder Liste von Listen
        if input_type == "tensor":
            try:
                x = torch.tensor(raw, dtype=torch.float32)
            except Exception as e:
                return {"type": "error", "message": f"Tensor-Konvertierung: {e}"}
        elif input_type == "text":
            # Text-Input: Feature-Encoding versuchen (einfaches Fallback)
            return {"type": "error", "message": "Text-Input erfordert einen Tokenizer – nutze ein seq_classification-Modell."}
        else:
            return {"type": "error", "message": f"Unbekannter input_type: {input_type}"}

        result = self.loaded.predict(x)
        result["type"] = "result"
        return result

    def run(self):
        try:
            self.load()
        except Exception as e:
            emit_error(f"Modell konnte nicht geladen werden: {e}")
            sys.exit(1)

        emit({
            "type": "ready",
            "num_classes": self.loaded.num_classes,
            "task_type": self.loaded.task_type,
            "device": self.loaded.device,
        })

        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError as e:
                emit_error(f"JSON parse: {e}")
                continue

            if req.get("cmd") == "shutdown":
                break

            try:
                result = self.infer(req)
                emit(result)
            except Exception as e:
                emit_error(f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FrameTrain Canvas Inference Server")
    parser.add_argument("--model-dir", required=True, help="Pfad zum Modell-Ordner (mit graph_metadata.json + model.pt)")
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        emit_error(f"Modell-Ordner nicht gefunden: {args.model_dir}")
        sys.exit(1)

    CanvasInferenceServer(args.model_dir).run()


if __name__ == "__main__":
    main()
