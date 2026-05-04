#!/usr/bin/env python3
"""
FrameTrain - Persistent Model Server
=====================================
Bleibt als Hintergrundprozess am Leben und beantwortet Inferenz-Anfragen
via stdin/stdout JSON-Protokoll.

Protokoll:
  Rust -> Python (stdin):   {"text": "..."}\n
  Python -> Rust (stdout):  {"predicted": "...", "confidence": 0.95, ...}\n

Startup:
  Python -> Rust:  {"type": "ready"}\n   (wenn Modell geladen)
  Python -> Rust:  {"type": "error", "message": "..."}\n  (bei Fehler)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Unbuffered line-by-line stdout (kritisch fuer IPC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def emit(obj: dict):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def emit_error(message: str):
    emit({"type": "error", "message": message})


class ModelServer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer  = None
        self.model      = None
        self.id2label   = {}
        self.device     = None
        self._torch     = None
        self._np        = None

    def load(self):
        try:
            import torch
            import numpy as np
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            raise ImportError(
                f"Fehlende Pakete: {e}. Installiere: pip install torch transformers"
            )

        self._torch = torch
        self._np    = np

        cfg_file = self.model_path / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Keine config.json in: {self.model_path}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        # Label-Mapping laden
        label_map_file = self.model_path / "label_mapping.json"
        if label_map_file.exists():
            with open(label_map_file, "r", encoding="utf-8") as f:
                lm = json.load(f)
            self.id2label = {int(k): v for k, v in lm.get("id2label", {}).items()}
        else:
            raw = model_cfg.get("id2label", {})
            self.id2label = {int(k): v for k, v in raw.items()}

        # Geraet waehlen (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

    def infer(self, text: str) -> dict:
        torch = self._torch
        np    = self._np

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        inference_time = time.time() - t0

        if isinstance(probs, float):
            probs = [probs]

        pred_id    = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        predicted  = self.id2label.get(pred_id, str(pred_id))

        top_n = min(5, len(probs))
        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
        top_predictions = [
            {"label": self.id2label.get(i, str(i)), "score": float(probs[i])}
            for i in sorted_ids
        ]

        return {
            "predicted":       predicted,
            "confidence":      confidence,
            "top_predictions": top_predictions,
            "inference_time":  inference_time,
        }

    def run(self):
        try:
            self.load()
        except Exception as e:
            emit_error(str(e))
            sys.exit(1)

        # Bereit-Signal an Rust
        emit({"type": "ready"})

        # Request-Loop: eine JSON-Zeile rein, eine JSON-Zeile raus
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError as e:
                emit_error(f"JSON parse error: {e}")
                continue

            if req.get("cmd") == "shutdown":
                break

            text = req.get("text", "")
            if not text:
                emit_error("Kein Text in der Anfrage")
                continue

            try:
                result = self.infer(text)
                emit(result)
            except Exception as e:
                emit_error(f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FrameTrain Persistent Model Server")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        emit_error(f"Modell-Pfad nicht gefunden: {args.model_path}")
        sys.exit(1)

    ModelServer(args.model_path).run()


if __name__ == "__main__":
    main()
