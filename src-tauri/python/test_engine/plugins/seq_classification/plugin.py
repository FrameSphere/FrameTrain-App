"""
plugins/seq_classification/plugin.py – Test/Inferenz-Plugin
=============================================================
Lädt ein trainiertes XLM-RoBERTa / BERT / DeBERTa Modell und führt
Sequenzklassifikations-Inferenz durch – entweder auf einem Dataset
oder auf einem einzelnen Text-Input.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    raise

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TestConfig
from core.protocol import TestProtocol

# Unterstützte Architekturen (muss mit train-Plugin übereinstimmen)
SUPPORTED_ARCHITECTURES = {
    "xlm-roberta", "roberta", "bert", "deberta", "deberta-v2",
    "distilbert", "albert", "camembert", "electra", "rembert",
    "xlm", "ernie", "funnel", "mpnet", "squeezebert", "layoutlm",
}

TEXT_COLUMN_NAMES = [
    "text", "sentence", "content", "review_body", "input", "document",
    "title", "body", "description", "abstract", "question", "passage",
    "premise", "hypothesis",
]
LABEL_COLUMN_NAMES = ["label", "labels", "category", "class", "target", "sentiment"]
ID_COLUMN_NAMES    = {"id", "idx", "index", "row_id", "sample_id", "uid", "uuid", "key"}


class Plugin:
    def __init__(self, config: TestConfig):
        self.config    = config
        self.tokenizer = None
        self.model     = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.device    = None
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    # ─── Setup ────────────────────────────────────────────────────────────

    def setup(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modell-Verzeichnis nicht gefunden: {model_path}")

        cfg_file = model_path / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Keine config.json in: {model_path}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        model_type = model_cfg.get("model_type", "").lower()
        if model_type not in SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ValueError(
                f"❌ Modell-Architektur '{model_type}' wird noch nicht unterstützt.\n"
                f"Unterstützte Architekturen: {supported}"
            )

        TestProtocol.status("init", f"Architektur: {model_type} | Lade Tokenizer & Modell...")

        # Label-Mapping aus config.json oder label_mapping.json
        label_map_file = model_path / "label_mapping.json"
        if label_map_file.exists():
            with open(label_map_file, "r", encoding="utf-8") as f:
                lm = json.load(f)
            self.label2id = lm.get("label2id", {})
            self.id2label = {int(k): v for k, v in lm.get("id2label", {}).items()}
        else:
            # Fallback: aus config.json
            raw_id2label = model_cfg.get("id2label", {})
            self.id2label = {int(k): v for k, v in raw_id2label.items()}
            self.label2id = {v: int(k) for k, v in raw_id2label.items()}

        # Gerät
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        TestProtocol.status("init", f"Gerät: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        TestProtocol.status("init", "✅ Modell geladen")

    # ─── Single-Input Inferenz ────────────────────────────────────────────

    def run_single(self):
        text = self.config.single_input
        if not text.strip():
            raise ValueError("Eingabetext ist leer.")

        t0 = time.time()
        predicted, confidence, top_preds = self._infer_text(text)
        inference_time = time.time() - t0

        TestProtocol.complete_single(
            predicted=predicted,
            confidence=confidence,
            top_predictions=top_preds,
            inference_time=inference_time,
        )

    def _infer_text(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        pred_id    = int(np.argmax(probs))
        confidence = float(probs[pred_id]) if len(probs) > 0 else None
        predicted  = self.id2label.get(pred_id, str(pred_id))

        top_n = min(5, len(probs))
        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
        top_preds = [
            {"label": self.id2label.get(i, str(i)), "score": float(probs[i])}
            for i in sorted_ids
        ]

        return predicted, confidence, top_preds

    # ─── Dataset-Inferenz ─────────────────────────────────────────────────

    def run_dataset(self):
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_path}")

        TestProtocol.status("loading", f"Lade Dataset: {dataset_path.name}")

        samples = self._load_samples(dataset_path)
        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        total = len(samples)
        TestProtocol.status("running", f"Inferenz auf {total} Samples...")

        predictions   = []
        correct_count = 0
        total_loss    = 0.0
        has_labels    = any("expected" in s for s in samples)
        t_start       = time.time()
        t_last_report = t_start

        for i, sample in enumerate(samples):
            if self.is_stopped:
                break

            text     = sample.get("text", "")
            expected = sample.get("expected")

            t0 = time.time()
            try:
                predicted, confidence, top_preds = self._infer_text(text)

                # Optional: Loss berechnen wenn Label vorhanden
                sample_loss = None
                if expected is not None and expected in self.label2id:
                    label_id = self.label2id[str(expected)]
                    inputs = self.tokenizer(
                        text, return_tensors="pt", truncation=True,
                        max_length=128, padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    label_tensor = torch.tensor([label_id], device=self.device)
                    with torch.no_grad():
                        out = self.model(**inputs, labels=label_tensor)
                        sample_loss = float(out.loss.item())
                    total_loss += sample_loss

                is_correct = (str(predicted) == str(expected)) if expected is not None else False
                if is_correct:
                    correct_count += 1

                inference_time = time.time() - t0
                predictions.append({
                    "sample_id":       i,
                    "input_text":      text[:500],
                    "expected_output": str(expected) if expected is not None else None,
                    "predicted_output": predicted,
                    "is_correct":      is_correct,
                    "loss":            sample_loss,
                    "confidence":      confidence,
                    "inference_time":  inference_time,
                    "top_predictions": top_preds,
                })
            except Exception as e:
                predictions.append({
                    "sample_id":       i,
                    "input_text":      text[:500],
                    "expected_output": str(expected) if expected is not None else None,
                    "predicted_output": "ERROR",
                    "is_correct":      False,
                    "loss":            None,
                    "confidence":      None,
                    "inference_time":  time.time() - t0,
                    "error_type":      type(e).__name__,
                })

            # Progress alle 0.5s oder alle 10 Samples
            now = time.time()
            if (i + 1) % 10 == 0 or (now - t_last_report) >= 0.5:
                elapsed = now - t_start
                sps     = (i + 1) / max(elapsed, 1e-6)
                eta     = (total - i - 1) / max(sps, 1e-6)
                TestProtocol.progress(i + 1, total, sps=sps, eta=eta)
                t_last_report = now

        # ── Metriken zusammenstellen ──────────────────────────────────────
        elapsed_total = time.time() - t_start
        completed     = len(predictions)
        sps_final     = completed / max(elapsed_total, 1e-6)
        avg_infer     = elapsed_total / max(completed, 1)
        accuracy      = correct_count / completed if completed > 0 and has_labels else None
        avg_loss      = total_loss / completed if completed > 0 and has_labels else None

        hard_examples = [p for p in predictions if not p["is_correct"] and p.get("expected_output")]

        # Ergebnis-JSON speichern
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "predictions": predictions,
                "metrics": {
                    "accuracy":             accuracy,
                    "correct_predictions":  correct_count,
                    "total_samples":        completed,
                    "average_loss":         avg_loss,
                    "average_inference_time": avg_infer,
                    "samples_per_second":   sps_final,
                    "total_time":           elapsed_total,
                },
            }, f, ensure_ascii=False, indent=2, default=str)

        # Hard-Examples speichern
        hard_file = None
        if hard_examples:
            hard_file = str(output_dir / "hard_examples.json")
            with open(hard_file, "w", encoding="utf-8") as f:
                json.dump(hard_examples, f, ensure_ascii=False, indent=2, default=str)

        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=completed,
            accuracy=accuracy,
            correct=correct_count if has_labels else None,
            average_loss=avg_loss,
            average_inference_time=avg_infer,
            samples_per_second=sps_final,
            hard_examples_file=hard_file,
        )

    # ─── Dataset-Loader ───────────────────────────────────────────────────

    def _load_samples(self, path: Path) -> List[Dict[str, Any]]:
        """Lädt Samples aus JSON/JSONL/CSV in eine einheitliche Struktur."""

        # Falls der Pfad ein Verzeichnis ist, suche darin nach einer unterstützten Datei
        if path.is_dir():
            for ext_try in (".jsonl", ".ndjson", ".json", ".csv"):
                for candidate in path.rglob(f"*{ext_try}"):
                    if candidate.is_file():
                        path = candidate
                        break
                if path.is_file():
                    break
            else:
                raise ValueError(f"Kein unterstütztes Dataset (jsonl/json/csv) in Verzeichnis: {path}")

        ext = path.suffix.lower()

        raw_rows: List[Dict] = []

        if ext in (".jsonl", ".ndjson"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_rows = data
            elif isinstance(data, dict):
                # {"data": [...]} o.ä.
                for v in data.values():
                    if isinstance(v, list):
                        raw_rows = v
                        break
        elif ext == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                raw_rows = list(reader)
        else:
            raise ValueError(f"Nicht unterstütztes Dataset-Format: {ext}")

        # Spalten normalisieren
        if not raw_rows:
            raise ValueError("Dataset ist leer.")

        sample_keys = list(raw_rows[0].keys())
        non_id_keys = [c for c in sample_keys if c.lower() not in ID_COLUMN_NAMES]

        # Text-Spalte: erst bekannte Namen, dann erste Nicht-ID-Spalte
        text_col = next((c for c in TEXT_COLUMN_NAMES if c in sample_keys), None)
        if text_col is None:
            text_col = non_id_keys[0] if non_id_keys else sample_keys[0]

        # Label-Spalte: erst bekannte Namen, dann letzte Nicht-ID-Spalte als Fallback
        label_col = next((c for c in LABEL_COLUMN_NAMES if c in sample_keys), None)
        if label_col is None and len(non_id_keys) >= 2:
            label_col = non_id_keys[-1]

        samples = []
        for row in raw_rows:
            raw_label = row.get(label_col) if label_col else None
            # Listen-Labels (z.B. prmu: ["P","R",...]) → erstes Element nehmen
            if isinstance(raw_label, list):
                raw_label = raw_label[0] if raw_label else None
            samples.append({
                "text":     str(row.get(text_col, "")),
                "expected": str(raw_label) if raw_label is not None else None,
            })
        return samples
