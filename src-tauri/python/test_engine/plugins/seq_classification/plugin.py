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

# Spaltenerkennung, Satzpaare und Label-Regeln gemeinsam mit dem Training.
from ft_data.text import detect_columns, expected_label, safe_text
from ft_data.media import sample as random_sample


class Plugin:
    def __init__(self, config: TestConfig):
        self.config    = config
        self.tokenizer = None
        self.model     = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        # Rohwert im Dataset -> Klassenname und die Spalten aus dem Training
        self.value_to_label: Dict[str, str] = {}
        self.train_columns: Dict[str, Optional[str]] = {}
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
                f"✗ Modell-Architektur '{model_type}' wird noch nicht unterstützt.\n"
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
            self.value_to_label = {str(k): str(v) for k, v in (lm.get("value_to_label") or {}).items()}
            self.train_columns = lm.get("columns") or {}
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

        TestProtocol.status("init", "✓ Modell geladen")

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

    def _infer_text(self, text: str, text_pair: Optional[str] = None):
        inputs = self.tokenizer(
            text,
            text_pair,
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
        # Zufaellige Stichprobe statt der ersten N: viele Splits sind nach Label
        # sortiert (imdb test: erst 12.500 negative) — die Accuracy mass sonst
        # nur eine Klasse.
        samples = random_sample(samples, self.config.max_samples)

        total = len(samples)
        TestProtocol.status("running", f"Inferenz auf {total} Samples...")

        predictions   = []
        correct_count = 0
        total_loss    = 0.0
        # Nur echte Labels zaehlen. Vorher war das immer wahr (der Schluessel
        # existiert immer) — ein ungelabeltes Dataset zeigte Accuracy 0 %.
        has_labels    = any(s.get("expected") is not None for s in samples)
        labelled      = 0
        t_start       = time.time()
        t_last_report = t_start

        for i, sample in enumerate(samples):
            if self.is_stopped:
                break

            text     = sample.get("text", "")
            text2    = sample.get("text_pair")
            expected = sample.get("expected")

            t0 = time.time()
            try:
                predicted, confidence, top_preds = self._infer_text(text, text2)

                # Optional: Loss berechnen wenn Label vorhanden
                sample_loss = None
                if expected is not None and expected in self.label2id:
                    label_id = self.label2id[str(expected)]
                    inputs = self.tokenizer(
                        text, text2, return_tensors="pt", truncation=True,
                        max_length=128, padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    label_tensor = torch.tensor([label_id], device=self.device)
                    with torch.no_grad():
                        out = self.model(**inputs, labels=label_tensor)
                        sample_loss = float(out.loss.item())
                    total_loss += sample_loss

                is_correct = (str(predicted) == str(expected)) if expected is not None else None
                if expected is not None:
                    labelled += 1
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
                    "is_correct":      False if expected is not None else None,
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
        accuracy      = correct_count / labelled if labelled > 0 else None
        avg_loss      = total_loss / labelled if labelled > 0 else None

        hard_examples = [p for p in predictions if p["is_correct"] is False and p.get("expected_output")]

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

    SUPPORTED_DATA_EXTS = (".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet", ".pq")

    def _find_dataset_files(self, root: Path) -> List[Path]:
        groups = self._dataset_file_groups(root)
        if not groups:
            raise ValueError(
                f"Kein unterstütztes Dataset (jsonl/json/csv/tsv/parquet) in Verzeichnis: {root}"
            )
        return groups[0]

    def _dataset_file_groups(self, root: Path) -> List[List[Path]]:
        """Dateigruppen je Split in Prioritaetsreihenfolge.

        WICHTIG: Bei gesplitteten Datasets MUSS der test-Split bevorzugt werden --
        sonst würde die Accuracy auf Trainingsdaten gemessen (Data Leakage).
        Priorität: test/ > val/ > Root-Dateien > alles andere (train/ zuletzt).
        Alle Shards eines Splits werden gelesen, nicht nur der erste.
        """
        META_NAMES = {"dataset_infos.json", "metadata.json", "config.json", "dataset.yaml"}

        def files_in(d: Path, recursive: bool = False) -> List[Path]:
            it = d.rglob("*") if recursive else d.iterdir()
            found = [
                f for f in it
                if f.is_file()
                and f.suffix.lower() in self.SUPPORTED_DATA_EXTS
                and f.name.lower() not in META_NAMES
                and ".frametrain_media" not in f.parts
            ]
            found.sort(key=lambda f: (0 if "test" in f.stem.lower() else 1, str(f)))
            return found

        def same_kind(files: List[Path]) -> List[Path]:
            ext = files[0].suffix.lower()
            return [f for f in files if f.suffix.lower() == ext]

        groups: List[List[Path]] = []
        # 1. Split-Ordner nach Priorität
        for sub in ("test", "testing", "val", "validation", "valid"):
            d = root / sub
            if d.is_dir():
                candidates = files_in(d, recursive=True)
                if candidates:
                    groups.append(same_kind(candidates))

        # 2. Dateien direkt im Root: bevorzugt die mit test/val im Namen
        candidates = files_in(root)
        if candidates:
            for key in ("test", "val"):
                named = [f for f in candidates if key in f.stem.lower()]
                if named:
                    groups.append(same_kind(named))
            no_train = [f for f in candidates if "train" not in f.stem.lower()]
            if no_train:
                groups.append(same_kind(no_train))

        # 3. Letzter Ausweg: train/ bzw. Trainingsdateien
        if not groups:
            all_files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in self.SUPPORTED_DATA_EXTS
                         and ".frametrain_media" not in f.parts]
            all_files.sort(key=lambda f: (1 if "train" in str(f.relative_to(root)).lower() else 0, str(f)))
            if all_files:
                groups.append(same_kind(all_files))
        return groups

    @staticmethod
    def _read_rows(path: Path) -> List[Dict[str, Any]]:
        ext = path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            rows = []
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if ext == ".json":
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
            return []
        if ext in (".csv", ".tsv"):
            import csv
            # utf-8-sig: Excel-Exporte beginnen mit einem BOM — sonst hiess die
            # erste Spalte "\ufefftext" und wurde nicht erkannt.
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
        if ext in (".parquet", ".pq"):
            # HuggingFace-Downloads liegen standardmäßig als Parquet vor.
            import pandas as pd
            df = pd.read_parquet(path)
            return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
        raise ValueError(f"Nicht unterstütztes Dataset-Format: {ext}")

    def _load_samples(self, path: Path) -> List[Dict[str, Any]]:
        """Lädt Samples aus JSON/JSONL/CSV/TSV/Parquet in eine einheitliche Struktur."""
        if not path.is_dir():
            return self._samples_from([path])

        groups = self._dataset_file_groups(path)
        if not groups:
            raise ValueError(
                f"Kein unterstütztes Dataset (jsonl/json/csv/tsv/parquet) in Verzeichnis: {path}"
            )
        first: Optional[List[Dict[str, Any]]] = None
        for files in groups:
            rel = ", ".join(str(f.relative_to(path)) for f in files[:3])
            more = f" (+{len(files) - 3} weitere)" if len(files) > 3 else ""
            samples = self._samples_from(files)
            if first is None:
                first = samples
            if any(s["expected"] is not None for s in samples):
                TestProtocol.status("loading", f"Verwende Datei(en): {rel}{more}")
                return samples
            # HF-Testsplits (GLUE u.a.) haben label = -1 — dann den naechsten
            # Split mit echten Labels nehmen, sonst gaebe es keine Accuracy.
            TestProtocol.status("loading", f"{rel}: keine Labels (z.B. -1) — suche gelabelten Split...")
        TestProtocol.status("loading", "Kein Split mit Labels gefunden — nur Vorhersagen, keine Accuracy.")
        return first or []

    def _samples_from(self, files: List[Path]) -> List[Dict[str, Any]]:
        raw_rows: List[Dict] = []
        for f in files:
            raw_rows.extend(self._read_rows(f))

        if not raw_rows:
            raise ValueError("Dataset ist leer.")

        sample_keys = list(raw_rows[0].keys())
        text_col, pair_col, label_col = detect_columns(sample_keys)
        # Spalten aus dem Training haben Vorrang, wenn es sie hier gibt.
        trained = self.train_columns or {}
        if trained.get("text") in sample_keys:
            text_col = trained["text"]
            pair_col = trained.get("text_pair") if trained.get("text_pair") in sample_keys else None
        if trained.get("label") in sample_keys:
            label_col = trained["label"]
        if text_col is None:
            text_col = sample_keys[0]

        samples = []
        for row in raw_rows:
            samples.append({
                "text":      safe_text(row.get(text_col)),
                "text_pair": safe_text(row.get(pair_col)) if pair_col else None,
                "expected":  expected_label(row.get(label_col), self.value_to_label) if label_col else None,
            })
        return samples
