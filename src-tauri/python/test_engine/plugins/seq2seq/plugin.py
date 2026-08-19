"""Test-Plugin: Seq2Seq-Inferenz (Zusammenfassung, Übersetzung, Umformung)."""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device

SOURCE_CANDIDATES = ["source", "input", "text", "article", "document", "de", "src", "question"]
TARGET_CANDIDATES = ["target", "output", "summary", "highlights", "translation", "en", "tgt", "answer"]
DATA_EXTS = (".json", ".jsonl", ".csv", ".tsv", ".parquet")


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = None
        self.prefix = str(config.plugin_config.get("task_prefix", "") or "")
        self.max_new_tokens = int(config.plugin_config.get("max_target_length", 64))
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")

        TestProtocol.status("loading", "Lade Seq2Seq-Modell...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)
        TestProtocol.status("loading", f"Modell geladen | Gerät: {self.device}")

    def _generate(self, text: str) -> str:
        import torch

        enc = self.tokenizer(f"{self.prefix}{text}", return_tensors="pt",
                             truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def run_single(self):
        text = self.config.single_input
        if not text.strip():
            raise ValueError("Eingabetext ist leer.")
        t0 = time.time()
        generated = self._generate(text)
        # Seq2Seq erzeugt freien Text — es gibt keine Klassen und damit auch
        # keine Konfidenz. Eine erfundene Zahl waere schlechter als keine.
        TestProtocol.complete_single(
            predicted=generated, confidence=None,
            top_predictions=[], inference_time=time.time() - t0,
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    def _find_file(self, root: Path) -> Path:
        for sub in ("test", "val", "validation", "train"):
            d = root / sub
            if d.is_dir():
                for f in sorted(d.rglob("*")):
                    if f.suffix.lower() in DATA_EXTS:
                        return f
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in DATA_EXTS:
                return f
        raise ValueError(f"Keine Datendatei in '{root}' gefunden (erwartet: {', '.join(DATA_EXTS)}).")

    def _load_rows(self, path: Path) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ext = path.suffix.lower()
        fmt = {"jsonl": "json", "json": "json", "parquet": "parquet"}.get(ext.lstrip("."), "csv")
        kwargs = {"data_files": str(path)}
        if ext == ".tsv":
            kwargs["delimiter"] = "\t"
        ds = load_dataset(fmt, **kwargs)["train"]
        return [dict(r) for r in ds]

    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

        rows = self._load_rows(self._find_file(root))
        if not rows:
            raise ValueError("Dataset enthaelt keine Zeilen.")
        cols = list(rows[0].keys())
        src = next((c for c in SOURCE_CANDIDATES if c in cols), None)
        tgt = next((c for c in TARGET_CANDIDATES if c in cols and c != src), None)
        if src is None:
            raise ValueError(f"Eingabespalte nicht erkannt. Vorhanden: {cols}")

        if self.config.max_samples:
            rows = rows[: int(self.config.max_samples)]
        TestProtocol.status("running", f"{len(rows)} Zeilen werden ausgewertet...")

        results: List[Dict[str, Any]] = []
        exact = 0
        labelled = 0
        total_time = 0.0
        started = time.time()

        for idx, row in enumerate(rows, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            generated = self._generate(str(row.get(src, "")))
            dt = time.time() - t0
            total_time += dt

            expected: Optional[str] = str(row[tgt]) if tgt and row.get(tgt) is not None else None
            is_correct: Optional[bool] = None
            if expected is not None:
                labelled += 1
                is_correct = generated.strip() == expected.strip()
                if is_correct:
                    exact += 1

            results.append({
                "sample_id": idx,
                "input_text": str(row.get(src, ""))[:500],
                "expected_output": expected,
                "predicted_output": generated,
                "is_correct": is_correct,
                "confidence": None,
                "inference_time": dt,
            })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(rows), sps=idx / elapsed)

        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=len(results),
            # "Accuracy" heisst hier: Wort fuer Wort identisch zum Ziel.
            accuracy=(exact / labelled) if labelled else None,
            correct=exact if labelled else None,
            average_loss=None,
            average_inference_time=total_time / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
        )
