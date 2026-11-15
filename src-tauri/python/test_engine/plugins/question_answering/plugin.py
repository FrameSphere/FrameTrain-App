"""Test-Plugin: Extraktives Frage-Antwort (SQuAD-Stil).

single:  Kontext (Textfeld) + Frage (plugin_config.question) oder "Frage ||| Kontext"
         -> Antwort, der Kontext mit markierter Stelle »…« und die besten Kandidaten.
dataset: SQuAD-JSON oder Tabelle -> Exact Match (accuracy) und F1 wie beim
         SQuAD-Skript; bei Fragen ohne Antwort (v2) zusaetzlich getrennt.
Daten, Fenster und Bewertung kommen aus ft_data.qa — wie im Training.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data.media import sample as random_sample
from ft_data.splits import evaluation_files
from ft_data import qa as Q


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.pc = config.plugin_config or {}
        self.model = None
        self.tokenizer = None
        self.device = None
        self.spec: Dict[str, Any] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Frage-Antwort-Modell...")
        self.spec = Q.load_spec(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.device = resolve_device()
        self.model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))
        self.model.eval()
        self.model.to(self.device)
        TestProtocol.status("loading", f"Modell geladen | Gerät: {self.device}")

    def _setting(self, key: str, default):
        v = self.pc.get(key)
        if v is None or v == "":
            v = self.spec.get(key, default)
        return v

    def _predict(self, examples: List[Dict[str, Any]], allow_no_answer: bool) -> Dict[str, Dict[str, Any]]:
        import torch
        max_len = int(self._setting("max_length", 384))
        stride = int(self._setting("doc_stride", 128))
        feats = Q.prepare_eval_features(self.tokenizer, examples, max_len, stride)
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in feats]
        bs = max(int(self.config.batch_size or 16), 4)
        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, len(feats["input_ids"]), bs):
                if self.is_stopped:
                    break
                batch = {k: torch.tensor(feats[k][i:i + bs]).to(self.device) for k in keys}
                out = self.model(**batch)
                starts.extend(out.start_logits.float().cpu().numpy())
                ends.extend(out.end_logits.float().cpu().numpy())
        if len(starts) < len(feats["input_ids"]):
            return {}
        return Q.postprocess(examples, feats, starts, ends,
                             max_answer_length=int(self._setting("max_answer_length", 30)),
                             allow_no_answer=allow_no_answer,
                             null_threshold=float(self._setting("null_threshold", 0.0)))

    def _allow_no_answer(self, default: bool) -> bool:
        v = self.pc.get("allow_no_answer")
        if v is None or str(v).strip().lower() in ("", "auto"):
            return bool(self.spec.get("allow_no_answer", default))
        return str(v).strip().lower() in ("1", "true", "yes", "ja", "on")

    # ── Einzel-Eingabe ──────────────────────────────────────────────────────
    def run_single(self):
        t0 = time.time()
        context, question = Q.parse_single_input(self.config.single_input or "", str(self.pc.get("question") or ""))
        if not context.strip():
            raise ValueError("Kontext ist leer.")
        ex = {"id": "single", "context": context, "question": question, "answers": {"text": [], "answer_start": []}}
        pred = self._predict([ex], self._allow_no_answer(False))["single"]
        found = bool(pred["text"])
        marked = Q.mark_answer(context, pred["start"], pred["end"])
        predicted = (f"{pred['text']}\n\n{marked}" if found
                     else "Keine Antwort im Text / No answer in the text")
        top = [{"label": c["text"], "score": float(c["score"])} for c in pred["candidates"]]
        TestProtocol.complete_single(
            predicted=predicted, confidence=Q.span_probability(pred), top_predictions=top,
            inference_time=time.time() - t0,
            extra={"output_kind": "qa_span", "answer": pred["text"], "answer_start": pred["start"],
                   "answer_end": pred["end"], "context": context, "question": question},
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split, files = evaluation_files(root, Q.DATA_EXTS)
        if not files:
            raise ValueError(f"Keine Datendatei in '{root}' gefunden (erwartet: {', '.join(Q.DATA_EXTS)}).")
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}")
        rows, stats = Q.normalize_rows(Q.load_files(files), self.pc)
        TestProtocol.status("loading", Q.describe(stats))
        if not rows:
            raise ValueError("Keine verwertbare Frage im Dataset.")
        rows = random_sample(rows, self.config.max_samples)
        allow = self._allow_no_answer(any(not r["answers"]["text"] for r in rows))
        total = len(rows)
        started = time.time()
        preds: Dict[str, Dict[str, Any]] = {}
        chunk = 32
        for i in range(0, total, chunk):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            part = self._predict(rows[i:i + chunk], allow)
            if not part:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            preds.update(part)
            done = min(i + chunk, total)
            el = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=done, total=total, sps=done / el, eta=(total - done) / max(done / el, 1e-6))

        results, correct = [], 0
        for n, r in enumerate(rows, start=1):
            p = preds[r["id"]]
            em, f1 = Q.score_one(p["text"], r["answers"]["text"])
            correct += int(em)
            results.append({
                "sample_id": n,
                "input_text": f"{r['question']} ||| {r['context'][:400]}",
                "expected_output": " | ".join(r["answers"]["text"]) or "(keine Antwort)",
                "predicted_output": p["text"] or "(keine Antwort)",
                "is_correct": bool(em),
                "f1": round(f1, 4),
                "confidence": Q.span_probability(p),
                "inference_time": 0.0,
            })
        metrics = Q.squad_metrics({k: v["text"] for k, v in preds.items()}, rows)
        TestProtocol.status("running", " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
        elapsed = max(time.time() - started, 1e-6)
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": results, "metrics": metrics}, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results),
            accuracy=metrics.get("exact_match"), correct=correct, average_loss=None,
            average_inference_time=elapsed / max(len(results), 1),
            samples_per_second=len(results) / elapsed, metrics=metrics,
        )
