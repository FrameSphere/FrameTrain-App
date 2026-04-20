"""
plugins/nlp/plugin.py
=====================
Test-Plugin für alle Text/NLP-Modelle via HuggingFace Transformers.

Unterstützte Modell-Klassen:
  causal_lm          – GPT-2, LLaMA, Mistral, Falcon, …
  seq2seq            – T5, BART, Pegasus, MarianMT, …
  classification     – BERT, RoBERTa, DistilBERT + SequenceClassification

Dataset-Formate (im test/ oder val/ Unterordner):
  *.txt    – eine Zeile pro Sample; Tab-separiert für "input\\texpected"
  *.csv    – Spalten: input, expected (oder text, label)
  *.jsonl  – {"input":…, "output":…}  oder {"text":…, "label":…}
  *.json   – Array der obigen Objekte

Single-Modus:
  Nimmt beliebigen Text und gibt die Modell-Antwort zurück.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol


class Plugin(TestPlugin):

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model_class: str = "causal_lm"   # causal_lm | seq2seq | classification

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.device = self.get_device()
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.proto.status("init", f"NLP-Plugin | device={self.device}")

    # ── Modell laden ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        import os
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # Offline-Modus: verhindert versehentliche Hub-Downloads
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        model_path = self.config.model_path
        self.proto.status("loading", f"Lade Modell: {Path(model_path).name} …")

        # Architektur aus config.json ermitteln.
        # Wenn die Version kein eigenes config.json hat (Adapter-only), schauen wir
        # im Basis-Modell-Ordner nach (parent.parent = models/{hf_id}/)
        def _find_base_path(vpath: str) -> str:
            p = Path(vpath)
            # Direkt: config.json im Versions-Verzeichnis?
            if (p / "config.json").exists():
                return vpath
            # Adapter-only: versions/ver_xxx → models/hf_id/
            candidate = p.parent.parent
            if (candidate / "config.json").exists():
                self.proto.status("loading", f"config.json nicht in Version gefunden — nutze Basis-Modell: {candidate.name}")
                return str(candidate)
            # Noch eine Ebene höher (falls unerwartete Struktur)
            candidate2 = p.parent
            if (candidate2 / "config.json").exists():
                return str(candidate2)
            # Fallback: originalen Pfad behalten (HF lädt mit local_files_only)
            return vpath

        base_config_path = _find_base_path(model_path)

        # Architektur aus config.json ermitteln
        hf_config = AutoConfig.from_pretrained(base_config_path, local_files_only=True)
        archs = getattr(hf_config, "architectures", []) or []

        if archs:
            arch = archs[0]
            if "SequenceClassification" in arch:
                self.model_class = "classification"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                # Label-Mapping speichern falls vorhanden
                self.id2label = getattr(hf_config, "id2label", None)
            elif any(x in arch for x in ["T5", "MT5", "Bart", "Pegasus", "Marian", "Mbart"]):
                self.model_class = "seq2seq"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
            elif "MaskedLM" in arch or "ForMaskedLM" in arch:
                # Encoder-Modelle (BERT, RoBERTa, XLM-R, etc.) trainiert mit MLM
                # → für Inferenz als fill-mask verwenden
                self.model_class = "fill_mask"
                from transformers import AutoModelForMaskedLM
                self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
            else:
                self.model_class = "causal_lm"
                self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        else:
            # task_type als Fallback
            task = self.config.task_type.lower()
            if task in ("seq_classification", "sequence_classification", "text_classification"):
                self.model_class = "classification"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            elif task in ("seq2seq", "seq2seq_lm", "summarization", "translation"):
                self.model_class = "seq2seq"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
            elif task in ("masked_lm", "fill_mask", "mlm"):
                self.model_class = "fill_mask"
                from transformers import AutoModelForMaskedLM
                self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
            else:
                self.model_class = "causal_lm"
                self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

        self.model.to(self.device)
        self.model.eval()

        # Tokenizer laden
        tokenizer_path = self._find_tokenizer_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.proto.status("loaded", f"Modell geladen ({self.model_class})")

    def _find_tokenizer_path(self, model_path: str) -> str:
        """Tokenizer-Pfad ermitteln – ggf. aus Original-Modell-ID."""
        p = Path(model_path)
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json", "spiece.model"]
        if any((p / f).exists() for f in tokenizer_files):
            return model_path

        # tokenizer_info.txt (von train_engine geschrieben)
        info_file = p / "tokenizer_info.txt"
        if info_file.exists():
            for line in info_file.read_text().splitlines():
                if line.startswith("Original model:"):
                    orig = line.split(":", 1)[1].strip()
                    if not orig.startswith("/"):
                        self.proto.status("loading", f"Tokenizer von Original: {orig}")
                        return orig

        # config.json: _name_or_path
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                orig = cfg.get("_name_or_path") or cfg.get("original_model_id")
                if orig and not orig.startswith("/"):
                    return orig
            except Exception:
                pass

        return model_path  # Fallback

    # ── Dataset laden ──────────────────────────────────────────────────────

    def _load_test_data(self) -> List[Dict[str, Any]]:
        dataset_root = Path(self.config.dataset_path)
        test_dir = dataset_root / "test"
        if not test_dir.exists():
            test_dir = dataset_root / "val"
        if not test_dir.exists():
            test_dir = dataset_root   # direkt im Root-Verzeichnis suchen

        samples: List[Dict[str, Any]] = []

        for fp in sorted(test_dir.rglob("*")):
            if not fp.is_file():
                continue

            if fp.suffix == ".txt":
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        parts = line.split("\t", 1)
                        samples.append({"input": parts[0], "expected": parts[1]})
                    else:
                        samples.append({"input": line, "expected": None})

            elif fp.suffix == ".csv":
                with open(fp, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        inp = row.get("input") or row.get("text") or row.get("prompt") or ""
                        exp = row.get("expected") or row.get("output") or row.get("label")
                        if inp:
                            samples.append({"input": inp.strip(), "expected": exp})

            elif fp.suffix == ".jsonl":
                for line in fp.read_text(encoding="utf-8").splitlines():
                    try:
                        d = json.loads(line)
                        inp = d.get("input") or d.get("text") or d.get("prompt") or ""
                        exp = d.get("output") or d.get("expected") or d.get("label") or d.get("target")
                        if inp:
                            samples.append({"input": inp, "expected": exp})
                    except Exception:
                        pass

            elif fp.suffix == ".json":
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for d in data:
                            inp = d.get("input") or d.get("text") or d.get("prompt") or ""
                            exp = d.get("output") or d.get("expected") or d.get("label") or d.get("target")
                            if inp:
                                samples.append({"input": inp, "expected": exp})
                except Exception:
                    pass

        if not samples:
            raise ValueError(
                f"Keine Test-Daten gefunden in: {test_dir}\n"
                f"Unterstützte Formate: .txt, .csv, .jsonl, .json"
            )

        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        self.proto.status("loaded", f"{len(samples)} Test-Samples geladen")
        return samples

    # ── Einzelnes Sample testen ────────────────────────────────────────────

    def _infer_one(self, inp: str, expected: Optional[str]) -> Dict[str, Any]:
        import torch

        t0 = time.time()
        with torch.no_grad():
            enc = self.tokenizer(
                inp, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            if self.model_class == "classification":
                out = self.model(**enc)
                logits = out.logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_idx].item()

                # Label-Mapping
                if hasattr(self, "id2label") and self.id2label:
                    predicted = self.id2label.get(str(pred_idx), str(pred_idx))
                else:
                    predicted = str(pred_idx)

                loss = None
                if expected is not None:
                    try:
                        lbl = torch.tensor([int(expected)]).to(self.device)
                        loss = self.model(**enc, labels=lbl).loss.item()
                    except Exception:
                        pass

            elif self.model_class == "seq2seq":
                out_ids = self.model.generate(
                    **enc, max_length=256, num_beams=4, early_stopping=True
                )
                predicted = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                confidence = None
                loss = None
                if expected:
                    try:
                        lbl_enc = self.tokenizer(
                            expected, return_tensors="pt", truncation=True, max_length=256
                        ).input_ids.to(self.device)
                        loss = self.model(**enc, labels=lbl_enc).loss.item()
                    except Exception:
                        pass

            elif self.model_class == "fill_mask":
                mask_token = self.tokenizer.mask_token or '[MASK]'
                max_pos = getattr(self.model.config, 'max_position_embeddings', 514)
                max_content = max_pos - 3

                if mask_token not in inp:
                    toks = self.tokenizer.encode(inp, add_special_tokens=False)
                    if len(toks) > max_content:
                        toks = toks[:max_content]
                        inp = self.tokenizer.decode(toks, skip_special_tokens=True)
                    masked_inp = inp.rstrip() + f' {mask_token}'
                else:
                    text_no_mask = inp.replace(mask_token, '').strip()
                    toks = self.tokenizer.encode(text_no_mask, add_special_tokens=False)
                    if len(toks) > max_content:
                        toks = toks[:max_content]
                        text_no_mask = self.tokenizer.decode(toks, skip_special_tokens=True)
                    masked_inp = text_no_mask.rstrip() + f' {mask_token}'

                enc_masked = self.tokenizer(
                    masked_inp, return_tensors='pt', truncation=True,
                    max_length=max_pos - 2
                ).to(self.device)

                out = self.model(**enc_masked)
                logits = out.logits

                mask_id = self.tokenizer.mask_token_id
                input_ids_list = enc_masked['input_ids'][0]
                mask_positions = (input_ids_list == mask_id).nonzero(as_tuple=True)[0]

                if len(mask_positions) > 0:
                    mask_pos = mask_positions[0].item()
                    mask_logits = logits[0, mask_pos]
                    top_k = torch.topk(torch.softmax(mask_logits, dim=-1), k=5)
                    top_tokens = [self.tokenizer.decode([idx]).strip() for idx in top_k.indices]
                    top_scores = top_k.values.tolist()
                    predicted = top_tokens[0]
                    confidence = top_scores[0]
                    # Top-predictions für run_single Output
                    self._fill_mask_tops = [
                        {'label': tok, 'confidence': sc}
                        for tok, sc in zip(top_tokens, top_scores)
                    ]
                    # Sequenz mit erstem Token
                    full_seq = masked_inp.replace(mask_token, predicted)
                    self._fill_mask_sequence = full_seq
                else:
                    predicted = '(kein Mask-Token erkannt)'
                    confidence = None
                    self._fill_mask_tops = []
                    self._fill_mask_sequence = masked_inp

                loss = None

            else:  # causal_lm
                input_len = enc.input_ids.shape[1]
                out_ids = self.model.generate(
                    **enc,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                full = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                predicted = full[len(inp):].strip() if full.startswith(inp) else full
                confidence = None
                loss = None

        inference_time = time.time() - t0
        is_correct = False
        if expected is not None:
            is_correct = predicted.strip().lower() == str(expected).strip().lower()

        return {
            "input_text": inp,
            "predicted_output": predicted,
            "expected_output": expected,
            "is_correct": is_correct,
            "loss": loss,
            "confidence": confidence,
            "inference_time": inference_time,
            "error_type": None,
        }

    # ── Dataset-Modus ─────────────────────────────────────────────────────

    def run_dataset(self) -> Dict[str, Any]:
        samples = self._load_test_data()
        self.proto.status("testing", f"Teste {len(samples)} Samples …")

        results: List[Dict[str, Any]] = []
        t_start = time.time()

        for i, s in enumerate(samples):
            if self.is_stopped:
                break
            r = self._infer_one(s["input"], s.get("expected"))
            r["sample_id"] = i
            results.append(r)

            elapsed = time.time() - t_start
            sps = (i + 1) / elapsed if elapsed > 0 else 0.0
            self.proto.progress(i + 1, len(samples), sps)

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        losses = [r["loss"] for r in results if r["loss"] is not None]
        times = [r["inference_time"] for r in results]
        elapsed_total = time.time() - t_start

        metrics = {
            "accuracy": (correct / total * 100) if total > 0 else 0.0,
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "average_loss": (sum(losses) / len(losses)) if losses else None,
            "average_inference_time": (sum(times) / len(times)) if times else 0.0,
            "samples_per_second": total / elapsed_total if elapsed_total > 0 else 0.0,
            "total_time": elapsed_total,
        }

        # Ergebnisse speichern
        out = Path(self.config.output_path)
        results_file = out / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": results}, f, indent=2, default=str)

        # Hard-Examples speichern (falsch + hoher Loss)
        hard = [r for r in results if not r["is_correct"]]
        loss_threshold = (sum(losses) / len(losses) * 1.5) if losses else None
        if loss_threshold:
            hard += [r for r in results if r.get("loss") and r["loss"] > loss_threshold and r not in hard]

        hard_file = None
        if hard:
            hard_file = str(out / "hard_examples.jsonl")
            with open(hard_file, "w", encoding="utf-8") as f:
                for ex in hard:
                    f.write(json.dumps(ex, default=str) + "\n")
            self.proto.status("saved", f"{len(hard)} Hard-Examples gespeichert")

        return {
            "metrics": metrics,
            "predictions": results,
            "results_file": str(results_file),
            "hard_examples_file": hard_file,
            "total_samples": total,
            "task_type": "nlp",
        }

    # ── Single-Modus ───────────────────────────────────────────────────────

    def run_single(self, input_data: str) -> Dict[str, Any]:
        self.proto.status("inferring", "Führe Inferenz durch …")
        result = self._infer_one(input_data, None)
        out: Dict[str, Any] = {
            "output": result["predicted_output"],
            "model_class": self.model_class,
            "confidence": result.get("confidence"),
            "inference_time": result["inference_time"],
        }
        # fill_mask: Top-Predictions und vollständige Sequenz mitgeben
        if self.model_class == "fill_mask":
            out["top_predictions"] = getattr(self, '_fill_mask_tops', [])
            out["sequence"] = getattr(self, '_fill_mask_sequence', input_data)
        return out
