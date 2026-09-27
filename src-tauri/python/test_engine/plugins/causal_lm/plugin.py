"""Test-Plugin: Decoder-LLM (causal_lm) — Antworten erzeugen und bewerten.

single:  Eingabetext (oder eine JSON-Liste von Chat-Nachrichten) → Antwort.
dataset: Chat-/Frage-Antwort-Daten → Exact Match (als accuracy) + ROUGE-L je
         Zeile; reiner Text → Perplexitaet (average_loss = mittlerer Loss).
Daten und Chat-Template kommen aus ft_data.llm — dieselbe Aufbereitung wie
im Training, sonst sieht das Modell im Test ein anderes Format.
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data import llm as D
from ft_data.media import sample as random_sample


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        pc = config.plugin_config or {}
        self.max_new_tokens = int(pc.get("max_new_tokens", 256) or 256)
        self.temperature = float(pc.get("temperature", 0.0) or 0.0)
        self.system_prompt = str(pc.get("system_prompt") or "")
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        meta_file = model_path / "frametrain_llm.json"
        if meta_file.exists() and not self.system_prompt:
            try:
                self.system_prompt = json.loads(meta_file.read_text(encoding="utf-8")).get("system_prompt", "")
            except Exception:
                pass

        TestProtocol.status("loading", "Lade Sprachmodell ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        D.ensure_chat_template(self.tokenizer)
        self.device = resolve_device()
        cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        big = (cfg.get("num_hidden_layers", 0) * cfg.get("hidden_size", 0) ** 2 * 12) > 1.5e9
        dtype = torch.float32
        if self.device.type == "cuda" or (self.device.type == "mps" and big):
            dtype = torch.bfloat16 if self.device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype)
        self.model.eval()
        self.model.to(self.device)
        TestProtocol.status("loading", f"Modell geladen | {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M Parameter | Gerät: {self.device}")

    def _messages(self, text: str) -> List[Dict[str, str]]:
        raw = text.strip()
        if raw.startswith("["):
            try:
                msgs = json.loads(raw)
                if isinstance(msgs, list) and all(isinstance(m, dict) and "role" in m for m in msgs):
                    return [{"role": m["role"], "content": str(m.get("content", ""))} for m in msgs]
            except json.JSONDecodeError:
                pass
        msgs = [{"role": "user", "content": raw}]
        if self.system_prompt:
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return msgs

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        import torch
        text = D.render_chat(self.tokenizer, messages, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens,
                                  "pad_token_id": self.tokenizer.pad_token_id}
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _text_loss(self, text: str) -> Optional[float]:
        import torch
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False,
                             truncation=True, max_length=1024)["input_ids"].to(self.device)
        if ids.shape[1] < 2:
            return None
        with torch.no_grad():
            return float(self.model(input_ids=ids, labels=ids).loss)

    def run_single(self):
        text = self.config.single_input or ""
        if not text.strip():
            raise ValueError("Eingabetext ist leer.")
        t0 = time.time()
        answer = self._generate(self._messages(text))
        # Freier Text hat keine ehrliche Konfidenz — wie bei Seq2Seq keine erfinden.
        TestProtocol.complete_single(predicted=answer, confidence=None,
                                     top_predictions=[], inference_time=time.time() - t0)

    def run_dataset(self):
        # val_fraction=0: nichts abtrennen — ohne eigenen Test-/Val-Split zaehlt jede Zeile.
        data = D.load_examples(self.config.dataset_path, self.config.plugin_config or {}, val_fraction=0)
        pool = data.test or data.val or data.train
        rows = random_sample(pool, self.config.max_samples)
        TestProtocol.status("running", f"{len(rows)} Beispiele ({'Chat' if data.kind == 'chat' else 'Text'}) werden ausgewertet ...")

        results: List[Dict[str, Any]] = []
        exact, losses, total_time = 0, [], 0.0
        started = time.time()
        for idx, ex in enumerate(rows, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            if ex.is_chat:
                prompt = ex.prompt_messages()
                ref = ex.reference() or ""
                gen = self._generate(prompt)
                ok = D.exact_match(gen, ref)
                exact += int(ok)
                results.append({
                    "sample_id": idx, "input_text": (prompt[-1]["content"] if prompt else "")[:500],
                    "expected_output": ref, "predicted_output": gen, "is_correct": ok,
                    "rougeL": round(D.rouge_l(gen, ref), 4), "confidence": None,
                    "inference_time": time.time() - t0,
                })
            else:
                loss = self._text_loss(ex.text or "")
                if loss is not None:
                    losses.append(loss)
                results.append({
                    "sample_id": idx, "input_text": (ex.text or "")[:500], "expected_output": None,
                    "predicted_output": f"Perplexität {math.exp(min(loss, 50)):.2f}" if loss is not None else "-",
                    "is_correct": None, "loss": loss, "confidence": None,
                    "inference_time": time.time() - t0,
                })
            total_time += time.time() - t0
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(rows), sps=idx / elapsed)

        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        chat = data.kind == "chat"
        if chat and results:
            TestProtocol.status("running", f"ROUGE-L im Mittel: {sum(r['rougeL'] for r in results)/len(results):.3f}")
        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results),
            # "Accuracy" heisst hier: Antwort (normalisiert) identisch zur Referenz.
            accuracy=(exact / len(results)) if chat and results else None,
            correct=exact if chat else None,
            average_loss=(sum(losses) / len(losses)) if losses else None,
            average_inference_time=total_time / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
        )
