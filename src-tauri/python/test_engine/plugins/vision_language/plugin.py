"""Test-Plugin: Vision-Language-Modell (Bild + Frage -> Antwort).

single:  single_input ist der Bildpfad, die Frage steht in plugin_config
         "question". Fuer Aufrufer mit nur einem Eingabefeld geht auch
         "pfad/zum/bild.jpg ||| Frage".
dataset: Antworten fuer die Val/Test-Beispiele, Kennzahlen exact match und
         ROUGE-L (results.json: metrics + predictions).
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol

SEPARATOR = "|||"


def split_single_input(raw: str, question: str = "") -> Tuple[str, str]:
    """(Bildpfad, Frage). Eine Frage aus plugin_config hat Vorrang vor der
    Kurzform 'pfad ||| frage'."""
    raw = (raw or "").strip()
    path, inline = raw, ""
    if SEPARATOR in raw:
        path, inline = (p.strip() for p in raw.split(SEPARATOR, 1))
    return path.strip().strip('"').strip("'"), (question or "").strip() or inline


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        pc = config.plugin_config or {}
        self.question = str(pc.get("question", "") or "")
        self.max_new_tokens = int(pc.get("max_new_tokens", 0) or 0)
        self.model = None
        self.processor = None
        self.family = "chat"
        self.info: Dict[str, Any] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from _shared_classify import resolve_device
        from ft_data import vlm as vlmlib

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Vision-Language-Modell...")
        device = resolve_device()
        self.model, self.processor, self.family, self.info = vlmlib.load_model(model_path, device)
        if not self.max_new_tokens:
            self.max_new_tokens = int(self.info.get("max_new_tokens") or 64)
        TestProtocol.status("loading", f"Modell geladen | Prompt-Format: {self.family} | Geraet: {device}")

    @property
    def default_prompt(self) -> str:
        from ft_data.vlm import DEFAULT_PROMPT
        # "" ist ein gueltiger Standard (BLIP beschreibt ohne Text-Praefix).
        return str(self.info.get("default_prompt", DEFAULT_PROMPT) or "")

    def _answer(self, images: List[Any], questions: List[str]) -> List[str]:
        from ft_data.vlm import generate
        return generate(self.model, self.processor, self.family, images, questions, self.max_new_tokens)

    def run_single(self):
        from PIL import Image

        path_str, question = split_single_input(self.config.single_input, self.question)
        path = Path(path_str)
        if not path_str or not path.exists():
            raise ValueError(f"Bilddatei nicht gefunden: {path_str or '(leer)'}")
        question = question or self.default_prompt
        t0 = time.time()
        with Image.open(path) as im:
            img = im.convert("RGB")
        answer = self._answer([img], [question])[0]
        TestProtocol.complete_single(
            predicted=answer, confidence=None, top_predictions=[],
            inference_time=time.time() - t0,
            extra={"question": question, "image_path": str(path)},
        )

    def run_dataset(self):
        from PIL import Image
        from ft_data.image_text import evaluation_samples, normalise_answer, text_scores
        from ft_data.media import sample as random_sample

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        samples, split = evaluation_samples(root, "vlm", status=lambda m: TestProtocol.status("loading", m))
        samples = random_sample(samples, self.config.max_samples)
        TestProtocol.status("running", f"{len(samples)} Beispiele aus '{split}' werden beantwortet...")

        bs = max(1, min(int(self.config.batch_size or 1), 8))
        results: List[Dict[str, Any]] = []
        preds: List[str] = []
        started = time.time()
        for i in range(0, len(samples), bs):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            chunk = samples[i:i + bs]
            imgs = []
            for s in chunk:
                with Image.open(s.image) as im:
                    imgs.append(im.convert("RGB"))
            questions = [s.prompt or self.default_prompt for s in chunk]
            t0 = time.time()
            answers = self._answer(imgs, questions)
            dt = (time.time() - t0) / len(chunk)
            for s, q, a in zip(chunk, questions, answers):
                preds.append(a)
                results.append({
                    "sample_id": len(results) + 1,
                    "input_text": q,
                    "input_path": str(s.image),
                    "expected_output": s.answer,
                    "predicted_output": a,
                    "is_correct": normalise_answer(a) == normalise_answer(s.answer),
                    "confidence": None,
                    "inference_time": dt,
                })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=len(results), total=len(samples), sps=len(results) / elapsed)

        scores = text_scores(preds, [s.answer for s in samples])
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        results_file = out / "results.json"
        results_file.write_text(json.dumps({"metrics": scores, "predictions": results},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        correct = sum(1 for r in results if r["is_correct"])
        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results),
            # "Accuracy" = exakter Treffer nach Normalisierung (Gross/Klein, Satzzeichen).
            accuracy=scores.get("exact_match"), correct=correct, average_loss=None,
            average_inference_time=sum(r["inference_time"] for r in results) / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
            extra={"metrics": scores},
        )
