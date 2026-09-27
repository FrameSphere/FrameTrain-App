"""Test-Plugin: Token-Klassifikation (NER, POS) mit einem trainierten HF-Modell.

single:  Text rein, Entitaeten raus ("Karol [PER], Berlin [LOC]").
dataset: Entitaeten-F1 (seqeval) ueber den Test-Split; "Treffer" zaehlt
         Saetze, deren Tags vollstaendig stimmen.
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
from _shared_classify import load_label_names, resolve_device
from ft_data.media import sample as random_sample
from ft_data.splits import evaluation_files
from ft_data import tokens as tk

PREFIX_SPACE_TYPES = {"roberta", "longformer", "deberta", "bart", "gpt2", "mvp", "led"}


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = None
        self.id2label: Dict[int, str] = {}
        self.bio = True
        self.spec: Dict[str, Any] = {}
        self.max_length = 512
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")

        TestProtocol.status("loading", "Lade Token-Klassifikationsmodell...")
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        model_type = str(getattr(self.model.config, "model_type", "") or "")
        kwargs = {"add_prefix_space": True} if model_type in PREFIX_SPACE_TYPES else {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **kwargs)
        self.model.eval()
        self.device = resolve_device()
        self.model.to(self.device)
        self.id2label = load_label_names(model_path, self.model.config)
        self.spec = tk.load_spec(model_path)
        self.bio = tk.is_bio_scheme(self.id2label.values())
        self.max_length = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        if self.max_length > 4096:  # "unendlich" bei manchen Tokenizern
            self.max_length = 512
        TestProtocol.status(
            "loading",
            f"Modell geladen | Labels: {len(self.id2label)} | Gerät: {self.device}",
        )

    def _predict(self, words: List[str]):
        preds = tk.predict_words(self.model, self.tokenizer, words, self.id2label,
                                 self.device, self.max_length)
        return [p[0] for p in preds], [p[1] for p in preds]

    def run_single(self):
        text = self.config.single_input or ""
        if not text.strip():
            raise ValueError("Eingabetext ist leer.")
        t0 = time.time()
        words = tk.split_words(text)
        tags, scores = self._predict([w for w, _, _ in words])
        ents = tk.entities_from_tags(words, tags, scores, text=text, bio=self.bio)
        predicted = tk.format_entities(ents) or "Keine Entitaeten gefunden"
        top = [{"label": f"{e['text']} [{e['label']}]", "score": e["score"],
                "entity": e["label"], "text": e["text"], "start": e["start"], "end": e["end"]}
               for e in ents]
        # Eine Konfidenz fuer den ganzen Satz: die unsicherste Entitaet — sie
        # entscheidet, ob man das Ergebnis nachpruefen sollte.
        confidence = min((e["score"] for e in ents), default=None)
        TestProtocol.complete_single(
            predicted=predicted, confidence=confidence,
            top_predictions=top, inference_time=time.time() - t0,
        )

    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split, files = evaluation_files(root, tk.DATA_EXTS)
        if not files:
            raise ValueError(f"Keine Datendatei in '{root}' gefunden (erwartet: {', '.join(tk.DATA_EXTS)}).")
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}")
        cfg = {k: v for k, v in self.spec.items() if k in ("tokens_column", "tags_column")}
        cfg.update(self.config.plugin_config or {})
        try:
            sentences, _ = tk.load_files(files, cfg)
        except ValueError:
            # Spalten aus dem Training fehlen hier — frei erkennen.
            sentences, _ = tk.load_files(files, self.config.plugin_config or {})
        if not sentences:
            raise ValueError("Dataset enthaelt keine verwertbaren Saetze.")
        sentences = random_sample(sentences, self.config.max_samples)
        TestProtocol.status("running", f"{len(sentences)} Saetze werden ausgewertet...")

        known = set(self.id2label.values())
        results: List[Dict[str, Any]] = []
        true_all, pred_all = [], []
        exact = 0
        total_time = 0.0
        started = time.time()
        for idx, s in enumerate(sentences, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            tags, scores = self._predict(s["tokens"])
            dt = time.time() - t0
            total_time += dt
            # Tags, die das Modell nicht kennt, koennen nicht getroffen werden —
            # sie bleiben als Soll stehen und zaehlen als verfehlt.
            true_all.append(s["tags"])
            pred_all.append(tags)
            ok = tags == s["tags"]
            exact += int(ok)
            exp_ents = tk.entities_from_tags(s["tokens"], s["tags"], bio=self.bio)
            pred_ents = tk.entities_from_tags(s["tokens"], tags, scores, bio=self.bio)
            results.append({
                "sample_id": idx,
                "input_text": " ".join(s["tokens"])[:500],
                "expected_output": tk.format_entities(exp_ents) or "(keine)",
                "predicted_output": tk.format_entities(pred_ents) or "(keine)",
                "is_correct": ok,
                "confidence": min(scores) if scores else None,
                "inference_time": dt,
            })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(sentences), sps=idx / elapsed)

        scores_all = tk.score_sequences(true_all, pred_all)
        unknown = sorted({t for seq in true_all for t in seq} - known)
        if unknown:
            TestProtocol.status("running", f"Hinweis: Tags ohne Gegenstueck im Modell: {', '.join(unknown[:10])}")
        entity = scores_all.get("scheme") == "entity"
        metrics = {
            ("entity_f1" if entity else "f1"): scores_all["f1"],
            ("entity_precision" if entity else "precision"): scores_all["precision"],
            ("entity_recall" if entity else "recall"): scores_all["recall"],
            "token_accuracy": scores_all["accuracy"],
            "sentence_exact_match": exact / max(len(results), 1),
        }
        TestProtocol.status(
            "running",
            f"{'Entitaeten-' if entity else ''}F1 {scores_all['f1']:.3f} | Precision {scores_all['precision']:.3f} | "
            f"Recall {scores_all['recall']:.3f} | Token-Accuracy {scores_all['accuracy']:.3f}",
        )

        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        # Objekt mit predictions + metrics (wie seq_classification): so legt die
        # App Einzelergebnisse und Entitaeten-Kennzahlen in der Testhistorie ab.
        results_file.write_text(json.dumps({"predictions": results, "metrics": metrics},
                                           indent=2, ensure_ascii=False), encoding="utf-8")

        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=len(results),
            # "Treffer" = Satz komplett richtig getaggt; die Entitaeten-Kennzahlen stehen in metrics.
            accuracy=exact / max(len(results), 1),
            correct=exact,
            average_loss=None,
            average_inference_time=total_time / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
            metrics=metrics,
        )
