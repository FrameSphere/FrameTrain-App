"""
FrameTrain - Test Engine Orchestrator
======================================
Einstiegspunkt für alle Tests.

Wird von Rust aufgerufen:
  python3 /path/to/test_engine/test_engine.py --config /path/to/config.json

Modi:
  dataset  – Ganzen Datensatz testen → Metriken + Hard-Examples + Predictions-Datei
  single   – Einzelnen Input testen  → Sofortige Antwort ohne Datensatz

Ablauf:
  1. config.json laden
  2. Plugin automatisch erkennen (task_type oder Auto-Detection aus model_path)
  3. Plugin dynamisch aus plugins/<name>/plugin.py laden
  4. Plugin-Pipeline: setup → load_model → run_dataset | run_single
  5. JSON-Messages über stdout an Rust senden

Plugins:
  plugins/nlp/plugin.py       ← HuggingFace (BERT, GPT, T5, LLaMA, …)
  plugins/vision/plugin.py    ← Bildklassifikation (timm, torchvision)
  plugins/detection/plugin.py ← Objekterkennung (YOLO, Faster-RCNN)
  plugins/audio/plugin.py     ← Audio / ASR (Whisper, wav2vec2)
  plugins/tabular/plugin.py   ← Tabular (sklearn, XGBoost, LightGBM)
"""

import argparse
import importlib.util
import json
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol


# ============================================================================
# PLUGIN-ERKENNUNG  (identisches Schema wie train_engine)
# ============================================================================

TASK_TYPE_MAP = {
    # ── NLP / HuggingFace ─────────────────────────────────────────────────
    "nlp":                       "nlp",
    "auto":                      "nlp",
    "fine_tuning":               "nlp",
    "causal_lm":                 "nlp",
    "text_generation":           "nlp",
    "text_classification":       "nlp",
    "seq_classification":        "nlp",
    "sequence_classification":   "nlp",
    "summarization":             "nlp",
    "translation":               "nlp",
    "question_answering":        "nlp",
    "token_classification":      "nlp",
    "masked_lm":                 "nlp",
    "seq2seq":                   "nlp",
    "seq2seq_lm":                "nlp",
    "language_modeling":         "nlp",
    # ── Vision ────────────────────────────────────────────────────────────
    "vision":                    "vision",
    "image_classification":      "vision",
    "image":                     "vision",
    "cv":                        "vision",
    # ── Detection ─────────────────────────────────────────────────────────
    "detection":                 "detection",
    "object_detection":          "detection",
    "yolo":                      "detection",
    # ── Audio ─────────────────────────────────────────────────────────────
    "audio":                     "audio",
    "speech":                    "audio",
    "asr":                       "audio",
    "speech_recognition":        "audio",
    "audio_classification":      "audio",
    "whisper":                   "audio",
    # ── Tabular ───────────────────────────────────────────────────────────
    "tabular":                   "tabular",
    "regression":                "tabular",
    "classification":            "tabular",
    "structured":                "tabular",
}


def detect_plugin_from_model_path(model_path: str) -> str:
    """Auto-Detection aus Modell-Pfad als letzter Fallback."""
    p = model_path.lower()
    if "yolo" in p:
        return "detection"
    if any(k in p for k in ["whisper", "wav2vec", "hubert", "speech", "audio"]):
        return "audio"
    if any(k in p for k in ["vit_", "resnet", "efficientnet", "convnext", "swin_", "deit_"]):
        return "vision"
    # Versuche HuggingFace config.json zu lesen
    try:
        cfg_path = Path(model_path) / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            if archs:
                arch = archs[0].lower()
                if "vision" in arch or "vit" in arch or "swin" in arch:
                    return "vision"
                if "audio" in arch or "wav2vec" in arch or "whisper" in arch:
                    return "audio"
    except Exception:
        pass
    return "nlp"


def resolve_plugin_name(config: TestConfig) -> str:
    """Plugin-Name aus task_type oder Auto-Detection."""
    task = (config.task_type or "auto").lower().strip()
    if task in TASK_TYPE_MAP:
        plugin = TASK_TYPE_MAP[task]
        if task != "auto":
            return plugin
    if task not in ("auto",):
        MessageProtocol.warning(f"Unbekannter task_type '{task}' — Auto-Detection...")
    return detect_plugin_from_model_path(config.model_path)


# ============================================================================
# PLUGIN-LOADER
# ============================================================================

def load_plugin(plugin_name: str, config: TestConfig) -> TestPlugin:
    """Lädt Plugin dynamisch aus plugins/<plugin_name>/plugin.py."""
    plugin_file = Path(__file__).parent / "plugins" / plugin_name / "plugin.py"

    if not plugin_file.exists():
        raise FileNotFoundError(
            f"Test-Plugin nicht gefunden: {plugin_file}\n"
            f"Verfügbare Plugins: nlp, vision, detection, audio, tabular"
        )

    try:
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_name}.plugin", plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(
            f"Plugin '{plugin_name}' konnte nicht geladen werden:\n{e}\n\n{traceback.format_exc()}"
        )

    if not hasattr(module, "Plugin"):
        raise AttributeError(
            f"Plugin '{plugin_name}' enthält keine Klasse 'Plugin'.\n"
            f"Jedes Plugin muss 'class Plugin(TestPlugin)' definieren."
        )

    return module.Plugin(config)


# ============================================================================
# FEHLERBEHANDLUNG
# ============================================================================

def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "memoryerror", "oom",
        "allocation failed", "mps backend out of memory",
    ])


def handle_exception(exc: Exception) -> None:
    tb = traceback.format_exc()

    if is_oom(exc):
        MessageProtocol.error(
            "RAM-Fehler: Nicht genug Arbeitsspeicher",
            "Empfehlungen:\n"
            "  1. Batch-Size halbieren\n"
            "  2. Max-Samples begrenzen\n"
            "  3. Andere Apps schließen\n"
            f"\nDetail: {exc}"
        )
        return

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "No module named" in str(exc):
        MessageProtocol.error("Fehlendes Python-Paket", f"{exc}\n\n{tb}")
        return

    if isinstance(exc, FileNotFoundError):
        MessageProtocol.error("Datei/Ordner nicht gefunden", str(exc))
        return

    if isinstance(exc, ValueError):
        MessageProtocol.error(f"Konfigurationsfehler: {exc}", tb)
        return

    MessageProtocol.error(f"{type(exc).__name__}: {exc}", tb)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class Orchestrator:
    """Steuert den gesamten Test-Ablauf."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.plugin: Optional[TestPlugin] = None

        signal.signal(signal.SIGINT,  self._on_stop)
        signal.signal(signal.SIGTERM, self._on_stop)

    def _on_stop(self, *_):
        MessageProtocol.status("stopping", "Test wird gestoppt...")
        if self.plugin:
            self.plugin.stop()

    def run(self):
        cfg = self.config

        try:
            # ── 1. Plugin erkennen ────────────────────────────────────────
            plugin_name = resolve_plugin_name(cfg)
            MessageProtocol.status(
                "init",
                f"Plugin: {plugin_name}  |  task_type='{cfg.task_type}'  |  "
                f"mode='{cfg.mode}'  |  model='{Path(cfg.model_path).name}'"
            )

            # ── 2. Plugin laden ───────────────────────────────────────────
            self.plugin = load_plugin(plugin_name, cfg)

            # ── 3. Setup ──────────────────────────────────────────────────
            self.plugin.setup()
            if self.plugin.is_stopped:
                return

            # ── 4. Modell laden ───────────────────────────────────────────
            self.plugin.load_model()
            if self.plugin.is_stopped:
                return

            # ── 5a. SINGLE-MODUS ──────────────────────────────────────────
            if cfg.mode == "single":
                MessageProtocol.status("testing", "Einzeltest läuft...")
                result = self.plugin.run_single(cfg.single_input)

                MessageProtocol.complete({
                    "mode": "single",
                    "task_type": plugin_name,
                    "input": cfg.single_input,
                    "input_type": cfg.single_input_type,
                    "result": result,
                })
                return

            # ── 5b. DATASET-MODUS ─────────────────────────────────────────
            MessageProtocol.status("testing", "Dataset-Test läuft...")
            output = self.plugin.run_dataset()

            if self.plugin.is_stopped:
                MessageProtocol.status("stopped", "Test durch User gestoppt")
                return

            # Pflichtfelder absichern
            metrics = output.get("metrics", {})
            metrics.setdefault("accuracy", 0.0)
            metrics.setdefault("total_samples", output.get("total_samples", 0))

            MessageProtocol.complete({
                "mode": "dataset",
                "task_type": plugin_name,
                "metrics": metrics,
                "total_samples": output.get("total_samples", 0),
                "results_file": output.get("results_file", ""),
                "hard_examples_file": output.get("hard_examples_file"),
                # Kompakt-Zusammenfassung für Rust (ohne volle Predictions-Liste)
                "accuracy": metrics.get("accuracy", 0.0),
                "correct_predictions": metrics.get("correct_predictions", 0),
                "incorrect_predictions": metrics.get("incorrect_predictions", 0),
                "average_loss": metrics.get("average_loss"),
                "average_inference_time": metrics.get("average_inference_time", 0.0),
                "samples_per_second": metrics.get("samples_per_second", 0.0),
            })

        except Exception as exc:
            print(f"[FrameTrain Test] UNCAUGHT EXCEPTION: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            handle_exception(exc)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(0)


# ============================================================================
# EINSTIEGSPUNKT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Test Engine")
    parser.add_argument("--config", required=True, help="Pfad zur config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        MessageProtocol.error("Config nicht gefunden", f"Erwartet: {config_path}")
        sys.stdout.flush()
        sys.exit(0)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        MessageProtocol.error("config.json ungültig", str(e))
        sys.stdout.flush()
        sys.exit(0)

    config = TestConfig.from_dict(config_dict)
    Orchestrator(config).run()


if __name__ == "__main__":
    main()
