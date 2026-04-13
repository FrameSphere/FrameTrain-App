"""
FrameTrain - Training Engine Orchestrator
=========================================
Einstiegspunkt für alle Trainings.

Wird von Rust aufgerufen:
  python3 /path/to/train_engine/train_engine.py --config /path/to/config.json

Der Rust-Code (run_training_process) startet Python als Subprocess mit
stdout: piped und stderr: piped. stdout wird Zeile für Zeile als JSON geparst.

Ablauf:
  1. config.json laden
  2. Plugin automatisch erkennen (task_type oder auto-detect aus model_path)
  3. Plugin dynamisch aus plugins/<name>/plugin.py laden
  4. Plugin-Pipeline: setup → load_data → build_model → train → validate → export
  5. JSON-Messages über stdout an Rust senden

Plugins:
  plugins/nlp/plugin.py       ← HuggingFace (BERT, GPT, T5, LLaMA, ...)
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

# Core-Paket liegt im selben Verzeichnis wie dieser Orchestrator
sys.path.insert(0, str(Path(__file__).parent))

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


# ============================================================================
# PLUGIN-ERKENNUNG
# ============================================================================

# Alle task_type-Werte die vom Rust-Frontend kommen können → Plugin-Name
# Rust-Defaults: "causal_lm" (Presets nutzen auch "seq_classification", "constant" etc.)
TASK_TYPE_MAP = {
    # ── NLP / HuggingFace ─────────────────────────────────────────────────
    "nlp":                  "nlp",
    "auto":                 "nlp",           # Rust-Default wenn nichts gesetzt
    "fine_tuning":          "nlp",           # training_type-Wert, auch als task_type möglich
    "causal_lm":            "nlp",           # Rust-Default task_type
    "text_generation":      "nlp",
    "text_classification":  "nlp",
    "seq_classification":   "nlp",           # Rust-Preset "classification_standard"
    "sequence_classification": "nlp",
    "summarization":        "nlp",
    "translation":          "nlp",
    "question_answering":   "nlp",
    "token_classification": "nlp",
    "masked_lm":            "nlp",
    "seq2seq":              "nlp",
    "seq2seq_lm":           "nlp",
    "language_modeling":    "nlp",
    # ── Vision ────────────────────────────────────────────────────────────
    "vision":               "vision",
    "image_classification": "vision",
    "image":                "vision",
    "cv":                   "vision",
    # ── Detection ─────────────────────────────────────────────────────────
    "detection":            "detection",
    "object_detection":     "detection",
    "yolo":                 "detection",
    # ── Audio ─────────────────────────────────────────────────────────────
    "audio":                "audio",
    "speech":               "audio",
    "asr":                  "audio",
    "speech_recognition":   "audio",
    "audio_classification": "audio",
    # ── Tabular ───────────────────────────────────────────────────────────
    "tabular":              "tabular",
    "regression":           "tabular",
    "classification":       "tabular",
    "structured":           "tabular",
}


def detect_plugin_from_model_path(model_path: str) -> str:
    """
    Versucht aus dem Modell-Pfad den Plugin-Typ zu erkennen.
    Wird als letzter Fallback genutzt wenn task_type nicht erkannt wird.
    """
    p = model_path.lower()

    # YOLO: Dateiname enthält "yolo" oder ist eine .yaml/.pt Datei mit YOLO-Inhalt
    if "yolo" in p:
        return "detection"

    # Audio-Modelle
    if any(k in p for k in ["whisper", "wav2vec", "hubert", "speech", "audio"]):
        return "audio"

    # Vision-Modelle (timm-Namenskonvention)
    if any(k in p for k in ["vit_", "resnet", "efficientnet", "convnext", "swin_", "deit_"]):
        return "vision"

    # Standard: NLP (HuggingFace Modell-Hub oder lokaler HF-Checkpoint)
    return "nlp"


def resolve_plugin_name(config: TrainingConfig) -> str:
    """
    Wählt den richtigen Plugin-Namen basierend auf der Config.

    Priorität:
    1. task_type direkt gemappt (aus TASK_TYPE_MAP)
    2. Fallback: Auto-Detection aus model_path
    """
    task = (config.task_type or "auto").lower().strip()

    # Direkter Treffer
    if task in TASK_TYPE_MAP:
        plugin = TASK_TYPE_MAP[task]
        if task not in ("auto",):  # "auto" loggen wir nicht als Treffer
            return plugin

    # Unbekannter task_type → warnen und auto-detect
    if task not in ("auto",):
        MessageProtocol.warning(
            f"Unbekannter task_type '{task}' — Auto-Detection über model_path..."
        )

    return detect_plugin_from_model_path(config.model_path)


# ============================================================================
# PLUGIN-LOADER
# ============================================================================

def load_plugin(plugin_name: str, config: TrainingConfig) -> TrainPlugin:
    """
    Lädt das Plugin dynamisch aus plugins/<plugin_name>/plugin.py.
    Die Plugin-Klasse muss 'Plugin' heißen und TrainPlugin implementieren.
    """
    plugin_file = Path(__file__).parent / "plugins" / plugin_name / "plugin.py"

    if not plugin_file.exists():
        raise FileNotFoundError(
            f"Plugin nicht gefunden: {plugin_file}\n"
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
            f"Plugin '{plugin_name}' konnte nicht geladen werden:\n"
            f"{e}\n\n{traceback.format_exc()}"
        )

    if not hasattr(module, "Plugin"):
        raise AttributeError(
            f"Plugin '{plugin_name}' enthält keine Klasse 'Plugin'.\n"
            f"Jedes Plugin muss 'class Plugin(TrainPlugin)' definieren."
        )

    return module.Plugin(config)


# ============================================================================
# FEHLERBEHANDLUNG
# ============================================================================

def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "memoryerror", "oom",
        "allocation failed", "mps backend out of memory", "not enough memory",
    ])


def handle_exception(exc: Exception) -> None:
    tb = traceback.format_exc()

    if is_oom(exc):
        MessageProtocol.error(
            "RAM-Fehler: Nicht genug Arbeitsspeicher",
            "Empfehlungen:\n"
            "  1. Batch-Size halbieren\n"
            "  2. Sequenzlänge / Bildgröße reduzieren\n"
            "  3. LoRA aktivieren (trainiert nur 1-5% der Parameter)\n"
            "  4. Andere Apps schließen\n"
            "  5. load_in_4bit aktivieren (QLoRA)\n"
            f"\nDetail: {exc}"
        )
        return

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "No module named" in str(exc):
        MessageProtocol.error(
            "Fehlendes Python-Paket",
            f"{exc}\n\nInstalliere fehlende Pakete mit pip.\n\n{tb}"
        )
        return

    if isinstance(exc, FileNotFoundError):
        MessageProtocol.error("Datei/Ordner nicht gefunden", str(exc))
        return

    if isinstance(exc, ValueError):
        MessageProtocol.error(f"Konfigurationsfehler: {exc}", tb)
        return

    # Generischer Fehler — vollständiger Traceback
    MessageProtocol.error(f"{type(exc).__name__}: {exc}", tb)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class Orchestrator:
    """Steuert den gesamten Trainings-Ablauf."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.plugin: Optional[TrainPlugin] = None

        # SIGINT (Ctrl+C) und SIGTERM (stop_training aus Rust) → sauberes Stoppen
        signal.signal(signal.SIGINT,  self._on_stop)
        signal.signal(signal.SIGTERM, self._on_stop)

    def _on_stop(self, *_):
        MessageProtocol.status("stopping", "Training wird gestoppt...")
        if self.plugin:
            self.plugin.stop()

    def run(self):
        cfg = self.config

        try:
            # ── 1. Plugin erkennen ────────────────────────────────────────
            plugin_name = resolve_plugin_name(cfg)
            MessageProtocol.status(
                "init",
                f"Plugin: {plugin_name}  |  "
                f"task_type='{cfg.task_type}'  |  "
                f"model='{Path(cfg.model_path).name}'"
            )

            # ── 2. Plugin laden ───────────────────────────────────────────
            self.plugin = load_plugin(plugin_name, cfg)

            # ── 3. Setup ──────────────────────────────────────────────────
            MessageProtocol.status("init", "Initialisierung...")
            self.plugin.setup()
            if self.plugin.is_stopped:
                return

            # ── 4. Daten laden ────────────────────────────────────────────
            self.plugin.load_data()
            if self.plugin.is_stopped:
                return

            # ── 5. Modell aufbauen ────────────────────────────────────────
            self.plugin.build_model()
            if self.plugin.is_stopped:
                return

            # ── 6. Training ───────────────────────────────────────────────
            MessageProtocol.status("training", "Training gestartet...")
            self.plugin.train()

            if self.plugin.is_stopped:
                MessageProtocol.status("stopped", "Training durch User gestoppt")
                return

            # ── 7. Validierung ────────────────────────────────────────────
            MessageProtocol.status("validating", "Finale Validierung...")
            metrics = self.plugin.validate()

            # Pflichtfelder sicherstellen (Rust-Kompatibilität)
            metrics.setdefault("final_train_loss", 0.0)
            metrics.setdefault("total_epochs", cfg.epochs)
            metrics.setdefault("total_steps", 0)
            metrics.setdefault("best_epoch", 0)
            metrics.setdefault("training_duration_seconds", 0)

            # ── 8. Export ─────────────────────────────────────────────────
            MessageProtocol.status("saving", f"Speichere nach: {cfg.output_path}")
            output_path = self.plugin.export()

            # ── 9. Fertig ─────────────────────────────────────────────────
            # Rust liest model_path und final_metrics aus diesem Message
            MessageProtocol.complete(output_path, metrics)

        except Exception as exc:
            # Traceback immer auch auf stderr für Rust-Logs
            import traceback as _tb
            print(f"[FrameTrain] UNCAUGHT EXCEPTION: {exc}", file=sys.stderr, flush=True)
            print(_tb.format_exc(), file=sys.stderr, flush=True)
            handle_exception(exc)
            # stdout flushen bevor der Prozess endet
            sys.stdout.flush()
            sys.stderr.flush()
            # Mit 0 beenden: wir haben den Fehler bereits sauber per JSON-Protokoll
            # an Rust gemeldet. sys.exit(1) würde den Rust-Fallback-Handler triggern
            # der eine generische "unerwartet beendet" Meldung ausgibt und die
            # eigentliche Fehlermeldung überschreibt.
            sys.exit(0)


# ============================================================================
# EINSTIEGSPUNKT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Training Engine")
    parser.add_argument("--config", required=True, help="Pfad zur config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        MessageProtocol.error(
            "Config nicht gefunden",
            f"Erwartet: {config_path}"
        )
        sys.stdout.flush()
        sys.exit(0)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        MessageProtocol.error("config.json ungültig", str(e))
        sys.stdout.flush()
        sys.exit(0)

    # PyTorch-Check vor allem anderen
    try:
        import torch  # noqa: F401
    except ImportError:
        MessageProtocol.error(
            "PyTorch nicht installiert",
            "Installiere mit: pip install torch\n"
            "Oder mit CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        sys.stdout.flush()
        sys.exit(0)

    config = TrainingConfig.from_dict(config_dict)
    Orchestrator(config).run()


if __name__ == "__main__":
    main()
