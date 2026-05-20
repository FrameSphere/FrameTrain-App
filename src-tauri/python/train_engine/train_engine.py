"""
FrameTrain - Train Engine
=========================
Plugin-basierter Orchestrator für KI-Modell-Training.

Wird von Rust aufgerufen:
  python3 train_engine.py --config /path/to/config.json

Verfügbare Plugins werden automatisch aus dem plugins/-Verzeichnis geladen.
Jedes Plugin-Verzeichnis braucht eine manifest.json mit task_type.
Das aktive Plugin wird über config.task_type ausgewählt.
"""

import argparse
import json
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.config import TrainingConfig
from core.protocol import MessageProtocol


class LoggingTee:
    """Schreibt stdout gleichzeitig auf die Konsole und in eine Log-Datei."""
    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file   = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def fileno(self):  return self._stream.fileno()
    def isatty(self):  return False
    def __getattr__(self, name): return getattr(self._stream, name)


# ============ Plugin-Loader ============

def load_plugin(config: TrainingConfig):
    """
    Dynamischer Plugin-Loader.
    Scannt plugins/-Verzeichnis, liest manifest.json jedes Plugins,
    und lädt das Plugin dessen task_type mit config.task_type übereinstimmt.
    """
    import importlib.util

    plugins_dir = Path(__file__).parent / "plugins"

    if not plugins_dir.exists():
        raise FileNotFoundError(f"plugins/-Verzeichnis nicht gefunden: {plugins_dir}")

    available = []

    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir():
            continue

        manifest_path = plugin_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            MessageProtocol.warning(f"manifest.json in '{plugin_dir.name}' konnte nicht gelesen werden: {e}")
            continue

        plugin_task = manifest.get("task_type", "")
        available.append(plugin_task)

        if plugin_task != config.task_type:
            continue

        # Passendes Plugin gefunden
        entry = manifest.get("entry", "plugin.py")
        plugin_file = plugin_dir / entry

        if not plugin_file.exists():
            raise FileNotFoundError(
                f"Plugin-Datei nicht gefunden: {plugin_file}\n"
                f"Prüfe 'entry' in {manifest_path}"
            )

        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_dir.name}.plugin", plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_name = manifest.get("class", "Plugin")
        if not hasattr(module, class_name):
            raise AttributeError(
                f"Plugin '{plugin_dir.name}' hat keine Klasse '{class_name}'.\n"
                f"Prüfe 'class' in {manifest_path}"
            )

        MessageProtocol.status(
            "init",
            f"Plugin geladen: {manifest.get('name', plugin_dir.name)} (task_type='{config.task_type}')"
        )
        return getattr(module, class_name)(config)

    raise ValueError(
        f"Kein Plugin für task_type='{config.task_type}' gefunden.\n"
        f"Verfügbare task_types: {available}\n"
        f"Prüfe config.task_type oder lege ein neues Plugin in plugins/<name>/manifest.json an."
    )


# ============ Fehlerbehandlung ============

def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "oom",
        "allocation failed", "mps backend out of memory",
    ])


def handle_exception(exc: Exception) -> None:
    tb = traceback.format_exc()

    if is_oom(exc):
        MessageProtocol.error(
            "RAM-Fehler: Nicht genug Arbeitsspeicher",
            "Empfehlungen:\n"
            "  1. Batch-Size halbieren (z.B. 8 → 4 → 2)\n"
            "  2. max_seq_length reduzieren (z.B. 128 → 64)\n"
            "  3. Gradient Checkpointing aktivieren\n"
            "  4. Andere Apps schließen\n"
            f"\nDetail: {exc}"
        )
        return

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        MessageProtocol.error(
            "Fehlendes Python-Paket",
            f"{exc}\n\nInstalliere mit:\n  pip install transformers datasets torch scikit-learn\n\n{tb}"
        )
        return

    if isinstance(exc, FileNotFoundError):
        MessageProtocol.error("Datei/Ordner nicht gefunden", str(exc))
        return

    if isinstance(exc, ValueError) and (
        "wird noch nicht unterstützt" in str(exc) or
        "Kein Plugin für" in str(exc)
    ):
        MessageProtocol.error("Konfigurationsfehler", str(exc))
        return

    MessageProtocol.error(f"{type(exc).__name__}: {exc}", tb)


# ============ Orchestrator ============

class _TeeLogger:
    """Schreibt stdout gleichzeitig in Datei und originalen stdout."""
    def __init__(self, filepath: Path, original):
        self._file = open(filepath, "a", encoding="utf-8", buffering=1)
        self._orig = original
    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
    def flush(self):
        self._orig.flush()
        self._file.flush()
    def close(self):
        self._file.close()
    def fileno(self):
        return self._orig.fileno()
    def __getattr__(self, name):
        return getattr(self._orig, name)


class Orchestrator:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.plugin = None
        self._tee   = None
        signal.signal(signal.SIGINT,  self._on_stop)
        signal.signal(signal.SIGTERM, self._on_stop)

    def _start_logging(self, output_dir: Path) -> None:
        log_dir = output_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Training gestartet: {datetime.now().isoformat()}\n")
            f.write(f"task_type: {self.config.task_type}\n")
            f.write(f"{'='*60}\n")
        self._tee = _TeeLogger(log_path, sys.stdout)
        sys.stdout = self._tee

    def _stop_logging(self) -> None:
        if self._tee:
            sys.stdout = self._tee._orig
            self._tee.close()
            self._tee = None

    def _on_stop(self, *_):
        MessageProtocol.status("stopping", "Training wird gestoppt...")
        if self.plugin:
            self.plugin.stop()

    def run(self):
        try:
            MessageProtocol.status(
                "init",
                f"Train Engine gestartet | task_type='{self.config.task_type}' | model='{Path(self.config.model_path).name}'"
            )

            self.plugin = load_plugin(self.config)

            log_dir = Path(self.config.effective_output_dir())
            self._start_logging(log_dir)

            MessageProtocol.status("init", "Setup...")
            self.plugin.setup()
            if self.plugin.is_stopped: return

            self.plugin.load_data()
            if self.plugin.is_stopped: return

            self.plugin.build_model()
            if self.plugin.is_stopped: return

            MessageProtocol.status("training", "Training gestartet...")
            self.plugin.train()
            if self.plugin.is_stopped:
                MessageProtocol.status("stopped", "Training gestoppt")
                return

            MessageProtocol.status("validating", "Finale Validierung...")
            metrics = self.plugin.validate()
            metrics.setdefault("final_train_loss", 0.0)
            metrics.setdefault("total_epochs", self.config.epochs)
            metrics.setdefault("total_steps", 0)
            metrics.setdefault("best_epoch", 0)
            metrics.setdefault("training_duration_seconds", 0)

            MessageProtocol.status("saving", f"Speichere nach: {self.config.output_path}")
            output_path = self.plugin.export()

            MessageProtocol.complete(output_path, metrics)

        except Exception as exc:
            print(f"[FrameTrain] EXCEPTION: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            handle_exception(exc)
            sys.stdout.flush()
            sys.exit(0)
        finally:
            self._stop_logging()


# ============ Einstiegspunkt ============

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Train Engine")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        MessageProtocol.error("Config nicht gefunden", f"Erwartet: {config_path}")
        sys.exit(0)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        MessageProtocol.error("config.json ungültig", str(e))
        sys.exit(0)

    try:
        import torch  # noqa
    except ImportError:
        MessageProtocol.error(
            "PyTorch nicht installiert",
            "Installiere mit: pip install torch transformers datasets scikit-learn"
        )
        sys.exit(0)

    config = TrainingConfig.from_dict(config_dict)

    log_dir = Path(config.effective_output_dir())
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"training_{ts}.log"
    tee = LoggingTee(sys.stdout, log_path)
    sys.stdout = tee

    Orchestrator(config).run()

    sys.stdout = tee._stream
    tee.close()


if __name__ == "__main__":
    main()
