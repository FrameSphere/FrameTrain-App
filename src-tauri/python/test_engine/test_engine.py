"""
FrameTrain – Test Engine
=========================
Orchestrator für Sequenzklassifikations-Inferenz.

Aufruf durch Rust:
  python3 test_engine.py --config /path/to/test_config.json

Unterstützte Modi:
  mode=dataset  → Batch-Inferenz auf einem Dataset
  mode=single   → Einzelner Text-Input
"""

import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.config import TestConfig
from core.protocol import TestProtocol


# ─── Plugin-Loader ────────────────────────────────────────────────────────────

def load_plugin(config: TestConfig):
    plugin_file = Path(__file__).parent / "plugins" / "seq_classification" / "plugin.py"
    if not plugin_file.exists():
        raise FileNotFoundError(f"Test-Plugin nicht gefunden: {plugin_file}")

    import importlib.util
    spec   = importlib.util.spec_from_file_location("test_plugins.seq_classification.plugin", plugin_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Plugin"):
        raise AttributeError("Plugin-Datei enthält keine Klasse 'Plugin'")

    return module.Plugin(config)


# ─── Fehlerbehandlung ─────────────────────────────────────────────────────────

def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "oom",
        "allocation failed", "mps backend out of memory",
    ])


def handle_exception(exc: Exception) -> None:
    tb = traceback.format_exc()

    if is_oom(exc):
        TestProtocol.error(
            "RAM-Fehler beim Test",
            "Empfehlungen:\n"
            "  1. Batch-Size verkleinern\n"
            "  2. max_samples setzen\n"
            "  3. Andere Apps schließen\n"
            f"\nDetail: {exc}"
        )
        return

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        TestProtocol.error(
            "Fehlendes Python-Paket",
            f"{exc}\n\nInstalliere mit: pip install transformers torch scikit-learn\n\n{tb}"
        )
        return

    if isinstance(exc, FileNotFoundError):
        TestProtocol.error("Datei nicht gefunden", str(exc))
        return

    if isinstance(exc, ValueError) and "nicht unterstützt" in str(exc):
        TestProtocol.error("Modell nicht unterstützt", str(exc))
        return

    TestProtocol.error(f"{type(exc).__name__}: {exc}", tb)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class TestOrchestrator:

    def __init__(self, config: TestConfig):
        self.config = config
        self.plugin = None
        signal.signal(signal.SIGINT,  self._on_stop)
        signal.signal(signal.SIGTERM, self._on_stop)

    def _on_stop(self, *_):
        TestProtocol.status("stopping", "Test wird gestoppt...")
        if self.plugin:
            self.plugin.stop()

    def run(self):
        try:
            TestProtocol.status("init", f"Test-Engine gestartet | Modus: {self.config.mode}")

            self.plugin = load_plugin(self.config)
            self.plugin.setup()

            if self.plugin.is_stopped:
                return

            if self.config.mode == "single":
                self.plugin.run_single()
            else:
                self.plugin.run_dataset()

        except Exception as exc:
            print(f"[TestEngine] EXCEPTION: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            handle_exception(exc)
            sys.stdout.flush()
            sys.exit(0)


# ─── Einstiegspunkt ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Test Engine")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        TestProtocol.error("Config nicht gefunden", f"Erwartet: {config_path}")
        sys.exit(0)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        TestProtocol.error("test_config.json ungültig", str(e))
        sys.exit(0)

    try:
        import torch  # noqa
    except ImportError:
        TestProtocol.error(
            "PyTorch nicht installiert",
            "Installiere mit: pip install torch transformers"
        )
        sys.exit(0)

    config = TestConfig.from_dict(config_dict)
    TestOrchestrator(config).run()


if __name__ == "__main__":
    main()
