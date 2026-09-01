"""
FrameTrain – Test Engine
=========================
Plugin-basierter Orchestrator für Modell-Inferenz.

Aufruf durch Rust:
  python3 test_engine.py --config /path/to/test_config.json

Das aktive Test-Plugin wird über config.task_type ausgewählt.
Verfügbare Plugins werden automatisch aus dem plugins/-Verzeichnis geladen.
"""

import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# stdout/stderr auf UTF-8 zwingen — MUSS vor jedem print laufen.
# Auf Windows ist stdout/stderr per Default cp1252; ein Unicode-Zeichen
# (Emoji, tqdm-Balken) laesst den print sonst mit UnicodeEncodeError sterben.
def _force_utf8_stdio() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_force_utf8_stdio()

from core.config import TestConfig
from core.protocol import TestProtocol


# ─── Plugin-Loader ────────────────────────────────────────────────────────────

def load_plugin(config: TestConfig):
    """
    Dynamischer Plugin-Loader (identisch zu train_engine).
    Scannt plugins/-Verzeichnis und lädt Plugin für config.task_type.
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
            TestProtocol.warning(f"manifest.json in '{plugin_dir.name}' konnte nicht gelesen werden: {e}")
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
            f"test_plugins.{plugin_dir.name}.plugin", plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_name = manifest.get("class", "Plugin")
        if not hasattr(module, class_name):
            raise AttributeError(
                f"Plugin '{plugin_dir.name}' hat keine Klasse '{class_name}'.\n"
                f"Prüfe 'class' in {manifest_path}"
            )

        TestProtocol.status(
            "init",
            f"Test-Plugin geladen: {manifest.get('name', plugin_dir.name)} (task_type='{config.task_type}')"
        )
        return getattr(module, class_name)(config)

    raise ValueError(
        f"Kein Test-Plugin für task_type='{config.task_type}' gefunden.\n"
        f"Verfügbare task_types: {available}"
    )


# ─── Fehlerbehandlung ─────────────────────────────────────────────────────────

def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "oom",
        "allocation failed", "mps backend out of memory",
    ])


def is_torch_ecosystem_conflict(text: str) -> bool:
    """torchvision/torchaudio passen nicht zur installierten torch-Version
    (z. B. 'operator torchvision::nms does not exist' beim transformers-Import)."""
    t = text.lower()
    return (
        "torchvision::nms" in t
        or ("torchvision" in t and "does not exist" in t)
        or ("torchvision" in t and "undefined symbol" in t)
        or ("libtorchaudio" in t and ("symbol not found" in t or "could not load" in t))
        or ("torchaudio" in t and "undefined symbol" in t)
    )


def handle_exception(exc: Exception) -> None:
    tb = traceback.format_exc()

    if is_torch_ecosystem_conflict(tb) or is_torch_ecosystem_conflict(str(exc)):
        TestProtocol.error(
            "PyTorch/torchvision/torchaudio Versionskonflikt",
            "Die installierte torchvision- bzw. torchaudio-Version passt nicht zur torch-Version.\n\n"
            "Behebe mit:\n"
            f"  {sys.executable} -m pip install --upgrade torch torchvision torchaudio\n\n"
            f"Detail: {exc}\n\n{tb}"
        )
        return

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

    if isinstance(exc, ValueError) and (
        "nicht unterstützt" in str(exc) or
        "Kein Test-Plugin" in str(exc)
    ):
        TestProtocol.error("Konfigurationsfehler", str(exc))
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
            TestProtocol.status(
                "init",
                f"Test Engine gestartet | task_type='{self.config.task_type}' | modus='{self.config.mode}'"
            )

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
