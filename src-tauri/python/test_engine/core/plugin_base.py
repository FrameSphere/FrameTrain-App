"""
core/plugin_base.py
===================
Abstrakte Basisklasse für alle FrameTrain Test-Plugins.

Jedes Plugin MUSS alle abstrakten Methoden implementieren.
Die Klasse MUSS 'Plugin' heißen und TestPlugin erweitern.

Fluss Dataset-Modus:
  setup() → load_model() → run_dataset() → [complete via Orchestrator]

Fluss Single-Modus:
  setup() → load_model() → run_single(input) → [complete via Orchestrator]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import TestConfig
from .protocol import MessageProtocol


class TestPlugin(ABC):
    """
    Basis-Interface für alle Test-Plugins.

    Ein Plugin kapselt domänen-spezifische Test-Logik:
    - Modell laden (HuggingFace, YOLO, sklearn, …)
    - Datensatz laden und iterieren
    - Metriken berechnen (Accuracy, WER, mAP, R², …)
    - Hard-Examples identifizieren und speichern
    - Single-Input-Inferenz für das interaktive Interface
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.proto = MessageProtocol   # Shortcut für Plugins
        self.model = None
        self.device = None
        self.is_stopped = False

    # ── Pflicht-Methoden ───────────────────────────────────────────────────

    @abstractmethod
    def setup(self) -> None:
        """Initialisierung: Device erkennen, Ordner anlegen, Seeds setzen."""

    @abstractmethod
    def load_model(self) -> None:
        """
        Modell und ggf. Tokenizer/Processor aus config.model_path laden.
        Ergebnis in self.model speichern.
        """

    @abstractmethod
    def run_dataset(self) -> Dict[str, Any]:
        """
        Vollständigen Testdatensatz durchlaufen.

        Gibt zurück:
        {
          "metrics":            {accuracy, loss, ...},   # Pflicht
          "predictions":        [...],                   # Liste aller Sample-Ergebnisse
          "results_file":       "/.../test_results.json",
          "hard_examples_file": "/.../hard_examples.jsonl" | None,
          "total_samples":      int,
          "task_type":          str,
        }
        """

    @abstractmethod
    def run_single(self, input_data: str) -> Dict[str, Any]:
        """
        Einzelnen Input testen.

        input_data: roher Text ODER absoluter Dateipfad (Bild/Audio)

        Gibt modality-spezifisches Dict zurück:
          NLP:       {"output": str, "confidence": float|None, ...}
          Vision:    {"top_predictions": [...], ...}
          Audio:     {"transcript": str, "confidence": float|None, ...}
          Detection: {"detections": [...], ...}
          Tabular:   {"prediction": Any, "probabilities": [...], ...}
        """

    # ── Optionale Methoden ─────────────────────────────────────────────────

    def stop(self) -> None:
        """Wird vom Orchestrator aufgerufen wenn der User abbricht."""
        self.is_stopped = True

    def get_device(self):
        """Bestes verfügbares Device ermitteln."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        except ImportError:
            return "cpu"

    def get_info(self) -> Dict[str, Any]:
        """Optionale Plugin-Metadaten für Debugging."""
        return {"plugin": self.__class__.__name__, "task_type": self.config.task_type}
