"""
core/plugin_base.py
===================
Abstrakte Basisklasse (ABC) für alle FrameTrain-Plugins.

JEDES Plugin MUSS alle Methoden implementieren.
Die Engine kennt das Plugin nur über dieses Interface — nie direkt.

Fluss:
  setup() → load_data() → build_model() → train() → validate() → export()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .config import TrainingConfig
from .protocol import MessageProtocol


class TrainPlugin(ABC):
    """
    Basis-Interface für alle Trainings-Plugins.

    Ein Plugin kapselt die gesamte domänen-spezifische Logik:
    - Wie Daten geladen werden (Vision, NLP, Audio, ...)
    - Wie das Modell aufgebaut / geladen wird
    - Wie das Training abläuft
    - Wie Metriken berechnet werden
    - Wie das Modell exportiert wird

    Die Engine (train_engine.py) ruft diese Methoden in der richtigen Reihenfolge
    auf und übernimmt Logging, Fehlerbehandlung und Protokoll-Kommunikation.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.proto = MessageProtocol   # Shortcut für Plugins
        self.is_stopped = False

    # ── Pflicht-Methoden ───────────────────────────────────────────────────

    @abstractmethod
    def setup(self) -> None:
        """
        Initialisierung vor dem Training.
        Z.B.: Device erkennen, Seeds setzen, Verzeichnisse anlegen.
        """

    @abstractmethod
    def load_data(self) -> None:
        """
        Datensatz laden und vorbereiten.
        Ergebnis wird intern im Plugin gespeichert (self.train_data etc.)
        """

    @abstractmethod
    def build_model(self) -> None:
        """
        Modell laden oder bauen.
        Z.B.: AutoModelForCausalLM.from_pretrained(), YOLO(), timm.create_model()
        """

    @abstractmethod
    def train(self) -> None:
        """
        Haupt-Training-Loop.
        Plugin ruft selbst MessageProtocol.progress() auf.
        """

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """
        Finale Validierung nach dem Training.
        Gibt Metrics-Dict zurück: {"val_loss": 0.3, "accuracy": 0.85, ...}
        """

    @abstractmethod
    def export(self) -> str:
        """
        Modell exportieren / speichern.
        Gibt den finalen output_path zurück.
        """

    # ── Optionale Methoden (können überschrieben werden) ──────────────────

    def stop(self) -> None:
        """Wird vom Orchestrator aufgerufen wenn der User abbricht."""
        self.is_stopped = True

    def get_info(self) -> Dict[str, Any]:
        """Optionale Plugin-Metadaten für Debugging."""
        return {"plugin": self.__class__.__name__}
