"""core/plugin_base.py – Abstrakte Basisklasse für Trainings-Plugins"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import TrainingConfig
from .protocol import MessageProtocol


class TrainPlugin(ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.proto = MessageProtocol
        self.is_stopped = False

    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def load_data(self) -> None: ...
    @abstractmethod
    def build_model(self) -> None: ...
    @abstractmethod
    def train(self) -> None: ...
    @abstractmethod
    def validate(self) -> Dict[str, float]: ...
    @abstractmethod
    def export(self) -> str: ...

    def stop(self) -> None:
        self.is_stopped = True

    def get_info(self) -> Dict[str, Any]:
        return {"plugin": self.__class__.__name__}
