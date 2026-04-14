"""
core/config.py
==============
Test-Konfiguration für alle Plugins.

Modi:
  dataset  – Ganzen Datensatz durchlaufen, Metriken + Hard-Examples ausgeben
  single   – Einzelnen Input testen (Text, Bildpfad, Audiopfad)
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class TestConfig:
    # ── Pfade ─────────────────────────────────────────────────────────────
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""

    # ── Test-Einstellungen ─────────────────────────────────────────────────
    batch_size: int = 8
    max_samples: Optional[int] = None

    # ── Modell-Typ ─────────────────────────────────────────────────────────
    # Gültige Werte: "auto" | "nlp" | "vision" | "audio" | "detection" | "tabular"
    # oder spezifische task_types wie "causal_lm", "image_classification", "asr", …
    task_type: str = "auto"

    # ── Modus ──────────────────────────────────────────────────────────────
    mode: str = "dataset"           # "dataset" | "single"
    single_input: str = ""          # Text ODER absoluter Dateipfad
    single_input_type: str = "text" # "text" | "image_path" | "audio_path" | "json"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
