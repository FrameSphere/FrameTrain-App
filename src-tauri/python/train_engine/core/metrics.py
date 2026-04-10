"""
core/metrics.py
===============
Metrics-Kollektion — unabhängig vom Plugin.

Sammelt Trainings-Metriken über Epochen hinweg und schreibt metrics.json
im Format das Rust (training_manager.rs) erwartet — flaches JSON auf Root-Level:

{
  "final_train_loss": 1.23,
  "final_val_loss":   1.45,
  "total_epochs":     3,
  "total_steps":      450,
  "best_epoch":       2,
  "training_duration_seconds": 600
}
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Sammelt Trainings-Metriken über den gesamten Trainingslauf."""

    def __init__(self):
        self._start_time = time.time()
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss: Optional[float] = None
        self.best_epoch: int = 0
        self.total_epochs: int = 0
        self.total_steps: int = 0

    def record(self, epoch: int, step: int, data: Dict[str, Any]) -> None:
        """Einen Messpunkt aufzeichnen."""
        entry = {
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            **data,
        }
        self.history.append(entry)

        # Bestes Val-Loss tracken
        val_loss = data.get("val_loss")
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

        # Fortschritt tracken
        if epoch > self.total_epochs:
            self.total_epochs = epoch
        if step > self.total_steps:
            self.total_steps = step

    def elapsed_seconds(self) -> int:
        """Verstrichene Zeit in Sekunden."""
        return int(time.time() - self._start_time)

    def final_metrics(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Zusammenfassung am Ende des Trainings.
        Gibt alle Pflichtfelder zurück die Rust in metrics.json erwartet.
        """
        duration = self.elapsed_seconds()

        # Letzten Train-Loss aus History
        final_train = next(
            (e["train_loss"] for e in reversed(self.history) if "train_loss" in e), None
        )
        final_val = next(
            (e["val_loss"] for e in reversed(self.history)
             if "val_loss" in e and e["val_loss"] is not None), None
        )

        # Pflichtfelder (Rust liest genau diese)
        result: Dict[str, Any] = {
            "final_train_loss":          final_train,
            "final_val_loss":            final_val,
            "total_epochs":              self.total_epochs,
            "total_steps":               self.total_steps,
            "best_epoch":                self.best_epoch,
            "best_val_loss":             self.best_val_loss,
            "training_duration_seconds": duration,
        }

        # Plugin-spezifische Extra-Felder (überschreiben ggf. obige)
        if extra:
            result.update(extra)

        return result

    def save(self, output_path: str) -> None:
        """
        Speichert metrics.json im Rust-kompatiblen Format.

        Rust (save_training_metrics_from_output) liest direkt:
          metrics["final_train_loss"], metrics["total_epochs"] etc.
        → Felder müssen auf Root-Level liegen, nicht in einem Sub-Objekt.
        """
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        summary = self.final_metrics()

        # Rust-kompatibles Format: Pflichtfelder auf Root-Level
        # + optionale history für Debugging
        data = {
            **summary,
            "history": self.history,
        }

        (out / "metrics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )

    def save_with_overrides(self, output_path: str, overrides: Dict[str, Any]) -> None:
        """
        Wie save(), aber überschreibt einzelne Felder (z.B. total_steps vom HF Trainer).
        Plugins nutzen dies wenn sie genauere Werte als die eigene History haben.
        """
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        summary = self.final_metrics(overrides)
        data = {
            **summary,
            "history": self.history,
        }

        (out / "metrics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )
