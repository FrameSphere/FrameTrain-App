"""
core/metrics.py
===============
Metrics-Kollektion — unabhängig vom Plugin.

Schreibt drei Dateien:
  metrics.json           → Rust-Pflichtformat (flach, Root-Level)
  training_logs.json     → LogEntry-Array (AnalysisPanel chart data)
  training_full_data.json→ Alles: Config, Hardware, Logs, Epoch-Summaries
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Sammelt alle Trainings-Metriken über den gesamten Trainingslauf."""

    def __init__(self):
        self._start_time   = time.time()
        # Step-level logs (für training_logs.json / Charts)
        self.history: List[Dict[str, Any]] = []
        # Epoch-level summaries
        self.epoch_summaries: List[Dict[str, Any]] = []
        self._epoch_start: float = self._start_time
        self._epoch_losses: List[float] = []

        self.best_val_loss: Optional[float] = None
        self.best_epoch: int = 0
        self.total_epochs: int = 0
        self.total_steps: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Recording API
    # ─────────────────────────────────────────────────────────────────────

    def record(self, epoch: int, step: int, data: Dict[str, Any]) -> None:
        """Step-Level Messpunkt aufzeichnen."""
        entry = {
            "epoch":           epoch,
            "step":            step,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            "timestamp":       datetime.now().isoformat(),
            **data,
        }
        self.history.append(entry)

        if "train_loss" in data:
            self._epoch_losses.append(float(data["train_loss"]))

        # Bestes Val-Loss
        val_loss = data.get("val_loss")
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_epoch    = epoch

        if epoch > self.total_epochs: self.total_epochs = epoch
        if step  > self.total_steps:  self.total_steps  = step

    def record_epoch_end(self, epoch: int, val_loss: Optional[float] = None) -> None:
        """Epoche abschließen und Epoch-Summary speichern."""
        now = time.time()
        avg_train = (
            sum(self._epoch_losses) / len(self._epoch_losses)
            if self._epoch_losses else None
        )
        # Minimum / Maximum dieser Epoche
        min_loss = min(self._epoch_losses) if self._epoch_losses else None
        max_loss = max(self._epoch_losses) if self._epoch_losses else None

        self.epoch_summaries.append({
            "epoch":              epoch,
            "avg_train_loss":     round(avg_train, 6) if avg_train is not None else None,
            "min_train_loss":     round(min_loss, 6) if min_loss is not None else None,
            "max_train_loss":     round(max_loss, 6) if max_loss is not None else None,
            "val_loss":           round(float(val_loss), 6) if val_loss is not None else None,
            "duration_seconds":   round(now - self._epoch_start, 1),
            "elapsed_total_sec":  round(now - self._start_time, 1),
            "steps":              len(self._epoch_losses),
        })

        self._epoch_losses = []
        self._epoch_start  = now

    # ─────────────────────────────────────────────────────────────────────

    def elapsed_seconds(self) -> int:
        return int(time.time() - self._start_time)

    def final_metrics(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        duration = self.elapsed_seconds()
        final_train = next(
            (e["train_loss"] for e in reversed(self.history) if "train_loss" in e), None
        )
        final_val = next(
            (e["val_loss"] for e in reversed(self.history)
             if "val_loss" in e and e["val_loss"] is not None), None
        )
        result: Dict[str, Any] = {
            "final_train_loss":          final_train,
            "final_val_loss":            final_val,
            "total_epochs":              self.total_epochs,
            "total_steps":               self.total_steps,
            "best_epoch":                self.best_epoch,
            "best_val_loss":             self.best_val_loss,
            "training_duration_seconds": duration,
        }
        if extra:
            result.update(extra)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Save API
    # ─────────────────────────────────────────────────────────────────────

    def _write_training_logs(self, out: Path) -> None:
        """
        Schreibt training_logs.json im LogEntry-Format (für AnalysisPanel Charts).
        Format: [{"epoch", "step", "train_loss", "val_loss", "learning_rate",
                   "grad_norm", "timestamp"}]
        """
        log_entries = []
        for e in self.history:
            log_entries.append({
                "epoch":         e.get("epoch", 0),
                "step":          e.get("step", 0),
                "train_loss":    e.get("train_loss", 0.0),
                "val_loss":      e.get("val_loss"),
                "learning_rate": e.get("learning_rate", 0.0),
                "grad_norm":     e.get("grad_norm"),
                "elapsed_seconds": e.get("elapsed_seconds"),
                "timestamp":     e.get("timestamp", datetime.now().isoformat()),
            })
        (out / "training_logs.json").write_text(
            json.dumps(log_entries, indent=2, ensure_ascii=False, default=str)
        )

    def save_full_data(
        self,
        output_path: str,
        config_dict: Optional[Dict[str, Any]] = None,
        hardware_info: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Schreibt training_full_data.json mit ALLEM was die KI-Analyse braucht.
        Wird von Plugin.export() aufgerufen.
        """
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        summary = self.final_metrics(overrides)

        full_data: Dict[str, Any] = {
            "exported_at":      datetime.now().isoformat(),
            "training_summary": summary,
            "config":           config_dict or {},
            "hardware":         hardware_info or {},
            "model_info":       model_info or {},
            "dataset_info":     dataset_info or {},
            "epoch_summaries":  self.epoch_summaries,
            "step_logs":        self.history,   # vollständige Step-History
        }

        # Abgeleitete Statistiken für schnelleren KI-Zugriff
        if self.history:
            losses = [e["train_loss"] for e in self.history if "train_loss" in e]
            val_losses = [e["val_loss"] for e in self.history
                          if "val_loss" in e and e["val_loss"] is not None]
            lrs = [e["learning_rate"] for e in self.history
                   if "learning_rate" in e and e["learning_rate"] > 0]
            grad_norms = [e["grad_norm"] for e in self.history
                          if e.get("grad_norm") is not None]
            full_data["derived_stats"] = {
                "loss_reduction_pct":    round(((losses[0] - losses[-1]) / max(losses[0], 1e-9)) * 100, 2) if len(losses) > 1 else 0,
                "initial_train_loss":    round(losses[0], 6)   if losses else None,
                "final_train_loss":      round(losses[-1], 6)  if losses else None,
                "min_train_loss":        round(min(losses), 6) if losses else None,
                "max_train_loss":        round(max(losses), 6) if losses else None,
                "initial_val_loss":      round(val_losses[0], 6)  if val_losses else None,
                "final_val_loss":        round(val_losses[-1], 6) if val_losses else None,
                "overfitting_gap_pct":   round(((val_losses[-1] - losses[-1]) / max(losses[-1], 1e-9)) * 100, 2) if val_losses and losses else None,
                "initial_lr":            lrs[0]  if lrs else None,
                "final_lr":              lrs[-1] if lrs else None,
                "avg_grad_norm":         round(sum(grad_norms) / len(grad_norms), 4) if grad_norms else None,
                "max_grad_norm":         round(max(grad_norms), 4) if grad_norms else None,
                "total_log_entries":     len(self.history),
            }

        (out / "training_full_data.json").write_text(
            json.dumps(full_data, indent=2, ensure_ascii=False, default=str)
        )

    def save(self, output_path: str) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.final_metrics()
        data = {**summary, "history": self.history}
        (out / "metrics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )
        self._write_training_logs(out)

    def save_with_overrides(self, output_path: str, overrides: Dict[str, Any]) -> None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.final_metrics(overrides)
        data = {**summary, "history": self.history}
        (out / "metrics.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )
        self._write_training_logs(out)
