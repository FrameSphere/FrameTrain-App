"""
core/protocol.py
================
JSON-Message-Protokoll für die Kommunikation zwischen Python und Rust (stdout).

Alle Nachrichten folgen dem Schema:
  {"type": "<typ>", "timestamp": "<iso>", "data": {...}}

Rust (training_manager.rs → run_training_process) liest stdout Zeile für Zeile
und parst jede Zeile als JSON. Die Typen werden so verarbeitet:

  "progress"   → emit "training-progress"
  "status"     → emit "training-status"
  "checkpoint" → emit "training-checkpoint"
  "complete"   → create model version + emit "training-complete"
  "error"      → emit "training-error"
  "warning"    → emit "training-status" (als Warnung dargestellt)

WICHTIG: stderr geht NUR in Rust-Logs (eprintln!), NICHT ans Frontend.
         Deshalb darf kein Plugin direkt auf stderr schreiben — nur über MessageProtocol.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional


class MessageProtocol:
    """Sendet JSON-Messages über stdout an das Rust-Backend."""

    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]) -> None:
        msg = {
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        # flush=True ist kritisch — Rust liest zeilenweise, ohne flush kommt nichts an
        print(json.dumps(msg, ensure_ascii=False, default=str), flush=True)

    # ── Fortschritt ────────────────────────────────────────────────────────
    # Rust: emit "training-progress" → TrainingProgress struct

    @staticmethod
    def progress(
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        # Fortschritt basierend auf globalem Step / Gesamtzahl Steps
        # step = absoluter Schritt über alle Epochen hinweg
        # total_steps = Gesamtzahl aller Steps (epochs * steps_per_epoch)
        pct = (
            step / max(total_steps, 1)
            * 100
        )
        MessageProtocol._send(
            "progress",
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "learning_rate": float(learning_rate),
                "metrics": metrics or {},
                "progress_percent": round(pct, 2),
            },
        )

    # ── Status ─────────────────────────────────────────────────────────────
    # Rust: emit "training-status"

    @staticmethod
    def status(status: str, message: str = "") -> None:
        MessageProtocol._send("status", {"status": status, "message": message})

    # ── Fehler ─────────────────────────────────────────────────────────────
    # Rust: emit "training-error" → rotes Div im UI

    @staticmethod
    def error(error: str, details: str = "") -> None:
        MessageProtocol._send("error", {"error": error, "details": details})

    # ── Warnung ────────────────────────────────────────────────────────────
    # Rust: emit "training-status" (wird als Warnung angezeigt)

    @staticmethod
    def warning(message: str) -> None:
        MessageProtocol._send("warning", {"message": message})

    # ── Debug (nur sichtbar wenn Rust-Logs aktiv) ──────────────────────────

    @staticmethod
    def debug(message: str, data: Optional[Dict[str, Any]] = None) -> None:
        MessageProtocol._send("debug", {"message": message, "data": data or {}})

    # ── Checkpoint ─────────────────────────────────────────────────────────
    # Rust: emit "training-checkpoint"
    # Spec: {"path": "...", "epoch": 1, "metrics": {}}

    @staticmethod
    def checkpoint(step: int, path: str, epoch: int = 0, metrics: Optional[Dict[str, Any]] = None) -> None:
        MessageProtocol._send("checkpoint", {
            "path": path,
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
        })

    # ── Abgeschlossen ──────────────────────────────────────────────────────
    # Rust: create_new_model_version + save_training_metrics_from_output
    # → final_metrics muss alle Pflichtfelder enthalten:
    #   final_train_loss, final_val_loss, total_epochs, total_steps,
    #   training_duration_seconds, best_epoch

    @staticmethod
    def complete(model_path: str, metrics: Dict[str, Any]) -> None:
        MessageProtocol._send(
            "complete",
            {
                "model_path": model_path,
                "output_path": model_path,
                "final_metrics": metrics,
            },
        )
