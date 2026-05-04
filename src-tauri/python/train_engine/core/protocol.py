"""core/protocol.py – JSON-Kommunikationsprotokoll (identisch zu desktop-app2)"""
import json
from datetime import datetime
from typing import Any, Dict, Optional


class MessageProtocol:

    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]) -> None:
        msg = {"type": msg_type, "timestamp": datetime.now().isoformat(), "data": data}
        print(json.dumps(msg, ensure_ascii=False, default=str), flush=True)

    @staticmethod
    def progress(epoch: int, total_epochs: int, step: int, total_steps: int,
                 train_loss: float, val_loss: Optional[float] = None,
                 learning_rate: float = 0.0,
                 metrics: Optional[Dict[str, float]] = None) -> None:
        pct = step / max(total_steps, 1) * 100
        MessageProtocol._send("progress", {
            "epoch": epoch, "total_epochs": total_epochs,
            "step": step, "total_steps": total_steps,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "learning_rate": float(learning_rate),
            "metrics": metrics or {},
            "progress_percent": round(pct, 2),
        })

    @staticmethod
    def status(status: str, message: str = "") -> None:
        MessageProtocol._send("status", {"status": status, "message": message})

    @staticmethod
    def error(error: str, details: str = "") -> None:
        MessageProtocol._send("error", {"error": error, "details": details})

    @staticmethod
    def warning(message: str) -> None:
        MessageProtocol._send("warning", {"message": message})

    @staticmethod
    def checkpoint(step: int, path: str, epoch: int = 0,
                   metrics: Optional[Dict[str, Any]] = None) -> None:
        MessageProtocol._send("checkpoint", {"path": path, "step": step, "epoch": epoch, "metrics": metrics or {}})

    @staticmethod
    def complete(model_path: str, metrics: Dict[str, Any]) -> None:
        MessageProtocol._send("complete", {
            "model_path": model_path,
            "output_path": model_path,
            "final_metrics": metrics,
        })
