"""core/protocol.py – JSON-Kommunikationsprotokoll für die Test-Engine"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class TestProtocol:

    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]) -> None:
        msg = {"type": msg_type, "timestamp": datetime.now().isoformat(), "data": data}
        print(json.dumps(msg, ensure_ascii=False, default=str), flush=True)

    @staticmethod
    def progress(current: int, total: int, sps: float = 0.0,
                 eta: Optional[float] = None) -> None:
        TestProtocol._send("progress", {
            "current_sample": current,
            "total_samples":  total,
            "progress_percent": round(current / max(total, 1) * 100, 2),
            "samples_per_second": sps,
            "estimated_time_remaining": eta,
        })

    @staticmethod
    def status(status: str, message: str = "") -> None:
        TestProtocol._send("status", {"status": status, "message": message})

    @staticmethod
    def error(error: str, details: str = "") -> None:
        TestProtocol._send("error", {"error": error, "details": details})

    @staticmethod
    def complete_single(predicted: str, confidence: Optional[float],
                        top_predictions: List[Dict[str, Any]],
                        inference_time: float) -> None:
        TestProtocol._send("complete", {
            "mode": "single",
            "predicted_output": predicted,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "inference_time": inference_time,
        })

    @staticmethod
    def complete_dataset(results_file: str, total_samples: int,
                         accuracy: Optional[float], correct: Optional[int],
                         average_loss: Optional[float],
                         average_inference_time: float,
                         samples_per_second: float,
                         hard_examples_file: Optional[str] = None) -> None:
        TestProtocol._send("complete", {
            "mode": "dataset",
            "results_file": results_file,
            "total_samples": total_samples,
            "accuracy": accuracy,
            "correct_predictions": correct,
            "average_loss": average_loss,
            "average_inference_time": average_inference_time,
            "samples_per_second": samples_per_second,
            "hard_examples_file": hard_examples_file,
        })
