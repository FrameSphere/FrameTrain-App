"""core/config.py – TestConfig für die Test-Engine"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TestConfig:
    model_path:         str = ""
    dataset_path:       str = ""
    output_path:        str = ""
    batch_size:         int = 16
    max_samples:        Optional[int] = None
    task_type:          str = "seq_classification"
    mode:               str = "dataset"   # "dataset" | "single"
    single_input:       str = ""
    single_input_type:  str = "text"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestConfig":
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
