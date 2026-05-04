"""
core/config.py – TrainingConfig für Sequenzklassifikation
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingConfig:
    # Pfade
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""
    checkpoint_dir: str = ""

    # Training-Grundlagen
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1

    # Lernrate
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.1

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    scheduler: str = "linear"

    # Regularisierung
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # Mixed Precision
    fp16: bool = False
    bf16: bool = False

    # Sequenzklassifikation spezifisch
    max_seq_length: int = 128

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = False
    dataloader_drop_last: bool = False
    group_by_length: bool = False

    # Evaluation & Speichern
    eval_steps: int = 500
    eval_strategy: str = "epoch"
    save_steps: int = 500
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10

    # Sonstiges
    seed: int = 42
    gradient_checkpointing: bool = False
    training_type: str = "fine_tuning"
    task_type: str = "seq_classification"

    # Extra-Felder (werden ignoriert, aber nicht verworfen)
    extra: Dict[str, Any] = field(default_factory=dict)

    # Felder aus desktop-app2 die vorkommen können (werden ignoriert)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=list)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    sgd_momentum: float = 0.9
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    cosine_min_lr: float = 0.0
    num_workers: int = 0  # noqa: F811

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        cfg = cls(**filtered)
        cfg.extra = extra
        return cfg

    def effective_output_dir(self) -> str:
        return self.checkpoint_dir or self.output_path
