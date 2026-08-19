"""
core/hf_training.py – Gemeinsame Bausteine für HuggingFace-Trainer-Plugins.

Hier liegt die Logik, die sich sonst in jedem Plugin wiederholen würde:
TrainingArguments aufbauen, das Gerät bestimmen, Klassifikationsmetriken
rechnen und die Abschluss-Metriken so benennen, wie die Analyse-Seite sie
erwartet. Ein Fix an dieser Stelle wirkt für alle Plugins.
"""
from typing import Any, Dict, List, Optional, Sequence

from .protocol import MessageProtocol


def device_name() -> str:
    """Das Gerät, auf dem tatsächlich gerechnet wird."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def optimizer_name(config_optimizer: str) -> str:
    """Bildet den UI-Namen auf den HuggingFace-Bezeichner ab."""
    return {
        "adamw": "adamw_torch",
        "adam": "adamw_torch",   # HF kennt kein reines Adam
        "sgd": "sgd",
        "adafactor": "adafactor",
    }.get(str(config_optimizer).lower(), "adamw_torch")


def build_training_arguments(config, output_dir: str, TrainingArguments, **overrides):
    """
    Baut TrainingArguments aus der FrameTrain-Config.

    Unbekannte Argumente werden verworfen statt das Training mit einem
    TypeError abzubrechen: die Felder von TrainingArguments ändern sich
    zwischen transformers-Versionen (5.x hat z.B. group_by_length entfernt).
    """
    import inspect

    warmup_steps = config.warmup_steps if config.warmup_ratio == 0 else 0
    warmup_ratio = config.warmup_ratio if warmup_steps == 0 else 0.0
    max_steps = int(config.max_steps) if int(config.max_steps) > 0 else -1
    eval_steps = (
        max(int(config.eval_steps), 1)
        if str(config.eval_strategy).lower() == "steps"
        else None
    )

    import torch
    use_fp16 = config.fp16 and torch.cuda.is_available()
    use_bf16 = config.bf16 and (
        torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    )

    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        optim=optimizer_name(config.optimizer),
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        lr_scheduler_type=config.scheduler,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy=config.eval_strategy,
        save_strategy="steps",
        save_steps=max(config.save_steps, 50),
        save_total_limit=2,
        logging_steps=config.logging_steps,
        seed=config.seed,
        dataloader_num_workers=0,    # MPS-sicher
        dataloader_pin_memory=False,  # MPS-sicher
        dataloader_drop_last=config.dataloader_drop_last,
        report_to=[],
        disable_tqdm=True,
        load_best_model_at_end=False,
    )
    if eval_steps is not None:
        kwargs["eval_steps"] = eval_steps
    kwargs.update(overrides)

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in supported)
    if dropped:
        import transformers
        MessageProtocol.status(
            "training",
            "Hinweis: diese Einstellungen kennt die installierte "
            f"transformers-Version ({transformers.__version__}) nicht und "
            f"werden ignoriert: {', '.join(dropped)}",
        )
        for k in dropped:
            kwargs.pop(k, None)

    return TrainingArguments(**kwargs)


def classification_scores(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    """Accuracy/F1/Precision/Recall — die Analyse-Seite zeigt genau diese vier."""
    if labels is None or preds is None or len(labels) == 0 or len(labels) != len(preds):
        return {}
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except ImportError:
        return {}
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def cap_eval_dataset(eval_ds, max_eval_samples: int, seed: int):
    """
    Deckelt den Eval-Split.

    Zufällig ziehen, nicht die ersten N: viele Splits sind nach Label sortiert
    (imdb: erst 12.500x Klasse 0). Ein Präfix enthielte nur eine Klasse und die
    Auswertung meldete Accuracy 0.0.
    """
    n = int(max_eval_samples or 0)
    if n <= 0 or eval_ds is None or len(eval_ds) <= n:
        return eval_ds
    MessageProtocol.status(
        "loading_data",
        f"Eval-Split auf {n} von {len(eval_ds)} Beispielen begrenzt (Max Eval Samples).",
    )
    return eval_ds.shuffle(seed=seed).select(range(n))


def progress_callback(TrainerCallback, plugin, total_steps_fallback: int):
    """
    Meldet Fortschritt und Evaluierungs-Phasen ans Frontend.

    Die Meldung zur Evaluierung ist wichtig: eine Auswertung über einen großen
    Split dauert länger als das Training selbst, und ohne Rückmeldung stand der
    Fortschritt minutenlang still und sah aus wie ein Absturz.
    """

    class _Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_evaluate and getattr(plugin, "eval_dataset", None) is not None:
                MessageProtocol.status(
                    "evaluating",
                    f"Evaluierung laeuft ueber {len(plugin.eval_dataset)} Beispiele "
                    "- das kann bei grossen Datasets dauern.",
                )

        def on_evaluate(self, args, state, control, **kwargs):
            if not plugin.is_stopped:
                MessageProtocol.status("training", "Evaluierung abgeschlossen, Training laeuft weiter.")

        def on_epoch_end(self, args, state, control, **kwargs):
            if plugin.is_stopped:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if plugin.is_stopped:
                control.should_training_stop = True
                return
            total = getattr(state, "max_steps", 0) or total_steps_fallback
            is_eval = "eval_loss" in logs
            if is_eval:
                extra = {k: v for k, v in logs.items()
                         if k not in ("eval_loss", "epoch", "eval_runtime",
                                      "eval_samples_per_second", "eval_steps_per_second")
                         and isinstance(v, (int, float))}
                MessageProtocol.progress(
                    epoch=int(state.epoch or 0), total_epochs=plugin.config.epochs,
                    step=state.global_step, total_steps=total,
                    train_loss=getattr(plugin, "_last_train_loss", None),
                    val_loss=logs.get("eval_loss"),
                    learning_rate=getattr(plugin, "_last_lr", plugin.config.learning_rate),
                    metrics=extra,
                )
            else:
                t_loss = logs.get("loss", logs.get("train_loss", 0.0))
                lr = logs.get("learning_rate", plugin.config.learning_rate)
                plugin._last_train_loss = t_loss
                plugin._last_lr = lr
                MessageProtocol.progress(
                    epoch=int(state.epoch or 0), total_epochs=plugin.config.epochs,
                    step=state.global_step, total_steps=total,
                    train_loss=t_loss, val_loss=None, learning_rate=lr, metrics={},
                )

    return _Progress()


def final_metrics(plugin, trainer, eval_result: Dict[str, Any], start_time: float,
                  architecture: str, num_labels: int) -> Dict[str, Any]:
    """Abschluss-Metriken in genau den Namen, die die Analyse-Seite liest."""
    import time
    epochs_done = getattr(trainer.state, "epoch", None)
    total_epochs = (max(1, round(float(epochs_done)))
                    if epochs_done is not None else plugin.config.epochs)
    return {
        "final_train_loss": float(getattr(plugin, "_last_train_loss", 0.0) or 0.0),
        "final_val_loss": float(eval_result.get("eval_loss", 0.0) or 0.0),
        "accuracy": float(eval_result.get("eval_accuracy", 0.0) or 0.0),
        "f1": float(eval_result.get("eval_f1", 0.0) or 0.0),
        "precision": float(eval_result.get("eval_precision", 0.0) or 0.0),
        "recall": float(eval_result.get("eval_recall", 0.0) or 0.0),
        "total_epochs": int(total_epochs),
        "total_steps": int(getattr(trainer.state, "global_step", 0)),
        "best_epoch": 0,
        "training_duration_seconds": int(time.time() - start_time),
        "architecture": architecture or "unbekannt",
        "num_labels": int(num_labels),
        "n_train": int(len(plugin.train_dataset)) if getattr(plugin, "train_dataset", None) is not None else 0,
        "n_val": int(len(plugin.eval_dataset)) if getattr(plugin, "eval_dataset", None) is not None else 0,
        "device": getattr(plugin, "device_used", "cpu"),
    }
