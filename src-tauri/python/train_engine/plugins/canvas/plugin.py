"""
Canvas Model Training Plugin — Runtime Graph IR

Trainiert Modelle aus config.canvas_graph (JSON IR):
  parse_ir → validate_ir_shapes → build_model_from_graph → train loop
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core.config import TrainingConfig
from core.protocol import MessageProtocol

# Sibling imports (train_engine loads this file via importlib, not as package)
_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)

from dataloaders import get_dataloaders  # noqa: E402
from ir import CanvasGraphIR, IRTrainingSpec, is_non_empty_ir, parse_ir  # noqa: E402
from model_builder import build_model_from_graph  # noqa: E402
from shape_propagate import ShapeValidationError  # noqa: E402


class CanvasPlugin:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Fix 1.2: MPS-Support für Apple Silicon
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )
        self.model: Optional[nn.Module] = None
        self.model_class = None
        self.ir = None
        self.is_stopped = False
        self._optimizer: Optional[optim.Optimizer] = None
        self._scheduler = None
        self._prev_optimizer_state = None
        self.train_history: Dict[str, List] = {
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
        }

    # ── K2: Legacy-Erkennung ─────────────────────────────────────────────────
    _LEGACY_ERROR = (
        "Dieses Modell stammt aus einer älteren FrameTrain-Version und enthält "
        "keine graphIR-Daten (canvas_graph fehlt oder ist leer).\n"
        "Training und Resume-Training werden für Legacy-Modelle nicht unterstützt.\n"
        "Lösung: Modell im Synapse-Builder öffnen, neu anordnen/speichern und "
        "erneut exportieren — danach steht der vollständige IR zur Verfügung."
    )


    def stop(self) -> None:
        """Abbruch aus der Oberflaeche.

        Diese Klasse erbt nicht von TrainPlugin, wo stop() definiert ist.
        Ohne die Methode lief der Signal-Handler der Engine in einen
        AttributeError: "Stoppen" blieb wirkungslos und das Training lief
        bis zur letzten Epoche weiter, obwohl is_stopped ueberall geprueft wird.
        """
        self.is_stopped = True

    def setup(self) -> bool:
        try:
            raw_graph = getattr(self.config, "canvas_graph", None) or {}
            if is_non_empty_ir(raw_graph):
                return self._setup_from_ir(raw_graph)

            # K2: Legacy-Pfad blockieren — kein Training ohne IR
            canvas_code = getattr(self.config, "canvas_model_code", None) or ""
            if canvas_code:
                MessageProtocol.error("Legacy-Modell — Training nicht möglich", self._LEGACY_ERROR)
                return False

            MessageProtocol.error(
                "Canvas Setup",
                "Weder canvas_graph (IR) noch canvas_model_code gesetzt.",
            )
            return False
        except ShapeValidationError as e:
            MessageProtocol.error("Shape Validation", str(e))
            try:
                MessageProtocol.status(
                    "train",
                    f"[DIAGNOSTIC_JSON]\n{json.dumps({'error_type': 'shape_mismatch', 'raw_error': str(e)})}\n[/DIAGNOSTIC_JSON]",
                )
            except Exception:
                pass
            return False
        except Exception as e:
            MessageProtocol.error("Canvas Setup", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            return False

    def _setup_from_ir(self, raw_graph: Any) -> bool:
        self.ir = parse_ir(raw_graph)
        MessageProtocol.status("setup", f"[Canvas IR] {len(self.ir.nodes)} nodes, {len(self.ir.edges)} edges")
        self.model = build_model_from_graph(self.ir).to(self.device)
        self.model_class = type(self.model)
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        MessageProtocol.status(
            "setup",
            f"✓ Runtime-Modell aus Graph-IR\n"
            f"  Parameters: {total:,} (trainable: {trainable:,})\n"
            f"  Device: {self.device}",
        )
        self._load_prev_checkpoint()
        return True

    def _load_prev_checkpoint(self) -> None:
        """Lädt vorherige Gewichte + Optimizer-State für echten Resume."""
        try:
            pc = getattr(self.config, "plugin_config", None) or {}
            if isinstance(pc, str):
                import json as _json
                pc = _json.loads(pc)
            prev = pc.get("prev_checkpoint", "") if isinstance(pc, dict) else ""
            if prev and Path(prev).exists():
                # weights_only=False nötig um optimizer_state_dict zu laden
                checkpoint = torch.load(prev, map_location=self.device, weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state)
                # Fix 1.3: Optimizer-State für echten Resume merken
                self._prev_optimizer_state = checkpoint.get("optimizer_state_dict", None)
                # W1 Edit 3d: Scheduler-State für echten Resume merken
                self._prev_scheduler_state = checkpoint.get("scheduler_state_dict", None)
                MessageProtocol.status("setup", "✓ Vorheriger Checkpoint geladen — iteratives Training aktiv")
            else:
                self._prev_optimizer_state = None
                self._prev_scheduler_state = None
                MessageProtocol.status("setup", "[Info] Kein vorheriger Checkpoint — Training startet von Null")
        except Exception as e:
            self._prev_optimizer_state = None
            MessageProtocol.status("setup", f"[Warn] Checkpoint konnte nicht geladen werden: {e} — starte von Null")

    def _setup_from_code(self, canvas_code: str) -> bool:
        # K2: Diese Methode ist für Training deaktiviert.
        # Wird nur noch als Fallback-Guard erreicht wenn setup() sie direkt aufruft —
        # was seit K2 nicht mehr passiert. Zweite Verteidigungslinie.
        MessageProtocol.error("Legacy-Modell — Training nicht möglich", self._LEGACY_ERROR)
        return False

    def load_data(self) -> None:
        pass

    def build_model(self) -> None:
        if self.model is None:
            MessageProtocol.error("Model nicht verfügbar", "setup() zuerst aufrufen")

    def _make_optimizer(self) -> optim.Optimizer:
        spec = self.ir.training if self.ir else None
        lr = spec.learning_rate if spec else self.config.learning_rate
        wd = spec.weight_decay if spec else self.config.weight_decay
        name = (spec.optimizer if spec else self.config.optimizer).lower()
        params = self.model.parameters()
        if name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        if name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "rmsprop":
            return optim.RMSprop(params, lr=lr, weight_decay=wd)
        return optim.AdamW(params, lr=lr, weight_decay=wd)

    def _make_loss(self) -> nn.Module:
        spec = self.ir.training if self.ir else None
        loss_name = spec.loss if spec else "cross_entropy"
        # reduction="none" liefert keinen Skalar → backward() bricht. Auf mean zurückfallen.
        reduction = str(getattr(spec, "loss_reduction", "mean") or "mean") if spec else "mean"
        if reduction not in ("mean", "sum"):
            if reduction == "none":
                MessageProtocol.status(
                    "train", "[Warn] Loss reduction='none' wird beim Training nicht unterstützt — nutze 'mean'"
                )
            reduction = "mean"
        smoothing = 0.0
        if spec is not None:
            smoothing = max(0.0, min(float(getattr(spec, "label_smoothing", 0.0) or 0.0), 0.9))
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(reduction=reduction, label_smoothing=smoothing),
            "mse": nn.MSELoss(reduction=reduction),
            "mae": nn.L1Loss(reduction=reduction),
            "bce": nn.BCEWithLogitsLoss(reduction=reduction),
            "huber": nn.HuberLoss(reduction=reduction),
            "nll": nn.NLLLoss(reduction=reduction),
        }
        return loss_map.get(loss_name, nn.CrossEntropyLoss(reduction=reduction, label_smoothing=smoothing))

    def _make_scheduler(self, optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int):
        """W1: Scheduler-Instanz basierend auf IR-Konfiguration.
        steps_per_epoch wird für one_cycle (Batch-Level) und die Umrechnung
        der warmupSteps in Warmup-Epochen benötigt.
        minLr/warmupSteps kommen aus dem Scheduler-Node im Canvas.
        """
        import math as _math

        spec = self.ir.training if self.ir else None
        name = (spec.scheduler if spec else "none").lower()
        lr = spec.learning_rate if spec else self.config.learning_rate
        min_lr = float(getattr(spec, "min_lr", 0.0) or 0.0) if spec else 0.0
        eta_min = min_lr if min_lr > 0 else lr * 0.01
        warmup_steps = int(getattr(spec, "warmup_steps", 0) or 0) if spec else 0

        if name == "one_cycle":
            # one_cycle hat eingebautes Warmup (pct_start) — warmupSteps hier ignorieren
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=max(steps_per_epoch, 1),
            )

        # warmupSteps (Optimizer-Steps) → ganze Epochen für Epoch-Level-Scheduler
        warm_epochs = 0
        if warmup_steps > 0:
            warm_epochs = min(
                _math.ceil(warmup_steps / max(steps_per_epoch, 1)),
                max(epochs - 1, 0),
            )
        main_epochs = max(epochs - warm_epochs, 1)

        base = None
        if name == "cosine":
            base = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_epochs, eta_min=eta_min
            )
        elif name == "linear":
            end_factor = max(eta_min / lr, 1e-8) if lr > 0 else 0.01
            base = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=end_factor, total_iters=main_epochs
            )
        elif name in ("exponential", "exp"):
            # gamma so dass LR nach allen Epochen auf eta_min gefallen ist
            target = max(eta_min / lr, 1e-8) if lr > 0 else 0.01
            gamma = target ** (1.0 / max(main_epochs, 1))
            base = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif name == "polynomial":
            base = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=main_epochs, power=2.0
            )
        # "constant"/"none"/unbekannt — kein Scheduler
        if base is None:
            return None

        if warm_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_epochs
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, base], milestones=[warm_epochs]
            )
        return base

    def train(self) -> bool:
        if not self.model:
            MessageProtocol.error("Modell nicht initialisiert", "setup() fehlgeschlagen")
            return False
        # K2/K3: Sicherheitsnetz — sollte durch Orchestrator-Fix nie mehr erreicht werden,
        # bleibt aber als stiller Guard falls setup() direkt umgangen wird.
        if self.ir is None:
            return False

        try:
            epochs = self.ir.training.epochs if self.ir else self.config.epochs
            batch_size = self.ir.training.batch_size if self.ir else self.config.batch_size
            is_clf = True
            if self.ir and self.ir.training.task_type == "regression":
                is_clf = False

            if self.ir:
                train_loader, val_loader = get_dataloaders(
                    self.ir, self.config.dataset_path, batch_size
                )
            else:
                stub = CanvasGraphIR(
                    version=1, nodes=[], edges=[], execution_order=[],
                    training=IRTrainingSpec(
                        epochs=epochs, batch_size=batch_size,
                        num_classes=10,
                    ),
                    data=None,
                )
                train_loader, val_loader = get_dataloaders(
                    stub, self.config.dataset_path, batch_size
                )

            optimizer = self._make_optimizer()
            self._optimizer = optimizer  # Fix 1.3: für save_model (optimizer_state_dict)

            # Fix 1.3: Echter Resume – Optimizer-State aus vorherigem Checkpoint laden
            if self._prev_optimizer_state is not None:
                try:
                    optimizer.load_state_dict(self._prev_optimizer_state)
                    MessageProtocol.status("train", "✓ Optimizer-State aus Checkpoint geladen — echter Resume")
                except Exception as e:
                    MessageProtocol.status("train", f"[Warn] Optimizer-State konnte nicht geladen werden: {e}")

            criterion = self._make_loss()
            # AMP/fp16 ist CUDA-gebunden. Auf MPS/CPU liefert autocast(cuda)+GradScaler
            # kein echtes fp16 (no-op/Warnung) → sauber auf fp32 zurückfallen.
            use_amp = bool(self.config.fp16) and self.device == "cuda"
            if self.config.fp16 and not use_amp:
                MessageProtocol.status(
                    "train",
                    f"[Warn] fp16 wird nur auf CUDA unterstützt (Device: {self.device}) — Training läuft in fp32.",
                )
            scaler = GradScaler() if use_amp else None

            # Shape test
            try:
                sample_x, _ = next(iter(train_loader))
                sample_x = sample_x.to(self.device)
                with torch.no_grad():
                    out = self.model(sample_x)
                MessageProtocol.status(
                    "train",
                    f"✓ Shape Test: {list(sample_x.shape)} → {list(out.shape)}",
                )
            except RuntimeError as e:
                err = str(e)
                MessageProtocol.error("Shape Test fehlgeschlagen", err)
                import re
                m = re.search(r"\((\d+)x(\d+)\s+and\s+(\d+)x(\d+)\)", err)
                diag = {"error_type": "shape_mismatch", "raw_error": err}
                if m:
                    diag["actual_output_features"] = int(m.group(2))
                    diag["expected_input_features"] = int(m.group(3))
                MessageProtocol.status("train", f"[DIAGNOSTIC_JSON]\n{json.dumps(diag)}\n[/DIAGNOSTIC_JSON]")
                return False

            # W2: Gradient Accumulation — aus IR lesen, default 1 (kein Effekt)
            # MUSS vor _make_scheduler() stehen damit OneCycleLR korrekte steps bekommt
            accum_steps = max(int(self.ir.training.grad_accum) if self.ir else 1, 1)
            if accum_steps > 1:
                MessageProtocol.status("train", f"✓ Gradient Accumulation aktiv: accum_steps={accum_steps}")

            # Gradient-Clipping: Wert aus dem Optimizer-Node im Canvas (clipGrad),
            # Fallback auf die globale Trainings-Config. 0 = deaktiviert.
            max_grad = (
                float(getattr(self.ir.training, "clip_grad", self.config.max_grad_norm))
                if self.ir else float(self.config.max_grad_norm)
            )

            # W1 Edit 3a: Scheduler anlegen — nach Shape-Test und accum_steps, steps_per_epoch jetzt bekannt
            # W2 Fix: effective_steps_per_epoch = ceil(len(train_loader) / accum_steps)
            # identische Logik wie do_step-Gate: is_last_batch fängt Rest-Batches ab
            import math
            steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
            scheduler = self._make_scheduler(optimizer, epochs, steps_per_epoch)
            self._scheduler = scheduler
            is_batch_scheduler = scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            )
            # W1 Edit 3a: Scheduler-State für Resume laden
            prev_scheduler_state = getattr(self, "_prev_scheduler_state", None)
            if scheduler is not None and prev_scheduler_state is not None:
                try:
                    scheduler.load_state_dict(prev_scheduler_state)
                    MessageProtocol.status("train", "✓ Scheduler-State aus Checkpoint geladen")
                except Exception as e:
                    MessageProtocol.status("train", f"[Warn] Scheduler-State konnte nicht geladen werden: {e}")
            if scheduler is not None:
                sched_name = self.ir.training.scheduler if self.ir else "none"
                MessageProtocol.status("train", f"✓ Scheduler aktiv: {sched_name}")

            best_val = float("inf")
            for epoch in range(epochs):
                if self.is_stopped:
                    break
                self.model.train()
                total_loss = 0.0
                correct = 0
                n_samples = 0
                steps = 0
                optimizer_steps = 0
                optimizer.zero_grad()  # W2: einmalig zu Beginn jeder Epoch

                for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                    if self.is_stopped:
                        break
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # W2: Loss skalieren für korrekte Gradientenmagnitude
                    if use_amp:
                        with autocast():
                            out = self.model(batch_x)
                            loss_raw = criterion(out, batch_y) if is_clf else criterion(out.squeeze(-1), batch_y.float())
                            loss = loss_raw / accum_steps
                        scaler.scale(loss).backward()
                    else:
                        out = self.model(batch_x)
                        loss_raw = criterion(out, batch_y) if is_clf else criterion(out.squeeze(-1), batch_y.float())
                        loss = loss_raw / accum_steps
                        loss.backward()

                    # W2: Gated optimizer.step() — nur alle accum_steps Batches
                    is_last_batch = (batch_idx + 1) == len(train_loader)
                    do_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

                    if do_step:
                        if use_amp:
                            if max_grad > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            if max_grad > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)
                            optimizer.step()

                        # W1 SAFE: one_cycle scheduler.step() nur bei echtem optimizer.step()
                        if scheduler is not None and is_batch_scheduler:
                            scheduler.step()

                        optimizer.zero_grad()  # W2: Gradienten erst nach echtem Step zurücksetzen
                        optimizer_steps += 1

                    # Metriken auf ungeskaltem loss_raw
                    total_loss += loss_raw.item()
                    if is_clf and out.dim() >= 2:
                        correct += (out.argmax(dim=-1) == batch_y).sum().item()
                    n_samples += batch_y.size(0)
                    steps += 1

                avg_loss = total_loss / max(steps, 1)
                acc = correct / max(n_samples, 1)

                self.model.eval()
                val_sum = 0.0
                val_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        vo = self.model(vx)
                        val_sum += (
                            criterion(vo, vy).item()
                            if is_clf
                            else criterion(vo.squeeze(-1), vy.float()).item()
                        )
                        val_steps += 1
                val_loss = val_sum / max(val_steps, 1)
                if val_loss < best_val:
                    best_val = val_loss

                # W1 Edit 3c: Epoch-Level Scheduler (cosine, linear, exponential)
                if scheduler is not None and not is_batch_scheduler:
                    scheduler.step()

                self.train_history["epochs"].append(epoch + 1)
                self.train_history["train_losses"].append(avg_loss)
                self.train_history["val_losses"].append(val_loss)

                lr_now = optimizer.param_groups[0]["lr"]
                total_steps = max(len(train_loader), 1) * epochs // accum_steps
                step_idx = (epoch + 1) * optimizer_steps
                MessageProtocol.progress(
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    step=step_idx,
                    total_steps=total_steps,
                    train_loss=avg_loss,
                    val_loss=val_loss,
                    learning_rate=lr_now,
                    metrics={"accuracy": acc},
                )
                MessageProtocol.status(
                    "train",
                    f"[Metric] epoch={epoch + 1} loss={avg_loss:.6f} val_loss={val_loss:.6f} "
                    f"accuracy={acc:.6f} lr={lr_now:.8f} optimizer_steps={optimizer_steps}",
                )

            MessageProtocol.status("train", "✓ Training komplett (Canvas IR Runtime)")
            return True
        except Exception as e:
            MessageProtocol.error("Training Fehler", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            return False

    def validate(self) -> Dict[str, float]:
        return {"val_loss": 0.0}

    def save_model(self, output_path: str, optimizer: Optional[optim.Optimizer] = None) -> bool:
        """Fix 1.3: Speichert vollständigen IR + optimizer_state_dict für Inference-Reload und echten Resume."""
        try:
            if not self.model:
                return False
            # K2: Legacy-Modell ohne IR nicht speichern — würde ein nutzloses Artefakt erzeugen
            if self.ir is None:
                MessageProtocol.error("Legacy-Modell — Save nicht möglich", self._LEGACY_ERROR)
                return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_state_dict": self.model.state_dict(),
                "training_history": self.train_history,
            }
            if self.ir:
                payload["graph_ir"] = {
                    "version": self.ir.version,
                    "node_count": len(self.ir.nodes),
                    "edge_count": len(self.ir.edges),
                    # Vollständige Nodes/Edges – nötig für build_model_from_graph() beim Reload
                    "nodes": [
                        {"id": n.id, "type": n.type, "category": n.category, "params": n.params}
                        for n in self.ir.nodes
                    ],
                    "edges": [
                        {"source": e.source, "target": e.target}
                        for e in self.ir.edges
                    ],
                    "execution_order": self.ir.execution_order,
                    "training": {
                        "epochs": self.ir.training.epochs,
                        "batchSize": self.ir.training.batch_size,
                        "learningRate": self.ir.training.learning_rate,
                        "weightDecay": self.ir.training.weight_decay,
                        "optimizer": self.ir.training.optimizer,
                        "loss": self.ir.training.loss,
                        "scheduler": self.ir.training.scheduler,
                        "numClasses": self.ir.training.num_classes,
                        "taskType": self.ir.training.task_type,
                        "precision": self.ir.training.precision,
                        "gradAccum": self.ir.training.grad_accum,
                        "gpu": self.ir.training.gpu,
                    },
                }
            # Optimizer-State für echten Resume (Epoche N+1 weitertrainieren)
            if optimizer is not None:
                try:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                except Exception:
                    pass  # Optional – Save nie blockieren
            # W1 Edit 3e: Scheduler-State für echten Resume
            if self._scheduler is not None:
                try:
                    payload["scheduler_state_dict"] = self._scheduler.state_dict()
                except Exception:
                    pass  # Optional – Save nie blockieren
            torch.save(payload, output_path)
            MessageProtocol.status("save", f"✓ Modell gespeichert: {output_path}")
            return True
        except Exception as e:
            MessageProtocol.error("Save Fehler", str(e))
            return False

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "model_class": self.model_class.__name__ if self.model_class else "DynamicGraphModule",
            "total_params": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "training_history": self.train_history,
            "runtime": "graph_ir" if self.ir else "legacy_code",
        }

    def export(self) -> str:
        try:
            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "model.pt"
            self.save_model(str(model_path), optimizer=self._optimizer)
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.get_metrics(), f, indent=2, default=str)
            return str(output_dir)
        except Exception as e:
            MessageProtocol.error("Export Fehler", str(e))
            return str(self.config.output_path)
# run() wurde entfernt -- toter Code, wird vom Orchestrator nie aufgerufen.
