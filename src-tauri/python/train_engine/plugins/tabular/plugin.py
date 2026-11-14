"""Tabellen-Klassifikation und -Regression (task_type 'tabular').

Braucht kein heruntergeladenes Basismodell: das Netz wird aus den Spalten des
Datensatzes gebaut. Ein vorher trainierter Stand (model.pt im Modellordner)
wird weitertrainiert, wenn Spalten und Klassen dazu passen.
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from core.plugin_base import TrainPlugin  # noqa: E402
from core.protocol import MessageProtocol  # noqa: E402

import tabular_core as tc  # noqa: E402


def resolve_device(name: str):
    import torch
    name = (name or "cpu").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def parse_hidden(value) -> list:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value or "128,64").replace(";", ",").split(",") if v.strip()]


class Plugin(TrainPlugin):
    def setup(self) -> None:
        import torch
        pc = self.config.plugin_config or {}
        self.target_column = str(pc.get("target_column", "") or "").strip()
        self.task_request = str(pc.get("task", "auto") or "auto").strip().lower()
        self.hidden = parse_hidden(pc.get("hidden_layers", "128,64"))
        self.dropout = float(pc.get("dropout", self.config.dropout if self.config.dropout else 0.1))
        self.val_ratio = float(pc.get("val_ratio", 0.15))
        self.test_ratio = float(pc.get("test_ratio", 0.15))
        self.patience = int(pc.get("patience", 10))
        self.device = resolve_device(str(pc.get("device", "cpu")))
        torch.manual_seed(int(self.config.seed or 42))
        import numpy as np
        np.random.seed(int(self.config.seed or 42))  # Batch-Reihenfolge reproduzierbar
        MessageProtocol.status("init", f"Tabellen-Plugin bereit (Geraet: {self.device})")

    def load_data(self) -> None:
        MessageProtocol.status("loading_data", "Lade Tabelle...")
        frames = tc.load_frames(self.config.dataset_path)
        self.splits = tc.resolve_splits(frames, self.val_ratio, self.test_ratio, int(self.config.seed or 42))
        train = self.splits["train"]
        self.target = tc.pick_target(train, self.target_column)
        self.prep = tc.Preprocessor().fit(train, self.target, self.task_request)
        self.tensors = {}
        for name, df in self.splits.items():
            y, mask = self.prep.transform_target(df[self.target])
            x_num, x_cat = self.prep.transform(df)
            self.tensors[name] = (x_num[mask], x_cat[mask], y[mask])
            dropped = int((~mask).sum())
            if dropped:
                MessageProtocol.warning(f"{name}: {dropped} Zeilen ohne gueltigen Zielwert uebersprungen.")
        info = (f"Ziel '{self.target}' -> {self.prep.task}"
                + (f" ({len(self.prep.classes)} Klassen)" if self.prep.task == "classification" else "")
                + f"; {len(self.prep.numeric)} Zahlen-, {len(self.prep.categorical)} Kategoriespalten"
                + (f"; ignoriert: {', '.join(self.prep.dropped)}" if self.prep.dropped else "")
                + "; Zeilen " + ", ".join(f"{k}={len(v[2])}" for k, v in self.tensors.items()))
        MessageProtocol.status("loading_data", info)
        if len(self.tensors["train"][2]) == 0:
            raise ValueError("Keine Trainingszeilen mit gueltigem Zielwert.")

    def build_model(self) -> None:
        import torch
        self.model = tc.build_model(self.prep, self.hidden, self.dropout)
        prev = Path(self.config.model_path or "") / tc.WEIGHTS_FILE
        prev_prep = Path(self.config.model_path or "") / tc.PREPROCESSING_FILE
        if prev.is_file() and prev_prep.is_file():
            try:
                old = json.loads(prev_prep.read_text(encoding="utf-8"))
                same = (old.get("target") == self.prep.target and old.get("classes") == self.prep.classes
                        and [c["name"] for c in old.get("numeric", [])] == [c["name"] for c in self.prep.numeric]
                        and [c["name"] for c in old.get("categorical", [])] == [c["name"] for c in self.prep.categorical])
                if same:
                    # Alte Kategorie-Tabellen und Skalierung weiter nutzen, sonst passen die Gewichte nicht.
                    self.prep = tc.Preprocessor.from_dict(old)
                    self.load_data_with_prep()
                    self.model = tc.build_model(self.prep, self.hidden, self.dropout)
                    self.model.load_state_dict(torch.load(prev, map_location="cpu", weights_only=True))
                    MessageProtocol.status("building_model", "Vorheriger Stand geladen, wird weitertrainiert.")
                else:
                    MessageProtocol.warning("Vorheriger Stand passt nicht zu den Spalten — Netz startet neu.")
            except Exception as exc:  # noqa: BLE001
                MessageProtocol.warning(f"Vorheriger Stand nicht ladbar ({exc}) — Netz startet neu.")
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"Tabellennetz: {n_params:,} Parameter, Schichten {self.hidden}")

    def load_data_with_prep(self) -> None:
        for name, df in self.splits.items():
            y, mask = self.prep.transform_target(df[self.target])
            x_num, x_cat = self.prep.transform(df)
            self.tensors[name] = (x_num[mask], x_cat[mask], y[mask])

    def _batches(self, split: str, shuffle: bool):
        import numpy as np
        import torch
        x_num, x_cat, y = self.tensors[split]
        idx = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
        bs = max(1, int(self.config.batch_size or 64))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            yield (torch.from_numpy(x_num[j]).to(self.device), torch.from_numpy(x_cat[j]).to(self.device),
                   torch.from_numpy(y[j]).to(self.device))

    def _loss(self, out, y):
        import torch.nn.functional as F
        if self.prep.task == "classification":
            return F.cross_entropy(out, y, label_smoothing=float(self.config.label_smoothing or 0.0))
        return F.mse_loss(out[:, 0], y)

    def _eval(self, split: str):
        import numpy as np
        import torch
        if split not in self.tensors or len(self.tensors[split][2]) == 0:
            return None, {}
        self.model.eval()
        losses, preds = [], []
        with torch.no_grad():
            for xn, xc, y in self._batches(split, False):
                out = self.model(xn, xc)
                losses.append(float(self._loss(out, y)) * len(y))
                preds.append(out.argmax(1).cpu().numpy() if self.prep.task == "classification"
                             else self.prep.decode_regression(out[:, 0].cpu().numpy()))
        y_true = self.tensors[split][2]
        pred = np.concatenate(preds)
        loss = sum(losses) / len(y_true)
        if self.prep.task == "classification":
            return loss, tc.classification_metrics(y_true, pred, len(self.prep.classes))
        return loss, tc.regression_metrics(self.prep.decode_regression(y_true), pred)

    def train(self) -> None:
        import torch
        self.t0 = time.time()
        epochs = max(1, int(self.config.epochs or 1))
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.config.learning_rate or 1e-3),
                                weight_decay=float(self.config.weight_decay or 0.0))
        steps_per_epoch = max(1, -(-len(self.tensors["train"][2]) // max(1, int(self.config.batch_size or 64))))
        total = epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(self.config.learning_rate or 1e-3),
                                                    total_steps=total, pct_start=0.1)
        self.best = (float("inf"), None, 0)
        self.history = []
        step, bad = 0, 0
        for epoch in range(1, epochs + 1):
            if self.is_stopped:
                break
            self.model.train()
            run, n = 0.0, 0
            for xn, xc, y in self._batches("train", True):
                if self.is_stopped:
                    break
                opt.zero_grad()
                loss = self._loss(self.model(xn, xc), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm or 1.0))
                opt.step()
                sched.step()
                step += 1
                run += loss.item() * len(y)
                n += len(y)
            train_loss = run / max(n, 1)
            val_loss, val_metrics = self._eval("val")
            ref = val_loss if val_loss is not None else train_loss
            if ref < self.best[0]:
                self.best = (ref, copy.deepcopy(self.model.state_dict()), epoch)
                bad = 0
            else:
                bad += 1
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
            MessageProtocol.progress(epoch=epoch, total_epochs=epochs, step=step, total_steps=total,
                                     train_loss=train_loss, val_loss=val_loss,
                                     learning_rate=sched.get_last_lr()[0],
                                     metrics={f"val_{k}": v for k, v in val_metrics.items()})
            if self.patience > 0 and bad >= self.patience:
                MessageProtocol.status("training", f"Early Stopping nach Epoche {epoch} (bester Stand: Epoche {self.best[2]})")
                break
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        self.steps = step

    def validate(self) -> dict:
        res = {"total_epochs": len(self.history), "total_steps": getattr(self, "steps", 0),
               "best_epoch": self.best[2], "training_duration_seconds": round(time.time() - self.t0, 2),
               "final_train_loss": self.history[-1]["train_loss"] if self.history else 0.0,
               "task": self.prep.task, "target_column": self.target}
        val_loss, _ = self._eval("val")
        res["final_val_loss"] = val_loss
        res["best_val_loss"] = self.best[0] if self.best[0] != float("inf") else None
        split = "test" if "test" in self.tensors and len(self.tensors["test"][2]) else "val"
        _, metrics = self._eval(split)
        y_train = self.tensors["train"][2]
        y_eval = self.tensors[split][2]
        if self.prep.task == "regression":
            y_train, y_eval = self.prep.decode_regression(y_train), self.prep.decode_regression(y_eval)
        base = tc.baseline_metrics(self.prep, y_train, y_eval)
        res.update({f"{split}_{k}": v for k, v in metrics.items()})
        res.update({f"{split}_{k}": v for k, v in base.items()})
        res["eval_split"] = split
        MessageProtocol.status("validating", "Ergebnis " + split + ": " + ", ".join(
            f"{k}={v:.4f}" for k, v in {**metrics, **base}.items()))
        self.final_metrics = res
        return res

    def export(self) -> str:
        out = Path(self.config.output_path)
        self.model.to("cpu")
        tc.save_bundle(out, self.model, self.prep, {"hidden": self.hidden, "dropout": self.dropout},
                       {"split": {"val_ratio": self.val_ratio, "test_ratio": self.test_ratio,
                                  "seed": int(self.config.seed or 42)},
                        "metrics": {k: v for k, v in getattr(self, "final_metrics", {}).items()
                                    if isinstance(v, (int, float, str)) or v is None}})
        (out / "training_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        MessageProtocol.status("saving", f"Modell gespeichert: {out}")
        return str(out)
