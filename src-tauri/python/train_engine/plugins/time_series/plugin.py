"""Zeitreihen-Vorhersage mit PatchTST (task_type 'time_series').

Modellquelle:
  * leeres FrameTrain-Modell (config.json model_type 'ft_time_series') oder
    plugin_config from_scratch=true -> PatchTST neu initialisiert,
  * heruntergeladener PatchTST von Hugging Face (model_type 'patchtst' mit
    Gewichten) -> wird feinabgestimmt,
  * eine bereits mit FrameTrain trainierte Version -> wird weitertrainiert.
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

import ts_core as ts  # noqa: E402

SUPPORTED_ARCHITECTURES = ["patchtst", "ft_time_series"]


def resolve_device(name: str):
    import torch
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class Plugin(TrainPlugin):
    def setup(self) -> None:
        import torch
        pc = self.config.plugin_config or {}
        self.pc = pc
        self.context = int(pc.get("context_length", 64))
        self.horizon = int(pc.get("prediction_length", 16))
        self.season = int(pc.get("season_length", 0) or 0)
        self.val_ratio = float(pc.get("val_ratio", 0.1))
        self.test_ratio = float(pc.get("test_ratio", 0.2))
        self.patience = int(pc.get("patience", 5))
        self.from_scratch = bool(pc.get("from_scratch", False))
        self.device = resolve_device(str(pc.get("device", "auto")))
        torch.manual_seed(int(self.config.seed or 42))
        if self.context < 4 or self.horizon < 1:
            raise ValueError("context_length muss >= 4 und prediction_length >= 1 sein.")
        # Weitertrainieren einer FrameTrain-Version: deren Fenster gelten.
        meta_file = Path(self.config.model_path or "") / ts.META_FILE
        self.prev_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else None
        if self.prev_meta and not self.from_scratch:
            self.context = int(self.prev_meta["context_length"])
            self.horizon = int(self.prev_meta["prediction_length"])
        elif ts.is_pretrained_patchtst(self.config.model_path) and not self.from_scratch:
            from transformers import PatchTSTConfig
            self.context = int(PatchTSTConfig.from_pretrained(self.config.model_path).context_length)
            MessageProtocol.status("init", f"Vortrainierter PatchTST: Kontextlaenge {self.context} aus dessen Konfiguration")
        MessageProtocol.status("init", f"Zeitreihen-Plugin: Kontext {self.context}, Vorhersage {self.horizon}, Geraet {self.device}")

    def load_data(self) -> None:
        MessageProtocol.status("loading_data", "Lade Zeitreihen...")
        df = ts.load_all(self.config.dataset_path)
        pc = self.pc
        self.cols = ts.detect_columns(df, str(pc.get("time_column", "") or ""), str(pc.get("target_columns", "") or ""),
                                      str(pc.get("id_column", "") or ""))
        if self.prev_meta and not self.from_scratch and self.prev_meta.get("value_columns") != self.cols.values:
            raise ValueError(f"Die trainierte Version erwartet die Wertspalten {self.prev_meta.get('value_columns')}, "
                             f"das Dataset liefert {self.cols.values}.")
        series = ts.build_series(df, self.cols)
        need = self.context + self.horizon * 3
        short = [s for s, a in series.items() if len(a) < need]
        for s in short:
            series.pop(s)
        if short:
            MessageProtocol.warning(f"{len(short)} Serie(n) kuerzer als {need} Werte uebersprungen.")
        if not series:
            raise ValueError(f"Keine Serie ist lang genug: noetig sind mindestens {need} Zeitpunkte "
                             f"(Kontext {self.context} + 3 x Vorhersage {self.horizon}). "
                             "Kontext- oder Vorhersagelaenge verkleinern.")
        self.series = series
        self.splits = {sid: ts.split_points(len(a), self.horizon, self.val_ratio, self.test_ratio)
                       for sid, a in series.items()}
        self.norm = ts.Normalizer.fit(series, {s: sp.train_end for s, sp in self.splits.items()})
        self.data = {p: ts.make_windows(series, self.splits, self.norm, p, self.context, self.horizon)
                     for p in ("train", "val", "test")}
        if len(self.data["train"][0]) == 0:
            raise ValueError("Der Trainingsabschnitt ist zu kurz fuer ein einziges Fenster. "
                             "Kontextlaenge verkleinern oder val/test-Anteil senken.")
        MessageProtocol.status("loading_data", (
            f"{len(series)} Serie(n), Werte: {', '.join(self.cols.values)}, Zeit: {self.cols.time or 'Zeilenfolge'}"
            + (f", Serien-ID: {self.cols.id}" if self.cols.id else "")
            + f"; Fenster train={len(self.data['train'][0])}, val={len(self.data['val'][0])}, test={len(self.data['test'][0])}"))

    def build_model(self) -> None:
        if self.prev_meta and not self.from_scratch:
            from transformers import PatchTSTForPrediction
            self.model = PatchTSTForPrediction.from_pretrained(self.config.model_path)
            self.source = "frametrain"
            MessageProtocol.status("building_model", "Trainierte Version geladen, wird weitertrainiert.")
        else:
            self.model, ctx, pretrained = ts.build_patchtst(len(self.cols.values), self.context, self.horizon,
                                                            self.pc, self.config.model_path, self.from_scratch)
            self.source = "huggingface" if pretrained else "scratch"
            MessageProtocol.status("building_model", "PatchTST " + (
                "von Hugging Face geladen (Vorhersagekopf an neue Laenge angepasst)" if pretrained
                else "neu initialisiert"))
        self.model.to(self.device)
        n = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"PatchTST: {n:,} Parameter")

    def _batches(self, part: str, shuffle: bool):
        import numpy as np
        import torch
        past, fut, _ = self.data[part]
        idx = np.random.permutation(len(past)) if shuffle else np.arange(len(past))
        bs = max(1, int(self.config.batch_size or 32))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            yield torch.from_numpy(past[j]).to(self.device), torch.from_numpy(fut[j]).to(self.device)

    def _val_loss(self, part="val"):
        import torch
        if len(self.data[part][0]) == 0:
            return None
        self.model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y in self._batches(part, False):
                out = self.model(past_values=x).prediction_outputs
                tot += float(torch.nn.functional.mse_loss(out, y)) * len(x)
                n += len(x)
        return tot / max(n, 1)

    def train(self) -> None:
        import torch
        self.t0 = time.time()
        epochs = max(1, int(self.config.epochs or 1))
        lr = float(self.config.learning_rate or 1e-3)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=float(self.config.weight_decay or 0.0))
        bs = max(1, int(self.config.batch_size or 32))
        per_epoch = max(1, -(-len(self.data["train"][0]) // bs))
        max_steps = int(self.config.max_steps or -1)
        total = per_epoch * epochs if max_steps <= 0 else min(per_epoch * epochs, max_steps)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total, 1), pct_start=0.1)
        self.best = (float("inf"), None, 0)
        self.history = []
        step, bad = 0, 0
        for epoch in range(1, epochs + 1):
            if self.is_stopped or step >= total:
                break
            self.model.train()
            run, n = 0.0, 0
            for x, y in self._batches("train", True):
                if self.is_stopped or step >= total:
                    break
                opt.zero_grad()
                out = self.model(past_values=x).prediction_outputs
                # Verlust auf normalisierter Skala: Serien unterschiedlicher Groesse zaehlen gleich.
                loss = torch.nn.functional.mse_loss(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm or 1.0))
                opt.step()
                sched.step()
                step += 1
                run += loss.item() * len(x)
                n += len(x)
            train_loss = run / max(n, 1)
            val_loss = self._val_loss("val")
            ref = val_loss if val_loss is not None else train_loss
            if ref < self.best[0]:
                self.best = (ref, copy.deepcopy(self.model.state_dict()), epoch)
                bad = 0
            else:
                bad += 1
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            MessageProtocol.progress(epoch=epoch, total_epochs=epochs, step=step, total_steps=total,
                                     train_loss=train_loss, val_loss=val_loss, learning_rate=sched.get_last_lr()[0])
            if self.patience > 0 and bad >= self.patience:
                MessageProtocol.status("training", f"Early Stopping nach Epoche {epoch} (bester Stand: Epoche {self.best[2]})")
                break
        if self.best[1] is not None:
            self.model.load_state_dict(self.best[1])
        self.steps = step

    def _meta(self) -> dict:
        return {"version": 1, "context_length": self.context, "prediction_length": self.horizon,
                "season_length": self.season, "val_ratio": self.val_ratio, "test_ratio": self.test_ratio,
                "time_column": self.cols.time, "value_columns": self.cols.values, "id_column": self.cols.id,
                "norm_mean": self.norm.mean, "norm_std": self.norm.std, "source": self.source}

    def validate(self) -> dict:
        res = {"total_epochs": len(self.history), "total_steps": getattr(self, "steps", 0),
               "best_epoch": self.best[2], "training_duration_seconds": round(time.time() - self.t0, 2),
               "final_train_loss": self.history[-1]["train_loss"] if self.history else 0.0,
               "final_val_loss": self._val_loss("val"),
               "best_val_loss": self.best[0] if self.best[0] != float("inf") else None}
        part = "test" if len(self.data["test"][0]) else "val"
        ev = ts.evaluate_part(self.model, self.series, self._meta(), part, self.device)
        res.update({f"{part}_{k}": v for k, v in ev["metrics"].items()})
        res.update({f"{part}_naive_{k}": v for k, v in ev["naive_metrics"].items()})
        res["eval_split"] = part
        MessageProtocol.status("validating", f"{part}: MAE {ev['metrics']['mae']:.4f} (naiv {ev['naive_metrics']['mae']:.4f}), "
                               f"MASE {ev['metrics'].get('mase', 0):.3f} (naiv {ev['naive_metrics'].get('mase', 0):.3f})")
        self.final_metrics = res
        return res

    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        self.model.save_pretrained(str(out))
        meta = self._meta()
        meta["metrics"] = {k: v for k, v in getattr(self, "final_metrics", {}).items()
                           if isinstance(v, (int, float, str)) or v is None}
        ts.save_meta(out, meta)
        (out / "training_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        MessageProtocol.status("saving", f"Modell gespeichert: {out}")
        return str(out)
