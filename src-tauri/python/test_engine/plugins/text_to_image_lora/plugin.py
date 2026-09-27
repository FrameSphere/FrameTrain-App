"""Test-Plugin: Bilder mit einem LoRA-trainierten Stable-Diffusion-Modell erzeugen.

single:  Prompt -> PNG; das Ergebnis ist der Pfad des Bildes (die Test-Seite
         zeigt es an).
dataset: fuer N Captions des Datasets je ein Bild. Eine Trefferquote gibt es
         bei Bilderzeugung nicht; results.json stellt Prompt, erzeugtes und
         Original-Bild nebeneinander.

plugin_config: num_inference_steps (25), guidance_scale (7.5), seed (42),
negative_prompt, max_images (16, Obergrenze fuer den Dataset-Lauf).
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        pc = config.plugin_config or {}
        self.steps = int(pc.get("num_inference_steps", pc.get("sample_steps", 25)) or 25)
        self.guidance = float(pc.get("guidance_scale", 7.5) or 7.5)
        self.seed = int(pc.get("seed", 42) or 0)
        self.negative_prompt = str(pc.get("negative_prompt", "") or "")
        self.max_images = int(pc.get("max_images", 16) or 16)
        self.pipe = None
        self.info: Dict[str, Any] = {}
        self.size = 512
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        from ft_data.diffusion import default_size, load_pipeline

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        self.pipe, self.info = load_pipeline(
            model_path, status=lambda m: TestProtocol.status("loading", m))
        self.size = default_size(self.pipe, self.info)
        dev = self.pipe.device
        TestProtocol.status("loading", f"Pipeline geladen | {self.size}px | {self.steps} Schritte | Geraet: {dev}")

    def _render(self, prompt: str, out_dir: Path, name: str, seed: int) -> Path:
        from ft_data.diffusion import generate_image

        img = generate_image(self.pipe, prompt, steps=self.steps, guidance=self.guidance,
                             seed=seed, negative_prompt=self.negative_prompt, size=self.size)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        img.save(path)
        return path

    def run_single(self):
        from ft_data.diffusion import safe_filename

        prompt = (self.config.single_input or "").strip() or str(self.info.get("validation_prompt") or "")
        if not prompt:
            raise ValueError("Prompt ist leer.")
        t0 = time.time()
        path = self._render(prompt, Path(self.config.output_path),
                            f"{safe_filename(prompt)}_{self.seed}.png", self.seed)
        TestProtocol.complete_single(
            predicted=str(path), confidence=None, top_predictions=[],
            inference_time=time.time() - t0,
            extra={"output_kind": "image", "image_path": str(path), "prompt": prompt},
        )

    def run_dataset(self):
        from ft_data.image_text import evaluation_samples
        from ft_data.media import sample as random_sample

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        samples, split = evaluation_samples(
            root, "caption", instance_prompt=str(self.info.get("instance_prompt") or ""),
            status=lambda m: TestProtocol.status("loading", m))
        n = self.config.max_samples or self.max_images
        if n > self.max_images:
            TestProtocol.status("loading", f"Bilderzeugung ist teuer — begrenzt auf {self.max_images} Bilder "
                                           "(plugin_config max_images).")
            n = self.max_images
        samples = random_sample(samples, n)
        TestProtocol.status("running", f"Erzeuge {len(samples)} Bilder aus Captions ({split})...")

        out_dir = Path(self.config.output_path) / "images"
        results: List[Dict[str, Any]] = []
        total_time = 0.0
        started = time.time()
        for idx, s in enumerate(samples, start=1):
            if self.is_stopped:
                TestProtocol.status("stopped", "Test abgebrochen.")
                return
            t0 = time.time()
            path = self._render(s.answer, out_dir, f"{idx:04d}.png", self.seed + idx)
            dt = time.time() - t0
            total_time += dt
            results.append({
                "sample_id": idx, "input_text": s.answer, "input_path": str(s.image),
                "expected_output": str(s.image), "predicted_output": str(path),
                # Kein richtig/falsch bei Bilderzeugung — bewusst null statt erfunden.
                "is_correct": None, "confidence": None, "inference_time": dt,
            })
            elapsed = max(time.time() - started, 1e-6)
            TestProtocol.progress(current=idx, total=len(samples), sps=idx / elapsed,
                                  eta=(len(samples) - idx) * elapsed / idx)

        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        results_file = out / "results.json"
        results_file.write_text(json.dumps({
            "metrics": {"generated_images": len(results), "images_dir": str(out_dir),
                        "num_inference_steps": self.steps, "guidance_scale": self.guidance},
            "predictions": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results),
            accuracy=None, correct=None, average_loss=None,
            average_inference_time=total_time / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
            extra={"output_kind": "image", "images_dir": str(out_dir),
                   "images": [r["predicted_output"] for r in results]},
        )
