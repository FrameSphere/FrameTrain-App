"""ft_data.deps – ehrliche Hinweise, wenn ein Python-Paket fehlt.

Ein Hinweis "pip install x" fuehrt beim Kunden oft ins Leere: das `pip` im
Terminal gehoert haeufig zu einem anderen Python als dem, mit dem FrameTrain
trainiert (System-Python, Homebrew, python.org, conda …). Darum nennen die
Meldungen zuerst den Weg in der App und dann den exakten Interpreter:

    "/usr/local/bin/python3.11" -m pip install "peft>=0.17.0"

Mindestversionen und Gruppen spiegeln MIN_VERSIONS / packages_for_task in
src-tauri/src/plugin_commands.rs.
"""
import sys
from typing import Iterable, Optional

# Import-Name → pip-Name (nur wo sie sich unterscheiden)
MODULE_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "rouge_score": "rouge-score",
    "sentence_transformers": "sentence-transformers",
    "mlx_lm": "mlx-lm",
    "mlx": "mlx-lm",
    "yaml": "pyyaml",
    "google": "protobuf",
    "huggingface_hub": "huggingface_hub",
}

MIN_VERSIONS = {
    "transformers": "4.56.0", "datasets": "2.14.0", "accelerate": "1.0.0",
    "peft": "0.17.0", "diffusers": "0.32.0", "sentence-transformers": "3.0.0",
    "seqeval": "1.2.0", "jiwer": "3.0.0", "rouge-score": "0.1.2", "sacrebleu": "2.0.0",
    "mlx-lm": "0.31.0", "bitsandbytes": "0.45.0", "ultralytics": "8.3.0",
    "opencv-python": "4.8.0", "librosa": "0.10.0", "soundfile": "0.12.0", "pillow": "9.0.0",
}

# Paket → Gruppe, wie sie in Einstellungen → Python-Pakete heisst
GROUP_OF = {
    "peft": "LLM Fine-Tuning", "mlx-lm": "LLM Fine-Tuning", "bitsandbytes": "LLM Fine-Tuning",
    "diffusers": "Generative Modelle",
    "ultralytics": "YOLO",
}
DEFAULT_GROUP = "HuggingFace-Stack"


def pip_name(module: str) -> str:
    root = (module or "").split(".")[0]
    return MODULE_TO_PIP.get(root, root.replace("_", "-"))


def _spec(pkg: str) -> str:
    return f'"{pkg}>={MIN_VERSIONS[pkg]}"' if pkg in MIN_VERSIONS else pkg


def install_hint(*packages: str) -> str:
    """Hinweis fuer fehlende Pakete: erst die App, dann der exakte Interpreter."""
    pkgs = [p for p in packages if p]
    groups = sorted({GROUP_OF.get(p, DEFAULT_GROUP) for p in pkgs}) or [DEFAULT_GROUP]
    return (
        f"In FrameTrain: Einstellungen → Python-Pakete → „{' / '.join(groups)}“ installieren.\n"
        f"Oder im Terminal (genau dieses Python nutzt FrameTrain):\n"
        f'  "{sys.executable}" -m pip install {" ".join(_spec(p) for p in pkgs)}'
    )


def missing(*packages: str, what: str = "") -> ImportError:
    """ImportError mit Hinweis — `raise missing("peft", what="LoRA")`."""
    label = ", ".join(packages)
    head = f"{what} braucht das Paket {label}." if what else f"Python-Paket fehlt: {label}."
    return ImportError(f"{head}\n\n{install_hint(*packages)}")


def hint_for_exception(exc: BaseException) -> Optional[str]:
    """Hinweis zu einem ImportError/ModuleNotFoundError, sonst None."""
    name = getattr(exc, "name", None)
    if not name:
        text = str(exc)
        if "No module named" in text:
            name = text.split("No module named", 1)[1].strip(" '\"")
    if not name:
        return None
    return install_hint(pip_name(name))


def packages_text(packages: Iterable[str]) -> str:
    return " ".join(_spec(p) for p in packages)
