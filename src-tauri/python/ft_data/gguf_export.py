"""ft_data.gguf_export – LLM als GGUF fuer Ollama / LM Studio schreiben.

Konvertiert wird mit dem offiziellen Skript von llama.cpp (convert_hf_to_gguf.py).
Kunden haben llama.cpp in der Regel nicht installiert; das Skript braucht aber
kein Kompilieren. Darum laedt FrameTrain einmalig die Konvertierungsdateien
eines festgelegten Releases (convert_hf_to_gguf.py, conversion/, gguf-py/)
in den Cache der App und startet sie mit genau dem Python, das auch trainiert.
Die mitgelieferte gguf-py-Version passt so immer zum Skript.

Ist llama.cpp doch installiert (LLAMA_CPP_DIR, ~/llama.cpp, Homebrew), wird
dessen Skript genommen.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Festes Release: dieselbe Skriptversion auf jedem Rechner. Geprueft am
# 27.09.2026 mit SmolLM2 (llama) — das GGUF laedt und antwortet wie das Original.
LLAMA_CPP_TAG = "v0.5.0"
SOURCE_URL = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_TAG}.tar.gz"
NEEDED = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/")
OUTTYPES = ("q8_0", "f16", "bf16", "f32")


def cache_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "FrameTrain"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FrameTrain" / "cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "frametrain"
    return base / f"llama.cpp-{LLAMA_CPP_TAG}"


def _installed_converter() -> Optional[Path]:
    for base in filter(None, [os.environ.get("LLAMA_CPP_DIR"), str(Path.home() / "llama.cpp"),
                              "/opt/homebrew/share/llama.cpp", "/usr/local/share/llama.cpp"]):
        p = Path(base) / "convert_hf_to_gguf.py"
        if p.exists():
            return p
    return None


def _safe_members(tar: tarfile.TarFile):
    """Nur die benoetigten Dateien, ohne absolute Pfade oder '..'."""
    for m in tar.getmembers():
        if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in Path(m.name).parts:
            continue
        rel = m.name.split("/", 1)[-1] if "/" in m.name else ""
        if rel.startswith(NEEDED):
            m.name = rel
            yield m


def ensure_converter(status: Callable[[str], None] = lambda _m: None) -> Path:
    """Pfad zu convert_hf_to_gguf.py — installiert oder aus dem Cache."""
    found = _installed_converter()
    if found:
        return found
    target = cache_dir()
    script = target / "convert_hf_to_gguf.py"
    if script.exists() and (target / "gguf-py").is_dir():
        return script
    status(f"Lade die GGUF-Konvertierung von llama.cpp {LLAMA_CPP_TAG} (einmalig, ca. 40 MB) ...")
    target.mkdir(parents=True, exist_ok=True)
    archive = target.parent / f"llama.cpp-{LLAMA_CPP_TAG}.tar.gz"
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp, open(archive, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(target, members=list(_safe_members(tar)))
    finally:
        archive.unlink(missing_ok=True)
    if not script.exists():
        raise RuntimeError(f"Im llama.cpp-Archiv {LLAMA_CPP_TAG} fehlt convert_hf_to_gguf.py.")
    return script


def convert(model_dir: Path, outfile: Path, outtype: str = "q8_0",
            status: Callable[[str], None] = lambda _m: None) -> Tuple[bool, str]:
    """(ok, Meldung). Scheitert nie mit Exception — der Export des HF-Modells
    ist dann trotzdem fertig, GGUF laesst sich spaeter nachholen."""
    outtype = outtype if outtype in OUTTYPES else "q8_0"
    try:
        script = ensure_converter(status)
    except Exception as exc:  # offline, Proxy, GitHub nicht erreichbar
        return False, (f"llama.cpp-Konvertierung nicht verfuegbar ({exc}). Das Modell liegt im "
                       "HF-Format vor; mit Internetverbindung laesst sich GGUF spaeter erzeugen.")
    env = dict(os.environ)
    gguf_py = script.parent / "gguf-py"
    if gguf_py.is_dir():
        env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
    cmd: List[str] = [sys.executable, str(script), str(model_dir), "--outfile", str(outfile), "--outtype", outtype]
    status(f"Schreibe GGUF ({outtype}) ...")
    kwargs = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, **kwargs)
    if res.returncode != 0 or not outfile.exists():
        tail = (res.stderr or res.stdout or "")[-800:]
        if "No module named" in tail:
            from ft_data.deps import hint_for_exception
            mod = tail.split("No module named", 1)[1].split("\n")[0].strip(" '\"")
            tail += "\n\n" + (hint_for_exception(ImportError(name=mod)) or "")
        return False, f"GGUF-Export fehlgeschlagen:\n{tail}"
    return True, f"GGUF gespeichert: {outfile} ({outfile.stat().st_size / 1e6:.0f} MB)"


def write_modelfile(out_dir: Path, gguf_name: str, system_prompt: str = "") -> Path:
    """Ollama-Modelfile neben dem GGUF. Das Chat-Template steckt im GGUF selbst."""
    lines = [f"FROM ./{gguf_name}"]
    if system_prompt:
        lines.append('SYSTEM """' + system_prompt.replace('"""', "'''") + '"""')
    path = out_dir / "Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
