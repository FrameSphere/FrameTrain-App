"""
plugin_manager.py
=================
Verwaltet Python-Abhängigkeiten für FrameTrain.

Wird von Rust (plugin_commands.rs) aufgerufen:

  # Beim ersten Start — gibt JSON-Array mit PluginInfo zurück:
  python plugin_manager.py --first-launch

  # Plugin installieren — gibt Fortschritt zeilenweise aus:
  python plugin_manager.py --install <plugin_id>

  # Nur prüfen was fehlt:
  python plugin_manager.py --check

Die PluginInfo-Struktur muss dem Rust-Struct entsprechen:
  id, name, description, category, icon, built_in,
  train_plugin, test_plugin, required_packages, optional_packages,
  estimated_size_mb, install_time_minutes, github_path, priority,
  is_selected, is_installed
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


# ============================================================================
# PLUGIN-DEFINITIONEN
# Jede Gruppe entspricht einem "Plugin" das der User beim Erststart auswählt.
# required_packages werden IMMER installiert, optional_packages nur wenn gewünscht.
# ============================================================================

PLUGINS: List[Dict[str, Any]] = [
    {
        "id":                   "core_nlp",
        "name":                 "NLP / LLM Training",
        "description":          "Fine-Tuning von Sprachmodellen: BERT, GPT, LLaMA, T5, mT5, Mistral und mehr. Unterstützt LoRA, 4-bit QLoRA und alle HuggingFace-Modelle.",
        "category":             "nlp",
        "icon":                 "brain",
        "built_in":             True,
        "train_plugin":         "nlp",
        "test_plugin":          None,
        "required_packages":    [
            "torch",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.26.0",
            "tokenizers>=0.15.0",
            "safetensors",
            "numpy",
            "psutil",
        ],
        "optional_packages":    [
            "peft>=0.7.0",
            "bitsandbytes",
            "sentencepiece",
            "protobuf",
        ],
        "estimated_size_mb":    2500,
        "install_time_minutes": 5,
        "github_path":          None,
        "priority":             1,
        "is_selected":          True,
    },
    {
        "id":                   "vision",
        "name":                 "Vision / Bildklassifikation",
        "description":          "Training von Bildklassifikations-Modellen: ViT, ResNet, EfficientNet, ConvNeXt, Swin Transformer. Unterstützt ImageFolder und Custom-Datensätze.",
        "category":             "vision",
        "icon":                 "image",
        "built_in":             True,
        "train_plugin":         "vision",
        "test_plugin":          None,
        "required_packages":    [
            "torch",
            "torchvision",
            "pillow",
        ],
        "optional_packages":    [
            "timm>=0.9.0",
            "albumentations",
        ],
        "estimated_size_mb":    800,
        "install_time_minutes": 3,
        "github_path":          None,
        "priority":             2,
        "is_selected":          False,
    },
    {
        "id":                   "detection",
        "name":                 "Objekt-Erkennung",
        "description":          "Training von Objekt-Erkennungs-Modellen: YOLO v5/v8/v10/v11, Faster R-CNN. Unterstützt YOLO-Format (data.yaml) und COCO-Annotierungen.",
        "category":             "vision",
        "icon":                 "scan",
        "built_in":             True,
        "train_plugin":         "detection",
        "test_plugin":          None,
        "required_packages":    [
            "torch",
            "torchvision",
            "pillow",
            "pyyaml",
        ],
        "optional_packages":    [
            "ultralytics",
        ],
        "estimated_size_mb":    600,
        "install_time_minutes": 3,
        "github_path":          None,
        "priority":             3,
        "is_selected":          False,
    },
    {
        "id":                   "audio",
        "name":                 "Audio / Speech Recognition",
        "description":          "Fine-Tuning von Audio-Modellen: Whisper (OpenAI), Wav2Vec2, HuBERT. Ideal für Spracherkennung und Audio-Klassifikation.",
        "category":             "audio",
        "icon":                 "mic",
        "built_in":             True,
        "train_plugin":         "audio",
        "test_plugin":          None,
        "required_packages":    [
            "torch",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.26.0",
            "librosa",
            "soundfile",
        ],
        "optional_packages":    [
            "evaluate",
            "jiwer",
        ],
        "estimated_size_mb":    900,
        "install_time_minutes": 4,
        "github_path":          None,
        "priority":             4,
        "is_selected":          False,
    },
    {
        "id":                   "tabular",
        "name":                 "Tabular / Strukturierte Daten",
        "description":          "Training auf CSV/Tabellen-Daten: XGBoost, LightGBM, Random Forest, Logistische Regression. Für Klassifikation und Regression.",
        "category":             "tabular",
        "icon":                 "table",
        "built_in":             True,
        "train_plugin":         "tabular",
        "test_plugin":          None,
        "required_packages":    [
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy",
            "joblib",
        ],
        "optional_packages":    [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
        "estimated_size_mb":    200,
        "install_time_minutes": 2,
        "github_path":          None,
        "priority":             5,
        "is_selected":          False,
    },
]


# ============================================================================
# PAKET-PRÜFUNG
# ============================================================================

def _import_name(package_spec: str) -> str:
    """Mappt pip-Paketnamen auf den Python-Import-Namen."""
    name = package_spec.split(">=")[0].split("==")[0].split(">")[0].split("<")[0].strip()
    mapping = {
        "torch":                  "torch",
        "torchvision":            "torchvision",
        "transformers":           "transformers",
        "datasets":               "datasets",
        "accelerate":             "accelerate",
        "tokenizers":             "tokenizers",
        "safetensors":            "safetensors",
        "numpy":                  "numpy",
        "psutil":                 "psutil",
        "peft":                   "peft",
        "bitsandbytes":           "bitsandbytes",
        "sentencepiece":          "sentencepiece",
        "protobuf":               "google.protobuf",
        "pillow":                 "PIL",
        "timm":                   "timm",
        "albumentations":         "albumentations",
        "ultralytics":            "ultralytics",
        "pyyaml":                 "yaml",
        "librosa":                "librosa",
        "soundfile":              "soundfile",
        "evaluate":               "evaluate",
        "jiwer":                  "jiwer",
        "scikit-learn":           "sklearn",
        "pandas":                 "pandas",
        "joblib":                 "joblib",
        "xgboost":                "xgboost",
        "lightgbm":               "lightgbm",
    }
    return mapping.get(name, name.replace("-", "_"))


def is_installed(package_spec: str) -> bool:
    """Prüft ob ein Paket importierbar ist."""
    import_name = _import_name(package_spec)
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def check_plugin(plugin: Dict[str, Any]) -> Dict[str, Any]:
    """Prüft welche Pakete eines Plugins fehlen. Gibt erweitertes Plugin zurück."""
    missing_required = [p for p in plugin["required_packages"]  if not is_installed(p)]
    missing_optional = [p for p in plugin["optional_packages"]  if not is_installed(p)]

    result = dict(plugin)
    result["is_installed"]     = len(missing_required) == 0
    result["missing_required"] = missing_required
    result["missing_optional"] = missing_optional
    return result


# ============================================================================
# INSTALLATION
# ============================================================================

def install_package(package_spec: str) -> bool:
    """Installiert ein einzelnes Paket mit pip. Gibt True zurück bei Erfolg."""
    print(f"[FrameTrain] Installiere: {package_spec}", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", package_spec,
             "--quiet", "--no-warn-script-location"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"[FrameTrain] ✓ {package_spec} installiert", flush=True)
            return True
        else:
            print(f"[FrameTrain] ✗ {package_spec} fehlgeschlagen: {result.stderr[:300]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"[FrameTrain] ✗ {package_spec} Timeout (>10min)", flush=True)
        return False
    except Exception as e:
        print(f"[FrameTrain] ✗ {package_spec} Fehler: {e}", flush=True)
        return False


def install_plugin(plugin_id: str) -> bool:
    """Installiert alle required_packages eines Plugins."""
    plugin = next((p for p in PLUGINS if p["id"] == plugin_id), None)
    if plugin is None:
        print(f"[FrameTrain] ✗ Plugin nicht gefunden: {plugin_id}", flush=True)
        return False

    info = check_plugin(plugin)
    missing = info["missing_required"]

    if not missing:
        print(f"[FrameTrain] ✓ {plugin['name']}: alle Pakete bereits installiert", flush=True)
        return True

    print(f"[FrameTrain] Installiere {len(missing)} Pakete für '{plugin['name']}'...", flush=True)
    success = True
    for pkg in missing:
        if not install_package(pkg):
            success = False

    if success:
        print(f"[FrameTrain] ✓ {plugin['name']} bereit", flush=True)
    else:
        print(f"[FrameTrain] ✗ {plugin['name']}: einige Pakete konnten nicht installiert werden", flush=True)

    return success


# ============================================================================
# HAUPTFUNKTIONEN
# ============================================================================

def cmd_first_launch():
    """
    Gibt JSON-Array aller Plugins mit Installationsstatus aus.
    Rust parst dieses JSON und zeigt es im FirstLaunchSetup an.
    """
    result = []
    for plugin in PLUGINS:
        info = check_plugin(plugin)
        # Nur Felder ausgeben die Rust erwartet (PluginInfo struct)
        result.append({
            "id":                   info["id"],
            "name":                 info["name"],
            "description":          info["description"],
            "category":             info["category"],
            "icon":                 info["icon"],
            "built_in":             info["built_in"],
            "train_plugin":         info["train_plugin"],
            "test_plugin":          info["test_plugin"],
            "required_packages":    info["required_packages"],
            "optional_packages":    info["optional_packages"],
            "estimated_size_mb":    info["estimated_size_mb"],
            "install_time_minutes": info["install_time_minutes"],
            "github_path":          info["github_path"],
            "priority":             info["priority"],
            "is_selected":          info["is_selected"],
            "is_installed":         info["is_installed"],
        })
    print(json.dumps(result, ensure_ascii=False))


def cmd_check():
    """Gibt Installationsstatus aller Plugins aus (für Debugging)."""
    for plugin in PLUGINS:
        info = check_plugin(plugin)
        status = "✓" if info["is_installed"] else "✗"
        print(f"{status} {info['name']}")
        if info["missing_required"]:
            print(f"  Fehlt (required): {', '.join(info['missing_required'])}")
        if info["missing_optional"]:
            print(f"  Fehlt (optional): {', '.join(info['missing_optional'])}")


def cmd_install(plugin_id: str):
    """Installiert ein Plugin. Fortschritt wird zeilenweise ausgegeben."""
    success = install_plugin(plugin_id)
    sys.exit(0 if success else 1)


def cmd_install_all_required():
    """Installiert alle als is_selected=True markierten Plugins."""
    selected = [p for p in PLUGINS if p.get("is_selected", False)]
    success = True
    for plugin in selected:
        if not install_plugin(plugin["id"]):
            success = False
    sys.exit(0 if success else 1)


# ============================================================================
# EINSTIEGSPUNKT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Plugin Manager")
    parser.add_argument("--first-launch",   action="store_true",
                        help="Gibt JSON-Liste aller Plugins aus (für Erststart-Dialog)")
    parser.add_argument("--install",        type=str, metavar="PLUGIN_ID",
                        help="Installiert ein bestimmtes Plugin")
    parser.add_argument("--install-all",    action="store_true",
                        help="Installiert alle vorselektierten Plugins")
    parser.add_argument("--check",          action="store_true",
                        help="Zeigt Installationsstatus aller Plugins")
    args = parser.parse_args()

    if args.first_launch:
        cmd_first_launch()
    elif args.install:
        cmd_install(args.install)
    elif args.install_all:
        cmd_install_all_required()
    elif args.check:
        cmd_check()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
