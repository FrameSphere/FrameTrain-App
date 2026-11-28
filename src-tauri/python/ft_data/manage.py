"""Datensatz-Ordner pruefen und reparieren (fuer KI-Werkzeuge).

Aufruf:  python -m ft_data.manage '{"kind": "tree", "path": "...", "max_depth": 4}'
         python -m ft_data.manage '{"kind": "repair", "path": "...", "actions": ["merge_subsets", ...]}'

tree    Ordnerbaum mit Zaehlungen je Ordner und konkreten Befunden. Jeder Befund
        nennt die Reparatur (datasets_repair action), die ihn behebt.
repair  Feste, gefahrlose Reparaturen direkt im Datensatz-Ordner:
          remove_hidden  versteckte Hilfsdateien (.cache, .DS_Store, __MACOSX) loeschen
          merge_subsets  Teil-Datensaetze (<teil>/images + <teil>/labels) zusammenfuehren
          pair_labels    Bilder und Labels aus getrennten Baeumen paaren
          write_yaml     data.yaml aus Ordneraufbau und Klassenliste neu schreiben

Warum: Live-Test 28.11. (FruitDet) – der Datensatz hatte einen Aufbau, den das
Training nicht fand, und die KI konnte ihn weder sehen noch korrigieren.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import fetch
from .fetch import IMAGE_EXTS, FetchError, emit

JUNK_DIRS = {".cache", "__MACOSX", ".ipynb_checkpoints", ".git", ".trash"}
JUNK_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SPLITS = ("train", "val", "valid", "validation", "test")
MAX_TREE_ENTRIES = 400


def _is_hidden(rel_parts) -> bool:
    return any(p.startswith(".") or p in JUNK_DIRS for p in rel_parts)


def _kind(f: Path) -> str:
    ext = f.suffix.lower()
    if ext in IMAGE_EXTS:
        return "images"
    if ext == ".txt" and f.name not in ("classes.txt", "labels.txt", "README.txt", "readme.txt"):
        return "labels"
    return "other"


def _label_for(img: Path) -> Path:
    from .detection import label_path_for
    return label_path_for(img)


def tree(root: Path, max_depth: int = 4) -> Dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FetchError("Dataset folder not found.")
    max_depth = max(1, min(int(max_depth or 4), 8))
    dirs: Dict[str, Dict[str, int]] = {}
    hidden_files = 0
    hidden_bytes = 0
    hidden_dirs = set()
    images: List[Path] = []
    labels: List[Path] = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        if _is_hidden(rel.parts) or f.name in JUNK_FILES:
            hidden_files += 1
            try:
                hidden_bytes += f.stat().st_size
            except OSError:
                pass
            top = next((p for p in rel.parts[:-1] if p.startswith(".") or p in JUNK_DIRS), None)
            hidden_dirs.add(top or f.name)
            continue
        kind = _kind(f)
        if kind == "images":
            images.append(f)
        elif kind == "labels" and "labels" in [p.lower() for p in rel.parts[:-1]]:
            labels.append(f)
        # Zaehlung fuer jeden Vorfahren bis max_depth
        parent_parts = rel.parts[:-1]
        for depth in range(0, min(len(parent_parts), max_depth) + 1):
            key = "/".join(parent_parts[:depth]) or "."
            d = dirs.setdefault(key, {"images": 0, "labels": 0, "other": 0, "files": 0})
            d[kind] += 1
            d["files"] += 1

    entries = [{"path": k, **v} for k, v in sorted(dirs.items(), key=lambda kv: (kv[0] != ".", kv[0]))]
    truncated = len(entries) > MAX_TREE_ENTRIES
    entries = entries[:MAX_TREE_ENTRIES]

    findings: List[Dict[str, Any]] = []
    if hidden_files:
        findings.append({
            "code": "hidden_files",
            "message": f"{hidden_files} hidden helper files ({hidden_bytes / 1e6:.1f} MB) in {sorted(hidden_dirs)[:5]}; they are not training data.",
            "fix": "remove_hidden",
        })
    subsets = [s for s in fetch._yolo_subsets(root) if s != root]
    canonical = fetch._canonical_yolo(root)
    if subsets and not canonical:
        names = [str(s.relative_to(root)) for s in subsets]
        findings.append({
            "code": "sub_datasets",
            "message": f"{len(subsets)} separate YOLO sub-datasets ({names[:6]}{'...' if len(names) > 6 else ''}); training expects one train/images + train/labels.",
            "fix": "merge_subsets",
        })
    if images and labels:
        label_set = {p.resolve() for p in labels}
        no_label = [p for p in images if _label_for(p).resolve() not in label_set]
        image_stems = {(_label_for(p)).resolve() for p in images}
        no_image = [p for p in labels if p.resolve() not in image_stems]
        if no_label and len(no_label) == len(images):
            findings.append({
                "code": "labels_not_next_to_images",
                "message": "No image has its label at the expected place (images/x.jpg -> labels/x.txt).",
                "fix": "pair_labels",
            })
        else:
            if no_label:
                findings.append({
                    "code": "images_without_labels",
                    "message": f"{len(no_label)} images have no label file, e.g. {[str(p.relative_to(root)) for p in no_label[:3]]}. Fine for background images, otherwise add labels or remove them.",
                    "fix": None,
                })
            if no_image:
                findings.append({
                    "code": "labels_without_images",
                    "message": f"{len(no_image)} label files have no image, e.g. {[str(p.relative_to(root)) for p in no_image[:3]]}.",
                    "fix": None,
                })
    if images and labels and not subsets:
        yaml = next((p for n in ("data.yaml", "dataset.yaml") for p in [root / n] if p.exists()), None)
        names = fetch._parse_names(yaml.read_text(encoding="utf-8", errors="ignore")) if yaml else []
        top = -1
        for lf in labels[:5000]:
            for line in lf.read_text(errors="ignore").splitlines():
                head = line.split(None, 1)[0] if line.strip() else ""
                if head.isdigit():
                    top = max(top, int(head))
        if not yaml:
            findings.append({"code": "no_yaml", "message": "No data.yaml with class names.", "fix": "write_yaml"})
        elif names and top >= len(names):
            findings.append({
                "code": "yaml_too_few_classes",
                "message": f"data.yaml lists {len(names)} classes but labels use class id {top}.",
                "fix": "write_yaml",
            })
    if images and not canonical and not subsets and not (root / "images").is_dir():
        findings.append({
            "code": "not_trainable_layout",
            "message": "Training expects train/images (+ train/labels) or images/train. Use datasets_move to arrange the folders, or datasets_split for a flat folder.",
            "fix": None,
        })

    return {
        "root": str(root),
        "layout": "yolo_split" if canonical else ("sub_datasets" if subsets else ("flat" if images else "other")),
        "images": len(images),
        "labels": len(labels),
        "hidden_files": hidden_files,
        "dirs": entries,
        "truncated": truncated,
        "findings": findings,
    }


def _replace_contents(root: Path, new: Path) -> None:
    """Inhalt von root durch new ersetzen (root selbst bleibt, damit Pfade gueltig sind)."""
    backup = root.parent / (root.name + ".repair_old")
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir()
    for p in list(root.iterdir()):
        shutil.move(str(p), backup / p.name)
    try:
        for p in list(new.iterdir()):
            shutil.move(str(p), root / p.name)
    except Exception:
        for p in list(root.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in list(backup.iterdir()):
            shutil.move(str(p), root / p.name)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(new, ignore_errors=True)


def remove_hidden(root: Path) -> int:
    removed = 0
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts)):
        if not p.exists():
            continue
        rel = p.relative_to(root)
        if (p.is_dir() and (p.name.startswith(".") or p.name in JUNK_DIRS)) or (p.is_file() and (p.name in JUNK_FILES or p.name.startswith("._"))):
            n = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 1
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed += n
    return removed


def repair(root: Path, actions: List[str]) -> Dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FetchError("Dataset folder not found.")
    allowed = {"remove_hidden", "merge_subsets", "pair_labels", "write_yaml"}
    unknown = [a for a in actions if a not in allowed]
    if unknown:
        raise FetchError(f"Unknown repair action(s) {unknown}. Allowed: {sorted(allowed)}.")
    done: List[Dict[str, Any]] = []
    work = root.parent / (root.name + ".repair_new")
    for action in actions:
        if action == "remove_hidden":
            done.append({"action": action, "removed_files": remove_hidden(root)})
        elif action == "merge_subsets":
            remove_hidden(root)
            if work.exists():
                shutil.rmtree(work)
            work.mkdir()
            merged = fetch.merge_yolo_subsets(root, work)
            if not merged:
                shutil.rmtree(work, ignore_errors=True)
                done.append({"action": action, "changed": False, "reason": "no sub-datasets found or already in train/images layout"})
                continue
            info = fetch.inspect(merged)
            _replace_contents(root, merged)
            shutil.rmtree(work, ignore_errors=True)
            done.append({"action": action, "changed": True, "images": info["images"], "classes": info["classes"], "splits": info["splits"]})
        elif action == "pair_labels":
            paired = fetch.pair_images_and_labels(root)
            if not paired:
                done.append({"action": action, "changed": False, "reason": "labels already next to images, or no matching label files"})
                continue
            info = fetch.inspect(paired)
            _replace_contents(root, paired)
            done.append({"action": action, "changed": True, "images": info["images"], "classes": info["classes"]})
        elif action == "write_yaml":
            names = fetch._read_class_list(root) or fetch._names_from_labels(root)
            for old in ("data.yaml", "dataset.yaml"):
                if (root / old).exists():
                    (root / old).unlink()
            if (root / "train" / "images").is_dir():
                fetch._write_data_yaml(root, names)
            else:
                fetch._write_data_yaml_inplace(root, names)
            done.append({"action": action, "changed": True, "classes": names})
    after = tree(root)
    return {"done": done, "tree": after}


def run(req: Dict[str, Any]) -> Dict[str, Any]:
    kind = req.get("kind")
    path = Path(req["path"])
    if kind == "tree":
        return tree(path, int(req.get("max_depth") or 4))
    if kind == "repair":
        return repair(path, [str(a) for a in (req.get("actions") or [])])
    raise FetchError(f"Unknown kind: {kind}")


def main(argv: List[str]) -> int:
    try:
        req = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        emit("result", **run(req))
        return 0
    except FetchError as e:
        emit("error", message=str(e))
        return 2
    except Exception as e:  # noqa: BLE001
        emit("error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
