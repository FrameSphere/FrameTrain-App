#!/usr/bin/env python3
"""
FrameTrain Laboratory Inference Script
Lädt ein lokal gespeichertes Modell und führt Inferenz auf einem Sample aus.
WICHTIG: Alle Modelle liegen LOKAL. Es wird NICHTS von HuggingFace heruntergeladen.
"""

import sys
import json
import time
import argparse
import os
from pathlib import Path


def detect_sample_type(sample_path: str) -> str:
    ext = Path(sample_path).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
        return 'image'
    elif ext in ['.txt', '.md']:
        return 'text'
    elif ext in ['.json', '.jsonl']:
        return 'json'
    elif ext in ['.csv', '.tsv']:
        return 'csv'
    elif ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        return 'audio'
    return 'unknown'


def find_actual_model_dir(model_path: str) -> tuple[str, str]:
    """
    Findet das tatsächliche Verzeichnis mit den Modell-Gewichten.
    
    Gibt zurück: (resolved_path, error_message)
    - resolved_path: Pfad der geladen werden kann (leer wenn Fehler)
    - error_message: Fehlerbeschreibung (leer wenn OK)
    
    FrameTrain speichert Modelle lokal in verschiedenen Strukturen:
    
    Struktur 1 - Basis-Modell (heruntergeladen von HF):
      /app_data/models/{model_id}/
        ├── config.json
        ├── model.safetensors  (oder pytorch_model.bin)
        ├── tokenizer.json
        └── model_info.json    (FrameTrain Metadata)
    
    Struktur 2 - Trainiertes Modell (nach Training):
      /app_data/training_outputs/{job_id}/
        ├── final_model/
        │   ├── config.json
        │   ├── model.safetensors
        │   └── tokenizer.json
        └── checkpoints/
    
    Struktur 3 - Modell-Version (in model_versions_new.path):
      Kann direkt auf final_model zeigen, oder auf einen Checkpoint
    """
    p = Path(model_path)
    
    if not p.exists():
        return '', (
            f"❌ Modell-Verzeichnis existiert nicht:\n{model_path}\n\n"
            f"Mögliche Ursachen:\n"
            f"• Das Modell wurde noch nicht trainiert\n"
            f"• Das Training wurde abgebrochen bevor das Modell gespeichert wurde\n"
            f"• Der Pfad in der Datenbank ist veraltet\n\n"
            f"💡 Tipp: Trainiere das Modell zuerst im Training-Panel"
        )
    
    if not p.is_dir():
        return '', f"❌ Kein Verzeichnis: {model_path}"
    
    def has_model_files(d: Path) -> bool:
        """Prüft ob ein Verzeichnis Modell-Gewichte enthält."""
        has_config = (d / 'config.json').exists()
        has_weights = (
            (d / 'model.safetensors').exists() or
            (d / 'pytorch_model.bin').exists() or
            any(d.glob('*.pt')) or
            any(d.glob('*.pth')) or
            any(d.glob('pytorch_model-*.bin'))  # sharded models
        )
        return has_config and has_weights
    
    # 1. Direkt prüfen
    if has_model_files(p):
        return str(p), ''
    
    # 2. final_model Unterverzeichnis prüfen (Standard-Output nach Training)
    final_model = p / 'final_model'
    if final_model.exists() and has_model_files(final_model):
        return str(final_model), ''
    
    # 3. Checkpoints prüfen (neuester zuerst)
    checkpoint_patterns = ['checkpoint-*', 'checkpoints/checkpoint-*']
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(p.glob(pattern))
    
    if checkpoints:
        # Sortiere nach Step-Nummer (neuester = höchste Zahl)
        def get_step(cp: Path) -> int:
            try:
                return int(cp.name.split('-')[-1])
            except ValueError:
                return 0
        checkpoints_with_files = [cp for cp in checkpoints if has_model_files(cp)]
        if checkpoints_with_files:
            latest = sorted(checkpoints_with_files, key=get_step, reverse=True)[0]
            return str(latest), ''
    
    # 4. Rekursiv suchen (max 2 Ebenen tief)
    for subdir in sorted(p.iterdir()):
        if subdir.is_dir() and has_model_files(subdir):
            return str(subdir), ''
    
    # 5. Nur config.json (ohne Gewichte) - transformers kann trotzdem versuchen zu laden
    if (p / 'config.json').exists():
        # Gib den Pfad zurück, transformers zeigt einen besseren Fehler
        return str(p), ''
    
    # Kein Modell gefunden
    files_found = [f.name for f in p.iterdir() if f.is_file()][:10]
    dirs_found = [d.name for d in p.iterdir() if d.is_dir()][:5]
    
    return '', (
        f"❌ Keine Modell-Gewichte in:\n{model_path}\n\n"
        f"Gefundene Dateien: {files_found}\n"
        f"Gefundene Ordner: {dirs_found}\n\n"
        f"Erwartet: config.json + model.safetensors oder pytorch_model.bin\n\n"
        f"💡 Das Training muss erst vollständig abgeschlossen sein.\n"
        f"   Unterbrochene Trainings speichern kein vollständiges Modell."
    )


def detect_task_type(model_dir: str) -> str:
    """Erkennt den Task-Typ aus config.json."""
    config_path = Path(model_dir) / 'config.json'
    if not config_path.exists():
        return 'auto'
    try:
        with open(config_path) as f:
            config = json.load(f)

        arch = config.get('architectures', [''])[0].lower()
        if 'forsequenceclassification' in arch:
            return 'text-classification'
        elif 'fortokenclassification' in arch:
            return 'token-classification'
        elif 'forobjectdetection' in arch or 'detr' in arch:
            return 'object-detection'
        elif 'forimageclassification' in arch or 'vit' in arch:
            return 'image-classification'
        elif 'causallm' in arch or 'gpt' in arch or 'llama' in arch or 'mistral' in arch:
            return 'text-generation'
        elif 'seq2seq' in arch or 't5' in arch or 'bart' in arch:
            return 'text2text-generation'
        elif 'forquestionanswering' in arch:
            return 'question-answering'

        tag = config.get('pipeline_tag', '')
        if tag:
            return tag
    except Exception:
        pass
    return 'auto'


def parse_hf_output(result, task_type: str) -> dict:
    """Parst HuggingFace pipeline output in RenderedOutput-Format."""
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']

    rendered = {
        "primary_label": None,
        "confidence": None,
        "labels": [],
        "bounding_boxes": [],
        "highlighted_spans": [],
        "generated_text": None,
        "key_values": [],
    }

    if isinstance(result, list):
        if not result:
            return rendered
        first = result[0]

        if isinstance(first, dict):
            if 'label' in first and 'score' in first:
                rendered['primary_label'] = first['label']
                rendered['confidence'] = float(first['score'])
                rendered['labels'] = [
                    {'label': str(r.get('label', '')), 'score': float(r.get('score', 0))}
                    for r in result[:8]
                ]

            elif 'entity' in first or 'entity_group' in first:
                label_colors = {}
                color_idx = 0
                spans = []
                for entity in result:
                    lbl = entity.get('entity_group') or entity.get('entity', '')
                    if lbl not in label_colors:
                        label_colors[lbl] = colors[color_idx % len(colors)]
                        color_idx += 1
                    spans.append({
                        'start': int(entity.get('start', 0)),
                        'end': int(entity.get('end', 0)),
                        'label': lbl,
                        'score': float(entity.get('score', 0)),
                        'color': label_colors[lbl],
                    })
                rendered['highlighted_spans'] = spans
                if spans:
                    rendered['primary_label'] = f"{len(spans)} Entität(en) erkannt"

            elif 'box' in first or 'xmin' in first:
                boxes = []
                for det in result:
                    box = det.get('box', {})
                    xmin = float(box.get('xmin', det.get('xmin', 0)))
                    ymin = float(box.get('ymin', det.get('ymin', 0)))
                    xmax = float(box.get('xmax', det.get('xmax', 0)))
                    ymax = float(box.get('ymax', det.get('ymax', 0)))
                    boxes.append({
                        'label': str(det.get('label', '')),
                        'score': float(det.get('score', 0)),
                        'x': xmin, 'y': ymin,
                        'width': xmax - xmin, 'height': ymax - ymin,
                    })
                rendered['bounding_boxes'] = boxes
                if boxes:
                    rendered['primary_label'] = f"{len(boxes)} Objekt(e) erkannt"

            elif 'generated_text' in first:
                rendered['generated_text'] = str(first['generated_text'])
                rendered['primary_label'] = 'Text generiert'

    elif isinstance(result, dict):
        if 'generated_text' in result:
            rendered['generated_text'] = str(result['generated_text'])
            rendered['primary_label'] = 'Text generiert'
        elif 'answer' in result:
            rendered['primary_label'] = str(result['answer'])
            rendered['confidence'] = float(result.get('score', 0))
            rendered['generated_text'] = str(result['answer'])
        elif 'translation_text' in result:
            rendered['generated_text'] = str(result['translation_text'])
            rendered['primary_label'] = 'Übersetzung'
        elif 'summary_text' in result:
            rendered['generated_text'] = str(result['summary_text'])
            rendered['primary_label'] = 'Zusammenfassung'
        else:
            rendered['key_values'] = [(str(k), str(v)) for k, v in list(result.items())[:10]]

    elif isinstance(result, str):
        rendered['generated_text'] = result
        rendered['primary_label'] = 'Ausgabe'

    return rendered


def map_task_to_output_type(task_type: str) -> str:
    mapping = {
        'text-classification': 'classification',
        'image-classification': 'classification',
        'audio-classification': 'classification',
        'token-classification': 'ner',
        'ner': 'ner',
        'object-detection': 'detection',
        'text-generation': 'generation',
        'text2text-generation': 'generation',
        'question-answering': 'generation',
        'translation': 'generation',
        'summarization': 'generation',
        'automatic-speech-recognition': 'generation',
        'fill-mask': 'classification',
    }
    return mapping.get(task_type, 'raw')


def run_inference(model_path: str, sample_path: str, task_type: str = 'auto'):
    start = time.time()
    sample_type = detect_sample_type(sample_path)

    # Finde das echte Modell-Verzeichnis (rein lokal, kein HF-Download)
    resolved_path, resolve_error = find_actual_model_dir(model_path)

    if resolve_error:
        elapsed_ms = int((time.time() - start) * 1000)
        print(json.dumps(build_error_output(sample_path, resolve_error, elapsed_ms), ensure_ascii=False))
        return

    if task_type == 'auto':
        task_type = detect_task_type(resolved_path)

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1

        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = -1  # transformers setzt MPS automatisch in neueren Versionen

        # KRITISCH: local_files_only=True verhindert jeglichen HF-Download
        # Das Modell MUSS lokal vorhanden sein.
        pipe = None
        last_error = None

        load_kwargs = {
            'local_files_only': True,  # NIEMALS HuggingFace kontaktieren
            'device': device,
        }

        try:
            if task_type == 'auto':
                pipe = pipeline(model=resolved_path, **load_kwargs)
            else:
                pipe = pipeline(task=task_type, model=resolved_path, **load_kwargs)
        except Exception as e:
            last_error = str(e)
            # Fallback ohne task
            try:
                pipe = pipeline(model=resolved_path, **load_kwargs)
            except Exception as e2:
                last_error = str(e2)

        if pipe is None:
            elapsed_ms = int((time.time() - start) * 1000)
            error_msg = (
                f"❌ Modell konnte nicht geladen werden\n\n"
                f"Pfad: {resolved_path}\n\n"
                f"Fehler: {last_error}\n\n"
                f"💡 Mögliche Ursachen:\n"
                f"• Zu wenig RAM/VRAM\n"
                f"• Inkompatible transformers-Version\n"
                f"• Modell-Dateien beschädigt\n"
                f"• Fehlende Abhängigkeiten (pip install transformers torch pillow)"
            )
            print(json.dumps(build_error_output(sample_path, error_msg, elapsed_ms), ensure_ascii=False))
            return

        # Input laden
        if sample_type == 'image':
            from PIL import Image
            input_data = Image.open(sample_path).convert('RGB')
        elif sample_type == 'audio':
            input_data = sample_path
        elif sample_type in ('text', 'json', 'csv', 'unknown'):
            with open(sample_path, 'r', encoding='utf-8', errors='replace') as f:
                input_data = f.read(4000)
        else:
            with open(sample_path, 'r', encoding='utf-8', errors='replace') as f:
                input_data = f.read(4000)

        # Inferenz
        if task_type in ('text-generation', 'text2text-generation'):
            raw_result = pipe(input_data, max_new_tokens=200)
        else:
            raw_result = pipe(input_data)

        elapsed_ms = int((time.time() - start) * 1000)

        actual_task = pipe.task if hasattr(pipe, 'task') else task_type
        rendered = parse_hf_output(raw_result, actual_task)

        try:
            json.dumps(raw_result)
            serializable_raw = raw_result
        except (TypeError, ValueError):
            serializable_raw = str(raw_result)

        output = {
            "sample_path": sample_path,
            "model_output_type": map_task_to_output_type(actual_task),
            "raw_output": serializable_raw,
            "rendered": rendered,
            "inference_time_ms": elapsed_ms,
            "error": None,
        }
        print(json.dumps(output, ensure_ascii=False))

    except ImportError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        error_msg = (
            f"⚠️ Fehlende Bibliothek: {e}\n\n"
            f"Python: {sys.executable}\n"
            f"Version: {sys.version.split()[0]}\n\n"
            f"Installation:\n"
            f"pip install torch transformers pillow\n"
            f"  oder\n"
            f"pip3 install torch transformers pillow"
        )
        print(json.dumps(build_error_output(sample_path, error_msg, elapsed_ms), ensure_ascii=False))

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        print(json.dumps(build_error_output(sample_path, str(e), elapsed_ms), ensure_ascii=False))


def build_error_output(sample_path: str, error_msg: str, elapsed_ms: int) -> dict:
    return {
        "sample_path": sample_path,
        "model_output_type": "raw",
        "raw_output": {},
        "rendered": {
            "primary_label": None,
            "confidence": None,
            "labels": [],
            "bounding_boxes": [],
            "highlighted_spans": [],
            "generated_text": None,
            "key_values": [("Fehler", error_msg)],
        },
        "inference_time_ms": elapsed_ms,
        "error": error_msg,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--sample_path', required=True)
    parser.add_argument('--task_type', default='auto')
    args = parser.parse_args()
    run_inference(args.model_path, args.sample_path, args.task_type)
