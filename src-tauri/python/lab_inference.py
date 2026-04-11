#!/usr/bin/env python3
"""
FrameTrain Laboratory Inference Script
Führt Modell-Inferenz auf einem einzelnen Sample aus und gibt strukturiertes JSON zurück.
Unterstützt: Klassifikation, Objekterkennung, NER, Textgenerierung, Audio, Tabellendaten
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


def detect_task_type(model_path: str) -> str:
    """Erkennt den Task-Typ aus der model config.json"""
    config_path = Path(model_path) / 'config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Aus architectures
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

            # Aus pipeline_tag
            tag = config.get('pipeline_tag', '')
            if tag:
                return tag

        except Exception:
            pass
    return 'auto'


def parse_hf_output(result, task_type: str) -> dict:
    """Parst HuggingFace pipeline output in unser RenderedOutput-Format"""
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
        if len(result) == 0:
            return rendered
        first = result[0]

        if isinstance(first, dict):
            # Klassifikation: [{label, score}, ...]
            if 'label' in first and 'score' in first:
                rendered['primary_label'] = first['label']
                rendered['confidence'] = float(first['score'])
                rendered['labels'] = [
                    {'label': str(r.get('label', '')), 'score': float(r.get('score', 0))}
                    for r in result[:8]
                ]

            # Token-Klassifikation / NER: [{entity, entity_group, start, end, score}, ...]
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

            # Objekterkennung: [{label, score, box: {xmin,ymin,xmax,ymax}}, ...]
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
                        'x': xmin,
                        'y': ymin,
                        'width': xmax - xmin,
                        'height': ymax - ymin,
                    })
                rendered['bounding_boxes'] = boxes
                if boxes:
                    rendered['primary_label'] = f"{len(boxes)} Objekt(e) erkannt"

            # Generierter Text als Liste
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

    if task_type == 'auto':
        task_type = detect_task_type(model_path)

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = -1  # pipeline nutzt mps automatisch in neueren Versionen

        # Pipeline erstellen
        if task_type == 'auto':
            pipe = pipeline(model=model_path, device=device)
        else:
            pipe = pipeline(task=task_type, model=model_path, device=device)

        # Input laden je nach Typ
        if sample_type == 'image':
            from PIL import Image
            input_data = Image.open(sample_path).convert('RGB')
        elif sample_type == 'audio':
            input_data = sample_path  # pipeline nimmt Pfad
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

        # Erkannten task_type für Mapping verwenden
        actual_task = pipe.task if hasattr(pipe, 'task') else task_type
        rendered = parse_hf_output(raw_result, actual_task)

        # raw_output serialisierbar machen
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

    except ImportError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        output = build_error_output(sample_path, f"Fehlende Bibliothek: {e}. Bitte installiere: pip install transformers torch pillow", elapsed_ms)

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        output = build_error_output(sample_path, str(e), elapsed_ms)

    print(json.dumps(output, ensure_ascii=False))


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
    parser = argparse.ArgumentParser(description='FrameTrain Laboratory Inference')
    parser.add_argument('--model_path', required=True, help='Pfad zum Modell-Verzeichnis')
    parser.add_argument('--sample_path', required=True, help='Pfad zum Sample')
    parser.add_argument('--task_type', default='auto', help='HuggingFace task type oder "auto"')
    args = parser.parse_args()

    run_inference(args.model_path, args.sample_path, args.task_type)
