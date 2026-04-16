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


def resolve_actual_model_path(model_path: str) -> str:
    """
    Resolves the actual loadable model path from a FrameTrain storage directory.
    
    FrameTrain stores models in app_data/models/{model_id}/ with a model_info.json.
    The actual HuggingFace model name or local weights path is stored in source_path.
    
    Priority:
    1. If path has HF model files (config.json + weights) -> use as-is
    2. If path has model_info.json -> use source_path from it
    3. Return as-is (might be a HF model ID string directly)
    """
    p = Path(model_path)
    
    # Check if it's a FrameTrain storage directory with model_info.json
    model_info_path = p / 'model_info.json'
    if model_info_path.exists():
        try:
            with open(model_info_path) as f:
                info = json.load(f)
            
            source_path = info.get('source_path')
            source = info.get('source', '')
            
            # Check if the FrameTrain dir itself has valid HF model files
            has_config = (p / 'config.json').exists()
            has_safetensors = (p / 'model.safetensors').exists()
            has_pytorch = (p / 'pytorch_model.bin').exists()
            
            if has_config and (has_safetensors or has_pytorch):
                # This directory IS a proper HF model directory - use it directly
                return model_path
            
            # Use source_path (HuggingFace repo ID or local path to original model)
            if source_path and source_path.strip():
                return source_path
                
        except Exception:
            pass
    
    # Check if path itself has valid HF model files
    if p.exists() and p.is_dir():
        has_config = (p / 'config.json').exists()
        has_safetensors = (p / 'model.safetensors').exists()
        has_pytorch = (p / 'pytorch_model.bin').exists()
        
        if has_config and (has_safetensors or has_pytorch):
            return model_path
    
    # Return as-is - might be a HuggingFace model ID ("microsoft/deberta-v3-base")
    return model_path


def detect_task_type(model_path: str) -> str:
    """Erkennt den Task-Typ aus der model config.json"""
    # First resolve the actual model path
    resolved = resolve_actual_model_path(model_path)
    config_path = Path(resolved) / 'config.json'
    
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

    # Resolve the actual model path (handles FrameTrain storage format)
    resolved_model_path = resolve_actual_model_path(model_path)

    if task_type == 'auto':
        task_type = detect_task_type(model_path)

    try:
        import torch
        from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM

        device = 0 if torch.cuda.is_available() else -1

        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = -1  # pipeline nutzt mps automatisch in neueren Versionen

        # ============ VALIDATE MODEL PATH ============
        model_path_obj = Path(resolved_model_path)
        
        # For local paths: validate they exist and have model files
        # For HF model IDs (like "microsoft/deberta-v3-base"): skip local validation
        is_local_path = model_path_obj.exists() or '/' in resolved_model_path or '\\' in resolved_model_path
        is_hf_id = '/' in resolved_model_path and not model_path_obj.exists()
        
        if is_local_path and not is_hf_id:
            if not model_path_obj.exists():
                # Path doesn't exist - provide helpful error
                original_name = Path(model_path).name
                raise FileNotFoundError(
                    f"❌ Modell nicht gefunden: '{original_name}'\n\n"
                    f"Gesuchter Pfad: {resolved_model_path}\n\n"
                    f"Mögliche Ursachen:\n"
                    f"1. Das Modell wurde noch nicht trainiert (nur das Original-Modell gespeichert)\n"
                    f"2. Das Modell wurde gelöscht oder verschoben\n"
                    f"3. Der Laboratory-Test braucht ein trainiertes Modell\n\n"
                    f"💡 Tipp: Trainiere zuerst das Modell, dann teste im Laboratory"
                )
            
            # Check if it's a valid model directory
            has_safetensors = (model_path_obj / 'model.safetensors').exists()
            has_pytorch = (model_path_obj / 'pytorch_model.bin').exists()
            has_config = (model_path_obj / 'config.json').exists()
            has_weights_any = list(model_path_obj.glob('*.pt')) or list(model_path_obj.glob('*.pth')) or list(model_path_obj.glob('*.onnx'))
            
            if not (has_safetensors or has_pytorch or has_config or has_weights_any):
                raise ValueError(
                    f"❌ Das Verzeichnis enthält keine Modell-Dateien.\n"
                    f"Pfad: {resolved_model_path}\n\n"
                    f"Erwartet: config.json, model.safetensors, pytorch_model.bin oder *.pt Dateien\n\n"
                    f"Wurde das Training erfolgreich abgeschlossen?"
                )

        # ============ LOAD MODEL ============
        pipe = None
        last_error = None
        
        try:
            if task_type == 'auto':
                pipe = pipeline(model=resolved_model_path, device=device)
            else:
                pipe = pipeline(task=task_type, model=resolved_model_path, device=device)
        except Exception as e:
            last_error = str(e)
            # Try without specifying task
            try:
                pipe = pipeline(model=resolved_model_path, device=device)
            except Exception as e2:
                last_error = str(e2)
        
        if pipe is None:
            raise ValueError(
                f"❌ Das Modell konnte nicht geladen werden.\n"
                f"Pfad: {resolved_model_path}\n\n"
                f"Fehler: {last_error}\n\n"
                f"💡 Mögliche Lösungen:\n"
                f"1. Ist die Model-Version korrekt gespeichert?\n"
                f"2. Fehlen Abhängigkeiten (z.B. transformers, torch)?\n"
                f"3. Ist genug RAM/VRAM verfügbar?\n"
                f"4. Ist das Modell mit transformers kompatibel?"
            )

        # Input laden je nach Typ
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

    except ImportError as e:
        elapsed_ms = int((time.time() - start) * 1000)
        error_msg = f"⚠️ Fehlende Bibliothek: {str(e)}\n\n"
        error_msg += f"Aktuelle Python: {sys.executable}\n"
        error_msg += f"Python Version: {sys.version.split()[0]}\n\n"
        error_msg += "Installation:\n"
        error_msg += "pip install torch transformers pillow\n"
        error_msg += "  ODER\n"
        error_msg += "pip3 install torch transformers pillow"
        output = build_error_output(sample_path, error_msg, elapsed_ms)

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
