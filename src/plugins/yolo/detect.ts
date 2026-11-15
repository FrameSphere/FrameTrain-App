// YOLO Object Detection Plugin – detect.ts
// Erkennt YOLOv5/v8/v9/v11 Modelle anhand model_type, config oder Modell-ID

import type { ModelConfig } from '../types';

/** model_type-Werte fuer YOLO — muss zu plugins/yolo/manifest.json passen (Test prueft das). */
export const YOLO_MODEL_TYPES = [
  'yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolo12', 'yolo26', 'yolo', 'rtdetr', 'yolo_world',
];

export function detectYOLO(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // config.json: model_type = "yolo" oder architecture-Hinweis
  if (configJson) {
    const mt = configJson.model_type?.toLowerCase() ?? '';
    if (YOLO_MODEL_TYPES.includes(mt)) return true;
    // YOLOS ist ein HuggingFace-Transformer (YolosForObjectDetection), kein Ultralytics-Modell.
    if (mt === 'yolos') return false;
    const archs = (configJson.architectures ?? []).map((a: string) => a.toLowerCase());
    if (archs.some((a: string) => a.includes('yolo') && !a.startsWith('yolos'))) return true;
  }

  // Modell-ID / Pfad Heuristik
  const id = modelPathOrId.toLowerCase();
  // hustvl/yolos-tiny enthielt "/yolo" und landete hier — YOLOS gehoert zu hf-object-detection.
  if (/(^|[^a-z0-9])yolos([^a-z0-9]|$)/.test(id)) return false;
  // Ultralytics-RT-DETR: rtdetr-l.pt / rtdetr-x.pt (HF-Format heisst rtdetr_r18vd).
  if (/(^|[^a-z0-9])rtdetr-[lx]([^a-z0-9]|$)/.test(id)) return true;
  return (
    id.includes('yolov5') ||
    id.includes('yolov8') ||
    id.includes('yolov9') ||
    id.includes('yolo11') ||
    id.includes('yolo-') ||
    id.includes('/yolo') ||
    id.startsWith('yolo') ||
    // Ultralytics HuggingFace Hub Konvention
    id.includes('ultralytics/') ||
    id.includes('yolo_')
  );
}

export type YoloTask = 'detect' | 'segment' | 'pose' | 'obb' | 'classify';

/**
 * Aufgabe aus dem Gewichtsnamen, wie Ultralytics sie benennt
 * (yolo11n-seg.pt -> segment). null, wenn der Name nichts verraet
 * (model.pt, best.pt) — dann entscheidet das Python-Plugin am Checkpoint.
 */
export function yoloTaskFromName(nameOrPath: string): YoloTask | null {
  const file = nameOrPath.split(/[\\/]/).pop() ?? '';
  const stem = file.toLowerCase().replace(/\.(pt|pth|onnx|engine|mlpackage)$/, '');
  if (!stem) return null;
  if (stem.endsWith('-seg')) return 'segment';
  if (stem.endsWith('-pose')) return 'pose';
  if (stem.endsWith('-obb')) return 'obb';
  if (stem.endsWith('-cls')) return 'classify';
  return stem.startsWith('yolo') || stem.startsWith('rtdetr') ? 'detect' : null;
}
