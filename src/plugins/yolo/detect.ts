// YOLO Object Detection Plugin – detect.ts
// Erkennt YOLOv5/v8/v9/v11 Modelle anhand model_type, config oder Modell-ID

import type { ModelConfig } from '../types';

/** model_type-Werte fuer YOLO — muss zu plugins/yolo/manifest.json passen (Test prueft das). */
export const YOLO_MODEL_TYPES = ['yolov5', 'yolov8', 'yolov9', 'yolo11', 'yolo'];

export function detectYOLO(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // config.json: model_type = "yolo" oder architecture-Hinweis
  if (configJson) {
    const mt = configJson.model_type?.toLowerCase() ?? '';
    if (YOLO_MODEL_TYPES.includes(mt)) return true;
    const archs = (configJson.architectures ?? []).map((a: string) => a.toLowerCase());
    if (archs.some((a: string) => a.includes('yolo'))) return true;
  }

  // Modell-ID / Pfad Heuristik
  const id = modelPathOrId.toLowerCase();
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
  return stem.startsWith('yolo') ? 'detect' : null;
}
