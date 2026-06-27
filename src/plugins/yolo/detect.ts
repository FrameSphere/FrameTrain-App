// YOLO Object Detection Plugin – detect.ts
// Erkennt YOLOv5/v8/v9/v11 Modelle anhand model_type, config oder Modell-ID

import type { ModelConfig } from '../types';

export function detectYOLO(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // config.json: model_type = "yolo" oder architecture-Hinweis
  if (configJson) {
    const mt = configJson.model_type?.toLowerCase() ?? '';
    if (mt === 'yolo' || mt === 'yolov5' || mt === 'yolov8' || mt === 'yolov9' || mt === 'yolo11') return true;
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
