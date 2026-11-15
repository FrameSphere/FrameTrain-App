// HuggingFace-Detektoren (AutoModelForObjectDetection) – detect.ts
//
// Ultralytics-Gewichte (yolo11n.pt, rtdetr-l.pt) gehoeren zum YOLO-Plugin;
// hier landen Modelle im HF-Format mit config.json: DETR, Conditional/
// Deformable DETR, RT-DETR(v2) wie PekingU/rtdetr_r18vd, YOLOS, D-FINE.

import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** model_type-Werte — muss zu plugins/hf_object_detection/manifest.json passen (Test prueft das). */
export const HF_DETECTION_MODEL_TYPES = [
  'detr', 'conditional_detr', 'deformable_detr', 'rt_detr', 'rt_detr_v2', 'yolos',
  'dab_detr', 'd_fine', 'table-transformer',
];

const SUPPORTED = new Set(HF_DETECTION_MODEL_TYPES);

/** Ultralytics-Datei oder -Name (rtdetr-l.pt, yolov8n) — nicht hierher. */
function isUltralyticsName(normalized: string): boolean {
  const name = modelNameSegment(normalized);
  return /\.(pt|pth)$/.test(normalized) || /^rtdetr-[lx]\b/.test(name) || normalized.startsWith('ultralytics/');
}

export function detectHFObjectDetection(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    if (SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/-/g, '_'))) return true;
    const archs = (configJson?.architectures ?? []).map((a: string) => a.toLowerCase());
    return archs.some((a: string) => a.endsWith('forobjectdetection'));
  }

  const normalized = normalizePath(modelPathOrId);
  if (isUltralyticsName(normalized)) return false;
  const name = modelNameSegment(normalized);
  const tokens = ['detr', 'yolos', 'conditional-detr', 'deformable-detr', 'dab-detr', 'd-fine', 'dfine',
    'table-transformer'];
  if (tokens.some(t => containsToken(name, t) || containsToken(normalized, t))) return true;
  // RT-DETR im HF-Format: PekingU/rtdetr_r18vd, rtdetr_v2_r50vd
  return /(^|[^a-z0-9])rt[-_]?detr(_v2)?_r\d/.test(normalized);
}
