import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** Videoklassifikatoren, die AutoModelForVideoClassification laedt. */
export const VIDEO_MODEL_TYPES = ['videomae', 'timesformer', 'vivit'];

export function detectVideoClassification(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return VIDEO_MODEL_TYPES.includes(modelType);
  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  return VIDEO_MODEL_TYPES.some(t => containsToken(name, t) || containsToken(normalized, t));
}
