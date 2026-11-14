import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** Muss zu train_engine/plugins/time_series/manifest.json passen. */
export const TIME_SERIES_MODEL_TYPES = ['patchtst', 'ft_time_series'];

/**
 * PatchTST von Hugging Face (model_type "patchtst") oder ein in FrameTrain
 * angelegtes leeres Zeitreihenmodell ("ft_time_series").
 */
export function detectTimeSeries(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return TIME_SERIES_MODEL_TYPES.includes(modelType);
  const normalized = normalizePath(modelPathOrId);
  return containsToken(modelNameSegment(normalized), 'patchtst') || containsToken(normalized, 'patchtst');
}
