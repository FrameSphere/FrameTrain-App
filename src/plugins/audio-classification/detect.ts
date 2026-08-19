import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

export const AUDIO_MODEL_TYPES = [
  'wav2vec2', 'wav2vec2-bert', 'hubert', 'wavlm', 'unispeech',
  'unispeech-sat', 'sew', 'sew-d', 'audio-spectrogram-transformer', 'whisper',
];

/** Audiomodelle, die keine Klassifikatoren sind (Sprachsynthese, Trennung). */
const NON_CLASSIFIER = ['speecht5', 'bark', 'musicgen', 'encodec', 'vits', 'seamless'];

const SUPPORTED = new Set(AUDIO_MODEL_TYPES);

export function detectAudioClassification(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    if (NON_CLASSIFIER.some(t => containsToken(modelType, t))) return false;
    return SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/_/g, '-'));
  }

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (NON_CLASSIFIER.some(t => containsToken(name, t) || containsToken(normalized, t))) return false;

  const tokens = ['wav2vec2', 'wav2vec', 'hubert', 'wavlm', 'unispeech', 'whisper', 'ast'];
  return tokens.some(t => containsToken(name, t) || containsToken(normalized, t));
}
