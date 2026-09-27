import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

export const AUDIO_MODEL_TYPES = [
  'wav2vec2', 'wav2vec2-bert', 'hubert', 'wavlm', 'unispeech',
  'unispeech-sat', 'sew', 'sew-d', 'audio-spectrogram-transformer',
];

/** Audiomodelle, die keine Klassifikatoren sind (Sprachsynthese, Trennung). */
const NON_CLASSIFIER = ['speecht5', 'bark', 'musicgen', 'encodec', 'vits', 'seamless'];

const SUPPORTED = new Set(AUDIO_MODEL_TYPES);

/**
 * Seit dem ASR-Plugin gilt: Whisper und *ForCTC sind Spracherkenner und gehoeren
 * zu speech-recognition. Vorher stand whisper hier in der Liste — ein
 * openai/whisper-tiny landete in der Audio-Klassifikation und bekam einen
 * zufaelligen Klassifikationskopf statt Transkripte. Whisper als Klassifikator
 * (WhisperForAudioClassification) bleibt ueber die Architektur erreichbar.
 */
export function detectAudioClassification(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const archs = (configJson?.architectures ?? []).filter((a): a is string => typeof a === 'string');
  if (archs.some(a => a.endsWith('ForAudioClassification'))) return true;
  if (archs.some(a => a.endsWith('ForCTC') || a.endsWith('ForSpeechSeq2Seq')
    || a.endsWith('ForConditionalGeneration'))) return false;

  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    if (NON_CLASSIFIER.some(t => containsToken(modelType, t))) return false;
    return SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/_/g, '-'));
  }

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (NON_CLASSIFIER.some(t => containsToken(name, t) || containsToken(normalized, t))) return false;

  const tokens = ['wav2vec2', 'wav2vec', 'hubert', 'wavlm', 'unispeech', 'ast'];
  return tokens.some(t => containsToken(name, t) || containsToken(normalized, t));
}
