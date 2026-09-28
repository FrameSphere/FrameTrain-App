import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** Encoder-Decoder fuer Spracherkennung: der Decoder schreibt den Text. */
export const ASR_SEQ2SEQ_MODEL_TYPES = [
  'whisper', 'moonshine', 'speech_to_text', 'speech-to-text', 'speech-encoder-decoder',
];

/**
 * Audio-Encoder, die mit CTC-Kopf Text ausgeben. Derselbe model_type kann auch
 * ein Klassifikator sein (wav2vec2 + ForSequenceClassification) — ohne
 * Architektur entscheidet deshalb der Name.
 */
export const CTC_MODEL_TYPES = [
  'wav2vec2', 'wav2vec2-bert', 'wav2vec2-conformer', 'hubert', 'wavlm', 'data2vec-audio',
  'unispeech', 'unispeech-sat', 'sew', 'sew-d',
];

/** Namensteile, die einen Audio-Encoder als Spracherkenner ausweisen. */
const ASR_NAME_TOKENS = [
  '960h', 'asr', 'ctc', 'stt', 'librispeech', 'commonvoice', 'common-voice', 'speech-recognition',
];

const SEQ2SEQ_NAME_TOKENS = ['whisper', 'moonshine', 's2t', 'speech-to-text', 'distil-whisper'];
const CTC_NAME_TOKENS = ['wav2vec2', 'wav2vec', 'hubert', 'wavlm', 'data2vec-audio', 'unispeech'];

const norm = (t: string) => t.toLowerCase().replace(/_/g, '-');

/** Architekturen, die sicher Spracherkennung sind bzw. sicher nicht. */
function byArchitecture(configJson?: ModelConfig): boolean | undefined {
  const archs = (configJson?.architectures ?? []).filter((a): a is string => typeof a === 'string');
  if (archs.length === 0) return undefined;
  if (archs.some(a => a.endsWith('ForCTC') || a.endsWith('ForSpeechSeq2Seq'))) return true;
  if (archs.some(a => a.endsWith('ForAudioClassification') || a.endsWith('ForSequenceClassification')
    || a.endsWith('ForAudioFrameClassification') || a.endsWith('ForXVector'))) return false;
  const modelType = norm(configJson?.model_type ?? '');
  if (ASR_SEQ2SEQ_MODEL_TYPES.map(norm).includes(modelType)
    && archs.some(a => a.endsWith('ForConditionalGeneration'))) return true;
  return undefined;
}

function nameSaysAsr(modelPathOrId: string): boolean {
  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  const has = (t: string) => containsToken(name, t) || containsToken(normalized, t);
  return ASR_NAME_TOKENS.some(has);
}

export function detectSpeechRecognition(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const arch = byArchitecture(configJson);
  if (arch !== undefined) return arch;

  const modelType = configJson?.model_type ? norm(configJson.model_type) : '';
  if (modelType) {
    if (ASR_SEQ2SEQ_MODEL_TYPES.map(norm).includes(modelType)) return true;
    if (CTC_MODEL_TYPES.includes(modelType)) return nameSaysAsr(modelPathOrId);
    return false;
  }

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  const has = (t: string) => containsToken(name, t) || containsToken(normalized, t);
  if (SEQ2SEQ_NAME_TOKENS.some(has)) return true;
  // "asr" allein im Namen genuegt ohne weitere Angaben (z.B. "mein-asr-modell").
  return (CTC_NAME_TOKENS.some(has) && nameSaysAsr(modelPathOrId))
    || ['asr', 'speech-recognition'].some(has);
}
