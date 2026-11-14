import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** Trainierbar: SpeechT5 als Text-to-Speech (microsoft/speecht5_tts). */
export const TTS_TRAINABLE_MODEL_TYPES = ['speecht5'];

/**
 * Nur Test, Labor und Hosting: VITS / MMS-TTS (facebook/mms-tts-*).
 * transformers bietet fuer VITS keine Trainingsschleife — das Train-Plugin
 * lehnt solche Modelle mit Begruendung ab.
 */
export const TTS_INFERENCE_ONLY_MODEL_TYPES = ['vits'];

const norm = (t: string) => t.toLowerCase().replace(/_/g, '-');

/** Architekturen, die sicher Sprachausgabe sind bzw. sicher nicht. */
function byArchitecture(configJson?: ModelConfig): boolean | undefined {
  const archs = (configJson?.architectures ?? []).filter((a): a is string => typeof a === 'string');
  if (archs.length === 0) return undefined;
  if (archs.some(a => a.endsWith('ForTextToSpeech') || a === 'VitsModel')) return true;
  // SpeechT5 gibt es auch als Spracherkenner und als Stimmwandler — gleicher
  // model_type, andere Aufgabe.
  if (archs.some(a => a.endsWith('ForSpeechToText') || a.endsWith('ForSpeechToSpeech'))) return false;
  return undefined;
}

/** true, wenn das Modell zwar abspielbar, aber nicht trainierbar ist (VITS / MMS-TTS). */
export function isInferenceOnlyTts(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type ? norm(configJson.model_type) : '';
  if (modelType) return TTS_INFERENCE_ONLY_MODEL_TYPES.includes(modelType);
  if ((configJson?.architectures ?? []).includes('VitsModel')) return true;
  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  return ['vits', 'mms-tts'].some(t => containsToken(name, t) || containsToken(normalized, t));
}

export function detectTextToSpeech(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const arch = byArchitecture(configJson);
  if (arch !== undefined) return arch;

  const modelType = configJson?.model_type ? norm(configJson.model_type) : '';
  if (modelType) {
    return TTS_TRAINABLE_MODEL_TYPES.includes(modelType) || TTS_INFERENCE_ONLY_MODEL_TYPES.includes(modelType);
  }

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  const has = (t: string) => containsToken(name, t) || containsToken(normalized, t);
  if (has('vits') || has('mms-tts')) return true;
  // "speecht5" allein reicht nicht: speecht5_asr und speecht5_vc sind andere Aufgaben.
  const family = ['speecht5', 'speech-t5', 'speecht5-tts'].some(has) || /speecht5/.test(name);
  if (family) return has('tts');
  return ['tts', 'text-to-speech'].some(t => containsToken(name, t));
}
