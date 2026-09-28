// Token-Klassifikation (NER, POS) – Erkennung
//
// Schwierigkeit: ein NER-Modell ist architektonisch ein BERT/RoBERTa wie jedes
// andere. Beim Import kennt die App nur model_type ("bert") und den Namen —
// daran erkennt hf-encoder zu Recht jedes BERT als Sequenzklassifikation.
// Dieses Plugin greift deshalb nur bei einem eindeutigen Signal:
//   1. config.architectures endet auf ...ForTokenClassification, oder
//   2. der Modellname nennt die Aufgabe (ner, pos, token-classification ...).
// Ein Basis-BERT ohne Hinweis bleibt bei der Sequenzklassifikation; wer damit
// NER trainieren will, waehlt das Plugin beim Import von Hand.

import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../hf-encoder/detect';

/** Encoder, die AutoModelForTokenClassification laden kann (Backend-Manifest). */
export const TOKEN_CLASSIFICATION_MODEL_TYPES: string[] = [
  ...HF_ENCODER_SUPPORTED_MODEL_TYPES,
  'longformer', 'mobilebert', 'convbert', 'modernbert',
];

const SUPPORTED = new Set(TOKEN_CLASSIFICATION_MODEL_TYPES);

/** Namensbestandteile, die die Aufgabe verraten. Nur an Wortgrenzen. */
export const TOKEN_TASK_TOKENS = [
  'ner', 'pos', 'token-classification', 'token-class', 'tokenclassification',
  'conll', 'conll03', 'conll2003', 'wikiann', 'wikineural', 'upos', 'pos-tagger', 'postag',
];

/** Decoder & Co.: "ner" im Namen macht aus einem Llama kein NER-Encoder-Modell. */
const NON_ENCODER_TOKENS = [
  'gpt', 'gpt2', 'llama', 'mistral', 'qwen', 'phi', 'gemma', 'falcon', 't5', 'bart', 'whisper',
];

export function hasTokenTaskHint(modelPathOrId: string): boolean {
  const name = modelNameSegment(normalizePath(modelPathOrId));
  return TOKEN_TASK_TOKENS.some(t => containsToken(name, t));
}

export function detectTokenClassification(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  // Ist die Architektur bekannt und kein Encoder, hilft auch ein Name nicht.
  if (modelType && !SUPPORTED.has(modelType)) return false;

  const archs = Array.isArray(configJson?.architectures) ? configJson!.architectures! : [];
  if (archs.some(a => typeof a === 'string' && a.endsWith('ForTokenClassification'))) return true;

  const name = modelNameSegment(normalizePath(modelPathOrId));
  if (NON_ENCODER_TOKENS.some(t => containsToken(name, t))) return false;
  return hasTokenTaskHint(modelPathOrId);
}
