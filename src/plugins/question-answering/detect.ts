// Extraktives Frage-Antwort (SQuAD) – Erkennung
//
// Ein QA-Modell ist architektonisch ein BERT/RoBERTa wie jedes andere. Erkannt
// wird es deshalb nur an einem eindeutigen Signal:
//   1. config.architectures endet auf ...ForQuestionAnswering, oder
//   2. der Modellname nennt die Aufgabe (squad, squad2, question-answering, qa).
// "qa" steht auch in Embedding-Namen (multi-qa-MiniLM) — die bleiben beim
// Embedding-Plugin. Ein Basis-BERT ohne Hinweis waehlt der Nutzer von Hand.

import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../hf-encoder/detect';
import { hasEmbeddingHint } from '../sentence-embedding/detect';

/** Encoder, die AutoModelForQuestionAnswering laden kann (Backend-Manifest). */
export const QA_MODEL_TYPES: string[] = [
  ...HF_ENCODER_SUPPORTED_MODEL_TYPES, 'modernbert', 'mobilebert', 'longformer', 'splinter',
];

const SUPPORTED = new Set(QA_MODEL_TYPES);

/** Namensbestandteile, die die Aufgabe verraten (Wortgrenzen). */
export const QA_TASK_TOKENS = [
  'squad', 'squad2', 'squad_v2', 'squadv2', 'squad-v2', 'question-answering', 'questionanswering',
  'extractive-qa', 'qa', 'mlqa', 'xquad', 'germanquad', 'tydiqa',
];

const NON_ENCODER_TOKENS = [
  'gpt', 'gpt2', 'llama', 'mistral', 'qwen', 'phi', 'gemma', 'falcon', 't5', 'bart', 'whisper',
];

export function hasQaTaskHint(modelPathOrId: string): boolean {
  const name = modelNameSegment(normalizePath(modelPathOrId));
  return QA_TASK_TOKENS.some(t => containsToken(name, t));
}

export function detectQuestionAnswering(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType && !SUPPORTED.has(modelType)) return false;

  const archs = Array.isArray(configJson?.architectures) ? configJson!.architectures! : [];
  if (archs.some(a => typeof a === 'string' && a.endsWith('ForQuestionAnswering'))) return true;

  const name = modelNameSegment(normalizePath(modelPathOrId));
  if (NON_ENCODER_TOKENS.some(t => containsToken(name, t))) return false;
  if (hasEmbeddingHint(modelPathOrId)) return false;
  return hasQaTaskHint(modelPathOrId);
}
