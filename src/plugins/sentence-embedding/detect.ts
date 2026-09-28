// Sentence Embeddings – Erkennung
//
// Ein Sentence-Transformers-Modell ist architektonisch ein BERT/MPNet. Die
// sichere Markierung (modules.json, sentence_bert_config.json) liegt im
// Modellordner — die Plugin-Erkennung sieht aber nur model_type und Namen.
// Erkannt wird deshalb ueber die bekannten Modellfamilien im Namen; alles
// andere ordnet der Nutzer beim Import von Hand zu. Das Plugin steht vor
// hf-encoder, sonst landete all-MiniLM bei der Sequenzklassifikation.

import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../hf-encoder/detect';

// Nur Architekturen, die ohne trust_remote_code laden (nomic_bert, "new" von
// gte-v1.5 brauchen fremden Code und bleiben aussen vor).
const SUPPORTED = new Set<string>([...HF_ENCODER_SUPPORTED_MODEL_TYPES, 'modernbert']);

/** Familien, die fast nur als Embedding-Modell vorkommen (Wortgrenzen-Match auf den Namen). */
export const EMBEDDING_NAME_TOKENS = [
  'all-minilm', 'all-mpnet', 'mpnet-base-v2', 'paraphrase', 'multi-qa', 'msmarco',
  'bge', 'gte', 'e5', 'simcse', 'labse', 'sbert', 'stsb', 'embedding', 'embeddings', 'embed',
];

/** Organisationen, deren Modelle Embeddings sind. */
const EMBEDDING_ORGS = ['sentence-transformers', 'baai', 'thenlper', 'intfloat', 'nomic-ai', 'mixedbread-ai'];

/**
 * Cross-Encoder und Reranker bewerten ein Textpaar mit EINER Zahl — das ist
 * Sequenzklassifikation, kein Embedding (cross-encoder/stsb-roberta-base,
 * bge-reranker-base).
 */
const CROSS_ENCODER_TOKENS = ['cross-encoder', 'reranker', 'rerank'];

const NON_ENCODER_TOKENS = [
  'gpt', 'gpt2', 'llama', 'mistral', 'qwen', 'qwen2', 'phi', 'gemma', 'falcon', 't5', 'bart', 'whisper', 'clip',
];

/** Zeigt der Name selbst auf ein Embedding-Modell? */
export function hasEmbeddingHint(modelPathOrId: string): boolean {
  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (EMBEDDING_NAME_TOKENS.some(t => containsToken(name, t))) return true;
  // e5 steht meist als "e5-small-v2" / "multilingual-e5-base" im Namen.
  if (/(^|[^a-z0-9])e5-/.test(name)) return true;
  const segments = normalized.split('/').filter(Boolean);
  const org = segments.length >= 2 ? segments[segments.length - 2] : '';
  return EMBEDDING_ORGS.includes(org);
}

export function detectSentenceEmbedding(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType && !SUPPORTED.has(modelType)) return false;
  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (CROSS_ENCODER_TOKENS.some(t => containsToken(normalized, t))) return false;
  // "gte-Qwen2-7B-instruct" ist ein Decoder — das trainiert dieses Plugin nicht.
  if (NON_ENCODER_TOKENS.some(t => containsToken(name, t))) return false;
  return hasEmbeddingHint(modelPathOrId);
}
