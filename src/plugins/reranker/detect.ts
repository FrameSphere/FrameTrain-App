// Reranker / Cross-Encoder – Erkennung
//
// Ein Cross-Encoder ist ein BERT/RoBERTa mit einem Kopf, der EINE Zahl
// ausgibt. Erkannt wird er am Namen (cross-encoder/…, bge-reranker, …-rerank)
// — dieselben Merkmale, mit denen sentence-embedding solche Modelle von sich
// weist. Decoder-Reranker (Qwen3-Reranker) trainiert dieses Plugin nicht.

import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../hf-encoder/detect';

const SUPPORTED = new Set<string>([...HF_ENCODER_SUPPORTED_MODEL_TYPES, 'modernbert']);

/** Merkmale im Namen oder in der Organisation (cross-encoder/ms-marco-MiniLM-L-6-v2). */
export const RERANKER_TOKENS = ['cross-encoder', 'crossencoder', 'reranker', 'rerank', 're-ranker'];

const NON_ENCODER_TOKENS = [
  'gpt', 'gpt2', 'llama', 'mistral', 'qwen', 'qwen2', 'qwen3', 'phi', 'gemma', 'falcon', 't5', 'bart', 'whisper', 'clip',
];

export function hasRerankerHint(modelPathOrId: string): boolean {
  const normalized = normalizePath(modelPathOrId);
  return RERANKER_TOKENS.some(t => containsToken(normalized, t));
}

export function detectReranker(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType && !SUPPORTED.has(modelType)) return false;
  const name = modelNameSegment(normalizePath(modelPathOrId));
  if (NON_ENCODER_TOKENS.some(t => containsToken(name, t))) return false;
  return hasRerankerHint(modelPathOrId);
}
