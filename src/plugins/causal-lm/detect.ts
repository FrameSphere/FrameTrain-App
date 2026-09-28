import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** Decoder-LLMs, die AutoModelForCausalLM laedt (gleiche Liste wie das Manifest). */
export const CAUSAL_LM_MODEL_TYPES = [
  'llama', 'mistral', 'mixtral', 'qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe',
  'gemma', 'gemma2', 'gemma3', 'gemma3_text', 'phi', 'phi3', 'phimoe',
  'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'falcon', 'bloom', 'opt', 'mpt',
  'stablelm', 'olmo', 'olmo2', 'granite', 'cohere', 'starcoder2', 'smollm3',
];

const SUPPORTED = new Set(CAUSAL_LM_MODEL_TYPES);

/** Namensbestandteile ohne config.json. */
const NAME_TOKENS = [
  'llama', 'tinyllama', 'mistral', 'mixtral', 'qwen', 'qwen2', 'qwen2.5', 'qwen3',
  'gemma', 'phi', 'phi-2', 'phi-3', 'phi-4', 'smollm', 'smollm2', 'smollm3',
  'gpt2', 'distilgpt2', 'gpt-neo', 'gpt-j', 'gpt-neox', 'pythia', 'falcon', 'bloom',
  'bloomz', 'opt', 'mpt', 'stablelm', 'olmo', 'granite', 'starcoder2', 'deepseek-r1-distill',
];

/**
 * Varianten derselben Familien, die KEINE Text-LLMs sind oder sich so nicht
 * laden lassen: Vision-Language (Qwen2-VL, Llava), Embeddings, fertige
 * Quantisierungen fuer andere Laufzeiten (GGUF, AWQ, GPTQ).
 */
const EXCLUDE = ['vl', 'vision', 'llava', 'embedding', 'embed', 'gguf', 'awq', 'gptq', 'audio', 'omni'];

export function detectCausalLM(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/-/g, '_'));

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (EXCLUDE.some(t => containsToken(name, t))) return false;
  return NAME_TOKENS.some(t => containsToken(name, t) || containsToken(normalized, t));
}
