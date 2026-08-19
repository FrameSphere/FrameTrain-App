import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

export const SEQ2SEQ_MODEL_TYPES = [
  't5', 'mt5', 'longt5', 'bart', 'mbart', 'pegasus', 'marian', 'm2m_100', 'blenderbot',
];

const SUPPORTED = new Set(SEQ2SEQ_MODEL_TYPES);

export function detectSeq2Seq(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/-/g, '_'));

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  const tokens = ['flan-t5', 'mt5', 'longt5', 't5', 'mbart', 'bart', 'pegasus',
                  'marian', 'opus-mt', 'blenderbot'];
  return tokens.some(t => containsToken(name, t) || containsToken(normalized, t));
}
