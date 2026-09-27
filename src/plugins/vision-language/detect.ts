import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/**
 * model_type-Werte, die AutoModelForImageTextToText laedt (siehe
 * ft_data/vlm.py VLM_MODEL_TYPES). CLIP/SigLIP fehlen absichtlich: das sind
 * Embedding-Modelle, sie erzeugen keinen Text.
 */
export const VLM_MODEL_TYPES = [
  'idefics3', 'smolvlm', 'idefics2', 'qwen2_vl', 'qwen2_5_vl', 'qwen3_vl', 'paligemma',
  'blip', 'blip-2', 'blip_2', 'llava', 'llava_next', 'llava_onevision', 'florence2',
  'mllama', 'pixtral', 'aya_vision', 'internvl',
];

const SUPPORTED = new Set(VLM_MODEL_TYPES);

const NAME_TOKENS = [
  'smolvlm', 'idefics', 'idefics2', 'idefics3', 'qwen2-vl', 'qwen2.5-vl', 'qwen3-vl', 'paligemma',
  'paligemma2', 'llava', 'blip', 'blip2', 'florence-2', 'internvl', 'pixtral',
];

/** BLIP-Varianten ohne Textausgabe (Bild-Text-Abgleich) bleiben aussen vor. */
const NON_GENERATIVE = /blip-itm|blip-image-text-matching|clip|siglip/i;

export function detectVisionLanguage(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return SUPPORTED.has(modelType);

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (NON_GENERATIVE.test(name)) return false;
  return NAME_TOKENS.some(t => containsToken(name, t) || containsToken(normalized, t));
}
