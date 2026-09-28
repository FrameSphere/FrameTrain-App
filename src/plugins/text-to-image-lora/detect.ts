import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/**
 * Pipeline-Klassen aus model_index.json, die das LoRA-Training kann.
 * FrameTrain meldet bei diffusers-Ordnern `_class_name` als model_type
 * (Rust detect_model_type); aeltere Importe tragen noch "diffusion".
 */
export const T2I_PIPELINE_CLASSES = ['stablediffusionpipeline', 'stablediffusionxlpipeline'];

/** Namen, die sicher eine SD-1.x/2.x/SDXL-Pipeline meinen. */
const NAME_TOKENS = [
  'stable-diffusion', 'stable-diffusion-xl', 'sdxl', 'sdxl-turbo', 'sd-turbo', 'tiny-sd', 'small-sd',
  'bk-sdm-tiny', 'bk-sdm-small', 'bk-sdm-base', 'tiny-stable-diffusion-pipe', 'tiny-stable-diffusion-xl-pipe',
];

/** Andere Diffusion-Architekturen (Transformer statt UNet) — kein LoRA-Pfad hier. */
const OTHER_DIFFUSION = /stable-diffusion-3|(^|[^a-z0-9])sd3([^a-z0-9]|$)|flux|pixart|kandinsky|hunyuan|sana|auraflow|wan2|ltx-video|cogvideo|playground-v2/i;

export function detectTextToImageLora(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const cls = typeof configJson?.['_class_name'] === 'string'
    ? (configJson['_class_name'] as string).toLowerCase()
    : undefined;
  if (cls) return T2I_PIPELINE_CLASSES.includes(cls);

  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return T2I_PIPELINE_CLASSES.includes(modelType) || modelType === 'diffusion';

  const normalized = normalizePath(modelPathOrId);
  if (OTHER_DIFFUSION.test(normalized)) return false;
  const name = modelNameSegment(normalized);
  return NAME_TOKENS.some(t => containsToken(name, t) || containsToken(normalized, t));
}
