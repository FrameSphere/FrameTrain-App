import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/**
 * model_type-Werte, die AutoModelForSemanticSegmentation laedt (identisch mit
 * supported_architectures in train_engine/plugins/semantic_segmentation).
 * Mask2Former/MaskFormer fehlen bewusst: sie trainieren mit Masken- und
 * Klassenlisten je Bild (Universal Segmentation), nicht mit Pixel-Labels.
 */
export const SEMSEG_MODEL_TYPES = [
  'segformer', 'upernet', 'dpt', 'beit', 'data2vec-vision',
  'mobilevit', 'mobilevitv2', 'mobilenet_v2',
];

/** Nur Segmentierung — bei diesen Typen reicht der model_type allein. */
const SEG_ONLY = new Set(['segformer', 'upernet']);

/**
 * Hinweise im Namen oder in architectures, dass ein BEiT/MobileViT/DPT ein
 * Segmentierer ist. Dieselben Typen gibt es als Klassifikator (BEiT, MobileViT)
 * und als Tiefenschaetzer (DPT) — der Typ allein entscheidet dort nicht.
 */
const SEG_HINT = /(forsemanticsegmentation|segformer|upernet|deeplab|segmentation|(^|[^a-z])seg([^a-z]|$)|ade20k|(^|[^a-z])ade([^a-z]|$)|cityscapes|pascal[-_]?voc|-voc|-coco-stuff)/i;
const DEPTH_HINT = /(depth|fordepthestimation|midas|dpt-hybrid-midas)/i;

export function detectSemanticSegmentation(modelPathOrId: string, configJson?: ModelConfig): boolean {
  const normalized = normalizePath(modelPathOrId);
  const archs = (configJson?.architectures ?? []).join(' ');
  const haystack = `${normalized} ${archs}`;
  if (DEPTH_HINT.test(haystack)) return false;

  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    const t = SEMSEG_MODEL_TYPES.includes(modelType) ? modelType : modelType.replace(/-/g, '_');
    if (!SEMSEG_MODEL_TYPES.includes(t)) return false;
    if (SEG_ONLY.has(t)) return true;
    return SEG_HINT.test(haystack);
  }

  // Ohne config.json: am Namen. SegFormer-Encoder heissen "mit-b0" … "mit-b5".
  const name = modelNameSegment(normalized);
  if (containsToken(name, 'segformer') || containsToken(name, 'upernet')) return true;
  if (/(^|\/)mit-b[0-5]$/.test(normalized)) return true;
  const seg = SEG_HINT.test(normalized);
  return seg && ['beit', 'dpt', 'mobilevit', 'deeplabv3', 'data2vec-vision', 'mobilenet']
    .some(tok => containsToken(name, tok) || containsToken(normalized, tok));
}
