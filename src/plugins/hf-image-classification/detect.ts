import type { ModelConfig } from '../types';
import { containsToken, modelNameSegment, normalizePath } from '../modelTokens';

/** model_type-Werte, die AutoModelForImageClassification bedienen kann. */
export const HF_IMAGE_MODEL_TYPES = [
  'vit', 'deit', 'beit', 'swin', 'swinv2', 'convnext', 'convnextv2',
  'resnet', 'regnet', 'efficientnet', 'mobilenet_v2', 'mobilevit',
  'levit', 'poolformer', 'dinat', 'cvt',
];

/** Bildmodelle, die keine Klassifikatoren sind. */
const NON_CLASSIFIER = [
  'detr', 'yolos', 'owlvit', 'owlv2', 'grounding-dino',
  'clip', 'siglip', 'blip', 'flava', 'align',
  'segformer', 'maskformer', 'mask2former', 'upernet', 'dpt', 'sam',
  'stable-diffusion', 'controlnet', 'vae',
];

const SUPPORTED = new Set(HF_IMAGE_MODEL_TYPES);

export function detectHFImageClassification(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // Ein torchvision-Modell gehört zum alten Plugin, nicht hierher.
  if (configJson && (configJson as Record<string, unknown>)['framework'] === 'torchvision') return false;

  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    if (NON_CLASSIFIER.some(t => containsToken(modelType, t))) return false;
    return SUPPORTED.has(modelType) || SUPPORTED.has(modelType.replace(/-/g, '_'));
  }

  const normalized = normalizePath(modelPathOrId);
  const name = modelNameSegment(normalized);
  if (NON_CLASSIFIER.some(t => containsToken(name, t) || containsToken(normalized, t))) return false;

  // timm ist ein reiner Bildmodell-Hub; transformers laedt diese Modelle ueber
  // TimmWrapperForImageClassification. Geprueft mit mobilenetv3_small_100.
  if (/^timm\//.test(normalized)) return true;

  const nameTokens = ['vit', 'deit', 'beit', 'swin', 'convnext', 'resnet', 'regnet',
    'efficientnet', 'mobilenet', 'mobilevit', 'levit', 'poolformer', 'cvt'];
  if (nameTokens.some(t => containsToken(name, t) || containsToken(normalized, t))) return true;

  // timm-Schreibweise haengt die Version direkt an: mobilenetv3, efficientnetv2,
  // convnextv2. Die Wortgrenze greift dort nicht, deshalb dieser Zusatz.
  return /(^|[^a-z])(mobilenet|efficientnet|convnext|regnet|resnext|resnet)v?\d/i.test(normalized);
}
