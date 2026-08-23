import type { ModelPlugin, ModelConfig } from '../types';
import ImageClassificationTestPlugin from './TestPlugin';

/**
 * Architekturen, die zwar Bilder verarbeiten, aber keine Klassifikatoren sind.
 *
 * Ohne diese Sperre reichte ein Teilstring: `facebook/detr-resnet-50`
 * (Objekterkennung) matchte auf "resnet", `openai/clip-vit-base-patch32`
 * (multimodal) auf "vit-b". Beide wurden als trainierbar gemeldet und waeren
 * erst beim Trainingsstart gescheitert — nach dem Download.
 */
const NON_CLASSIFIER_IMAGE_TOKENS = [
  'detr', 'yolos', 'owlvit', 'owlv2', 'grounding-dino', 'dino-detr',
  'clip', 'siglip', 'blip', 'flava', 'align',
  'segformer', 'maskformer', 'mask2former', 'upernet', 'dpt', 'sam',
  'stable-diffusion', 'controlnet', 'vae',
];

/** Von torchvision tatsaechlich baubare Backbones (siehe Backend-Manifest). */
const TORCHVISION_TOKENS = [
  'resnet18', 'resnet50', 'resnet',
  'efficientnet', 'mobilenet', 'mobilenetv3',
  'vit_b_16', 'vit_b', 'vit-b', 'vit',
  'deit',
];

function containsToken(haystack: string, token: string): boolean {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i').test(haystack);
}

function detectImageClassification(modelPath: string, configJson?: ModelConfig): boolean {
  if (configJson && (configJson as Record<string, unknown>)['framework'] === 'torchvision') return true;

  // config.json ist massgeblich, wenn vorhanden.
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) {
    if (NON_CLASSIFIER_IMAGE_TOKENS.some(t => containsToken(modelType, t))) return false;
    return TORCHVISION_TOKENS.some(t => containsToken(modelType, t));
  }

  const p = modelPath.toLowerCase().replace(/\\/g, '/');
  // Der Modellname schlaegt den uebrigen Pfad — sonst gewinnt bei
  // "detr-resnet-50" das Wort "resnet" gegen die Architektur davor.
  if (NON_CLASSIFIER_IMAGE_TOKENS.some(t => containsToken(p, t))) return false;
  return TORCHVISION_TOKENS.some(t => containsToken(p, t));
}

const imageClassificationPlugin: ModelPlugin = {
  id: 'image-classification',
  name: 'Image Classification',
  description: 'Transfer Learning fuer Bildklassifikation: ResNet, EfficientNet, ViT, MobileNet. Erwartet einen Ordner pro Klasse (ImageFolder).',
  taskType: 'image_classification',
  // Nur der Klassifikationskopf wird trainiert — das vertraegt 1e-3.
  defaultTrainingConfig: { learning_rate: 1e-3, batch_size: 32, epochs: 10 },
  defaultPluginConfig: { arch: 'resnet18', image_size: 224, freeze_base: true, unfreeze_at: -1, pretrained: true, augment: true },
  detect: detectImageClassification,
  TestComponent: ImageClassificationTestPlugin,
  supportedDatasetTypes: ['folder_class', 'pre_split'],
  preferredDatasetType: 'folder_class',
};

export default imageClassificationPlugin;
