import type { ModelPlugin, ModelConfig } from '../types';
import ImageClassificationTrainPlugin from './TrainPlugin';
import ImageClassificationTestPlugin from './TestPlugin';

function detectImageClassification(modelPath: string, configJson?: ModelConfig): boolean {
  if (configJson && (configJson as Record<string, unknown>)['framework'] === 'torchvision') return true;
  const p = modelPath.toLowerCase();
  return p.includes('resnet') || p.includes('efficientnet') || p.includes('mobilenet') || p.includes('vit_b') || p.includes('vit-b');
}

const imageClassificationPlugin: ModelPlugin = {
  id: 'image-classification',
  name: 'Image Classification',
  description: 'Transfer Learning fuer Bildklassifikation: ResNet, EfficientNet, ViT, MobileNet. Erwartet einen Ordner pro Klasse (ImageFolder).',
  taskType: 'image_classification',
  defaultPluginConfig: { arch: 'resnet18', image_size: 224, freeze_base: true, unfreeze_at: -1, pretrained: true, augment: true },
  detect: detectImageClassification,
  TrainComponent: ImageClassificationTrainPlugin,
  TestComponent: ImageClassificationTestPlugin,
  supportedDatasetTypes: ['folder_class', 'pre_split'],
  preferredDatasetType: 'folder_class',
};

export default imageClassificationPlugin;
