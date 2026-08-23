import type { ModelPlugin } from '../types';
import { detectHFImageClassification } from './detect';
import HFImageTestPlugin from './TestPlugin';

const hfImageClassificationPlugin: ModelPlugin = {
  id: 'hf-image-classification',
  name: 'Image Classification (HuggingFace)',
  description: 'Trainiert das heruntergeladene HuggingFace-Bildmodell selbst (ViT, DeiT, ConvNeXt, Swin, ResNet).',
  taskType: 'hf_image_classification',
  // ViT/ConvNeXt-Feintuning laeuft ueblicherweise bei 5e-5.
  defaultTrainingConfig: { learning_rate: 5e-5, batch_size: 16, epochs: 5 },
  defaultPluginConfig: {},
  detect: detectHFImageClassification,
  TestComponent: HFImageTestPlugin,
  supportedDatasetTypes: ['folder_class', 'pre_split'],
  preferredDatasetType: 'folder_class',
};

export default hfImageClassificationPlugin;
