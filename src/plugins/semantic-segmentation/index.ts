import type { ModelPlugin } from '../types';
import { detectSemanticSegmentation } from './detect';
import SemanticSegmentationTestPlugin from './TestPlugin';

const semanticSegmentationPlugin: ModelPlugin = {
  id: 'semantic-segmentation',
  name: 'Semantic Segmentation (HuggingFace)',
  description: 'Trainiert HuggingFace-Segmentierungsmodelle (SegFormer, UperNet, DPT, BEiT, MobileViT) auf Bild + Maske; Kennzahlen mIoU und Pixel-Accuracy.',
  taskType: 'semantic_segmentation',
  // SegFormer-Feintuning: 6e-5 (Paper), kleine Batches wegen der Bildgroesse.
  defaultTrainingConfig: { learning_rate: 6e-5, batch_size: 4, epochs: 10 },
  // image_size: Trainingsgroesse in px (0 = wie der Bild-Prozessor, meist 512).
  defaultPluginConfig: { image_size: 512, augment: true, resume_from_checkpoint: false },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'lora_r', 'lora_alpha',
    'lora_dropout', 'lora_target_modules', 'load_in_4bit', 'load_in_8bit',
  ],
  detect: detectSemanticSegmentation,
  TestComponent: SemanticSegmentationTestPlugin,
  // image_mask: images/ + masks/; pre_split: train/images + train/masks (App-Split).
  supportedDatasetTypes: ['image_mask', 'pre_split'],
  preferredDatasetType: 'image_mask',
};

export default semanticSegmentationPlugin;
