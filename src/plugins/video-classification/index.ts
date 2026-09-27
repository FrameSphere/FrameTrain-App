import type { ModelPlugin } from '../types';
import { detectVideoClassification } from './detect';
import VideoTestPlugin from './TestPlugin';

const videoClassificationPlugin: ModelPlugin = {
  id: 'video-classification',
  name: 'Video Classification',
  description: 'Klassifiziert Videoclips mit VideoMAE, TimeSformer oder ViViT.',
  taskType: 'video_classification',
  // Ein Clip sind 16 Bilder durch einen Transformer — kleine Batches.
  defaultTrainingConfig: { learning_rate: 5e-5, batch_size: 2, epochs: 5 },
  // num_frames 0 = aus der config.json des Modells (VideoMAE: 16).
  defaultPluginConfig: { num_frames: 0, resume_from_checkpoint: false },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'lora_r', 'lora_alpha',
    'lora_dropout', 'lora_target_modules', 'load_in_4bit', 'load_in_8bit',
  ],
  detect: detectVideoClassification,
  TestComponent: VideoTestPlugin,
  supportedDatasetTypes: ['folder_class', 'pre_split'],
  preferredDatasetType: 'folder_class',
};

export default videoClassificationPlugin;
