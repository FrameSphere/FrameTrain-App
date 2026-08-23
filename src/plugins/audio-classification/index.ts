import type { ModelPlugin } from '../types';
import { detectAudioClassification } from './detect';
import AudioTestPlugin from './TestPlugin';

const audioClassificationPlugin: ModelPlugin = {
  id: 'audio-classification',
  name: 'Audio Classification',
  description: 'Klassifiziert Audio mit Wav2Vec2, HuBERT, WavLM, AST oder dem Whisper-Encoder.',
  taskType: 'audio_classification',
  // Audio-Encoder sind empfindlich; kleine Batches wegen langer Sequenzen.
  defaultTrainingConfig: { learning_rate: 3e-5, batch_size: 4, epochs: 5 },
  defaultPluginConfig: { max_seconds: 10 },
  detect: detectAudioClassification,
  TestComponent: AudioTestPlugin,
  supportedDatasetTypes: ['folder_class', 'pre_split'],
  preferredDatasetType: 'folder_class',
};

export default audioClassificationPlugin;
