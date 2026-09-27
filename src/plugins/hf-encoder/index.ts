// HF Encoder Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectHFEncoder } from './detect';
import HFEncoderTestPlugin from './TestPlugin';

const hfEncoderPlugin: ModelPlugin = {
  id: 'hf-encoder',
  name: 'HF Encoder (Generic)',
  description: 'Sequence Classification für unterstützte HuggingFace Encoder-Modelle (BERT/RoBERTa/DeBERTa/...)',
  taskType: 'seq_classification',
  // resume_from_checkpoint: nach "Stoppen" die gespeicherte Version waehlen und einschalten
  // (oder einen Checkpoint-Pfad eintragen) - das Training laeuft ab dem letzten Schritt weiter.
  defaultPluginConfig: { resume_from_checkpoint: false },
  detect: detectHFEncoder,
  TestComponent: HFEncoderTestPlugin,
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default hfEncoderPlugin;

