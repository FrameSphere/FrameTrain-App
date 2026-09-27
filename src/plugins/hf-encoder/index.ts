// HF Encoder Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectHFEncoder } from './detect';
import HFEncoderTestPlugin from './TestPlugin';

const hfEncoderPlugin: ModelPlugin = {
  id: 'hf-encoder',
  name: 'HF Encoder (Generic)',
  description: 'Sequence Classification für unterstützte HuggingFace Encoder-Modelle (BERT/RoBERTa/DeBERTa/...)',
  taskType: 'seq_classification',
  // "auto": Multi-Label, wenn die Label-Spalte Listen oder "a;b" enthaelt;
  // Regression bei Kommazahlen mit vielen Werten. Sonst Single-Label wie bisher.
  // problem_type: auto | single_label_classification | multi_label_classification | regression
  defaultPluginConfig: { multi_label: 'auto', problem_type: 'auto', threshold: 0.5 },
  detect: detectHFEncoder,
  TestComponent: HFEncoderTestPlugin,
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default hfEncoderPlugin;

