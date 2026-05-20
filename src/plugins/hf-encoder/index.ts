// HF Encoder Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectHFEncoder } from './detect';
import HFEncoderTrainPlugin from './TrainPlugin';
import HFEncoderTestPlugin from './TestPlugin';

const hfEncoderPlugin: ModelPlugin = {
  id: 'hf-encoder',
  name: 'HF Encoder (Generic)',
  description: 'Sequence Classification für unterstützte HuggingFace Encoder-Modelle (BERT/RoBERTa/DeBERTa/...)',
  taskType: 'seq_classification',
  defaultPluginConfig: {},
  detect: detectHFEncoder,
  TrainComponent: HFEncoderTrainPlugin,
  TestComponent: HFEncoderTestPlugin,
};

export default hfEncoderPlugin;

