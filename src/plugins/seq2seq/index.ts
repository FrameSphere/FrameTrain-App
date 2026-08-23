import type { ModelPlugin } from '../types';
import { detectSeq2Seq } from './detect';
import Seq2SeqTestPlugin from './TestPlugin';

const seq2seqPlugin: ModelPlugin = {
  id: 'seq2seq',
  name: 'Seq2Seq (T5/BART)',
  description: 'Zusammenfassung, Übersetzung und Textumformung mit Encoder-Decoder-Modellen.',
  taskType: 'seq2seq',
  // T5 & Co. brauchen deutlich hoehere Lernraten als Encoder-Modelle.
  defaultTrainingConfig: { learning_rate: 3e-4, batch_size: 8, epochs: 3 },
  defaultPluginConfig: { max_target_length: 128 },
  detect: detectSeq2Seq,
  TestComponent: Seq2SeqTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default seq2seqPlugin;
