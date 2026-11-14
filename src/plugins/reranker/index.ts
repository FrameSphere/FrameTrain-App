import type { ModelPlugin } from '../types';
import { detectReranker } from './detect';
import RerankerTestPlugin from './TestPlugin';

const rerankerPlugin: ModelPlugin = {
  id: 'reranker',
  name: 'Reranker (Cross-Encoder)',
  description: 'Trainiert Cross-Encoder, die Anfrage und Passage gemeinsam lesen und ihre Relevanz bewerten (ms-marco-MiniLM, BGE-Reranker oder jeder BERT-Encoder). Zweite Stufe für Suche und RAG.',
  taskType: 'reranker',
  defaultTrainingConfig: { learning_rate: 2e-5, batch_size: 16, epochs: 2, max_seq_length: 256 },
  // Leere Felder = automatisch erkennen. negatives_per_query: so viele Passagen
  // anderer Anfragen kommen als Negative dazu, wenn das Dataset keine hat.
  defaultPluginConfig: {
    query_column: '', passage_column: '', score_column: '', positive_column: '', negative_column: '',
    negatives_per_query: 3,
  },
  hiddenTrainingFields: ['lora', 'group_by_length', 'label_smoothing'],
  detect: detectReranker,
  TestComponent: RerankerTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default rerankerPlugin;
