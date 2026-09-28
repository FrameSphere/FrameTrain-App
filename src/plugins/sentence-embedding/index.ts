import type { ModelPlugin } from '../types';
import { detectSentenceEmbedding } from './detect';
import SentenceEmbeddingTestPlugin from './TestPlugin';

const sentenceEmbeddingPlugin: ModelPlugin = {
  id: 'sentence-embedding',
  name: 'Sentence Embeddings (Suche/RAG)',
  description: 'Trainiert Embedding-Modelle für semantische Suche, RAG und Ähnlichkeit (all-MiniLM, BGE, E5, GTE, MPNet oder jeder BERT-Encoder).',
  taskType: 'sentence_embedding',
  // In-Batch-Negative: groessere Batches geben mehr Gegenbeispiele pro Schritt.
  defaultTrainingConfig: { learning_rate: 2e-5, batch_size: 32, epochs: 3 },
  // Leere Felder = automatisch erkennen. Die Praefixe braucht z.B. E5
  // ("query: " / "passage: ").
  defaultPluginConfig: {
    anchor_column: '', positive_column: '', negative_column: '',
    query_prefix: '', document_prefix: '',
  },
  // Label Smoothing gibt es fuer Kontrastiv-Losses nicht.
  hiddenTrainingFields: ['lora', 'group_by_length', 'label_smoothing'],
  detect: detectSentenceEmbedding,
  TestComponent: SentenceEmbeddingTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default sentenceEmbeddingPlugin;
