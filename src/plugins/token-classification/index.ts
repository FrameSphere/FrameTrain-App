import type { ModelPlugin } from '../types';
import { detectTokenClassification } from './detect';
import TokenClassificationTestPlugin from './TestPlugin';

const tokenClassificationPlugin: ModelPlugin = {
  id: 'token-classification',
  name: 'Token Classification (NER/POS)',
  description: 'Ein Label pro Wort: Named Entity Recognition (Personen, Orte, Firmen) oder Wortarten, mit BERT, RoBERTa, DeBERTa & Co.',
  taskType: 'token_classification',
  // Kleine Datensaetze, ein frischer Klassifikationskopf je Token — etwas
  // mehr Lernrate und Epochen als bei der Sequenzklassifikation.
  defaultTrainingConfig: { learning_rate: 5e-5, batch_size: 16, epochs: 5 },
  // Leere Felder = automatisch erkennen (tokens/words, ner_tags/labels/tags).
  defaultPluginConfig: { tokens_column: '', tags_column: '', label_all_tokens: false },
  // LoRA und Quantisierung wertet das Plugin nicht aus.
  hiddenTrainingFields: ['lora', 'group_by_length'],
  detect: detectTokenClassification,
  TestComponent: TokenClassificationTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard', 'unknown'],
  preferredDatasetType: 'pre_split',
};

export default tokenClassificationPlugin;
