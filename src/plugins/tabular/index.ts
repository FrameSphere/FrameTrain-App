import type { ModelPlugin } from '../types';
import { detectTabular } from './detect';
import TabularTestPlugin from './TestPlugin';

const tabularPlugin: ModelPlugin = {
  id: 'tabular',
  name: 'Tabellen-Klassifikation und -Regression',
  description: 'Sagt eine Spalte aus den übrigen voraus (CSV/Parquet). Zahlen werden normalisiert, Kategorien als Embeddings gelernt, fehlende Werte behandelt; Klassen oder Zahl werden automatisch erkannt.',
  taskType: 'tabular',
  defaultTrainingConfig: { learning_rate: 0.003, batch_size: 64, epochs: 30 },
  // Leere Zielspalte = label/target/y oder die letzte Spalte. task: auto | classification | regression
  defaultPluginConfig: {
    target_column: '', task: 'auto', hidden_layers: '128,64',
    val_ratio: 0.15, test_ratio: 0.15, patience: 10, device: 'cpu',
  },
  hiddenTrainingFields: [
    'max_seq_length', 'warmup_ratio', 'warmup_steps', 'lora', 'gradient_checkpointing',
    'group_by_length', 'scheduler',
  ],
  detect: detectTabular,
  TestComponent: TabularTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default tabularPlugin;
