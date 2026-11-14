import type { ModelPlugin } from '../types';
import { detectTimeSeries } from './detect';
import TimeSeriesTestPlugin from './TestPlugin';

const timeSeriesPlugin: ModelPlugin = {
  id: 'time-series',
  name: 'Zeitreihen-Vorhersage (PatchTST)',
  description: 'Sagt die nächsten Werte einer oder mehrerer Zeitreihen voraus (CSV/Parquet mit Zeitspalte). PatchTST neu angelegt oder von Hugging Face; zeitliche Aufteilung, Vergleich mit naiver Vorhersage.',
  taskType: 'time_series',
  defaultTrainingConfig: { learning_rate: 0.001, batch_size: 32, epochs: 20 },
  // Leere Spaltennamen = automatisch erkennen. season_length 0 = letzter Wert als Vergleich.
  defaultPluginConfig: {
    time_column: '', target_columns: '', id_column: '',
    context_length: 64, prediction_length: 16, season_length: 0,
    val_ratio: 0.1, test_ratio: 0.2, patience: 5,
    d_model: 64, num_layers: 2, from_scratch: false, device: 'auto',
  },
  hiddenTrainingFields: [
    'max_seq_length', 'warmup_ratio', 'warmup_steps', 'lora', 'gradient_checkpointing',
    'group_by_length', 'label_smoothing', 'scheduler',
  ],
  detect: detectTimeSeries,
  TestComponent: TimeSeriesTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default timeSeriesPlugin;
