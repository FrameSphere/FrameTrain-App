// XLM-RoBERTa Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectXLMRoberta } from './detect';
import XLMRobertaTestPlugin from './TestPlugin';

const xlmRobertaPlugin: ModelPlugin = {
  id: 'xlm-roberta',
  name: 'XLM-RoBERTa',
  description: 'Keyword Recognition & Sequence Classification mit XLM-RoBERTa base/large',
  taskType: 'seq_classification',
  // resume_from_checkpoint: nach "Stoppen" die gespeicherte Version waehlen und einschalten
  // (oder einen Checkpoint-Pfad eintragen) - das Training laeuft ab dem letzten Schritt weiter.
  // multi_label/problem_type "auto": Multi-Label, wenn die Label-Spalte Listen oder "a;b"
  // enthaelt; Regression bei Kommazahlen mit vielen Werten. Sonst Single-Label wie bisher.
  // problem_type: auto | single_label_classification | multi_label_classification | regression
  defaultPluginConfig: { multi_label: 'auto', problem_type: 'auto', threshold: 0.5, resume_from_checkpoint: false },
  detect: detectXLMRoberta,
  TestComponent: XLMRobertaTestPlugin,
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default xlmRobertaPlugin;
