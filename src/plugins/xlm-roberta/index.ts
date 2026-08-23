// XLM-RoBERTa Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectXLMRoberta } from './detect';
import XLMRobertaTestPlugin from './TestPlugin';

const xlmRobertaPlugin: ModelPlugin = {
  id: 'xlm-roberta',
  name: 'XLM-RoBERTa',
  description: 'Keyword Recognition & Sequence Classification mit XLM-RoBERTa base/large',
  taskType: 'seq_classification',
  defaultPluginConfig: {},
  detect: detectXLMRoberta,
  TestComponent: XLMRobertaTestPlugin,
  // Phase 7: Dataset-Kompatibilität
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default xlmRobertaPlugin;
