// XLM-RoBERTa Plugin – Einstiegspunkt

import type { ModelPlugin } from '../types';
import { detectXLMRoberta } from './detect';
import XLMRobertaTrainPlugin from './TrainPlugin';
import XLMRobertaTestPlugin from './TestPlugin';

const xlmRobertaPlugin: ModelPlugin = {
  id: 'xlm-roberta',
  name: 'XLM-RoBERTa',
  description: 'Keyword Recognition & Sequence Classification mit XLM-RoBERTa base/large',
  taskType: 'seq_classification',
  defaultPluginConfig: {},
  detect: detectXLMRoberta,
  TrainComponent: XLMRobertaTrainPlugin,
  TestComponent: XLMRobertaTestPlugin,
};

export default xlmRobertaPlugin;
