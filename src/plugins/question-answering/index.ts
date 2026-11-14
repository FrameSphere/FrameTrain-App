import type { ModelPlugin } from '../types';
import { detectQuestionAnswering } from './detect';
import QuestionAnsweringTestPlugin from './TestPlugin';

const questionAnsweringPlugin: ModelPlugin = {
  id: 'question-answering',
  name: 'Frage-Antwort (extraktiv)',
  description: 'Findet die Antwort auf eine Frage als Textstelle im Kontext (SQuAD-Stil, auch „keine Antwort“) mit BERT, RoBERTa, DeBERTa, XLM-R & Co.',
  taskType: 'question_answering',
  // Kontexte sind lang: 384 Tokens je Fenster, Ueberlappung 128 (doc_stride).
  defaultTrainingConfig: { learning_rate: 3e-5, batch_size: 12, epochs: 2, max_seq_length: 384 },
  // Leere Spaltennamen = automatisch erkennen. allow_no_answer "auto" schaltet
  // SQuAD-v2-Verhalten ein, sobald das Dataset Fragen ohne Antwort enthaelt.
  defaultPluginConfig: {
    context_column: '', question_column: '', answer_column: '', answer_start_column: '',
    doc_stride: 128, max_answer_length: 30, allow_no_answer: 'auto',
  },
  hiddenTrainingFields: ['lora', 'group_by_length', 'label_smoothing'],
  detect: detectQuestionAnswering,
  TestComponent: QuestionAnsweringTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard', 'unknown'],
  preferredDatasetType: 'pre_split',
};

export default questionAnsweringPlugin;
