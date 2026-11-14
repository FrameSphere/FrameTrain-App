import type { ModelPlugin } from '../types';
import { detectTextToSpeech } from './detect';
import TextToSpeechTestPlugin from './TestPlugin';

const textToSpeechPlugin: ModelPlugin = {
  id: 'text-to-speech',
  name: 'Text-to-Speech (TTS)',
  description: 'Sprachausgabe: trainiert SpeechT5 auf Audio + Transkript auf eine eigene Stimme. VITS / MMS-TTS lässt sich testen und hosten, aber nicht trainieren. Misst die Verständlichkeit per Rücktranskription (WER/CER).',
  taskType: 'text_to_speech',
  // SpeechT5 verlernt bei hohen Lernraten die Aussprache; kleine Batches
  // reichen, weil jede Aufnahme viele Spektrogramm-Frames liefert.
  defaultTrainingConfig: { learning_rate: 1e-5, batch_size: 4, epochs: 30, gradient_accumulation_steps: 2 },
  // speaker_embedding auto = x-vector mit speechbrain, sonst Standardvektor.
  defaultPluginConfig: {
    speaker_embedding: 'auto',
    language: '',
    max_seconds: 15,
    sample_text: '',
  },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'lora_r', 'lora_alpha',
    'lora_dropout', 'lora_target_modules', 'load_in_4bit', 'load_in_8bit', 'label_smoothing',
  ],
  detect: detectTextToSpeech,
  TestComponent: TextToSpeechTestPlugin,
  // Dieselben Daten wie die Spracherkennung: Aufnahme + Transkript.
  supportedDatasetTypes: ['audio_transcript', 'common_voice', 'pre_split', 'flat_file', 'multi_shard'],
  preferredDatasetType: 'audio_transcript',
};

export default textToSpeechPlugin;
