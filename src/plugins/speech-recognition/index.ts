import type { ModelPlugin } from '../types';
import { detectSpeechRecognition } from './detect';
import SpeechRecognitionTestPlugin from './TestPlugin';

const speechRecognitionPlugin: ModelPlugin = {
  id: 'speech-recognition',
  name: 'Speech Recognition (ASR)',
  description: 'Spracherkennung: Whisper/Moonshine als Seq2Seq, wav2vec2/HuBERT/WavLM mit CTC. Misst WER und CER.',
  taskType: 'speech_recognition',
  // Whisper vergisst bei hohen Lernraten schnell, was es schon konnte.
  defaultTrainingConfig: { learning_rate: 3e-5, batch_size: 8, epochs: 5 },
  // language leer = am Trainingsmaterial erkennen (nur Whisper).
  defaultPluginConfig: {
    language: '',
    task: 'transcribe',
    max_seconds: 30,
    freeze_feature_encoder: true,
    eval_before_training: true,
  },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'lora_r', 'lora_alpha',
    'lora_dropout', 'lora_target_modules', 'load_in_4bit', 'load_in_8bit',
  ],
  detect: detectSpeechRecognition,
  TestComponent: SpeechRecognitionTestPlugin,
  // audio_transcript: so exportiert die Datensatz-Werkstatt (Audio + gleichnamige .txt).
  // flat_file/multi_shard: Parquet mit Audio- und Textspalte.
  supportedDatasetTypes: ['audio_transcript', 'common_voice', 'pre_split', 'flat_file', 'multi_shard'],
  preferredDatasetType: 'audio_transcript',
};

export default speechRecognitionPlugin;
