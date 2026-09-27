import type { ModelPlugin } from '../types';
import { detectVisionLanguage } from './detect';
import VisionLanguageTestPlugin from './TestPlugin';

const visionLanguagePlugin: ModelPlugin = {
  id: 'vision-language',
  name: 'Vision-Language (VLM, LoRA)',
  description: 'Bild + Frage -> Antwort, Bildbeschreibung, Texterkennung: SmolVLM, Qwen2-VL, PaliGemma, LLaVA, BLIP.',
  taskType: 'vision_language',
  // LoRA auf einem kleinen VLM: 2e-4, kleine Batches (ein Bild sind hunderte Tokens).
  defaultTrainingConfig: {
    learning_rate: 2e-4, batch_size: 4, epochs: 3,
    lora_r: 16, lora_alpha: 32, lora_dropout: 0.05, warmup_ratio: 0.1,
  },
  defaultPluginConfig: {
    // Frage fuer Beispiele ohne eigene (Bilder + .txt, Klassenordner). Leer = "Beschreibe das Bild."
    prompt: '',
    max_new_tokens: 64,
    // Antworten ueber so viele Val-Beispiele erzeugen (exact match, ROUGE-L), vorher und nachher.
    eval_generate_samples: 50,
    eval_before_training: true,
    // false: ein Bild = eine Kachel statt bis zu 17 (SmolVLM) — viel schneller, fuer Fotos meist genug.
    image_splitting: false,
    train_vision: false,
  },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'load_in_4bit', 'load_in_8bit', 'label_smoothing',
  ],
  detect: detectVisionLanguage,
  TestComponent: VisionLanguageTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'folder_class'],
  preferredDatasetType: 'flat_file',
};

export default visionLanguagePlugin;
