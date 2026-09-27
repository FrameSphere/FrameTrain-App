import type { ModelPlugin } from '../types';
import { detectTextToImageLora } from './detect';
import TextToImageTestPlugin from './TestPlugin';

const textToImageLoraPlugin: ModelPlugin = {
  id: 'text-to-image-lora',
  name: 'Text-to-Image LoRA (Stable Diffusion)',
  description: 'Bringt Stable Diffusion 1.x/2.x oder SDXL per LoRA einen eigenen Stil oder ein eigenes Objekt bei.',
  taskType: 'text_to_image_lora',
  // Werte wie im diffusers-Referenzskript: LoRA lernt mit 1e-4, ein Bild pro
  // Schritt, einige hundert Schritte. alpha = r haelt die LoRA-Skala bei 1.
  defaultTrainingConfig: {
    learning_rate: 1e-4, batch_size: 1, epochs: 20, max_steps: 500,
    lora_r: 8, lora_alpha: 8, lora_dropout: 0, scheduler: 'constant', warmup_ratio: 0,
  },
  defaultPluginConfig: {
    // 512 ist die native Aufloesung von SD 1.x; auf dem Mac spart 256 viel Zeit.
    resolution: 512,
    // DreamBooth: gilt fuer alle Bilder ohne eigene Caption, z. B. "a photo of sks dog".
    instance_prompt: '',
    validation_prompt: '',
    num_validation_images: 2,
    sample_steps: 25,
    guidance_scale: 7.5,
    center_crop: true,
    random_flip: true,
    sample_before_training: true,
    checkpoint_every: 0,
    // true: zusaetzlich die komplette Pipeline speichern (laedt ohne Basismodell, braucht GB).
    merge_lora: false,
  },
  hiddenTrainingFields: [
    'max_seq_length', 'group_by_length', 'use_lora', 'load_in_4bit', 'load_in_8bit',
    'label_smoothing', 'eval_steps', 'eval_strategy', 'max_eval_samples', 'dropout',
  ],
  detect: detectTextToImageLora,
  TestComponent: TextToImageTestPlugin,
  supportedDatasetTypes: ['flat_file', 'folder_class', 'pre_split'],
  preferredDatasetType: 'flat_file',
};

export default textToImageLoraPlugin;
