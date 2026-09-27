import type { ModelPlugin } from '../types';
import { detectCausalLM } from './detect';
import CausalLMTestPlugin from './TestPlugin';

const causalLMPlugin: ModelPlugin = {
  id: 'causal-lm',
  name: 'LLM Fine-Tuning (LoRA)',
  description: 'Feinjustiert Decoder-LLMs (Llama, Qwen, Mistral, Gemma, Phi, SmolLM) mit LoRA oder QLoRA auf Chat-, Frage-Antwort- oder Textdaten; Praeferenzpaare (chosen/rejected) per DPO.',
  taskType: 'causal_lm',
  // LoRA ist der Normalfall: volles Fine-Tuning eines 7B-Modells braucht
  // >100 GB. LoRA-Lernraten liegen etwa 10x ueber denen des vollen Trainings.
  defaultTrainingConfig: {
    use_lora: true, lora_r: 16, lora_alpha: 32, lora_dropout: 0.05,
    learning_rate: 2e-4, batch_size: 4, epochs: 3, max_seq_length: 1024,
    warmup_ratio: 0.03, scheduler: 'cosine', label_smoothing: 0,
  },
  // backend: auto = MLX auf Apple Silicon (schneller, 4-bit moeglich), sonst PyTorch.
  // eval_samples: so viele Val-Beispiele werden vor/nach dem Training generiert
  // und mit der Referenz verglichen (Exact Match, ROUGE-L).
  // dpo_*: greifen nur bei Praeferenzdaten (chosen/rejected). dpo_sft_weight
  // haelt das Antwortformat stabil — reines DPO (0) zerlegte es im Test.
  defaultPluginConfig: {
    backend: 'auto', system_prompt: '', eval_samples: 10, eval_baseline: true,
    max_new_tokens: 256, export_gguf: false, dpo_beta: 0.1, dpo_sft_weight: 1.0,
  },
  hiddenTrainingFields: ['label_smoothing', 'group_by_length'],
  detect: detectCausalLM,
  TestComponent: CausalLMTestPlugin,
  supportedDatasetTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredDatasetType: 'flat_file',
};

export default causalLMPlugin;
