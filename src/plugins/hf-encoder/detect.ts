// HF Encoder (Sequence Classification) – Erkennung
//
// Ziel: Frontend soll alle Modelle als "supported" erkennen, die das aktuelle
// Backend-Plugin `task_type="seq_classification"` trainieren kann.
//
// Primär-Signal: HuggingFace `config.json` -> `model_type`
// Fallback: heuristische Erkennung anhand Model-ID/Pfad.

import type { ModelConfig } from '../types';

// Muss zum Backend-Manifest passen:
// /src-tauri/python/train_engine/plugins/seq_classification/manifest.json -> supported_architectures
export const HF_ENCODER_SUPPORTED_MODEL_TYPES: string[] = [
  'xlm-roberta', 'roberta', 'bert', 'deberta', 'deberta-v2',
  'distilbert', 'albert', 'camembert', 'electra', 'rembert',
  'xlm', 'ernie', 'funnel', 'mpnet', 'squeezebert', 'layoutlm',
];

const SUPPORTED_MODEL_TYPES = new Set<string>(HF_ENCODER_SUPPORTED_MODEL_TYPES);

function containsToken(normalized: string, token: string): boolean {
  // Token muss an "Wortgrenzen" sitzen (verhindert z.B. bert in albert).
  // normalized ist lowercase und nutzt / als Pfadseparator.
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i');
  return re.test(normalized);
}

export function detectHFEncoder(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // 1) config.json (lokale Modelle) – robusteste Quelle
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType && SUPPORTED_MODEL_TYPES.has(modelType)) return true;

  // 2) Fallback: Model-ID/Pfad
  const normalized = modelPathOrId.toLowerCase().replace(/\\/g, '/');
  const lastPart = normalized.split('/').pop() ?? normalized;

  // Spezifischere Tokens zuerst (vermeidet false positives)
  const tokens = [
    'xlm-roberta',
    'distilbert',
    'deberta-v2',
    'deberta',
    'camembert',
    'layoutlm',
    'squeezebert',
    'electra',
    'rembert',
    'roberta',
    'albert',
    'funnel',
    'mpnet',
    'ernie',
    'bert',
    'xlm',
  ];

  for (const t of tokens) {
    if (containsToken(normalized, t) || containsToken(lastPart, t)) return true;
  }

  return false;
}
