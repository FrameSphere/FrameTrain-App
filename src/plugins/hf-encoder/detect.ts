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

/** Verzeichnisnamen, die kein Modellname sind – dort lohnt der Blick eine Ebene höher. */
const OPAQUE_DIR = /^(ver_[a-z0-9]+|hf_[a-z0-9]+|snapshots|blobs|refs|model|models|versions|original|latest|[0-9a-f]{12,})$/;

/**
 * Reduziert Pfad oder Repo-ID auf den Teil, der tatsächlich den Modellnamen trägt.
 *
 * Wichtig: NUR dieses Segment darf für die Token-Heuristik verwendet werden.
 * Bei der Repo-ID `distilbert/distilgpt2` ist `distilbert` der Organisations-
 * name — wer den ganzen String durchsucht, hält ein GPT-2 für einen BERT und
 * meldet es als unterstützt, bis das Training es beim Start abweist.
 */
function modelNameSegment(normalized: string): string {
  const segments = normalized.split('/').filter(Boolean);
  if (segments.length === 0) return normalized;
  for (let i = segments.length - 1; i >= 0; i--) {
    if (!OPAQUE_DIR.test(segments[i])) return segments[i];
  }
  return segments[segments.length - 1];
}

/**
 * Namensbestandteile, die eindeutig auf eine Nicht-Encoder-Architektur zeigen.
 *
 * Sie schlagen die Pfad-Heuristik: `distilbert/distilgpt2` enthält im
 * Organisationsnamen „distilbert“, ist aber ein GPT-2. Ohne diesen Vorrang
 * galt das Modell als unterstützt — bis das Training es nach 1,5 GB Download
 * abwies.
 */
const NON_ENCODER_TOKENS = [
  'gpt2', 'distilgpt2', 'gptj', 'gpt-neo', 'gpt-neox', 'gpt',
  'llama', 'mistral', 'mixtral', 'qwen', 'falcon', 'phi', 'gemma',
  'bloom', 'mpt', 'opt', 't5', 'bart', 'pegasus', 'marian',
  'whisper', 'clip', 'stable-diffusion', 'wav2vec2',
];

export function detectHFEncoder(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // 1) config.json (lokale Modelle) – robusteste Quelle.
  //    Ist ein model_type bekannt, ist er allein maßgeblich: eine bekannte,
  //    aber nicht unterstützte Architektur darf nicht über die Namens-
  //    Heuristik doch noch als unterstützt durchrutschen.
  const modelType = configJson?.model_type?.toLowerCase();
  if (modelType) return SUPPORTED_MODEL_TYPES.has(modelType);

  // 2) Fallback: Namens-Heuristik
  const normalized = modelPathOrId.toLowerCase().replace(/\\/g, '/');
  const lastPart = modelNameSegment(normalized);

  // Widerspricht der Modellname selbst, zählt er mehr als der übrige Pfad.
  for (const t of NON_ENCODER_TOKENS) {
    if (containsToken(lastPart, t)) return false;
  }

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
    // Verbreitete Encoder, deren Name die Architektur nicht nennt.
    // sentence-transformers/all-MiniLM-L6-v2 ist ein BERT, wurde aber als
    // "nicht unterstuetzt" gemeldet — eines der meistgenutzten Modelle ueberhaupt.
    'minilm',
    'mobilebert',
    'tinybert',
    'bert',
    'xlm',
  ];

  for (const t of tokens) {
    if (containsToken(lastPart, t) || containsToken(normalized, t)) return true;
  }

  return false;
}
