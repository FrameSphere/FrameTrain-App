// Welche Art Code braucht ein Modell in den Dev-Bereichen (Train und Test)?
//
// Beide Panels haben bisher immer Text-Code erzeugt — auch fuer wav2vec2 oder
// resnet-50. Die Erkennung laeuft ueber dieselbe Plugin-Registry wie das
// Training, damit Vorlage und Engine nie auseinanderlaufen.

import { detectPlugin } from '../plugins/registry';

export type ScriptModality = 'text' | 'image' | 'audio' | 'seq2seq';

interface ModelLike {
  name: string;
  local_path?: string | null;
  source_path?: string | null;
  model_type?: string | null;
}

export function detectScriptModality(model: ModelLike | null): ScriptModality {
  if (!model) return 'text';
  const r = detectPlugin(
    model.source_path || model.local_path || model.name,
    model.model_type ? { model_type: model.model_type } : undefined,
  );
  if (!r.supported) return 'text';
  switch (r.plugin.taskType) {
    case 'hf_image_classification':
    case 'image_classification':
      return 'image';
    case 'audio_classification':
      return 'audio';
    case 'seq2seq':
      return 'seq2seq';
    default:
      return 'text';
  }
}
