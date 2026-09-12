// Welche Art Code braucht ein Modell in den Dev-Bereichen (Train und Test)?
//
// Beide Panels haben bisher immer Text-Code erzeugt — auch fuer wav2vec2 oder
// resnet-50. Die Erkennung laeuft ueber dieselbe Plugin-Registry wie das
// Training, damit Vorlage und Engine nie auseinanderlaufen.

import { detectPlugin } from '../plugins/registry';

export type ScriptModality = 'text' | 'image' | 'audio' | 'seq2seq' | 'detection';

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
    // YOLO: Ultralytics mit dataset.yaml statt Transformers mit Tabellen.
    case 'detect':
      return 'detection';
    default:
      return 'text';
  }
}

/**
 * Pfad als Inhalt eines Python-String-Literals ("...").
 * Ohne Escaping war jede Vorlage auf Windows kaputt: "C:\Users\..." ist in
 * Python ein ungueltiges \U-Escape und bricht schon beim Parsen ab.
 */
export function pyPath(path: string | null | undefined): string {
  return (path ?? '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

/** Standard-OUTPUT_PATH einer Vorlage, falls sie ausserhalb von FrameTrain laeuft. */
export function templateOutputPath(outputPath: string, name: 'dev_train' | 'dev_test'): string {
  return outputPath.replace(/(dev_)?<job_id>|\{wird beim Start gesetzt\}/, name);
}
