// Welches Modell zu welchem Projekt passt — ohne React, damit es auch die
// KI-Werkzeuge (mcp/tools/studio.ts) benutzen koennen.

import { detectPluginForModel, type ModelDetectionInfo } from '../../plugins/registry';
import { yoloTaskFromName } from '../../plugins/yolo/detect';
import type { StudioProject } from './studioTypes';

export type StudioModel = ModelDetectionInfo;

/** Welche Plugin-Aufgaben aus diesem Projekt trainieren koennen. */
export function passendeAufgaben(project: Pick<StudioProject, 'modality' | 'task'>): string[] {
  if (project.modality === 'text') {
    // Paare (source/target) trainieren Seq2Seq-Modelle und ebenso Decoder-LLMs.
    return project.task === 'pairs' ? ['seq2seq', 'causal_lm'] : ['seq_classification'];
  }
  if (project.modality === 'video') return ['video_classification'];
  if (project.modality === 'image' && project.task === 'classify') {
    return ['hf_image_classification', 'image_classification'];
  }
  if (project.modality === 'audio') {
    // Transkripte trainieren Spracherkenner (Whisper, Wav2Vec2-CTC). Die
    // liefen frueher mangels eigenem Plugin als seq2seq bzw. Audio-Klassifikation.
    // Dieselben Paare aus Aufnahme und Transkript trainieren auch Sprachausgabe (SpeechT5).
    return project.task === 'transcript' ? ['speech_recognition', 'text_to_speech'] : ['audio_classification'];
  }
  return ['detect'];
}

export function passtZumProjekt(model: StudioModel, project: Pick<StudioProject, 'modality' | 'task'>): boolean {
  const r = detectPluginForModel(model);
  if (!r.supported || !passendeAufgaben(project).includes(r.plugin.taskType)) return false;
  // Box-Projekte: YOLO-Varianten fuer gedrehte Boxen, Masken oder Posen
  // zeichnen keine normalen Boxen (yolo-obb stand sonst vorausgewaehlt).
  if (r.plugin.taskType === 'detect') {
    const yoloTask = yoloTaskFromName(model.name ?? '') ?? yoloTaskFromName(model.source_path ?? '');
    if (yoloTask && yoloTask !== 'detect') return false;
  }
  return true;
}

/** Passende zuerst, Reihenfolge innerhalb der Gruppen bleibt erhalten. */
export function ordneModelle<T extends StudioModel>(
  models: T[], project: Pick<StudioProject, 'modality' | 'task'>,
): { passend: T[]; andere: T[] } {
  const passend: T[] = [];
  const andere: T[] = [];
  for (const m of models) (passtZumProjekt(m, project) ? passend : andere).push(m);
  return { passend, andere };
}

/** Ein Modell, das zur Projektart passt und frei verfuegbar ist — als Tipp,
 *  wenn noch keines da ist. Einsteiger wissen sonst nicht, wonach sie suchen sollen. */
export function empfohlenesModell(project: Pick<StudioProject, 'modality' | 'task'>): string {
  if (project.modality === 'video') return 'MCG-NJU/videomae-base';
  if (project.modality === 'audio') return project.task === 'transcript' ? 'openai/whisper-small' : 'facebook/wav2vec2-base';
  if (project.modality === 'text') return project.task === 'pairs' ? 'google/mt5-small' : 'distilbert-base-multilingual-cased';
  if (project.task === 'classify') return 'google/vit-base-patch16-224';
  return 'yolo11n';
}
