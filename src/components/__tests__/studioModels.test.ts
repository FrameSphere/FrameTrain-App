// Welches Modell zu welchem Projekt passt.
//
// Vorher nahm der Export einfach das erste Modell der Liste. Bei einem
// Textprojekt war das yolo8n — und der Datensatz haengte danach an einem
// Bildmodell, mit dem er nie trainiert werden kann.

import { describe, it, expect } from 'vitest';
import { ordneModelle, passtZumProjekt } from '../studio/studioModels';

const YOLO  = { id: 'm_yolo', name: 'yolo8n', source_path: 'ultralytics/yolov8n', model_type: 'yolo' };
const BERT  = { id: 'm_bert', name: 'bert', source_path: 'bert-base-uncased', model_type: 'bert' };
const T5    = { id: 'm_t5',   name: 't5',   source_path: 't5-small', model_type: 't5' };
const WAV   = { id: 'm_wav',  name: 'wav',  source_path: 'facebook/wav2vec2-base', model_type: 'wav2vec2' };

const projekt = (modality: string, task: string) => ({ modality, task });

describe('studioModels', () => {
  it('stellt bei Textklassifikation das Textmodell vor das Bildmodell', () => {
    const { passend, andere } = ordneModelle([YOLO, BERT], projekt('text', 'classification'));
    expect(passend.map(m => m.id)).toEqual(['m_bert']);
    expect(andere.map(m => m.id)).toEqual(['m_yolo']);
  });

  it('ordnet jede Projektart dem richtigen Modell zu', () => {
    expect(passtZumProjekt(YOLO, projekt('image', 'bbox'))).toBe(true);
    expect(passtZumProjekt(BERT, projekt('image', 'bbox'))).toBe(false);
    expect(passtZumProjekt(T5,   projekt('text', 'pairs'))).toBe(true);
    expect(passtZumProjekt(BERT, projekt('text', 'pairs'))).toBe(false);
    expect(passtZumProjekt(WAV,  projekt('audio', 'classification'))).toBe(true);
  });

  it('beachtet die Plugin-Zuordnung aus dem Import', () => {
    // Ein lokal importiertes Modell traegt im Pfad keinen Architekturnamen mehr.
    const lokal = { id: 'm_x', name: 'mein-modell', source_path: '/models/local_abc',
      model_type: null, plugin_override: 'yolo' };
    expect(passtZumProjekt(lokal, projekt('image', 'bbox'))).toBe(true);
  });
});
