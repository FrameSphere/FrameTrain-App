// Objekterkennung ueber YOLO hinaus: HF-Detektoren (DETR, RT-DETR, YOLOS)
// bekommen ein eigenes Plugin, Ultralytics-RT-DETR (rtdetr-l.pt) bleibt bei YOLO.
//
// Frueher landete hustvl/yolos-tiny beim YOLO-Plugin (der Name enthaelt "/yolo")
// und scheiterte im Training; DETR galt als "nicht unterstuetzt".

import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel, hiddenTrainingFieldsForTaskType } from '../registry';
import { yoloTaskFromName } from '../yolo/detect';
import { checkDatasetCompat } from '../datasetCompat';
import hfObjectDetection from '../hf-object-detection';
import type { DatasetAnalysis } from '../datasetCompatHelpers';
import de from '../../locales/de.json';
import en from '../../locales/en.json';

const pluginOf = (id: string, config?: Record<string, unknown>) => {
  const r = detectPlugin(id, config as never);
  return r.supported ? r.plugin.id : null;
};

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 80, pairing_status: null, warnings: [], file_count: 10, dir_count: 3, extensions, schema_hint: null,
});

describe('HF-Detektoren -> hf-object-detection', () => {
  it.each([
    'facebook/detr-resnet-50',
    'hustvl/yolos-tiny',
    'hustvl/yolos-small',
    'PekingU/rtdetr_r18vd',
    'PekingU/rtdetr_v2_r50vd',
    'microsoft/conditional-detr-resnet-50',
    'SenseTime/deformable-detr',
    'microsoft/table-transformer-detection',
  ])('%s', (id) => expect(pluginOf(id)).toBe('hf-object-detection'));

  it.each(['rt_detr', 'rt_detr_v2', 'yolos', 'detr', 'conditional_detr', 'deformable_detr', 'd_fine'])(
    'config.json model_type %s', (mt) => expect(pluginOf('/models/local_abc', { model_type: mt })).toBe('hf-object-detection'));

  it('YOLOS mit config.json geht nicht an YOLO', () => {
    expect(pluginOf('/models/yolos', { model_type: 'yolos', architectures: ['YolosForObjectDetection'] }))
      .toBe('hf-object-detection');
  });

  it('lokal importiert: der vergebene Name entscheidet', () => {
    const r = detectPluginForModel({ id: 'm1', name: 'mein-detr', source_path: '/data/models/local_ab12' });
    expect(r.supported && r.plugin.id).toBe('hf-object-detection');
  });
});

describe('Ultralytics bleibt bei YOLO', () => {
  it.each(['rtdetr-l.pt', 'rtdetr-x.pt', '/models/hf_x/rtdetr-l.pt', 'yolov10n.pt', 'yolo12n.pt', 'yolo26n.pt',
    'yolov8s-world.pt', 'yolo11n.pt', 'Ultralytics/YOLOv8'])('%s -> yolo', (id) => expect(pluginOf(id)).toBe('yolo'));

  it('rtdetr-l.pt ist eine Erkennungsaufgabe', () => {
    expect(yoloTaskFromName('rtdetr-l.pt')).toBe('detect');
  });
});

describe('Plugin-Definition', () => {
  it('task_type, Datenformate und ausgeblendete Felder', () => {
    expect(hfObjectDetection.taskType).toBe('hf_object_detection');
    expect(hfObjectDetection.supportedDatasetTypes).toEqual(['yolo_bbox', 'coco_json', 'pre_split']);
    expect(hiddenTrainingFieldsForTaskType('hf_object_detection')).toContain('max_seq_length');
  });

  it('YOLO- und COCO-Daten passen, Ordner-Klassen nicht', () => {
    const level = (a: DatasetAnalysis) => checkDatasetCompat('hf-object-detection', [], a, hfObjectDetection).overallLevel;
    expect(level(analysis('yolo_bbox', ['.jpg', '.txt']))).toBe('perfect');
    expect(level(analysis('coco_json', ['.jpg', '.json']))).toBe('ok');
    expect(level(analysis('folder_class', ['.jpg']))).toBe('bad');
  });

  it('ohne Labels gibt es den eigenen Hinweis (EN+DE)', () => {
    const r = checkDatasetCompat('hf-object-detection', [], analysis('unknown', ['.jpg']), hfObjectDetection);
    expect(r.overallLevel).toBe('bad');
    expect(r.summaryMsg?.key).toMatch(/missing\.hfDetect$/);
    const de_ = (de as any).datasetCompat ?? {};
    const en_ = (en as any).datasetCompat ?? {};
    expect(JSON.stringify(de_)).toContain('hfDetect');
    expect(JSON.stringify(en_)).toContain('hfDetect');
  });

  it('Ergebnistext fuer Boxen in beiden Sprachen', () => {
    expect((de as any).testPlugins.generic.detectedObjects).toBe('Erkannte Objekte');
    expect((en as any).testPlugins.generic.detectedObjects).toBe('Detected objects');
  });
});
