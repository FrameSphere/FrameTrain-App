// Kompatibilitaet fuer Plugins ohne eigene Pruefung.
// Regression aus dem App-Durchgang vom 12.09.2026: Das YOLO-Dataset "images"
// (nur .jpg, keine Labels) stand auf der Trainingsseite als "Geeignet".

import { describe, it, expect } from 'vitest';
import { checkDatasetCompat } from '../datasetCompat';
import yolo from '../yolo';
import canvas from '../canvas';
import hfImage from '../hf-image-classification';
import type { DatasetAnalysis } from '../datasetCompatHelpers';

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[], pairing = null as DatasetAnalysis['pairing_status']): DatasetAnalysis => ({
  detected_type, confidence: 80, pairing_status: pairing, warnings: [], file_count: 10, dir_count: 0, extensions, schema_hint: null,
});

describe('genericDatasetCompat', () => {
  it('YOLO ohne Label-Dateien ist nicht geeignet', () => {
    const r = checkDatasetCompat('yolo', ['.jpg', '.cache', '.yaml'], analysis('pre_split', ['.jpg', '.cache', '.yaml']), yolo);
    expect(r.overallLevel).toBe('bad');
  });

  it('YOLO mit Labels ist perfekt, Pascal VOC geeignet', () => {
    expect(checkDatasetCompat('yolo', [], analysis('yolo_bbox', ['.jpg', '.txt']), yolo).overallLevel).toBe('perfect');
    expect(checkDatasetCompat('yolo', [], analysis('pascal_voc', ['.jpg', '.xml']), yolo).overallLevel).toBe('ok');
  });

  it('YOLO mit teilweise fehlenden Labels warnt', () => {
    const pairing = { is_paired: false, primary_count: 10, paired_count: 6, orphan_primaries: [], orphan_secondaries: [] };
    expect(checkDatasetCompat('yolo', [], analysis('yolo_bbox', ['.jpg', '.txt'], pairing), yolo).overallLevel).toBe('warning');
  });

  it('Canvas liest keine YOLO-Datasets', () => {
    expect(checkDatasetCompat('canvas', [], analysis('yolo_bbox', ['.jpg', '.txt']), canvas).overallLevel).toBe('bad');
    expect(checkDatasetCompat('canvas', [], analysis('folder_class', ['.png']), canvas).overallLevel).toBe('ok');
  });

  it('HF-Bildmodell akzeptiert Parquet (HF-Download), aber keine reinen Textdateien', () => {
    expect(checkDatasetCompat('hf-image-classification', [], analysis('flat_file', ['.parquet']), hfImage).overallLevel).toBe('ok');
    expect(checkDatasetCompat('hf-image-classification', [], analysis('flat_file', ['.csv']), hfImage).overallLevel).toBe('bad');
  });

  it('ohne Plugin-Info bleibt das alte Verhalten', () => {
    expect(checkDatasetCompat('unbekannt', ['.jpg']).overallLevel).toBe('ok');
  });
});
