// YOLO kann mehr als Boxen: Segmentierung, Keypoints, gedrehte Boxen und
// Klassifikation laufen ueber dasselbe Plugin (task_type 'detect'); die
// Aufgabe steckt in den Gewichten.

import { describe, it, expect } from 'vitest';
import { detectPlugin } from '../registry';
import { yoloTaskFromName } from '../yolo/detect';
import { summarizeYoloResult } from '../yolo/TestPlugin';
import { checkDatasetCompat } from '../datasetCompat';
import yolo from '../yolo';
import type { DatasetAnalysis } from '../datasetCompatHelpers';
import de from '../../locales/de.json';
import en from '../../locales/en.json';

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 80, pairing_status: null, warnings: [], file_count: 10, dir_count: 3, extensions, schema_hint: null,
});

describe('YOLO-Aufgaben', () => {
  it('erkennt Seg-, Pose-, OBB- und Cls-Gewichte als YOLO', () => {
    for (const name of ['yolo11n-seg.pt', 'yolo11n-pose.pt', 'yolo11n-obb.pt', 'yolo11n-cls.pt', 'yolov8s-seg.pt']) {
      const r = detectPlugin(name);
      expect(r.supported, name).toBe(true);
      if (r.supported) expect(r.plugin.id, name).toBe('yolo');
    }
  });

  it('liest die Aufgabe aus dem Gewichtsnamen', () => {
    expect(yoloTaskFromName('yolo11n.pt')).toBe('detect');
    expect(yoloTaskFromName('/models/x/yolo11n-seg.pt')).toBe('segment');
    expect(yoloTaskFromName('C:\\m\\yolov8m-pose.pt')).toBe('pose');
    expect(yoloTaskFromName('yolo11s-obb.pt')).toBe('obb');
    expect(yoloTaskFromName('yolo11n-cls.pt')).toBe('classify');
    // Trainierte Versionen verraten die Aufgabe nicht im Namen.
    expect(yoloTaskFromName('model.pt')).toBeNull();
    expect(yoloTaskFromName('best.pt')).toBeNull();
  });

  it('bietet task (Standard auto) und resume als Parameter an', () => {
    expect(yolo.defaultPluginConfig?.task).toBe('auto');
    expect(yolo.defaultPluginConfig).toHaveProperty('resume', '');
  });

  it('Ordner pro Klasse ist fuer YOLO-cls geeignet, auch ohne Label-Dateien', () => {
    const r = checkDatasetCompat('yolo', [], analysis('folder_class', ['.jpg']), yolo);
    expect(r.overallLevel).not.toBe('bad');
  });

  it('Box-Datasets ohne Labels bleiben ungeeignet', () => {
    expect(checkDatasetCompat('yolo', [], analysis('pre_split', ['.jpg']), yolo).overallLevel).toBe('bad');
  });
});

describe('YOLO-Testergebnis', () => {
  it('zaehlt Masken und sichere Keypoints', () => {
    const s = summarizeYoloResult({
      task: 'pose', inference_time_ms: 1, image_path: 'x',
      detections: [{ label: 'person', confidence: 0.9, bbox: [0, 0, 1, 1], keypoints: [[1, 1, 0.9], [2, 2, 0.2], [3, 3, 0.6]] }],
    });
    expect(s).toMatchObject({ task: 'pose', keypoints: 2, masks: 0 });
    const seg = summarizeYoloResult({
      task: 'segment', inference_time_ms: 1, image_path: 'x',
      detections: [
        { label: 'a', confidence: 0.9, bbox: [0, 0, 1, 1], polygon: [[0, 0], [1, 0], [1, 1]] },
        { label: 'b', confidence: 0.5, bbox: [0, 0, 1, 1] },
      ],
    });
    expect(seg.masks).toBe(1);
  });

  it('aeltere Backends ohne task gelten als Detektion', () => {
    const s = summarizeYoloResult({ detections: [], inference_time_ms: 1, image_path: 'x' });
    expect(s.task).toBe('detect');
    expect(s.classes).toEqual([]);
  });

  it('Texte fuer alle Aufgaben gibt es auf Deutsch und Englisch', () => {
    for (const loc of [de, en] as Array<{ testPlugins: { yolo: Record<string, unknown> } }>) {
      const y = loc.testPlugins.yolo;
      expect(Object.keys(y.taskNames as object).sort()).toEqual(['classify', 'detect', 'obb', 'pose', 'segment']);
      for (const k of ['taskLabel', 'maskBadge', 'obbBadge', 'keypointsBadge', 'topClasses', 'noClasses', 'summaryMasks', 'summaryKeypoints']) {
        expect(y[k], k).toBeTruthy();
      }
    }
  });
});
