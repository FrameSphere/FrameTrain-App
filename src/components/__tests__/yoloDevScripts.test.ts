// Dev-Vorlagen fuer YOLO und Pfade in Vorlagen.
// Regression aus dem App-Durchgang vom 12.09.2026: Fuer YOLOv8 erzeugten Dev
// Train und Dev Test ein Text-Klassifikations-Skript (AutoTokenizer,
// load_dataset), das beim ersten Lauf abstuerzte.

import { describe, it, expect, vi } from 'vitest';

vi.mock('../../plugins/registry', () => ({
  detectPlugin: (path: string) => path.includes('yolo')
    ? { supported: true, plugin: { taskType: 'detect' } }
    : { supported: true, plugin: { taskType: 'seq_classification' } },
}));

import { generateYoloTrainScript, generateYoloTestScript } from '../yoloDevScripts';
import { detectScriptModality, pyPath, templateOutputPath } from '../scriptModality';

const paths = {
  modelPath: '/m/yolo', datasetPath: '/ds/a', datasetYaml: '/ds/a/dataset.yaml', outputPath: '/out/dev_train',
};

describe('YOLO Dev-Vorlagen', () => {
  it('YOLO-Modelle bekommen die Detection-Vorlage', () => {
    expect(detectScriptModality({ name: 'yolov8', source_path: '/models/yolov8' })).toBe('detection');
    expect(detectScriptModality({ name: 'bert', source_path: '/models/bert' })).toBe('text');
  });

  it('Training uebergibt Ultralytics die yaml, nicht den Ordner', () => {
    const s = generateYoloTrainScript(paths);
    expect(s).toContain('DATASET_YAML = os.environ.get("DATASET_YAML", "/ds/a/dataset.yaml")');
    expect(s).toContain('data=data_yaml');
    expect(s).toContain('emit(\n    "complete"');
    expect(s).not.toMatch(/AutoTokenizer|load_dataset/);
  });

  it('Test wertet test aus, wenn die yaml einen test-Split hat', () => {
    const s = generateYoloTestScript(paths);
    expect(s).toContain('split = "test" if has_split(data_yaml, "test") else "val"');
    expect(s).toContain('results.json');
  });
});

describe('Pfade in Vorlagen', () => {
  it('Windows-Pfade werden fuer Python-Strings escaped', () => {
    expect(pyPath('C:\\Users\\karol')).toBe('C:\\\\Users\\\\karol');
    expect(pyPath('a"b')).toBe('a\\"b');
    expect(pyPath(undefined)).toBe('');
  });

  it('Standard-OUTPUT_PATH ohne doppeltes dev_', () => {
    expect(templateOutputPath('/x/training_outputs/dev_<job_id>', 'dev_train')).toBe('/x/training_outputs/dev_train');
    expect(templateOutputPath('/x/test_outputs/dev_<job_id>', 'dev_test')).toBe('/x/test_outputs/dev_test');
  });
});
