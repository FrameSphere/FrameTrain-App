// Pfad-Referenzen fuer Dev Train / Dev Test.
// Regression aus dem App-Durchgang vom 12.09.2026: Nach dem Wechsel des
// Datasets zeigte "Dataset-Pfad 2" ein anderes Dataset als vorher, und fuer
// YOLO gab es keine Referenz auf die dataset.yaml.

import { describe, it, expect } from 'vitest';
import { buildDatasetRefs, refsToEnv, refsForPrompt, selectedFirst } from '../devDatasetRefs';

const images   = { id: 'a', name: 'images',   storage_path: '/ds/a' };
const skiTrain = { id: 'b', name: 'SkiTrain', storage_path: '/ds/b', dataset_yaml_path: '/ds/b/dataset.yaml' };
const neu      = { id: 'c', name: 'neu',      storage_path: '/ds/c', dataset_yaml_path: '/ds/c/dataset.yaml' };
const all = [images, skiTrain, neu];

const env = (selected: string | null) => refsToEnv('/model', buildDatasetRefs(all, selected));

describe('buildDatasetRefs', () => {
  it('DATASET_PATH ist das gewaehlte Dataset', () => {
    expect(env('c').DATASET_PATH).toBe('/ds/c');
    expect(env('c').DATASET_YAML).toBe('/ds/c/dataset.yaml');
    expect(env(null).DATASET_PATH).toBe('/ds/a');
  });

  it('nummerierte Referenzen haengen nicht von der Auswahl ab', () => {
    for (const sel of ['a', 'b', 'c']) {
      const e = env(sel);
      expect(e.DATASET_PATH_1).toBe('/ds/a');
      expect(e.DATASET_PATH_2).toBe('/ds/b');
      expect(e.DATASET_PATH_3).toBe('/ds/c');
      expect(e.DATASET_YAML_2).toBe('/ds/b/dataset.yaml');
    }
  });

  it('DATASET_YAML nur, wenn es eine yaml gibt', () => {
    const e = env('a');
    expect(e.DATASET_YAML).toBeUndefined();
    expect(e.DATASET_YAML_1).toBeUndefined();
  });

  it('bei nur einem Dataset keine doppelten Nummern', () => {
    const e = refsToEnv('/m', buildDatasetRefs([skiTrain], null));
    expect(Object.keys(e).sort()).toEqual(['DATASET_PATH', 'DATASET_YAML', 'MODEL_PATH']);
  });

  it('ohne Pfad keine leere Variable und kein Name als Pfad', () => {
    const ohnePfad = { id: 'x', name: 'Ohne' };
    const refs = buildDatasetRefs([ohnePfad], null);
    expect(refsToEnv('/m', refs).DATASET_PATH).toBeUndefined();
    expect(refsForPrompt(refs)).toContain('(kein Pfad)');
    expect(refsForPrompt(refs)).not.toContain('"Ohne"');
  });

  it('selectedFirst stellt das gewaehlte Dataset nach vorne', () => {
    expect(selectedFirst(all, 'c').map(d => d.id)).toEqual(['c', 'a', 'b']);
    expect(selectedFirst(all, 'unbekannt').map(d => d.id)).toEqual(['a', 'b', 'c']);
  });
});
