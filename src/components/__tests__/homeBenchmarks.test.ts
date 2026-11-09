import { describe, it, expect } from 'vitest';
import { buildBenchmarks } from '../homeBenchmarks';
import { buildPipeline, type PipelineInput, type PipelineTest } from '../homePipeline';

const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60000).toISOString();
const test = (id: string, model: string, dataset: string, accuracy: number, minAgo: number, extra: Record<string, unknown> = {}, version = 'v1'): PipelineTest => ({
  id, model_id: model, model_name: model, version_id: `${model}-${version}`, version_name: version, dataset_name: dataset, status: 'completed',
  created_at: iso(minAgo), completed_at: iso(minAgo - 1),
  results: { accuracy, ...extra } as PipelineTest['results'],
});

function bench(tests: PipelineTest[], trainings: PipelineInput['trainings'] = [], split = false) {
  const rows = buildPipeline({
    models: [{ id: 'a', name: 'a' }, { id: 'b', name: 'b' }, { id: 'c', name: 'c' }],
    datasets: [], trainings, tests, hosts: [],
  });
  return buildBenchmarks(rows, () => 'text', () => split);
}

describe('buildBenchmarks', () => {
  it('rankt nur auf demselben Test-Dataset und haelt Modellfarben fest', () => {
    const { tasks, withoutResult } = bench([
      test('1', 'a', 'reviews', 0.88, 30, { total_samples: 500, average_inference_time: 0.014 }),
      test('2', 'b', 'reviews', 0.91, 20, { total_samples: 500, average_inference_time: 0.041 }),
      test('3', 'a', 'tweets', 0.79, 10, { total_samples: 200 }),
    ]);
    expect(withoutResult.map(r => r.name)).toEqual(['c']);
    const [task] = tasks;
    expect(task.series.map(m => [m.model, m.color, m.shade])).toEqual([['a', 1, 1], ['b', 2, 1]]);
    const reviews = task.datasets.find(d => d.name === 'reviews')!;
    expect(reviews.ranked).toBe(true);
    expect(reviews.results.map(r => [r.model, r.rank, Math.round(r.msPerSample!)])).toEqual([['b', 1, 41], ['a', 2, 14]]);
    const tweets = task.datasets.find(d => d.name === 'tweets')!;
    expect(tweets.ranked).toBe(false);
    expect(tweets.results[0].rank).toBeNull();
  });

  it('nimmt je Modell und Dataset den neuesten Test und meldet deutliche Tempo-Aenderungen', () => {
    const { tasks } = bench([
      test('new', 'a', 'reviews', 0.9, 10, { average_inference_time: 0.037 }),
      test('old', 'a', 'reviews', 0.9, 100, { average_inference_time: 0.079 }),
    ]);
    const r = tasks[0].datasets[0].results[0];
    expect(Math.round(r.msPerSample!)).toBe(37);
    expect(Math.round(r.msBefore!)).toBe(79);
  });

  it('warnt bei wenigen Beispielen und bei Test auf den Trainingsdaten', () => {
    const trainings = [{ id: 't', model_id: 'a', model_name: 'a', dataset_name: 'own', status: 'completed', created_at: iso(300), started_at: iso(300), completed_at: iso(290), progress: { train_loss: 0.2 } }];
    const r = bench([test('1', 'a', 'own', 1, 10, { total_samples: 2 })], trainings).tasks[0].datasets[0].results[0];
    expect(r.fewSamples).toBe(true);
    expect(r.sameData).toBe(true);
    expect(bench([test('1', 'a', 'own', 1, 10, { total_samples: 2 })], trainings, true).tasks[0].datasets[0].results[0].sameData).toBe(false);
  });

  it('fuehrt gescheiterte Tests ohne Ergebnis als offen in ihrer Aufgabe', () => {
    const { tasks } = bench([{ ...test('1', 'a', 'x', 0, 10), status: 'failed', results: null }]);
    expect(tasks[0].pending.map(r => r.name)).toEqual(['a']);
    expect(tasks[0].datasets).toEqual([]);
  });

  it('fuehrt jede getestete Version als eigenen Eintrag und vergleicht sie', () => {
    const { tasks } = bench([
      test('3', 'a', 'reviews', 0.91, 10, {}, 'v2'),
      test('2', 'a', 'reviews', 0.88, 50, {}, 'v1'),
      test('1', 'b', 'reviews', 0.90, 30, {}, 'v3'),
    ]);
    const [task] = tasks;
    expect(task.modelCount).toBe(2);
    expect(task.series.map(s => [s.model, s.version, s.shade, s.multi])).toEqual([
      ['a', 'v1', 0.7, true], ['a', 'v2', 1, true], ['b', 'v3', 1, false],
    ]);
    const set = task.datasets[0];
    expect(set.results.map(r => [r.model, r.version, r.rank])).toEqual([['a', 'v2', 1], ['b', 'v3', 2], ['a', 'v1', 3]]);
    expect(set.steps).toHaveLength(1);
    expect(set.steps[0]).toMatchObject({ model: 'a', newer: 'v2', older: 'v1' });
    expect(set.steps[0].points).toBeCloseTo(3, 5);
  });
});

describe('Beschriftungen', () => {
  it('kuerzt den Modellnamen aus dem Versionsnamen', async () => {
    const { versionLabel } = await import('../homeBenchmarks');
    expect(versionLabel('bench-electra-small', 'bench-electra-small v1')).toBe('v1');
    expect(versionLabel('t1', 'v2')).toBe('v2');
    expect(versionLabel('ner-distilbert', 'Original')).toBe('Original');
    expect(versionLabel('myFirstTextclassifier', 'myFirstTextclassifier')).toBe('myFirstTextclassifier');
    expect(versionLabel('x', null)).toBeNull();
  });

  it('fasst deckungsgleiche Punkte im Streudiagramm zusammen und vermeidet Ueberlappung', async () => {
    const { placeScatterLabels } = await import('../BenchmarkView');
    const labels = placeScatterLabels([
      { x: 650, y: 60, model: 'bench-electra-small', version: 'v1' },
      { x: 653, y: 62, model: 'bench-electra-small', version: 'v2' },
      { x: 300, y: 64, model: 'bench-distilbert', version: null },
    ], 720);
    expect(labels.map(l => l.text).sort()).toEqual(['bench-distilbert', 'bench-electra-small · v1, v2']);
    const electra = labels.find(l => l.text.startsWith('bench-electra'))!;
    expect(electra.anchor).toBe('end');
    const ys = labels.map(l => l.y);
    expect(new Set(ys).size).toBe(2);
  });
});

describe('Lupe im Streudiagramm', () => {
  it('findet die Gruppe zu dicht liegender Punkte, verkettet', async () => {
    const { findTightCluster } = await import('../BenchmarkView');
    expect(findTightCluster([
      { key: 'd', x: 319, y: 54 }, { key: 'e1', x: 370, y: 49 }, { key: 'e2', x: 381, y: 49 }, { key: 'e3', x: 391, y: 50 },
    ]).sort()).toEqual(['e1', 'e2', 'e3']);
    expect(findTightCluster([{ key: 'a', x: 0, y: 0 }, { key: 'b', x: 50, y: 0 }])).toEqual([]);
  });
});

describe('Beschriftung neben der Lupe', () => {
  it('weicht dem Lupen-Rahmen aus', async () => {
    const { placeScatterLabels } = await import('../BenchmarkView');
    const [label] = placeScatterLabels([{ x: 319, y: 54, model: 'bench-distilbert', version: null }], 480,
      [{ left: 359, right: 392, top: 38, bottom: 60 }]);
    expect(label.anchor).not.toBe('start');
  });
});
