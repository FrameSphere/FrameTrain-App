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
