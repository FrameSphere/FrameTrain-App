import { describe, it, expect } from 'vitest';
import { buildCompare, buildPipeline, homeHeadline, testMetric, type PipelineInput, type PipelineTest, type PipelineTraining } from '../homePipeline';

const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60000).toISOString();

const train = (over: Partial<PipelineTraining> = {}): PipelineTraining => ({
  id: 'j1', model_id: 'm1', model_name: 'bert', dataset_id: 'd1', dataset_name: 'reviews', status: 'completed',
  created_at: iso(120), started_at: iso(120), completed_at: iso(100), progress: { train_loss: 0.3, val_loss: 0.31 }, ...over,
});
const test_ = (over: Partial<PipelineTest> = {}): PipelineTest => ({
  id: 't1', model_id: 'm1', model_name: 'bert', version_id: 'v1', version_name: 'v1', dataset_name: 'holdout', status: 'completed',
  created_at: iso(60), started_at: iso(60), completed_at: iso(50), results: { accuracy: 0.9 }, ...over,
});
const input = (over: Partial<PipelineInput> = {}): PipelineInput => ({
  models: [{ id: 'm1', name: 'bert' }], datasets: [], trainings: [], tests: [], hosts: [], ...over,
});
const one = (over: Partial<PipelineInput>) => buildPipeline(input(over))[0];

describe('buildPipeline', () => {
  describe('Rueckschritt', () => {
    const setup = { epochs: 3, learning_rate: 0.001, batch_size: 8 };
    const run = (id: string, minAgo: number, loss: number, over: Partial<PipelineTraining> = {}) => train({
      id, created_at: iso(minAgo), started_at: iso(minAgo), completed_at: iso(minAgo - 1), progress: { train_loss: loss }, config: setup, ...over,
    });

    it('meldet einen deutlich schlechteren Lauf bei gleichen Daten und Einstellungen', () => {
      const row = one({ trainings: [run('a', 200, 0.4), run('b', 100, 0.6)] });
      expect(row.note).toEqual({ kind: 'regression', params: { best: '0.4000', last: '0.6000' } });
      expect(row.group).toBe('attention');
    });

    it('vergleicht nicht ueber andere Daten oder Einstellungen hinweg', () => {
      expect(one({ trainings: [run('a', 200, 0.4, { dataset_name: 'andere' }), run('b', 100, 2.9)] }).note).toBeNull();
      expect(one({ trainings: [run('a', 200, 0.4, { config: { ...setup, epochs: 5 } }), run('b', 100, 2.9)] }).note).toBeNull();
      expect(one({ trainings: [run('a', 200, 0.4, { config: undefined }), run('b', 100, 2.9, { config: undefined })] }).note).toBeNull();
    });

    it('bleibt bei kleinen Schwankungen und nach einer Verbesserung still', () => {
      expect(one({ trainings: [run('a', 200, 0.4), run('b', 100, 0.45)] }).note).toBeNull();
      expect(one({ trainings: [run('a', 300, 0.4), run('b', 200, 0.9), run('c', 100, 0.41)] }).note).toBeNull();
    });
  });

  it('schlaegt ohne Daten zuerst ein Dataset vor', () => {
    const row = one({});
    expect(row.next).toBe('addDataset');
    expect(row.group).toBe('next');
  });

  it('nimmt das Dataset des Modells, wenn es noch nie trainiert wurde', () => {
    const row = one({ datasets: [{ id: 'd9', name: 'eigene', model_id: 'm1' }] });
    expect(row.data.name).toBe('eigene');
    expect(row.next).toBe('startTest');
  });

  it('fuehrt nach dem Training zum Test und nach dem Test zum Hosting', () => {
    expect(one({ trainings: [train()] }).next).toBe('startTest');
    expect(one({ trainings: [train()], tests: [test_()] }).next).toBe('host');
  });

  it('zeigt ein gehostetes Modell ohne weitere Aktion', () => {
    const row = one({ trainings: [train()], tests: [test_()], hosts: [{ model_id: 'm1', status: 'ready' }] });
    expect(row.group).toBe('hosted');
    expect(row.next).toBeNull();
  });

  it('zeigt einen gestoppten oder nicht geladenen Host weiter als gehostet', () => {
    const row = one({ trainings: [train()], tests: [test_()], hosts: [{ model_id: 'm1', status: 'sleeping' }] });
    expect(row.hosting.state).toBe('idle');
    expect(row.next).toBeNull();
  });

  it('wiederholt einen gescheiterten Schritt in seiner eigenen Spalte', () => {
    const failedTest = one({ trainings: [train()], tests: [test_({ status: 'failed', results: null, error: 'boom' })] });
    expect(failedTest.next).toBe('retryTest');
    expect(failedTest.note).toEqual({ kind: 'testFailed', params: { error: 'boom' } });
    expect(failedTest.group).toBe('attention');

    const failedTrain = one({ trainings: [train({ status: 'failed', error: 'oom', progress: {} })] });
    expect(failedTrain.next).toBe('retryTrain');
    expect(failedTrain.note?.kind).toBe('trainingFailed');
  });

  it('verlangt nach neuem Training einen neuen Test und vergisst alte Fehler', () => {
    const row = one({
      tests: [test_({ status: 'failed', results: null, completed_at: iso(300), created_at: iso(310) })],
      trainings: [train()],
    });
    expect(row.next).toBe('startTest');
    expect(row.note).toBeNull();
  });

  it('meldet Overfitting nur beim betroffenen Modell', () => {
    const row = one({ trainings: [train({ progress: { train_loss: 0.02, val_loss: 0.09 } })] });
    expect(row.note?.kind).toBe('overfitting');
    expect(row.next).toBe('startTest');
  });

  it('meldet schwache Ergebnisse', () => {
    expect(one({ trainings: [train()], tests: [test_({ results: { accuracy: 0.4 } })] }).note?.kind).toBe('weakResult');
  });

  it('nimmt Dauer und Zeit vom selben juengsten Ereignis', () => {
    const row = one({ trainings: [train()], tests: [test_()] });
    expect(row.last?.kind).toBe('test');
    expect(row.last?.durationSec).toBeCloseTo(600, 0);
  });

  it('ordnet Probleme vor naechsten Schritten vor Gehostetem', () => {
    const rows = buildPipeline(input({
      models: [{ id: 'a', name: 'hosted' }, { id: 'b', name: 'next' }, { id: 'c', name: 'broken' }],
      trainings: [
        train({ id: '1', model_id: 'a', model_name: 'hosted' }),
        train({ id: '2', model_id: 'b', model_name: 'next' }),
        train({ id: '3', model_id: 'c', model_name: 'broken', status: 'failed', progress: {} }),
      ],
      tests: [test_({ model_id: 'a', model_name: 'hosted' })],
      hosts: [{ model_id: 'a', status: 'ready' }],
    }));
    expect(rows.map(r => r.name)).toEqual(['broken', 'next', 'hosted']);
  });

  it('sammelt die Laeufe aelteste zuerst und ignoriert Laeufe ohne Loss', () => {
    const row = one({
      trainings: [
        train({ id: 'neu', progress: { train_loss: 0.2 } }),
        train({ id: 'alt', completed_at: iso(500), created_at: iso(510), started_at: iso(510), progress: { train_loss: 0.5 } }),
        train({ id: 'leer', completed_at: iso(400), created_at: iso(410), progress: { train_loss: 0 } }),
      ],
    });
    expect(row.training.runs.map(r => r.id)).toEqual(['alt', 'neu']);
  });
});

describe('testMetric', () => {
  it('nimmt Accuracy mit Skala, andere Kennzahlen ohne', () => {
    expect(testMetric(test_())?.fraction).toBe(0.9);
    expect(testMetric(test_({ results: { metrics: { map50: 0.7, label: 'x' } } }))).toEqual({ label: 'map50', value: 0.7, fraction: null });
    expect(testMetric(test_({ results: {} }))).toBeNull();
  });
});

describe('buildCompare', () => {
  const rowsFor = (tests: PipelineTest[]) => buildPipeline(input({
    models: [{ id: 'a', name: 'a' }, { id: 'b', name: 'b' }, { id: 'c', name: 'c' }],
    trainings: [],
    tests,
  }));

  it('vergleicht nur innerhalb einer Aufgabe', () => {
    const rows = rowsFor([
      test_({ id: '1', model_id: 'a', model_name: 'a' }),
      test_({ id: '2', model_id: 'b', model_name: 'b' }),
    ]);
    const { groups } = buildCompare(rows, id => (id === 'a' ? 'text' : 'yolo'));
    expect(groups).toHaveLength(2);
    expect(groups.flatMap(g => g.rows).some(r => r.best)).toBe(false);
  });

  it('vergibt "Bestes" nur auf demselben Test-Dataset', () => {
    const same = buildCompare(rowsFor([
      test_({ id: '1', model_id: 'a', model_name: 'a', results: { accuracy: 0.8 } }),
      test_({ id: '2', model_id: 'b', model_name: 'b', results: { accuracy: 0.95 } }),
    ]), () => 'text').groups[0];
    expect(same.sameDataset).toBe(true);
    expect(same.rows[0]).toMatchObject({ best: true, row: { name: 'b' } });

    const mixed = buildCompare(rowsFor([
      test_({ id: '1', model_id: 'a', model_name: 'a', dataset_name: 'x' }),
      test_({ id: '2', model_id: 'b', model_name: 'b', dataset_name: 'y' }),
    ]), () => 'text').groups[0];
    expect(mixed.rows.some(r => r.best)).toBe(false);
  });

  it('fuehrt Modelle ohne Test getrennt', () => {
    const { withoutResult } = buildCompare(rowsFor([test_({ model_id: 'a', model_name: 'a' })]), () => 'text');
    expect(withoutResult.map(r => r.name)).toEqual(['b', 'c']);
  });
});

describe('homeHeadline', () => {
  it('nennt erst laufendes Training, dann offene Punkte', () => {
    const rows = buildPipeline(input({ trainings: [train({ status: 'failed', progress: {} })] }));
    expect(homeHeadline(rows, 'bert', true)).toEqual({ kind: 'training', model: 'bert' });
    expect(homeHeadline(rows, null, false)).toEqual({ kind: 'attention', count: 1 });
    expect(homeHeadline([], null, false)).toEqual({ kind: 'empty' });
    expect(homeHeadline(buildPipeline(input()), null, false)).toEqual({ kind: 'noData' });
    expect(homeHeadline(buildPipeline(input({ trainings: [train()] })), null, false)).toEqual({ kind: 'clear' });
  });
});
