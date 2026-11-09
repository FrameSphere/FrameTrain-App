import { describe, it, expect } from 'vitest';
import en from '../../locales/en.json';
import { buildHomeCoachContext } from '../homeCoachContext';
import { buildPipeline, type PipelineInput } from '../homePipeline';

const t = (key: string, params?: Record<string, string>) => {
  const value = key.split('.').reduce<unknown>((o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), en);
  let s = typeof value === 'string' ? value : key;
  for (const [k, v] of Object.entries(params ?? {})) s = s.split(`{${k}}`).join(v);
  return s;
};
const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60000).toISOString();
const input: PipelineInput = {
  models: [{ id: 'm1', name: 'YOLO11' }, { id: 'm2', name: 't5-small' }, { id: 'm3', name: 'leer' }],
  datasets: [{ id: 'd3', name: 'bilder', model_id: 'm3' }],
  trainings: [
    { id: 'a', model_id: 'm1', model_name: 'YOLO11', dataset_name: 'data', status: 'completed', created_at: iso(300), started_at: iso(300), completed_at: iso(290), progress: { train_loss: 6.6288 } },
    { id: 'b', model_id: 'm1', model_name: 'YOLO11', dataset_name: 'data', status: 'stopped', created_at: iso(200), started_at: iso(200), completed_at: iso(190) },
    { id: 'c', model_id: 'm2', model_name: 't5-small', dataset_name: 'trans', status: 'completed', created_at: iso(100), started_at: iso(100), completed_at: iso(90), progress: { train_loss: 2.9755 } },
  ],
  tests: [],
  hosts: [],
};

function context() {
  return buildHomeCoachContext({
    language: 'en', t, ago: () => '2 days ago', number: (v, d) => v.toFixed(d),
    title: '1 model needs you', view: 'pipeline', rows: buildPipeline(input), running: null, sleepRisk: false,
    compare: { groups: [], withoutResult: [] }, groupLabel: k => k,
  });
}

describe('buildHomeCoachContext', () => {
  it('nutzt die Gruppen und Woerter der Oberflaeche', () => {
    const text = context();
    expect(text.split('\n')[0]).toBe('=== FrameTrain Home ===');
    expect(text).toContain('## Needs attention (1)');
    expect(text).toContain('## Ready to test (1)');
    expect(text).toContain('Button "Start test" in column Test');
    expect(text).toContain('Button "Retry" in column Training');
    expect(text).not.toMatch(/startTest|retryTrain|\bnone\b/);
  });

  it('nennt den letzten fertigen Lauf auch bei gestopptem Training', () => {
    expect(context()).toContain('Training: Stopped (last finished run: loss 6.6288)');
  });

  it('zaehlt Modelle ohne Lauf wie die Seite als zugeklappte Gruppe', () => {
    expect(context()).toContain('## Without a run (1, collapsed): leer');
  });
});
