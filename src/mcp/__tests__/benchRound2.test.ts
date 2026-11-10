// Befunde aus dem zweiten Benchmark-Lauf (09.11.): Einzeltest ohne Ergebnis,
// falsche Analyse-Zahlen, rohe Uebersetzungsschluessel, kein Warten auf Trainings,
// Labor meldete "nicht gestartet", Detektor- und LLM-Kennzahlen ohne Skala.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: async () => '9.9.9' }));

import { registry } from '../registry';
import type { McpCallContext } from '../types';
import { dedupeSteps, summarizeMetrics } from '../tools/analysis';
import { resolveI18n } from '../tools/system';
import { etaSeconds } from '../tools/training';
import { testMetric } from '../../components/homePipeline';

const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'local', userId: 'u1' };
const run = (name: string, args: Record<string, unknown> = {}) => registry.get(name)!.run(args, ctx);
const data = (r: Awaited<ReturnType<typeof run>>) => r.structuredContent as Record<string, unknown>;

beforeEach(() => { mockInvoke.mockReset(); });

describe('tests_single_input', () => {
  it('wartet auf das Ergebnis statt nur die Test-ID zu liefern', async () => {
    mockInvoke.mockImplementation(async (cmd: string, args: Record<string, unknown>) => {
      if (cmd === 'test_single_input') {
        expect(args.wait).toBe(true);
        return JSON.stringify({ mode: 'single', predicted_output: 'neg', confidence: 0.9 });
      }
      throw new Error(`unexpected ${cmd}`);
    });
    const r = await run('tests_single_input', { version_id: 'v1', input: 'bad film', input_type: 'text' });
    expect(data(r).result).toMatchObject({ predicted_output: 'neg' });
    expect(r.content?.[0]).toMatchObject({ text: expect.stringContaining('neg') });
  });
});

describe('Analyse', () => {
  const steps = [
    { epoch: 1, step: 260, train_loss: 0.42, val_loss: null },
    { epoch: 2, step: 267, train_loss: 0.42, val_loss: 0.432 },
    { epoch: 2, step: 534, train_loss: 0.30, val_loss: 0.390 },
    { epoch: 3, step: 801, train_loss: 0.217, val_loss: 0.459 },
    { epoch: 3, step: 801, train_loss: 0.333, val_loss: null, learning_rate: 0.0001 },
    { epoch: 3, step: 801, train_loss: 0.333, val_loss: 0.459, learning_rate: 0.0001 },
  ];

  it('entfernt Abschlusszeilen desselben Schritts', () => {
    expect(dedupeSteps(steps)).toHaveLength(4);
  });

  it('nimmt das echte Minimum und erkennt Overfitting', () => {
    const s = summarizeMetrics({ final_val_loss: 0.459, best_epoch: 2 }, { step_logs: steps })!;
    expect(s.best_val_loss).toBe(0.39);
    expect(s.points).toBe(4);
    expect(s.overfitting_hint).toBe(true);
  });

  it('kuert keinen Besten ueber verschiedene Datasets', async () => {
    mockInvoke.mockImplementation(async (cmd: string, args: { versionId: string }) => {
      if (cmd === 'get_training_metrics') return { final_val_loss: 0.4 };
      if (cmd === 'get_training_full_data') return { config: { dataset_path: `/d/${args.versionId === 'a' ? 'ds_1' : 'ds_2'}` }, step_logs: [{ step: 1, val_loss: 0.4 }] };
      throw new Error(`unexpected ${cmd}`);
    });
    const d = data(await run('analysis_runs_compare', { version_ids: ['a', 'b'] }));
    expect(d.best_version_id).toBeNull();
  });
});

describe('system_capabilities', () => {
  it('loest Uebersetzungsschluessel auf', () => {
    expect(resolveI18n('firstLaunch.pluginRegistry.yolo.name')).toBe('YOLO Object Detection (Ultralytics)');
    expect(resolveI18n('Plain name')).toBe('Plain name');
    expect(resolveI18n('firstLaunch.gibtEsNicht.name')).toBe('firstLaunch.gibtEsNicht.name');
  });
});

describe('training_status', () => {
  it('schaetzt die Restzeit aus Fortschritt und Laufzeit', () => {
    const job = { id: 't', model_id: 'm', model_name: 'm', dataset_id: 'd', dataset_name: 'd', status: 'running',
      started_at: '2026-11-09T20:00:00Z', progress: { progress_percent: 25 } };
    expect(etaSeconds(job, Date.parse('2026-11-09T20:01:00Z'))).toBe(180);
    expect(etaSeconds({ ...job, status: 'completed' })).toBeNull();
  });
});

describe('lab_server_status', () => {
  it('meldet einen bereiten Server als bereit', async () => {
    mockInvoke.mockResolvedValue({ status: 'ready', version_id: 'v3' });
    const r = await run('lab_server_status');
    expect(r.content?.[0]).toMatchObject({ text: expect.stringContaining('ready') });
  });
});

describe('Benchmark-Kennzahl', () => {
  it('rankt Detektoren nach mAP50 und Sprachmodelle nach ROUGE-L', () => {
    const det = testMetric({ results: { total_samples: 16, accuracy: null, metrics: { mAP50: 0.56, 'mAP50-95': 0.42 } } } as never);
    expect(det).toEqual({ label: 'mAP50', value: 0.56, fraction: 0.56 });
    const llm = testMetric({ results: { total_samples: 30, accuracy: null, metrics: { exact_match: 0.03, rouge_l: 0.32 } } } as never);
    expect(llm).toEqual({ label: 'rouge_l', value: 0.32, fraction: 0.32 });
  });
});
