// tests_status und tests_history aus Sicht eines Assistenten (Benchmark-Lauf
// 09.11.): nach dem Ende kam { job: null }, der Verlauf stand aelteste zuerst
// und ohne Ergebnis.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: async () => '9.9.9' }));

import { registry } from '../registry';
import type { McpCallContext } from '../types';

const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'local', userId: 'u1' };
const run = (name: string, args: Record<string, unknown> = {}) => registry.get(name)!.run(args, ctx);
const data = (r: Awaited<ReturnType<typeof run>>) => r.structuredContent as Record<string, unknown>;

const job = (id: string, completed: string, accuracy: number) => ({
  id, status: 'completed', created_at: completed, completed_at: completed, version_id: 'v1',
  results: { total_samples: 1066, accuracy, average_inference_time: 0.0214, predictions: [] },
});
const HISTORY = [job('old', '2026-11-09T13:00:00Z', 0.82), job('new', '2026-11-09T14:00:00Z', 0.8386)];

function backend(handlers: Record<string, unknown>) {
  mockInvoke.mockImplementation(async (cmd: string) => {
    if (!(cmd in handlers)) throw new Error(`unexpected command ${cmd}`);
    return handlers[cmd];
  });
}

beforeEach(() => { mockInvoke.mockReset(); });

describe('tests_status', () => {
  it('meldet nach dem Ende den zuletzt beendeten Test mit Ergebnis', async () => {
    backend({ get_active_test_job: null, get_current_test: null, get_test_history: HISTORY });
    const d = data(await run('tests_status'));
    expect(d.running).toBe(false);
    expect((d.job as { id: string }).id).toBe('new');
    expect(d.result).toMatchObject({ total_samples: 1066, accuracy: 0.8386, ms_per_sample: 21.4 });
  });

  it('meldet einen laufenden Test als running', async () => {
    backend({ get_active_test_job: { id: 'run', status: 'running', progress: { progress_percent: 40 } } });
    const d = data(await run('tests_status'));
    expect(d.running).toBe(true);
    expect((d.job as { status: string }).status).toBe('running');
  });
});

describe('tests_history', () => {
  it('liefert neueste zuerst mit Kennzahlen statt Vorhersagen', async () => {
    backend({ get_test_history: HISTORY });
    const items = data(await run('tests_history', { limit: 1 })).items as Array<Record<string, unknown>>;
    expect(items.map((i) => i.id)).toEqual(['new']);
    expect(items[0].result).toMatchObject({ accuracy: 0.8386, total_samples: 1066 });
    expect(items[0]).not.toHaveProperty('results');
  });
});

describe('models_download_hf', () => {
  it('warnt, wenn das Modell keine erkennbare Architektur hat', async () => {
    const { withTypeWarning } = await import('../tools/models');
    const base = { id: 'm', name: 'x', source: 'hf', local_path: '/m', size_bytes: 1, file_count: 1, created_at: '' };
    expect(withTypeWarning({ ...base, model_type: 'transformer' }).warning).toMatch(/model_type/);
    expect(withTypeWarning({ ...base, model_type: 'bert' }).warning).toBeUndefined();
  });
});
