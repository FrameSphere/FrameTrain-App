// system_info darf nicht auf die langsame Python-Pruefung warten (Live-Test 1.6.14: 12,7 s).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn() }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: async () => '9.9.9' }));

import { registry } from '../registry';
import { INFO_WAIT_MS, resetProbeCache } from '../tools/system';
import type { McpCallContext } from '../types';

const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'local', userId: 'u1' };
const run = () => registry.get('system_info')!.run({}, ctx);

let releasePython: (() => void) | null = null;

beforeEach(() => {
  vi.useFakeTimers();
  resetProbeCache();
  mockInvoke.mockReset();
  mockInvoke.mockImplementation(async (cmd: string) => {
    if (cmd === 'get_system_ram_gb') return 32;
    if (cmd === 'get_python_info') return { version: '3.11' };
    if (cmd === 'check_training_requirements') {
      await new Promise<void>((r) => { releasePython = r; });
      return { mps_available: true };
    }
    throw new Error(`unexpected ${cmd}`);
  });
});
afterEach(() => vi.useRealTimers());

describe('system_info', () => {
  it('antwortet nach der Wartezeit mit Hinweis, statt auf Python zu warten', async () => {
    const p = run();
    await vi.advanceTimersByTimeAsync(INFO_WAIT_MS + 10);
    const r = await p;
    expect(r.isError).toBeFalsy();
    expect(r.content[0].text).toMatch(/still being checked/);
    expect((r.structuredContent as { requirements: unknown }).requirements).toEqual({ status: 'checking' });
  });

  it('nutzt danach das gemerkte Ergebnis ohne neuen Python-Aufruf', async () => {
    const first = run();
    await vi.advanceTimersByTimeAsync(INFO_WAIT_MS + 10);
    await first;
    releasePython?.();
    await vi.advanceTimersByTimeAsync(0);

    const second = await run();
    expect(second.content[0].text).toMatch(/Apple MPS GPU/);
    expect(mockInvoke.mock.calls.filter((c) => c[0] === 'check_training_requirements')).toHaveLength(1);
  });
});
