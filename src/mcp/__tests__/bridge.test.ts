// Bridge-Ablauf mit gemocktem invoke/listen: Manifest melden, mcp://call
// annehmen, Freigabe, Ausfuehrung, mcp_tool_result. Und: ohne Rust-Seite
// laeuft alles weiter.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import { executeCall, MSG_APPROVAL_TIMEOUT, MSG_DENIED, MSG_NOT_SIGNED_IN, startMcpBridge, type BridgeDeps } from '../bridge';
import { createRegistry } from '../registry';
import { listGrants, resetApprovalQueue } from '../approval';
import { defineTool, ok, reqStr, S, ToolFailure } from '../util';
import type { McpCallEvent } from '../types';

type Handler = (e: { payload: unknown }) => void | Promise<void>;
let handlers: Record<string, Handler[]> = {};

const echo = defineTool({
  name: 'models_list', title: 'Echo', description: 'Echo tool for tests, returns its argument back to the caller.',
  properties: { word: S.str('A word.') },
  run: async (args, ctx) => ok(`echo ${args.word ?? ''}`, { word: args.word ?? null, user: ctx.userId }),
});
const guarded = defineTool({
  name: 'training_start', title: 'Start', description: 'Guarded tool for tests that needs explicit user approval.',
  properties: { model_id: S.str('Model.') }, required: ['model_id'], approval: 'always',
  run: async (args) => ok(`started ${reqStr(args, 'model_id')}`),
});
const failing = defineTool({
  name: 'datasets_list', title: 'Fail', description: 'Always fails with a readable message for the tests.',
  run: async () => { throw new ToolFailure('Datensatz kaputt'); },
});
const crashing = defineTool({
  name: 'datasets_info', title: 'Crash', description: 'Throws an unexpected error to test the generic handler.',
  run: async () => { throw new Error('boom'); },
});
const slow = defineTool({
  name: 'lab_infer', title: 'Slow', description: 'Never finishes to test the bridge timeout handling here.',
  run: () => new Promise(() => {}),
});
const sys = defineTool({
  name: 'system_ping', title: 'Ping', description: 'System tool that works without a signed-in user for tests.',
  run: async () => ok('pong'),
});
const reg = createRegistry([echo, guarded, failing, crashing, slow, sys]);

function call(tool: string, args: Record<string, unknown> = {}, client = 'Claude Code'): McpCallEvent {
  return { callId: `c-${Math.random()}`, tool, args, source: 'local', client: { name: client } };
}

function deps(over: Partial<BridgeDeps> = {}): BridgeDeps {
  return { registry: reg, getUserId: () => 'u1', requestApproval: vi.fn(async () => 'allow' as const), focusApp: vi.fn(), ...over };
}

beforeEach(() => {
  localStorage.clear();
  resetApprovalQueue();
  handlers = {};
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockImplementation(async (event: string, cb: Handler) => {
    (handlers[event] ??= []).push(cb);
    return () => { handlers[event] = handlers[event].filter((h) => h !== cb); };
  });
});

describe('executeCall', () => {
  it('fuehrt lesende Werkzeuge ohne Freigabe aus', async () => {
    const d = deps();
    const r = await executeCall(call('models_list', { word: 'hi' }), d);
    expect(r.isError).toBeFalsy();
    expect(r.content[0].text).toBe('echo hi');
    expect(r.structuredContent).toEqual({ word: 'hi', user: 'u1' });
    expect(d.requestApproval).not.toHaveBeenCalled();
  });

  it('unbekanntes Werkzeug und ungueltige Argumente werden isError', async () => {
    expect((await executeCall(call('nope'), deps())).isError).toBe(true);
    const bad = await executeCall(call('models_list', { word: 3 }), deps());
    expect(bad.isError).toBe(true);
    expect(bad.content[0].text).toMatch(/word: expected string/);
    const extra = await executeCall(call('models_list', { other: 1 }), deps());
    expect(extra.content[0].text).toMatch(/unknown argument/);
    const missing = await executeCall(call('training_start', {}), deps());
    expect(missing.content[0].text).toMatch(/model_id: missing/);
  });

  it('ohne angemeldeten Nutzer laufen nur system-Werkzeuge', async () => {
    const d = deps({ getUserId: () => null });
    expect((await executeCall(call('models_list'), d)).content[0].text).toBe(MSG_NOT_SIGNED_IN);
    expect((await executeCall(call('system_ping'), d)).content[0].text).toBe('pong');
  });

  it('Freigabe: Ablehnen, Zeitlimit, Erlauben', async () => {
    const deny = deps({ requestApproval: vi.fn(async () => 'deny' as const) });
    const r1 = await executeCall(call('training_start', { model_id: 'm' }), deny);
    expect(r1).toMatchObject({ isError: true, content: [{ text: MSG_DENIED }] });
    expect(deny.focusApp).toHaveBeenCalled();

    const late = deps({ requestApproval: vi.fn(async () => 'timeout' as const) });
    expect((await executeCall(call('training_start', { model_id: 'm' }), late)).content[0].text).toBe(MSG_APPROVAL_TIMEOUT);

    const allow = deps();
    const r3 = await executeCall(call('training_start', { model_id: 'm' }), allow);
    expect(r3.content[0].text).toBe('started m');
    const req = (allow.requestApproval as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(req.tool.name).toBe('training_start');
    expect(req.args).toEqual({ model_id: 'm' });
    expect(req.client.name).toBe('Claude Code');
    expect(listGrants()).toEqual([]);
  });

  it('"immer erlauben" wird gespeichert und spart den naechsten Dialog', async () => {
    const d = deps({ requestApproval: vi.fn(async () => 'always' as const) });
    await executeCall(call('training_start', { model_id: 'a' }), d);
    await executeCall(call('training_start', { model_id: 'b' }), d);
    expect(d.requestApproval).toHaveBeenCalledTimes(1);
    expect(listGrants()).toHaveLength(1);
  });

  it('Fehler im Werkzeug werden lesbare isError-Ergebnisse', async () => {
    expect(await executeCall(call('datasets_list'), deps())).toMatchObject({ isError: true, content: [{ text: 'Datensatz kaputt' }] });
    expect((await executeCall(call('datasets_info'), deps())).content[0].text).toBe('datasets_info failed: boom');
  });

  it('antwortet rechtzeitig, auch wenn ein Werkzeug haengt', async () => {
    const r = await executeCall(call('lab_infer'), deps({ timeoutMs: 5000 }));
    expect(r.isError).toBe(true);
    expect(r.content[0].text).toMatch(/did not finish/);
  }, 10_000);
});

describe('startMcpBridge', () => {
  it('meldet das Manifest, nimmt Aufrufe an und schickt das Ergebnis zurueck', async () => {
    mockInvoke.mockResolvedValue(undefined);
    const stop = await startMcpBridge({ registry: reg, getUserId: () => 'u1', requestApproval: async () => 'allow', focusApp: () => {} });
    const reg1 = mockInvoke.mock.calls.find((c) => c[0] === 'mcp_register_manifest');
    expect(reg1?.[1].tools.map((t: { name: string }) => t.name)).toEqual(reg.manifest().map((t) => t.name));
    expect(reg1?.[1].tools[0].run).toBeUndefined();

    await handlers['mcp://call'][0]({ payload: { callId: 'x1', tool: 'models_list', args: { word: 'yo' }, source: 'remote', client: { name: 'ChatGPT' } } });
    const res = mockInvoke.mock.calls.find((c) => c[0] === 'mcp_tool_result');
    expect(res?.[1]).toEqual({ callId: 'x1', result: { content: [{ type: 'text', text: 'echo yo' }], structuredContent: { word: 'yo', user: 'u1' } } });

    // Kaputte Payloads werden ignoriert, nicht beantwortet
    mockInvoke.mockClear();
    await handlers['mcp://call'][0]({ payload: { nope: true } });
    expect(mockInvoke).not.toHaveBeenCalled();
    stop();
    expect(handlers['mcp://call']).toEqual([]);
  });

  it('laeuft ohne Rust-Commands weiter (alles abgesichert)', async () => {
    mockInvoke.mockRejectedValue('command mcp_register_manifest not found');
    const info = vi.spyOn(console, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const stop = await startMcpBridge({ registry: reg, getUserId: () => 'u1', requestApproval: async () => 'allow', focusApp: () => {} });
    await expect(handlers['mcp://call'][0]({ payload: { callId: 'x2', tool: 'models_list', args: {}, source: 'local', client: { name: 'c' } } })).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
    stop();
    info.mockRestore(); warn.mockRestore();
  });

  it('startet nur einmal', async () => {
    mockInvoke.mockResolvedValue(undefined);
    const a = startMcpBridge({ registry: reg });
    const b = startMcpBridge({ registry: reg });
    expect(a).toBe(b);
    (await a)();
  });
});
