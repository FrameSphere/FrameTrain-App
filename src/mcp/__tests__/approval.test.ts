// Freigabe-Logik: none/once/always, "immer erlauben", destruktive Werkzeuge,
// Widerruf, Warteschlange mit Zeitlimit.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  answerApproval, listGrants, needsApproval, recordDecision, requestApprovalViaDialog, resetApprovalQueue,
  revokeAllGrants, revokeClient, revokeGrant, subscribeApprovals, subscribeGrants,
} from '../approval';
import { defineTool, ok } from '../util';

const noop = async () => ok('ok');
const readTool = defineTool({ name: 'models_list', title: 'L', description: 'x'.repeat(50), run: noop });
const onceTool = defineTool({ name: 'models_rename', title: 'R', description: 'x'.repeat(50), approval: 'once', run: noop });
const alwaysTool = defineTool({ name: 'training_start', title: 'S', description: 'x'.repeat(50), approval: 'always', run: noop });
const deleteTool = defineTool({
  name: 'models_delete', title: 'D', description: 'x'.repeat(50), approval: 'always',
  annotations: { destructiveHint: true }, run: noop,
});
const claude = { name: 'Claude Desktop' };
const gpt = { name: 'ChatGPT' };

beforeEach(() => { localStorage.clear(); resetApprovalQueue(); });
afterEach(() => { vi.useRealTimers(); });

describe('Freigabe-Regeln', () => {
  it('lesende Werkzeuge brauchen keine Freigabe', () => {
    expect(needsApproval(readTool, claude)).toBe(false);
  });

  it('once fragt einmal je Client und merkt sich die Zustimmung', () => {
    expect(needsApproval(onceTool, claude)).toBe(true);
    recordDecision(onceTool, claude, 'allow');
    expect(needsApproval(onceTool, claude)).toBe(false);
    expect(needsApproval(onceTool, { name: ' claude desktop ' })).toBe(false);
    expect(needsApproval(onceTool, gpt)).toBe(true);
  });

  it('always fragt jedes Mal, ausser "immer erlauben" wurde gewaehlt', () => {
    recordDecision(alwaysTool, claude, 'allow');
    expect(needsApproval(alwaysTool, claude)).toBe(true);
    recordDecision(alwaysTool, claude, 'always');
    expect(needsApproval(alwaysTool, claude)).toBe(false);
    expect(needsApproval(alwaysTool, gpt)).toBe(true);
  });

  it('destruktive Werkzeuge fragen immer, auch nach "immer erlauben"', () => {
    recordDecision(deleteTool, claude, 'always');
    expect(needsApproval(deleteTool, claude)).toBe(true);
    expect(listGrants()).toEqual([]);
  });

  it('Ablehnen speichert nichts', () => {
    recordDecision(onceTool, claude, 'deny');
    expect(listGrants()).toEqual([]);
  });

  it('Widerruf je Werkzeug, je Client und komplett', () => {
    const changes = vi.fn();
    const off = subscribeGrants(changes);
    recordDecision(onceTool, claude, 'allow');
    recordDecision(alwaysTool, claude, 'always');
    recordDecision(onceTool, gpt, 'allow');
    expect(listGrants().map((g) => `${g.client}/${g.tool}/${g.kind}`)).toEqual([
      'ChatGPT/models_rename/once', 'Claude Desktop/models_rename/once', 'Claude Desktop/training_start/always',
    ]);
    revokeGrant('Claude Desktop', 'training_start');
    expect(needsApproval(alwaysTool, claude)).toBe(true);
    revokeClient('ChatGPT');
    expect(needsApproval(onceTool, gpt)).toBe(true);
    revokeAllGrants();
    expect(listGrants()).toEqual([]);
    expect(changes).toHaveBeenCalledTimes(6);
    off();
  });

  it('defekter Speicher gilt als leer', () => {
    localStorage.setItem('ft_mcp_grants_v1', '{kaputt');
    expect(needsApproval(onceTool, claude)).toBe(true);
    expect(listGrants()).toEqual([]);
  });
});

describe('Warteschlange', () => {
  it('liefert die Antwort des Dialogs und zeigt die naechste Anfrage', async () => {
    const heads: (string | null)[] = [];
    const off = subscribeApprovals((head) => heads.push(head?.tool.name ?? null));
    const p1 = requestApprovalViaDialog({ tool: alwaysTool, args: {}, client: claude, source: 'local' });
    const p2 = requestApprovalViaDialog({ tool: deleteTool, args: { model_id: 'm1' }, client: gpt, source: 'remote' });
    const first = heads[heads.length - 1];
    expect(first).toBe('training_start');
    let current: { id: string; canRemember: boolean } | null = null;
    const off2 = subscribeApprovals((h) => { current = h; });
    answerApproval(current!.id, 'always');
    await expect(p1).resolves.toBe('always');
    expect(current!.canRemember).toBe(false); // Loeschen: keine Merk-Option
    answerApproval(current!.id, 'deny');
    await expect(p2).resolves.toBe('deny');
    expect(current).toBeNull();
    off(); off2();
  });

  it('ohne Antwort laeuft die Anfrage ab', async () => {
    vi.useFakeTimers();
    const p = requestApprovalViaDialog({ tool: alwaysTool, args: {}, client: claude, source: 'local' }, 1000);
    vi.advanceTimersByTime(1001);
    await expect(p).resolves.toBe('timeout');
  });
});

describe('Schnittstelle fuer Strang B (approvals.ts)', () => {
  it('liest und widerruft mit Anzeigenamen', async () => {
    const { listApprovals, revokeApproval, onApprovalsChanged } = await import('../approvals');
    const cb = vi.fn();
    const off = onApprovalsChanged(cb);
    recordDecision(alwaysTool, claude, 'always');
    expect(listApprovals()).toEqual([expect.objectContaining({ client: 'Claude Desktop', tool: 'training_start', kind: 'always' })]);
    revokeApproval('Claude Desktop', 'training_start');
    expect(listApprovals()).toEqual([]);
    expect(cb).toHaveBeenCalledTimes(2);
    off();
  });
});
