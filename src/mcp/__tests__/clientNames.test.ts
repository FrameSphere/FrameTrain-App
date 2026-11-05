import { beforeEach, describe, expect, it } from 'vitest';
import { friendlyClientName, UNKNOWN_CLIENT } from '../clientNames';
import { clientKey, GRANTS_KEY, listGrants, needsApproval } from '../approval';
import type { Tool } from '../types';

describe('friendlyClientName', () => {
  it('bildet technische Kennungen auf Produktnamen ab', () => {
    expect(friendlyClientName('local-agent-mode-frametrain')).toBe('Claude Code');
    expect(friendlyClientName('claude-code')).toBe('Claude Code');
    expect(friendlyClientName('claude-ai')).toBe('Claude');
    expect(friendlyClientName('Claude')).toBe('Claude');
    expect(friendlyClientName('openai-mcp')).toBe('ChatGPT');
    expect(friendlyClientName('cursor-vscode')).toBe('Cursor');
    expect(friendlyClientName('my-own-agent')).toBe('my-own-agent');
    expect(friendlyClientName('')).toBe(UNKNOWN_CLIENT);
    expect(friendlyClientName(undefined)).toBe(UNKNOWN_CLIENT);
  });

  it('Freigaben gelten fuer alle Kennungen desselben Programms', () => {
    expect(clientKey('local-agent-mode-frametrain')).toBe(clientKey('claude-code'));
    expect(clientKey({ name: 'claude-ai' })).toBe(clientKey('Claude'));
  });
});

describe('Freigaben unter alter Kennung', () => {
  beforeEach(() => localStorage.clear());

  it('werden beim Lesen auf den Produktnamen umgezogen', () => {
    localStorage.setItem(GRANTS_KEY, JSON.stringify({
      'local-agent-mode-frametrain': { training_start: { kind: 'always', grantedAt: '2026-11-05T10:00:00Z', clientName: 'local-agent-mode-frametrain' } },
    }));
    const grants = listGrants();
    expect(grants).toHaveLength(1);
    expect(grants[0].client).toBe('Claude Code');
    const tool = { name: 'training_start', annotations: {}, _meta: { 'frametrain/approval': 'always' } } as unknown as Tool;
    expect(needsApproval(tool, { name: 'claude-code' })).toBe(false);
  });
});
