import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { setCurrentUser, userScopedKey } from '../currentUser';
import { GRANTS_KEY, listGrants, needsApproval } from '../../mcp/approval';
import type { Tool } from '../../mcp/types';

describe('Daten pro Account', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => setCurrentUser(null));

  it('trennt Schluessel nach angemeldetem Account', () => {
    setCurrentUser('a');
    expect(userScopedKey('ft_training_sessions')).toBe('ft_training_sessions_a');
    setCurrentUser('b');
    expect(userScopedKey('ft_training_sessions')).toBe('ft_training_sessions_b');
    setCurrentUser(null);
    expect(userScopedKey('ft_training_sessions')).toBe('ft_training_sessions_anon');
  });

  it('eine "immer erlauben"-Freigabe von Account A gilt nicht fuer Account B', () => {
    setCurrentUser('a');
    localStorage.setItem(userScopedKey(GRANTS_KEY), JSON.stringify({
      'Claude Code': { training_start: { kind: 'always', grantedAt: '2026-11-08T10:00:00Z', clientName: 'Claude Code' } },
    }));
    const tool = { name: 'training_start', annotations: {}, _meta: { 'frametrain/approval': 'always' } } as unknown as Tool;
    expect(listGrants()).toHaveLength(1);
    expect(needsApproval(tool, { name: 'claude-code' })).toBe(false);

    setCurrentUser('b');
    expect(listGrants()).toHaveLength(0);
    expect(needsApproval(tool, { name: 'claude-code' })).toBe(true);
  });

  it('alte gemeinsame Freigaben werden keinem Account zugeschlagen', () => {
    localStorage.setItem(GRANTS_KEY, JSON.stringify({
      'Claude Code': { training_start: { kind: 'always', grantedAt: '2026-11-08T10:00:00Z', clientName: 'Claude Code' } },
    }));
    setCurrentUser('a');
    expect(listGrants()).toHaveLength(0);
  });
});
