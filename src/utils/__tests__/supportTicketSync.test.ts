import { beforeEach, describe, expect, it, vi } from 'vitest';

const { invokeMock } = vi.hoisted(() => ({ invokeMock: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }));

import { readStoredTickets, storeTicket, syncSupportTickets } from '../supportTicket';

const T = (id: number) => ({ ticket_id: id, user_token: `tok${id}`, subject: `S${id}` });

describe('syncSupportTickets', () => {
  beforeEach(() => { localStorage.clear(); invokeMock.mockReset(); });

  it('schickt lokale Tickets und uebernimmt die des Kontos', async () => {
    storeTicket('u1', T(3));
    invokeMock.mockResolvedValue([T(8), T(3)]);
    const list = await syncSupportTickets('u1', 'ft_key');
    expect(invokeMock).toHaveBeenCalledWith('sync_support_tickets', { apiKey: 'ft_key', tickets: [T(3)] });
    expect(list.map((t) => t.ticket_id).sort()).toEqual([3, 8]);
    expect(readStoredTickets('u1')).toHaveLength(2);
  });

  it('bleibt ohne Netz bei der lokalen Liste', async () => {
    storeTicket('u1', T(3));
    invokeMock.mockRejectedValue(new Error('offline'));
    expect(await syncSupportTickets('u1', 'ft_key')).toEqual([T(3)]);
  });

  it('fragt ohne API-Key gar nicht erst', async () => {
    expect(await syncSupportTickets('u1', '')).toEqual([]);
    expect(invokeMock).not.toHaveBeenCalled();
  });
});
