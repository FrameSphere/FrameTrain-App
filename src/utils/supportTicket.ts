// Support-Tickets an den FrameSphere Manager (WebControl HQ).
//
// Bis 1.2.81 lag diese Logik nur in Settings.tsx. Seit der Plugin-Wunsch aus
// der Modell-Verwaltung denselben Weg nimmt, liegt sie hier: ein Ticket, eine
// Ablage, ein Thread — der Nutzer findet die Antwort in den Einstellungen
// wieder, egal wo er das Ticket geoeffnet hat.
//
// Bewusst getrennt vom Fehler-Kanal (siehe errorReport.ts): ein Wunsch ist
// kein Fehler und hat in der Auto-Fix-Pipeline nichts zu suchen.

export const MANAGER_API = 'https://webcontrol-hq-api.karol-paschek.workers.dev';

export interface StoredTicket {
  ticket_id: number;
  user_token: string;
  subject: string;
}

function storageKey(userId: string): string {
  return `ft_tickets_${userId || 'anon'}`;
}

/** Alle lokal bekannten Tickets eines Nutzers, neuestes zuerst. */
export function readStoredTickets(userId: string): StoredTicket[] {
  try {
    return JSON.parse(localStorage.getItem(storageKey(userId)) || '[]');
  } catch {
    return [];
  }
}

/** Legt ein Ticket ab; ein bereits bekanntes wird ersetzt statt verdoppelt. */
export function storeTicket(userId: string, ticket: StoredTicket): void {
  const list = readStoredTickets(userId).filter(x => x.ticket_id !== ticket.ticket_id);
  try {
    localStorage.setItem(storageKey(userId), JSON.stringify([ticket, ...list]));
  } catch {
    // Privater Modus o. ae. — das Ticket existiert serverseitig trotzdem.
  }
}

export interface SupportTicketInput {
  userId: string;
  email?: string | null;
  subject: string;
  message: string;
}

/**
 * Schickt ein Support-Ticket und legt es lokal ab.
 *
 * Wirft bei Netzwerkfehler oder wenn der Manager `success: false` meldet —
 * der Aufrufer entscheidet, was der Nutzer dann sieht.
 */
export async function submitSupportTicket(input: SupportTicketInput): Promise<StoredTicket> {
  const res = await fetch(`${MANAGER_API}/api/support/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: input.userId,
      name: input.email?.split('@')[0] || 'FrameTrain User',
      email: input.email || '',
      subject: input.subject.trim(),
      message: input.message.trim(),
    }),
  });
  const data = await res.json();
  if (!data.success) throw new Error('Manager hat das Ticket abgelehnt');

  const ticket: StoredTicket = {
    ticket_id: data.ticket_id,
    user_token: data.user_token,
    subject: input.subject.trim(),
  };
  storeTicket(input.userId, ticket);
  return ticket;
}

/**
 * Gleicht die lokalen Tickets mit dem Konto ab (frame-train.com, per API-Key).
 * Danach kennt die App auch Tickets aus Website-Chat und Dashboard, und die
 * Website kennt die aus der App. Ohne Netz bleibt die lokale Liste unveraendert.
 */
export async function syncSupportTickets(userId: string, apiKey: string): Promise<StoredTicket[]> {
  const local = readStoredTickets(userId);
  if (!apiKey) return local;
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    const remote = await invoke<StoredTicket[]>('sync_support_tickets', { apiKey, tickets: local });
    if (Array.isArray(remote)) {
      for (const t of [...remote].reverse()) {
        if (t && typeof t.ticket_id === 'number' && typeof t.user_token === 'string') storeTicket(userId, t);
      }
    }
  } catch {
    // Offline oder Website nicht erreichbar: lokale Tickets reichen.
  }
  return readStoredTickets(userId);
}
