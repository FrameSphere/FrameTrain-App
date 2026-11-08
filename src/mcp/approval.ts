// Freigaben fuer KI-Clients.
//
// Vertrag Abschnitt 1/3: `none` laeuft sofort, `once` fragt beim ersten
// Aufruf je Client und merkt sich die Zustimmung, `always` fragt jedes Mal —
// ausser der Nutzer hat "Fuer diesen Client immer erlauben" gewaehlt.
// Destruktive Werkzeuge (Loeschen) fragen immer, ohne Merk-Option.
//
// Die Lese-/Widerruf-Funktionen (`listGrants`, `revokeGrant`, `revokeClient`,
// `revokeAllGrants`, `subscribeGrants`) sind fuer den Einstellungs-Tab
// "KI-Verbindungen" (Strang B) gedacht.

import { friendlyClientName } from './clientNames';
import type { ApprovalDecision, ApprovalRequest, McpClientInfo, Tool, ToolArgs, McpCallSource } from './types';

export const GRANTS_KEY = 'ft_mcp_grants_v1';

/** `once`: erste Zustimmung zu einem `once`-Werkzeug; `always`: Nutzer hat "immer erlauben" gewaehlt. */
export type GrantKind = 'once' | 'always';

export interface Grant {
  /** Anzeigename des Clients, wie er beim Freigeben hiess. */
  client: string;
  tool: string;
  kind: GrantKind;
  grantedAt: string;
}

type Store = Record<string, Record<string, { kind: GrantKind; grantedAt: string; clientName?: string }>>;

const listeners = new Set<() => void>();

function readStore(): Store {
  try {
    const raw = localStorage.getItem(GRANTS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? migrateNames(parsed as Store) : {};
  } catch {
    return {};
  }
}

/** Alte Eintraege unter technischen Kennungen ("local-agent-mode-x") auf den Produktnamen zusammenfuehren. */
function migrateNames(store: Store): Store {
  const out: Store = {};
  for (const [key, tools] of Object.entries(store)) {
    const sample = Object.values(tools ?? {})[0];
    const label = friendlyClientName(sample?.clientName ?? key);
    const k = clientKey(label);
    for (const [tool, g] of Object.entries(tools ?? {})) {
      out[k] = { ...(out[k] ?? {}), [tool]: { ...g, clientName: label } };
    }
  }
  return out;
}

function writeStore(store: Store) {
  try { localStorage.setItem(GRANTS_KEY, JSON.stringify(store)); } catch { /* Speicher voll */ }
  listeners.forEach((l) => { try { l(); } catch { /* Abonnent defekt */ } });
}

/** Client-Name normalisiert (Gross/Klein und Leerraum spielen keine Rolle). */
export function clientKey(client: McpClientInfo | string): string {
  const name = typeof client === 'string' ? client : client.name;
  return friendlyClientName(name).trim().toLowerCase();
}

/**
 * Unwiderruflich (Loeschen): jedes Mal fragen, nichts merken, Warnung zeigen.
 * Nicht dasselbe wie destructiveHint – das sagt Claude/ChatGPT nur, dass Daten
 * ueberschrieben werden koennen (z. B. Annotation setzen).
 */
export function isDestructive(tool: Tool): boolean {
  const flag = tool._meta['frametrain/irreversible'];
  if (flag !== undefined) return flag === true;
  return tool.annotations.destructiveHint === true && tool._meta['frametrain/approval'] === 'always';
}

/** Muss der Nutzer diesen Aufruf bestaetigen? */
export function needsApproval(tool: Tool, client: McpClientInfo): boolean {
  const level = tool._meta['frametrain/approval'];
  if (level === 'none') return false;
  if (isDestructive(tool)) return true;
  const grant = readStore()[clientKey(client)]?.[tool.name];
  if (!grant) return true;
  if (level === 'once') return false;
  return grant.kind !== 'always';
}

export function canRemember(tool: Tool): boolean {
  return tool._meta['frametrain/approval'] !== 'none' && !isDestructive(tool);
}

/** Merkt eine Zustimmung. `always` nur fuer nicht-destruktive Werkzeuge. */
export function recordDecision(tool: Tool, client: McpClientInfo, decision: ApprovalDecision) {
  if (decision === 'deny') return;
  const level = tool._meta['frametrain/approval'];
  let kind: GrantKind | null = null;
  if (decision === 'always' && canRemember(tool)) kind = 'always';
  else if (level === 'once' && !isDestructive(tool)) kind = 'once';
  if (!kind) return;
  const store = readStore();
  const key = clientKey(client);
  store[key] = { ...(store[key] ?? {}), [tool.name]: { kind, grantedAt: new Date().toISOString(), clientName: client.name } };
  writeStore(store);
}

// ── Lesen / Widerrufen (Einstellungs-Tab) ───────────────────────────────────

export function listGrants(): Grant[] {
  const store = readStore();
  const out: Grant[] = [];
  for (const [client, tools] of Object.entries(store)) {
    for (const [tool, g] of Object.entries(tools ?? {})) {
      out.push({ client: g.clientName ?? client, tool, kind: g.kind, grantedAt: g.grantedAt });
    }
  }
  return out.sort((a, b) => clientKey(a.client).localeCompare(clientKey(b.client)) || a.tool.localeCompare(b.tool));
}

export function revokeGrant(client: string, tool: string) {
  const store = readStore();
  const key = clientKey(client);
  if (!store[key]?.[tool]) return;
  delete store[key][tool];
  if (Object.keys(store[key]).length === 0) delete store[key];
  writeStore(store);
}

export function revokeClient(client: string) {
  const store = readStore();
  delete store[clientKey(client)];
  writeStore(store);
}

export function revokeAllGrants() {
  writeStore({});
}

export function subscribeGrants(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

// ── Warteschlange fuer den Dialog ───────────────────────────────────────────

/** Ohne Antwort gilt die Anfrage nach dieser Zeit als abgelehnt. Rust wartet 60 s auf
 *  das ganze Ergebnis: 30 s Freigabe + max. 20 s Werkzeug (INLINE_WAIT_MS) passen hinein. */
export const APPROVAL_TIMEOUT_MS = 30_000;

interface Pending {
  request: ApprovalRequest;
  resolve: (d: ApprovalDecision | 'timeout') => void;
  timer: ReturnType<typeof setTimeout>;
}

const queue: Pending[] = [];
const queueListeners = new Set<(head: ApprovalRequest | null, waiting: number) => void>();
let reqSeq = 0;

function notifyQueue() {
  const head = queue[0]?.request ?? null;
  queueListeners.forEach((l) => l(head, queue.length));
}

/** Der Dialog abonniert die Warteschlange und zeigt immer die erste Anfrage. */
export function subscribeApprovals(listener: (head: ApprovalRequest | null, waiting: number) => void): () => void {
  queueListeners.add(listener);
  listener(queue[0]?.request ?? null, queue.length);
  return () => { queueListeners.delete(listener); };
}

export function hasApprovalHost(): boolean {
  return queueListeners.size > 0;
}

/** Antwort des Dialogs auf die Anfrage `id`. */
export function answerApproval(id: string, decision: ApprovalDecision) {
  const idx = queue.findIndex((p) => p.request.id === id);
  if (idx < 0) return;
  const [p] = queue.splice(idx, 1);
  clearTimeout(p.timer);
  p.resolve(decision);
  notifyQueue();
}

/** Stellt eine Anfrage in die Warteschlange und wartet auf die Antwort. */
export function requestApprovalViaDialog(input: {
  tool: Tool; args: ToolArgs; client: McpClientInfo; source: McpCallSource;
}, timeoutMs = APPROVAL_TIMEOUT_MS): Promise<ApprovalDecision | 'timeout'> {
  reqSeq += 1;
  const request: ApprovalRequest = {
    id: `appr_${Date.now().toString(36)}_${reqSeq}`,
    tool: input.tool,
    args: input.args,
    client: input.client,
    source: input.source,
    canRemember: canRemember(input.tool),
  };
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      const idx = queue.findIndex((p) => p.request.id === request.id);
      if (idx >= 0) queue.splice(idx, 1);
      resolve('timeout');
      notifyQueue();
    }, timeoutMs);
    queue.push({ request, resolve, timer });
    notifyQueue();
  });
}

/** Nur fuer Tests. */
export function resetApprovalQueue() {
  while (queue.length) clearTimeout(queue.pop()!.timer);
  notifyQueue();
}
