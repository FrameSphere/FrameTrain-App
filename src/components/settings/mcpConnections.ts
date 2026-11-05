// KI-Verbindungen (MCP): Typen und reine Hilfen fuer den Einstellungs-Tab.
// Rust-Seite: src-tauri/src/mcp/, Vertrag: docs/mcp/CONTRACT.md

export const REMOTE_MCP_URL = 'https://frame-train.com/api/mcp';
export const TOKEN_MASK = 'ftmcp_••••••••••••••••';

export interface McpConfig {
  enabled: boolean;
  port: number;
  token: string;
  deviceId: string;
  remoteEnabled: boolean;
}

export interface McpCallLog {
  tool: string;
  clientName: string;
  source: 'local' | 'remote';
  isError: boolean;
  at: string;
  durationMs: number;
}

export interface McpStatus {
  config: McpConfig;
  server: { running: boolean; url: string | null; error: string | null };
  relay: { running: boolean; connected: boolean; lastHeartbeat: string | null; lastError: string | null; jobsHandled: number };
  manifestTools: number;
  manifestLoaded: boolean;
  pendingCalls: number;
  recentCalls: McpCallLog[];
  localClients: { name: string; version?: string }[];
  exePath: string;
  exeTranslocated: boolean;
  claudeDesktop: { configPath: string | null; installed: boolean; configured: boolean };
}

/** Antwort von /api/desktop/mcp/status (Strang C). `id` ist optional – nur damit geht "Trennen". */
export interface RemoteStatus {
  connections: { id?: string; clientName: string; createdAt: string; lastUsedAt: string | null }[];
  recentJobs: { tool: string; clientName: string; status: string; createdAt: string }[];
}

export function endpointUrl(port: number): string {
  return `http://127.0.0.1:${port}/mcp`;
}

export function claudeCodeCommand(port: number, token: string): string {
  return `claude mcp add --transport http frametrain ${endpointUrl(port)} --header "Authorization: Bearer ${token}"`;
}

export function cursorJson(port: number, token: string): string {
  return JSON.stringify({
    mcpServers: {
      frametrain: { url: endpointUrl(port), headers: { Authorization: `Bearer ${token}` } },
    },
  }, null, 2);
}

export function parseRemoteStatus(v: unknown): RemoteStatus {
  const o = (v && typeof v === 'object' ? v : {}) as Partial<RemoteStatus>;
  return {
    connections: Array.isArray(o.connections) ? o.connections : [],
    recentJobs: Array.isArray(o.recentJobs) ? o.recentJobs : [],
  };
}

/** Fehlercodes aus Rust (mcp_connect_claude_desktop) auf Uebersetzungsschluessel. */
export function connectErrorKey(err: string): string | null {
  if (err.startsWith('CLAUDE_NOT_FOUND')) return 'mcp.settings.claudeDesktop.notFound';
  if (err.startsWith('CLAUDE_CONFIG_INVALID')) return 'mcp.settings.claudeDesktop.invalidConfig';
  return null;
}

// ============ Freigaben (kommen aus src/mcp/approvals.ts, Strang A) ============

export interface McpApproval {
  client: string;
  tool: string;
  grantedAt?: string;
}

export interface ApprovalsApi {
  listApprovals: () => McpApproval[] | Promise<McpApproval[]>;
  revokeApproval: (client: string, tool: string) => void | Promise<void>;
  onApprovalsChanged?: (cb: () => void) => () => void;
}

// Optionaler Import: fehlt das Modul (Strang A noch nicht da), ist die Liste leer statt kaputt.
const approvalModules = import.meta.glob('../../mcp/approvals.ts');

export async function loadApprovalsApi(): Promise<ApprovalsApi | null> {
  const loader = Object.values(approvalModules)[0];
  if (!loader) return null;
  try {
    const m = (await loader()) as Partial<ApprovalsApi>;
    if (typeof m.listApprovals !== 'function' || typeof m.revokeApproval !== 'function') return null;
    return m as ApprovalsApi;
  } catch {
    return null;
  }
}

/** Freigaben je Client gruppiert, Clients und Werkzeuge alphabetisch. */
export function groupApprovals(list: McpApproval[]): { client: string; items: McpApproval[] }[] {
  const map = new Map<string, McpApproval[]>();
  for (const a of list) {
    if (!a || !a.client || !a.tool) continue;
    map.set(a.client, [...(map.get(a.client) ?? []), a]);
  }
  return [...map.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([client, items]) => ({ client, items: items.sort((x, y) => x.tool.localeCompare(y.tool)) }));
}
