// Webview-Seite der MCP-Bridge (Vertrag Abschnitt 3).
//
//   Start:   mcp_register_manifest({ tools })  — Rust merkt sich das Manifest
//   Aufruf:  Event "mcp://call" { callId, tool, args, source, client }
//   Antwort: mcp_tool_result({ callId, result })
//
// Die Rust-Commands baut Strang B. Fehlen sie (aelterer Stand, Tests, Vite-
// Vorschau), laeuft die App ohne MCP weiter: jeder Aufruf ist abgesichert.

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { needsApproval, recordDecision, requestApprovalViaDialog, hasApprovalHost } from './approval';
import { registry as defaultRegistry, type ToolRegistry } from './registry';
import { startRunLog } from './runLog';
import type {
  ApprovalDecision, CallToolResult, McpCallContext, McpCallEvent, McpClientInfo, McpCallSource, Tool, ToolArgs,
} from './types';
import { errorText, fail, ToolArgError, ToolFailure, validateArgs } from './util';
import { announceChange } from './tools/common';

export const MSG_DENIED = 'Der Nutzer hat die Aktion in FrameTrain abgelehnt.';
export const MSG_APPROVAL_TIMEOUT = 'Keine Freigabe in FrameTrain erteilt (Zeitlimit). Bitte den Nutzer bitten, die App zu oeffnen und erneut versuchen.';
export const MSG_NOT_SIGNED_IN = 'In FrameTrain ist niemand angemeldet. Bitte den Nutzer bitten, sich in der App anzumelden.';
/**
 * Rust wartet 60 s auf die Antwort (inklusive Freigabe-Dialog). Die Bridge
 * antwortet spaetestens nach diesem Budget; ein laengeres Werkzeug laeuft in
 * der App weiter.
 */
export const TOOL_TIMEOUT_MS = 57_000;
/** Mindestzeit fuer das Werkzeug, auch wenn die Freigabe lange gedauert hat. */
const MIN_RUN_MS = 5_000;

export type ApprovalRequester = (input: {
  tool: Tool; args: ToolArgs; client: McpClientInfo; source: McpCallSource;
}) => Promise<ApprovalDecision | 'timeout'>;

export interface BridgeDeps {
  registry: ToolRegistry;
  getUserId: () => string | null;
  requestApproval: ApprovalRequester;
  /** Holt das Hauptfenster nach vorn, wenn eine Freigabe noetig ist. */
  focusApp?: () => void;
  timeoutMs?: number;
}

let currentUserId: string | null = null;

/** App.tsx meldet den angemeldeten Nutzer (null nach Logout). */
export function setMcpUser(userId: string | null) {
  currentUserId = userId;
}

export function getMcpUser(): string | null {
  return currentUserId;
}

function normalizeCall(raw: unknown): McpCallEvent | null {
  if (!raw || typeof raw !== 'object') return null;
  const c = raw as Record<string, unknown>;
  if (typeof c.callId !== 'string' || typeof c.tool !== 'string') return null;
  const client = (c.client && typeof c.client === 'object' ? c.client : {}) as Record<string, unknown>;
  return {
    callId: c.callId,
    tool: c.tool,
    args: c.args && typeof c.args === 'object' && !Array.isArray(c.args) ? c.args as ToolArgs : {},
    source: c.source === 'remote' ? 'remote' : 'local',
    client: { name: typeof client.name === 'string' && client.name ? client.name : 'Unbekannter Client', version: typeof client.version === 'string' ? client.version : undefined },
  };
}

/**
 * Fuehrt einen Aufruf aus: Werkzeug finden, Argumente pruefen, Freigabe
 * einholen, ausfuehren. Wirft nie — Fehler werden zu `isError`-Ergebnissen.
 */
export async function executeCall(call: McpCallEvent, deps: BridgeDeps): Promise<CallToolResult> {
  const startedAt = Date.now();
  const tool = deps.registry.get(call.tool);
  if (!tool) return fail(`Unknown tool "${call.tool}". Call tools/list to see available tools.`);

  const argErrors = validateArgs(tool.inputSchema, call.args);
  if (argErrors.length) return fail(`Invalid arguments for ${tool.name}: ${argErrors.join('; ')}`, { errors: argErrors });

  const userId = deps.getUserId();
  if (tool.requiresUser && !userId) return fail(MSG_NOT_SIGNED_IN);

  if (needsApproval(tool, call.client)) {
    deps.focusApp?.();
    const decision = await deps.requestApproval({ tool, args: call.args, client: call.client, source: call.source });
    if (decision === 'timeout') return fail(MSG_APPROVAL_TIMEOUT);
    if (decision === 'deny') return fail(MSG_DENIED);
    recordDecision(tool, call.client, decision);
  }

  const ctx: McpCallContext = { callId: call.callId, client: call.client, source: call.source, userId };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const budget = Math.max(MIN_RUN_MS, (deps.timeoutMs ?? TOOL_TIMEOUT_MS) - (Date.now() - startedAt));
  try {
    const timeout = new Promise<CallToolResult>((resolve) => {
      timer = setTimeout(() => resolve(fail(`${tool.name} did not finish within ${Math.round(budget / 1000)} s; it continues in the app. Check the matching *_status tool.`)), budget);
    });
    const result = await Promise.race([tool.run(call.args, ctx), timeout]);
    if (!result.isError && tool.annotations.readOnlyHint === false) {
      announceChange(tool._meta['frametrain/domain'], { tool: tool.name });
    }
    return result;
  } catch (e) {
    if (e instanceof ToolArgError) return fail(`Invalid arguments for ${tool.name}: ${e.message}`);
    if (e instanceof ToolFailure) return fail(e.message);
    return fail(`${tool.name} failed: ${errorText(e)}`);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Ergebnis an Rust zurueck; fehlt der Command, wird es nur protokolliert. */
async function sendResult(callId: string, result: CallToolResult) {
  try {
    await invoke('mcp_tool_result', { callId, result });
  } catch (e) {
    console.warn('[MCP] mcp_tool_result nicht verfuegbar:', errorText(e));
  }
}

/** Manifest an Rust melden. Rueckgabe: ob Rust es angenommen hat. */
export async function registerManifest(reg: ToolRegistry = defaultRegistry): Promise<boolean> {
  try {
    await invoke('mcp_register_manifest', { tools: reg.manifest() });
    return true;
  } catch (e) {
    console.info('[MCP] mcp_register_manifest nicht verfuegbar — MCP bleibt aus:', errorText(e));
    return false;
  }
}

function focusMainWindow() {
  invoke('hosting_open_main', { view: null }).catch(() => {});
}

let running: Promise<() => void> | null = null;

/**
 * Startet die Bridge einmal pro App-Lauf: Manifest melden, Events abonnieren.
 * Rueckgabe: Abmelden (fuer Tests und Hot Reload).
 */
export function startMcpBridge(overrides: Partial<BridgeDeps> = {}): Promise<() => void> {
  if (running) return running;
  const deps: BridgeDeps = {
    registry: overrides.registry ?? defaultRegistry,
    getUserId: overrides.getUserId ?? getMcpUser,
    requestApproval: overrides.requestApproval ?? ((input) => (
      hasApprovalHost() ? requestApprovalViaDialog(input) : Promise.resolve('timeout' as const)
    )),
    focusApp: overrides.focusApp ?? focusMainWindow,
    timeoutMs: overrides.timeoutMs,
  };
  running = (async () => {
    const offs: (() => void)[] = [];
    offs.push(await startRunLog());
    try {
      offs.push(await listen<unknown>('mcp://call', async (event) => {
        const call = normalizeCall(event.payload);
        if (!call) return;
        const result = await executeCall(call, deps);
        await sendResult(call.callId, result);
      }));
    } catch (e) {
      console.info('[MCP] Event-Listener nicht verfuegbar:', errorText(e));
    }
    await registerManifest(deps.registry);
    return () => {
      offs.forEach((off) => off());
      running = null;
    };
  })();
  return running;
}
