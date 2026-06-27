// ─────────────────────────────────────────────────────────────────────────────
// Synapse Agent Debug Logger
// Writes a detailed JSON-Lines log to the OS temp dir.
// Enable via: localStorage.setItem('synapse_debug', '1')
// File location logged to console on first write.
// ─────────────────────────────────────────────────────────────────────────────

import { writeTextFile, BaseDirectory } from '@tauri-apps/plugin-fs';

const SESSION_ID = Date.now().toString(36);
let callIndex = 0;
let logLines: string[] = [];
let filePath: string | null = null;

function isEnabled(): boolean {
  try { return import.meta.env.DEV && localStorage.getItem('synapse_debug') === '1'; } catch { return false; }
}

/** Rough token estimate: ~4 chars per token */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Detect duplicate/near-duplicate content blocks in a message list */
function findDuplicates(messages: Array<{ role: string; content: string }>): string[] {
  const warnings: string[] = [];
  const seen = new Map<string, number>();

  for (let i = 0; i < messages.length; i++) {
    const key = messages[i].content.slice(0, 120).trim();
    if (seen.has(key)) {
      warnings.push(`⚠ messages[${i}] is near-duplicate of messages[${seen.get(key)}]`);
    } else {
      seen.set(key, i);
    }
  }

  // Check if system prompt content leaks into messages
  return warnings;
}

export type DebugCallEntry = {
  session: string;
  call: number;
  step: number;
  ts: string;

  // What was sent
  system: string;
  messages: Array<{ role: string; content: string }>;

  // Token estimates
  tokens: {
    system: number;
    messages: number;
    total: number;
    byMessage: Array<{ role: string; tokens: number; preview: string }>;
  };

  // Warnings
  warnings: string[];

  // What came back
  reply?: string;
  replyTokens?: number;
  parsedTool?: { tool: string; args: unknown } | null;
  error?: string;
  durationMs?: number;
};

function buildEntry(
  step: number,
  system: string,
  messages: Array<{ role: string; content: string }>,
): DebugCallEntry {
  callIndex++;
  const systemTokens = estimateTokens(system);
  const byMessage = messages.map((m) => ({
    role: m.role,
    tokens: estimateTokens(m.content),
    preview: m.content.slice(0, 80).replace(/\n/g, '↵'),
  }));
  const messagesTokens = byMessage.reduce((s, m) => s + m.tokens, 0);
  const warnings = findDuplicates(messages);

  // Extra warnings
  if (systemTokens > 800)  warnings.push(`⚠ System prompt is large: ~${systemTokens} tokens`);
  if (messagesTokens > 2000) warnings.push(`⚠ Messages are large: ~${messagesTokens} tokens`);
  if (messages.length > 8) warnings.push(`⚠ Many messages: ${messages.length} (consider trimming)`);

  return {
    session: SESSION_ID,
    call: callIndex,
    step,
    ts: new Date().toISOString(),
    system,
    messages,
    tokens: { system: systemTokens, messages: messagesTokens, total: systemTokens + messagesTokens, byMessage },
    warnings,
  };
}

async function flush(entry: DebugCallEntry): Promise<void> {
  if (!isEnabled()) return;

  const line = JSON.stringify(entry, null, 2);
  logLines.push(line);

  const fileName = `synapse_agent_debug_${SESSION_ID}.jsonl`;

  try {
    const content = logLines.map((l) => l).join('\n---\n');
    await writeTextFile(fileName, content, { baseDir: BaseDirectory.Temp });

    if (!filePath) {
      filePath = fileName;
      console.log(`[SynapseDebug] Writing to TEMP/${fileName}`);
    }
  } catch (e) {
    // Fallback: dump to console so it's never lost
    console.group(`[SynapseDebug] call #${entry.call} step ${entry.step}`);
    console.log('System tokens:', entry.tokens.system);
    console.log('Messages tokens:', entry.tokens.messages);
    console.log('Total tokens:', entry.tokens.total);
    if (entry.warnings.length) console.warn('Warnings:', entry.warnings.join('\n'));
    console.log('System:\n', entry.system);
    console.table(entry.tokens.byMessage);
    if (entry.reply) console.log('Reply:\n', entry.reply);
    if (entry.error) console.error('Error:', entry.error);
    console.groupEnd();
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export type DebugHandle = {
  onReply: (reply: string, parsedTool: { tool: string; args: unknown } | null) => Promise<void>;
  onError: (error: string) => Promise<void>;
};

/**
 * Call before each AI request. Returns a handle to log the reply/error.
 * Early-exit when disabled: zero overhead in Production builds.
 */
export async function debugLogRequest(
  step: number,
  system: string,
  messages: Array<{ role: string; content: string }>,
): Promise<DebugHandle> {
  // Early-exit: im Production-Build ist import.meta.env.DEV === false,
  // d.h. isEnabled() ist immer false und Vite eliminiert den Rest per Tree-Shaking.
  // Kein buildEntry(), keine Token-Schätzung, keine Objekterzeugung.
  if (!isEnabled()) {
    return {
      onReply: async () => {},
      onError: async () => {},
    };
  }

  const startMs = Date.now();
  const entry   = buildEntry(step, system, messages);

  console.log(
    `[SynapseDebug] call #${entry.call} step ${entry.step} | tokens: sys=${entry.tokens.system} msgs=${entry.tokens.messages} total=${entry.tokens.total}`,
    entry.warnings.length ? `\n${entry.warnings.join('\n')}` : '',
  );

  return {
    async onReply(reply: string, parsedTool) {
      entry.reply       = reply;
      entry.replyTokens = estimateTokens(reply);
      entry.parsedTool  = parsedTool;
      entry.durationMs  = Date.now() - startMs;
      await flush(entry);
    },
    async onError(error: string) {
      entry.error      = error;
      entry.durationMs = Date.now() - startMs;
      await flush(entry);
    },
  };
}

/**
 * Returns a short readable summary of the last N calls (for in-app display if needed).
 */
export function getDebugSummary(): string {
  if (logLines.length === 0) return 'No debug calls recorded yet.';
  try {
    const last = JSON.parse(logLines[logLines.length - 1]) as DebugCallEntry;
    return [
      `Session: ${last.session}`,
      `Total calls: ${last.call}`,
      `Last call: step ${last.step} | ${last.tokens.total} tokens | ${last.durationMs}ms`,
      last.warnings.length ? `Warnings: ${last.warnings.join(', ')}` : 'No warnings',
    ].join('\n');
  } catch { return 'Error reading debug log.'; }
}
