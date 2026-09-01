import { invoke } from '@tauri-apps/api/core';
import type { AISettings, AIProvider } from '../contexts/AISettingsContext';
import { PROVIDER_META, resolveModel } from './providerMeta';

/**
 * Alle KI-HTTP-Aufrufe laufen ueber das Rust-Backend (Tauri-Command
 * `ai_http_post`), NICHT direkt aus dem WebView. Grund: Anthropic-Abo-/OAuth-
 * Orgs blockieren CORS-Anfragen aus dem Browser ("CORS requests are not allowed
 * for this Organization"). Serverseitig gibt es keinen Origin/Preflight — genau
 * wie bei der Claude-Code-CLI. Zusaetzlich liegt der Key so nie in einer
 * Browser-Netzwerkschicht.
 */
type ProxyResponse = { status: number; body: string };

async function backendPost(
  url: string,
  headers: Record<string, string>,
  bodyObj: unknown,
): Promise<{ status: number; data: any }> {
  const res = await invoke<ProxyResponse>('ai_http_post', {
    url,
    headers,
    body: JSON.stringify(bodyObj),
  });
  let data: any = {};
  try { data = res.body ? JSON.parse(res.body) : {}; } catch { data = { raw: res.body }; }
  return { status: res.status, data };
}

export type ChatRole = 'system' | 'user' | 'assistant';
export type ChatMessage = { role: Exclude<ChatRole, 'system'>; content: string };

export type CallAIOptions = {
  system: string;
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  responseLanguage?: string;
};

function requireEnabled(settings: AISettings) {
  if (!settings.enabled) throw new Error('KI-Assistent deaktiviert. Bitte in Einstellungen aktivieren.');
  const meta = PROVIDER_META[settings.provider];
  if (meta.needsKey && !settings.apiKey) throw new Error(`API-Key für ${meta.label} fehlt.`);
}

/**
 * Anthropic kennt zwei Key-Typen mit UNTERSCHIEDLICHER Authentifizierung:
 *
 *  - `sk-ant-api…` (Console-Key, Dollar-Guthaben): Header `x-api-key`.
 *  - `sk-ant-oat…` (OAuth-Abo-Token aus `claude setup-token`, zählt gegen die
 *    Abo-Grenzen): Header `Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20`.
 *    Zusätzlich akzeptiert Anthropic OAuth-Token nur für „Claude Code"-förmige
 *    Anfragen — deshalb wird der Claude-Code-Identitätssatz als erster
 *    System-Block vorangestellt (genau das macht die CLI intern). Ohne ihn
 *    lehnt die API den Abo-Token ab.
 */
function isOAuthToken(key: string): boolean {
  return key.trim().startsWith('sk-ant-oat');
}

const CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude.";

async function callAnthropic(apiKey: string, model: string, system: string, messages: ChatMessage[], maxTokens: number, temperature: number, unlimited = false) {
  const key = apiKey.trim();
  const oauth = isOAuthToken(key);

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01',
  };
  if (oauth) {
    headers['Authorization'] = `Bearer ${key}`;
    headers['anthropic-beta'] = 'oauth-2025-04-20';
  } else {
    headers['x-api-key'] = key;
  }

  // Bei OAuth muss der Claude-Code-Identitätssatz zuerst kommen, sonst 401/403.
  const systemField = oauth
    ? [
        { type: 'text', text: CLAUDE_CODE_IDENTITY },
        { type: 'text', text: system },
      ]
    : system;

  // Die Claude-5-Familie (opus-5, sonnet-5, fable-5) sowie opus-4.6/4.7/4.8 und
  // sonnet-4.6 lehnen Sampling-Parameter (temperature/top_p) mit HTTP 400 ab.
  // Nur ältere Modelle wie haiku-4-5 akzeptieren `temperature`. Deshalb wird der
  // Parameter für die neueren Modelle weggelassen (Default greift).
  const rejectsSampling = /claude-(opus-5|sonnet-5|fable-5|mythos-5|opus-4-[678]|sonnet-4-6)/.test(model);

  const body: Record<string, unknown> = {
    model,
    max_tokens: maxTokens,
    system: systemField,
    messages: messages.map(m => ({ role: m.role, content: m.content })),
  };
  if (!rejectsSampling) body.temperature = temperature;

  // "Unlimited"-Modus: Modelle mit Effort-Steuerung (Claude-5-Familie,
  // opus-4.6/4.7/4.8, sonnet-4.6) laufen auf höchster Stufe — maximale Tiefe
  // fürs interne Nachdenken. Modelle ohne Effort-Support (z.B. haiku-4-5)
  // ignorieren das bewusst, um kein HTTP 400 zu provozieren.
  const supportsEffort = /claude-(opus-5|sonnet-5|fable-5|mythos-5|opus-4-[5678]|sonnet-4-6)/.test(model);
  if (unlimited && supportsEffort) {
    body.output_config = { effort: 'max' };
  }

  const { status, data } = await backendPost('https://api.anthropic.com/v1/messages', headers, body);
  if (status < 200 || status >= 300) {
    throw new Error(data?.error?.message || `HTTP ${status}`);
  }
  // WICHTIG: Bei Thinking-Modellen (Claude-5-Familie: opus-5/sonnet-5 — Thinking
  // ist dort per Default an) ist content[0] ein `thinking`-Block; der eigentliche
  // Text steht in einem SPÄTEREN `text`-Block. Deshalb ALLE text-Blöcke einsammeln,
  // nicht nur den ersten — sonst kommt fälschlich ein leerer String zurück.
  const blocks: any[] = Array.isArray(data?.content) ? data.content : [];
  return blocks
    .filter(b => b?.type === 'text' && typeof b.text === 'string')
    .map(b => b.text)
    .join('')
    .trim();
}

/**
 * Modelle, die vor der Antwort unsichtbar "nachdenken".
 *
 * Ihre Reasoning-Tokens zaehlen gegen max_tokens, tauchen aber nicht in
 * `content` auf. Ein knappes Budget (z.B. 20 Tokens fuer einen Chat-Titel)
 * wird komplett vom Reasoning aufgebraucht — die Antwort kommt dann leer
 * zurueck, ohne Fehler. Deshalb bekommen diese Modelle einen Aufschlag.
 */
const REASONING_MODEL_PATTERN = /gpt-oss|^o[1-9]([-.]|$)|qwen3|deepseek-r1|compound/i;
const REASONING_RESERVE = 1024;

export function effectiveMaxTokens(model: string, maxTokens: number): number {
  return REASONING_MODEL_PATTERN.test(model) ? maxTokens + REASONING_RESERVE : maxTokens;
}

async function callOpenAICompat(url: string, apiKey: string, model: string, system: string, messages: ChatMessage[], maxTokens: number, temperature: number) {
  const { status, data } = await backendPost(
    url,
    { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey.trim()}` },
    {
      model,
      max_tokens: effectiveMaxTokens(model, maxTokens),
      temperature,
      messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
    },
  );
  if (status < 200 || status >= 300) {
    throw new Error(data?.error?.message || `HTTP ${status}`);
  }
  return data?.choices?.[0]?.message?.content || '';
}

async function callOllama(model: string, system: string, messages: ChatMessage[], temperature: number) {
  let status: number, data: any;
  try {
    ({ status, data } = await backendPost(
      'http://localhost:11434/api/chat',
      { 'Content-Type': 'application/json' },
      {
        model,
        stream: false,
        options: { temperature, num_ctx: 4096 },
        messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
      },
    ));
  } catch {
    // Verbindungsfehler (Backend erreicht Ollama nicht)
    throw new Error('Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
  }
  if (status < 200 || status >= 300) {
    throw new Error(data?.error || 'Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
  }
  return data?.message?.content || '';
}

function withResponseLanguage(system: string, responseLanguage?: string) {
  const lang = responseLanguage?.trim();
  if (!lang) return system;
  return `${system}\n\nANTWORTSPRACHE:\n- Antworte ausschließlich auf ${lang}.`;
}

/**
 * Prueft mit einem minimalen Request, ob Provider + Key + Modell zusammen
 * funktionieren. Wirft mit einer sprechenden Fehlermeldung, wenn nicht.
 * `enabled` wird intern erzwungen, damit auch vor dem Aktivieren getestet
 * werden kann.
 */
export async function testAIConnection(settings: AISettings): Promise<void> {
  // maxTokens großzügig genug, dass auch Thinking-Modelle (die einen Teil des
  // Budgets fürs interne Nachdenken verbrauchen) noch echten Text ausgeben.
  const reply = await callAI(
    { ...settings, enabled: true },
    {
      system: 'Reply with exactly the word: OK',
      messages: [{ role: 'user', content: 'ping' }],
      maxTokens: 256,
      temperature: 0,
    },
  );
  // Ohne diese Prüfung galt der Test schon als bestanden, wenn der Aufruf nur
  // nicht warf — eine leere Antwort (z.B. weil nur ein Thinking-Block kam) wäre
  // faelschlich als Erfolg durchgegangen.
  if (!reply || !reply.trim()) {
    throw new Error('Verbindung steht, aber das Modell lieferte keinen Text zurück. Bitte anderes Modell/Budget probieren.');
  }
}

export async function callAI(settings: AISettings, options: CallAIOptions): Promise<string> {
  requireEnabled(settings);
  const provider: AIProvider = settings.provider;
  const model = resolveModel(provider, settings.selectedModel, settings.ollamaModel);
  const maxTokens = options.maxTokens ?? 2000;
  const temperature = options.temperature ?? 0.7;
  const system = withResponseLanguage(options.system, options.responseLanguage);
  const messages = options.messages;
  const unlimited = settings.tokenBudget === 'unlimited';

  if (provider === 'anthropic') return callAnthropic(settings.apiKey, model, system, messages, maxTokens, temperature, unlimited);
  if (provider === 'openai') return callOpenAICompat('https://api.openai.com/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature);
  if (provider === 'groq') return callOpenAICompat('https://api.groq.com/openai/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature);
  return callOllama(model, system, messages, temperature);
}
