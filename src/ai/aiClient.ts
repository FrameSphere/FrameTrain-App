import type { AISettings, AIProvider } from '../contexts/AISettingsContext';
import { PROVIDER_META, resolveModel } from './providerMeta';

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

async function callAnthropic(apiKey: string, model: string, system: string, messages: ChatMessage[], maxTokens: number, temperature: number) {
  const key = apiKey.trim();
  const oauth = isOAuthToken(key);

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'anthropic-version': '2023-06-01',
    'anthropic-dangerous-direct-browser-access': 'true',
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

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e?.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return data?.content?.[0]?.text || '';
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
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model,
      max_tokens: effectiveMaxTokens(model, maxTokens),
      temperature,
      messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
    }),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e?.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return data?.choices?.[0]?.message?.content || '';
}

async function callOllama(model: string, system: string, messages: ChatMessage[], temperature: number) {
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      options: { temperature, num_ctx: 4096 },
      messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
    }),
  });
  if (!res.ok) throw new Error('Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
  const data = await res.json();
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
  await callAI(
    { ...settings, enabled: true },
    { system: 'ping', messages: [{ role: 'user', content: 'ping' }], maxTokens: 5, temperature: 0 },
  );
}

export async function callAI(settings: AISettings, options: CallAIOptions): Promise<string> {
  requireEnabled(settings);
  const provider: AIProvider = settings.provider;
  const model = resolveModel(provider, settings.selectedModel, settings.ollamaModel);
  const maxTokens = options.maxTokens ?? 2000;
  const temperature = options.temperature ?? 0.7;
  const system = withResponseLanguage(options.system, options.responseLanguage);
  const messages = options.messages;

  if (provider === 'anthropic') return callAnthropic(settings.apiKey, model, system, messages, maxTokens, temperature);
  if (provider === 'openai') return callOpenAICompat('https://api.openai.com/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature);
  if (provider === 'groq') return callOpenAICompat('https://api.groq.com/openai/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature);
  return callOllama(model, system, messages, temperature);
}
