import { invoke } from '@tauri-apps/api/core';
import type { AISettings, AIProvider, TokenBudget } from '../contexts/AISettingsContext';
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

/**
 * 'chat'  — Fliesstext fuer Menschen. Bekommt eine Laengen-Vorgabe passend zum
 *           Token-Budget (kurz bei Minimal, ausfuehrlich bei Maximum).
 * 'raw'   — Strukturierte Ausgabe (JSON-Plan, Code-Edits, Chat-Titel).
 *           Keine Laengen-Vorgabe, sonst kuerzt das Modell die Struktur weg.
 */
export type ResponseStyle = 'chat' | 'raw';

export type CallAIOptions = {
  system: string;
  messages: ChatMessage[];
  /**
   * Ziel-Laenge der SICHTBAREN Antwort. Ist bewusst KEIN hartes Limit mehr:
   * das tatsaechlich gesendete max_tokens liegt darueber (siehe
   * `thinkingReserve`), damit nichts mitten im Satz abbricht. Die Laenge
   * steuert bei `style: 'chat'` die Vorgabe im System-Prompt.
   */
  maxTokens?: number;
  temperature?: number;
  responseLanguage?: string;
  style?: ResponseStyle;
  /**
   * Wird aufgerufen, wenn das Modell mitten im Satz aufgehoert hat, weil
   * max_tokens erreicht war. Ohne diesen Hinweis wirkt eine abgeschnittene
   * Antwort wie eine vollstaendige — inklusive halbem JSON-Block, aus dem
   * dann keine Parameter mehr gelesen werden koennen.
   */
  onTruncated?: () => void;
};

type Bilingual = { de: string; en: string };

/**
 * Was ein Token-Budget in der Praxis bedeutet.
 *
 * Vorher war `maxTokens` gleichzeitig Laengen-Steuerung UND hartes Limit. Das
 * ging bei Modellen mit unsichtbarem Nachdenken (Claude-5-Familie, gpt-oss,
 * qwen3 …) schief: die Denk-Tokens zaehlen gegen max_tokens, verbrauchten bei
 * "Minimal" das komplette Budget und die Antwort kam LEER zurueck — Analyse,
 * Metrik-Assistent und Analyse-Chat lieferten sichtbar nichts.
 *
 * Jetzt sind beide Rollen getrennt:
 *   - `thinkingReserve` wird auf max_tokens DRAUFGESCHLAGEN. Der sichtbare
 *     Text hat damit immer Platz, egal wie viel das Modell nachdenkt.
 *   - `effort` begrenzt bei Anthropic, wie tief nachgedacht wird — das ist der
 *     Hebel, der die Kosten bei kleinem Budget wirklich klein haelt.
 *   - `brevity` steuert die Laenge dort, wo sie hingehoert: im Prompt.
 * Ergebnis: jedes Budget funktioniert, hoeheres Budget = mehr Tiefe.
 */
type BudgetProfile = {
  effort: 'low' | 'medium' | 'high' | 'max';
  thinkingReserve: number;
  brevity: Bilingual | null;
};

export const BUDGET_PROFILE: Record<TokenBudget, BudgetProfile> = {
  minimal: {
    effort: 'low',
    thinkingReserve: 2000,
    brevity: {
      de: 'Fasse dich sehr kurz: hoechstens 3 Saetze Fliesstext, keine Wiederholungen, keine Einleitung.',
      en: 'Be very brief: at most 3 sentences of prose, no repetition, no preamble.',
    },
  },
  balanced: {
    effort: 'low',
    thinkingReserve: 3000,
    brevity: {
      de: 'Fasse dich knapp: hoechstens 6 Saetze Fliesstext, keine Einleitung.',
      en: 'Be concise: at most 6 sentences of prose, no preamble.',
    },
  },
  quality: {
    effort: 'medium',
    thinkingReserve: 5000,
    brevity: {
      de: 'Antworte ausfuehrlich, aber ohne Fuellwoerter: hoechstens 12 Saetze Fliesstext, je Empfehlung eine kurze Begruendung.',
      en: 'Answer thoroughly but without filler: at most 12 sentences of prose, one short rationale per recommendation.',
    },
  },
  max: {
    effort: 'high',
    thinkingReserve: 8000,
    brevity: {
      de: 'Antworte gruendlich und begruendet. Struktur mit Zwischenueberschriften und Listen ist erwuenscht, Fuellwoerter nicht.',
      en: 'Answer thoroughly with reasoning. Use headings and lists where they help; no filler.',
    },
  },
  unlimited: {
    effort: 'max',
    thinkingReserve: 16000,
    brevity: {
      de: 'Nimm dir Raum: gehe in die Tiefe, nenne Alternativen und Trade-offs, belege deine Empfehlungen. Keine kuenstliche Kuerzung.',
      en: 'Take your space: go deep, name alternatives and trade-offs, justify recommendations. Do not artificially shorten.',
    },
  },
};

/** Laengen- und Denk-Profil der aktuellen Einstellung. */
export function budgetProfile(settings: AISettings): BudgetProfile {
  return BUDGET_PROFILE[settings.tokenBudget ?? 'balanced'] ?? BUDGET_PROFILE.balanced;
}

/**
 * Bereinigt den Verlauf, bevor er an einen Provider geht.
 *
 * - Leere / nur aus Leerzeichen bestehende Nachrichten fliegen raus: Anthropic
 *   lehnt leere Text-Bloecke mit HTTP 400 ab. Eine leer zurueckgekommene
 *   Antwort landete so als leere assistant-Nachricht im Verlauf und legte
 *   jeden weiteren Aufruf lahm.
 * - Aufeinanderfolgende Nachrichten derselben Rolle werden zusammengefasst.
 */
function normalizeMessages(messages: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of messages) {
    if (typeof m.content !== 'string' || !m.content.trim()) continue;
    const last = out[out.length - 1];
    if (last && last.role === m.role) last.content = `${last.content}\n\n${m.content}`;
    else out.push({ role: m.role, content: m.content });
  }
  return out;
}

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

/**
 * Anthropic verlangt, dass die ERSTE Nachricht die Rolle `user` hat.
 * Der Metrik-Assistent stellt den fertigen Bericht als erste
 * assistant-Nachricht in den Verlauf, und der Coach kann nach dem
 * History-Trimmen ebenfalls mit einer assistant-Nachricht beginnen — beides
 * quittierte die API mit HTTP 400, der Chat blieb komplett tot.
 * Der Inhalt geht nicht verloren: er steckt bei beiden bereits im
 * System-Prompt.
 */
function stripLeadingAssistant(messages: ChatMessage[]): ChatMessage[] {
  let i = 0;
  while (i < messages.length && messages[i].role === 'assistant') i++;
  return messages.slice(i);
}

/**
 * Claude-Modelle, die vor der Antwort unsichtbar nachdenken (Thinking ist dort
 * per Default aktiv). Ihre Denk-Tokens zaehlen gegen max_tokens.
 */
const ANTHROPIC_THINKING_PATTERN = /claude-(opus-5|sonnet-5|fable-5|mythos-5|opus-4-[678]|sonnet-4-6)/;

/** Modelle, die `output_config.effort` kennen. */
const ANTHROPIC_EFFORT_PATTERN = /claude-(opus-5|sonnet-5|fable-5|mythos-5|opus-4-[5678]|sonnet-4-6)/;

/** Nur die neueren Familien kennen die Stufe `max`; opus-4.5 kann nur bis `high`. */
const ANTHROPIC_MAX_EFFORT_PATTERN = /claude-(opus-5|sonnet-5|fable-5|mythos-5|opus-4-[678]|sonnet-4-6)/;

function anthropicEffort(model: string, profile: BudgetProfile): string | null {
  if (!ANTHROPIC_EFFORT_PATTERN.test(model)) return null;
  if (profile.effort === 'max' && !ANTHROPIC_MAX_EFFORT_PATTERN.test(model)) return 'high';
  return profile.effort;
}

async function callAnthropic(
  apiKey: string, model: string, system: string, messages: ChatMessage[],
  maxTokens: number, temperature: number, profile: BudgetProfile, onTruncated?: () => void,
) {
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
  const rejectsSampling = ANTHROPIC_THINKING_PATTERN.test(model);

  const sendable = stripLeadingAssistant(messages);
  if (sendable.length === 0) throw new Error('Keine Nachricht zum Senden (Verlauf enthaelt nur Antworten).');

  const budget = ANTHROPIC_THINKING_PATTERN.test(model)
    ? maxTokens + profile.thinkingReserve
    : maxTokens;

  const body: Record<string, unknown> = {
    model,
    max_tokens: budget,
    system: systemField,
    messages: sendable.map(m => ({ role: m.role, content: m.content })),
  };
  if (!rejectsSampling) body.temperature = temperature;

  // Der Denk-Aufwand folgt jetzt dem Token-Budget (frueher nur "unlimited" =
  // effort:max, sonst gar nichts). Genau das macht "Minimal" bezahlbar: das
  // Modell denkt kurz statt lange, statt dass wir ihm den Text abschneiden.
  const effort = anthropicEffort(model, profile);
  if (effort) body.output_config = { effort };

  const { status, data } = await backendPost('https://api.anthropic.com/v1/messages', headers, body);
  if (status < 200 || status >= 300) {
    throw new Error(data?.error?.message || `HTTP ${status}`);
  }
  // WICHTIG: Bei Thinking-Modellen (Claude-5-Familie: opus-5/sonnet-5 — Thinking
  // ist dort per Default an) ist content[0] ein `thinking`-Block; der eigentliche
  // Text steht in einem SPÄTEREN `text`-Block. Deshalb ALLE text-Blöcke einsammeln,
  // nicht nur den ersten — sonst kommt fälschlich ein leerer String zurück.
  if (data?.stop_reason === 'max_tokens') onTruncated?.();
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
const REASONING_MODEL_PATTERN = /gpt-oss|^o[1-9]([-.]|$)|qwen3|deepseek-r1|magistral|compound|thinking|reasoning/i;

/** OpenAI-Reasoning-Modelle nehmen `max_completion_tokens` statt `max_tokens`. */
const OPENAI_REASONING_PATTERN = /^(o[1-9]|gpt-5)/i;

export function effectiveMaxTokens(model: string, maxTokens: number, reserve = 1024): number {
  return REASONING_MODEL_PATTERN.test(model) ? maxTokens + reserve : maxTokens;
}

/**
 * Entfernt sichtbare Denk-Bloecke aus der Antwort.
 *
 * Lokale Reasoning-Modelle (qwen3, deepseek-r1, gpt-oss über Ollama) schreiben
 * ihr Nachdenken als `<think>…</think>` mitten in den Text. Ungefiltert stand
 * das komplette Selbstgespraech im Chat.
 */
export function stripThinkTags(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
    // Unabgeschlossener Block (am Limit abgeschnitten): Rest verwerfen.
    .replace(/<think(?:ing)?>[\s\S]*$/i, '')
    .trim();
}

async function callOpenAICompat(
  url: string, apiKey: string, model: string, system: string, messages: ChatMessage[],
  maxTokens: number, temperature: number, profile: BudgetProfile, onTruncated?: () => void,
) {
  const ceiling = effectiveMaxTokens(model, maxTokens, profile.thinkingReserve);
  const isOpenAIReasoning = OPENAI_REASONING_PATTERN.test(model);
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
  };
  // o-Serie und GPT-5 lehnen `max_tokens` sowie abweichende Temperaturen mit
  // HTTP 400 ab — dort gilt `max_completion_tokens` und der Default-Sampler.
  if (isOpenAIReasoning) {
    body.max_completion_tokens = ceiling;
  } else {
    body.max_tokens = ceiling;
    body.temperature = temperature;
  }

  const { status, data } = await backendPost(
    url,
    { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey.trim()}` },
    body,
  );
  if (status < 200 || status >= 300) {
    throw new Error(data?.error?.message || `HTTP ${status}`);
  }
  if (data?.choices?.[0]?.finish_reason === 'length') onTruncated?.();
  const msg = data?.choices?.[0]?.message;
  const raw = typeof msg?.content === 'string' ? msg.content : '';
  return stripThinkTags(raw);
}

/** Ollama-Kontextfenster in sinnvollen Stufen (Speicherverbrauch waechst mit). */
const OLLAMA_CTX_STEPS = [4096, 8192, 16384, 32768];

/**
 * Waehlt `num_ctx` so, dass Prompt UND Antwort hineinpassen.
 *
 * Vorher standen hier fest 4096 Tokens. Ollama schneidet alles darueber
 * STILL ab — und zwar vorne, also genau den System-Prompt. Bei Seitenkontext,
 * Trainingsanalyse oder einem laengeren Dev-Skript antwortete das lokale
 * Modell deshalb an der Frage vorbei, ohne dass ein Fehler sichtbar wurde.
 */
export function ollamaContextSize(promptChars: number, numPredict: number): number {
  const needed = Math.ceil(promptChars / 3.5) + numPredict + 512;
  return OLLAMA_CTX_STEPS.find(step => step >= needed) ?? OLLAMA_CTX_STEPS[OLLAMA_CTX_STEPS.length - 1];
}

async function callOllama(
  model: string, system: string, messages: ChatMessage[],
  maxTokens: number, temperature: number, profile: BudgetProfile, onTruncated?: () => void,
) {
  const numPredict = effectiveMaxTokens(model, maxTokens, profile.thinkingReserve);
  const promptChars = system.length + messages.reduce((n, m) => n + m.content.length, 0);
  let status: number, data: any;
  try {
    ({ status, data } = await backendPost(
      'http://localhost:11434/api/chat',
      { 'Content-Type': 'application/json' },
      {
        model,
        stream: false,
        options: {
          temperature,
          num_ctx: ollamaContextSize(promptChars, numPredict),
          // Ohne num_predict lieferte Ollama bis zum Modell-Default (oft sehr
          // lang) — das eingestellte Token-Budget galt hier ueberhaupt nicht.
          num_predict: numPredict,
        },
        messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.content }))],
      },
    ));
  } catch {
    // Verbindungsfehler (Backend erreicht Ollama nicht)
    throw new Error('Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
  }
  if (status === 404) {
    throw new Error(
      data?.error
        ? `Ollama: ${data.error} — Modell zuerst laden: "ollama pull ${model}".`
        : `Ollama kennt das Modell "${model}" nicht. Zuerst "ollama pull ${model}" ausführen.`,
    );
  }
  if (status < 200 || status >= 300) {
    throw new Error(data?.error || 'Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
  }
  if (data?.done_reason === 'length') onTruncated?.();
  const raw = typeof data?.message?.content === 'string' ? data.message.content : '';
  return stripThinkTags(raw);
}

function decorateSystem(system: string, opts: { responseLanguage?: string; style: ResponseStyle; profile: BudgetProfile }): string {
  const parts = [system];
  const lang = opts.responseLanguage?.trim();
  if (lang) parts.push(`\n\nANTWORTSPRACHE:\n- Antworte ausschließlich auf ${lang}.`);
  // Die Laenge steuert der Prompt, nicht mehr das harte Token-Limit. So bricht
  // keine Antwort mehr mitten im Satz ab und ein hoeheres Budget bedeutet
  // wirklich mehr Tiefe statt nur mehr erlaubte Zeichen.
  if (opts.style === 'chat' && opts.profile.brevity) {
    const en = (lang ?? '').toLowerCase().startsWith('en');
    const text = en ? opts.profile.brevity.en : opts.profile.brevity.de;
    // Die Vorgabe gilt nur fuer Fliesstext. Ohne diesen Zusatz haetten die
    // Code-Assistenten bei "Minimal" ihren Code-Block gekuerzt, statt kurz
    // zu erklaeren und den Block vollstaendig zu liefern.
    const exemption = en
      ? 'Required sections, code blocks, JSON blocks and edit blocks do not count as prose and always stay complete.'
      : 'Geforderte Abschnitte, Code-, JSON- und Edit-Bloecke zaehlen nicht als Fliesstext und bleiben immer vollstaendig.';
    parts.push(`\n\n${en ? 'RESPONSE LENGTH' : 'ANTWORTLÄNGE'}:\n- ${text}\n- ${exemption}`);
  }
  return parts.join('');
}

/**
 * Prueft mit einem minimalen Request, ob Provider + Key + Modell zusammen
 * funktionieren. Wirft mit einer sprechenden Fehlermeldung, wenn nicht.
 * `enabled` wird intern erzwungen, damit auch vor dem Aktivieren getestet
 * werden kann.
 */
export async function testAIConnection(settings: AISettings): Promise<void> {
  const reply = await callAI(
    { ...settings, enabled: true },
    {
      system: 'Reply with exactly the word: OK',
      messages: [{ role: 'user', content: 'ping' }],
      maxTokens: 256,
      temperature: 0,
    },
  );
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
  const profile = budgetProfile(settings);
  const system = decorateSystem(options.system, {
    responseLanguage: options.responseLanguage,
    style: options.style ?? 'raw',
    profile,
  });
  const messages = normalizeMessages(options.messages);
  const onTruncated = options.onTruncated;

  let text: string;
  if (provider === 'anthropic') {
    text = await callAnthropic(settings.apiKey, model, system, messages, maxTokens, temperature, profile, onTruncated);
  } else if (provider === 'openai') {
    text = await callOpenAICompat('https://api.openai.com/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature, profile, onTruncated);
  } else if (provider === 'groq') {
    text = await callOpenAICompat('https://api.groq.com/openai/v1/chat/completions', settings.apiKey, model, system, messages, maxTokens, temperature, profile, onTruncated);
  } else {
    text = await callOllama(model, system, messages, maxTokens, temperature, profile, onTruncated);
  }

  // Eine leere Antwort ist KEIN Erfolg. Sie entstand regelmaessig, wenn ein
  // Thinking-Modell das komplette Budget verdacht hatte: die Analyse wurde als
  // leerer Bericht gespeichert, der Chat zeigte eine leere Blase. Statt still
  // nichts zu liefern, gibt es jetzt eine Fehlermeldung, die sagt was zu tun ist.
  if (!text.trim()) {
    throw new Error(
      'Das Modell hat keinen Text zurückgegeben (das Antwort-Budget ging vermutlich vollständig für internes Nachdenken drauf). '
      + 'In den Einstellungen ein größeres Token-Budget wählen oder ein anderes Modell verwenden.',
    );
  }
  return text;
}
