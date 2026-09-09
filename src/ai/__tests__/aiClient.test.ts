// Absicherung der Verlaufs-Aufbereitung in callAI.
// Ausfuehren: npx vitest run src/ai/__tests__/aiClient.test.ts --config vitest.config.ts

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { AISettings } from '../../contexts/AISettingsContext';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));

import { callAI } from '../aiClient';

/** Antwortet wie die Anthropic-API und merkt sich den gesendeten Body. */
function respondAnthropic(text = 'ok', stopReason = 'end_turn') {
  mockInvoke.mockImplementation(async () => ({
    status: 200,
    body: JSON.stringify({ content: [{ type: 'text', text }], stop_reason: stopReason }),
  }));
}

function sentBody(): any {
  return JSON.parse(mockInvoke.mock.calls[0][1].body);
}

const ANTHROPIC: AISettings = {
  enabled: true, provider: 'anthropic', apiKey: 'sk-ant-api03-test',
  selectedModel: 'claude-haiku-4-5', ollamaModel: '', tokenBudget: 'balanced',
};

describe('callAI – Verlaufsaufbereitung', () => {
  beforeEach(() => { mockInvoke.mockReset(); });

  // Der Metrik-Assistent stellt den fertigen Bericht als erste
  // assistant-Nachricht in den Verlauf. Anthropic verlangt aber `user` als
  // erste Rolle und antwortete sonst mit HTTP 400 — der Chat war damit tot.
  it('entfernt fuehrende assistant-Nachrichten fuer Anthropic', async () => {
    respondAnthropic();
    await callAI(ANTHROPIC, {
      system: 'sys',
      messages: [
        { role: 'assistant', content: 'Bericht aus der Analyse' },
        { role: 'user', content: 'Wie ist die mAP?' },
      ],
    });
    expect(sentBody().messages).toEqual([{ role: 'user', content: 'Wie ist die mAP?' }]);
  });

  it('wirft verstaendlich, wenn nach dem Bereinigen nichts uebrig ist', async () => {
    respondAnthropic();
    await expect(callAI(ANTHROPIC, {
      system: 'sys',
      messages: [{ role: 'assistant', content: 'nur ein Bericht' }],
    })).rejects.toThrow(/Keine Nachricht/);
  });

  // Eine leer zurueckgekommene Antwort landete als leere assistant-Nachricht
  // im Verlauf; Anthropic lehnt leere Text-Bloecke ab.
  it('verwirft leere Nachrichten und fasst gleiche Rollen zusammen', async () => {
    respondAnthropic();
    await callAI(ANTHROPIC, {
      system: 'sys',
      messages: [
        { role: 'user', content: 'erste Frage' },
        { role: 'assistant', content: '   ' },
        { role: 'user', content: 'zweite Frage' },
      ],
    });
    expect(sentBody().messages).toEqual([
      { role: 'user', content: 'erste Frage\n\nzweite Frage' },
    ]);
  });

  it('meldet ein Abschneiden am Token-Limit', async () => {
    respondAnthropic('halber Satz', 'max_tokens');
    const onTruncated = vi.fn();
    await callAI(ANTHROPIC, {
      system: 'sys',
      messages: [{ role: 'user', content: 'analysiere' }],
      onTruncated,
    });
    expect(onTruncated).toHaveBeenCalledTimes(1);
  });

  it('meldet kein Abschneiden bei normalem Ende', async () => {
    respondAnthropic('ganzer Satz');
    const onTruncated = vi.fn();
    await callAI(ANTHROPIC, {
      system: 'sys',
      messages: [{ role: 'user', content: 'analysiere' }],
      onTruncated,
    });
    expect(onTruncated).not.toHaveBeenCalled();
  });

  // Thinking-Modelle verbrauchen einen Teil von max_tokens fuers Nachdenken.
  it('schlaegt Thinking-Modellen Budget auf, anderen nicht', async () => {
    respondAnthropic();
    await callAI({ ...ANTHROPIC, selectedModel: 'claude-opus-5' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 2000,
    });
    expect(sentBody().max_tokens).toBeGreaterThan(2000);

    mockInvoke.mockReset();
    respondAnthropic();
    await callAI(ANTHROPIC, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 2000,
    });
    expect(sentBody().max_tokens).toBe(2000);
  });
});

// ---------------------------------------------------------------------------
// Token-Budget: Denk-Reserve, Effort und Laengen-Vorgabe
//
// Der Praxisfall, der das noetig machte: bei "Minimal" (400 Tokens) und
// claude-sonnet-5 verbrauchte das unsichtbare Nachdenken das komplette
// Budget. Trainingsanalyse und Analyse-Chat lieferten daraufhin einen leeren
// Text — gespeichert und angezeigt als "Erfolg".
// ---------------------------------------------------------------------------
describe('callAI – Token-Budget', () => {
  beforeEach(() => { mockInvoke.mockReset(); });

  const THINKING: AISettings = { ...ANTHROPIC, selectedModel: 'claude-sonnet-5' };

  it('schlaegt die Denk-Reserve auf das sichtbare Budget auf', async () => {
    respondAnthropic();
    await callAI({ ...THINKING, tokenBudget: 'minimal' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 400,
    });
    expect(sentBody().max_tokens).toBe(400 + 2000);
  });

  it('waehlt den Denk-Aufwand passend zum Budget', async () => {
    respondAnthropic();
    await callAI({ ...THINKING, tokenBudget: 'minimal' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }],
    });
    expect(sentBody().output_config).toEqual({ effort: 'low' });

    mockInvoke.mockReset();
    respondAnthropic();
    await callAI({ ...THINKING, tokenBudget: 'unlimited' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }],
    });
    expect(sentBody().output_config).toEqual({ effort: 'max' });
  });

  it('setzt keinen Effort bei Modellen ohne Effort-Stufen', async () => {
    respondAnthropic();
    await callAI(ANTHROPIC, { system: 'sys', messages: [{ role: 'user', content: 'x' }] });
    expect(sentBody().output_config).toBeUndefined();
  });

  it('gibt die Laenge nur bei style="chat" im System-Prompt vor', async () => {
    respondAnthropic();
    await callAI({ ...ANTHROPIC, tokenBudget: 'minimal' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], style: 'chat',
    });
    expect(sentBody().system).toMatch(/hoechstens 3 Saetze/);

    mockInvoke.mockReset();
    respondAnthropic();
    await callAI({ ...ANTHROPIC, tokenBudget: 'minimal' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], style: 'raw',
    });
    expect(sentBody().system).not.toMatch(/hoechstens/);
  });

  it('wirft statt still leer zu antworten', async () => {
    mockInvoke.mockImplementation(async () => ({
      status: 200,
      body: JSON.stringify({ content: [{ type: 'thinking', thinking: '…' }], stop_reason: 'max_tokens' }),
    }));
    await expect(callAI(THINKING, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }],
    })).rejects.toThrow(/keinen Text/);
  });
});

// ---------------------------------------------------------------------------
// Ollama: Kontextfenster und Antwortlaenge
// ---------------------------------------------------------------------------
describe('callAI – Ollama', () => {
  beforeEach(() => { mockInvoke.mockReset(); });

  const OLLAMA: AISettings = {
    enabled: true, provider: 'ollama', apiKey: '',
    selectedModel: 'llama3.2', ollamaModel: 'llama3.2', tokenBudget: 'balanced',
  };

  function respondOllama(content = 'ok', extra: Record<string, unknown> = {}) {
    mockInvoke.mockImplementation(async () => ({
      status: 200,
      body: JSON.stringify({ message: { content }, ...extra }),
    }));
  }

  it('begrenzt die Antwortlaenge (num_predict) statt sie laufen zu lassen', async () => {
    respondOllama();
    await callAI(OLLAMA, { system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 800 });
    expect(sentBody().options.num_predict).toBe(800);
  });

  // Ollama schneidet alles ab, was nicht in num_ctx passt — und zwar vorne,
  // also genau den System-Prompt. Bei 4096 fest verdrahtet ging der
  // Seitenkontext still verloren.
  it('vergroessert das Kontextfenster fuer lange Prompts', async () => {
    respondOllama();
    await callAI(OLLAMA, {
      system: 'S'.repeat(60000), messages: [{ role: 'user', content: 'x' }], maxTokens: 800,
    });
    expect(sentBody().options.num_ctx).toBeGreaterThanOrEqual(16384);
  });

  it('entfernt sichtbare Denk-Bloecke lokaler Reasoning-Modelle', async () => {
    respondOllama('<think>lange Ueberlegung</think>Die Antwort.');
    const reply = await callAI(OLLAMA, { system: 'sys', messages: [{ role: 'user', content: 'x' }] });
    expect(reply).toBe('Die Antwort.');
  });

  it('erklaert ein fehlendes Modell', async () => {
    mockInvoke.mockImplementation(async () => ({
      status: 404, body: JSON.stringify({ error: 'model "llama3.2" not found' }),
    }));
    await expect(callAI(OLLAMA, { system: 'sys', messages: [{ role: 'user', content: 'x' }] }))
      .rejects.toThrow(/ollama pull/);
  });

  it('meldet ein Abschneiden am Limit', async () => {
    respondOllama('halber Satz', { done_reason: 'length' });
    const onTruncated = vi.fn();
    await callAI(OLLAMA, { system: 'sys', messages: [{ role: 'user', content: 'x' }], onTruncated });
    expect(onTruncated).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// OpenAI-kompatible Provider (OpenAI, Groq)
// ---------------------------------------------------------------------------
describe('callAI – OpenAI/Groq', () => {
  beforeEach(() => { mockInvoke.mockReset(); });

  const GROQ: AISettings = {
    enabled: true, provider: 'groq', apiKey: 'gsk_test',
    selectedModel: 'llama-3.3-70b-versatile', ollamaModel: '', tokenBudget: 'balanced',
  };

  function respondChat(content = 'ok', finish = 'stop') {
    mockInvoke.mockImplementation(async () => ({
      status: 200,
      body: JSON.stringify({ choices: [{ message: { content }, finish_reason: finish }] }),
    }));
  }

  it('schickt max_tokens und temperature fuer normale Modelle', async () => {
    respondChat();
    await callAI(GROQ, { system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 800 });
    const body = sentBody();
    expect(body.max_tokens).toBe(800);
    expect(body.temperature).toBeDefined();
  });

  it('gibt Reasoning-Modellen zusaetzliches Budget', async () => {
    respondChat();
    await callAI({ ...GROQ, selectedModel: 'openai/gpt-oss-120b' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 800,
    });
    expect(sentBody().max_tokens).toBe(800 + 3000);
  });

  // o-Serie und GPT-5 lehnen max_tokens sowie abweichende Temperaturen ab.
  it('nutzt max_completion_tokens fuer die OpenAI-o-Serie', async () => {
    respondChat();
    await callAI({ ...GROQ, provider: 'openai', apiKey: 'sk-test', selectedModel: 'o3-mini' }, {
      system: 'sys', messages: [{ role: 'user', content: 'x' }], maxTokens: 800,
    });
    const body = sentBody();
    expect(body.max_completion_tokens).toBeGreaterThan(800);
    expect(body.max_tokens).toBeUndefined();
    expect(body.temperature).toBeUndefined();
  });
});
