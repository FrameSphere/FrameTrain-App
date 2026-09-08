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
