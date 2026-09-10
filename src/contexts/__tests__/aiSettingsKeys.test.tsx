// Ein API-Key PRO ANBIETER.
//
// Vorher teilten sich alle Anbieter ein einziges Schluesselbund-Konto: wer von
// Claude auf Groq wechselte und dort speicherte, hatte den Claude-Key verloren
// und musste ihn beim Zurueckwechseln neu eintippen. Zusaetzlich loeschte ein
// Speichern mit leerem Feld den hinterlegten Key — auch dann, wenn im Feld nur
// deshalb nichts stand, weil gerade erst umgeschaltet worden war.
//
// Ausfuehren: npx vitest run src/contexts/__tests__/aiSettingsKeys.test.tsx --config vitest.config.ts

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));

import { AISettingsProvider, useAISettings } from '../AISettingsContext';

/** Schluesselbund-Attrappe: merkt sich Werte pro Konto. */
function keychain(initial: Record<string, string> = {}, calls?: string[]) {
  const store: Record<string, string> = { ...initial };
  mockInvoke.mockImplementation(async (cmd: string, args: any) => {
    calls?.push(cmd);
    if (cmd === 'secret_get') return store[args.key] ?? null;
    if (cmd === 'secret_set') { store[args.key] = args.value; return null; }
    if (cmd === 'secret_delete') { delete store[args.key]; return null; }
    return null;
  });
  return store;
}

const wrapper = ({ children }: { children: ReactNode }) => (
  <AISettingsProvider userId="u1">{children}</AISettingsProvider>
);

/** Startzustand in localStorage — nur Nicht-Geheimes, wie in der App. */
function storedSettings(provider: string) {
  localStorage.setItem('ft_ai_settings_u1', JSON.stringify({
    enabled: true, provider, apiKey: '', selectedModel: 'claude-sonnet-5',
    ollamaModel: 'llama3.2', tokenBudget: 'balanced',
  }));
}

async function mounted() {
  const hook = renderHook(() => useAISettings(), { wrapper });
  await waitFor(() => expect(hook.result.current.keyLoading).toBe(false));
  return hook;
}

describe('AISettings – Key pro Anbieter', () => {
  beforeEach(() => {
    localStorage.clear();
    mockInvoke.mockReset();
  });

  it('laedt den Key aus dem Konto des gewaehlten Anbieters', async () => {
    keychain({ 'ft_ai_key_u1_anthropic': 'sk-ant-claude', 'ft_ai_key_u1_groq': 'gsk_groq' });
    storedSettings('anthropic');
    const { result } = await mounted();
    expect(result.current.settings.apiKey).toBe('sk-ant-claude');
  });

  it('holt beim Anbieterwechsel den Key des neuen Anbieters und behaelt den alten', async () => {
    const store = keychain({ 'ft_ai_key_u1_anthropic': 'sk-ant-claude', 'ft_ai_key_u1_groq': 'gsk_groq' });
    storedSettings('anthropic');
    const { result } = await mounted();

    act(() => { result.current.updateDraft({ provider: 'groq' }); });
    await waitFor(() => expect(result.current.draft.apiKey).toBe('gsk_groq'));

    await act(async () => { await result.current.saveSettings(); });
    // Groq wurde geschrieben, Claude blieb unangetastet.
    expect(store['ft_ai_key_u1_groq']).toBe('gsk_groq');
    expect(store['ft_ai_key_u1_anthropic']).toBe('sk-ant-claude');

    // Zurueck auf Claude: der Key ist noch da.
    act(() => { result.current.updateDraft({ provider: 'anthropic' }); });
    await waitFor(() => expect(result.current.draft.apiKey).toBe('sk-ant-claude'));
  });

  it('loescht beim Speichern keinen Key, wenn nie einer geladen war', async () => {
    const store = keychain({ 'ft_ai_key_u1_anthropic': 'sk-ant-claude' });
    storedSettings('ollama');
    const { result } = await mounted();

    // Wechsel auf einen Anbieter ohne hinterlegten Key und sofort speichern.
    act(() => { result.current.updateDraft({ provider: 'openai' }); });
    await act(async () => { await result.current.saveSettings(); });

    expect(store['ft_ai_key_u1_anthropic']).toBe('sk-ant-claude');
    expect(store['ft_ai_key_u1_openai']).toBeUndefined();
  });

  it('loescht den Key, wenn ein geladener Key bewusst geleert wird', async () => {
    const store = keychain({ 'ft_ai_key_u1_anthropic': 'sk-ant-claude' });
    storedSettings('anthropic');
    const { result } = await mounted();

    act(() => { result.current.updateDraft({ apiKey: '' }); });
    await act(async () => { await result.current.saveSettings(); });

    expect(store['ft_ai_key_u1_anthropic']).toBeUndefined();
  });

  // Wer die App vorher genutzt hat, hat seinen Key im alten gemeinsamen Konto.
  it('uebernimmt den Key aus dem frueheren gemeinsamen Konto', async () => {
    const store = keychain({ 'ft_ai_key_u1': 'sk-ant-alt' });
    storedSettings('anthropic');
    const { result } = await mounted();

    expect(result.current.settings.apiKey).toBe('sk-ant-alt');
    expect(store['ft_ai_key_u1_anthropic']).toBe('sk-ant-alt');
  });

  // Die Anzeige meldete "API-Key noetig" fuer Anbieter, deren Key laengst im
  // Schluesselbund lag — sichtbar wurde er erst beim Umschalten. Genau die
  // Beruhigung, die die Anzeige geben soll, fehlte damit.
  it('meldet ALLE Anbieter mit hinterlegtem Key, ohne dass man umschalten muss', async () => {
    keychain({ 'ft_ai_key_u1_anthropic': 'sk-ant-claude', 'ft_ai_key_u1_groq': 'gsk_groq' });
    storedSettings('anthropic');
    const { result } = await mounted();

    await waitFor(() => expect(result.current.providersWithKey).toEqual(['anthropic', 'groq']));

    // Umschalten aendert daran nichts und verliert keinen Key.
    act(() => { result.current.updateDraft({ provider: 'groq' }); });
    await waitFor(() => expect(result.current.draft.apiKey).toBe('gsk_groq'));
    expect(result.current.providersWithKey).toEqual(['anthropic', 'groq']);
  });

  // Ein reiner Ollama-Nutzer soll den Schluesselbund gar nicht erst wecken.
  it('fasst den Schluesselbund nicht an, solange nur Ollama eingestellt ist', async () => {
    const calls: string[] = [];
    keychain({}, calls);
    storedSettings('ollama');
    const { result } = await mounted();

    await waitFor(() => expect(result.current.keyLoading).toBe(false));
    expect(calls).toHaveLength(0);
    expect(result.current.providersWithKey).toEqual([]);
  });
});
