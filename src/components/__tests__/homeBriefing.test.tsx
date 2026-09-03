// Das Briefing ist die einzige Stelle der Startseite, die Geld kostet.
// Getestet wird deshalb vor allem, WANN es die KI ruft — und wann eben nicht.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const callAIMock = vi.fn();
vi.mock('../../ai/aiClient', () => ({ callAI: (...a: unknown[]) => callAIMock(...a) }));

vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));

const settingsRef = { current: { enabled: true, provider: 'anthropic', apiKey: 'sk-ant-api-x', selectedModel: 'claude-haiku-4-5', ollamaModel: '', tokenBudget: 'balanced' } };
vi.mock('../../contexts/AISettingsContext', () => ({
  useAISettings: () => ({ settings: settingsRef.current }),
}));

vi.mock('../../contexts/LanguageContext', async () => {
  const locales = (await import('../../locales/de.json')).default as Record<string, unknown>;
  return {
    useLanguage: () => ({
      language: 'de' as const,
      t: (key: string, params?: Record<string, string>) => {
        const v = key.split('.').reduce<unknown>(
          (o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), locales);
        let s = typeof v === 'string' ? v : key;
        for (const [k, val] of Object.entries(params ?? {})) s = s.split(`{${k}}`).join(val);
        return s;
      },
    }),
  };
});

import de from '../../locales/de.json';
import HomeBriefing from '../HomeBriefing';

const CACHE_KEY = 'ft_home_briefing_u1';
const enabled = { enabled: true, provider: 'anthropic', apiKey: 'sk-ant-api-x', selectedModel: 'claude-haiku-4-5', ollamaModel: '', tokenBudget: 'balanced' };

describe('HomeBriefing', () => {
  beforeEach(() => {
    localStorage.clear();
    callAIMock.mockReset();
    settingsRef.current = { ...enabled };
  });
  afterEach(() => vi.restoreAllMocks());

  it('verweist auf die Einstellungen, solange die KI nicht eingerichtet ist', () => {
    settingsRef.current = { ...enabled, enabled: false };
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);
    expect(screen.getByText(de.home.briefing.toSettings)).toBeTruthy();
    expect(screen.queryByText(de.home.briefing.create)).toBeNull();
  });

  it('verweist auch bei fehlendem API-Key auf die Einstellungen', () => {
    settingsRef.current = { ...enabled, apiKey: '' };
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);
    expect(screen.getByText(de.home.briefing.toSettings)).toBeTruthy();
  });

  it('ruft die KI nicht von selbst — erst auf Klick', async () => {
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);
    await waitFor(() => expect(screen.getByText(de.home.briefing.create)).toBeTruthy());
    expect(callAIMock).not.toHaveBeenCalled();
  });

  it('erzeugt das Briefing auf Klick und merkt es sich', async () => {
    callAIMock.mockResolvedValue('  Das Training laeuft gut.  ');
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);

    await userEvent.click(screen.getByText(de.home.briefing.create));

    await waitFor(() => expect(screen.getByText('Das Training laeuft gut.')).toBeTruthy());
    const stored = JSON.parse(localStorage.getItem(CACHE_KEY)!);
    expect(stored.hash).toBe('h1');
    expect(stored.text).toBe('Das Training laeuft gut.');
    // Die Fakten muessen wirklich mitgehen, sonst schreibt die KI ins Blaue.
    expect(callAIMock.mock.calls[0][1].messages[0].content).toContain('stand');
  });

  it('zeigt einen gespeicherten Text zum selben Stand ohne neuen Aufruf', async () => {
    localStorage.setItem(CACHE_KEY, JSON.stringify({
      hash: 'h1', text: 'Alter Text', at: new Date().toISOString(), model: 'claude-haiku-4-5',
    }));
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);

    await waitFor(() => expect(screen.getByText('Alter Text')).toBeTruthy());
    expect(callAIMock).not.toHaveBeenCalled();
    expect(screen.queryByText(de.home.briefing.refresh)).toBeNull();
  });

  it('markiert einen Text als veraltet, sobald sich der Stand geaendert hat', async () => {
    localStorage.setItem(CACHE_KEY, JSON.stringify({
      hash: 'alt', text: 'Alter Text', at: new Date().toISOString(), model: 'claude-haiku-4-5',
    }));
    render(<HomeBriefing facts="stand" factsKey="neu" userId="u1" />);

    await waitFor(() => expect(screen.getByText(/veraltet/)).toBeTruthy());
    expect(screen.getByText(de.home.briefing.refresh)).toBeTruthy();
    expect(callAIMock).not.toHaveBeenCalled();
  });

  it('rendert die Antwort als Markdown statt als Rohtext', async () => {
    callAIMock.mockResolvedValue(
      'Der Lauf ist durch.\n- **xlm-roberta** liegt bei **0.3142**\n\n**Naechster Schritt:** eine Epoche weniger',
    );
    const { container } = render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);

    await userEvent.click(screen.getByText(de.home.briefing.create));

    await waitFor(() => expect(container.querySelector('ul li')).toBeTruthy());
    // Die Sternchen duerfen nicht mehr im sichtbaren Text stehen.
    expect(container.textContent).not.toContain('**');
    expect(screen.getByText('xlm-roberta').tagName).toBe('STRONG');
    // Die Empfehlung steht abgesetzt, nicht als vierter Absatz im Fliesstext.
    expect(screen.getByText('Naechster Schritt:').tagName).toBe('STRONG');
    expect(screen.getByText(/eine Epoche weniger/)).toBeTruthy();
  });

  it('zeigt einen Fehler der KI an, statt ihn zu verschlucken', async () => {
    callAIMock.mockRejectedValue(new Error('API-Key abgelaufen'));
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);

    await userEvent.click(screen.getByText(de.home.briefing.create));

    await waitFor(() => expect(screen.getByText('API-Key abgelaufen')).toBeTruthy());
    expect(localStorage.getItem(CACHE_KEY)).toBeNull();
  });

  it('behandelt eine leere Antwort als Fehler statt sie zu speichern', async () => {
    callAIMock.mockResolvedValue('   ');
    render(<HomeBriefing facts="stand" factsKey="h1" userId="u1" />);

    await userEvent.click(screen.getByText(de.home.briefing.create));

    await waitFor(() => expect(screen.getByText(de.home.briefing.emptyAnswer)).toBeTruthy());
    expect(localStorage.getItem(CACHE_KEY)).toBeNull();
  });
});
