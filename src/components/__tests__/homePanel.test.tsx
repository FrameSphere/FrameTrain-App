// Die Startseite beantwortet vor allem eine Frage: "Was ist passiert, während
// ich weg war?" Diese Antwort haengt an zwei leicht zu brechenden Details —
// dem eingefrorenen last-seen-Zeitstempel und dem Filter darauf. Beides hier
// abgesichert, plus die Robustheit gegen einzelne fehlschlagende Commands.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));

vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));
vi.mock('../../contexts/PageContext', () => ({
  usePageContext: () => ({ setCurrentPageContent: vi.fn() }),
}));
vi.mock('../../contexts/AISettingsContext', () => ({
  useAISettings: () => ({
    settings: { enabled: false, provider: 'anthropic', apiKey: '', selectedModel: '', ollamaModel: '', tokenBudget: 'balanced' },
  }),
}));

import de from '../../locales/de.json';
vi.mock('../../contexts/LanguageContext', async () => {
  const locales = (await import('../../locales/de.json')).default as Record<string, unknown>;
  return {
    useLanguage: () => ({
      language: 'de' as const,
      t: (key: string, params?: Record<string, string>) => {
        const value = key.split('.').reduce<unknown>(
          (o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined),
          locales,
        );
        let s = typeof value === 'string' ? value : key;
        for (const [k, v] of Object.entries(params ?? {})) s = s.split(`{${k}}`).join(v);
        return s;
      },
    }),
  };
});

import HomePanel from '../HomePanel';

const LAST_SEEN_KEY = 'ft_home_last_seen_u1';

/** Antwort je Command; nicht gesetzte Commands liefern eine leere Liste. */
function mockBackend(responses: Record<string, unknown>) {
  invokeMock.mockImplementation((cmd: string) =>
    cmd in responses ? Promise.resolve(responses[cmd]) : Promise.resolve([]),
  );
}

function training(over: Record<string, unknown> = {}) {
  return {
    id: 'j1', model_name: 'xlm-roberta', dataset_name: 'reviews',
    status: 'completed', created_at: '2026-09-03T02:00:00Z',
    started_at: '2026-09-03T01:00:00Z', completed_at: '2026-09-03T02:00:00Z',
    progress: { epoch: 3, total_epochs: 3, step: 60, total_steps: 60, train_loss: 0.3142, progress_percent: 100 },
    ...over,
  };
}

describe('HomePanel', () => {
  beforeEach(() => {
    localStorage.clear();
    invokeMock.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  it('zeigt ein Training, das seit dem letzten Besuch fertig wurde', async () => {
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-03T00:00:00Z');
    mockBackend({ get_training_history: [training()] });

    render(<HomePanel userEmail="karol@example.com" userId="u1" />);

    await waitFor(() => expect(screen.getByText(de.home.since.title)).toBeTruthy());
    expect(screen.getByText(/Training abgeschlossen: xlm-roberta/)).toBeTruthy();
  });

  it('blendet den Block aus, wenn seit dem letzten Besuch nichts passiert ist', async () => {
    // Letzter Besuch NACH dem Trainingsende → keine Neuigkeiten.
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-04T00:00:00Z');
    mockBackend({ get_training_history: [training()] });

    render(<HomePanel userEmail="karol@example.com" userId="u1" />);

    await waitFor(() => expect(screen.getByText(de.home.recentTrainings.title)).toBeTruthy());
    expect(screen.queryByText(de.home.since.title)).toBeNull();
  });

  it('schreibt den last-seen-Zeitstempel erst beim Verlassen der Seite', async () => {
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-03T00:00:00Z');
    mockBackend({ get_training_history: [training()] });

    const { unmount } = render(<HomePanel userEmail="karol@example.com" userId="u1" />);
    await waitFor(() => expect(screen.getByText(de.home.since.title)).toBeTruthy());

    // Solange die Seite offen ist, bleibt der alte Stand stehen — sonst
    // verschwaende die Liste beim ersten Re-Render unter den Haenden.
    expect(localStorage.getItem(LAST_SEEN_KEY)).toBe('2026-09-03T00:00:00Z');
    unmount();
    expect(localStorage.getItem(LAST_SEEN_KEY)).not.toBe('2026-09-03T00:00:00Z');
  });

  it('bleibt nutzbar, wenn einzelne Commands fehlschlagen', async () => {
    invokeMock.mockImplementation((cmd: string) =>
      cmd === 'get_training_history'
        ? Promise.reject(new Error('DB weg'))
        : Promise.resolve([]),
    );

    render(<HomePanel userEmail="karol@example.com" userId="u1" />);

    // Kein Absturz, die uebrigen Bereiche rendern weiter.
    await waitFor(() => expect(screen.getByText(de.home.recentTrainings.empty)).toBeTruthy());
    expect(screen.getByText(de.home.recentTests.empty)).toBeTruthy();
  });

  it('zeigt ein laufendes Training mit Fortschritt', async () => {
    mockBackend({
      list_active_trainings: [{
        training_id: 't1', status: 'running', current_epoch: 1, total_epochs: 3,
        current_step: 20, total_steps: 60, progress_percentage: 33.3,
        train_loss: 0.5, elapsed_time_seconds: 125,
      }],
    });

    render(<HomePanel userEmail="karol@example.com" userId="u1" />);

    await waitFor(() => expect(screen.getByText(de.home.running.title)).toBeTruthy());
    expect(screen.getByText(/Schritt 20\/60/)).toBeTruthy();
  });
});
