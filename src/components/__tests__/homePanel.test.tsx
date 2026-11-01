// Die Startseite beantwortet: Laeuft etwas, was wartet auf mich, wo steht jedes
// Modell? Abgesichert sind das Lesen der Daten, der eingefrorene last-seen-
// Zeitstempel, die Spruenge mit Modell und die Robustheit gegen einzelne
// fehlschlagende Commands. Die Regeln selbst testet homePipeline.test.ts.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));

vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));
vi.mock('../../contexts/TrainingContext', () => ({
  useTrainingContext: () => ({
    state: { modelName: '', lossPoints: [], currentJob: null },
    setShowDashboard: vi.fn(),
    setIsDashMinimized: vi.fn(),
  }),
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
import { takeModelFocus } from '../../ui/navigationEvents';

const LAST_SEEN_KEY = 'ft_home_last_seen_u1';

/** Antwort je Command; nicht gesetzte Commands liefern eine leere Liste. */
function mockBackend(responses: Record<string, unknown>) {
  invokeMock.mockImplementation((cmd: string) =>
    cmd in responses ? Promise.resolve(responses[cmd]) : Promise.resolve([]),
  );
}

function training(over: Record<string, unknown> = {}) {
  return {
    id: 'j1', model_id: 'm1', model_name: 'xlm-roberta', dataset_name: 'reviews',
    status: 'completed', created_at: '2026-09-03T02:00:00Z',
    started_at: '2026-09-03T01:00:00Z', completed_at: '2026-09-03T02:00:00Z',
    progress: { epoch: 3, total_epochs: 3, step: 60, total_steps: 60, train_loss: 0.3142, progress_percent: 100 },
    ...over,
  };
}

const MODEL = { id: 'm1', name: 'xlm-roberta' };

describe('HomePanel', () => {
  beforeEach(() => {
    localStorage.clear();
    invokeMock.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  it('zeigt jedes Modell mit Stand und naechster Aktion', async () => {
    mockBackend({ list_models: [MODEL], get_training_history: [training()] });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByText('xlm-roberta')).toBeTruthy());
    expect(screen.getByText('reviews')).toBeTruthy();
    expect(screen.getByText('0,3142')).toBeTruthy();
    expect(screen.getByText(de.home.pipeline.next.startTest)).toBeTruthy();
    expect(screen.queryByText(de.home.pipeline.title.empty)).toBeNull();
  });

  it('nennt im Titel, wie viele Modelle warten', async () => {
    mockBackend({ list_models: [MODEL], get_training_history: [training({ status: 'failed', error: 'CUDA out of memory' })] });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('1 Modell braucht dich'));
    expect(screen.getByText(/CUDA out of memory/)).toBeTruthy();
  });

  it('springt mit dem Modell in den naechsten Schritt', async () => {
    mockBackend({ list_models: [MODEL], get_training_history: [training()] });
    render(<HomePanel userId="u1" />);

    fireEvent.click(await screen.findByText(de.home.pipeline.next.startTest));
    expect(takeModelFocus('tests')?.modelId).toBe('m1');
  });

  it('markiert, was seit dem letzten Besuch passiert ist', async () => {
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-03T00:00:00Z');
    mockBackend({ list_models: [MODEL], get_training_history: [training()] });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByText(de.home.pipeline.new)).toBeTruthy());
  });

  it('markiert nichts, wenn seit dem letzten Besuch nichts passiert ist', async () => {
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-04T00:00:00Z');
    mockBackend({ list_models: [MODEL], get_training_history: [training()] });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByText('xlm-roberta')).toBeTruthy());
    expect(screen.queryByText(de.home.pipeline.new)).toBeNull();
  });

  it('schreibt den last-seen-Zeitstempel erst beim Verlassen der Seite', async () => {
    localStorage.setItem(LAST_SEEN_KEY, '2026-09-03T00:00:00Z');
    mockBackend({ list_models: [MODEL], get_training_history: [training()] });
    const { unmount } = render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByText(de.home.pipeline.new)).toBeTruthy());
    expect(localStorage.getItem(LAST_SEEN_KEY)).toBe('2026-09-03T00:00:00Z');
    unmount();
    expect(localStorage.getItem(LAST_SEEN_KEY)).not.toBe('2026-09-03T00:00:00Z');
  });

  it('bleibt nutzbar, wenn einzelne Commands fehlschlagen', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'list_models') return Promise.resolve([MODEL]);
      if (cmd === 'get_training_history') return Promise.reject(new Error('kaputt'));
      return Promise.resolve([]);
    });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByText('xlm-roberta')).toBeTruthy());
  });

  it('startet neue Nutzer mit den vier Schritten', async () => {
    mockBackend({});
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByRole('heading', { level: 1 }).textContent).toBe(de.home.pipeline.title.empty));
    expect(screen.getByText(de.home.pipeline.first.intro)).toBeTruthy();
    expect(screen.getByText(de.home.firstRun.cta)).toBeTruthy();
  });

  it('zeigt ein laufendes Training mit Fortschritt', async () => {
    mockBackend({
      list_models: [MODEL],
      get_training_history: [training({ status: 'running', completed_at: null })],
      list_active_trainings: [{
        training_id: 't1', status: 'running', current_epoch: 1, total_epochs: 3,
        current_step: 20, total_steps: 60, progress_percentage: 33.3,
        train_loss: 0.5, elapsed_time_seconds: 125,
      }],
    });
    render(<HomePanel userId="u1" />);

    await waitFor(() => expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('xlm-roberta trainiert'));
    expect(screen.getByText('20 / 60')).toBeTruthy();
    expect(screen.getByText('Training, 33 %')).toBeTruthy();
  });

  it('vergleicht getestete Modelle je Aufgabe', async () => {
    mockBackend({
      list_models: [MODEL],
      get_test_history: [{
        id: 's1', model_id: 'm1', model_name: 'xlm-roberta', version_name: 'v1', dataset_name: 'holdout',
        status: 'completed', created_at: '2026-09-03T02:00:00Z', completed_at: '2026-09-03T02:10:00Z',
        results: { total_samples: 10, accuracy: 0.923 },
      }],
    });
    render(<HomePanel userId="u1" />);

    fireEvent.click(await screen.findByText(de.home.pipeline.compare.title));
    expect(screen.getByText('92,3%')).toBeTruthy();
    expect(screen.getByText('holdout')).toBeTruthy();
  });
});
