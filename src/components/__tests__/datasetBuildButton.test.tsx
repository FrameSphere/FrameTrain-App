// Der Einstieg vom Dataset-Bereich in die Werkstatt.
//
// Der Knopf ist der einzige Weg dorthin — die Seitenleiste hat keinen eigenen
// Eintrag. Verschwindet er bei einem Umbau des Kopfbereichs, ist das Studio
// fuer den Nutzer nicht mehr erreichbar, ohne dass irgendetwas fehlschlaegt.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({ onDragDropEvent: () => Promise.resolve(() => {}) }),
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));
vi.mock('../../contexts/PageContext', () => ({
  usePageContext: () => ({ setCurrentPageContent: vi.fn() }),
}));
vi.mock('../../ui/contextMenuRegistry', () => ({ useContextMenuActions: () => {} }));
vi.mock('../../ai/coachToolEvents', () => ({
  onCoachCommand: () => () => {},
  consumePendingCoachCommand: () => null,
}));

import DatasetUpload from '../DatasetUpload';

describe('Dataset-Bereich', () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models') return [{ id: 'm1', name: 'yolo8n', source: 'local' }];
      if (cmd === 'list_datasets_for_model') return [];
      if (cmd === 'get_dataset_filter_options') return { tasks: [], languages: [], sizes: [] };
      return null;
    });
  });

  it('fuehrt mit "Datensatz bauen" in die Werkstatt', async () => {
    const seen: string[] = [];
    const listener = (e: Event) => seen.push((e as CustomEvent<string>).detail);
    window.addEventListener('ft_navigate', listener as EventListener);
    try {
      render(<DatasetUpload />);
      const button = await screen.findByRole('button', { name: /Datensatz bauen/ });
      fireEvent.click(button);
      expect(seen).toEqual(['studio']);
    } finally {
      window.removeEventListener('ft_navigate', listener as EventListener);
    }
  });

  it('haelt den Import daneben unveraendert erreichbar', async () => {
    render(<DatasetUpload />);
    // Der Knopf steht im Kopf und noch einmal im leeren Zustand.
    expect((await screen.findAllByRole('button', { name: /Dataset hinzufügen/ })).length)
      .toBeGreaterThanOrEqual(1);
  });
});
