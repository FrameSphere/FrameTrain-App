// Rechtsklick und Tastenkuerzel im Dataset-Bereich.
//
// Auf einer Karte geht es um diesen Datensatz (Dateien, Aufteilen, Halbieren,
// Finder, Loeschen), daneben um die Seite. ⌘N fuegt einen Datensatz hinzu,
// ⌘B oeffnet die Werkstatt.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

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
vi.mock('../../ai/coachToolEvents', () => ({
  onCoachCommand: () => () => {},
  consumePendingCoachCommand: () => null,
}));

import DatasetUpload from '../DatasetUpload';
import { collectContextMenuActions } from '../../ui/contextMenuRegistry';

const DATENSATZ = {
  id: 'ds_1', name: 'imdb', model_id: 'm1', source: 'local', source_path: null,
  storage_path: '/daten/imdb', size_bytes: 1000, file_count: 3, created_at: '2026-09-20T10:00:00Z',
  status: 'unused', split_info: null, training_count: 0, last_used_at: null,
  extensions: ['.csv'], dataset_type: 'flat_file', warnings: [],
};

// Die Seite laedt erst Modelle, dann Datensaetze — unter Last dauert das.
describe('Dataset-Bereich: Rechtsklick und Kuerzel', { timeout: 15000 }, () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models') return [{ id: 'm1', name: 'bert', source: 'local' }];
      if (cmd === 'list_datasets_for_model') return [DATENSATZ];
      if (cmd === 'get_dataset_filter_options') return { tasks: [], languages: [], sizes: [] };
      return null;
    });
  });

  it('bietet auf einer Karte die Aktionen dieses Datensatzes an', async () => {
    render(<DatasetUpload />);
    const titel = await screen.findByText('imdb', {}, { timeout: 5000 });
    const karte = titel.closest('[data-dataset-id]') as HTMLElement;
    expect(karte).toBeTruthy();

    const aktionen = collectContextMenuActions({ target: titel as HTMLElement });
    expect(aktionen.slice(0, 5).map(a => a.id))
      .toEqual(['ds-card-files', 'ds-card-split', 'ds-card-halve', 'ds-card-finder', 'ds-card-delete']);
    expect(aktionen[0].group).toBe('imdb');
    expect(aktionen.find(a => a.id === 'ds-card-delete')?.danger).toBe(true);

    aktionen.find(a => a.id === 'ds-card-finder')!.onSelect();
    expect(invokeMock).toHaveBeenCalledWith('open_path_in_finder', { path: '/daten/imdb' });
  });

  it('zeigt neben den Karten nur die Seitenaktionen mit ihren Kuerzeln', async () => {
    render(<DatasetUpload />);
    await screen.findByText('imdb', {}, { timeout: 5000 });
    const aktionen = collectContextMenuActions({ target: document.body });
    expect(aktionen.map(a => a.id)).toEqual(['ds-import', 'ds-build', 'ds-refresh']);
    expect(aktionen.map(a => a.shortcut)).toEqual(['⌘N', '⌘B', undefined]);
    // Nicht der Text aus dem leeren Zustand ("ersten Datensatz") — es gibt ja schon welche.
    expect(aktionen[0].label).toBe('Dataset hinzufügen');
  });

  it('oeffnet mit Cmd+B die Werkstatt und mit Cmd+N den Import', async () => {
    const seen: string[] = [];
    const listener = (e: Event) => seen.push((e as CustomEvent<string>).detail);
    window.addEventListener('ft_navigate', listener as EventListener);
    try {
      render(<DatasetUpload />);
      await screen.findByText('imdb', {}, { timeout: 5000 });
      fireEvent.keyDown(window, { key: 'b', metaKey: true });
      expect(seen).toEqual(['studio']);

      fireEvent.keyDown(window, { key: 'n', metaKey: true });
      await waitFor(() => expect(screen.getAllByText(/Dataset hinzufügen|Datensatz hinzufügen/).length)
        .toBeGreaterThanOrEqual(2));
    } finally {
      window.removeEventListener('ft_navigate', listener as EventListener);
    }
  });
});
