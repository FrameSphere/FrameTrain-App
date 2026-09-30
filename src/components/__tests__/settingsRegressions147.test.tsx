// Befunde aus 1.4.7 (Durchgang fuer die Webseiten-Bilder).
//
// - Beim Oeffnen der Einstellungen kam der macOS-Passwortdialog: der erste Tab
//   (Konto) las den HuggingFace-Token aus dem Schluesselbund, nur um ihn ins
//   Feld zu schreiben. Jetzt wird nur gefragt, ob einer da ist.
// - Nach dem Wechsel auf Englisch stand die Bestaetigung noch auf Deutsch.
// - Ein Klassenprojekt ohne Klassen zeigte die Option "Klasse fuer alles
//   Geholte" nicht und sagte nicht, warum.
// - Der Export-Bericht wurde bei vielen Klassen hoeher als das Fenster.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: () => Promise.resolve('1.4.8') }));
vi.mock('@tauri-apps/plugin-shell', () => ({ open: vi.fn() }));
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({
    currentTheme: { id: 'purple', colors: { gradient: 'from-purple-600 to-pink-600', primary: '#a855f7' } },
    setTheme: vi.fn(), themes: [],
  }),
}));
vi.mock('../../contexts/PageContext', () => ({
  usePageContext: () => ({ setCurrentPageContent: vi.fn() }),
}));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));
vi.mock('../../contexts/AISettingsContext', async (orig) => ({
  ...(await orig<typeof import('../../contexts/AISettingsContext')>()),
  useAISettings: () => ({
    draft: { provider: 'ollama', apiKey: '', model: '', tokenBudget: 'normal' },
    updateDraft: vi.fn(), isDirty: false, saveSettings: vi.fn(), discardDraft: vi.fn(),
    keyLoading: false, keychainAvailable: true, providersWithKey: [], ensureProviderKeysLoaded: vi.fn(),
  }),
}));

import Settings from '../Settings';
import { LanguageProvider } from '../../contexts/LanguageContext';
import FetchDialog from '../studio/FetchDialog';
import ExportReportView from '../studio/ExportReportView';

const USER = { id: 'u1', email: 'a@b.de', name: 'A' };

describe('Einstellungen', { timeout: 15000 }, () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'secret_exists') return true;
      if (cmd === 'secret_get') return 'hf_geheim';
      return null;
    });
  });
  afterEach(() => localStorage.removeItem('ft_language'));

  it('liest beim Oeffnen keinen Token aus dem Schluesselbund', async () => {
    render(<LanguageProvider><Settings userData={USER as never} onLogout={vi.fn()} /></LanguageProvider>);
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('secret_exists', { key: 'ft_hf_token' }));
    expect(invokeMock.mock.calls.filter(([cmd]) => cmd === 'secret_get')).toEqual([]);
    // Ein gespeicherter Token wird angezeigt, ohne ihn zu kennen.
    expect(await screen.findByPlaceholderText('Gespeichert – zum Ersetzen neuen Token eingeben')).toBeInTheDocument();
  });

  it('bestaetigt den Sprachwechsel in der neuen Sprache', async () => {
    render(<LanguageProvider><Settings userData={USER as never} onLogout={vi.fn()} /></LanguageProvider>);
    fireEvent.click(screen.getByRole('button', { name: /^Sprache/ }));
    fireEvent.click(await screen.findByRole('button', { name: /English/ }));
    expect(await screen.findByText('Language changed — English')).toBeInTheDocument();
    expect(screen.queryByText(/Sprache geändert/)).toBeNull();
  });
});

describe('Aus dem Netz holen', () => {
  const PROJEKT = {
    id: 'sp', name: 'Tiere', modality: 'image', task: 'classify', target_format: 'folder_class',
    classes: [] as string[], created_at: '', updated_at: '',
  };

  it('sagt bei einem Klassenprojekt ohne Klassen, warum die Klassenwahl fehlt', () => {
    render(<FetchDialog project={PROJEKT} onClose={vi.fn()} onDone={vi.fn()} />);
    expect(screen.getByTestId('label-needs-classes')).toHaveTextContent('Leg zuerst Klassen an');
  });

  it('bei Boxen gibt es weder Klassenwahl noch den Hinweis', () => {
    render(<FetchDialog project={{ ...PROJEKT, task: 'bbox', classes: ['Lift'] }} onClose={vi.fn()} onDone={vi.fn()} />);
    expect(screen.queryByTestId('label-needs-classes')).toBeNull();
    expect(screen.queryByDisplayValue(/Keine/)).toBeNull();
  });
});

describe('Export-Bericht', () => {
  it('scrollt in sich, statt ueber das Fenster hinauszuwachsen', () => {
    const report = {
      total: 150, per_class: Array.from({ length: 15 }, (_, i) => [`k${i}`, 10] as [string, number]),
      per_split: [['train', 120], ['val', 30]] as [string, number][], group_leaks: [], groups: 20,
      near_checked: true, near_duplicates: 0, near_duplicates_across: 0, warnings: [], hints: [],
    };
    render(<ExportReportView report={report as never} modality="image" />);
    expect(screen.getByTestId('export-report').className).toMatch(/max-h-\[55vh\].*overflow-y-auto/);
  });
});
