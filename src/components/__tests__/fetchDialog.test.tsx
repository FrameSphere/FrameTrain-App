// Adressen aus dem Netz holen.
//
// Zwei Dinge muessen stimmen: es gehen nur echte Adressen raus, und der
// Bericht sagt hinterher, was robots.txt untersagt hat — sonst glaubt man,
// alles sei geholt worden.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));

import FetchDialog, { startGrenzen } from '../studio/FetchDialog';

const PROJECT = {
  id: 'sp_img', name: 'ski', modality: 'image', task: 'bbox',
  target_format: 'yolo_bbox', classes: [], created_at: '', updated_at: '',
};

describe('FetchDialog', () => {
  beforeEach(() => invokeMock.mockReset());

  it('erkennt nur echte Adressen', async () => {
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);
    const feld = screen.getByPlaceholderText(/https/);

    fireEvent.change(feld, { target: {
      value: 'https://a.de/1.jpg\nkeine-adresse\nhttp://b.de/2.png\nftp://c.de/3.png',
    } });

    expect(await screen.findByText('2 gültige Adressen erkannt')).toBeInTheDocument();
  });

  it('schickt Adressen und Lizenz ans Backend und zeigt den Bericht', async () => {
    invokeMock.mockResolvedValue({
      fetched: 2, duplicates: 1, blocked: ['https://c.de/geheim.jpg'], failed: [], skipped_type: [],
      pages_visited: 0, too_large: [], too_small: 0, outside_allowlist: 0, limit_reached: false, cancelled: false,
    });
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/https/), {
      target: { value: 'https://a.de/1.jpg https://b.de/2.png' },
    });
    fireEvent.change(screen.getByPlaceholderText('CC-BY-4.0'), { target: { value: 'CC0' } });
    fireEvent.click(screen.getByRole('button', { name: /Holen/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_fetch_web', expect.objectContaining({
        projectId: 'sp_img',
        urls: ['https://a.de/1.jpg', 'https://b.de/2.png'],
        options: expect.objectContaining({ mode: 'urls', license: 'CC0', max_mb: 25, max_files: 500 }),
      }));
    });

    // Der Bericht muss das Untersagte benennen, nicht verschweigen.
    expect(await screen.findByText('Von robots.txt untersagt')).toBeInTheDocument();
    expect(screen.getByText(/geheim\.jpg/)).toBeInTheDocument();
  });

  it('holt nichts ohne Adresse', () => {
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);
    expect(screen.getByRole('button', { name: /Holen/ })).toBeDisabled();
  });

  it('Website-Modus schickt Tiefe, Grenzen und Allowlist mit', async () => {
    invokeMock.mockResolvedValue({
      fetched: 12, duplicates: 0, blocked: [], failed: [], skipped_type: [], pages_visited: 5,
      too_large: ['https://a.de/riesig.jpg'], too_small: 3, outside_allowlist: 7, limit_reached: true, cancelled: false,
    });
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Website' }));
    fireEvent.change(screen.getByPlaceholderText(/https/), { target: { value: 'https://a.de/galerie' } });
    fireEvent.click(screen.getByText('Grenzen'));
    fireEvent.change(screen.getByPlaceholderText('example.org, wikimedia.org'), { target: { value: 'a.de, cdn.a.de' } });
    fireEvent.click(screen.getByRole('button', { name: /Holen/ }));

    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_fetch_web', expect.objectContaining({
      options: expect.objectContaining({ mode: 'crawl', max_depth: 1, max_pages: 50, allowlist: ['a.de', 'cdn.a.de'], min_side: 64 }),
    })));
    expect(await screen.findByText('Seiten gelesen')).toBeInTheDocument();
    expect(screen.getByText('Außerhalb der Allowlist')).toBeInTheDocument();
    expect(screen.getByText(/Die Grenze ist erreicht/)).toBeInTheDocument();
    expect(screen.getByText(/riesig\.jpg/)).toBeInTheDocument();
  });

  it('Video bekommt Grenzen, die zu Videos passen', () => {
    expect(startGrenzen('video')).toEqual({ files: 100, mb: 4000, pages: 50 });
    render(<FetchDialog project={{ ...PROJECT, modality: 'video' }} onClose={vi.fn()} onDone={vi.fn()} />);
    // Im Adressen-Modus zaehlen Seiten nicht — sie stehen auch nicht da.
    expect(screen.getByText('bis 100 Dateien · je Datei bis 4 GB')).toBeInTheDocument();
  });

  it('eine Klasse fuer alles Geholte geht mit', async () => {
    invokeMock.mockResolvedValue({
      fetched: 1, duplicates: 0, blocked: [], failed: [], skipped_type: [], pages_visited: 1,
      too_large: [], too_small: 0, outside_allowlist: 0, limit_reached: false, cancelled: false,
    });
    render(<FetchDialog project={{ ...PROJECT, task: 'classify', classes: ['Katze', 'Hund'] }} onClose={vi.fn()} onDone={vi.fn()} />);
    fireEvent.change(screen.getByPlaceholderText(/https/), { target: { value: 'https://a.de/katzen' } });
    fireEvent.change(screen.getByDisplayValue(/Keine/), { target: { value: 'Katze' } });
    fireEvent.click(screen.getByRole('button', { name: /Holen/ }));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_fetch_web', expect.objectContaining({
      options: expect.objectContaining({ label: 'Katze' }),
    })));
  });

  // gnu.org liess die Verbindung nicht zu; der Bericht sagte nur
  // "Nicht erreichbar: 1" und man wusste nicht, woran es lag.
  it('nennt fuer jede nicht geladene Adresse den Grund', async () => {
    invokeMock.mockResolvedValue({
      fetched: 0, duplicates: 0, blocked: [], skipped_type: [], pages_visited: 0,
      failed: [
        { url: 'https://www.gnu.org/', reason: 'connect' },
        { url: 'https://a.de/zu', reason: 'http 403' },
        { url: 'https://a.de/kaputt', reason: 'http 502' },
      ],
      too_large: [], too_small: 0, outside_allowlist: 0, limit_reached: false, cancelled: false,
    });
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);
    fireEvent.change(screen.getByPlaceholderText(/https/), { target: { value: 'https://www.gnu.org/' } });
    fireEvent.click(screen.getByRole('button', { name: /Holen/ }));

    const box = await screen.findByTestId('fetch-failed');
    expect(box).toHaveTextContent('www.gnu.org');
    expect(box).toHaveTextContent('Server nicht erreichbar');
    expect(box).toHaveTextContent('Server verweigert den Zugriff (403)');
    expect(box).toHaveTextContent('Server antwortet mit 502');
  });

  it('bei Boxenprojekten gibt es keine Klasse fuer alles', () => {
    render(<FetchDialog project={{ ...PROJECT, classes: ['Lift'] }} onClose={vi.fn()} onDone={vi.fn()} />);
    expect(screen.queryByDisplayValue(/Keine/)).toBeNull();
  });
});
