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

import FetchDialog from '../studio/FetchDialog';

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
    });
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/https/), {
      target: { value: 'https://a.de/1.jpg https://b.de/2.png' },
    });
    fireEvent.change(screen.getByPlaceholderText('CC-BY-4.0'), { target: { value: 'CC0' } });
    fireEvent.click(screen.getByRole('button', { name: /Holen/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_fetch_urls', {
        projectId: 'sp_img',
        urls: ['https://a.de/1.jpg', 'https://b.de/2.png'],
        license: 'CC0',
      });
    });

    // Der Bericht muss das Untersagte benennen, nicht verschweigen.
    expect(await screen.findByText('Von robots.txt untersagt')).toBeInTheDocument();
    expect(screen.getByText(/geheim\.jpg/)).toBeInTheDocument();
  });

  it('holt nichts ohne Adresse', () => {
    render(<FetchDialog project={PROJECT} onClose={vi.fn()} onDone={vi.fn()} />);
    expect(screen.getByRole('button', { name: /Holen/ })).toBeDisabled();
  });
});
