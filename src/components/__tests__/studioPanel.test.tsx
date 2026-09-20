// Projektliste und die Weiche zur passenden Werkbank.
//
// Die Weiche ist der Punkt, an dem ein stiller Fehler entsteht: fehlt sie,
// oeffnet ein Textprojekt den Bild-Editor und meldet "Noch keine Bilder".
// Nichts schlaegt fehl, nichts steht im Log — es ist einfach die falsche Seite.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));

import StudioPanel from '../studio/StudioPanel';

const PROJEKTE = [
  { id: 'sp_img', name: 'Ski-Bilder', modality: 'image', task: 'bbox',
    target_format: 'yolo_bbox', classes: ['Ski'], created_at: '', updated_at: '2026-09-20T10:00:00Z',
    sample_count: 0, confirmed_count: 0 },
  { id: 'sp_txt', name: 'Rückmeldungen', modality: 'text', task: 'classification',
    target_format: 'flat_file', classes: ['lob'], created_at: '', updated_at: '2026-09-20T09:00:00Z',
    sample_count: 0, confirmed_count: 0 },
  { id: 'sp_pair', name: 'Umformulieren', modality: 'text', task: 'pairs',
    target_format: 'flat_file', classes: [], created_at: '', updated_at: '2026-09-20T08:00:00Z',
    sample_count: 0, confirmed_count: 0 },
];

describe('StudioPanel', () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_projects') return PROJEKTE;
      if (cmd === 'studio_list_samples') return { total: 0, items: [] };
      if (cmd === 'studio_stats') return { total: 0, new: 0, suggested: 0, confirmed: 0,
        skipped: 0, boxes_total: 0, per_class: [], empty_confirmed: 0, doubts: 0 };
      return null;
    });
  });

  it('oeffnet ein Bildprojekt in der Bild-Werkbank', async () => {
    render(<StudioPanel />);
    fireEvent.click(await screen.findByText('Ski-Bilder'));
    expect(await screen.findByText('Noch keine Bilder')).toBeInTheDocument();
  });

  it('oeffnet ein Textprojekt in der Text-Werkbank', async () => {
    render(<StudioPanel />);
    fireEvent.click(await screen.findByText('Rückmeldungen'));
    expect(await screen.findByText('Noch keine Texte')).toBeInTheDocument();
    expect(screen.queryByText('Noch keine Bilder')).not.toBeInTheDocument();
  });

  it('oeffnet ein Paar-Projekt ebenfalls in der Text-Werkbank', async () => {
    render(<StudioPanel />);
    fireEvent.click(await screen.findByText('Umformulieren'));
    expect(await screen.findByText('Noch keine Texte')).toBeInTheDocument();
  });
});
