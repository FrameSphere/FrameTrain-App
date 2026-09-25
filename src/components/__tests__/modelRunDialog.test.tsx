// Vorschlagen mit einem Modell.
//
// Im Test lief ein Lauf ueber 178 Bilder 28 Sekunden lang mit nichts als einem
// Spinner — der Fortschritt stand hinter dem Dialog. Und ob ein Modell taugt,
// wollte man an 20 Bildern sehen, nicht erst an allen.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
const handlers: Record<string, (ev: { payload: unknown }) => void> = {};
vi.mock('@tauri-apps/api/event', () => ({
  listen: (name: string, cb: (ev: { payload: unknown }) => void) => {
    handlers[name] = cb;
    return Promise.resolve(() => {});
  },
}));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));

import ModelRunDialog from '../studio/ModelRunDialog';

const PROJEKT = {
  id: 'sp_img', name: 'Ski', modality: 'image', task: 'bbox', target_format: 'yolo_bbox',
  classes: ['Sky'], created_at: '', updated_at: '',
};

describe('ModelRunDialog', () => {
  let fertig: (v: unknown) => void;
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models_with_version_tree') return [
        { id: 'm_yolo', name: 'yolo8n', versions: [{ id: 'v1', name: 'Original', version_number: 0 }] },
      ];
      if (cmd === 'list_models') return [
        { id: 'm_yolo', name: 'yolo8n', source_path: 'ultralytics/yolov8n', model_type: 'yolo' },
      ];
      if (cmd === 'studio_suggest') return new Promise(r => { fertig = r; });
      return null;
    });
  });

  it('laesst die Menge begrenzen und zeigt den Fortschritt im Dialog', async () => {
    render(<ModelRunDialog mode="suggest" project={PROJEKT} onClose={vi.fn()} onDone={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole('button', { name: /Starten|Start/ })).not.toBeDisabled());

    fireEvent.click(screen.getByRole('button', { name: 'Die nächsten 20' }));
    fireEvent.click(screen.getByRole('button', { name: /Starten|Start/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_suggest', expect.objectContaining({ limit: 20 }));
    });
    act(() => {
      handlers['studio-suggest-progress']({ payload: { project_id: 'sp_img', current: 2, total: 20 } });
    });
    expect(await screen.findByText('3 von 20')).toBeInTheDocument();

    await act(async () => {
      fertig({ processed: 20, with_boxes: 5, boxes_total: 7, left_confirmed: 0, without_boxes: 15,
        unmapped_classes: [], classes_added: [], model_classes: [], failed: 0 });
    });
  });
});
