// Der Dialog, der beim Import nach dem Plugin fragt.
//
// Ausloeser: ein importiertes YOLO wurde nicht erkannt und fiel erst im Labor
// auf ("wird noch nicht unterstuetzt"), wo es keinen Weg nach vorn gab. Der
// Dialog muss deshalb beides koennen: ein Plugin zuordnen und den Wunsch nach
// einem fehlenden Plugin aufnehmen.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));

const success = vi.fn();
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success, error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));
vi.mock('../../contexts/PageContext', () => ({
  usePageContext: () => ({ setCurrentPageContent: vi.fn() }),
}));

import { UnknownModelDialog } from '../ModelManager';

const MODEL = {
  id: 'local_abc123', name: 'mein-experiment', source: 'local' as const,
  source_path: '/tmp/models/local_abc123', size_bytes: 10, file_count: 1,
  created_at: '2026-09-16T08:00:00Z', model_type: 'pytorch',
};

describe('UnknownModelDialog', () => {
  beforeEach(() => { invokeMock.mockReset(); invokeMock.mockResolvedValue(undefined); success.mockReset(); });

  it('bietet die unterstuetzten Plugins an – ohne Canvas', () => {
    render(<UnknownModelDialog model={MODEL} onClose={vi.fn()} onAssigned={vi.fn()} />);
    expect(screen.getByText('YOLO Object Detection')).toBeTruthy();
    expect(screen.getByText('Seq2Seq (T5/BART)')).toBeTruthy();
    // Canvas-Netze entstehen im Synapse Builder, nicht beim Import.
    expect(screen.queryByText('Canvas Neural Net')).toBeNull();
  });

  it('speichert die Zuordnung erst nach der Auswahl', async () => {
    const onAssigned = vi.fn();
    render(<UnknownModelDialog model={MODEL} onClose={vi.fn()} onAssigned={onAssigned} />);

    const assign = screen.getByRole('button', { name: 'Zuordnen' }) as HTMLButtonElement;
    expect(assign.disabled).toBe(true);

    fireEvent.click(screen.getByText('YOLO Object Detection'));
    fireEvent.click(assign);

    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('set_model_plugin', {
      modelId: 'local_abc123', pluginId: 'yolo',
    }));
    await waitFor(() => expect(onAssigned).toHaveBeenCalled());
  });

  it('nimmt ueber "Keins davon passt" den Plugin-Wunsch auf', async () => {
    const onClose = vi.fn();
    render(<UnknownModelDialog model={MODEL} onClose={onClose} onAssigned={vi.fn()} />);

    fireEvent.click(screen.getByText('Keins davon passt'));
    // Ohne Plugin bleibt Dev Train / Dev Script – das muss dort stehen.
    expect(screen.getByText(/Dev Train/)).toBeTruthy();

    const send = screen.getByRole('button', { name: 'Wunsch speichern' }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Whisper fuer Spracherkennung' } });
    fireEvent.click(send);

    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('record_plugin_request', {
      modelName: 'mein-experiment', modelType: 'pytorch', note: 'Whisper fuer Spracherkennung',
    }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('laesst eine falsche Erkennung korrigieren und wieder zuruecknehmen', async () => {
    const onAssigned = vi.fn();
    const wrong = { ...MODEL, name: 'resnet-experiment', plugin_override: 'image-classification' };
    render(<UnknownModelDialog model={wrong} mode="change" onClose={vi.fn()} onAssigned={onAssigned} />);

    // Die aktuelle Zuordnung steht schon zur Wahl: "Zuordnen" ist sofort nutzbar.
    expect((screen.getByRole('button', { name: 'Zuordnen' }) as HTMLButtonElement).disabled).toBe(false);

    fireEvent.click(screen.getByText('Zuordnung entfernen'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('set_model_plugin', {
      modelId: 'local_abc123', pluginId: null,
    }));
    await waitFor(() => expect(onAssigned).toHaveBeenCalled());
  });

  it('bietet das Entfernen nur an, wenn wirklich von Hand zugeordnet wurde', () => {
    render(<UnknownModelDialog model={MODEL} mode="change" onClose={vi.fn()} onAssigned={vi.fn()} />);
    expect(screen.queryByText('Zuordnung entfernen')).toBeNull();
  });
});
