// UI-Komponenten Tests – FrameTrain Plugin-System
// Ausführen: npx vitest run src/plugins/__tests__/ui.test.tsx --config vitest.config.ts

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// ── Tauri Mocks ────────────────────────────────────────────────────────────
// vi.mock() wird von Vitest ans Datei-Top gehoisted – deshalb müssen die
// Mock-Funktionen mit vi.hoisted() deklariert werden, sonst sind sie beim
// Ausführen der Factory noch undefined (ReferenceError).

const { mockInvoke, mockListen } = vi.hoisted(() => ({
  mockInvoke: vi.fn(),
  mockListen: vi.fn(),
}));

vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

// Hilfsfunktion: simuliert Tauri-Events die an registrierte Listener geschickt werden
type ListenerMap = Record<string, ((e: { payload: unknown }) => void)[]>;
let listeners: ListenerMap = {};

function setupListenMock() {
  listeners = {};
  mockListen.mockImplementation((event: string, cb: (e: { payload: unknown }) => void) => {
    if (!listeners[event]) listeners[event] = [];
    listeners[event].push(cb);
    return Promise.resolve(() => {
      listeners[event] = listeners[event].filter(fn => fn !== cb);
    });
  });
}

function emitEvent(event: string, payload: unknown) {
  listeners[event]?.forEach(cb => cb({ payload }));
}

// ── Komponenten ────────────────────────────────────────────────────────────

import XLMRobertaTestPlugin from '../xlm-roberta/TestPlugin';
import HFEncoderTestPlugin from '../hf-encoder/TestPlugin';
import type { DatasetInfo } from '../types';

// ── Test-Fixtures ──────────────────────────────────────────────────────────

const SPLIT_DATASET: DatasetInfo = {
  id: 'ds-1',
  name: 'Keyword-Dataset',
  model_id: 'model-1',
  status: 'split',
  file_count: 3,
  size_bytes: 1024 * 512,
};

const UNUSED_DATASET: DatasetInfo = {
  id: 'ds-2',
  name: 'Rohes Dataset',
  model_id: 'model-1',
  status: 'unused',
  file_count: 5,
  size_bytes: 1024 * 256,
};

const BASE_TEST_PROPS = {
  modelPath: '/models/xlm-roberta-base',
  versionId: 'v-abc123',
  modelId: 'model-1',
  modelName: 'XLM-RoBERTa Base',
  versionName: 'v1.0',
  datasets: [SPLIT_DATASET],
};

// ─────────────────────────────────────────────────────────────────────────────
// XLMRobertaTrainPlugin
// ─────────────────────────────────────────────────────────────────────────────

describe('XLMRobertaTestPlugin – Text-Tab (Standard)', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    setupListenMock();
  });

  it('rendert ohne Absturz', () => {
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    expect(screen.getByText(/Text-Eingabe/i)).toBeInTheDocument();
  });

  it('"Text-Eingabe"-Tab ist initial aktiv', () => {
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    expect(screen.getByPlaceholderText(/Text zum Testen/i)).toBeInTheDocument();
  });

  it('Testen-Button disabled wenn Textarea leer', () => {
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    const btn = screen.getByRole('button', { name: /▶ Testen/i });
    expect(btn).toBeDisabled();
  });

  it('Testen-Button aktiv wenn Text eingegeben', async () => {
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Hallo Welt');
    const btn = screen.getByRole('button', { name: /▶ Testen/i });
    expect(btn).not.toBeDisabled();
  });

  it('invoke("test_single_input") mit korrekten Parametern', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Hallo Welt');
    fireEvent.click(screen.getByRole('button', { name: /▶ Testen/i }));

    await waitFor(() => {
      expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
        versionId: 'v-abc123',
        singleInput: 'Hallo Welt',
        singleInputType: 'text',
        taskType: 'seq_classification',
      }));
    });
  });

  it('test-single-complete Event → Ergebnis-Karte erscheint', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Hallo');
    fireEvent.click(screen.getByRole('button', { name: /▶ Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-single-complete', {
        test_id: 'test-id-1',
        data: {
          predicted_output: 'Begrüßung',
          confidence: 0.97,
          top_predictions: [{ label: 'Begrüßung', score: 0.97 }],
          inference_time: 0.042,
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Begrüßung')).toBeInTheDocument();
    });
  });

  it('test-single-complete mit falscher test_id → wird ignoriert', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Hallo');
    fireEvent.click(screen.getByRole('button', { name: /▶ Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-single-complete', {
        test_id: 'ANDERE-ID',
        data: { predicted_output: 'SollteNichtErscheinen', confidence: 0.5, inference_time: 0.01 },
      });
    });

    await new Promise(r => setTimeout(r, 50));
    expect(screen.queryByText('SollteNichtErscheinen')).not.toBeInTheDocument();
  });

  it('test-error Event → Fehlermeldung sichtbar', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Hallo');
    fireEvent.click(screen.getByRole('button', { name: /▶ Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-error', {
        test_id: 'test-id-1',
        data: { error: 'Modell nicht geladen' },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/Modell nicht geladen/i)).toBeInTheDocument();
    });
  });

  it('confidence wird in Prozent angezeigt (×100)', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text zum Testen/i), 'Test');
    fireEvent.click(screen.getByRole('button', { name: /▶ Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-single-complete', {
        test_id: 'test-id-1',
        data: { predicted_output: 'Label-A', confidence: 0.95, inference_time: 0.03 },
      });
    });

    await waitFor(() => {
      expect(screen.getByText('95.0%')).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// XLMRobertaTestPlugin – Dataset-Tab
// ─────────────────────────────────────────────────────────────────────────────

describe('XLMRobertaTestPlugin – Dataset-Tab', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    setupListenMock();
  });

  it('Tab-Wechsel zu "Dataset" funktioniert', async () => {
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    expect(screen.getByRole('button', { name: /▶ Dataset testen/i })).toBeInTheDocument();
  });

  it('split-Dataset wird auto-selektiert (Priorität vor unused)', async () => {
    render(<XLMRobertaTestPlugin
      {...BASE_TEST_PROPS}
      datasets={[UNUSED_DATASET, SPLIT_DATASET]}
    />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));

    const select = screen.getByRole('combobox') as HTMLSelectElement;
    expect(select.value).toBe(SPLIT_DATASET.id);
  });

  it('unused-Dataset → "Dataset testen"-Button disabled', async () => {
    render(<XLMRobertaTestPlugin
      {...BASE_TEST_PROPS}
      datasets={[UNUSED_DATASET]}
    />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    expect(screen.getByRole('button', { name: /▶ Dataset testen/i })).toBeDisabled();
  });

  it('unused-Dataset → Warnhinweis sichtbar', async () => {
    render(<XLMRobertaTestPlugin
      {...BASE_TEST_PROPS}
      datasets={[UNUSED_DATASET]}
    />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    expect(screen.getByText(/noch keinen Split/i)).toBeInTheDocument();
  });

  it('invoke("start_test") mit taskType="seq_classification"', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-1' });
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Dataset testen/i }));

    await waitFor(() => {
      expect(mockInvoke).toHaveBeenCalledWith('start_test', expect.objectContaining({
        taskType: 'seq_classification',
        versionId: 'v-abc123',
        datasetId: SPLIT_DATASET.id,
      }));
    });
  });

  it('test-progress Event → Fortschrittsanzeige erscheint', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-1' });
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Dataset testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-progress', {
        test_id: 'job-1',
        data: { current_sample: 50, total_samples: 100, progress_percent: 50, samples_per_second: 12.5 },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/50\/100/)).toBeInTheDocument();
    });
  });

  it('"Test stoppen"-Button ruft invoke("stop_test") auf', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-1' });
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Dataset testen/i }));

    await waitFor(() => expect(screen.getByText(/Test stoppen/i)).toBeInTheDocument());

    mockInvoke.mockResolvedValue(undefined);
    fireEvent.click(screen.getByText(/Test stoppen/i));

    await waitFor(() => {
      expect(mockInvoke).toHaveBeenCalledWith('stop_test');
    });
  });

  it('test-complete Event → Accuracy-Karte erscheint', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-1' });
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Dataset testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-complete', {
        test_id: 'job-1',
        data: {
          accuracy: 0.94,
          correct_predictions: 94,
          total_samples: 100,
          average_inference_time: 0.021,
          samples_per_second: 47.6,
          predictions: [],
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText('94.0%')).toBeInTheDocument();
    });
  });

  it('accuracy > 90% → kein text-red-400', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-1' });
    render(<XLMRobertaTestPlugin {...BASE_TEST_PROPS} />);
    fireEvent.click(screen.getByRole('button', { name: /^Dataset$/i }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Dataset testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-complete', {
        test_id: 'job-1',
        data: { accuracy: 0.95, total_samples: 100, average_inference_time: 0.02, predictions: [] },
      });
    });

    await waitFor(() => {
      const el = screen.getByText('95.0%');
      expect(el).not.toHaveClass('text-red-400');
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// HFEncoderTrainPlugin
// ─────────────────────────────────────────────────────────────────────────────

describe('HFEncoderTestPlugin – Single Input', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    setupListenMock();
  });

  it('rendert ohne Absturz', () => {
    render(<HFEncoderTestPlugin {...BASE_TEST_PROPS} />);
    expect(screen.getByPlaceholderText(/Text eingeben/i)).toBeInTheDocument();
  });

  it('Testen-Button disabled wenn leer', () => {
    render(<HFEncoderTestPlugin {...BASE_TEST_PROPS} />);
    expect(screen.getByRole('button', { name: /Testen/i })).toBeDisabled();
  });

  it('Reset-Button löscht Input und Ergebnis', async () => {
    mockInvoke.mockResolvedValue('test-id-1');
    render(<HFEncoderTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text eingeben/i), 'Test-Text');
    fireEvent.click(screen.getByRole('button', { name: /Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-single-complete', {
        test_id: 'test-id-1',
        data: { predicted_output: 'Label-X', confidence: 0.88, inference_time: 0.02 },
      });
    });

    await waitFor(() => expect(screen.getByText('Label-X')).toBeInTheDocument());

    const resetBtns = screen.getAllByRole('button', { name: /Reset/i });
    fireEvent.click(resetBtns[0]);

    await waitFor(() => {
      expect(screen.queryByText('Label-X')).not.toBeInTheDocument();
      expect(screen.getByPlaceholderText(/Text eingeben/i)).toHaveValue('');
    });
  });

  it('Ergebnis zeigt predicted_output an', async () => {
    mockInvoke.mockResolvedValue('test-id-2');
    render(<HFEncoderTestPlugin {...BASE_TEST_PROPS} />);

    await userEvent.type(screen.getByPlaceholderText(/Text eingeben/i), 'Anfrage');
    fireEvent.click(screen.getByRole('button', { name: /Testen/i }));

    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-single-complete', {
        test_id: 'test-id-2',
        data: { predicted_output: 'KlasseB', confidence: 0.72, inference_time: 0.015 },
      });
    });

    await waitFor(() => {
      expect(screen.getByText('KlasseB')).toBeInTheDocument();
    });
  });
});

describe('HFEncoderTestPlugin – Dataset-Test Reset', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    setupListenMock();
  });

  it('Reset-Button setzt Dataset-Ergebnis zurück', async () => {
    mockInvoke.mockResolvedValue({ id: 'job-2' });
    render(<HFEncoderTestPlugin {...BASE_TEST_PROPS} />);

    fireEvent.click(screen.getByRole('button', { name: /^Start$/i }));
    await waitFor(() => expect(mockListen).toHaveBeenCalled());

    act(() => {
      emitEvent('test-complete', {
        test_id: 'job-2',
        data: { total_samples: 50, accuracy: 0.88, average_inference_time: 0.02, predictions: [] },
      });
    });

    // Die Komponente rendert "88.00" und "%" als getrennte Text-Nodes –
    // daher Regex-Matcher statt exaktem String.
    await waitFor(() => expect(screen.getByText(/88\.00/)).toBeInTheDocument());

    const resetBtns = screen.getAllByRole('button', { name: /Reset/i });
    fireEvent.click(resetBtns[resetBtns.length - 1]);

    await waitFor(() => {
      expect(screen.queryByText(/88\.00/)).not.toBeInTheDocument();
    });
  });
});
