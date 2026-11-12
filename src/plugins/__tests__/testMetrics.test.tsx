// Neue Test-Kennzahlen in der Oberflaeche und dort, wo Tests ausgewertet werden:
//  * Konfusionsmatrix + Werte je Klasse in der Test-Ansicht (begrenzt auf die
//    am haeufigsten verwechselten Klassen)
//  * Macro-F1, tolerante Uebereinstimmung, CLIP-Score richtig formatiert
//  * Hinweis ohne CLIP-Modell uebersetzt statt als Rohtext
//  * Benchmark-Vergleich nimmt clip_score statt einer beliebigen Zahl
//  * MCP tests_results gibt verschachtelte Kennzahlen unveraendert weiter

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
  convertFileSrc: (p: string) => `asset://localhost/${encodeURIComponent(p)}`,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: async () => '9.9.9' }));

import { limitConfusion, readConfusion } from '../ConfusionMatrixView';
import { formatTestMetric } from '../GenericTestPanel';
import HFImageTestPlugin from '../hf-image-classification/TestPlugin';
import TextToImageTestPlugin from '../text-to-image-lora/TestPlugin';
import ImageClassificationTestPlugin from '../image-classification/TestPlugin';
import { testMetric, type PipelineTest } from '../../components/homePipeline';
import { registry } from '../../mcp/registry';
import type { McpCallContext } from '../../mcp/types';
import type { DatasetInfo } from '../types';

type Cb = (e: { payload: unknown }) => void;
let listeners: Record<string, Cb[]> = {};
const emit = (event: string, payload: unknown) => listeners[event]?.forEach(cb => cb({ payload }));

const DS: DatasetInfo = { id: 'ds', name: 'Tiere', model_id: 'm', status: 'split', file_count: 6, size_bytes: 10 };
const props = { modelPath: '/m', versionId: 'v1', modelId: 'm', modelName: 'M', versionName: 'V', datasets: [DS] };

// So meldet die Test-Engine einen Klassifikationstest (gleiche Zahlen wie
// test_shared_classify.py: 6 Beispiele, 3 Klassen).
const CLASSIFY_METRICS = {
  accuracy: 0.6667, macro_f1: 0.6556, macro_precision: 0.7222, macro_recall: 0.6667, weighted_f1: 0.6556,
  per_class_f1: { hund: 0.6667, katze: 0.8, vogel: 0.5 },
  per_class: {
    hund: { precision: 1, recall: 0.5, f1: 0.6667, support: 2 },
    katze: { precision: 0.6667, recall: 1, f1: 0.8, support: 2 },
    vogel: { precision: 0.5, recall: 0.5, f1: 0.5, support: 2 },
  },
  confusion_matrix: { labels: ['hund', 'katze', 'vogel'], matrix: [[1, 0, 1], [0, 2, 0], [0, 1, 1]], total_classes: 3, truncated: false },
  top_confusions: [{ expected: 'hund', predicted: 'vogel', count: 1 }],
};

beforeEach(() => {
  listeners = {};
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockImplementation((event: string, cb: Cb) => {
    (listeners[event] ??= []).push(cb);
    return Promise.resolve(() => { listeners[event] = listeners[event].filter(f => f !== cb); });
  });
});

async function runDataset(button: RegExp, payload: unknown) {
  mockInvoke.mockResolvedValue({ id: 'job1' });
  fireEvent.click(screen.getByRole('button', { name: button }));
  await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(0));
  act(() => emit('test-complete', payload));
}

describe('Konfusionsmatrix begrenzen', () => {
  it('behaelt die am haeufigsten verwechselten Klassen in Originalreihenfolge', () => {
    const labels = ['a', 'b', 'c', 'd', 'e'];
    const matrix = [
      [5, 0, 0, 0, 0],
      [0, 5, 0, 4, 0],
      [0, 0, 9, 0, 0],
      [1, 0, 0, 5, 0],
      [0, 0, 0, 0, 5],
    ];
    const cut = limitConfusion({ labels, matrix }, 3);
    // b<->d: 5 Fehler, a: 1 Fehler; c und e ohne Verwechslung
    expect(cut.labels).toEqual(['a', 'b', 'd']);
    expect(cut.matrix).toEqual([[5, 0, 0], [0, 5, 4], [1, 0, 5]]);
    expect(cut.hidden).toBe(2);
    expect(cut.truncated).toBe(true);
  });

  it('zaehlt in der Engine schon gekuerzte Klassen als verborgen', () => {
    const cut = limitConfusion({ labels: ['a', 'b'], matrix: [[1, 0], [0, 1]], total_classes: 30, truncated: true }, 10);
    expect(cut.hidden).toBe(28);
  });

  it('ignoriert kaputte Matrizen', () => {
    expect(readConfusion({ confusion_matrix: { labels: ['a'], matrix: [] } })).toBeNull();
    expect(readConfusion({})).toBeNull();
  });
});

describe('Formatierung', () => {
  it('zeigt Anteile in Prozent, CLIP-Score auf 0..100', () => {
    expect(formatTestMetric('macro_f1', 0.6556)).toBe('65.6 %');
    expect(formatTestMetric('tolerant_match', 2 / 3)).toBe('66.7 %');
    expect(formatTestMetric('cer', 0.25)).toBe('25.0 %');
    expect(formatTestMetric('clip_score', 31.234)).toBe('31.2');
    expect(formatTestMetric('clip_score_delta', -1.06)).toBe('-1.1');
  });
});

describe('Test-Ansicht Klassifikation', () => {
  it('zeigt Macro-F1, Konfusionsmatrix und Werte je Klasse (HF-Bild)', async () => {
    render(<HFImageTestPlugin {...props} />);
    await runDataset(/Test starten|Start test/, {
      test_id: 'job1', data: { total_samples: 6, accuracy: 0.6667, correct_predictions: 4, metrics: CLASSIFY_METRICS },
    });
    const label = await screen.findByText('Macro-F1');
    expect(label.parentElement?.textContent).toContain('65.6 %');
    expect(screen.queryByText('Accuracy')).toBeNull(); // nicht doppelt neben der Trefferzeile
    const view = screen.getByTestId('confusion-view');
    const matrix = within(view).getAllByRole('table')[0];
    expect(within(matrix).getByTitle('hund → vogel: 1').textContent).toBe('1');
    expect(within(matrix).getByTitle('katze → katze: 2').textContent).toBe('2');
    // Je Klasse, schlechteste zuerst
    const perClass = within(view).getAllByRole('table')[1];
    const rows = within(perClass).getAllByRole('row').slice(1);
    expect(rows[0].textContent).toContain('vogel');
    expect(rows[2].textContent).toContain('katze');
  });

  it('torchvision-Ansicht zeigt dieselben Kennzahlen', async () => {
    render(<ImageClassificationTestPlugin {...props} />);
    await runDataset(/Genauigkeit messen|Measure accuracy|Measure/i, {
      test_id: 'job1', data: { total_samples: 6, accuracy: 0.6667, metrics: CLASSIFY_METRICS },
    });
    expect(await screen.findByText('65.6 %')).toBeTruthy();
    expect(screen.getByTestId('confusion-view')).toBeTruthy();
    expect(mockInvoke).toHaveBeenCalledWith('start_test', expect.objectContaining({ taskType: 'image_classification' }));
  });
});

describe('Test-Ansicht Text-zu-Bild', () => {
  it('zeigt den CLIP-Score und keine Matrix', async () => {
    render(<TextToImageTestPlugin {...props} />);
    await runDataset(/Test starten|Start test/, {
      test_id: 'job1', data: { total_samples: 2, accuracy: null, metrics: { clip_score: 28.5, clip_score_base: 26.1, clip_score_delta: 2.4 }, images: [] },
    });
    expect(await screen.findByText('28.5')).toBeTruthy();
    expect(screen.getByText('2.4')).toBeTruthy();
    expect(screen.queryByTestId('confusion-view')).toBeNull();
  });

  it('nennt ohne CLIP-Modell den Grund in der UI-Sprache', async () => {
    render(<TextToImageTestPlugin {...props} />);
    await runDataset(/Test starten|Start test/, {
      test_id: 'job1', data: { total_samples: 2, accuracy: null, metrics: {
        metrics_note: 'CLIP-Score nicht berechnet: ...', metrics_note_codes: ['clip_unavailable'], clip_model: 'openai/clip-vit-base-patch32',
      } },
    });
    expect(await screen.findByText(/openai\/clip-vit-base-patch32/)).toBeTruthy();
    expect(screen.queryByText(/CLIP-Score nicht berechnet: \.\.\./)).toBeNull();
  });
});

describe('Auswertung ausserhalb der Test-Ansicht', () => {
  const test_ = (results: PipelineTest['results']): PipelineTest => ({
    id: 't', model_name: 'm', status: 'completed', created_at: '2026-11-11T10:00:00Z', results,
  });

  it('Benchmark nimmt clip_score vor Bildanzahl und Schrittzahl', () => {
    expect(testMetric(test_({ accuracy: null, metrics: { generated_images: 4, num_inference_steps: 25, clip_score: 27.9 } })))
      .toEqual({ label: 'clip_score', value: 27.9, fraction: null });
  });

  it('Benchmark rankt Klassifikation weiter ueber Accuracy', () => {
    expect(testMetric(test_({ accuracy: 0.6667, metrics: CLASSIFY_METRICS }))?.label).toBe('accuracy');
  });

  it('MCP tests_results gibt Konfusionsmatrix und Werte je Klasse weiter', async () => {
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === 'get_test_results_for_version') {
        return [{ total_samples: 6, accuracy: 0.6667, average_inference_time: 0.01, metrics: CLASSIFY_METRICS, predictions: [] }];
      }
      throw new Error(`unexpected ${cmd}`);
    });
    const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'local', userId: 'u1' };
    const r = await registry.get('tests_results')!.run({ version_id: 'v1' }, ctx);
    const items = (r.structuredContent as { items?: unknown[] }).items ?? (r.structuredContent as unknown[]);
    const first = (Array.isArray(items) ? items[0] : null) as { metrics: typeof CLASSIFY_METRICS; accuracy: number };
    expect(first.accuracy).toBe(0.6667);
    expect(first.metrics.macro_f1).toBe(0.6556);
    expect(first.metrics.confusion_matrix.matrix).toEqual([[1, 0, 1], [0, 2, 0], [0, 1, 1]]);
    expect(first.metrics.per_class.katze.support).toBe(2);
  });
});
