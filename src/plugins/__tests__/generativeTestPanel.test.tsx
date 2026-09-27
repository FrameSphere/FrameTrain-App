// Test-Oberflaechen der generativen Plugins:
//  * Text-to-Image zeigt das erzeugte Bild (Pfad -> convertFileSrc), nicht den Pfad als Text.
//  * VLM hat ein zweites Feld fuer die Frage; sie geht als plugin_config.question
//    an die Engine, single_input bleibt der Bildpfad.
//  * Dataset-Lauf zeigt Zusatzkennzahlen (ROUGE-L).

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));

vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
  convertFileSrc: (p: string) => `asset://localhost/${encodeURIComponent(p)}`,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import TextToImageTestPlugin from '../text-to-image-lora/TestPlugin';
import VisionLanguageTestPlugin from '../vision-language/TestPlugin';
import type { DatasetInfo } from '../types';

type Cb = (e: { payload: unknown }) => void;
let listeners: Record<string, Cb[]> = {};
const emit = (event: string, payload: unknown) => listeners[event]?.forEach(cb => cb({ payload }));

const DS: DatasetInfo = { id: 'ds', name: 'Formen', model_id: 'm', status: 'split', file_count: 3, size_bytes: 10 };
const props = { modelPath: '/m', versionId: 'v1', modelId: 'm', modelName: 'M', versionName: 'V', datasets: [DS] };

beforeEach(() => {
  listeners = {};
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockImplementation((event: string, cb: Cb) => {
    (listeners[event] ??= []).push(cb);
    return Promise.resolve(() => { listeners[event] = listeners[event].filter(f => f !== cb); });
  });
});

describe('Text-to-Image-Test', () => {
  it('zeigt das erzeugte Bild aus dem Ergebnis-Pfad', async () => {
    mockInvoke.mockResolvedValue('single_1');
    render(<TextToImageTestPlugin {...props} />);
    fireEvent.change(screen.getAllByRole('textbox')[0], { target: { value: 'a sks icon' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
      singleInput: 'a sks icon', taskType: 'text_to_image_lora',
    })));
    await waitFor(() => expect(listeners['test-single-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-single-complete', {
      test_id: 'single_1',
      data: { predicted_output: '/out/a_sks_icon_42.png', inference_time: 2 },
    }));
    const img = await screen.findByAltText('a sks icon');
    expect(img.getAttribute('src')).toContain(encodeURIComponent('/out/a_sks_icon_42.png'));
  });
});

describe('VLM-Test', () => {
  it('schickt die Frage als plugin_config.question, den Bildpfad als single_input', async () => {
    mockInvoke.mockResolvedValue('single_2');
    render(<VisionLanguageTestPlugin {...props} />);
    const [pathField] = screen.getAllByRole('textbox');
    fireEvent.change(pathField, { target: { value: '/bilder/kreis.png' } });
    const question = screen.getByLabelText(/Frage|Question/);
    fireEvent.change(question, { target: { value: 'Welche Farbe hat die Form?' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
      singleInput: '/bilder/kreis.png',
      singleInputType: 'file',
      taskType: 'vision_language',
      pluginConfig: { question: 'Welche Farbe hat die Form?' },
    })));
  });

  it('ohne Frage bleibt plugin_config leer (Standardprompt aus dem Training)', async () => {
    mockInvoke.mockResolvedValue('single_3');
    render(<VisionLanguageTestPlugin {...props} />);
    fireEvent.change(screen.getAllByRole('textbox')[0], { target: { value: '/b.png' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input',
      expect.objectContaining({ pluginConfig: {} })));
  });

  it('Dataset-Lauf zeigt ROUGE-L neben dem exakten Treffer', async () => {
    mockInvoke.mockResolvedValue({ id: 'job' });
    render(<VisionLanguageTestPlugin {...props} />);
    const buttons = screen.getAllByRole('button');
    fireEvent.click(buttons[buttons.length - 1]);
    await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-complete', {
      data: { total_samples: 20, accuracy: 0.9, correct_predictions: 18, metrics: { exact_match: 0.9, rougeL: 0.95 } },
    }));
    // Zusatzkennzahlen stehen als Beschriftung + Wert (gemeinsame Darstellung mit NER/Embeddings).
    expect(await screen.findByText('RougeL')).toBeTruthy();
    expect(screen.getByText('0.950')).toBeTruthy();
  });
});
