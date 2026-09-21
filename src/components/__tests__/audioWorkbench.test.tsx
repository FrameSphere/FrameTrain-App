// Die Audio-Werkbank des Dataset Studio.
//
// Zwei Dinge sind hier eigen: der Weg hinein fuehrt ueber das Mikrofon, und
// wenn macOS das verweigert, muss die App sagen wo man es erlaubt — sonst
// sucht man den Fehler in der Anwendung.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
const error = vi.fn();
const warning = vi.fn();
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error, warning, info: vi.fn() }),
}));

import AudioWorkbench, { audioExtForMime } from '../studio/AudioWorkbench';

const AUFNAHMEN = [
  { id: 's_1', media: 'ab/a.wav', mime: 'audio/wav', status: 'new' as const,
    ann: { boxes: [] }, src: { kind: 'record', origin: 'Aufnahme', at: '' },
    meta: { w: 0, h: 0 }, abs_path: '/media/ab/a.wav' },
  { id: 's_2', media: 'cd/b.wav', mime: 'audio/wav', status: 'new' as const,
    ann: { boxes: [] }, src: { kind: 'record', origin: 'Aufnahme', at: '' },
    meta: { w: 0, h: 0 }, abs_path: '/media/cd/b.wav' },
];

const STATS = { total: 2, new: 2, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 0, per_class: [0], empty_confirmed: 0, doubts: 0 };

function projekt(task: 'classification' | 'transcript') {
  return {
    id: 'sp_audio', name: 'Durchsagen', modality: 'audio', task,
    target_format: task === 'transcript' ? 'audio_transcript' : 'folder_class',
    classes: task === 'transcript' ? [] : ['ansage', 'stoerung'],
    created_at: '', updated_at: '',
  };
}

const props = (task: 'classification' | 'transcript' = 'classification') => ({
  project: projekt(task), onBack: vi.fn(), onProjectChanged: vi.fn(),
});

describe('AudioWorkbench', () => {
  beforeEach(() => {
    invokeMock.mockReset();
    error.mockReset();
    warning.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: AUFNAHMEN.length, items: AUFNAHMEN };
      if (cmd === 'studio_stats') return STATS;
      if (cmd === 'list_models') return [];
      return null;
    });
  });

  it('weist per Zahl eine Klasse zu und springt weiter', async () => {
    render(<AudioWorkbench {...props()} />);
    await screen.findByText('1 / 2');

    fireEvent.keyDown(window, { key: '1' });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'confirmed', label: 'ansage', boxes: [],
      }));
    });
    await screen.findByText('2 / 2');
  });

  it('bestaetigt ein Transkript nicht ohne Wortlaut', async () => {
    render(<AudioWorkbench {...props('transcript')} />);
    const feld = await screen.findByPlaceholderText('Wortlaut der Aufnahme…');

    fireEvent.keyDown(feld, { key: 'Enter', metaKey: true });

    await waitFor(() => expect(warning).toHaveBeenCalled());
    expect(invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation')).toBeUndefined();
  });

  it('speichert den Wortlaut mit Cmd+Enter', async () => {
    render(<AudioWorkbench {...props('transcript')} />);
    const feld = await screen.findByPlaceholderText('Wortlaut der Aufnahme…');

    fireEvent.change(feld, { target: { value: 'Der Lift fährt gleich weiter' } });
    fireEvent.keyDown(feld, { key: 'Enter', metaKey: true });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'confirmed', target: 'Der Lift fährt gleich weiter',
      }));
    });
  });

  it('legt die Aufnahme unter der Endung ab, die der Recorder liefert', async () => {
    // WKWebView nimmt MP4 auf, egal was man sich wuenscht. Stand "webm" fest
    // im Aufruf, lag ein MP4 als .webm im Projekt: der Abspieler blieb stumm
    // und der Dataset-Import erkannte den Export nicht als Audio.
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: () => Promise.resolve({ getTracks: () => [] }) },
    });
    class FakeRecorder {
      state = 'recording';
      mimeType = 'audio/mp4';
      ondataavailable: ((e: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;
      start() { /* nimmt auf */ }
      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob([new Uint8Array([1, 2, 3])]) });
        this.onstop?.();
      }
    }
    (globalThis as unknown as { MediaRecorder: unknown }).MediaRecorder = FakeRecorder;

    render(<AudioWorkbench {...props()} />);
    fireEvent.click(await screen.findByRole('button', { name: /Aufnehmen/ }));
    fireEvent.click(await screen.findByRole('button', { name: /Stopp/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_add_audio', expect.objectContaining({
        ext: 'm4a',
      }));
    });
  });

  it('kennt zu jedem Aufnahmeformat die Endung', () => {
    expect(audioExtForMime('audio/mp4')).toBe('m4a');
    expect(audioExtForMime('audio/mp4; codecs="mp4a.40.2"')).toBe('m4a');
    expect(audioExtForMime('audio/webm;codecs=opus')).toBe('webm');
    expect(audioExtForMime('audio/ogg')).toBe('ogg');
    expect(audioExtForMime('')).toBe('webm');
  });

  it('sagt beim verweigerten Mikrofon, wo man es erlaubt', async () => {
    // Genau hier verliert man sonst eine Stunde: macOS lehnt still ab, und in
    // der App sieht es aus, als waere die Aufnahme kaputt.
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: () => Promise.reject(new Error('NotAllowedError')) },
    });

    render(<AudioWorkbench {...props()} />);
    fireEvent.click(await screen.findByRole('button', { name: /Aufnehmen/ }));

    await waitFor(() => {
      expect(error).toHaveBeenCalled();
      const [, detail] = error.mock.calls[0];
      expect(String(detail)).toMatch(/Systemeinstellungen/);
    });
  });
});
