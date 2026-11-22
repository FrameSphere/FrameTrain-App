// @vitest-environment jsdom
// Dateien ins Fenster ziehen und das Mikrofon: Wege ins Projekt, die im
// Testlauf vom 22.11. offen geblieben waren.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

const { invokeMock, drop } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
  drop: { handler: null as null | ((e: { payload: unknown }) => void), abgemeldet: 0 },
}));
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a), convertFileSrc: (p: string) => p }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: async (h: (e: { payload: unknown }) => void) => { drop.handler = h; return () => { drop.abgemeldet += 1; }; },
  }),
}));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/LanguageContext', () => ({
  useLanguage: () => ({ t: (k: string, p?: Record<string, unknown>) => (p?.count !== undefined ? `${k}:${p.count}` : k), language: 'de' }),
}));
const melden = { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() };
vi.mock('../../contexts/NotificationContext', () => ({ useNotification: () => melden }));

import MediaWorkbench from '../studio/MediaWorkbench';
import ImageWorkbench from '../studio/ImageWorkbench';
import TextWorkbench from '../studio/TextWorkbench';
import { istStille, encodeWav } from '../studio/wavEncode';
import { mikrofonFehler } from '../studio/mikrofon';
import type { StudioProject } from '../studio/studioTypes';

const projekt = (modality: string, task: string): StudioProject => ({
  id: 'p', name: 'QA', modality, task, target_format: 'x', classes: ['eins', 'zwei'], created_at: 'x', updated_at: 'x',
} as StudioProject);
const REPORT = { added: 2, duplicates: 0, unreadable: 0, with_labels: 0, classes_added: [], unknown_ids: [], labels_ignored: 0 };
const LEER = { folders: [], images: [], audio: [], videos: [], texts: [], other: 0 };
const STATS = { total: 0, new: 0, suggested: 0, confirmed: 0, skipped: 0, boxes_total: 0, per_class: [0, 0], empty_confirmed: 0, doubts: 0 };
let ablage: Record<string, unknown> = LEER;
let labels = 0;

beforeEach(() => {
  drop.handler = null; ablage = LEER; labels = 0;
  Object.values(melden).forEach(m => { m.mockReset(); });
  invokeMock.mockReset();
  invokeMock.mockImplementation(async (cmd: string) => {
    if (cmd === 'studio_list_samples') return { total: 0, items: [] };
    if (cmd === 'studio_stats') return STATS;
    if (cmd === 'studio_list_projects' || cmd === 'list_models') return [];
    if (cmd === 'studio_sort_drop') return ablage;
    if (cmd === 'studio_inspect_folder') return { images: 3, with_labels: labels, needs_class_list: false, source_classes: ['a'], max_class_id: 0 };
    if (cmd === 'studio_inspect_text') return { rows: 2, kind: 'csv', columns: ['text', 'label'], labels: ['x'], text_column: 'text', label_column: 'label', samples: [], preview: ['a'] };
    if (cmd.startsWith('studio_import')) return REPORT;
    return null;
  });
});
afterEach(cleanup);

const aufrufe = (cmd: string) => invokeMock.mock.calls.filter(c => c[0] === cmd).map(c => c[1]);
const werfen = async (a: Record<string, unknown>) => {
  ablage = { ...LEER, ...a };
  await waitFor(() => expect(drop.handler).toBeTruthy());
  await act(async () => { drop.handler!({ payload: { type: 'drop', paths: ['/x'] } }); });
};
const p = { onBack: () => {}, onProjectChanged: () => {} };

describe('Ins Fenster ziehen', () => {
  it('zeigt beim Ziehen, dass abgelegt werden kann, und nimmt es wieder weg', async () => {
    render(<MediaWorkbench project={projekt('image', 'classify')} {...p} />);
    await waitFor(() => expect(drop.handler).toBeTruthy());
    act(() => { drop.handler!({ payload: { type: 'enter', paths: ['/a.png'] } }); });
    expect(screen.getByTestId('datei-ablage')).toBeTruthy();
    act(() => { drop.handler!({ payload: { type: 'leave' } }); });
    expect(screen.queryByTestId('datei-ablage')).toBeNull();
    // Gezogener Text bringt keine Pfade mit.
    act(() => { drop.handler!({ payload: { type: 'enter', paths: [] } }); });
    expect(screen.queryByTestId('datei-ablage')).toBeNull();
  });

  it('Bildklassen: Dateien zusammen, jeder Ordner fuer sich', async () => {
    render(<MediaWorkbench project={projekt('image', 'classify')} {...p} />);
    await werfen({ images: ['/a.png', '/b.jpg'], folders: ['/o1', '/o2'] });
    await waitFor(() => expect(aufrufe('studio_import_folder')).toHaveLength(3));
    expect(aufrufe('studio_import_folder')[0]).toMatchObject({ files: ['/a.png', '/b.jpg'] });
    expect(aufrufe('studio_import_folder')[2]).toMatchObject({ sourcePath: '/o2', files: null });
  });

  it('Audio nimmt Audiodateien', async () => {
    render(<MediaWorkbench project={projekt('audio', 'transcript')} {...p} />);
    await werfen({ audio: ['/a.wav'] });
    await waitFor(() => expect(aufrufe('studio_import_audio')).toEqual([{ projectId: 'p', sourcePath: '', ignoreLabels: false, files: ['/a.wav'] }]));
  });

  it('Video fragt nach den Abschnitten und liest dann die abgelegten Quellen', async () => {
    render(<MediaWorkbench project={projekt('video', 'classify')} {...p} />);
    await werfen({ videos: ['/a.mp4'], folders: ['/o'] });
    fireEvent.click(await screen.findByText('studio.drop.importVideos:2'));
    await waitFor(() => expect(aufrufe('studio_import_videos')[0]).toMatchObject({ paths: ['/o', '/a.mp4'], clipSeconds: null }));
  });

  it('Boxen: Bilder direkt, ein Ordner mit Labels fragt nach der Klassenliste', async () => {
    labels = 3;
    render(<ImageWorkbench project={projekt('image', 'bbox')} {...p} />);
    await werfen({ images: ['/a.png'], folders: ['/o1', '/o2'] });
    await waitFor(() => expect(aufrufe('studio_import_folder')).toHaveLength(1));
    expect(aufrufe('studio_import_folder')[0]).toMatchObject({ files: ['/a.png'], ignoreLabels: true });
    await screen.findByText('studio.import.classDialogTitle');
    expect(melden.info).toHaveBeenCalledWith('studio.drop.restTitle', 'studio.drop.rest:1');
  });

  it('Boxen: ein Video oeffnet den Einzelbild-Dialog', async () => {
    render(<ImageWorkbench project={projekt('image', 'bbox')} {...p} />);
    await werfen({ videos: ['/a.mp4'] });
    await waitFor(() => expect(screen.getByText('studio.video.title')).toBeTruthy());
    expect(melden.warning).not.toHaveBeenCalled();
  });

  it('Text: zeigt die Spaltenauswahl', async () => {
    render(<TextWorkbench project={projekt('text', 'classification')} {...p} />);
    await werfen({ texts: ['/a.csv', '/b.csv'] });
    await waitFor(() => expect(aufrufe('studio_inspect_text')).toEqual([{ sourcePath: '/a.csv', pairs: false }]));
    expect(melden.info).toHaveBeenCalledWith('studio.drop.restTitle', 'studio.drop.rest:1');
  });

  it('sagt, wenn nichts Passendes dabei ist', async () => {
    render(<MediaWorkbench project={projekt('audio', 'classification')} {...p} />);
    await werfen({ images: ['/a.png'], other: 1 });
    await waitFor(() => expect(melden.warning).toHaveBeenCalledWith('studio.drop.nothingTitle', 'studio.drop.audio'));
    expect(aufrufe('studio_import_audio')).toHaveLength(0);
  });

  it('nimmt nichts an, solange ein Dialog offen ist, und meldet sich ab', async () => {
    const { unmount } = render(<MediaWorkbench project={projekt('video', 'classify')} {...p} />);
    fireEvent.click((await screen.findAllByText('studio.media.videoImportButton'))[0]);
    await werfen({ videos: ['/a.mp4'] });
    expect(aufrufe('studio_sort_drop')).toHaveLength(0);
    const vorher = drop.abgemeldet;
    unmount();
    expect(drop.abgemeldet).toBe(vorher + 1);
  });
});

describe('Mikrofon', () => {
  let tracks: { stop: ReturnType<typeof vi.fn> }[];
  const mikro = (getUserMedia: () => Promise<unknown>) => Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
  class Rec {
    state = 'recording'; mimeType = 'audio/mp4';
    ondataavailable: ((e: { data: Blob }) => void) | null = null;
    onstop: (() => void) | null = null;
    start() { /* nimmt auf */ }
    stop() { this.state = 'inactive'; this.ondataavailable?.({ data: new Blob([new Uint8Array([1, 2])]) }); this.onstop?.(); }
  }
  const G = globalThis as unknown as { MediaRecorder: unknown };
  beforeEach(() => { tracks = [{ stop: vi.fn() }]; G.MediaRecorder = Rec; mikro(async () => ({ getTracks: () => tracks })); });

  const aufnehmen = async (convert: () => Promise<Uint8Array>, klasse?: string) => {
    render(<MediaWorkbench project={projekt('audio', 'classification')} {...p} convertToWav={convert} />);
    if (klasse) fireEvent.change((await screen.findAllByLabelText('studio.audio.recordAs'))[0], { target: { value: klasse } });
    fireEvent.click((await screen.findAllByText('studio.audio.record'))[0]);
    fireEvent.click((await screen.findAllByText('studio.audio.stop'))[0]);
  };

  it('legt die Aufnahme unter der gewaehlten Klasse ab', async () => {
    await aufnehmen(async () => encodeWav(new Float32Array([0, 0.5, -0.5])), 'zwei');
    await waitFor(() => expect(aufrufe('studio_add_audio')[0]).toMatchObject({ ext: 'wav', label: 'zwei' }));
    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('speichert reine Stille nicht', async () => {
    await aufnehmen(async () => encodeWav(new Float32Array(1600)));
    await waitFor(() => expect(melden.warning).toHaveBeenCalledWith('studio.audio.emptyTitle', 'studio.audio.emptyDetail'));
    expect(aufrufe('studio_add_audio')).toHaveLength(0);
  });

  it('gibt das Mikrofon frei, wenn der Rekorder nicht startet', async () => {
    G.MediaRecorder = class { constructor() { throw new DOMException('x', 'NotSupportedError'); } };
    render(<MediaWorkbench project={projekt('audio', 'classification')} {...p} />);
    fireEvent.click((await screen.findAllByText('studio.audio.record'))[0]);
    await waitFor(() => expect(melden.error).toHaveBeenCalledWith('studio.audio.micUnsupportedTitle', 'studio.audio.micUnsupportedDetail'));
    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('gibt es auch frei, wenn man waehrend der Freigabe weggeht', async () => {
    let frei: (s: unknown) => void = () => {};
    mikro(() => new Promise(r => { frei = r; }));
    const { unmount } = render(<MediaWorkbench project={projekt('audio', 'classification')} {...p} />);
    fireEvent.click((await screen.findAllByText('studio.audio.record'))[0]);
    unmount();
    await act(async () => { frei({ getTracks: () => tracks }); });
    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('unterscheidet die Gruende', () => {
    expect(mikrofonFehler(new DOMException('x', 'NotFoundError')).title).toBe('studio.audio.micNoneTitle');
    expect(mikrofonFehler(new DOMException('x', 'NotReadableError')).title).toBe('studio.audio.micBusyTitle');
    expect(mikrofonFehler(new TypeError('undefined')).title).toBe('studio.audio.micUnsupportedTitle');
    expect(mikrofonFehler(new DOMException('x', 'NotAllowedError')).title).toBe('studio.audio.micErrorTitle');
  });

  it('haelt leise Aufnahmen nicht fuer Stille', () => {
    expect(istStille(encodeWav(new Float32Array([0, 0.001, 0])))).toBe(false);
    expect(istStille(encodeWav(new Float32Array(100)))).toBe(true);
    expect(istStille(new Uint8Array([1, 2, 3]))).toBe(false);
  });
});
