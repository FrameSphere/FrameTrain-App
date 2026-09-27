// Die Medien-Werkbank fuer Bildklassen, Video und Audio.
//
// Geprueft wird, was es vorher nicht gab: eine Klasse je Bild, Abschnitte
// eines Videos (anzeigen, teilen), alte Aufnahmen werden WAV, der Filter
// "Unsicher" und der Bericht nach dem Export.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
const info = vi.fn();
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info }),
}));

import MediaWorkbench, { abschnittText } from '../studio/MediaWorkbench';
import { werkbankFuer, PROJEKTARTEN } from '../studio/StudioPanel';

const sample = (id: string, extra: Record<string, unknown> = {}) => ({
  id, media: `ab/${id}.jpg`, mime: 'image/jpeg', status: 'new' as const,
  ann: { boxes: [] }, src: { kind: 'import', origin: `/daten/${id}.jpg`, at: '' },
  meta: { w: 10, h: 10 }, abs_path: `/media/ab/${id}.jpg`, ...extra,
});

const STATS = { total: 2, new: 2, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 0, per_class: [0, 0], empty_confirmed: 0, doubts: 0 };

function projekt(modality: string, task: string) {
  return { id: 'sp_m', name: 'Tiere', modality, task, target_format: 'folder_class',
    classes: ['katze', 'hund'], created_at: '', updated_at: '' };
}

function antworten(items: unknown[], extra: Record<string, unknown> = {}) {
  invokeMock.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
    if (cmd in extra) return typeof extra[cmd] === 'function' ? (extra[cmd] as (a: unknown) => unknown)(args) : extra[cmd];
    if (cmd === 'studio_list_samples') return { total: items.length, items };
    if (cmd === 'studio_stats') return STATS;
    if (cmd === 'list_models') return [{ id: 'm1', name: 'vit' }];
    return null;
  });
}

describe('MediaWorkbench', () => {
  beforeEach(() => { invokeMock.mockReset(); info.mockReset(); });

  it('Bildklassen: Zahl weist die Klasse zu, das Bild steht in der Mitte', async () => {
    antworten([sample('s_1'), sample('s_2')]);
    render(<MediaWorkbench project={projekt('image', 'classify')} onBack={vi.fn()} onProjectChanged={vi.fn()} />);
    expect(await screen.findByAltText('/daten/s_1.jpg')).toBeInTheDocument();
    fireEvent.keyDown(window, { key: '2' });
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
      sampleId: 's_1', status: 'confirmed', label: 'hund', boxes: [],
    })));
    await screen.findByText('2 / 2');
  });

  it('Video: zeigt den Abschnitt und teilt ihn mit X an der aktuellen Stelle', async () => {
    const clip = sample('s_v', { media: 'ab/v.mp4', mime: 'video/mp4', abs_path: '/media/ab/v.mp4', meta: { w: 0, h: 0, start: 2, end: 6 } });
    antworten([clip], { studio_split_segment: 's_neu' });
    const { container } = render(<MediaWorkbench project={projekt('video', 'classify')} onBack={vi.fn()} onProjectChanged={vi.fn()} />);
    expect((await screen.findAllByText('2,0–6,0 s')).length).toBeGreaterThan(0);
    const video = container.querySelector('video') as HTMLVideoElement;
    Object.defineProperty(video, 'currentTime', { value: 4.5, writable: true });
    Object.defineProperty(video, 'duration', { value: 30 });
    fireEvent.keyDown(window, { key: 'x' });
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_split_segment', {
      projectId: 'sp_m', sampleId: 's_v', at: 4.5, duration: 30,
    }));
  });

  it('Audio: alte M4A-Aufnahmen werden beim Oeffnen WAV', async () => {
    const alt = sample('s_a', { media: 'ab/a.m4a', mime: 'audio/mp4', abs_path: '/media/ab/a.m4a',
      src: { kind: 'record', origin: 'Aufnahme', at: '' } });
    const importiert = sample('s_b', { media: 'ab/b.mp3', mime: 'audio/mpeg', abs_path: '/media/ab/b.mp3' });
    antworten([alt, importiert]);
    globalThis.fetch = vi.fn(async () => ({ arrayBuffer: async () => new ArrayBuffer(4) })) as unknown as typeof fetch;
    const wav = new Uint8Array([82, 73, 70, 70]);
    const convert = vi.fn(async () => wav);
    render(<MediaWorkbench project={{ ...projekt('audio', 'classification') }} onBack={vi.fn()} onProjectChanged={vi.fn()}
      convertToWav={convert} />);
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_replace_audio', {
      projectId: 'sp_m', sampleId: 's_a', bytes: [82, 73, 70, 70],
    }));
    expect(convert).toHaveBeenCalledTimes(1);   // die importierte MP3 bleibt
    await waitFor(() => expect(info).toHaveBeenCalled());
  });

  it('Filter "Unsicher" fragt das Backend nach den unsichersten Vorschlaegen', async () => {
    antworten([sample('s_1', { status: 'suggested', ann: { boxes: [], label: 'katze', confidence: 0.31 } })]);
    render(<MediaWorkbench project={projekt('image', 'classify')} onBack={vi.fn()} onProjectChanged={vi.fn()} />);
    expect(await screen.findAllByText('31 %')).not.toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: 'Unsicher' }));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_list_samples', expect.objectContaining({ status: 'uncertain' })));
  });

  it('Export teilt auf und zeigt danach den Bericht', async () => {
    const report = { total: 20, per_class: [['katze', 18], ['hund', 2]], per_split: [['train', 16], ['val', 4]],
      groups: 20, group_leaks: [], near_duplicates: 0, near_duplicates_across: 0, near_checked: true,
      licenses: [], without_license: 0, sources: [['import', 20]],
      warnings: ['Klasse „hund“ hat nur 2 Beispiele (empfohlen: mindestens 10).'] };
    antworten([sample('s_1')], {
      studio_stats: { ...STATS, confirmed: 20, per_class: [18, 2] },
      studio_export: { dataset: { id: 'd', name: 'Tiere' }, report, path: '/x' },
    });
    render(<MediaWorkbench project={projekt('image', 'classify')} onBack={vi.fn()} onProjectChanged={vi.fn()} />);
    // Balance-Hinweis schon in der Werkbank.
    expect(await screen.findByTestId('balance-warnings')).toHaveTextContent('„hund“ hat erst 2');
    fireEvent.click(screen.getByRole('button', { name: /Exportieren/ }));
    const knopf = await screen.findAllByRole('button', { name: /Exportieren/ });
    await waitFor(() => expect(knopf[knopf.length - 1]).not.toBeDisabled());
    fireEvent.click(knopf[knopf.length - 1]);
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_export', expect.objectContaining({
      trainRatio: 0.8, valRatio: 0.2,
    })));
    expect(await screen.findByTestId('export-report')).toHaveTextContent('hat nur 2 Beispiele');
  });

  it('Export warnt vorher: nur eine Klasse, kein passendes Modell', async () => {
    antworten([sample('s_1')], {
      studio_stats: { ...STATS, confirmed: 3, per_class: [3, 0] },
      list_models: [{ id: 'y', name: 'yolo8n', source_path: 'yolo8n' }],
    });
    render(<MediaWorkbench project={projekt('video', 'classify')} onBack={vi.fn()} onProjectChanged={vi.fn()} />);
    await screen.findByText('1 / 1');
    fireEvent.click(screen.getByRole('button', { name: /Exportieren/ }));
    expect(await screen.findByTestId('one-class-warning')).toBeInTheDocument();
    expect(await screen.findByTestId('no-fitting-model')).toHaveTextContent('MCG-NJU/videomae-base');
  });

  it('Abschnitte werden lesbar angezeigt', () => {
    expect(abschnittText(null, null)).toBeNull();
    expect(abschnittText(0, 4.25)).toBe('0,0–4,3 s');
  });
});

describe('Projektarten', () => {
  it('oeffnen die richtige Werkbank', () => {
    expect(werkbankFuer({ modality: 'image', task: 'bbox' })).toBe('boxes');
    expect(werkbankFuer({ modality: 'image', task: 'classify' })).toBe('media');
    expect(werkbankFuer({ modality: 'video', task: 'classify' })).toBe('media');
    expect(werkbankFuer({ modality: 'audio', task: 'transcript' })).toBe('media');
    expect(werkbankFuer({ modality: 'text', task: 'pairs' })).toBe('text');
    expect(PROJEKTARTEN.video).toMatchObject({ modality: 'video', task: 'classify', targetFormat: 'folder_class' });
  });
});
