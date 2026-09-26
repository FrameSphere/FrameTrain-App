// Tastenkuerzel und Rechtsklick-Menue der Bild-Werkbank.
//
// Zeichnen geht nur mit der Maus, aber wer schnell labelt, hat die Hand an der
// Tastatur: B legt eine Box an, Tab waehlt, Pfeile verschieben, ⌥ + Pfeile
// aendern die Groesse, ⌘D dupliziert, ⌘C/⌘V bringt Boxen auf ein anderes Bild.
// Das Rechtsklick-Menue zeigt dieselben Handgriffe mit ihrem Kuerzel.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

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

import ImageWorkbench from '../studio/ImageWorkbench';
import { collectContextMenuActions } from '../../ui/contextMenuRegistry';

const PROJECT = {
  id: 'sp_test', name: 'Ski-Lift', modality: 'image', task: 'bbox',
  target_format: 'yolo_bbox', classes: ['Lift', 'Sky'],
  created_at: '', updated_at: '',
};
// s_1 hat eine Box: Mitte (0.5, 0.5), 0.2 x 0.4 — in Pixeln 400..600 x 150..350.
const SAMPLES = [
  { id: 's_1', media: 'ab/a.jpg', mime: 'image/jpeg', status: 'new' as const,
    ann: { boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 }] },
    src: { kind: 'import', origin: '/a.jpg', at: '' }, meta: { w: 1000, h: 500 }, abs_path: '/m/a.jpg' },
  { id: 's_2', media: 'cd/b.jpg', mime: 'image/jpeg', status: 'new' as const,
    ann: { boxes: [] },
    src: { kind: 'import', origin: '/b.jpg', at: '' }, meta: { w: 1000, h: 500 }, abs_path: '/m/b.jpg' },
];
const STATS = { total: 2, new: 2, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 1, per_class: [1, 0], empty_confirmed: 0, doubts: 0 };

const props = { project: PROJECT, onBack: vi.fn(), onProjectChanged: vi.fn() };

type Box = { cls: number; x: number; y: number; w: number; h: number };
const gespeichert = async (sampleId: string): Promise<Box[]> => {
  let boxes: Box[] = [];
  await waitFor(() => {
    const call = [...invokeMock.mock.calls].reverse()
      .find(c => c[0] === 'studio_set_annotation' && (c[1] as { sampleId: string }).sampleId === sampleId);
    expect(call, `nichts gespeichert fuer ${sampleId}`).toBeTruthy();
    boxes = (call![1] as { boxes: Box[] }).boxes;
    // Gespeichert wird 400 ms verzoegert; auf einer ausgelasteten Maschine
    // dauert der erste Test samt Modulladen deutlich laenger.
  }, { timeout: 6000 });
  return boxes;
};

// Mehr Zeit je Test: gespeichert wird verzoegert, und der erste Test bezahlt
// das Laden der ganzen Werkbank mit.
describe('Bild-Werkbank: Tastatur und Menue', { timeout: 15000 }, () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: 2, items: SAMPLES };
      if (cmd === 'studio_stats') return STATS;
      return null;
    });
  });

  it('legt mit B eine Box in der Mitte an', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'b' });

    const boxes = await gespeichert('s_1');
    expect(boxes).toHaveLength(2);
    expect(boxes[1]).toMatchObject({ cls: 0, x: 0.5, y: 0.5 });
    expect(boxes[1].w).toBeCloseTo(0.25, 6);
  });

  it('waehlt mit Tab und verschiebt mit den Pfeilen, statt zu blaettern', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'Tab' });
    fireEvent.keyDown(window, { key: 'ArrowRight' });

    // Ein Schritt ist 1 % der kuerzeren Seite: 5 Pixel bei 1000 x 500.
    const boxes = await gespeichert('s_1');
    expect(boxes[0].x).toBeCloseTo(0.505, 6);
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
  });

  it('aendert mit Alt und Pfeil die Groesse', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'Tab' });
    fireEvent.keyDown(window, { key: 'ArrowRight', altKey: true });

    const boxes = await gespeichert('s_1');
    expect(boxes[0].w).toBeCloseTo(0.21, 6);
    expect(boxes[0].x).toBeCloseTo(0.5, 6);
  });

  it('dupliziert die ausgewaehlte Box mit Cmd+D', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'Tab' });
    fireEvent.keyDown(window, { key: 'd', metaKey: true });

    const boxes = await gespeichert('s_1');
    expect(boxes).toHaveLength(2);
    expect(boxes[1].x).toBeGreaterThan(boxes[0].x);
  });

  it('laesst Cmd+V durch, statt es als V abzufangen', async () => {
    // "V" (Boxen vom vorigen Bild) fing auch ⌘V ab. Das Einfuegen eines Bildes
    // aus der Zwischenablage kam dadurch nie an.
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    await screen.findByText('2 / 2');

    const durch = fireEvent.keyDown(window, { key: 'v', metaKey: true });
    expect(durch, '⌘V wurde abgefangen').toBe(true);
    await new Promise(r => setTimeout(r, 600));
    expect(invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation')).toBeUndefined();
  });

  it('bringt eine Box mit Cmd+C und Einfuegen auf ein anderes Bild', async () => {
    const geschrieben: string[] = [];
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: (t: string) => { geschrieben.push(t); return Promise.resolve(); } },
    });
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'Tab' });
    fireEvent.keyDown(window, { key: 'c', metaKey: true });
    await waitFor(() => expect(geschrieben).toHaveLength(1), { timeout: 3000 });

    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    await screen.findByText('2 / 2');
    const ereignis = new Event('paste') as Event & { clipboardData: unknown };
    ereignis.clipboardData = { items: [], getData: () => geschrieben[0] };
    window.dispatchEvent(ereignis);

    const boxes = await gespeichert('s_2');
    expect(boxes).toHaveLength(1);
    expect(boxes[0]).toMatchObject({ cls: 0, x: 0.5, y: 0.5 });
  });

  it('zeigt beim Rechtsklick auf eine Box deren Aktionen mit Kuerzel', async () => {
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.contextMenu(img, { clientX: 500, clientY: 250 });
    const aktionen = collectContextMenuActions({ target: img });
    const ids = aktionen.map(a => a.id);
    expect(ids.slice(0, 4)).toEqual(['st-box-class', 'st-box-dup', 'st-box-copy', 'st-box-del']);
    expect(aktionen.find(a => a.id === 'st-box-dup')?.shortcut).toBe('⌘D');
    // Die Klassen des Projekts als Untermenue, die aktuelle ist ausgegraut.
    const klassen = aktionen.find(a => a.id === 'st-box-class')!.submenu!;
    expect(klassen.map(k => k.label)).toEqual(['Lift', 'Sky']);
    expect(klassen[0].disabled).toBe(true);

    klassen[1].onSelect();
    const boxes = await gespeichert('s_1');
    expect(boxes[0].cls).toBe(1);
  });

  it('zeigt neben dem Bild nur Bild- und Projektaktionen', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    const ids = collectContextMenuActions({ target: document.body }).map(a => a.id);
    expect(ids).not.toContain('st-box-dup');
    expect(ids).toContain('st-img-new');
    expect(ids).toContain('st-prj-export');
  });

  it('blendet die Boxen mit H aus und wieder ein', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.keyDown(window, { key: 'h' });
    expect(await screen.findByText(/Boxen ausgeblendet/)).toBeInTheDocument();
    fireEvent.keyDown(window, { key: 'h' });
    await waitFor(() => expect(screen.queryByText(/Boxen ausgeblendet/)).not.toBeInTheDocument());
  });
});
