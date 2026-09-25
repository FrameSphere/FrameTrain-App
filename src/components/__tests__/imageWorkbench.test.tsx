// Die Bild-Werkbank des Dataset Studio.
//
// Geprueft wird der Weg, auf dem beim Labeln alles zusammenlaeuft: Enter muss
// speichern UND zum naechsten offenen Bild springen. Springt es ohne zu
// speichern, ist die Arbeit weg; speichert es ohne zu springen, landet der
// naechste Tastendruck auf demselben Bild.

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

import ImageWorkbench, { hatBilder } from '../studio/ImageWorkbench';

const PROJECT = {
  id: 'sp_test', name: 'Ski-Lift', modality: 'image', task: 'bbox',
  target_format: 'yolo_bbox', classes: ['Lift', 'Sky'],
  created_at: '2026-09-16T08:00:00Z', updated_at: '2026-09-16T08:00:00Z',
};

const SAMPLES = [
  {
    id: 's_1', media: 'ab/ab.jpg', mime: 'image/jpeg', status: 'new' as const,
    ann: { boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 }] },
    src: { kind: 'import', origin: '/daten/bild_1.jpg', at: '2026-09-16T08:00:00Z' },
    meta: { w: 1000, h: 500 }, abs_path: '/media/ab/ab.jpg',
  },
  {
    id: 's_2', media: 'cd/cd.jpg', mime: 'image/jpeg', status: 'new' as const,
    ann: { boxes: [] },
    src: { kind: 'import', origin: '/daten/bild_2.jpg', at: '2026-09-16T08:00:00Z' },
    meta: { w: 800, h: 600 }, abs_path: '/media/cd/cd.jpg',
  },
];

const STATS = {
  total: 2, new: 2, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 1, per_class: [1, 0], empty_confirmed: 0,
};

function mockBackend(samples = SAMPLES) {
  invokeMock.mockImplementation(async (cmd: string) => {
    if (cmd === 'studio_list_samples') return { total: samples.length, items: samples };
    if (cmd === 'studio_stats') return STATS;
    if (cmd === 'list_models') return [];
    return null;
  });
}

const props = { project: PROJECT, onBack: vi.fn(), onProjectChanged: vi.fn() };

describe('hatBilder', () => {
  it('laesst Textsammlungen aus der Bildquelle heraus', () => {
    expect(hatBilder({ extensions: ['.csv', '.md'] })).toBe(false);
    expect(hatBilder({ extensions: ['.PNG', '.txt'] })).toBe(true);
    expect(hatBilder({ extensions: ['jpg'] })).toBe(true);
    // Unbekannt heisst nicht "keine Bilder".
    expect(hatBilder({ extensions: [] })).toBe(true);
  });
});

describe('ImageWorkbench', () => {
  beforeEach(() => { invokeMock.mockReset(); mockBackend(); });

  it('zeigt die vorhandenen Boxen in der Warteschlange mit Klassennamen', async () => {
    render(<ImageWorkbench {...props} />);
    // "Lift" steht in der Warteschlange und in der Klassenliste.
    expect((await screen.findAllByText('Lift')).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('ohne Box')).toBeInTheDocument();
  });

  it('speichert bei Enter als bestaetigt und springt zum naechsten offenen Bild', async () => {
    // Enter faellt hier bewusst so frueh wie moeglich: die Boxen des Bildes
    // muessen schon stehen, sobald es zu sehen ist. Wurden sie erst in einem
    // Effekt nachgereicht, bestaetigte ein schneller Tastendruck ein leeres
    // Bild und die Labels waren weg.
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');

    fireEvent.keyDown(window, { key: 'Enter' });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        projectId: 'sp_test', sampleId: 's_1', status: 'confirmed',
        boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 }],
      }));
      // Unter Last (volle Suite parallel) brauchte das einmal 1041 ms — knapp
      // ueber dem Standard. Mehr Luft, ohne dass der Test etwas anderes prueft.
    }, { timeout: 3000 });
    // Die Box muss die Umrechnung in Pixel und zurueck unveraendert ueberstehen.
    await screen.findByText('2 / 2');
  });

  it('merkt sich beim Ueberspringen den Status, ohne die Boxen zu verlieren', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');

    fireEvent.keyDown(window, { key: 's' });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'skipped',
        boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 }],
      }));
    });
  });

  it('reagiert nicht auf Tastenkuerzel waehrend in ein Feld getippt wird', async () => {
    render(<ImageWorkbench {...props} />);
    const input = await screen.findByPlaceholderText('Neue Klasse');

    fireEvent.keyDown(input, { key: 's' });

    expect(invokeMock).not.toHaveBeenCalledWith('studio_set_annotation', expect.anything());
  });

  it('behaelt die Auswahl, wenn zwischendurch gespeichert wird', async () => {
    // Gezeichnete Box, 400 ms spaeter faellt der automatische Speichervorgang
    // an. Wird dabei der Bildzustand neu aufgebaut, verliert die Box ihre
    // Auswahl — und die Klassentaste wirkt danach auf nichts mehr.
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.pointerDown(img, { clientX: 600, clientY: 50 });
    fireEvent.pointerMove(img, { clientX: 800, clientY: 250 });
    fireEvent.pointerUp(img, { clientX: 800, clientY: 250 });

    // Nur bei ausgewaehlter Box gibt es diesen Knopf.
    expect(screen.getByRole('button', { name: /Box löschen/ })).toBeInTheDocument();

    // Der verzoegerte Speichervorgang laeuft durch und schickt beide Boxen.
    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call).toBeTruthy();
      expect((call![1] as { boxes: unknown[] }).boxes).toHaveLength(2);
    }, { timeout: 2000 });

    expect(screen.getByRole('button', { name: /Box löschen/ })).toBeInTheDocument();

    invokeMock.mockClear();
    fireEvent.keyDown(window, { key: '2' });

    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call, 'die Klassentaste hat die ausgewaehlte Box nicht erreicht').toBeTruthy();
      expect((call![1] as { boxes: { cls: number }[] }).boxes[1].cls).toBe(1);
    }, { timeout: 2000 });
  });

  it('macht die zuletzt gezeichnete Box rueckgaengig', async () => {
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.pointerDown(img, { clientX: 600, clientY: 50 });
    fireEvent.pointerMove(img, { clientX: 800, clientY: 250 });
    fireEvent.pointerUp(img, { clientX: 800, clientY: 250 });
    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect((call![1] as { boxes: unknown[] }).boxes).toHaveLength(2);
    }, { timeout: 2000 });

    invokeMock.mockClear();
    fireEvent.keyDown(window, { key: 'z', metaKey: true });

    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call, 'der Stand davor wurde nicht gespeichert').toBeTruthy();
      // Zurueck auf die eine Box, mit der das Bild geladen wurde.
      expect((call![1] as { boxes: unknown[] }).boxes).toHaveLength(1);
    }, { timeout: 2000 });
  });

  it('uebernimmt auf Tastendruck die Boxen des vorigen Bildes', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    // Auf das zweite Bild, das keine Boxen hat.
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    await screen.findByText('2 / 2');

    invokeMock.mockClear();
    fireEvent.keyDown(window, { key: 'v' });

    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call).toBeTruthy();
      expect((call![1] as { sampleId: string }).sampleId).toBe('s_2');
      const boxes = (call![1] as { boxes: { cls: number }[] }).boxes;
      expect(boxes).toHaveLength(1);
      expect(boxes[0].cls).toBe(0);
    }, { timeout: 2000 });
  });

  it('haelt ein breites Bild in seiner Spalte', async () => {
    // Ein 16:9-Videobild wurde bei 520 Pixel Hoehe 924 breit und schob sich
    // ueber Warteschlange und Klassenliste. Ohne Zoom muss die Breite begrenzt
    // bleiben; erst der Zoom darf darueber hinaus.
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    expect(img.className).toContain('max-w-full');
    expect(img.className).not.toContain('max-w-none');
  });

  it('vergroessert ein kleines Bild auf die Flaeche', async () => {
    // Ein 224er Bild blieb 224 Pixel gross — zu klein, um genau zu zeichnen.
    // Ohne Zoom richtet sich die Breite nach Hoehe 520 und Seitenverhaeltnis.
    const klein = [{ ...SAMPLES[0], meta: { w: 224, h: 224 } }, SAMPLES[1]];
    mockBackend(klein);
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    const rahmen = img.parentElement as HTMLElement;
    expect(rahmen.style.width).toBe('520px');
    expect(rahmen.style.maxWidth).toBe('100%');
    expect(img.className).toContain('w-full');
  });

  it('vergroessert beim Zoom wirklich', async () => {
    // maxHeight deckelt nur: ein Bild, das kleiner ist als der Deckel, blieb
    // bei jeder Zoomstufe gleich gross. Im Zoom muss die Hoehe gesetzt werden.
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation') as HTMLImageElement;
    expect(img.style.height).toBe('');

    fireEvent.keyDown(window, { key: '+' });
    expect(img.style.height).toBe('780px');
    expect(img.className).toContain('max-w-none');

    fireEvent.keyDown(window, { key: '0' });
    expect(img.style.height).toBe('');
    expect(img.className).toContain('max-w-full');
  });

  it('laesst die Klasse einer Box direkt an der Box aendern', async () => {
    // Die Tastenkuerzel gab es schon, sichtbar war davon nichts. Wer die Box
    // anklickt, muss dort die Klassen finden, die in der Seitenliste stehen.
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    // Die vorhandene Box liegt mittig (0.5/0.5, 0.2x0.4) — also bei 500/250.
    fireEvent.pointerDown(img, { clientX: 500, clientY: 250 });
    fireEvent.pointerUp(img, { clientX: 500, clientY: 250 });

    const auswahl = await screen.findByRole('button', { name: /Box löschen/ });
    expect(auswahl).toBeInTheDocument();

    invokeMock.mockClear();
    // "Sky" steht in der Seitenliste und muss auch am Bild waehlbar sein.
    const skyKnoepfe = screen.getAllByRole('button', { name: 'Sky' });
    fireEvent.click(skyKnoepfe[skyKnoepfe.length - 1]);

    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call).toBeTruthy();
      expect((call![1] as { boxes: { cls: number }[] }).boxes[0].cls).toBe(1);
    }, { timeout: 2000 });
  });

  it('speichert nichts, wenn eine Box nur ausgewaehlt wird', async () => {
    // Auswaehlen ist keine Aenderung. Frueher legte jeder Klick einen
    // Undo-Schritt an und schrieb die unveraenderte Box zurueck.
    render(<ImageWorkbench {...props} />);
    const img = await screen.findByRole('presentation');
    vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500, x: 0, y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    invokeMock.mockClear();
    fireEvent.pointerDown(img, { clientX: 500, clientY: 250 });
    fireEvent.pointerUp(img, { clientX: 500, clientY: 250 });

    await screen.findByRole('button', { name: /Box löschen/ });
    await new Promise(r => setTimeout(r, 600));
    expect(invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation')).toBeUndefined();
    expect(screen.getByRole('button', { name: /Rückgängig/ })).toBeDisabled();
  });

  it('nimmt ein Bild aus der Zwischenablage an', async () => {
    // Ein Screenshot soll nicht erst als Datei gespeichert und dann als Ordner
    // importiert werden muessen.
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');

    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const datei = new File([png], 'ausschnitt.png', { type: 'image/png' });
    const ereignis = new Event('paste') as Event & { clipboardData: unknown };
    ereignis.clipboardData = {
      items: [{ type: 'image/png', getAsFile: () => datei }],
    };

    invokeMock.mockClear();
    window.dispatchEvent(ereignis);

    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_add_image');
      expect(call, 'das eingefuegte Bild kam nicht an').toBeTruthy();
      const args = call![1] as { projectId: string; bytes: number[] };
      expect(args.projectId).toBe('sp_test');
      expect(args.bytes.slice(0, 4)).toEqual([0x89, 0x50, 0x4e, 0x47]);
    }, { timeout: 2000 });
  });

  it('zeigt nach dem Neuladen die neuen Boxen desselben Bildes', async () => {
    // Nach "Vorschlagen" kommt das gezeigte Bild mit Boxen zurueck — unter
    // derselben ID. Wurden die Boxen nur beim ID-Wechsel uebernommen, blieb
    // das Bild leer, und Enter bestaetigte leere Boxen: der Vorschlag war weg.
    const leer = [{ ...SAMPLES[0], ann: { boxes: [] } }, SAMPLES[1]];
    const vorgeschlagen = [
      { ...SAMPLES[0], status: 'suggested' as const,
        ann: { boxes: [{ cls: 1, x: 0.4, y: 0.4, w: 0.2, h: 0.2 }] } },
      SAMPLES[1],
    ];
    let liste: unknown[] = leer;
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: liste.length, items: liste };
      if (cmd === 'studio_stats') return STATS;
      if (cmd === 'studio_add_image') {
        liste = vorgeschlagen;
        return { added: 1, duplicates: 0, unreadable: 0, with_labels: 0,
          classes_added: [], unknown_ids: [], labels_ignored: 0 };
      }
      return null;
    });

    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');

    // Irgendein Weg, der von der Platte neu laedt — hier die Zwischenablage.
    const datei = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'x.png', { type: 'image/png' });
    const ereignis = new Event('paste') as Event & { clipboardData: unknown };
    ereignis.clipboardData = { items: [{ type: 'image/png', getAsFile: () => datei }] };
    window.dispatchEvent(ereignis);

    // Die neue Box steht in der Warteschlange ...
    await waitFor(() => expect(screen.getAllByText('Sky').length).toBeGreaterThanOrEqual(2));

    // ... und Enter bestaetigt sie, statt sie zu loeschen.
    invokeMock.mockClear();
    fireEvent.keyDown(window, { key: 'Enter' });
    await waitFor(() => {
      const call = invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation');
      expect(call, 'Enter hat nichts gespeichert').toBeTruthy();
      const args = call![1] as { sampleId: string; status: string; boxes: { cls: number; w: number }[] };
      expect(args.sampleId).toBe('s_1');
      expect(args.status).toBe('confirmed');
      expect(args.boxes).toHaveLength(1);
      expect(args.boxes[0].cls).toBe(1);
      expect(args.boxes[0].w).toBeCloseTo(0.2, 6);
    });
  });

  it('nimmt ein misslungenes Bild nach zweitem Klick aus dem Projekt', async () => {
    render(<ImageWorkbench {...props} />);
    await screen.findByText('1 / 2');
    fireEvent.click(screen.getByRole('button', { name: /Entfernen/ }));
    fireEvent.click(screen.getByRole('button', { name: /Wirklich entfernen/ }));
    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_delete_samples', {
        projectId: 'sp_test', sampleIds: ['s_1'],
      });
    });
  });

  it('beschriftet die Regler der Aufteilung', async () => {
    // Vorher standen zwei Regler ohne Namen untereinander.
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: 2, items: SAMPLES };
      if (cmd === 'studio_stats') return { ...STATS, confirmed: 3 };
      if (cmd === 'list_models') return [];
      return null;
    });
    render(<ImageWorkbench {...props} />);
    const knopf = await screen.findByRole('button', { name: /Exportieren/ });
    await waitFor(() => expect(knopf).not.toBeDisabled());
    fireEvent.click(knopf);
    fireEvent.click(await screen.findByLabelText(/aufteilen/i));

    fireEvent.change(screen.getByRole('slider', { name: 'Training' }), { target: { value: '80' } });
    fireEvent.change(screen.getByRole('slider', { name: 'Validierung' }), { target: { value: '15' } });
    expect(screen.getByText('5 %')).toBeInTheDocument();
    // Bei drei Bildern sagt der Dialog, dass Teile leer bleiben koennen.
    expect(screen.getByText(/Bei 3 Bildern/)).toBeInTheDocument();
  });

  it('bietet den Export erst an, wenn etwas bestaetigt ist', async () => {
    render(<ImageWorkbench {...props} />);
    const button = await screen.findByRole('button', { name: /Exportieren/ });
    expect(button).toBeDisabled();
  });

  it('fuehrt ohne Bilder zum Import statt in eine leere Flaeche', async () => {
    mockBackend([]);
    render(<ImageWorkbench {...props} />);
    expect(await screen.findByText('Noch keine Bilder')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Bilder importieren/ }).length).toBeGreaterThan(0);
  });
});
