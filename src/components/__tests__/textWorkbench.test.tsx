// Die Text-Werkbank des Dataset Studio.
//
// Der Griff, auf den es ankommt: eine Zahl weist die Klasse zu, speichert und
// springt zur naechsten offenen Zeile. Wer 2000 Rueckmeldungen einsortiert,
// macht genau das zweitausendmal.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
const warning = vi.fn();
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning, info: vi.fn() }),
}));

import TextWorkbench from '../studio/TextWorkbench';

const TEXTE = [
  {
    id: 's_1', media: '', mime: 'text/plain', content: 'Der Lift stand still, zweimal',
    status: 'new' as const, ann: { boxes: [] },
    src: { kind: 'import', origin: '/daten/feedback.csv', at: '' },
    meta: { w: 0, h: 0 }, abs_path: '',
  },
  {
    id: 's_2', media: '', mime: 'text/plain', content: 'Tolle Piste heute',
    status: 'new' as const, ann: { boxes: [] },
    src: { kind: 'import', origin: '/daten/feedback.csv', at: '' },
    meta: { w: 0, h: 0 }, abs_path: '',
  },
];

const STATS = { total: 2, new: 2, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 0, per_class: [0, 0], empty_confirmed: 0, doubts: 0 };

function projekt(task: 'classification' | 'pairs') {
  return {
    id: 'sp_text', name: 'Rückmeldungen', modality: 'text', task,
    target_format: 'flat_file',
    classes: task === 'pairs' ? [] : ['beschwerde', 'lob'],
    created_at: '', updated_at: '',
  };
}

const props = (task: 'classification' | 'pairs' = 'classification') => ({
  project: projekt(task), onBack: vi.fn(), onProjectChanged: vi.fn(),
});

describe('TextWorkbench', () => {
  beforeEach(() => {
    invokeMock.mockReset();
    warning.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: TEXTE.length, items: TEXTE };
      if (cmd === 'studio_stats') return STATS;
      if (cmd === 'list_models') return [];
      return null;
    });
  });

  it('zeigt den Text und die Klassen des Projekts', async () => {
    render(<TextWorkbench {...props()} />);
    // Der Text steht in der Warteschlange und gross in der Mitte.
    expect((await screen.findAllByText('Der Lift stand still, zweimal')).length)
      .toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('beschwerde').length).toBeGreaterThanOrEqual(1);
  });

  it('weist per Zahl eine Klasse zu und springt zur naechsten Zeile', async () => {
    render(<TextWorkbench {...props()} />);
    await screen.findByText('1 / 2');

    fireEvent.keyDown(window, { key: '1' });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'confirmed', label: 'beschwerde', boxes: [],
      }));
    });
    await screen.findByText('2 / 2');
  });

  it('nimmt eine Klasse auch per Klick', async () => {
    render(<TextWorkbench {...props()} />);
    await screen.findByText('1 / 2');

    const knoepfe = await screen.findAllByRole('button', { name: /lob/ });
    fireEvent.click(knoepfe[0]);

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', label: 'lob',
      }));
    });
  });

  it('bietet Vorschlagen und Pruefen an, Pruefen erst mit bestaetigten Zeilen', async () => {
    // Textprojekte hatten die beiden Knoepfe zuerst gar nicht — sie sind der
    // Unterschied zwischen "kann man benutzen" und "muss man alles tippen".
    render(<TextWorkbench {...props()} />);
    expect(await screen.findByRole('button', { name: /Vorschlagen/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Prüfen/ })).toBeDisabled();
  });

  it('zeigt einen Vorschlag als solchen, bis er bestaetigt ist', async () => {
    const mitVorschlag = [{
      ...TEXTE[0], status: 'suggested' as const,
      ann: { boxes: [], label: 'lob' },
    }, TEXTE[1]];
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: 2, items: mitVorschlag };
      if (cmd === 'studio_stats') return { ...STATS, suggested: 1, new: 1 };
      return null;
    });

    render(<TextWorkbench {...props()} />);
    expect(await screen.findByText(/Vorschlag des Modells/)).toBeInTheDocument();

    // Enter uebernimmt den vorgeschlagenen Wert unveraendert.
    fireEvent.keyDown(window, { key: 'Enter' });
    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'confirmed', label: 'lob',
      }));
    });
  });

  it('legt selbst geschriebene Texte an, eine Zeile je Text', async () => {
    // Was nirgends liegt, muss man schreiben koennen — ohne Umweg ueber eine
    // CSV in einem anderen Programm.
    render(<TextWorkbench {...props()} />);
    // Der Weg fuehrt ueber die Quellenauswahl.
    fireEvent.click(await screen.findByRole('button', { name: /Texte holen/ }));
    fireEvent.click(await screen.findByRole('button', { name: /Selbst schreiben/ }));

    const feld = await screen.findByPlaceholderText('Ein Text je Zeile…');
    fireEvent.change(feld, { target: { value: 'Lift kaputt\nPiste top\n\n  Kasse zu  ' } });

    // Leere Zeilen zaehlen nicht, Leerzeichen werden abgeschnitten.
    expect(await screen.findByText('3 Texte werden angelegt')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /^Anlegen$/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_add_texts', expect.objectContaining({
        projectId: 'sp_text',
        texts: ['Lift kaputt', 'Piste top', 'Kasse zu'],
        label: null,
      }));
    });
  });

  it('gibt den geschriebenen Texten auf Wunsch gleich eine Klasse', async () => {
    render(<TextWorkbench {...props()} />);
    fireEvent.click(await screen.findByRole('button', { name: /Texte holen/ }));
    fireEvent.click(await screen.findByRole('button', { name: /Selbst schreiben/ }));
    fireEvent.change(await screen.findByPlaceholderText('Ein Text je Zeile…'),
      { target: { value: 'Lift kaputt' } });

    // Im Dialog steht die Klassenliste des Projekts.
    const knoepfe = screen.getAllByRole('button', { name: 'beschwerde' });
    fireEvent.click(knoepfe[knoepfe.length - 1]);
    fireEvent.click(screen.getByRole('button', { name: /^Anlegen$/ }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_add_texts', expect.objectContaining({
        texts: ['Lift kaputt'], label: 'beschwerde',
      }));
    });
  });

  it('bestaetigt ein Paar nicht ohne Zieltext', async () => {
    render(<TextWorkbench {...props('pairs')} />);
    const feld = await screen.findByPlaceholderText('Zieltext…');

    fireEvent.keyDown(feld, { key: 'Enter', metaKey: true });

    await waitFor(() => expect(warning).toHaveBeenCalled());
    expect(invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation')).toBeUndefined();
  });

  it('speichert den Zieltext eines Paares mit Cmd+Enter', async () => {
    render(<TextWorkbench {...props('pairs')} />);
    const feld = await screen.findByPlaceholderText('Zieltext…');

    fireEvent.change(feld, { target: { value: 'Die Bahn ist außer Betrieb' } });
    fireEvent.keyDown(feld, { key: 'Enter', metaKey: true });

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalledWith('studio_set_annotation', expect.objectContaining({
        sampleId: 's_1', status: 'confirmed', target: 'Die Bahn ist außer Betrieb',
      }));
    });
  });

  it('laesst Zahlen im Zieltext-Feld in Ruhe', async () => {
    // Im Textfeld darf eine 1 nicht als Klassenkuerzel gelten.
    render(<TextWorkbench {...props('pairs')} />);
    const feld = await screen.findByPlaceholderText('Zieltext…');

    fireEvent.keyDown(feld, { key: '1' });

    expect(invokeMock.mock.calls.find(c => c[0] === 'studio_set_annotation')).toBeUndefined();
  });
});
