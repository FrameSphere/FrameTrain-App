// @vitest-environment jsdom
// Befunde aus dem Testlauf vom 22.11.: Umbenennen, Mehrfachauswahl, Abgleich
// mit KI-Clients, Dataset-Import mit Zuordnung, Adressen ohne https.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act, renderHook } from '@testing-library/react';

const { invokeMock } = vi.hoisted(() => ({ invokeMock: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...a: unknown[]) => invokeMock(...a), convertFileSrc: (p: string) => p }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/LanguageContext', () => ({
  useLanguage: () => ({ t: (k: string, p?: Record<string, unknown>) => (p?.count !== undefined ? `${k}:${p.count}` : k), language: 'de' }),
}));
const melden = { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() };
vi.mock('../../contexts/NotificationContext', () => ({ useNotification: () => melden }));

import EditableName from '../studio/EditableName';
import TextWorkbench from '../studio/TextWorkbench';
import DatasetImportDialog, { WEG, naheliegend, passtZurQuelle, zuordnung } from '../studio/DatasetImportDialog';
import ModelRunDialog, { IGNORIEREN, klassenZuordnung } from '../studio/ModelRunDialog';
import { adressenAus, zerlegeAdressen } from '../studio/FetchDialog';
import { filterListe, herkunftText, useMehrfachauswahl } from '../studio/werkbankHilfen';
import { importDatasetIntoProject } from '../studio/datasetImport';
import type { StudioProject, StudioSample } from '../studio/studioTypes';

const projekt = (over: Partial<StudioProject> = {}): StudioProject => ({
  id: 'p', name: 'SMS', modality: 'text', task: 'classification', target_format: 'flat_file',
  classes: ['spam', 'ham'], created_at: 'x', updated_at: 'x', ...over,
});
const sample = (id: string, label?: string): StudioSample => ({
  id, media: '', mime: 'text/plain', content: `Text ${id}`, status: label ? 'confirmed' : 'new',
  ann: { boxes: [], label }, src: { kind: 'import', at: 'x', origin: '/a/b.csv' }, meta: { w: 0, h: 0 }, abs_path: '',
});

beforeEach(() => { invokeMock.mockReset(); Object.values(melden).forEach(m => m.mockReset()); });
afterEach(cleanup);

describe('EditableName', () => {
  it('speichert mit Enter, verwirft mit Escape und laesst Gleiches in Ruhe', async () => {
    const onSave = vi.fn(async () => true);
    render(<EditableName value="spam" label="umbenennen" onSave={onSave} />);
    fireEvent.click(screen.getByLabelText('umbenennen'));
    fireEvent.change(screen.getByLabelText('umbenennen'), { target: { value: 'werbung' } });
    fireEvent.keyDown(screen.getByLabelText('umbenennen'), { key: 'Enter' });
    await waitFor(() => expect(onSave).toHaveBeenCalledWith('werbung'));
    fireEvent.click(screen.getByLabelText('umbenennen'));
    fireEvent.change(screen.getByLabelText('umbenennen'), { target: { value: 'x' } });
    fireEvent.keyDown(screen.getByLabelText('umbenennen'), { key: 'Escape' });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(screen.getByText('spam')).toBeTruthy();
  });
});

describe('Mehrfachauswahl', () => {
  it('⌘-Klick sammelt, ⇧-Klick nimmt den Bereich, ein einfacher Klick hebt auf', () => {
    const samples = ['a', 'b', 'c', 'd'].map(i => sample(i));
    const { result } = renderHook(() => useMehrfachauswahl(samples));
    const k = (m: object) => m as React.MouseEvent;
    act(() => { expect(result.current.klick(k({ metaKey: true }), 2, 0)).toBe(true); });
    expect([...result.current.ids].sort()).toEqual(['a', 'c']);
    act(() => { result.current.klick(k({ shiftKey: true }), 3, 0); });
    expect([...result.current.ids].sort()).toEqual(['a', 'c', 'd']);
    act(() => { expect(result.current.klick(k({}), 1, 0)).toBe(false); });
    expect(result.current.ids.size).toBe(0);
  });
});

describe('Textwerkbank', () => {
  const backend = (samples: StudioSample[]) => invokeMock.mockImplementation(async (cmd: string) => {
    if (cmd === 'studio_list_samples') return { total: samples.length, items: samples };
    if (cmd === 'studio_stats') return { total: samples.length, new: 1, suggested: 0, confirmed: 2, skipped: 0, boxes_total: 0, per_class: [1, 1], empty_confirmed: 0, incomplete: 1, doubts: 0 };
    if (cmd === 'studio_list_projects') return [projekt({ classes: ['spam', 'ham', 'neu'] })];
    if (cmd === 'studio_update_project') return projekt({ classes: ['werbung', 'ham'] });
    return null;
  });

  it('benennt eine Klasse um und liest die Samples neu', async () => {
    backend([sample('1', 'spam'), sample('2', 'ham')]);
    const onChanged = vi.fn();
    render(<TextWorkbench project={projekt()} onBack={() => {}} onProjectChanged={onChanged} />);
    const stift = (await screen.findAllByLabelText('studio.rename.class'))[0];
    fireEvent.click(stift);
    const feld = screen.getAllByLabelText('studio.rename.class').find(e => e.tagName === 'INPUT') as HTMLInputElement;
    fireEvent.change(feld, { target: { value: 'werbung' } });
    fireEvent.keyDown(feld, { key: 'Enter' });
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_update_project', { projectId: 'p', classes: ['werbung', 'ham'] }));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it('gibt mehreren Samples auf einmal eine Klasse und nimmt nichts sonst weg', async () => {
    backend([sample('1'), sample('2'), sample('3')]);
    render(<TextWorkbench project={projekt()} onBack={() => {}} onProjectChanged={() => {}} />);
    fireEvent.click(await screen.findByText('Text 2'), { metaKey: true });
    expect(screen.getByTestId('auswahl-leiste').textContent).toContain('studio.selection.count:2');
    fireEvent.change(screen.getByLabelText('studio.menu.assignClass'), { target: { value: 'ham' } });
    await waitFor(() => {
      const c = invokeMock.mock.calls.filter(x => x[0] === 'studio_set_annotation').map(x => x[1]);
      expect(c).toHaveLength(2);
      expect(c[0]).toMatchObject({ sampleId: '1', status: 'confirmed', label: 'ham', keep: true });
    });
  });

  it('uebernimmt Aenderungen eines KI-Clients — nur die des eigenen Projekts', async () => {
    const samples = [sample('1')];
    backend(samples);
    const onChanged = vi.fn();
    render(<TextWorkbench project={projekt()} onBack={() => {}} onProjectChanged={onChanged} />);
    await screen.findByText('Text 1', { selector: 'span' });
    window.dispatchEvent(new CustomEvent('ft-mcp-changed', { detail: { domain: 'studio', projectId: 'anderes' } }));
    await new Promise(r => setTimeout(r, 600));
    expect(onChanged).not.toHaveBeenCalled();
    samples.push(sample('9'));
    window.dispatchEvent(new CustomEvent('ft-mcp-changed', { detail: { domain: 'studio', projectId: 'p' } }));
    await waitFor(() => expect(screen.getByText('Text 9')).toBeTruthy(), { timeout: 2000 });
    expect(onChanged).toHaveBeenCalled();
  });

  it('zeigt unvollstaendige Samples und den Filter Uebersprungen', async () => {
    backend([sample('1', 'spam')]);
    render(<TextWorkbench project={projekt()} onBack={() => {}} onProjectChanged={() => {}} />);
    expect(await screen.findByText('studio.stats.incomplete:1')).toBeTruthy();
    fireEvent.click(screen.getByText('studio.filter.skipped'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_list_samples', expect.objectContaining({ status: 'skipped' })));
    expect(filterListe(false)).not.toContain('doubt');
  });

  it('exportiert Text mit Aufteilung', async () => {
    backend([sample('1', 'spam')]);
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_samples') return { total: 1, items: [sample('1', 'spam')] };
      if (cmd === 'studio_stats') return { total: 1, new: 0, suggested: 0, confirmed: 1, skipped: 0, boxes_total: 0, per_class: [1, 0], empty_confirmed: 0, doubts: 0 };
      if (cmd === 'list_models') return [{ id: 'm', name: 'distilbert', model_type: 'distilbert' }];
      if (cmd === 'studio_export') return { dataset: { id: 'd', name: 'x' }, path: '/x', report: { total: 1, per_class: [], per_split: [], groups: 1, group_leaks: [], near_duplicates: 0, near_duplicates_across: 0, near_checked: true, licenses: [], without_license: 0, sources: [], warnings: [], hints: [] } };
      return [];
    });
    render(<TextWorkbench project={projekt()} onBack={() => {}} onProjectChanged={() => {}} />);
    fireEvent.click(await screen.findByText('studio.workbench.exportButton'));
    fireEvent.click(await screen.findByText('studio.media.split.701515'));
    await waitFor(() => expect((screen.getByText('studio.export.confirmButton').closest('button') as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByText('studio.export.confirmButton'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_export', expect.objectContaining({ trainRatio: 0.7, valRatio: 0.15 })));
  });
});

describe('Dataset-Import', () => {
  it('bietet nur passende Datasets an', () => {
    const box = projekt({ modality: 'image', task: 'bbox' });
    expect(passtZurQuelle({ extensions: ['.csv', '.md'] }, box)).toBe(false);
    expect(passtZurQuelle({ extensions: ['.PNG', '.txt'] }, box)).toBe(true);
    expect(passtZurQuelle({ extensions: ['jpg'] }, box)).toBe(true);
    expect(passtZurQuelle({ extensions: [] }, box)).toBe(true);
    expect(passtZurQuelle({ extensions: ['.jpg'], dataset_type: 'folder_class' }, box)).toBe(false);
    expect(passtZurQuelle({ extensions: ['.jpg'], dataset_type: 'yolo_bbox' }, projekt({ modality: 'image', task: 'classify' }))).toBe(false);
    expect(passtZurQuelle({ extensions: ['.jpg', '.csv'] }, projekt())).toBe(false);
    expect(passtZurQuelle({ extensions: ['.csv'] }, projekt())).toBe(true);
  });

  it('stellt Parquet-Datasets fremder Modelle zurueck', () => {
    const box = projekt({ modality: 'image', task: 'bbox' });
    const yolo = { id: 'm', name: 'YOLOv8', model_type: 'yolo' } as never;
    const bert = { id: 'm2', name: 'bench-distilbert', model_type: 'distilbert' } as never;
    expect(naheliegend({ extensions: ['.parquet'] }, box, bert)).toBe(false);
    expect(naheliegend({ extensions: ['.parquet'] }, box, yolo)).toBe(true);
    expect(naheliegend({ extensions: ['.parquet'] }, box, undefined)).toBe(true);
    // Echte Bilder zaehlen, egal bei welchem Modell sie liegen.
    expect(naheliegend({ extensions: ['.jpg'] }, box, bert)).toBe(true);
    expect(naheliegend({ extensions: ['.parquet'] }, projekt(), bert)).toBe(true);
    expect(naheliegend({ extensions: ['.parquet'] }, projekt(), yolo)).toBe(false);
  });

  it('stellt aufgeteilte Klassenordner in Boxprojekten zurueck', () => {
    const box = projekt({ modality: 'image', task: 'bbox' });
    const yolo = { id: 'm', name: 'YOLOv8', model_type: 'yolo' } as never;
    const ordner = { extensions: ['jpg', '.png'], dataset_type: 'pre_split' };
    expect(naheliegend(ordner, box, yolo)).toBe(false);
    expect(naheliegend(ordner, box, undefined)).toBe(false);
    expect(naheliegend(ordner, projekt({ modality: 'image', task: 'classify' }), undefined)).toBe(true);
    // Labels daneben (auch COCO-JSON) sprechen fuer Boxen.
    expect(naheliegend({ ...ordner, extensions: ['.jpg', '.json'] }, box, yolo)).toBe(true);
    expect(naheliegend({ ...ordner, dataset_type: 'yolo_bbox' }, box, yolo)).toBe(true);
  });

  it('zeigt die zurueckgestellten erst auf Wunsch', async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_all_datasets') return [
        { id: 'd1', name: 'coco', model_id: 'm1', storage_path: '/d', file_count: 9, extensions: ['.jpg'] },
        { id: 'd2', name: 'imdb', model_id: 'm2', storage_path: '/d', file_count: 1, extensions: ['.parquet'] }];
      if (cmd === 'list_models') return [{ id: 'm1', name: 'YOLOv8', model_type: 'yolo' }, { id: 'm2', name: 'distilbert', model_type: 'distilbert' }];
      return null;
    });
    render(<DatasetImportDialog project={projekt({ modality: 'image', task: 'bbox' })} onClose={() => {}} onDone={() => {}} />);
    fireEvent.click(await screen.findByText('studio.datasetImport.more:1'));
    expect(screen.getByText('imdb')).toBeTruthy();
  });

  it('uebersetzt die Zuordnung', () => {
    const q = ['person', 'car', 'dog'];
    expect(zuordnung(q, { car: 'vehicle', dog: WEG }, false, ['person', 'vehicle']))
      .toEqual({ classMap: { car: 'vehicle', dog: null }, keepUnmapped: true });
    // "Rest weglassen": gleichnamige Klassen gelten als zugeordnet.
    expect(zuordnung(q, { car: 'vehicle' }, true, ['Person', 'vehicle']))
      .toEqual({ classMap: { person: 'Person', car: 'vehicle' }, keepUnmapped: false });
  });

  it('laeuft bis zum Import mit Stichprobe', async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_all_datasets') return [{ id: 'd1', name: 'coco', model_id: 'm1', storage_path: '/d', file_count: 9, dataset_type: 'yolo_bbox', extensions: ['.jpg'] }];
      if (cmd === 'list_models') return [{ id: 'm1', name: 'YOLOv8' }];
      if (cmd === 'inspect_dataset') return { classes: ['person', 'car'] };
      if (cmd === 'stage_dataset') return { studio_source: 'folder', path: '/w/s', work_dir: '/w', label_classes: ['vehicle'], samples: 5 };
      if (cmd === 'studio_import_folder') return { added: 5, duplicates: 0, unreadable: 0 };
      return null;
    });
    const onDone = vi.fn();
    render(<DatasetImportDialog project={projekt({ modality: 'image', task: 'bbox', classes: ['vehicle'] })} onClose={() => {}} onDone={onDone} />);
    expect(await screen.findByText(/YOLOv8/)).toBeTruthy();
    fireEvent.click(screen.getByText('coco'));
    const selects = await screen.findAllByLabelText('studio.datasetImport.mapFor');
    fireEvent.change(selects[1], { target: { value: 'vehicle' } });
    fireEvent.click(screen.getByText('studio.datasetImport.dropRest'));
    fireEvent.change(screen.getByPlaceholderText('studio.datasetImport.maxAll'), { target: { value: '5' } });
    fireEvent.click(screen.getByText('studio.import.startButton'));
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    const req = invokeMock.mock.calls.find(c => c[0] === 'stage_dataset')?.[1];
    expect(req.request).toMatchObject({ project_kind: 'image_boxes', class_map: { car: 'vehicle' }, keep_unmapped: false, max_samples: 5, per_class: true });
    expect(invokeMock).toHaveBeenCalledWith('studio_rewrite_origin', { projectId: 'p', prefix: '/w', datasetName: 'coco' });
  });

  it('raeumt auch nach einem Fehler auf und holt Videos als Ordner', async () => {
    const calls: [string, Record<string, unknown> | undefined][] = [];
    const call = async (cmd: string, a?: Record<string, unknown>) => {
      calls.push([cmd, a]);
      if (cmd === 'stage_dataset') return { studio_source: 'videos', path: '/w/s', work_dir: '/w', samples: 1 };
      if (cmd === 'studio_import_videos') throw new Error('kaputt');
      return null;
    };
    await expect(importDatasetIntoProject(call as never, { id: 'p', modality: 'video', task: 'classify' }, { datasetId: 'd' })).rejects.toThrow('kaputt');
    expect(calls.find(c => c[0] === 'studio_import_videos')?.[1]).toMatchObject({ paths: ['/w/s'] });
    expect(calls[calls.length - 1][0]).toBe('discard_staged');
  });
});

describe('Modell-Dialog', () => {
  it('waehlt ohne passendes Modell nichts vor und warnt', async () => {
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models_with_version_tree') return [{ id: 'm', name: 'smollm', versions: [{ id: 'v', name: 'Original', version_number: 0 }] }];
      if (cmd === 'list_models') return [{ id: 'm', name: 'smollm', model_type: 'llama' }];
      return null;
    });
    render(<ModelRunDialog mode="suggest" project={projekt({ modality: 'video', task: 'classify' })} onClose={() => {}} onDone={() => {}} offen={3} />);
    expect(await screen.findByTestId('no-fitting-model')).toBeTruthy();
    expect((screen.getByText('studio.suggest.runButton').closest('button') as HTMLButtonElement).disabled).toBe(true);
  });

  it('gibt die Klassenzuordnung weiter', async () => {
    expect(klassenZuordnung({ a: '', b: '' })).toBeNull();
    expect(klassenZuordnung({ pedestrian: 'person', bike: IGNORIEREN, x: '' })).toEqual({ pedestrian: 'person', bike: null });
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models_with_version_tree') return [{ id: 'm', name: 'yolov8n', versions: [{ id: 'v1', name: 'v1', version_number: 1 }] }];
      if (cmd === 'list_models') return [{ id: 'm', name: 'yolov8n', model_type: 'yolo' }];
      if (cmd === 'studio_model_classes') return ['pedestrian', 'car'];
      if (cmd === 'studio_review') return { checked: 2, doubts: 0, agree: 1, failed: 0, unknown_classes: ['dog'], not_comparable: 1 };
      return null;
    });
    render(<ModelRunDialog mode="review" project={projekt({ modality: 'image', task: 'bbox', classes: ['person', 'dog'] })} onClose={() => {}} onDone={() => {}} />);
    fireEvent.click(await screen.findByText('studio.suggest.mapLoad'));
    fireEvent.change((await screen.findAllByLabelText('studio.datasetImport.mapFor'))[0], { target: { value: 'person' } });
    fireEvent.click(screen.getByText('studio.suggest.runButton'));
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('studio_review', expect.objectContaining({ classMap: { pedestrian: 'person' } })));
    expect(await screen.findByText('studio.review.report.unknownTitle')).toBeTruthy();
  });
});

describe('Kleinigkeiten', () => {
  it('Adressen ohne https zaehlen, der Rest wird genannt', () => {
    expect(adressenAus('example.com\nhttps://a.de/x, www.b.org/seite?q=1')).toEqual(['https://example.com', 'https://a.de/x', 'https://www.b.org/seite?q=1']);
    expect(zerlegeAdressen('hallo ftp://x.y example.com').unklar).toEqual(['hallo', 'ftp://x.y']);
  });

  it('Herkunft steht in der Sprache der Oberflaeche', () => {
    const t = (k: string) => k;
    expect(herkunftText({ kind: 'create', at: 'x', origin: 'von Hand angelegt' }, t)).toBe('studio.origin.byHand');
    expect(herkunftText({ kind: 'create', at: 'x', origin: 'Zwischenablage' }, t)).toBe('studio.origin.clipboard');
    expect(herkunftText({ kind: 'record', at: 'x', origin: 'Aufnahme' }, t)).toBe('studio.workbench.recording');
    expect(herkunftText({ kind: 'import', at: 'x', origin: '/a/b/c.jpg' }, t)).toBe('c.jpg');
    expect(herkunftText({ kind: 'import', at: 'x', origin: null }, t)).toBe('');
  });
});
