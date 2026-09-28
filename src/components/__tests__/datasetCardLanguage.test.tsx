// Dataset-Karten in der Sprache der Oberflaeche.
//
// Die Erkennung im Backend schrieb deutsche Saetze in die Metadaten, die Typ-
// Namen standen deutsch im Code. Bei englischer Oberflaeche stand auf den
// Karten "Dataset-Typ konnte nicht erkannt werden." und "Voraufgeteilt".
// Jetzt kommen Codes, und alte Metadaten (nur deutscher Text) werden erkannt.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => p,
}));
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({ onDragDropEvent: () => Promise.resolve(() => {}) }),
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: () => Promise.resolve(() => {}) }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotification: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }),
}));
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { colors: { gradient: 'from-purple-600 to-pink-600' } } }),
}));
vi.mock('../../contexts/PageContext', () => ({
  usePageContext: () => ({ setCurrentPageContent: vi.fn() }),
}));
vi.mock('../../ai/coachToolEvents', () => ({
  onCoachCommand: () => () => {},
  consumePendingCoachCommand: () => null,
}));

import DatasetUpload from '../DatasetUpload';
import { LanguageProvider } from '../../contexts/LanguageContext';
import { checkDatasetCompat } from '../../plugins/datasetCompat';
import { genericDatasetCompat } from '../../plugins/genericDatasetCompat';
import { msgText } from '../../plugins/datasetMessages';
import { translate } from '../../contexts/LanguageContext';

const basis = {
  model_id: 'm1', source: 'local', source_path: null, size_bytes: 1000, file_count: 3,
  created_at: '2026-09-20T10:00:00Z', status: 'unused', split_info: null, training_count: 0,
  last_used_at: null, extensions: ['.csv', '.md'],
};

// Neu erkannt: Code + Parameter, deutscher Text nur als Fallback.
const NEU = {
  ...basis, id: 'ds_neu', name: 'neu', storage_path: '/d/neu', dataset_type: 'flat_file',
  warnings: ['Dataset enthält gemischte Dateitypen.'],
  hints: [{ code: 'mixed_file_types', params: {}, text: 'Dataset enthält gemischte Dateitypen.' }],
};
// Aus einer aelteren Version: nur die deutschen Saetze.
const ALT = {
  ...basis, id: 'ds_alt', name: 'alt', storage_path: '/d/alt', dataset_type: 'pre_split',
  warnings: ['3 Bild(er) ohne Label.', 'Dataset enthaelt gemischte Dateitypen.'],
};
const UNBEKANNT = {
  ...basis, id: 'ds_unb', name: 'unbekannt', storage_path: '/d/unb', dataset_type: 'unknown',
  warnings: ['Dataset-Typ konnte nicht erkannt werden.'],
};

const DEUTSCH = [
  'Dataset-Typ konnte nicht erkannt werden', 'gemischte Dateitypen', 'Bild(er) ohne Label',
  'Voraufgeteilt', 'Flat File passt', 'Unbekannt',
];

const en = (key: string, p?: string | Record<string, string | number>) => translate('en', key, p);

describe('Dataset-Karten auf Englisch', { timeout: 15000 }, () => {
  beforeEach(() => {
    localStorage.setItem('ft_language', 'en');
    invokeMock.mockReset();
    invokeMock.mockImplementation(async (cmd: string) => {
      if (cmd === 'list_models') return [{ id: 'm1', name: 'bert', source: 'local' }];
      if (cmd === 'list_datasets_for_model') return [NEU, ALT, UNBEKANNT];
      if (cmd === 'get_dataset_filter_options') return { tasks: [], languages: [], sizes: [] };
      return null;
    });
  });
  afterEach(() => localStorage.removeItem('ft_language'));

  it('zeigt Warnungen und Typen englisch, auch fuer alte Metadaten', async () => {
    render(<LanguageProvider><DatasetUpload /></LanguageProvider>);
    await screen.findByText('unbekannt', {}, { timeout: 5000 });

    expect(screen.getAllByText('Dataset contains mixed file types.')).toHaveLength(2);
    expect(screen.getByText('3 image(s) without a label.')).toBeInTheDocument();
    expect(screen.getByText('Dataset type could not be detected.')).toBeInTheDocument();
    expect(screen.getByText('Pre-Split')).toBeInTheDocument();

    const text = document.body.textContent ?? '';
    for (const de of DEUTSCH) expect(text).not.toContain(de);
  });
});

describe('Kompatibilitaetsmeldungen', () => {
  it('liefern einen Schluessel und behalten den deutschen Text', () => {
    const r = checkDatasetCompat('xlm-roberta', ['.csv'], {
      detected_type: 'pre_split', confidence: 90, pairing_status: null, warnings: [],
      file_count: 1, dir_count: 2, extensions: ['.csv'], schema_hint: null,
    });
    expect(r.summary).toBe('Voraufgeteiltes Dataset. 1 von 1 Formaten sind ideal für XLM-RoBERTa.');
    expect(msgText(en, r.summaryMsg!)).toBe('Pre-split dataset. 1 of 1 formats are ideal for XLM-RoBERTa.');
    expect(msgText(en, r.fileResults[0].reasonMsg!)).toBe('CSV with text/label columns works directly.');
  });

  it('uebersetzen auch Typnamen in Parametern', () => {
    const r = genericDatasetCompat(
      { name: 'YOLO', taskType: 'detect', supportedDatasetTypes: ['yolo_bbox', 'pre_split'] },
      { type: 'folder_class', extensions: ['.txt'], modalities: ['image'], pairingStatus: null }, ['.txt']);
    expect(msgText(en, r.summaryMsg!)).toBe('YOLO cannot read Folder Classes.');
    expect(msgText(en, r.hintMsg!)).toBe('Suitable: YOLO Bounding Box, Pre-Split');
    expect(r.hint).toBe('Geeignet: YOLO Bounding Box, Voraufgeteilt');
  });
});
