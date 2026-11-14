// Zeitreihen- und Tabellen-Plugin: Erkennung, Manifest-Abgleich, Formular,
// Uebersetzungen, Kurve und der Dialog "Leeres Modell anlegen".

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke, convertFileSrc: (p: string) => p }));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import { detectPlugin, detectPluginForModel, getPluginById, hiddenTrainingFieldsForTaskType } from '../registry';
import { TIME_SERIES_MODEL_TYPES } from '../time-series/detect';
import { TABULAR_MODEL_TYPES } from '../tabular/detect';
import { linePath, valueRange, isForecastCurve } from '../time-series/forecastChart';
import TimeSeriesTestPlugin from '../time-series/TestPlugin';
import TabularTestPlugin from '../tabular/TestPlugin';
import BlankModelPanel from '../../components/BlankModelPanel';
import type { DatasetInfo } from '../types';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const manifest = (dir: string) => JSON.parse(readFileSync(
  join(ROOT, 'src-tauri', 'python', 'train_engine', 'plugins', dir, 'manifest.json'), 'utf-8'));
const locale = (lang: string) => JSON.parse(readFileSync(join(ROOT, 'src', 'locales', `${lang}.json`), 'utf-8'));

type Cb = (e: { payload: unknown }) => void;
let listeners: Record<string, Cb[]> = {};
const emit = (event: string, payload: unknown) => listeners[event]?.forEach(cb => cb({ payload }));
const DS: DatasetInfo = { id: 'ds', name: 'Umsatz', model_id: 'm', status: 'unused', file_count: 1, size_bytes: 10 };
const props = { modelPath: '/m', versionId: 'v1', modelId: 'm', modelName: 'M', versionName: 'V', datasets: [DS] };

beforeEach(() => {
  listeners = {};
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockImplementation((event: string, cb: Cb) => {
    (listeners[event] ??= []).push(cb);
    return Promise.resolve(() => { listeners[event] = (listeners[event] ?? []).filter(f => f !== cb); });
  });
});

describe('Erkennung', () => {
  it('leeres Zeitreihenmodell und PatchTST von Hugging Face -> time_series', () => {
    for (const cfg of [{ model_type: 'ft_time_series' }, { model_type: 'patchtst' }]) {
      const r = detectPlugin('/models/local_abc', cfg);
      expect(r.supported && r.plugin.taskType).toBe('time_series');
    }
    const byName = detectPlugin('ibm-granite/granite-timeseries-patchtst');
    expect(byName.supported && byName.plugin.id).toBe('time-series');
  });

  it('leeres Tabellenmodell -> tabular, auch nur ueber die Zuordnung im Ordner', () => {
    const r = detectPluginForModel({ id: 'local_1', name: 'Kunden', source_path: '/m/local_1', model_type: 'ft_tabular' });
    expect(r.supported && r.plugin.taskType).toBe('tabular');
    const o = detectPluginForModel({ id: 'local_2', name: 'x', source_path: '/m/local_2', plugin_override: 'tabular' });
    expect(o.supported && o.plugin.id).toBe('tabular');
  });

  it('greift nicht bei fremden Modellen; Chronos bleibt ohne Engine-Plugin', () => {
    const bert = detectPlugin('bert-base-uncased', { model_type: 'bert' });
    expect(bert.supported && bert.plugin.id).not.toBe('tabular');
    expect(bert.supported && bert.plugin.id).not.toBe('time-series');
    expect(detectPlugin('amazon/chronos-2', { model_type: 't5' }).supported).toBe(false);
  });

  it('Modelltypen passen zu den Python-Manifesten', () => {
    expect([...TIME_SERIES_MODEL_TYPES].sort()).toEqual([...manifest('time_series').supported_architectures].sort());
    expect(manifest('tabular').task_type).toBe('tabular');
    expect(TABULAR_MODEL_TYPES).toEqual(['ft_tabular']);
    const core = readFileSync(join(ROOT, 'src-tauri', 'python', 'train_engine', 'plugins', 'tabular', 'tabular_core.py'), 'utf-8');
    expect(core).toContain('MODEL_TYPE = "ft_tabular"');
  });
});

describe('Formular und Uebersetzungen', () => {
  it('blendet Text-Felder aus, die die Plugins nicht auswerten', () => {
    expect(hiddenTrainingFieldsForTaskType('time_series')).toContain('max_seq_length');
    expect(hiddenTrainingFieldsForTaskType('tabular')).toContain('lora');
  });

  it('jeder Plugin-Parameter und jede neue Kennzahl hat einen Text in EN und DE', () => {
    const keys = [
      ...Object.keys(getPluginById('time-series')!.defaultPluginConfig!),
      ...Object.keys(getPluginById('tabular')!.defaultPluginConfig!),
    ];
    const metrics = ['mae', 'rmse', 'smape', 'mase', 'naive_mae', 'naive_mase', 'mae_skill', 'r2', 'macro_f1', 'baseline_accuracy', 'baseline_mae'];
    for (const lang of ['en', 'de']) {
      const l = locale(lang);
      for (const k of keys) expect(l.trainingPanel.pluginParamLabels[k], `${lang}: ${k}`).toBeTruthy();
      for (const k of metrics) expect(l.testPlugins.generic.metricLabels[k], `${lang}: ${k}`).toBeTruthy();
      expect(l.plugins['time-series'].name).toBeTruthy();
      expect(l.plugins.tabular.name).toBeTruthy();
      expect(Object.keys(l.modelManager.blankModel).sort()).toEqual(Object.keys(locale('en').modelManager.blankModel).sort());
    }
  });
});

describe('Kurve', () => {
  it('Pfad bricht bei null ab und skaliert in die Flaeche', () => {
    expect(linePath([0, 1, null, 1], 30, 10, 0, 1)).toBe('M0.0,10.0L10.0,0.0M30.0,0.0');
    expect(valueRange([null, 2, 4])).toEqual([1.9, 4.1]);
    expect(isForecastCurve({ actual: [1, 2], forecast: [null, 2], forecast_start: 1 })).toBe(true);
    expect(isForecastCurve({ actual: [1] })).toBe(false);
  });

  it('Zeitreihen-Test zeigt nach dem Datensatz-Lauf Kennzahlen und Kurve', async () => {
    mockInvoke.mockResolvedValue({ id: 'test_1' });
    render(<TimeSeriesTestPlugin {...props} />);
    const startButtons = screen.getAllByRole('button');
    fireEvent.click(startButtons[startButtons.length - 1]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('start_test', expect.objectContaining({ taskType: 'time_series' })));
    await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(1));
    act(() => emit('test-complete', {
      test_id: 'test_1', version_id: 'v1',
      data: {
        total_samples: 16, accuracy: null, metrics: { mae: 0.18, naive_mae: 2.9, mase: 0.18 },
        forecast_curve: { series_id: 'a', channel: 'sales', forecast_start: 2, actual: [1, 2, 3, 4], forecast: [null, null, 3.1, 3.9], naive: [null, null, 2, 2] },
      },
    }));
    expect(await screen.findByTestId('forecast-chart')).toBeTruthy();
    expect(screen.getAllByText('0.180')).toHaveLength(2); // MAE und MASE
    expect(screen.getByText('MAE naive Vorhersage')).toBeTruthy();
  });
});

describe('Tabellen-Einzeltest', () => {
  it('schickt die Zeile als Text an die Engine', async () => {
    mockInvoke.mockResolvedValue('single_1');
    render(<TabularTestPlugin {...props} />);
    fireEvent.change(screen.getAllByRole('textbox')[0], { target: { value: 'x1=2; city=Mainz' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
      singleInput: 'x1=2; city=Mainz', taskType: 'tabular',
    })));
  });
});

describe('Leeres Modell anlegen', () => {
  it('ruft create_task_model mit Aufgabe und Name auf', async () => {
    mockInvoke.mockResolvedValue({ id: 'local_x', name: 'Kunden' });
    const onCreated = vi.fn();
    render(<BlankModelPanel gradientClass="" onCreated={onCreated} onError={vi.fn()} />);
    fireEvent.click(screen.getAllByRole('radio')[1]);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Kunden' } });
    fireEvent.click(screen.getAllByRole('button').slice(-1)[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('create_task_model', { task: 'tabular', modelName: 'Kunden' }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
  });

  it('meldet Fehler statt sie zu verschlucken', async () => {
    mockInvoke.mockRejectedValue('Kein User eingeloggt');
    const onError = vi.fn();
    render(<BlankModelPanel gradientClass="" onCreated={vi.fn()} onError={onError} />);
    fireEvent.click(screen.getAllByRole('button').slice(-1)[0]);
    await waitFor(() => expect(onError).toHaveBeenCalledWith('Kein User eingeloggt'));
    expect(mockInvoke).toHaveBeenCalledWith('create_task_model', { task: 'time_series', modelName: 'PatchTST' });
  });
});
