// Semantische Segmentierung (HF): Erkennung, Manifest-Abgleich, Dataset-Typ
// image_mask und Test-Oberflaeche (Masken-Ueberlagerung, Legende, IoU je Klasse).

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
  convertFileSrc: (p: string) => `asset://localhost/${encodeURIComponent(p)}`,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import { detectSemanticSegmentation, SEMSEG_MODEL_TYPES } from '../semantic-segmentation/detect';
import SegTestPlugin, { parseDataset, parseSingle } from '../semantic-segmentation/TestPlugin';
import { detectPlugin, detectPluginForModel } from '../registry';
import { checkDatasetCompat, DATASET_TYPE_LABELS } from '../datasetCompat';
import type { DatasetAnalysis } from '../datasetCompatHelpers';
import type { DatasetInfo } from '../types';
import en from '../../locales/en.json';
import de from '../../locales/de.json';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const manifest = (engine: string) => JSON.parse(readFileSync(
  join(ROOT, 'src-tauri', 'python', engine, 'plugins', 'semantic_segmentation', 'manifest.json'), 'utf-8'));

describe('Erkennung', () => {
  it('SEMSEG_MODEL_TYPES == supported_architectures im Train-Manifest', () => {
    expect([...SEMSEG_MODEL_TYPES].sort()).toEqual([...manifest('train_engine').supported_architectures].sort());
    expect(manifest('test_engine').task_type).toBe('semantic_segmentation');
  });

  it('SegFormer/UperNet am model_type, auch der reine Encoder mit-b0', () => {
    expect(detectPlugin('nvidia/mit-b0', { model_type: 'segformer' })).toMatchObject({ supported: true, plugin: { id: 'semantic-segmentation' } });
    expect(detectPlugin('openmmlab/upernet-convnext-tiny', { model_type: 'upernet' })).toMatchObject({ plugin: { id: 'semantic-segmentation' } });
    expect(detectSemanticSegmentation('nvidia/mit-b0')).toBe(true);
    expect(detectSemanticSegmentation('nvidia/segformer-b0-finetuned-ade-512-512')).toBe(true);
  });

  it('BEiT/MobileViT nur mit Segmentierungs-Hinweis, sonst Bildklassifikation', () => {
    expect(detectPlugin('microsoft/beit-base-patch16-224', { model_type: 'beit' }))
      .toMatchObject({ plugin: { id: 'hf-image-classification' } });
    expect(detectPlugin('microsoft/beit-base-finetuned-ade-640-640', { model_type: 'beit' }))
      .toMatchObject({ plugin: { id: 'semantic-segmentation' } });
    expect(detectPlugin('apple/deeplabv3-mobilevit-small', { model_type: 'mobilevit' }))
      .toMatchObject({ plugin: { id: 'semantic-segmentation' } });
    expect(detectPlugin('x', { model_type: 'mobilevit', architectures: ['MobileViTForSemanticSegmentation'] }))
      .toMatchObject({ plugin: { id: 'semantic-segmentation' } });
    expect(detectPlugin('apple/mobilevit-small', { model_type: 'mobilevit' }))
      .toMatchObject({ plugin: { id: 'hf-image-classification' } });
  });

  it('DPT zur Tiefenschaetzung und Mask2Former werden nicht beansprucht', () => {
    expect(detectSemanticSegmentation('Intel/dpt-large', { model_type: 'dpt', architectures: ['DPTForDepthEstimation'] })).toBe(false);
    expect(detectSemanticSegmentation('Intel/dpt-large-ade', { model_type: 'dpt', architectures: ['DPTForSemanticSegmentation'] })).toBe(true);
    expect(detectSemanticSegmentation('facebook/mask2former-swin-tiny-ade-semantic', { model_type: 'mask2former' })).toBe(false);
    expect(detectSemanticSegmentation('google/vit-base-patch16-224', { model_type: 'vit' })).toBe(false);
  });

  it('lokal importiertes Modell mit Namen "segformer-strasse"', () => {
    const r = detectPluginForModel({ id: 'x', name: 'segformer-strasse', source_path: '/data/models/local_abc123', model_type: 'segformer' });
    expect(r).toMatchObject({ supported: true, plugin: { id: 'semantic-segmentation' } });
  });
});

describe('Dataset-Typ image_mask', () => {
  const plugin = { name: 'Semantic Segmentation (HuggingFace)', taskType: 'semantic_segmentation',
    supportedDatasetTypes: ['image_mask', 'pre_split'] as const, preferredDatasetType: 'image_mask' as const };
  const analysis = (t: DatasetAnalysis['detected_type'], exts: string[]): DatasetAnalysis => ({
    detected_type: t, confidence: 90, pairing_status: null, warnings: [], file_count: 4, dir_count: 2, extensions: exts, schema_hint: null,
  });

  it('hat Label und Uebersetzung in beiden Sprachen', () => {
    expect(DATASET_TYPE_LABELS.image_mask.modality).toBe('image');
    expect(en.datasetCompat.datasetTypes.image_mask).toBe('Image + Mask');
    expect(de.datasetCompat.datasetTypes.image_mask).toBe('Bild + Maske');
    expect(Object.keys(en.testPlugins.segmentation).sort()).toEqual(Object.keys(de.testPlugins.segmentation).sort());
  });

  it('Bild + Maske ist perfekt, Ordner-Klassen passen nicht, ohne PNG fehlt die Maske', () => {
    const p = { ...plugin, supportedDatasetTypes: [...plugin.supportedDatasetTypes] };
    expect(checkDatasetCompat('semantic-segmentation', ['.jpg', '.png'], analysis('image_mask', ['.jpg', '.png']), p).overallLevel).toBe('perfect');
    expect(checkDatasetCompat('semantic-segmentation', ['.jpg', '.png'], analysis('pre_split', ['.jpg', '.png']), p).overallLevel).toBe('ok');
    expect(checkDatasetCompat('semantic-segmentation', ['.jpg', '.png'], analysis('folder_class', ['.jpg', '.png']), p).overallLevel).toBe('bad');
    const noMask = checkDatasetCompat('semantic-segmentation', ['.jpg'], analysis('image_mask', ['.jpg']), p);
    expect(noMask.overallLevel).toBe('bad');
    expect(noMask.summaryMsg?.key).toBe('datasetCompat.msg.missing.segmentation');
  });
});

describe('Antworten der Engine', () => {
  it('parseSingle/parseDataset lesen Pfade, Legende und Kennzahlen', () => {
    expect(parseSingle({ image_path: '/a.jpg' })).toBeNull();
    const s = parseSingle({ image_path: '/a.jpg', color_mask_path: '/a_color.png', overlay_path: '/a_overlay.png',
      legend: [{ id: 0, label: 'bg', color: '#000000', share: 0.8 }], inference_time: 0.0123 });
    expect(s).toMatchObject({ colorMaskPath: '/a_color.png', ms: 12 });
    const d = parseDataset({ total_samples: 6, split: 'test', accuracy: 0.98,
      metrics: { mean_iou: 0.94, pixel_accuracy: 0.98, mean_accuracy: 0.97 },
      per_class: [{ id: 1, label: 'kreis', color: '#008000', iou: 0.91 }], images: ['/o1.png', 3] });
    expect(d).toMatchObject({ total: 6, meanIou: 0.94, pixelAccuracy: 0.98, overlays: ['/o1.png'] });
  });
});

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

describe('Test-Oberflaeche', () => {
  it('Einzelbild: Maske liegt ueber dem Bild, Deckkraft regelbar, Legende mit Anteil', async () => {
    mockInvoke.mockResolvedValue('single_1');
    render(<SegTestPlugin {...props} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '/bilder/s00.jpg' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
      singleInput: '/bilder/s00.jpg', singleInputType: 'file', taskType: 'semantic_segmentation',
    })));
    await waitFor(() => expect(listeners['test-single-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-single-complete', { test_id: 'single_1', data: {
      image_path: '/bilder/s00.jpg', color_mask_path: '/out/s00_color.png', overlay_path: '/out/s00_overlay.png',
      inference_time: 0.05, legend: [
        { id: 0, label: 'hintergrund', color: '#000000', share: 0.855 },
        { id: 2, label: 'kreis', color: '#008000', share: 0.045 }],
    } }));
    const mask = await screen.findByAltText(/mask|Maske/i);
    expect(mask.getAttribute('src')).toContain(encodeURIComponent('/out/s00_color.png'));
    expect((mask as HTMLImageElement).style.opacity).toBe('0.55');
    fireEvent.change(screen.getByRole('slider'), { target: { value: '20' } });
    expect((mask as HTMLImageElement).style.opacity).toBe('0.2');
    expect(screen.getByText('kreis')).toBeTruthy();
    expect(screen.getByText('85.5 %')).toBeTruthy();
  });

  it('Datensatz: mIoU, Pixel-Accuracy und IoU je Klasse, abwesende Klasse markiert', async () => {
    mockInvoke.mockResolvedValue({ id: 'job' });
    render(<SegTestPlugin {...props} />);
    const start = screen.getAllByRole('button').find(b => /test/i.test(b.textContent ?? ''))!;
    fireEvent.click(start);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('start_test', expect.objectContaining({
      datasetId: 'ds', taskType: 'semantic_segmentation',
    })));
    await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-complete', { data: {
      total_samples: 6, split: 'test', metrics: { mean_iou: 0.9457, pixel_accuracy: 0.9884, mean_accuracy: 0.987 },
      per_class: [
        { id: 0, label: 'hintergrund', color: '#000000', iou: 0.987 },
        { id: 1, label: 'quadrat', color: '#800000', iou: 0.938 },
        { id: 2, label: 'baum', color: '#008000', iou: null }],
      images: ['/out/overlays/0001_s00_overlay.png'],
    } }));
    const box = await screen.findByTestId('seg-summary');
    expect(box.textContent).toContain('94.6 %');
    expect(box.textContent).toContain('98.8 %');
    expect(box.textContent).toContain('quadrat');
    expect(box.textContent).toContain('93.8 %');
    expect(box.querySelectorAll('img').length).toBe(1);
  });
});
