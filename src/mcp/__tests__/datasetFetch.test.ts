// Neue Dataset-Werkzeuge: Bericht, Erkennung von Bild-/Detektionsmodellen, Weg ueber fetch_dataset.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke } = vi.hoisted(() => ({ mockInvoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn() }));

import { registry } from '../registry';
import { describeReport, isDetectionModel, isVisionModel } from '../tools/datasets';
import { waitForJob } from '../jobs';
import type { McpCallContext } from '../types';

const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'remote', userId: 'u1' };
const yolo = { id: 'm1', name: 'yolo8n', model_type: 'yolo', plugin_override: 'yolo', source: 'local', local_path: '/m', size_bytes: 1, file_count: 1, created_at: '' };
const bert = { ...yolo, id: 'm2', name: 'bert', model_type: 'bert', plugin_override: null };
const report = (over = {}) => ({ format: 'parquet', images: 0, yolo_labels: 0, classes: [], splits: {}, problems: ['No image files found.'], ...over });

beforeEach(() => mockInvoke.mockReset());

describe('dataset check helpers', () => {
  it('detects detection and vision models', () => {
    expect(isDetectionModel(yolo)).toBe(true);
    expect(isVisionModel({ ...bert, model_type: 'resnet' })).toBe(true);
    expect(isVisionModel(bert)).toBe(false);
  });

  it('describes tables with task, columns and labels', () => {
    const text = describeReport(report({ format: 'parquet', problems: [], table: { rows: 16000, columns: ['text', 'label'], task: 'classification',
      detected: { text_column: 'text', label_column: 'label' }, labels: { joy: 5362, sadness: 4666 }, label_count: 2, examples: [] } }), bert);
    expect(text).toMatch(/16000 row\(s\), task classification/);
    expect(text).toMatch(/labels joy 5362, sadness 4666/);
    expect(text).not.toMatch(/image/);
  });

  it('warns when a vision model gets a dataset without images', () => {
    expect(describeReport(report(), yolo)).toMatch(/Not usable for "yolo8n"/);
    expect(describeReport(report({ format: 'yolo', images: 120, yolo_labels: 120, classes: ['person'], problems: [] }), yolo)).not.toMatch(/Problems/);
  });
});

describe('datasets_download_hf for a YOLO model', () => {
  it('uses fetch_dataset with include, max_rows and YOLO conversion', async () => {
    mockInvoke.mockImplementation(async (cmd: string, a: Record<string, unknown>) => {
      if (cmd === 'list_models') return [yolo];
      if (cmd === 'fetch_dataset') {
        expect(a.source).toMatchObject({ kind: 'hf', repo_id: 'org/roads', include: ['sample/'], max_rows: 100, yolo: true });
        return { dataset: { id: 'ds1', name: 'roads' }, report: report({ format: 'yolo', images: 100, yolo_labels: 100, classes: ['person', 'car'], problems: [] }), conversion: null };
      }
      return null;
    });
    const r = await registry.get('datasets_download_hf')!.run({ repo_id: 'org/roads', model_id: 'm1', include: ['sample/'], max_rows: 100 }, ctx);
    const jobId = (r.structuredContent as { job_id: string }).job_id;
    const job = await waitForJob(jobId, 2000);
    expect(job?.state).toBe('done');
    expect(mockInvoke).toHaveBeenCalledWith('fetch_dataset', expect.anything());
  });

  it('rejects non-https URLs for datasets_import_url', async () => {
    mockInvoke.mockImplementation(async (cmd: string) => (cmd === 'list_models' ? [yolo] : null));
    await expect(registry.get('datasets_import_url')!.run({ url: 'http://x/a.zip', model_id: 'm1', dataset_name: 'x' }, ctx))
      .rejects.toThrow(/https/);
  });
});

describe('studio import from a dataset', () => {
  it('stages with class_map and imports with the right studio command, then cleans up', async () => {
    const calls: string[] = [];
    mockInvoke.mockImplementation(async (cmd: string, a: Record<string, unknown>) => {
      calls.push(cmd);
      if (cmd === 'studio_list_projects') return [{ id: 'p1', name: 'Roads', modality: 'image', task: 'bbox', classes: [] }];
      if (cmd === 'stage_dataset') {
        expect(a.request).toMatchObject({ project_kind: 'image_boxes', class_map: { truck: 'vehicle' }, keep_unmapped: false, max_samples: 50 });
        return { studio_source: 'folder', path: '/w/staged', work_dir: '/w', label_classes: ['person', 'vehicle'], samples: 50, per_class: { person: 30, vehicle: 20 } };
      }
      if (cmd === 'studio_import_folder') {
        expect(a).toMatchObject({ sourcePath: '/w/staged', labelClasses: ['person', 'vehicle'] });
        return { added: 50, duplicates: 0, unreadable: 0 };
      }
      return null;
    });
    const r = await registry.get('studio_import')!.run(
      { project_id: 'p1', source: 'dataset', dataset_id: 'ds1', class_map: { truck: 'vehicle' }, keep_unmapped: false, max_samples: 50 }, ctx);
    const text = r.content[0].text;
    expect(text).toMatch(/Added 50 sample/);
    expect(text).toMatch(/person 30, vehicle 20/);
    expect(calls).toContain('discard_staged');
  });
});

describe('studio_view_samples', () => {
  it('returns images with box legend and text for text samples', async () => {
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === 'studio_list_projects') return [{ id: 'p1', name: 'Roads', modality: 'image', task: 'bbox', classes: ['person', 'vehicle'] }];
      if (cmd === 'studio_list_samples') return { total: 1, items: [{ id: 's1', status: 'suggested', mime: 'image/jpeg', abs_path: '/d/a.jpg', ann: { boxes: [{ cls: 1, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }] } }] };
      if (cmd === 'preview_media') return { data: 'QUJD', mime: 'image/jpeg' };
      return null;
    });
    const r = await registry.get('studio_view_samples')!.run({ project_id: 'p1' }, ctx);
    expect(r.content[1]).toEqual({ type: 'image', data: 'QUJD', mimeType: 'image/jpeg' });
    expect(r.content[0].text).toMatch(/boxes 1:vehicle/);
  });
});
