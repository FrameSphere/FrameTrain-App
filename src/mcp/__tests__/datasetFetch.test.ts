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
