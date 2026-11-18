import { describe, expect, it } from 'vitest';
import {
  entryState, entrySummary, formatBytes, mergeEntries, pendingAutoShares, storageRelPath, versionTitle,
  type CatalogEntry, type LocalModel, type LocalVersion,
} from '../catalog';

const entry = (over: Partial<CatalogEntry>): CatalogEntry => ({
  id: 'e', kind: 'version', localRef: 'v', authorKey: 'seat-a', authorName: 'Anna', title: 'T', payload: {},
  storagePath: null, sizeBytes: null, auto: false, deleted: false, mine: false, canEdit: false,
  createdAt: '2026-11-18T10:00:00Z', updatedAt: '2026-11-18T10:00:00Z', ...over,
});

describe('mergeEntries', () => {
  const a = entry({ id: 'a', updatedAt: '2026-11-18T10:00:00Z' });
  const b = entry({ id: 'b', updatedAt: '2026-11-18T11:00:00Z' });
  it('full answer replaces, newest first', () => {
    expect(mergeEntries([entry({ id: 'old' })], [a, b], true).map(e => e.id)).toEqual(['b', 'a']);
  });
  it('partial answer updates and drops removed entries', () => {
    const a2 = entry({ id: 'a', title: 'neu', updatedAt: '2026-11-18T12:00:00Z' });
    const merged = mergeEntries([a, b], [a2, entry({ id: 'b', deleted: true })], false);
    expect(merged.map(e => [e.id, e.title])).toEqual([['a', 'neu']]);
  });
});

describe('storage paths', () => {
  it('separates authors and kinds, strips odd characters', () => {
    expect(storageRelPath('version', 'owner', 'ver_1')).toBe('models/owner-ver_1');
    expect(storageRelPath('dataset', 'seat:x', 'a/b')).toBe('datasets/seat_x-a_b');
  });
});

describe('entry state and summary', () => {
  it('state', () => {
    expect(entryState(entry({ storagePath: 'models/x' }))).toBe('stored');
    expect(entryState(entry({ mine: true }))).toBe('mine_local');
    expect(entryState(entry({}))).toBe('author_only');
  });
  it('summary prefers validation loss', () => {
    const s = entrySummary(entry({ payload: { modelType: 'detection', metrics: { final_train_loss: 0.5, final_val_loss: 0.3, total_epochs: 40 } } }));
    expect(s).toMatchObject({ task: 'detection', loss: 0.3, epochs: 40 });
    expect(entrySummary(entry({ payload: null }))).toMatchObject({ task: null, loss: null });
  });
});

describe('auto share', () => {
  const model: LocalModel = { id: 'm', name: 'Defekt', source: 'local', model_type: 'detection' };
  const v = (id: string, over: Partial<LocalVersion> = {}): LocalVersion => ({
    id, model_id: 'm', version_name: '', version_number: 2, path: '/x', size_bytes: 1, created_at: '', is_root: false,
    training_metrics: { final_train_loss: 1, final_val_loss: null, total_epochs: 3, total_steps: 10, best_epoch: null, training_duration_seconds: null },
    ...over,
  });
  it('only finished, non-base versions not shared yet', () => {
    const items = [
      { model, version: v('done') },
      { model, version: v('base', { is_root: true }) },
      { model, version: v('running', { training_metrics: null }) },
      { model, version: v('old') },
    ];
    expect(pendingAutoShares(items, new Set(['old'])).map(i => i.version.id)).toEqual(['done']);
  });
  it('titles fall back to the version number', () => {
    expect(versionTitle(model, v('x'))).toBe('Defekt · v2');
    expect(versionTitle(model, v('x', { version_name: 'Nachtlauf' }))).toBe('Defekt · Nachtlauf');
  });
});

describe('formatBytes', () => {
  it('reads like a human', () => {
    expect(formatBytes(null)).toBe('');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(3.1 * 1024 ** 3)).toBe('3.1 GB');
    expect(formatBytes(250 * 1024 ** 2)).toBe('250 MB');
  });
});
