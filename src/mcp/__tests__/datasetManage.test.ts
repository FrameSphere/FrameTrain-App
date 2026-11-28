import { describe, expect, it } from 'vitest';
import { registry } from '../registry';
import { isDestructive } from '../approval';
import { describeTree } from '../tools/datasets';

describe('Datensatz-Dateiverwaltung', () => {
  it('Freigaben: pruefen frei, aendern einmal, loeschen jedes Mal', () => {
    const tree = registry.get('datasets_tree')!;
    const repair = registry.get('datasets_repair')!;
    const move = registry.get('datasets_files_move')!;
    const del = registry.get('datasets_files_delete')!;
    expect(tree._meta['frametrain/approval']).toBe('none');
    expect(tree.annotations.readOnlyHint).toBe(true);
    for (const t of [repair, move]) {
      expect(t._meta['frametrain/approval']).toBe('once');
      expect(t.annotations.destructiveHint).toBe(true);
      expect(isDestructive(t)).toBe(false);
    }
    expect(del._meta['frametrain/approval']).toBe('always');
    expect(isDestructive(del)).toBe(true);
  });

  it('Befunde nennen die Reparatur', () => {
    const text = describeTree('FruitDet', {
      layout: 'sub_datasets', images: 150, labels: 150, hidden_files: 634, truncated: false,
      dirs: [{ path: '.', images: 150, labels: 150, other: 16, files: 316 }],
      findings: [
        { code: 'hidden_files', message: '634 hidden helper files.', fix: 'remove_hidden' },
        { code: 'sub_datasets', message: '5 separate YOLO sub-datasets.', fix: 'merge_subsets' },
      ],
    });
    expect(text).toContain('layout sub_datasets');
    expect(text).toContain('Fix: datasets_repair action merge_subsets');
    expect(text).toContain('634 hidden files');
  });
});
