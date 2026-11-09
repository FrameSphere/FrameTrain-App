// Wechselt ein Modell die Gruppe, markiert die Liste seine Zeile (Gleiten und
// kurzes Aufleuchten); Zeilen, die in ihrer Gruppe bleiben, bewegen sich nicht.
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';

vi.mock('../../contexts/LanguageContext', () => ({
  useLanguage: () => ({ language: 'en' as const, t: (key: string) => key }),
}));

import { PipelineList } from '../HomeStellwerk';
import { buildPipeline, type PipelineInput } from '../homePipeline';

const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60000).toISOString();
const f = { date: () => '', ago: () => '', duration: () => '', number: (v: number) => String(v) };
const base: PipelineInput = {
  models: [{ id: 'a', name: 'a' }, { id: 'b', name: 'b' }],
  datasets: [],
  trainings: [
    { id: 'ta', model_id: 'a', model_name: 'a', dataset_name: 'd', status: 'completed', created_at: iso(90), started_at: iso(90), completed_at: iso(80), progress: { train_loss: 0.3 } },
    { id: 'tb', model_id: 'b', model_name: 'b', dataset_name: 'd', status: 'completed', created_at: iso(70), started_at: iso(70), completed_at: iso(60), progress: { train_loss: 0.4 } },
  ],
  tests: [],
  hosts: [],
};

describe('PipelineList: Gruppenwechsel', () => {
  it('markiert nur die Zeile, die in eine andere Gruppe wandert', () => {
    const { container, rerender } = render(<PipelineList rows={buildPipeline(base)} f={f} lastSeen={null} limit={20} />);
    const failed = { ...base, tests: [{ id: 's', model_id: 'b', model_name: 'b', dataset_name: 'd', status: 'failed', created_at: iso(10), completed_at: iso(9), error: 'boom' }] };
    rerender(<PipelineList rows={buildPipeline(failed)} f={f} lastSeen={null} limit={20} />);
    const row = (id: string) => container.querySelector(`[data-row="${id}"]`)!;
    expect(row('b').getAttribute('data-section')).toBe('attention');
    expect(row('b').classList.contains('is-moved')).toBe(true);
    expect(row('a').classList.contains('is-moved')).toBe(false);
  });
});
