import { describe, it, expect } from 'vitest';
import { buildTaskMetricTiles, formatMetric, improved } from '../analysis/taskMetrics';

describe('Aufgabenspezifische Kennzahlen', () => {
  it('legt Vorher-Werte zusammen und zeigt sie zuerst', () => {
    const tiles = buildTaskMetricTiles({
      perplexity: 1.09, exact_match: 0.75, baseline_exact_match: 0.3,
      recall_at_1: 1.0, recall_at_1_before: 0.25, tokens_per_second: 280,
    });
    expect(tiles.map(t => t.key)).toEqual(['exact_match', 'recall_at_1', 'perplexity']);
    expect(tiles[0].before).toBe(0.3);
    expect(improved(tiles[0])).toBe(true);
  });

  it('kleiner ist besser bei WER und Perplexitaet', () => {
    const [wer] = buildTaskMetricTiles({ wer: 0.12, wer_before: 0.4 });
    expect(wer.rate).toBe(true);
    expect(improved(wer)).toBe(true);
    expect(formatMetric(wer.value, wer.rate)).toBe('12.0%');
  });

  it('grosse Zahlen bleiben Zahlen', () => {
    const [p] = buildTaskMetricTiles({ trainable_params: 4884480 });
    expect(p.rate).toBe(false);
    expect(formatMetric(p.value, p.rate)).toBe((4884480).toLocaleString());
  });

  it('leer ohne Daten', () => {
    expect(buildTaskMetricTiles(null)).toEqual([]);
  });
});
