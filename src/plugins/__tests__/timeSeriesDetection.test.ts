// Zeitreihen-Modelle in der Plugin-Erkennung.
// Regression aus dem Chronos-2-Test vom 28.10.2026: config.json nennt
// model_type "t5", das Seq2Seq-Plugin griff zu — Test-Engine und Labor
// stuerzten ab statt klar zu sagen, dass es dafuer kein Engine-Plugin gibt.

import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel, timeSeriesFamily } from '../registry';

describe('Zeitreihen-Modelle', () => {
  it('Chronos-2 ist kein Seq2Seq-Modell, sondern ein Zeitreihen-Modell mit klarem Hinweis', () => {
    const r = detectPluginForModel({ name: 'chronos-2-short', source_path: 'amazon/chronos-2', model_type: 't5' } as never);
    expect(r.supported).toBe(false);
    // Ohne strict verengt TS die Union nicht — Grund explizit lesen.
    const reason = (r as { reason?: string }).reason ?? '';
    expect(reason).toContain('Chronos');
    expect(reason).toContain('Zeitreihen-Modell');
    expect(reason).toContain('Dev Test');
  });

  it('erkennt Chronos auch an der Architektur, ohne Namen', () => {
    expect(timeSeriesFamily('/models/hf_0f8ffdcd434d', { model_type: 't5', architectures: ['Chronos2Model'] })).toBe('Chronos');
    expect(timeSeriesFamily('/models/x', { model_type: 't5', chronos_pipeline_class: 'Chronos2Pipeline' })).toBe('Chronos');
  });

  it('andere Zeitreihen-Familien', () => {
    expect(timeSeriesFamily('google/timesfm-2.0-500m-pytorch')).toBe('TimesFM');
    expect(timeSeriesFamily('Salesforce/moirai-1.1-R-small')).toBe('Moirai');
    expect(timeSeriesFamily('time-series-foundation-models/Lag-Llama')).toBe('Lag-Llama');
  });

  it('ein normales T5 bleibt Seq2Seq', () => {
    const r = detectPlugin('google-t5/t5-small', { model_type: 't5' });
    expect(r.supported).toBe(true);
    expect((r as { plugin?: { taskType: string } }).plugin?.taskType).toBe('seq2seq');
  });

  it('kein Fehltreffer mitten im Wort', () => {
    expect(timeSeriesFamily('my-synchronos-classifier')).toBeUndefined();
  });
});
