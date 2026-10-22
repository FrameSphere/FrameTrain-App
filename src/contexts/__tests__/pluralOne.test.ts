import { describe, it, expect } from 'vitest';
import { translate } from '../LanguageContext';
import { relocalizeDate } from '../../utils/templateText';
import de from '../../locales/de.json';
import en from '../../locales/en.json';

describe('Einzahl ueber <key>_one', () => {
  it('nimmt bei count 1 die Einzahl, sonst die Mehrzahl', () => {
    expect(translate('en', 'modelManager.card.filesLabel', { count: 1 })).toBe('1 file');
    expect(translate('en', 'modelManager.card.filesLabel', { count: 3 })).toBe('3 files');
    expect(translate('de', 'modelManager.card.filesLabel', { count: '1' })).toBe('1 Datei');
    expect(translate('de', 'hosting.page.running', { count: 1, gb: '0.4' })).toBe('1 läuft · 0.4 GB');
  });

  it('jede Einzahl hat in beiden Sprachen ihr Gegenstueck', () => {
    const keys = (o: unknown, p = ''): string[] =>
      o && typeof o === 'object'
        ? Object.entries(o as Record<string, unknown>).flatMap(([k, v]) => keys(v, p ? `${p}.${k}` : k))
        : [p];
    const enKeys = new Set(keys(en));
    const deKeys = new Set(keys(de));
    for (const k of [...enKeys, ...deKeys].filter(k => k.endsWith('_one'))) {
      expect(enKeys.has(k.slice(0, -4)) || deKeys.has(k.slice(0, -4)), k).toBe(true);
    }
  });
});

describe('relocalizeDate', () => {
  it('stellt ein deutsches Datum im Vorlagentext auf Englisch um', () => {
    expect(relocalizeDate('AI-recommended parameters · 07.09.2026, 23:33 · Claude', 'en'))
      .toBe('AI-recommended parameters · 07/09/2026, 23:33 · Claude');
  });
  it('laesst Text ohne Datum stehen', () => {
    expect(relocalizeDate('Created automatically', 'de')).toBe('Created automatically');
  });
});
