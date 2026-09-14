// Regression aus dem App-Durchgang mit 1.2.76: Das Labor zeigte fuer eine
// formatierte JSON-Liste (textklassifizierer_trainingsdaten, 6801 Zeilen)
// Samples wie '"label": "Textgenerierung",' — die Datei kam als 200-Zeilen-
// Vorschau an, und jede Zeile wurde ein eigenes Sample.
import { describe, it, expect } from 'vitest';
import { parseSamples } from '../LaboratoryPanel';

const pretty = (n: number) => JSON.stringify(
  Array.from({ length: n }, (_, i) => ({ label: ['Mathematik', 'Übersetzung'][i % 2], text: `Satz ${i}` })), null, 2);

describe('parseSamples (Labor)', () => {
  it('liest eine formatierte JSON-Liste als Datensaetze mit Text und Label', () => {
    const samples = parseSamples('﻿' + pretty(1360), 'trainingsdaten.json');
    expect(samples).toHaveLength(1360);
    expect(samples[0]).toMatchObject({ text: 'Satz 0', label: 'Mathematik' });
    expect(samples[1]).toMatchObject({ text: 'Satz 1', label: 'Übersetzung' });
  });

  it('macht aus abgeschnittenem JSON keine Zeilen-Samples, sondern meldet einen Fehler', () => {
    const cut = pretty(100).split('\n').slice(0, 200).join('\n') + '\n\n--- [Vorschau: 200 von 402 Zeilen] ---';
    expect(() => parseSamples(cut, 'trainingsdaten.json')).toThrow(/kein gültiges JSON/);
  });

  it('akzeptiert JSONL mit .json-Endung', () => {
    const jsonl = '{"text": "a", "label": "x"}\n{"text": "b", "label": "y"}\n';
    expect(parseSamples(jsonl, 'daten.json').map(s => s.label)).toEqual(['x', 'y']);
  });

  it('der Vorschau-Hinweis wird nie ein Sample', () => {
    const txt = 'erste Zeile\nzweite Zeile\n\n--- [Vorschau: 2 von 900 Zeilen] ---';
    expect(parseSamples(txt, 'daten.txt').map(s => s.text)).toEqual(['erste Zeile', 'zweite Zeile']);
  });
});
