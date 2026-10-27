// Code-KI in Dev Train / Dev Test.
// Regression aus Karols Test vom 27.10.2026 (Chronos-2 + Zeitreihen-Parquet):
// Die KI kannte nur Pfade, riet ein Skript mit YOLO-/Text-/LLM-Zweig, und die
// Antwort zeigte den Code doppelt, ohne ihn in die Datei zu schreiben.

import { describe, it, expect } from 'vitest';
import { devContextPrompt, stripPythonBlocks } from '../devContext';

const facts = `MODELL (hf_x):
- config.json: model_type=t5, architectures=['Chronos2Model']
PYTHON: /usr/local/bin/python3 (3.11.1)
- NICHT installiert: chronos-forecasting`;

describe('devContextPrompt', () => {
  it('gibt die gelesenen Fakten verbindlich weiter', () => {
    const p = devContextPrompt(facts, 'train');
    expect(p).toContain('Chronos2Model');
    expect(p).toContain('NICHT installiert: chronos-forecasting');
    expect(p).toContain('Keine');
    expect(p).not.toContain('INSTALLIERTE PAKETE');
  });

  it('erklaert im Training Fortschritts-Events und Speicherort', () => {
    const p = devContextPrompt(facts, 'train');
    expect(p).toContain('"type": "progress"');
    expect(p).toContain('"type": "complete"');
    expect(p).toContain('DIREKT nach OUTPUT_PATH');
  });

  it('nennt im Test results.json statt Trainings-Events', () => {
    const p = devContextPrompt(facts, 'test');
    expect(p).toContain('results.json');
    expect(p).not.toContain('"type": "progress"');
  });

  it('faellt ohne Fakten auf die feste Paketliste zurueck', () => {
    expect(devContextPrompt(null, 'train')).toContain('INSTALLIERTE PAKETE');
  });
});

describe('stripPythonBlocks', () => {
  it('entfernt Codebloecke aus der Erklaerung, Text bleibt', () => {
    const text = 'Hier das Skript:\n\n```python\nprint(1)\n```\n\n\nViel Erfolg.';
    expect(stripPythonBlocks(text)).toBe('Hier das Skript:\n\nViel Erfolg.');
  });
});
