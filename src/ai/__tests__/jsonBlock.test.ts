// Ausfuehren: npx vitest run src/ai/__tests__/jsonBlock.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { findLastJsonObject } from '../jsonBlock';

describe('findLastJsonObject', () => {
  it('liest einen geschlossenen json-Codeblock', () => {
    const text = 'Vorschlag:\n```json\n{"epochs": 5, "optimizer": "sgd"}\n```';
    expect(findLastJsonObject(text)).toEqual({ epochs: 5, optimizer: 'sgd' });
  });

  it('liest rohes JSON am Ende der Antwort', () => {
    const text = 'Kurze Analyse.\n\n{"epochs": 4, "fp16": true}';
    expect(findLastJsonObject(text)).toEqual({ epochs: 4, fp16: true });
  });

  // Berichte nennen erst den Ist-Zustand und danach die Empfehlung.
  it('nimmt bei mehreren Objekten das letzte', () => {
    const text = 'Aktuell: {"epochs": 3}\nEmpfehlung: {"epochs": 100}';
    expect(findLastJsonObject(text)).toEqual({ epochs: 100 });
  });

  // Der Fall aus der Praxis: max_tokens erreicht, kein schliessendes }.
  it('repariert ein am Token-Limit abgeschnittenes Objekt', () => {
    const text = '```json\n{"epochs":100,"optimizer":"sgd","scheduler":"cosine","logging_steps":20,';
    expect(findLastJsonObject(text)).toEqual({
      epochs: 100, optimizer: 'sgd', scheduler: 'cosine', logging_steps: 20,
    });
  });

  it('repariert auch ohne Codeblock', () => {
    const text = 'Empfehlung: {"epochs":50,"batch_size":16,';
    expect(findLastJsonObject(text)).toEqual({ epochs: 50, batch_size: 16 });
  });

  it('gibt null zurueck, wenn nichts Brauchbares da ist', () => {
    expect(findLastJsonObject('Kein JSON weit und breit.')).toBeNull();
    expect(findLastJsonObject('Angefangen: {"epochs"')).toBeNull();
  });

  it('ignoriert Arrays', () => {
    expect(findLastJsonObject('```json\n[1, 2, 3]\n```')).toBeNull();
  });
});
