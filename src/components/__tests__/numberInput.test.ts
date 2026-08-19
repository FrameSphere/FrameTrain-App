// Regression aus dem E2E-Test vom 19.08.2026:
// "0.001" in die Learning Rate getippt -> im Feld stand "0001".

import { describe, it, expect } from 'vitest';
import { parseNumberInput, isIncompleteNumber, clampNumber } from '../numberInput';

describe('parseNumberInput', () => {
  it('liest Dezimalzahlen mit Punkt', () => {
    expect(parseNumberInput('0.001')).toBe(0.001);
    expect(parseNumberInput('2e-5')).toBe(0.00002);
  });

  it('liest Dezimalzahlen mit Komma – die App zeigt "0,06" an', () => {
    expect(parseNumberInput('0,06')).toBe(0.06);
    expect(parseNumberInput('1,5')).toBe(1.5);
  });

  it('gibt bei Zwischenstaenden null zurueck, statt sie zu 0 zu machen', () => {
    // Genau hier lag der Fehler: "0." wurde zu 0 und loeschte den Punkt.
    for (const zwischen of ['', '-', '0.', '0,', '.', ',', '.5', '-0.']) {
      expect(parseNumberInput(zwischen)).toBeNull();
    }
  });

  it('erkennt Muell als ungueltig', () => {
    expect(parseNumberInput('abc')).toBeNull();
    expect(parseNumberInput('1.2.3')).toBeNull();
  });

  it('ganze Zahlen und Null bleiben erhalten', () => {
    expect(parseNumberInput('0')).toBe(0);
    expect(parseNumberInput('42')).toBe(42);
    expect(parseNumberInput('-3')).toBe(-3);
  });
});

describe('isIncompleteNumber', () => {
  it('erkennt angefangene Eingaben', () => {
    expect(isIncompleteNumber('0.')).toBe(true);
    expect(isIncompleteNumber('0,')).toBe(true);
    expect(isIncompleteNumber('-')).toBe(true);
  });

  it('fertige Zahlen sind nicht unvollstaendig', () => {
    expect(isIncompleteNumber('0.001')).toBe(false);
    expect(isIncompleteNumber('42')).toBe(false);
  });
});

describe('clampNumber', () => {
  it('haelt Grenzen ein', () => {
    expect(clampNumber(5, 0, 3)).toBe(3);
    expect(clampNumber(-1, 0, 3)).toBe(0);
    expect(clampNumber(2, 0, 3)).toBe(2);
  });

  it('ohne Grenzen unveraendert', () => {
    expect(clampNumber(1234)).toBe(1234);
  });
});
