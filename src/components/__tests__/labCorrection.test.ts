// Korrekturen: "so haette es aussehen muessen".
//
// Die Form der Korrektur haengt an der Modalitaet, die Vorbelegung daran, ob
// ueberhaupt eine Loesung vorliegt. Beides hier festgenagelt.

import { describe, it, expect } from 'vitest';
import {
  correctionKindFor, initialCorrection, isCorrectionEmpty, describeCorrection,
  clientToImagePoint, boxFromPoints, isUsableBox,
} from '../labCorrection';

const BOX = { label: 'Tree', x1: 10, y1: 20, x2: 60, y2: 80 };

describe('correctionKindFor', () => {
  it('bestimmt die Editor-Form aus der Modalitaet', () => {
    expect(correctionKindFor('detect')).toBe('boxes');
    expect(correctionKindFor('seq2seq')).toBe('text');
    expect(correctionKindFor('text')).toBe('label');
    expect(correctionKindFor('image')).toBe('label');
    expect(correctionKindFor('audio')).toBe('label');
    expect(correctionKindFor('canvas')).toBe('label');
  });

  it('faellt im Dev-Script-Modus auf die Eingabeart zurueck', () => {
    // Ohne Server gibt es keine Modalitaet – dann entscheidet das Sample.
    expect(correctionKindFor(null, 'image')).toBe('label');
    expect(correctionKindFor(undefined)).toBe('text');
  });
});

describe('initialCorrection', () => {
  it('nimmt bei Boxen das Dataset-Soll vor der Vorhersage', () => {
    const truth = [{ ...BOX, label: 'Soll' }];
    const pred  = [{ ...BOX, label: 'Vorhersage' }];
    const c = initialCorrection('boxes', { truthBoxes: truth, predictedBoxes: pred });
    expect(c.kind === 'boxes' && c.boxes[0].label).toBe('Soll');
  });

  it('nimmt die Vorhersage, wenn es kein Soll gibt', () => {
    // Eine fast richtige Box verschieben ist schneller als neu zeichnen.
    const c = initialCorrection('boxes', { predictedBoxes: [BOX] });
    expect(c.kind === 'boxes' && c.boxes[0].label).toBe('Tree');
  });

  it('kopiert die Boxen, statt die Soll-Liste zu veraendern', () => {
    const truth = [{ ...BOX }];
    const c = initialCorrection('boxes', { truthBoxes: truth });
    if (c.kind === 'boxes') c.boxes[0].x1 = 999;
    expect(truth[0].x1).toBe(10);
  });

  it('startet leer, wenn nichts vorliegt', () => {
    expect(initialCorrection('boxes', {})).toEqual({ kind: 'boxes', boxes: [] });
    expect(initialCorrection('label', {})).toEqual({ kind: 'label', label: '' });
    expect(initialCorrection('text', {})).toEqual({ kind: 'text', text: '' });
  });

  it('belegt Text und Label mit dem Soll, sonst mit der Vorhersage', () => {
    expect(initialCorrection('text', { expectedLabel: 'Soll', predictedText: 'Ist' }))
      .toEqual({ kind: 'text', text: 'Soll' });
    expect(initialCorrection('label', { predictedText: 'Ist' }))
      .toEqual({ kind: 'label', label: 'Ist' });
  });
});

describe('isCorrectionEmpty / describeCorrection', () => {
  it('erkennt leere Korrekturen', () => {
    expect(isCorrectionEmpty(null)).toBe(true);
    expect(isCorrectionEmpty({ kind: 'boxes', boxes: [] })).toBe(true);
    expect(isCorrectionEmpty({ kind: 'label', label: '  ' })).toBe(true);
    expect(isCorrectionEmpty({ kind: 'text', text: 'x' })).toBe(false);
  });

  it('fasst Boxen wie die Vorhersage zusammen', () => {
    const c = { kind: 'boxes' as const, boxes: [BOX, { ...BOX, label: 'Sky' }, BOX] };
    expect(describeCorrection(c)).toBe('3x Tree, Sky');
  });

  it('gibt Text und Label getrimmt zurueck', () => {
    expect(describeCorrection({ kind: 'text', text: '  neuer Text ' })).toBe('neuer Text');
    expect(describeCorrection({ kind: 'label', label: 'Katze' })).toBe('Katze');
    expect(describeCorrection(undefined)).toBe('');
  });
});

describe('clientToImagePoint', () => {
  const rect = { left: 100, top: 50, width: 400, height: 400 };

  it('trifft bei quadratischem Bild die Bildmitte', () => {
    const p = clientToImagePoint(rect, 512, 512, 300, 250);
    expect(p).toEqual({ x: 256, y: 256 });
  });

  it('rechnet die Randstreifen heraus', () => {
    // 800x400-Bild in 400x400-Flaeche: oben und unten je 100px leer.
    const p = clientToImagePoint(rect, 800, 400, 100, 150);
    expect(p).toEqual({ x: 0, y: 0 });
  });

  it('bleibt im Bild, auch wenn die Maus rauslaeuft', () => {
    const p = clientToImagePoint(rect, 512, 512, 9999, -9999);
    expect(p).toEqual({ x: 512, y: 0 });
  });

  it('liefert 0/0 statt NaN bei fehlenden Massen', () => {
    expect(clientToImagePoint(rect, 0, 0, 300, 250)).toEqual({ x: 0, y: 0 });
  });
});

describe('boxFromPoints / isUsableBox', () => {
  it('dreht die Box, wenn rueckwaerts gezogen wurde', () => {
    expect(boxFromPoints({ x: 80, y: 90 }, { x: 10, y: 20 }, 'Tree'))
      .toEqual({ label: 'Tree', x1: 10, y1: 20, x2: 80, y2: 90 });
  });

  it('verwirft Fehlklicks', () => {
    expect(isUsableBox({ label: 'x', x1: 10, y1: 10, x2: 11, y2: 60 }, 512, 512)).toBe(false);
    expect(isUsableBox(BOX, 512, 512)).toBe(true);
  });
});
