// Die Zahlen rechts in der Werkbank nach einer einzelnen Aenderung.
//
// Bisher wurde nur der Status verschoben. "Boxen 0" stand neben einem Bild
// mit Box, und bei Text gab es gar keine Verteilung je Klasse.

import { describe, it, expect } from 'vitest';
import { statsNachAenderung, classIndexFor } from '../studio/studioStats';

const LEER = {
  total: 3, new: 3, suggested: 0, confirmed: 0, skipped: 0,
  boxes_total: 0, per_class: [0, 0], empty_confirmed: 0, doubts: 0,
};
const KLASSEN = ['Tree', 'Sky'];

describe('statsNachAenderung', () => {
  it('zaehlt eine bestaetigte Box mit — gesamt und je Klasse', () => {
    const st = statsNachAenderung(LEER, KLASSEN,
      { status: 'new', boxes: [] },
      { status: 'confirmed', boxes: [{ cls: 1 }] }, true);
    expect(st.boxes_total).toBe(1);
    expect(st.per_class).toEqual([0, 1]);
    expect(st.confirmed).toBe(1);
    expect(st.new).toBe(2);
  });

  it('verschiebt beim Klassenwechsel einer Box zwischen den Klassen', () => {
    const vorher = { ...LEER, boxes_total: 1, per_class: [1, 0] };
    const st = statsNachAenderung(vorher, KLASSEN,
      { status: 'new', boxes: [{ cls: 0 }] },
      { status: 'new', boxes: [{ cls: 1 }] }, true);
    expect(st.boxes_total).toBe(1);
    expect(st.per_class).toEqual([0, 1]);
  });

  it('zaehlt ein Label nur, wenn es bestaetigt ist', () => {
    // Ein Modellvorschlag ist noch keine Aussage ueber die Verteilung.
    const vorgeschlagen = statsNachAenderung(LEER, KLASSEN,
      { status: 'new', boxes: [] },
      { status: 'suggested', boxes: [], label: 'Sky' }, false);
    expect(vorgeschlagen.per_class).toEqual([0, 0]);

    const bestaetigt = statsNachAenderung(vorgeschlagen, KLASSEN,
      { status: 'suggested', boxes: [], label: 'Sky' },
      { status: 'confirmed', boxes: [], label: 'sky' }, false);
    expect(bestaetigt.per_class).toEqual([0, 1]);
    expect(bestaetigt.suggested).toBe(0);
  });

  it('kennt "bestaetigt ohne Box" nur bei Bildern', () => {
    const bild = statsNachAenderung(LEER, KLASSEN,
      { status: 'new', boxes: [] }, { status: 'confirmed', boxes: [] }, true);
    expect(bild.empty_confirmed).toBe(1);
    const text = statsNachAenderung(LEER, KLASSEN,
      { status: 'new', boxes: [] }, { status: 'confirmed', boxes: [], label: 'Tree' }, false);
    expect(text.empty_confirmed).toBe(0);
  });

  it('ordnet Labels wie das Backend zu', () => {
    expect(classIndexFor(' sky ', KLASSEN)).toBe(1);
    expect(classIndexFor('Stone', KLASSEN)).toBe(-1);
    expect(classIndexFor(null, KLASSEN)).toBe(-1);
  });
});
