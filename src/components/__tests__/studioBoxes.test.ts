// Boxen im Dataset Studio.
//
// Der wunde Punkt ist die Grenze zwischen Speicher- und Arbeitsformat: auf der
// Platte liegen Boxen normalisiert (Mittelpunkt + Groesse), bearbeitet werden
// sie in Bildpixeln. Geht dort etwas schief, sitzen die Boxen im exportierten
// Datensatz verschoben — und das faellt erst im Training auf.

import { describe, it, expect } from 'vitest';
import {
  toPixel, toNormalized, isUsable, hitTest, handleAt, resizeTo, movedBy,
  classLabel, summarize, nextOpenIndex, MIN_SIDE,
  type PixelBox,
} from '../studio/studioBoxes';

describe('Speicher- und Arbeitsformat', () => {
  it('rechnet den Mittelpunkt in Kanten um', () => {
    const px = toPixel({ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.4 }, 1000, 500);
    expect(px).toEqual({ cls: 0, x1: 400, y1: 150, x2: 600, y2: 350 });
  });

  it('kommt hin und zurueck beim selben Wert an', () => {
    const before = { cls: 3, x: 0.31, y: 0.62, w: 0.12, h: 0.08 };
    const after = toNormalized(toPixel(before, 1920, 1080), 1920, 1080);
    expect(after.cls).toBe(3);
    (['x', 'y', 'w', 'h'] as const).forEach(k => {
      expect(after[k]).toBeCloseTo(before[k], 10);
    });
  });

  it('dreht verkehrt herum gezogene Rechtecke um', () => {
    // Von rechts unten nach links oben gezogen: x1 > x2.
    const b = toNormalized({ cls: 1, x1: 800, y1: 400, x2: 200, y2: 100 }, 1000, 500);
    expect(b.x).toBeCloseTo(0.5, 10);
    expect(b.w).toBeCloseTo(0.6, 10);
    expect(b.h).toBeCloseTo(0.6, 10);
  });

  it('haelt Boxen im Bild, auch wenn daneben gezogen wurde', () => {
    const b = toNormalized({ cls: 0, x1: -200, y1: -50, x2: 1400, y2: 900 }, 1000, 500);
    expect(b.w).toBeLessThanOrEqual(1);
    expect(b.h).toBeLessThanOrEqual(1);
  });

  it('meldet Bildmasse von null, statt durch null zu teilen', () => {
    expect(toNormalized({ cls: 0, x1: 0, y1: 0, x2: 10, y2: 10 }, 0, 0))
      .toEqual({ cls: 0, x: 0, y: 0, w: 0, h: 0 });
  });

  it('haelt Fehlklicks fuer unbrauchbar', () => {
    expect(isUsable({ cls: 0, x: 0.5, y: 0.5, w: MIN_SIDE, h: MIN_SIDE })).toBe(true);
    expect(isUsable({ cls: 0, x: 0.5, y: 0.5, w: 0.0001, h: 0.3 })).toBe(false);
  });
});

describe('Auswahl', () => {
  const gross: PixelBox = { cls: 0, x1: 0, y1: 0, x2: 500, y2: 500 };
  const klein: PixelBox = { cls: 1, x1: 100, y1: 100, x2: 200, y2: 200 };

  it('waehlt bei Ueberlappung die kleinere Box', () => {
    // Sonst waere eine Box, die ganz in einer grossen liegt, nie anklickbar.
    expect(hitTest([gross, klein], 150, 150)).toBe(1);
  });

  it('gibt ausserhalb aller Boxen -1 zurueck', () => {
    expect(hitTest([gross, klein], 900, 900)).toBe(-1);
  });

  it('findet die Ecken nur in der Naehe', () => {
    expect(handleAt(klein, 102, 98, 6)).toBe('nw');
    expect(handleAt(klein, 198, 202, 6)).toBe('se');
    expect(handleAt(klein, 150, 150, 6)).toBe(null);
  });
});

describe('Verschieben und Groesse', () => {
  const b: PixelBox = { cls: 0, x1: 100, y1: 100, x2: 200, y2: 200 };

  it('behaelt beim Verschieben die Groesse', () => {
    const m = movedBy(b, 50, -30, 1000, 1000);
    expect(m).toEqual({ cls: 0, x1: 150, y1: 70, x2: 250, y2: 170 });
  });

  it('laesst die Box nicht aus dem Bild laufen', () => {
    const m = movedBy(b, -500, 5000, 1000, 1000);
    expect(m.x1).toBe(0);
    expect(m.y2).toBe(1000);
    expect(m.x2 - m.x1).toBe(100);
  });

  it('klappt beim Ziehen ueber die Gegenecke um statt negativ zu werden', () => {
    const r = resizeTo(b, 'se', 50, 40);
    expect(r.x1).toBe(50);
    expect(r.x2).toBe(100);
    expect(r.y1).toBe(40);
    expect(r.y2).toBe(100);
  });
});

describe('Beschriftung', () => {
  const classes = ['Lift', 'Sky'];

  it('nimmt Namen aus dem Projekt', () => {
    expect(classLabel(1, classes)).toBe('Sky');
  });

  it('erfindet keinen Namen fuer eine unbekannte Klasse', () => {
    expect(classLabel(7, classes)).toBe('#7');
  });

  it('fasst Boxen mit Anzahl zusammen', () => {
    expect(summarize([{ cls: 0 }, { cls: 0 }, { cls: 1 }], classes)).toBe('2x Lift, Sky');
    expect(summarize([], classes)).toBe('');
  });
});

describe('naechstes offenes Bild', () => {
  it('springt vorwaerts zum naechsten unbearbeiteten', () => {
    expect(nextOpenIndex(['confirmed', 'confirmed', 'new', 'new'], 0)).toBe(2);
  });

  it('ueberspringt bereits bestaetigte und uebersprungene', () => {
    expect(nextOpenIndex(['new', 'confirmed', 'skipped', 'suggested'], 0)).toBe(3);
  });

  it('faengt am Ende wieder von vorn an', () => {
    // Sonst bliebe eine Luecke am Listenanfang fuer immer liegen.
    expect(nextOpenIndex(['new', 'confirmed', 'confirmed'], 1)).toBe(0);
  });

  it('meldet -1 wenn nichts mehr offen ist', () => {
    expect(nextOpenIndex(['confirmed', 'skipped'], 0)).toBe(-1);
  });
});
