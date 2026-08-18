// Fortschrittsanzeige fuer eigene Dev-Train-Scripts.
// Regression aus dem E2E-Test vom 18.08.2026: Der Balken stand auf 0 %,
// waehrend daneben "Step 30 / 60" zu lesen war.

import { describe, it, expect } from 'vitest';
import { devProgressPercent } from '../DevTrainPanel';

describe('devProgressPercent', () => {
  it('leitet aus step/total_steps ab, wenn kein Prozentwert kommt', () => {
    expect(devProgressPercent({ step: 30, total_steps: 60 })).toBe(50);
  });

  it('nimmt einen gemeldeten Prozentwert unveraendert', () => {
    expect(devProgressPercent({ progress_percent: 42, step: 30, total_steps: 60 })).toBe(42);
  });

  it('faellt auf Epochen zurueck, wenn Schritte fehlen', () => {
    expect(devProgressPercent({ epoch: 1, total_epochs: 4 })).toBe(25);
  });

  it('deckelt bei 100 %', () => {
    expect(devProgressPercent({ step: 120, total_steps: 60 })).toBe(100);
  });

  it('ohne verwertbare Angaben 0 statt NaN', () => {
    expect(devProgressPercent({})).toBe(0);
    expect(devProgressPercent(null)).toBe(0);
    expect(devProgressPercent({ step: 5, total_steps: 0 })).toBe(0);
  });
});
