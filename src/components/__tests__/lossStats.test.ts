// Regression aus dem E2E-Test vom 19.08.2026:
// Der Loss-Verlauf zeigte "Start 0.0000 ↑ Infinity%".

import { describe, it, expect } from 'vitest';
import { firstUsableLoss, lossImprovementPct } from '../lossStats';

describe('lossImprovementPct', () => {
  it('ignoriert einen Startwert von 0 und nimmt den ersten echten Loss', () => {
    const punkte = [{ train_loss: 0 }, { train_loss: 0.8 }, { train_loss: 0.4 }];
    expect(firstUsableLoss(punkte)).toBe(0.8);
    expect(lossImprovementPct(punkte)).toBeCloseTo(50);
  });

  it('rechnet normale Verlaeufe korrekt', () => {
    expect(lossImprovementPct([{ train_loss: 1.0 }, { train_loss: 0.25 }])).toBeCloseTo(75);
  });

  it('meldet einen Anstieg negativ', () => {
    expect(lossImprovementPct([{ train_loss: 0.4 }, { train_loss: 0.5 }])).toBeCloseTo(-25);
  });

  it('gibt null statt Infinity, wenn es keinen brauchbaren Startwert gibt', () => {
    expect(lossImprovementPct([{ train_loss: 0 }])).toBeNull();
    expect(lossImprovementPct([{ train_loss: 0 }, { train_loss: 0 }])).toBeNull();
    expect(lossImprovementPct([])).toBeNull();
  });

  it('gibt null bei unveraendertem Loss', () => {
    expect(lossImprovementPct([{ train_loss: 0.5 }, { train_loss: 0.5 }])).toBeNull();
  });

  it('kommt mit fehlenden Werten klar', () => {
    expect(lossImprovementPct([{ train_loss: null }, { train_loss: 0.6 }, { train_loss: 0.3 }])).toBeCloseTo(50);
  });
});
