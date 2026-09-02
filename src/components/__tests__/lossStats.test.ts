// Regression aus dem E2E-Test vom 19.08.2026:
// Der Loss-Verlauf zeigte "Start 0.0000 ↑ Infinity%".

import { describe, it, expect } from 'vitest';
import { firstUsableLoss, lossImprovementPct, appendLossPoint } from '../lossStats';

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

// Regression: Bei max_steps=60 lief der Graph bis "Punkt 9" (schien 90 Steps),
// weil Eval-Events (finale Evaluation, step-basierte Eval) als eigene Punkte
// mit demselben step angehaengt wurden.
describe('appendLossPoint', () => {
  type P = { step: number; epoch: number; train_loss: number; val_loss?: number | null };
  const p = (step: number, train: number, val?: number): P =>
    ({ step, epoch: 1, train_loss: train, val_loss: val });

  it('haengt Punkte mit unterschiedlichem step an', () => {
    let pts: P[] = [];
    pts = appendLossPoint(pts, p(10, 0.7));
    pts = appendLossPoint(pts, p(20, 0.6));
    expect(pts.map(x => x.step)).toEqual([10, 20]);
  });

  it('fuehrt ein Eval-Event mit demselben step zusammen statt anzuhaengen', () => {
    let pts: P[] = [p(60, 0.66)];
    // finale Evaluation kommt mit step=60 und val_loss
    pts = appendLossPoint(pts, { step: 60, epoch: 1, train_loss: 0.66, val_loss: 0.69 });
    expect(pts).toHaveLength(1);
    expect(pts[0].val_loss).toBe(0.69);
    expect(pts[0].step).toBe(60);
  });

  it('behaelt ein vorhandenes val_loss, wenn das neue Event keins bringt', () => {
    let pts: P[] = [{ step: 20, epoch: 1, train_loss: 0.6, val_loss: 0.5 }];
    pts = appendLossPoint(pts, { step: 20, epoch: 1, train_loss: 0.58, val_loss: undefined });
    expect(pts).toHaveLength(1);
    expect(pts[0].val_loss).toBe(0.5);
    expect(pts[0].train_loss).toBe(0.58);
  });

  it('60 Trainingsschritte + finale Eval ergeben 6 Punkte, nicht 7', () => {
    let pts: P[] = [];
    for (let s = 10; s <= 60; s += 10) pts = appendLossPoint(pts, p(s, 0.7 - s / 1000));
    expect(pts).toHaveLength(6);
    pts = appendLossPoint(pts, { step: 60, epoch: 1, train_loss: 0.64, val_loss: 0.66 });
    expect(pts).toHaveLength(6);              // kein Phantom-Punkt
    expect(pts[pts.length - 1].val_loss).toBe(0.66);
  });
});
