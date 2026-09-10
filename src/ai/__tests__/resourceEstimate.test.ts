// Die RAM-Schaetzung — eine Rechnung fuer Panel und Coach.
//
// Praxisfall: auf der Training-Seite stand im RAM-Rechner "~12,5 GB", waehrend
// der Coach im selben Moment "~17,7 GB" antwortete. Zwei getrennte Rechnungen
// mit unterschiedlichen Faktoren, und dem Kontext fehlte der Overhead ganz.
//
// Ausfuehren: npx vitest run src/ai/__tests__/resourceEstimate.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import {
  estimateTrainingRam,
  ramVerdict,
  ramEstimateLines,
  type RamRelevantConfig,
} from '../resourceEstimate';

const base: RamRelevantConfig = {
  batch_size: 8,
  max_seq_length: 128,
  gradient_checkpointing: false,
  fp16: false,
  bf16: false,
  use_lora: false,
  load_in_4bit: false,
  load_in_8bit: false,
  optimizer: 'adamw',
};

describe('estimateTrainingRam', () => {
  it('zaehlt den Framework-Overhead mit — er fehlte im Coach-Kontext', () => {
    const est = estimateTrainingRam(base, 0.28);
    expect(est.overhead).toBeGreaterThan(0);
    const sum = est.weights + est.gradients + est.optimizer + est.activations + est.overhead;
    expect(est.total).toBeCloseTo(sum, 6);
  });

  it('skaliert die Aktivierungen mit der Batch-Groesse', () => {
    const small = estimateTrainingRam({ ...base, batch_size: 8 }, 0.28);
    const large = estimateTrainingRam({ ...base, batch_size: 32 }, 0.28);
    expect(large.activations).toBeCloseTo(small.activations * 4, 6);
    expect(large.total).toBeGreaterThan(small.total);
  });

  it('LoRA und Quantisierung trainieren nur Adapter', () => {
    const full = estimateTrainingRam(base, 4);
    const lora = estimateTrainingRam({ ...base, use_lora: true }, 4);
    expect(lora.adapterOnly).toBe(true);
    expect(lora.gradients).toBeLessThan(full.gradients);
    expect(lora.optimizer).toBeLessThan(full.optimizer);
    expect(lora.total).toBeLessThan(full.total);
  });

  it('Mixed Precision halbiert die Gewichte, Adafactor spart am Optimizer', () => {
    const fp32 = estimateTrainingRam(base, 4);
    const bf16 = estimateTrainingRam({ ...base, bf16: true }, 4);
    expect(bf16.weights).toBeCloseTo(fp32.weights / 2, 6);
    expect(bf16.mixedPrecision).toBe(true);

    const adafactor = estimateTrainingRam({ ...base, optimizer: 'adafactor' }, 4);
    expect(adafactor.optimizer).toBeLessThan(fp32.optimizer);
  });

  it('Gradient Checkpointing senkt die Aktivierungen', () => {
    const off = estimateTrainingRam(base, 1);
    const on = estimateTrainingRam({ ...base, gradient_checkpointing: true }, 1);
    expect(on.activations).toBeLessThan(off.activations);
  });

  it('bleibt bei fehlender Modellgroesse endlich', () => {
    const est = estimateTrainingRam(base, 0);
    expect(Number.isFinite(est.total)).toBe(true);
    expect(est.total).toBeGreaterThan(0);
  });
});

describe('ramVerdict', () => {
  it('meldet OOM, wenn die Schaetzung den nutzbaren Speicher uebersteigt', () => {
    expect(ramVerdict(30, 16)).toBe('exceeds');
  });

  it('meldet knapp, wenn kaum Reserve bleibt', () => {
    // 16 GB → nutzbar 12.8, knapp ab 9.6
    expect(ramVerdict(11, 16)).toBe('tight');
  });

  it('meldet passend bei reichlich Reserve', () => {
    expect(ramVerdict(11, 64)).toBe('fits');
  });

  it('sagt nichts, wenn der System-RAM unbekannt ist', () => {
    expect(ramVerdict(11, null)).toBe('unknown');
    expect(ramVerdict(11, 0)).toBe('unknown');
  });
});

describe('ramEstimateLines', () => {
  it('nennt Peak, System-RAM und eine klare Bewertung', () => {
    const est = estimateTrainingRam({ ...base, batch_size: 64 }, 4);
    const text = ramEstimateLines(est, 4, 16).join('\n');
    expect(text).toContain('Geschätzter Peak-RAM/VRAM');
    expect(text).toContain('Verfügbarer System-RAM: 16.0 GB');
    expect(text).toMatch(/PASST NICHT/);
  });

  it('verbietet eine Aussage, wenn der System-RAM fehlt', () => {
    const text = ramEstimateLines(estimateTrainingRam(base, 1), 1, null).join('\n');
    expect(text).toContain('unbekannt');
    expect(text).not.toMatch(/PASST/);
  });
});
