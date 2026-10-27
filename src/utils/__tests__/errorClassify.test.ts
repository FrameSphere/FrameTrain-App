// Fehler-Einstufung im Trainings-Dashboard.
// Regression aus dem Chronos-2-Test vom 27.10.2026: ein einsum-Formfehler
// wurde wegen "quantile_loss(" im Traceback als "NaN/Inf" eingestuft.

import { describe, it, expect } from 'vitest';
import { classifyError } from '../errorClassify';

describe('classifyError', () => {
  it('Formfehler aus torch sind Code-Fehler, auch mit "loss" im Traceback', () => {
    const tb = `Traceback (most recent call last):
  File "dev_x.py", line 157, in <module>
    loss = quantile_loss(model(ctx, cov), fut)
RuntimeError: einsum(): the number of subscripts in the equation (2) does not match the number of dimensions (3) for operand 0 and no ellipsis was given`;
    expect(classifyError(tb)).toBe('code');
    expect(classifyError('RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x7 and 16x768)')).toBe('code');
  });

  it('echte NaN/Inf-Meldungen bleiben Konfigurationsfehler', () => {
    expect(classifyError('ValueError: loss is nan at step 12')).toBe('config');
    expect(classifyError('RuntimeError: Function returned inf values')).toBe('config');
  });

  it('ein Funktionsname mit loss allein ist kein numerisches Problem', () => {
    expect(classifyError('  File "x.py", line 3, in quantile_loss\nSomeCustomError: boom')).toBe('unknown');
  });

  it('"oom" nur als eigenes Wort', () => {
    expect(classifyError('RuntimeError: MPS backend out of memory')).toBe('memory');
    expect(classifyError('OOM when allocating tensor')).toBe('memory');
    expect(classifyError('ValueError: unknown model bigscience/bloom-560m')).not.toBe('memory');
  });

  it('fehlende Pakete bleiben Paketfehler', () => {
    expect(classifyError("ModuleNotFoundError: No module named 'chronos'")).toBe('packages');
  });
});
