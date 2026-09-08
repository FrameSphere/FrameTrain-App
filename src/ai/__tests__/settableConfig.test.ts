// Haelt die KI-Whitelist deckungsgleich mit der Trainings-Config.
// Ausfuehren: npx vitest run src/ai/__tests__/settableConfig.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { SETTABLE_CONFIG, coercePatchFromRecord } from '../coachContext';
import { DEFAULT_CONFIG } from '../../components/TrainingPanel';

/** Plugin-Routing, keine Groesse die der Nutzer im Training einstellt. */
const NICHT_EINSTELLBAR = new Set(['task_type', 'plugin_config']);

describe('SETTABLE_CONFIG', () => {
  // Anspruch: Was der Nutzer einstellen kann, darf die KI auch vorschlagen.
  // Vorher fehlten u.a. optimizer, scheduler und dropout — Empfehlungen dazu
  // wurden beim Uebernehmen kommentarlos verworfen.
  it('deckt jedes Feld der Trainings-Config ab', () => {
    const fehlend = Object.keys(DEFAULT_CONFIG)
      .filter(k => !NICHT_EINSTELLBAR.has(k))
      .filter(k => !(k in SETTABLE_CONFIG));
    expect(fehlend).toEqual([]);
  });

  it('enthaelt keine Felder, die es in der Config gar nicht gibt', () => {
    const unbekannt = Object.keys(SETTABLE_CONFIG).filter(k => !(k in DEFAULT_CONFIG));
    expect(unbekannt).toEqual([]);
  });

  it('passt der Typ zum Default-Wert', () => {
    for (const [key, meta] of Object.entries(SETTABLE_CONFIG)) {
      const def = (DEFAULT_CONFIG as unknown as Record<string, unknown>)[key];
      const erwartet = typeof def === 'boolean' ? ['bool']
        : typeof def === 'number' ? ['int', 'float']
        : ['enum', 'text'];
      expect(erwartet, `${key} (default ${JSON.stringify(def)})`).toContain(meta.type);
    }
  });
});

describe('coercePatchFromRecord', () => {
  // Genau der Fall aus der Praxis: die KI empfahl SGD fuer einen YOLO-Lauf.
  it('uebernimmt Optimizer und Scheduler', () => {
    expect(coercePatchFromRecord({ optimizer: 'sgd', scheduler: 'cosine' }))
      .toEqual({ optimizer: 'sgd', scheduler: 'cosine' });
  });

  it('verwirft unbekannte Enum-Werte', () => {
    expect(coercePatchFromRecord({ optimizer: 'zauberstab' })).toEqual({});
  });

  it('uebernimmt Regularisierung und Ablauf-Felder', () => {
    expect(coercePatchFromRecord({
      dropout: 0.1, label_smoothing: 0.05, eval_strategy: 'epoch',
      max_steps: 500, seed: 42, group_by_length: false,
    })).toEqual({
      dropout: 0.1, label_smoothing: 0.05, eval_strategy: 'epoch',
      max_steps: 500, seed: 42, group_by_length: false,
    });
  });
});
