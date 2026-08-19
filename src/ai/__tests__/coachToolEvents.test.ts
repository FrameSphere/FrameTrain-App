// Übergabe von KI-Empfehlungen ans Training.
// Befund aus dem E2E-Test vom 18.08.2026: Der Chip meldete "Übernommen",
// obwohl das Training-Panel gar nicht gemountet war und den Wert nie bekam.

import { describe, it, expect, beforeEach } from 'vitest';
import {
  applyCoachConfig,
  onApplyCoachConfig,
  consumePendingCoachConfig,
} from '../coachToolEvents';

beforeEach(() => { consumePendingCoachConfig(); });

describe('applyCoachConfig', () => {
  it('meldet false und merkt vor, wenn niemand zuhoert', () => {
    expect(applyCoachConfig({ batch_size: 32 })).toBe(false);
    expect(consumePendingCoachConfig()).toEqual({ batch_size: 32 });
  });

  it('liefert einen vorgemerkten Patch nur einmal', () => {
    applyCoachConfig({ batch_size: 32 });
    expect(consumePendingCoachConfig()).toEqual({ batch_size: 32 });
    expect(consumePendingCoachConfig()).toBeNull();
  });

  it('meldet true und liefert sofort aus, wenn eine Seite zuhoert', () => {
    const empfangen: Record<string, unknown>[] = [];
    const off = onApplyCoachConfig(p => { empfangen.push(p); });
    expect(applyCoachConfig({ batch_size: 16, epochs: 2 })).toBe(true);
    expect(empfangen).toEqual([{ batch_size: 16, epochs: 2 }]);
    expect(consumePendingCoachConfig()).toBeNull();
    off();
  });

  it('merkt wieder vor, nachdem die Seite verlassen wurde', () => {
    const off = onApplyCoachConfig(() => {});
    off();
    expect(applyCoachConfig({ epochs: 5 })).toBe(false);
    expect(consumePendingCoachConfig()).toEqual({ epochs: 5 });
  });
});
