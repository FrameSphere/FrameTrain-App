import { describe, expect, it } from 'vitest';
import { errorMessage } from '../tools/training';

describe('Trainingsfehler fuer die KI lesbar', () => {
  it('liest Text, Objekte und verschachtelte Events', () => {
    expect(errorMessage('YAML-Generierung fehlgeschlagen: Keine Trainings-Bilder')).toBe('YAML-Generierung fehlgeschlagen: Keine Trainings-Bilder');
    expect(errorMessage({ message: 'YAML-Generierung fehlgeschlagen', details: 'Keine Trainings-Bilder in /x' }))
      .toBe('YAML-Generierung fehlgeschlagen: Keine Trainings-Bilder in /x');
    expect(errorMessage({ data: { error: 'CUDA out of memory' } })).toBe('CUDA out of memory');
    expect(errorMessage(null)).toBeNull();
    expect(errorMessage({})).toBeNull();
  });
});
