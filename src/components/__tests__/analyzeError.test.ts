// Fehler-Kategorisierung im Trainings-Dashboard.
// Regressionen aus dem E2E-Test vom 18.08.2026.

import { describe, it, expect } from 'vitest';
import { analyzeError } from '../TrainingDashboard';

// Der Übersetzer gibt den Key zurück – so sieht man im Test direkt die Kategorie.
const t = (key: string) => key;
const cat = (msg: string) => analyzeError(msg, t).category;

describe('analyzeError – Python-Fehler schlagen die Pfad-Heuristik', () => {
  it('NameError in einem Dev-Script ist ein Code-Fehler, kein Dataset-Fehler', () => {
    // Frueher gewann e.includes('dataset'): Der Nutzer bekam
    // "Dataset / Pfad Fehler" und suchte den Fehler an der falschen Stelle.
    const msg = [
      'Script beendet mit Exit-Code 1.',
      '  File "/Users/x/dev_scripts/dev_abc.py", line 38, in <module>',
      '    dataset = load_frametrain_dataset(DATASET_PATH)',
      "NameError: name 'AutoTokenizer' is not defined",
    ].join('\n');
    expect(cat(msg)).toBe('code');
  });

  it('IndentationError bleibt ein Code-Fehler, auch mit DATASET_PATH im Traceback', () => {
    const msg = 'File "dev.py", line 40\n    # DATASET_PATH\nIndentationError: expected an indented block';
    expect(cat(msg)).toBe('code');
  });

  it('Ein blosses "path" im Traceback macht daraus keinen Dataset-Fehler', () => {
    expect(cat('Traceback: sys.path manipuliert, Abbruch ohne weitere Angabe')).not.toBe('dataset');
  });
});

describe('analyzeError – echte Kategorien bleiben erhalten', () => {
  it('FileNotFoundError auf dem Dataset-Pfad', () => {
    expect(cat('FileNotFoundError: DATASET_PATH existiert nicht: /tmp/weg')).toBe('dataset');
  });

  it('ModuleNotFoundError ist ein Paket-Problem', () => {
    expect(cat("ModuleNotFoundError: No module named 'transformers'")).toBe('packages');
  });

  it('Out of Memory', () => {
    expect(cat('RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB')).toBe('memory');
  });

  it('mse_loss bleibt ein Label-Problem, kein Geraete-Problem', () => {
    expect(cat('RuntimeError: mse_loss_out_mps: only defined for floating types')).toBe('labels');
  });

  it('Echter Geraetefehler wird als solcher erkannt', () => {
    expect(cat('RuntimeError: device-side assert triggered')).toBe('cuda');
  });

  it('Nicht unterstuetzte Architektur', () => {
    expect(cat('Die Modell-Architektur gpt2 wird noch nicht unterstützt')).toBe('architecture');
  });
});

describe('Netzwerk- und Zertifikatsfehler', () => {
  it('SSL-Zertifikatsfehler ist kein Paket-Problem', () => {
    // Beim Bild-Training laedt torchvision Gewichte nach. Das Wort
    // "torchvision" im Traceback fuehrte zu "Fehlende Python-Pakete"
    // samt pip-install-Rat — der nichts half.
    const msg = [
      'URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:',
      'unable to get local issuer certificate (_ssl.c:993)>',
      '  File ".../torchvision/models/_api.py", line 63, in load_state_dict',
    ].join('\n');
    expect(cat(msg)).toBe('network');
  });

  it('Verbindungsabbruch wird als Netzwerkfehler erkannt', () => {
    expect(cat('ConnectionError: Max retries exceeded with url: /models/resnet50.pth')).toBe('network');
  });

  it('Echtes Paket-Problem bleibt Paket-Problem', () => {
    expect(cat("ModuleNotFoundError: No module named 'torchvision'")).toBe('packages');
  });
});
