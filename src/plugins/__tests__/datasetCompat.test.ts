// Dataset-Compat Tests – FrameTrain Plugin-System
// Ausführen: npx vitest run src/plugins/__tests__/datasetCompat.test.ts

import { describe, it, expect } from 'vitest';
import { xlmRobertaCompatPlugin } from '../xlm-roberta/datasetCompat';
import { hfEncoderCompatPlugin } from '../hf-encoder/datasetCompat';
import { checkDatasetCompat, worstLevel, LEVEL_META } from '../datasetCompat';
import type { CompatLevel } from '../datasetCompat';

// ─────────────────────────────────────────────────────────────────────────────
// worstLevel() Helper
// ─────────────────────────────────────────────────────────────────────────────

describe('worstLevel()', () => {
  it('["perfect"] → "perfect"', () => {
    expect(worstLevel(['perfect'])).toBe('perfect');
  });

  it('["perfect", "ok"] → "ok"', () => {
    expect(worstLevel(['perfect', 'ok'])).toBe('ok');
  });

  it('["perfect", "ok", "warning"] → "warning"', () => {
    expect(worstLevel(['perfect', 'ok', 'warning'])).toBe('warning');
  });

  it('["perfect", "warning", "bad"] → "bad"', () => {
    expect(worstLevel(['perfect', 'warning', 'bad'])).toBe('bad');
  });

  it('["bad"] → "bad"', () => {
    expect(worstLevel(['bad'])).toBe('bad');
  });

  it('Reihenfolge egal: ["bad", "perfect"] → "bad"', () => {
    expect(worstLevel(['bad', 'perfect'])).toBe('bad');
  });

  it('[] leeres Array → "perfect" (kein schlimmster Wert)', () => {
    expect(worstLevel([])).toBe('perfect');
  });

  it('Alle vier Level → "bad"', () => {
    expect(worstLevel(['perfect', 'ok', 'warning', 'bad'])).toBe('bad');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// LEVEL_META – alle CompatLevel abgedeckt
// ─────────────────────────────────────────────────────────────────────────────

describe('LEVEL_META', () => {
  const levels: CompatLevel[] = ['perfect', 'ok', 'warning', 'bad'];

  it.each(levels)('Level "%s" hat alle Pflichtfelder', (level) => {
    const meta = LEVEL_META[level];
    expect(meta).toBeDefined();
    expect(meta.label).toBeTruthy();
    expect(meta.color).toBeTruthy();
    expect(meta.bg).toBeTruthy();
    expect(meta.border).toBeTruthy();
    expect(meta.icon).toBeTruthy();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// xlmRobertaCompatPlugin
// ─────────────────────────────────────────────────────────────────────────────

describe('xlmRobertaCompatPlugin – modelPluginId', () => {
  it('modelPluginId ist exakt "xlm-roberta"', () => {
    expect(xlmRobertaCompatPlugin.modelPluginId).toBe('xlm-roberta');
  });
});

describe('xlmRobertaCompatPlugin – leeres Dataset', () => {
  it('[] → overallLevel: "warning"', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions([]);
    expect(r.overallLevel).toBe('warning');
  });

  it('[] → fileResults ist leer', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions([]);
    expect(r.fileResults).toHaveLength(0);
  });

  it('[] → summary ist nicht leer', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions([]);
    expect(r.summary.length).toBeGreaterThan(0);
  });

  it('[] → hint enthält Empfehlung (.jsonl/.csv o.ä.)', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions([]);
    expect(r.hint).toBeTruthy();
  });
});

describe('xlmRobertaCompatPlugin – perfect-Formate', () => {
  it('[".jsonl"] → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.jsonl']);
    expect(r.overallLevel).toBe('perfect');
    expect(r.fileResults[0].level).toBe('perfect');
  });

  it('[".json"] → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.json']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('[".csv"] → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.csv']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('[".parquet"] → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.parquet']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('[".csv", ".json"] → overall perfect, 2 fileResults', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.csv', '.json']);
    expect(r.overallLevel).toBe('perfect');
    expect(r.fileResults).toHaveLength(2);
  });

  it('summary enthält "2 von 2" wenn beide perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.csv', '.json']);
    expect(r.summary).toContain('2 von 2');
  });
});

describe('xlmRobertaCompatPlugin – ok-Formate', () => {
  it('[".tsv"] → ok', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.tsv']);
    expect(r.overallLevel).toBe('ok');
    expect(r.fileResults[0].level).toBe('ok');
  });

  it('[".arrow"] → ok', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.arrow']);
    expect(r.overallLevel).toBe('ok');
  });

  it('[".jsonl", ".tsv"] → overall ok (worstLevel)', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.jsonl', '.tsv']);
    expect(r.overallLevel).toBe('ok');
  });
});

describe('xlmRobertaCompatPlugin – warning-Formate', () => {
  it('[".txt"] → warning', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.txt']);
    expect(r.overallLevel).toBe('warning');
    expect(r.fileResults[0].level).toBe('warning');
  });

  it('[".jsonl", ".txt"] → overall warning', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.jsonl', '.txt']);
    expect(r.overallLevel).toBe('warning');
  });

  it('[".exe"] → warning mit "nicht direkt erkannt" im reason', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.exe']);
    expect(r.overallLevel).toBe('warning');
    expect(r.fileResults[0].reason).toMatch(/nicht direkt erkannt/i);
  });
});

describe('xlmRobertaCompatPlugin – Groß-/Kleinschreibung', () => {
  it('[".CSV"] → wie ".csv" behandelt → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.CSV']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('[".JSONL"] → perfect', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.JSONL']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('[".TXT"] → warning', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.TXT']);
    expect(r.overallLevel).toBe('warning');
  });
});

describe('xlmRobertaCompatPlugin – fileResults Struktur', () => {
  it('Jedes fileResult hat extension, level und reason', () => {
    const r = xlmRobertaCompatPlugin.checkExtensions(['.csv', '.txt', '.exe']);
    for (const fr of r.fileResults) {
      expect(fr.extension).toBeTruthy();
      expect(['perfect', 'ok', 'warning', 'bad']).toContain(fr.level);
      expect(fr.reason.length).toBeGreaterThan(0);
    }
  });

  it('fileResults.length === extensions.length', () => {
    const exts = ['.csv', '.json', '.tsv', '.txt'];
    const r = xlmRobertaCompatPlugin.checkExtensions(exts);
    expect(r.fileResults).toHaveLength(exts.length);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// hfEncoderCompatPlugin
// ─────────────────────────────────────────────────────────────────────────────

describe('hfEncoderCompatPlugin – modelPluginId', () => {
  it('modelPluginId ist exakt "hf-encoder"', () => {
    expect(hfEncoderCompatPlugin.modelPluginId).toBe('hf-encoder');
  });

  it('modelPluginId ist NICHT "xlm-roberta"', () => {
    expect(hfEncoderCompatPlugin.modelPluginId).not.toBe('xlm-roberta');
  });
});

describe('hfEncoderCompatPlugin – Formate (analog xlm-roberta)', () => {
  it('[".jsonl"] → perfect', () => {
    expect(hfEncoderCompatPlugin.checkExtensions(['.jsonl']).overallLevel).toBe('perfect');
  });

  it('[".csv"] → perfect', () => {
    expect(hfEncoderCompatPlugin.checkExtensions(['.csv']).overallLevel).toBe('perfect');
  });

  it('[".tsv"] → ok', () => {
    expect(hfEncoderCompatPlugin.checkExtensions(['.tsv']).overallLevel).toBe('ok');
  });

  it('[".txt"] → warning', () => {
    expect(hfEncoderCompatPlugin.checkExtensions(['.txt']).overallLevel).toBe('warning');
  });

  it('[] → warning', () => {
    expect(hfEncoderCompatPlugin.checkExtensions([]).overallLevel).toBe('warning');
  });
});

describe('hfEncoderCompatPlugin – summary-Text', () => {
  it('summary enthält "Encoder-Modelle" (nicht "XLM-RoBERTa")', () => {
    const r = hfEncoderCompatPlugin.checkExtensions(['.csv']);
    expect(r.summary).toContain('Encoder-Modelle');
    expect(r.summary).not.toContain('XLM-RoBERTa');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// checkDatasetCompat() – zentrale Dispatch-Funktion
// ─────────────────────────────────────────────────────────────────────────────

describe('checkDatasetCompat() – Routing', () => {
  it('"xlm-roberta" → findet xlmRobertaCompatPlugin', () => {
    const r = checkDatasetCompat('xlm-roberta', ['.csv']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('"hf-encoder" → findet hfEncoderCompatPlugin', () => {
    const r = checkDatasetCompat('hf-encoder', ['.csv']);
    expect(r.overallLevel).toBe('perfect');
  });

  it('Unbekannte Plugin-ID → Fallback, kein Crash', () => {
    const r = checkDatasetCompat('unbekanntes-plugin', ['.csv']);
    expect(r).toBeDefined();
    expect(r.overallLevel).toBe('ok');
  });

  it('Unbekannte Plugin-ID → summary signalisiert Unklarheit', () => {
    const r = checkDatasetCompat('unbekanntes-plugin', ['.csv']);
    expect(r.summary.length).toBeGreaterThan(0);
  });

  it('Tippfehler "xlm-roberta " (Leerzeichen) → Fallback, nicht xlm-roberta-Plugin', () => {
    // Tippfehler → kein Plugin gefunden → Fallback
    const r = checkDatasetCompat('xlm-roberta ', ['.csv']);
    expect(r.overallLevel).toBe('ok');
  });
});
