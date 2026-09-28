// Dateiformat-Pruefung fuer Text-Modelle (XLM-RoBERTa, HF-Encoder).
// Beide Plugins hatten dieselben Regeln als Kopie mit deutschen Saetzen.

import type { CompatLevel, DatasetCompatResult, FileCompatResult, MsgParam } from './datasetCompatHelpers';
import { compatResult, fileResult, worstLevel } from './datasetCompatHelpers';

const M = 'datasetCompat.msg';

const FORMAT_RULES: Record<string, CompatLevel> = {
  '.jsonl':   'perfect',
  '.json':    'perfect',
  '.csv':     'perfect',
  '.parquet': 'perfect',
  '.tsv':     'ok',
  '.txt':     'warning',
  '.arrow':   'ok',
};

/** Bewertet die Dateiendungen; `model` erscheint in der Zusammenfassung. */
export function checkTextFormats(extensions: string[], model: MsgParam): DatasetCompatResult {
  if (!extensions || extensions.length === 0) {
    return compatResult('warning', [], { key: `${M}.empty` }, { key: `${M}.emptyHint` });
  }

  const fileResults: FileCompatResult[] = extensions.map(ext => {
    const level = FORMAT_RULES[ext.toLowerCase()];
    return level
      ? fileResult(ext, level, { key: `${M}.format.${ext.toLowerCase().slice(1)}` })
      : fileResult(ext, 'warning', { key: `${M}.format.unknown`, params: { ext } });
  });

  const overallLevel = worstLevel(fileResults.map(r => r.level));
  const perfect = fileResults.filter(r => r.level === 'perfect').length;
  const summary = perfect > 0
    ? { key: `${M}.idealFormats`, params: { perfect, total: fileResults.length, model } }
    : { key: overallLevel === 'ok' ? `${M}.usable` : `${M}.needsWork` };
  return compatResult(overallLevel, fileResults, summary);
}
