// XLM-RoBERTa Dataset-Kompatibilitäts-Plugin
// Erkannte Formate: .json, .jsonl, .csv, .parquet, .tsv, .txt

import type { DatasetCompatPlugin, DatasetCompatResult, FileCompatResult, CompatLevel } from '../datasetCompat';
import { worstLevel } from '../datasetCompat';

const FORMAT_RULES: Record<string, { level: CompatLevel; reason: string }> = {
  '.jsonl': { level: 'perfect', reason: 'JSONL ist das bevorzugte Format – jede Zeile ein JSON-Objekt mit text + label.' },
  '.json':  { level: 'perfect', reason: 'JSON-Arrays mit {text, label} Einträgen werden vollständig unterstützt.' },
  '.csv':   { level: 'perfect', reason: 'CSV mit text/label Spalten funktioniert direkt.' },
  '.parquet': { level: 'perfect', reason: 'Parquet wird von HuggingFace datasets nativ unterstützt.' },
  '.tsv':   { level: 'ok', reason: 'TSV (tab-separated) funktioniert, muss in CSV konvertiert werden.' },
  '.txt':   { level: 'warning', reason: 'Reine Textdateien brauchen eigenes Parsing-Skript.' },
  '.arrow': { level: 'ok', reason: 'Arrow-Format wird von HuggingFace datasets unterstützt.' },
};

export const xlmRobertaCompatPlugin: DatasetCompatPlugin = {
  modelPluginId: 'xlm-roberta',

  checkExtensions(extensions: string[]): DatasetCompatResult {
    if (!extensions || extensions.length === 0) {
      return {
        overallLevel: 'warning',
        fileResults: [],
        summary: 'Keine Dateien gefunden – Dataset scheint leer zu sein.',
        hint: 'Füge .jsonl, .json, .csv oder .parquet Dateien hinzu.',
      };
    }

    const fileResults: FileCompatResult[] = extensions.map(ext => {
      const rule = FORMAT_RULES[ext.toLowerCase()];
      return rule
        ? { extension: ext, level: rule.level, reason: rule.reason }
        : { extension: ext, level: 'warning' as CompatLevel, reason: `Format "${ext}" wird nicht direkt erkannt – evtl. manuelles Parsing nötig.` };
    });

    const overallLevel = worstLevel(fileResults.map(r => r.level));

    const perfectCount = fileResults.filter(r => r.level === 'perfect').length;
    const summary = perfectCount > 0
      ? `${perfectCount} von ${fileResults.length} Formaten sind ideal für XLM-RoBERTa.`
      : overallLevel === 'ok'
        ? 'Nutzbar, aber nicht optimale Formate vorhanden.'
        : 'Einige Formate benötigen Aufbereitung.';

    return { overallLevel, fileResults, summary };
  },
};
