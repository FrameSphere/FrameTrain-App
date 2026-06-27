// XLM-RoBERTa Dataset-Kompatibilitäts-Plugin
// Erkannte Formate: .json, .jsonl, .csv, .parquet, .tsv, .txt

import type {
  DatasetCompatPlugin, DatasetCompatResult, DatasetCheckInput,
  FileCompatResult, CompatLevel,
} from '../datasetCompatHelpers';
import { worstLevel } from '../datasetCompatHelpers';

const FORMAT_RULES: Record<string, { level: CompatLevel; reason: string }> = {
  '.jsonl':   { level: 'perfect', reason: 'JSONL ist das bevorzugte Format – jede Zeile ein JSON-Objekt mit text + label.' },
  '.json':    { level: 'perfect', reason: 'JSON-Arrays mit {text, label} Einträgen werden vollständig unterstützt.' },
  '.csv':     { level: 'perfect', reason: 'CSV mit text/label Spalten funktioniert direkt.' },
  '.parquet': { level: 'perfect', reason: 'Parquet wird von HuggingFace datasets nativ unterstützt.' },
  '.tsv':     { level: 'ok',      reason: 'TSV (tab-separated) funktioniert, muss in CSV konvertiert werden.' },
  '.txt':     { level: 'warning', reason: 'Reine Textdateien brauchen eigenes Parsing-Skript.' },
  '.arrow':   { level: 'ok',      reason: 'Arrow-Format wird von HuggingFace datasets unterstützt.' },
};

function checkExts(extensions: string[]): DatasetCompatResult {
  if (!extensions || extensions.length === 0) {
    return {
      overallLevel: 'warning',
      fileResults:  [],
      summary:      'Keine Dateien gefunden – Dataset scheint leer zu sein.',
      hint:         'Füge .jsonl, .json, .csv oder .parquet Dateien hinzu.',
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
}

export const xlmRobertaCompatPlugin: DatasetCompatPlugin = {
  modelPluginId: 'xlm-roberta',

  supportedTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredType:  'flat_file',

  checkExtensions(extensions: string[]): DatasetCompatResult {
    return checkExts(extensions);
  },

  checkDataset(info: DatasetCheckInput): DatasetCompatResult {
    const { type, extensions, pairingStatus } = info;

    // Typ-basierte Bewertung zuerst
    if (type === 'yolo_bbox' || type === 'coco_json' || type === 'pascal_voc') {
      return {
        overallLevel: 'bad',
        fileResults:  [],
        summary:      'Bild-Dataset erkannt – XLM-RoBERTa braucht Textdaten.',
        hint:         'Verwende .jsonl, .csv oder .parquet mit Textspalten.',
      };
    }

    if (type === 'audio_transcript' || type === 'common_voice') {
      return {
        overallLevel: 'bad',
        fileResults:  [],
        summary:      'Audio-Dataset erkannt – XLM-RoBERTa ist ein Text-Modell.',
        hint:         'Für Spracherkennung nutze ein Whisper-Plugin.',
      };
    }

    if (type === 'folder_class') {
      // Könnte Text-Klassifikation sein wenn keine Bilder
      if (info.modalities.includes('image')) {
        return {
          overallLevel: 'bad',
          fileResults:  [],
          summary:      'Bild-Klassifikations-Dataset – nicht kompatibel mit XLM-RoBERTa.',
        };
      }
    }

    if (type === 'pre_split') {
      const result = checkExts(extensions);
      return {
        ...result,
        summary:  `Voraufgeteiltes Dataset. ${result.summary}`,
      };
    }

    if (type === 'multi_shard') {
      return {
        overallLevel: 'perfect',
        fileResults:  [{ extension: '.parquet', level: 'perfect', reason: 'Multi-Shard Parquet – ideal für große Datasets.' }],
        summary:      'Multi-Shard Parquet wird nativ unterstützt.',
      };
    }

    // Pairing-Warnung bei paired types
    if (pairingStatus && !pairingStatus.is_paired) {
      const base = checkExts(extensions);
      return {
        ...base,
        overallLevel: worstLevel([base.overallLevel, 'warning']),
        summary:      `${base.summary} ${pairingStatus.orphan_primaries.length} Dateien ohne Partner.`,
      };
    }

    return checkExts(extensions);
  },
};
