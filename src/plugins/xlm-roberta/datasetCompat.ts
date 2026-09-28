// XLM-RoBERTa Dataset-Kompatibilitäts-Plugin
// Erkannte Formate: .json, .jsonl, .csv, .parquet, .tsv, .txt

import type { DatasetCompatPlugin, DatasetCompatResult, DatasetCheckInput } from '../datasetCompatHelpers';
import { compatResult, fileResult, worstLevel } from '../datasetCompatHelpers';
import { checkTextFormats } from '../textFormatCompat';

const M = 'datasetCompat.msg';
const MODEL = 'XLM-RoBERTa';

export const xlmRobertaCompatPlugin: DatasetCompatPlugin = {
  modelPluginId: 'xlm-roberta',

  supportedTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredType:  'flat_file',

  checkExtensions(extensions: string[]): DatasetCompatResult {
    return checkTextFormats(extensions, MODEL);
  },

  checkDataset(info: DatasetCheckInput): DatasetCompatResult {
    const { type, extensions, pairingStatus } = info;

    // Typ-basierte Bewertung zuerst
    if (type === 'yolo_bbox' || type === 'coco_json' || type === 'pascal_voc') {
      return compatResult('bad', [], { key: `${M}.xlm.imageDataset` }, { key: `${M}.textHint` });
    }

    if (type === 'audio_transcript' || type === 'common_voice') {
      return compatResult('bad', [], { key: `${M}.xlm.audioDataset` }, { key: `${M}.xlm.audioHint` });
    }

    // Ordner-Klassen koennten Text sein; nur Bilder sind sicher falsch.
    if (type === 'folder_class' && info.modalities.includes('image')) {
      return compatResult('bad', [], { key: `${M}.xlm.imageClassDataset` });
    }

    if (type === 'pre_split') {
      const result = checkTextFormats(extensions, MODEL);
      return compatResult(result.overallLevel, result.fileResults,
        { key: `${M}.preSplit`, params: { rest: result.summaryMsg! } }, result.hintMsg);
    }

    if (type === 'multi_shard') {
      return compatResult('perfect',
        [fileResult('.parquet', 'perfect', { key: `${M}.xlm.multiShardReason` })],
        { key: `${M}.xlm.multiShard` });
    }

    // Pairing-Warnung bei paired types
    if (pairingStatus && !pairingStatus.is_paired) {
      const base = checkTextFormats(extensions, MODEL);
      return compatResult(worstLevel([base.overallLevel, 'warning']), base.fileResults,
        { key: `${M}.withOrphans`, params: { rest: base.summaryMsg!, count: pairingStatus.orphan_primaries.length } },
        base.hintMsg);
    }

    return checkTextFormats(extensions, MODEL);
  },
};
