// HF Encoder Dataset-Kompatibilitäts-Plugin
// Basierend auf denselben Formaten wie das Backend-seq_classification Plugin.

import type { DatasetCompatPlugin, DatasetCompatResult, DatasetCheckInput } from '../datasetCompatHelpers';
import { compatResult, fileResult, worstLevel } from '../datasetCompatHelpers';
import { checkTextFormats } from '../textFormatCompat';

const M = 'datasetCompat.msg';
const MODEL = { key: `${M}.encoder.modelName` };

export const hfEncoderCompatPlugin: DatasetCompatPlugin = {
  modelPluginId: 'hf-encoder',

  supportedTypes: ['flat_file', 'pre_split', 'multi_shard'],
  preferredType:  'flat_file',

  checkExtensions(extensions: string[]): DatasetCompatResult {
    return checkTextFormats(extensions, MODEL);
  },

  checkDataset(info: DatasetCheckInput): DatasetCompatResult {
    const { type, extensions, pairingStatus } = info;

    if (['yolo_bbox', 'coco_json', 'pascal_voc'].includes(type)) {
      return compatResult('bad', [], { key: `${M}.encoder.imageDataset` }, { key: `${M}.textHint` });
    }
    if (['audio_transcript', 'common_voice'].includes(type)) {
      return compatResult('bad', [], { key: `${M}.encoder.audioDataset` });
    }
    if (type === 'multi_shard') {
      return compatResult('perfect',
        [fileResult('.parquet', 'perfect', { key: `${M}.encoder.multiShardReason` })],
        { key: `${M}.encoder.multiShard` });
    }
    if (pairingStatus && !pairingStatus.is_paired) {
      const base = checkTextFormats(extensions, MODEL);
      return compatResult(worstLevel([base.overallLevel, 'warning']), base.fileResults,
        { key: `${M}.withOrphans`, params: { rest: base.summaryMsg!, count: pairingStatus.orphan_primaries.length } },
        base.hintMsg);
    }
    return checkTextFormats(extensions, MODEL);
  },
};
