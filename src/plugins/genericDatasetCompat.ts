// Kompatibilitaetspruefung fuer Plugins ohne eigene datasetCompat.ts.
//
// Vorher zeigte die Trainingsseite fuer YOLO, Bild-, Audio-, Seq2Seq- und
// Canvas-Modelle immer "Geeignet — Kompatibilitaet noch unbekannt", auch fuer
// ein YOLO-Dataset ganz ohne Label-Dateien. Hier entscheiden die Typen, die das
// Plugin wirklich laden kann (supportedDatasetTypes), plus die Dateien, ohne
// die das Training nichts lernt.

import type { CompatLevel, DatasetCheckInput, DatasetCompatResult, DatasetType } from './datasetCompatHelpers';
import { compatResult, typeMsg } from './datasetCompatHelpers';

export interface CompatPluginInfo {
  name: string;
  taskType: string;
  supportedDatasetTypes?: DatasetType[];
  preferredDatasetType?: DatasetType;
}

const M = 'datasetCompat.msg';

const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff'];
const AUDIO_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'];
const VIDEO_EXTS = ['.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'];
const TABLE_EXTS = ['.csv', '.tsv', '.json', '.jsonl', '.parquet'];

/** Welche Dateien braucht die Aufgabe mindestens? `missing` ist der Meldungsschluessel. */
function requiredFiles(taskType: string): { exts: string[]; missing: string } | null {
  switch (taskType) {
    case 'detect':
      return { exts: ['.txt', '.xml'], missing: 'detect' };
    case 'hf_image_classification':
    case 'image_classification':
      return { exts: [...IMAGE_EXTS, '.parquet'], missing: 'image' };
    case 'audio_classification':
      return { exts: [...AUDIO_EXTS, '.parquet'], missing: 'audio' };
    case 'video_classification':
      return { exts: VIDEO_EXTS, missing: 'video' };
    case 'seq2seq':
    case 'seq_classification':
      return { exts: TABLE_EXTS, missing: 'table' };
    default:
      return null;
  }
}

export function genericDatasetCompat(plugin: CompatPluginInfo, info: DatasetCheckInput | null, extensions: string[]): DatasetCompatResult {
  const exts = (info?.extensions?.length ? info.extensions : extensions).map(e => e.toLowerCase());
  const supported = plugin.supportedDatasetTypes ?? [];
  const model = plugin.name;

  const required = requiredFiles(plugin.taskType);
  if (required && exts.length > 0 && !required.exts.some(e => exts.includes(e))) {
    return compatResult('bad', [], { key: `${M}.missing.${required.missing}` },
      { key: `${M}.cannotTrain`, params: { model } });
  }

  if (!info || info.type === 'unknown') {
    return compatResult(
      supported.length && info?.type === 'unknown' ? 'warning' : 'ok', [],
      info?.type === 'unknown'
        ? { key: `${M}.typeUnknown`, params: { model } }
        : { key: `${M}.formatsFit`, params: { model } });
  }

  if (supported.length && !supported.includes(info.type)) {
    return compatResult('bad', [],
      { key: `${M}.typeUnsupported`, params: { type: typeMsg(info.type), model } },
      { key: `${M}.suitableTypes`, params: { types: supported.map(typeMsg) } });
  }

  const pairing = info.pairingStatus;
  if (plugin.taskType === 'detect' && pairing && !pairing.is_paired && pairing.primary_count > 0) {
    const level: CompatLevel = pairing.paired_count === 0 ? 'bad' : 'warning';
    return compatResult(level, [],
      { key: `${M}.labelsPartial`, params: { paired: pairing.paired_count, total: pairing.primary_count } },
      { key: `${M}.unlabeledBackground` });
  }

  return compatResult(plugin.preferredDatasetType === info.type ? 'perfect' : 'ok', [],
    { key: `${M}.typeFits`, params: { type: typeMsg(info.type), model } });
}
