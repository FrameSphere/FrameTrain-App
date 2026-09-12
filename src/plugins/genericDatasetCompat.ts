// Kompatibilitaetspruefung fuer Plugins ohne eigene datasetCompat.ts.
//
// Vorher zeigte die Trainingsseite fuer YOLO, Bild-, Audio-, Seq2Seq- und
// Canvas-Modelle immer "Geeignet — Kompatibilitaet noch unbekannt", auch fuer
// ein YOLO-Dataset ganz ohne Label-Dateien. Hier entscheiden die Typen, die das
// Plugin wirklich laden kann (supportedDatasetTypes), plus die Dateien, ohne
// die das Training nichts lernt.

import type { CompatLevel, DatasetCheckInput, DatasetCompatResult, DatasetType } from './datasetCompatHelpers';
import { DATASET_TYPE_LABELS } from './datasetCompatHelpers';

export interface CompatPluginInfo {
  name: string;
  taskType: string;
  supportedDatasetTypes?: DatasetType[];
  preferredDatasetType?: DatasetType;
}

const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tif', '.tiff'];
const AUDIO_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'];
const TABLE_EXTS = ['.csv', '.tsv', '.json', '.jsonl', '.parquet'];

/** Welche Dateien braucht die Aufgabe mindestens? */
function requiredFiles(taskType: string): { exts: string[]; missing: string } | null {
  switch (taskType) {
    case 'detect':
      return { exts: ['.txt', '.xml'], missing: 'Keine Label-Dateien (.txt oder Pascal-VOC-.xml) gefunden — ohne Labels lernt ein Objekterkennungs-Modell nichts.' };
    case 'hf_image_classification':
    case 'image_classification':
      return { exts: [...IMAGE_EXTS, '.parquet'], missing: 'Keine Bilddateien (oder Parquet mit Bildspalte) gefunden.' };
    case 'audio_classification':
      return { exts: [...AUDIO_EXTS, '.parquet'], missing: 'Keine Audiodateien (oder Parquet mit Audiospalte) gefunden.' };
    case 'seq2seq':
    case 'seq_classification':
      return { exts: TABLE_EXTS, missing: 'Keine Tabellendateien (.csv, .tsv, .json, .jsonl, .parquet) gefunden.' };
    default:
      return null;
  }
}

const typeLabel = (t: DatasetType) => DATASET_TYPE_LABELS[t]?.label ?? t;

export function genericDatasetCompat(plugin: CompatPluginInfo, info: DatasetCheckInput | null, extensions: string[]): DatasetCompatResult {
  const exts = (info?.extensions?.length ? info.extensions : extensions).map(e => e.toLowerCase());
  const supported = plugin.supportedDatasetTypes ?? [];

  const required = requiredFiles(plugin.taskType);
  if (required && exts.length > 0 && !required.exts.some(e => exts.includes(e))) {
    return { overallLevel: 'bad', fileResults: [], summary: required.missing, hint: `${plugin.name} kann dieses Dataset nicht trainieren.` };
  }

  if (!info || info.type === 'unknown') {
    return {
      overallLevel: supported.length && info?.type === 'unknown' ? 'warning' : 'ok',
      fileResults: [],
      summary: info?.type === 'unknown'
        ? `Dataset-Typ nicht erkannt — ${plugin.name} kann es eventuell nicht lesen.`
        : `Dateiformate passen zu ${plugin.name}.`,
    };
  }

  if (supported.length && !supported.includes(info.type)) {
    return {
      overallLevel: 'bad',
      fileResults: [],
      summary: `${typeLabel(info.type)} kann ${plugin.name} nicht lesen.`,
      hint: `Geeignet: ${supported.map(typeLabel).join(', ')}`,
    };
  }

  const pairing = info.pairingStatus;
  if (plugin.taskType === 'detect' && pairing && !pairing.is_paired && pairing.primary_count > 0) {
    const level: CompatLevel = pairing.paired_count === 0 ? 'bad' : 'warning';
    return {
      overallLevel: level,
      fileResults: [],
      summary: `Nur ${pairing.paired_count} von ${pairing.primary_count} Bildern haben ein Label.`,
      hint: 'Bilder ohne Label zaehlen im Training als Hintergrund.',
    };
  }

  return {
    overallLevel: plugin.preferredDatasetType === info.type ? 'perfect' : 'ok',
    fileResults: [],
    summary: `${typeLabel(info.type)} passt zu ${plugin.name}.`,
  };
}
