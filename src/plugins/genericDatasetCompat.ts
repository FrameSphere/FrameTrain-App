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
/** CoNLL-Text fuer NER/POS: ein Token pro Zeile, Tag in der letzten Spalte. */
const CONLL_EXTS = ['.conll', '.conllu', '.iob', '.bio', '.txt'];

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
    case 'speech_recognition':
      return { exts: [...AUDIO_EXTS, '.parquet'], missing: 'speech' };
    case 'token_classification':
      return { exts: ['.jsonl', '.json', '.parquet', ...CONLL_EXTS], missing: 'tokens' };
    case 'sentence_embedding':
      return { exts: TABLE_EXTS, missing: 'pairs' };
    case 'text_to_image_lora':
      return { exts: [...IMAGE_EXTS, '.parquet'], missing: 'textToImage' };
    case 'vision_language':
      return { exts: [...IMAGE_EXTS, '.parquet'], missing: 'visionLanguage' };
    case 'causal_lm':
      // Chat-/Frage-Antwort-Tabellen oder reiner Text fuer weiteres Vortraining.
      return { exts: [...TABLE_EXTS, '.txt', '.md'], missing: 'llm' };
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

  // YOLO-cls liest Ordner pro Klasse — dort gibt es keine Label-Dateien.
  const yoloClassify = plugin.taskType === 'detect' && info?.type === 'folder_class' && supported.includes('folder_class');
  const required = requiredFiles(yoloClassify ? 'image_classification' : plugin.taskType);
  // CoNLL-Ordner erkennt die Dataset-Analyse nicht als Typ — fuer NER sind
  // sie aber genau das richtige Format.
  if (plugin.taskType === 'token_classification' && exts.some(e => ['.conll', '.conllu', '.iob', '.bio'].includes(e))) {
    return compatResult('perfect', [], { key: `${M}.conllFits`, params: { model } });
  }

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
