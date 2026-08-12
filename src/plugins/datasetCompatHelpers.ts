// Dataset-Compat Hilfsfunktionen + Typen
//
// Plugin-eigene datasetCompat.ts Dateien importieren von hier statt von ../datasetCompat
// (verhindert zirkuläre Abhängigkeiten)

// ══════════════════════════════════════════════════════════════════
// DATASET TYPE SYSTEM (spiegelt Rust DatasetType enum)
// ══════════════════════════════════════════════════════════════════

/** Spiegelt dataset_manager.rs::DatasetType (serde snake_case) */
export type DatasetType =
  | 'flat_file'        // .jsonl .json .csv .parquet .tsv – eine oder mehrere Dateien
  | 'yolo_bbox'        // images/ + labels/ Ordner, gleiche Basenames
  | 'coco_json'        // annotations.json + images/
  | 'pascal_voc'       // images/ + annotations/*.xml, gleiche Basenames
  | 'folder_class'     // class_name/image.ext Struktur
  | 'audio_transcript' // .wav/.mp3 + .txt gleiche Basenames
  | 'common_voice'     // clips/*.mp3 + metadata.tsv
  | 'pre_split'        // train/ val/ test/ bereits vorhanden
  | 'multi_shard'      // part-XXXX.parquet oder train-XXXX-of-XXXX.parquet
  | 'unknown';

/** Modality – für zukünftige Plugin-Erweiterungen */
export type Modality = 'text' | 'image' | 'audio' | 'video' | 'multimodal';

/** Schema-Hint – spezifische Struktur innerhalb eines Typs */
export type DatasetSchema =
  | 'instruction_response' // { instruction, response }
  | 'text_label'           // { text, label }
  | 'qa'                   // { question, answer }
  | 'token_classification' // { tokens[], labels[] }
  | 'seq2seq'              // { source, target }
  | 'causal_lm'            // reiner Text
  | 'unknown';

/** Pairing-Status (spiegelt Rust PairingStatus) */
export interface PairingStatus {
  is_paired:          boolean;
  primary_count:      number;
  paired_count:       number;
  orphan_primaries:   string[];
  orphan_secondaries: string[];
}

/** Vollständige Analyse eines Datasets (Rückgabe von analyze_dataset_path) */
export interface DatasetAnalysis {
  detected_type:  DatasetType;
  confidence:     number;           // 0–100
  pairing_status: PairingStatus | null;
  warnings:       string[];
  file_count:     number;
  dir_count:      number;
  extensions:     string[];
  schema_hint:    Record<string, unknown> | null;
}

/** Input für Plugin checkDataset() – alle Infos auf einen Blick */
export interface DatasetCheckInput {
  type:           DatasetType;
  extensions:     string[];
  modalities:     Modality[];
  pairingStatus:  PairingStatus | null;
  schema?:        DatasetSchema;
}

// ══════════════════════════════════════════════════════════════════
// COMPAT LEVEL SYSTEM (unverändert)
// ══════════════════════════════════════════════════════════════════

export type CompatLevel =
  | 'perfect'
  | 'ok'
  | 'warning'
  | 'bad';

export interface FileCompatResult {
  extension: string;
  level:     CompatLevel;
  reason:    string;
}

export interface DatasetCompatResult {
  overallLevel: CompatLevel;
  fileResults:  FileCompatResult[];
  summary:      string;
  hint?:        string;
}

// ══════════════════════════════════════════════════════════════════
// PLUGIN INTERFACE (erweitert, rückwärtskompatibel)
// ══════════════════════════════════════════════════════════════════

export interface DatasetCompatPlugin {
  modelPluginId: string;

  /**
   * Alte API – bleibt für Rückwärtskompatibilität.
   * Wird aufgerufen wenn checkDataset nicht definiert ist.
   */
  checkExtensions?: (extensions: string[]) => DatasetCompatResult;

  /**
   * Neue API – kennt Typ + Schema + Pairing-Status.
   * Wird bevorzugt, wenn vorhanden.
   */
  checkDataset?: (info: DatasetCheckInput) => DatasetCompatResult;

  /** Welche DatasetTypes kann dieses Modell trainieren? */
  supportedTypes?: DatasetType[];

  /** Bevorzugter Import-Typ (für Vorschlag im Import-Modal) */
  preferredType?: DatasetType;
}

// ══════════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════════

/** Gibt die "schlechteste" Bewertung aus einer Liste zurück */
export function worstLevel(levels: CompatLevel[]): CompatLevel {
  const order: CompatLevel[] = ['bad', 'warning', 'ok', 'perfect'];
  let worst: CompatLevel = 'perfect';
  for (const l of levels) {
    if (order.indexOf(l) < order.indexOf(worst)) worst = l;
  }
  return worst;
}

/**
 * `labelKey` ist die Quelle für die Anzeige – `label` bleibt als deutscher
 * Fallback für Kontexte ohne Übersetzungsfunktion (Tests, Logs) bestehen.
 */
export const LEVEL_META: Record<CompatLevel, { label: string; labelKey: string; color: string; bg: string; border: string; icon: 'check' | 'info' | 'alert' | 'ban' }> = {
  perfect: { label: 'Perfekt geeignet', labelKey: 'datasetCompat.levels.perfect', color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: 'check' },
  ok:      { label: 'Geeignet',         labelKey: 'datasetCompat.levels.ok',      color: 'text-blue-400',    bg: 'bg-blue-500/10',    border: 'border-blue-500/30',    icon: 'info' },
  warning: { label: 'Bedingt geeignet', labelKey: 'datasetCompat.levels.warning', color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   icon: 'alert' },
  bad:     { label: 'Nicht geeignet',   labelKey: 'datasetCompat.levels.bad',     color: 'text-red-400',     bg: 'bg-red-500/10',     border: 'border-red-500/30',     icon: 'ban' },
};

/** Lesbare Labels für DatasetType */
export const DATASET_TYPE_LABELS: Record<DatasetType, { label: string; icon: string; color: string; modality: Modality }> = {
  flat_file:        { label: 'Flat File',           icon: '📄', color: 'text-violet-400',  modality: 'text'  },
  yolo_bbox:        { label: 'YOLO Bounding Box',   icon: '🎯', color: 'text-orange-400',  modality: 'image' },
  coco_json:        { label: 'COCO JSON',           icon: '🖼️', color: 'text-amber-400',   modality: 'image' },
  pascal_voc:       { label: 'Pascal VOC',          icon: '🗂️', color: 'text-yellow-400',  modality: 'image' },
  folder_class:     { label: 'Ordner-Klassen',      icon: '📁', color: 'text-blue-400',    modality: 'image' },
  audio_transcript: { label: 'Audio + Transkript',  icon: '🎤', color: 'text-cyan-400',    modality: 'audio' },
  common_voice:     { label: 'Common Voice',        icon: '🔊', color: 'text-teal-400',    modality: 'audio' },
  pre_split:        { label: 'Voraufgeteilt',       icon: '✂️', color: 'text-emerald-400', modality: 'text'  },
  multi_shard:      { label: 'Multi-Shard Parquet', icon: '🧉', color: 'text-indigo-400',  modality: 'text'  },
  unknown:          { label: 'Unbekannt',           icon: '❓', color: 'text-gray-400',    modality: 'text'  },
};

/**
 * Konvertiert DatasetAnalysis zu DatasetCheckInput für Plugins.
 */
export function analysisToCheckInput(analysis: DatasetAnalysis): DatasetCheckInput {
  const typeMeta = DATASET_TYPE_LABELS[analysis.detected_type];
  return {
    type:          analysis.detected_type,
    extensions:    analysis.extensions,
    modalities:    [typeMeta?.modality ?? 'text'],
    pairingStatus: analysis.pairing_status,
    schema:        'unknown',
  };
}
