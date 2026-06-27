// Dataset-Kompatibilitäts-System
//
// Um eine neue Modell-Familie zu unterstützen:
//   1. DatasetCompatPlugin implementieren (in plugins/<name>/datasetCompat.ts)
//   2. In COMPAT_PLUGINS eintragen – fertig.

// ── Re-export für Abwärtskompatibilität ────────────────────────────────────
export type {
  CompatLevel,
  FileCompatResult,
  DatasetCompatResult,
  DatasetCompatPlugin,
  DatasetType,
  DatasetAnalysis,
  PairingStatus,
  DatasetCheckInput,
  DatasetSchema,
  Modality,
} from './datasetCompatHelpers';

export {
  worstLevel,
  LEVEL_META,
  DATASET_TYPE_LABELS,
  analysisToCheckInput,
} from './datasetCompatHelpers';

// ── Registry ───────────────────────────────────────────────────────────────

import type { DatasetCompatPlugin, DatasetCompatResult, DatasetAnalysis } from './datasetCompatHelpers';
import { analysisToCheckInput } from './datasetCompatHelpers';
import { xlmRobertaCompatPlugin } from './xlm-roberta/datasetCompat';
import { hfEncoderCompatPlugin } from './hf-encoder/datasetCompat';

const COMPAT_PLUGINS: DatasetCompatPlugin[] = [
  xlmRobertaCompatPlugin,
  hfEncoderCompatPlugin,
  // Neue Modell-Familien hier eintragen:
  // bertCompatPlugin,
  // whisperCompatPlugin,
  // yolov8CompatPlugin,
];

/**
 * Prüft ob ein Dataset mit einem Modell kompatibel ist.
 * Bevorzugt die neue checkDataset-API (kennt Typ + Pairing),
 * fällt zurück auf checkExtensions wenn nicht vorhanden.
 *
 * @param modelPluginId  Die Plugin-ID des Modells (z.B. "xlm-roberta")
 * @param extensions     Dateiendungen im Dataset (lowercase, mit Punkt)
 * @param analysis       Optional: vollständige Analyse für neue API
 */
export function checkDatasetCompat(
  modelPluginId: string,
  extensions:    string[],
  analysis?:     DatasetAnalysis | null,
): DatasetCompatResult {
  const plugin = COMPAT_PLUGINS.find(p => p.modelPluginId === modelPluginId);

  if (!plugin) {
    return {
      overallLevel: 'ok',
      fileResults:  [],
      summary:      'Kompatibilität für dieses Modell noch unbekannt.',
      hint:         'Dieses Modell wird in einer späteren Version geprüft.',
    };
  }

  // Neue API bevorzugen
  if (plugin.checkDataset && analysis) {
    return plugin.checkDataset(analysisToCheckInput(analysis));
  }

  // Alte API als Fallback
  if (plugin.checkExtensions) {
    return plugin.checkExtensions(extensions);
  }

  // Plugin ohne Check-Funktion – sollte nicht vorkommen
  return {
    overallLevel: 'ok',
    fileResults:  [],
    summary:      'Keine Kompatibilitätsprüfung konfiguriert.',
  };
}
