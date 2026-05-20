// Dataset-Kompatibilitäts-System
//
// Um eine neue Modell-Familie zu unterstützen:
//   1. DatasetCompatPlugin implementieren
//   2. In COMPAT_PLUGINS eintragen – fertig.
//
// Typen und Hilfsfunktionen liegen in datasetCompatHelpers.ts um
// zirkuläre Imports zu vermeiden (Plugin-Dateien importieren von dort,
// diese Datei importiert die Plugins – das wäre ein Kreis).

// ── Re-export für Abwärtskompatibilität ────────────────────────────────────
export type {
  CompatLevel,
  FileCompatResult,
  DatasetCompatResult,
  DatasetCompatPlugin,
} from './datasetCompatHelpers';

export { worstLevel, LEVEL_META } from './datasetCompatHelpers';

// ── Registry ───────────────────────────────────────────────────────────────

import type { DatasetCompatPlugin, DatasetCompatResult } from './datasetCompatHelpers';
import { xlmRobertaCompatPlugin } from './xlm-roberta/datasetCompat';
import { hfEncoderCompatPlugin } from './hf-encoder/datasetCompat';

const COMPAT_PLUGINS: DatasetCompatPlugin[] = [
  xlmRobertaCompatPlugin,
  hfEncoderCompatPlugin,
  // Neue Modell-Familien hier eintragen:
  // bertCompatPlugin,
  // whisperCompatPlugin,
];

/**
 * Prüft ob ein Dataset mit einem Modell kompatibel ist.
 * @param modelPluginId  Die Plugin-ID des Modells (z.B. "xlm-roberta")
 * @param extensions     Dateiendungen im Dataset (lowercase, mit Punkt)
 */
export function checkDatasetCompat(
  modelPluginId: string,
  extensions: string[]
): DatasetCompatResult {
  const plugin = COMPAT_PLUGINS.find(p => p.modelPluginId === modelPluginId);

  if (!plugin) {
    return {
      overallLevel: 'ok',
      fileResults: [],
      summary: 'Kompatibilität für dieses Modell noch unbekannt.',
      hint: 'Dieses Modell wird in einer späteren Version geprüft.',
    };
  }

  return plugin.checkExtensions(extensions);
}
