// Dataset-Kompatibilitäts-System
//
// Um eine neue Modell-Familie zu unterstützen:
//   1. DatasetCompatPlugin implementieren
//   2. In COMPAT_PLUGINS eintragen – fertig.

// ── Types ──────────────────────────────────────────────────────────────────

export type CompatLevel =
  | 'perfect'    // Ideal für dieses Modell
  | 'ok'         // Nutzbar, aber nicht ideal
  | 'warning'    // Nutzbar mit Einschränkungen
  | 'bad';       // Nicht geeignet

export interface FileCompatResult {
  extension: string;
  level: CompatLevel;
  reason: string;
}

export interface DatasetCompatResult {
  overallLevel: CompatLevel;         // Schlechteste Bewertung aller Dateien
  fileResults: FileCompatResult[];   // Pro Datei-Typ
  summary: string;                   // Kurze Erklärung für den User
  hint?: string;                     // Optionaler Tipp
}

/** Interface das jedes Plugin implementieren muss */
export interface DatasetCompatPlugin {
  /** Plugin-ID – muss mit ModelPlugin.id übereinstimmen */
  modelPluginId: string;
  /** Prüft eine Liste von Dateiendungen (lowercase, mit Punkt, z.B. ".csv") */
  checkExtensions: (extensions: string[]) => DatasetCompatResult;
}

// ── Registry ───────────────────────────────────────────────────────────────

import { xlmRobertaCompatPlugin } from './xlm-roberta/datasetCompat';

const COMPAT_PLUGINS: DatasetCompatPlugin[] = [
  xlmRobertaCompatPlugin,
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

// ── Helpers ────────────────────────────────────────────────────────────────

/** Gibt die "schlechteste" Bewertung aus einer Liste zurück */
export function worstLevel(levels: CompatLevel[]): CompatLevel {
  const order: CompatLevel[] = ['bad', 'warning', 'ok', 'perfect'];
  let worst: CompatLevel = 'perfect';
  for (const l of levels) {
    if (order.indexOf(l) < order.indexOf(worst)) worst = l;
  }
  return worst;
}

export const LEVEL_META: Record<CompatLevel, { label: string; color: string; bg: string; border: string; emoji: string }> = {
  perfect: { label: 'Perfekt geeignet',   color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', emoji: '✅' },
  ok:      { label: 'Geeignet',           color: 'text-blue-400',    bg: 'bg-blue-500/10',    border: 'border-blue-500/30',    emoji: '✔️' },
  warning: { label: 'Bedingt geeignet',   color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   emoji: '⚠️' },
  bad:     { label: 'Nicht geeignet',     color: 'text-red-400',     bg: 'bg-red-500/10',     border: 'border-red-500/30',     emoji: '🚫' },
};
