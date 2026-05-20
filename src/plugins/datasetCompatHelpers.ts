// Dataset-Compat Hilfsfunktionen – kein Import aus dieser Datei zurück nach oben
// Ausgelagert um zirkuläre Abhängigkeit zu vermeiden:
//   datasetCompat.ts → xlm-roberta/datasetCompat.ts → ../datasetCompat  ← Kreis
//
// Plugin-eigene datasetCompat.ts Dateien importieren von hier statt von ../datasetCompat

export type CompatLevel =
  | 'perfect'
  | 'ok'
  | 'warning'
  | 'bad';

export interface FileCompatResult {
  extension: string;
  level: CompatLevel;
  reason: string;
}

export interface DatasetCompatResult {
  overallLevel: CompatLevel;
  fileResults: FileCompatResult[];
  summary: string;
  hint?: string;
}

export interface DatasetCompatPlugin {
  modelPluginId: string;
  checkExtensions: (extensions: string[]) => DatasetCompatResult;
}

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
