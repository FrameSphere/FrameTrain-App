// Plugin-Registry – hier werden alle Plugins registriert
//
// Um ein neues Modell zu unterstützen:
//   1. Plugin-Ordner unter src/plugins/<name>/ anlegen
//   2. Plugin in PLUGINS eintragen
//   Das war's.

import type { ModelPlugin, ModelConfig } from './types';
import xlmRobertaPlugin from './xlm-roberta';

/** Alle registrierten Plugins – Reihenfolge bestimmt Priorität bei der Erkennung */
const PLUGINS: ModelPlugin[] = [
  xlmRobertaPlugin,
  // Zukünftige Plugins hier eintragen:
  // bertPlugin,
  // whisperPlugin,
  // mistralPlugin,
];

export type DetectionResult =
  | { supported: true;  plugin: ModelPlugin }
  | { supported: false; reason: string };

/**
 * Erkennt welches Plugin für ein Modell zuständig ist.
 * @param modelPathOrId  Lokaler Pfad oder HuggingFace Model-ID
 * @param configJson     Optional: bereits geparste config.json des Modells
 */
export function detectPlugin(
  modelPathOrId: string,
  configJson?: ModelConfig
): DetectionResult {
  const trimmed = modelPathOrId.trim();
  if (!trimmed) {
    return { supported: false, reason: 'Kein Modellpfad angegeben.' };
  }

  for (const plugin of PLUGINS) {
    if (plugin.detect(trimmed, configJson)) {
      return { supported: true, plugin };
    }
  }

  return {
    supported: false,
    reason: `Dieses Modell wird noch nicht unterstützt. Aktuell verfügbar: ${PLUGINS.map((p) => p.name).join(', ')}.`,
  };
}

export { PLUGINS };
