// Plugin-Registry – hier werden alle Plugins registriert
//
// Um ein neues Modell zu unterstützen:
//   1. Plugin-Ordner unter src/plugins/<name>/ anlegen
//   2. Plugin in PLUGINS eintragen
//   Das war's.

import type { ModelPlugin, ModelConfig } from './types';
import xlmRobertaPlugin from './xlm-roberta';
import hfEncoderPlugin from './hf-encoder';
import canvasPlugin from './canvas';
import yoloPlugin from './yolo';
import imageClassificationPlugin from './image-classification';

/** Alle registrierten Plugins – Reihenfolge bestimmt Priorität bei der Erkennung */
const PLUGINS: ModelPlugin[] = [
  canvasPlugin,              // Canvas Neural Net (Synapse Builder) — muss vor generischen stehen
  yoloPlugin,                // YOLO Object Detection (YOLOv5/v8/v9/v11)
  imageClassificationPlugin, // ResNet / EfficientNet / ViT / MobileNet
  xlmRobertaPlugin,
  hfEncoderPlugin,
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

  // Bekannte, aber (noch) nicht trainierbare Architekturen konkret benennen.
  // Ein pauschales "wird nicht unterstützt" ließ Nutzer erst nach dem
  // vollständigen Download und der kompletten Konfiguration auflaufen.
  const modelType = configJson?.model_type?.toLowerCase();
  const known = modelType ? KNOWN_UNSUPPORTED[modelType] : undefined;
  if (known) {
    return {
      supported: false,
      reason: `${known} FrameTrain trainiert derzeit Encoder-Modelle für Sequenzklassifikation (BERT, DistilBERT, RoBERTa, XLM-RoBERTa, DeBERTa und verwandte).`,
    };
  }

  return {
    supported: false,
    reason: `Dieses Modell wird noch nicht unterstützt${modelType ? ` (Architektur: ${modelType})` : ''}. Aktuell verfügbar: ${PLUGINS.map((p) => p.name).join(', ')}.`,
  };
}

/**
 * Architekturen, die häufig gesucht werden, für die es aber kein Plugin gibt.
 * Der Text erklärt den Grund – das ist der Unterschied zwischen "geht nicht"
 * und "geht nicht, weil …, nimm stattdessen …".
 */
const KNOWN_UNSUPPORTED: Record<string, string> = {
  gpt2: 'GPT-2 ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  gptj: 'GPT-J ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  gpt_neo: 'GPT-Neo ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  gpt_neox: 'GPT-NeoX ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  llama: 'Llama ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  mistral: 'Mistral ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  qwen2: 'Qwen ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  falcon: 'Falcon ist ein Decoder-Modell für Textgenerierung, kein Encoder.',
  t5: 'T5 ist ein Encoder-Decoder-Modell (Seq2Seq), kein reiner Encoder.',
  mt5: 'mT5 ist ein Encoder-Decoder-Modell (Seq2Seq), kein reiner Encoder.',
  bart: 'BART ist ein Encoder-Decoder-Modell (Seq2Seq), kein reiner Encoder.',
  marian: 'Marian ist ein Übersetzungsmodell (Seq2Seq), kein reiner Encoder.',
  whisper: 'Whisper ist ein Audio-Seq2Seq-Modell.',
  clip: 'CLIP ist ein multimodales Embedding-Modell.',
};

/** Minimal-Shape für die Modell-Vorauswahl – deckt ModelInfo aus den Panels ab. */
export interface ModelDetectionInfo {
  id: string;
  name: string;
  source_path?: string | null;
  model_type?: string | null;
}

/** Prüft, ob für ein Modell ein Plugin existiert. */
export function isModelSupported(model: ModelDetectionInfo): boolean {
  return detectPlugin(
    model.source_path ?? model.name,
    model.model_type ? { model_type: model.model_type } : undefined,
  ).supported;
}

/**
 * Wählt das erste Modell aus, für das ein Plugin zuständig ist.
 *
 * Ohne diese Vorauswahl landet man auf dem zuletzt geladenen Modell – das ist
 * häufig eines, das (noch) kein Plugin unterstützt, und die Seite startet
 * direkt im Fehlerzustand.
 *
 * `withVersions` liefert die Reihenfolge und die IDs, `models` die für die
 * Erkennung nötigen Felder (source_path / model_type). Findet sich kein
 * unterstütztes Modell, wird auf das erste zurückgefallen, damit die Auswahl
 * nie leer bleibt.
 */
export function pickPreferredModelId(
  withVersions: { id: string; name: string }[],
  models: ModelDetectionInfo[],
): string | null {
  if (withVersions.length === 0) return null;
  const byId = new Map(models.map((m) => [m.id, m]));
  const supported = withVersions.find((entry) => {
    const info = byId.get(entry.id);
    return isModelSupported(info ?? { id: entry.id, name: entry.name });
  });
  return (supported ?? withVersions[0]).id;
}

export { PLUGINS };
