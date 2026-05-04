// XLM-RoBERTa Plugin – Erkennung
//
// Erkannte Modelltypen:
//   - HuggingFace IDs die "xlm-roberta" enthalten  (z.B. "xlm-roberta-base")
//   - Lokale Modelle mit config.json model_type == "xlm-roberta"
//   - Architekturen: XLMRobertaForSequenceClassification, XLMRobertaForTokenClassification, XLMRobertaModel

import type { ModelConfig } from '../types';

const XLM_ROBERTA_ARCHITECTURES = [
  'XLMRobertaForSequenceClassification',
  'XLMRobertaForTokenClassification',
  'XLMRobertaModel',
  'XLMRobertaForMaskedLM',
];

export function detectXLMRoberta(modelPathOrId: string, configJson?: ModelConfig): boolean {
  // 1. Prüfe den Namen / Pfad (HuggingFace Model-ID oder lokaler Ordnername)
  const normalizedId = modelPathOrId.toLowerCase().replace(/\\/g, '/');
  const lastPart = normalizedId.split('/').pop() ?? normalizedId;

  if (
    normalizedId.includes('xlm-roberta') ||
    normalizedId.includes('xlmroberta') ||
    lastPart.includes('xlm_roberta')
  ) {
    return true;
  }

  // 2. Prüfe config.json falls vorhanden
  if (configJson) {
    if (configJson.model_type === 'xlm-roberta') return true;
    if (
      Array.isArray(configJson.architectures) &&
      configJson.architectures.some((a) => XLM_ROBERTA_ARCHITECTURES.includes(a))
    ) {
      return true;
    }
  }

  return false;
}
