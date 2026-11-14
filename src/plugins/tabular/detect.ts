import type { ModelConfig } from '../types';

/** Muss zu train_engine/plugins/tabular (tabular_core.MODEL_TYPE) passen. */
export const TABULAR_MODEL_TYPES = ['ft_tabular'];

/** Nur das in FrameTrain angelegte Tabellenmodell — es gibt kein HF-Basismodell dafuer. */
export function detectTabular(_modelPathOrId: string, configJson?: ModelConfig): boolean {
  const modelType = configJson?.model_type?.toLowerCase();
  return !!modelType && TABULAR_MODEL_TYPES.includes(modelType);
}
