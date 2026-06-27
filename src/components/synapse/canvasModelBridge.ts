/**
 * Synapse Canvas Model → Real Model Library Bridge
 * Converts Canvas-based networks to trainable models in the real model library
 */

import { invoke } from "@tauri-apps/api/core";
import type { ModelGraphConfig } from "./graphToModel";
import type { CanvasGraphIR } from "./graphIR";

export interface CanvasModelMetadata {
  type: "canvas";
  graphConfig: ModelGraphConfig;
  /** Vollständiger Graph-IR für Training via TrainingPanel */
  graphIR?: CanvasGraphIR;
  pythonCode?: string;
  createdAt: number;
}

/**
 * Export Canvas Network as a Real Model to the Model Library
 * This creates a new model that the training system can use
 */
export async function exportCanvasNetworkToModelLibrary(
  config: ModelGraphConfig,
  pythonCode: string,
  modelName: string,
  graphIR?: CanvasGraphIR
): Promise<{ modelId: string; path: string }> {
  try {
    const metadata: CanvasModelMetadata = {
      type: "canvas",
      graphConfig: config,
      graphIR,
      pythonCode,
      createdAt: Date.now(),
    };

    // Invoke Tauri command to create/register the model
    const result = await invoke<{ model_id: string; path: string }>(
      "create_canvas_network_model",
      {
        modelName,
        metadata: JSON.stringify(metadata),
        pythonCode,
      }
    );

    return {
      modelId: result.model_id,
      path: result.path,
    };
  } catch (e: any) {
    throw new Error(`Failed to export to model library: ${e.message}`);
  }
}

/**
 * Aktualisiert ein bestehendes Canvas-Modell (graph_metadata.json + canvas_model.py).
 * Wird verwendet wenn man Änderungen an einem bereits gespeicherten Modell speichert.
 */
export async function updateCanvasNetworkModel(
  modelId: string,
  config: ModelGraphConfig,
  pythonCode: string,
  graphIR?: CanvasGraphIR
): Promise<void> {
  const metadata: CanvasModelMetadata = {
    type: "canvas",
    graphConfig: config,
    graphIR,
    pythonCode,
    createdAt: Date.now(),
  };
  await invoke("update_canvas_network_model", {
    modelId,
    metadata: JSON.stringify(metadata),
    pythonCode,
  });
}

/**
 * Check if a model is a Canvas-based network
 */
export async function isCanvasNetworkModel(modelId: string): Promise<boolean> {
  try {
    const result = await invoke<boolean>("is_canvas_network_model", { modelId });
    return result;
  } catch {
    return false;
  }
}

/**
 * Get the generated PyTorch code for a canvas network model
 */
export async function getCanvasNetworkCode(modelId: string): Promise<string> {
  try {
    const code = await invoke<string>("get_canvas_network_code", { modelId });
    return code;
  } catch (e: any) {
    throw new Error(`Failed to get canvas network code: ${e.message}`);
  }
}
