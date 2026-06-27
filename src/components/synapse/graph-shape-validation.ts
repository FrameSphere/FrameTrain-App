/**
 * Phase 3: Shape Validation & Error Checking
 * 
 * Verhindert ungültige Graphen durch Shape-Flow Analyse:
 * - Conv2D→LSTM: Shape [B,C,H,W] → [B,T,D] ❌
 * - Dense→Dense: Shape [B,D] → [B,D'] ✅
 * - Transformer→Attention: [B,T,D] → [B,T,D'] ✅
 * 
 * VALIDATION PHASES:
 * 1. Connection Validation (beim Edge-Erstellen)
 * 2. Shape Flow Analysis (gesamter Graph)
 * 3. Layer Parameter Validation (inputSize matches outputSize)
 * 4. Error Recovery (suggestions für fix)
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeDefinition } from "./nodeTypes";

/** Synapse nodes store type on `_def.type` (React Flow data). */
export function getSynapseNodeType(node: Node): string {
  const d = node.data as Record<string, unknown> | undefined;
  const def = d?._def as { type?: string } | undefined;
  return def?.type ?? (d?.nodeType as string) ?? "unknown";
}

/**
 * Shape Information für jeden Layer-Typ
 * Format: "Tensor dimensionen" oder "Type" (z.B. "BHWC", "BTC", "BD")
 */
export interface ShapeInfo {
  inputShape: "BHWC" | "BTC" | "BD" | "BT" | string;
  outputShape: "BHWC" | "BTC" | "BD" | "BT" | string;
  description: string;
  flexible: boolean; // Kann verschiedene Formen akzeptieren?
}

export const LAYER_SHAPE_METADATA: Record<string, ShapeInfo> = {
  // Input/Output
  input: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Input node - shape depends on dataset",
    flexible: true,
  },
  output: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Output node - accepts any shape",
    flexible: true,
  },

  // Dense Layer: [B, D] → [B, D']
  dense: {
    inputShape: "BD",
    outputShape: "BD",
    description: "Dense/Linear layer - flattens to [Batch, Features]",
    flexible: false,
  },

  // Conv2D: [B, C, H, W] → [B, C', H', W']
  conv2d: {
    inputShape: "BHWC",
    outputShape: "BHWC",
    description: "2D Convolution - requires image format [Batch, Channels, Height, Width]",
    flexible: false,
  },

  // LSTM: [B, T, D] → [B, T, D'] or [B, D']
  lstm: {
    inputShape: "BTC",
    outputShape: "BTC",
    description: "LSTM - recurrent, requires sequence [Batch, Time, Features]",
    flexible: false,
  },

  // Embedding: [B, T] → [B, T, D]
  embedding: {
    inputShape: "BT",
    outputShape: "BTC",
    description: "Embedding - converts token IDs to dense vectors",
    flexible: false,
  },

  // Attention: [B, T, D] → [B, T, D]
  attention: {
    inputShape: "BTC",
    outputShape: "BTC",
    description: "Multi-head Attention - preserves sequence length",
    flexible: false,
  },

  // Transformer Block: [B, T, D] → [B, T, D]
  transformer_block: {
    inputShape: "BTC",
    outputShape: "BTC",
    description: "Transformer encoder - sequence-to-sequence",
    flexible: false,
  },

  // Activation Functions: pass-through shape
  relu: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "ReLU activation - preserves shape",
    flexible: true,
  },
  sigmoid: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Sigmoid activation - preserves shape",
    flexible: true,
  },
  tanh: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Tanh activation - preserves shape",
    flexible: true,
  },
  gelu: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "GELU activation - preserves shape",
    flexible: true,
  },

  // Normalization: pass-through shape
  layernorm: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Layer Normalization - preserves shape",
    flexible: true,
  },
  batchnorm: {
    inputShape: "BHWC",
    outputShape: "BHWC",
    description: "Batch Normalization - preserves shape (expects BHWC)",
    flexible: false,
  },

  // Dropout: pass-through shape
  dropout: {
    inputShape: "Variable",
    outputShape: "Variable",
    description: "Dropout - preserves shape",
    flexible: true,
  },

  // Concatenation: combines multiple inputs
  concatenate: {
    inputShape: "Variable (multiple)",
    outputShape: "Variable",
    description: "Concatenates multiple inputs along feature dimension",
    flexible: true,
  },
};

/**
 * VALIDATION ERROR TYPES
 */
export enum ValidationErrorType {
  SHAPE_MISMATCH = "SHAPE_MISMATCH",
  INVALID_EDGE = "INVALID_EDGE",
  MISSING_PARAMETER = "MISSING_PARAMETER",
  CYCLE_DETECTED = "CYCLE_DETECTED",
  DIMENSION_ERROR = "DIMENSION_ERROR",
}

export interface ValidationError {
  type: ValidationErrorType;
  severity: "error" | "warning" | "info";
  sourceNodeId: string;
  targetNodeId?: string;
  message: string;
  suggestion?: string;
  details?: Record<string, any>;
}

/**
 * Validiere eine neue Edge bevor sie erstellt wird
 * Dies wird vom SynapseBuilder aufgerufen in onConnect()
 */
export function validateEdgeConnection(
  sourceNodeId: string,
  targetNodeId: string,
  nodes: Node[],
  nodeDefs: NodeDefinition[]
): { valid: boolean; errors: ValidationError[] } {
  const errors: ValidationError[] = [];

  const sourceNode = nodes.find((n) => n.id === sourceNodeId);
  const targetNode = nodes.find((n) => n.id === targetNodeId);

  if (!sourceNode || !targetNode) {
    errors.push({
      type: ValidationErrorType.INVALID_EDGE,
      severity: "error",
      sourceNodeId,
      targetNodeId,
      message: "Source or target node not found",
    });
    return { valid: false, errors };
  }

  const sourceType = getSynapseNodeType(sourceNode);
  const targetType = getSynapseNodeType(targetNode);

  // Get shape metadata
  const sourceShape = LAYER_SHAPE_METADATA[sourceType];
  const targetShape = LAYER_SHAPE_METADATA[targetType];

  // If either is not defined, assume flexible (unknown types)
  if (!sourceShape || !targetShape) {
    return { valid: true, errors: [] };
  }

  // VALIDATION 1: Check if both nodes are flexible (always OK)
  if (sourceShape.flexible && targetShape.flexible) {
    return { valid: true, errors: [] };
  }

  // VALIDATION 2: Check shape compatibility
  if (!sourceShape.flexible && !targetShape.flexible) {
    const compatible = isShapeCompatible(sourceShape.outputShape, targetShape.inputShape);

    if (!compatible) {
      errors.push({
        type: ValidationErrorType.SHAPE_MISMATCH,
        severity: "error",
        sourceNodeId,
        targetNodeId,
        message: `Shape mismatch: ${sourceType} outputs ${sourceShape.outputShape}, but ${targetType} expects ${targetShape.inputShape}`,
        suggestion: `Consider adding a reshape/flatten layer between ${sourceType} and ${targetType}`,
        details: {
          sourceOutput: sourceShape.outputShape,
          targetInput: targetShape.inputShape,
          sourceDescription: sourceShape.description,
          targetDescription: targetShape.description,
        },
      });
      return { valid: false, errors };
    }
  }

  return { valid: true, errors };
}

/**
 * Check if two shapes are compatible
 * BHWC = image [Batch, Height, Width, Channels]
 * BTC = sequence [Batch, Time, Features]
 * BD = dense [Batch, Features]
 * BT = tokens [Batch, Time]
 */
function isShapeCompatible(outputShape: string, inputShape: string): boolean {
  // If either is flexible, it's compatible
  if (outputShape.includes("Variable") || inputShape.includes("Variable")) {
    return true;
  }

  // Exact match
  if (outputShape === inputShape) {
    return true;
  }

  // Special rules:
  // - BHWC can be flattened to BD (Conv2D → Dense)
  // - BTC can be used as BD if Time=1 (Sequence → Dense)
  // - BD can be expanded to BTC if reshaped
  if (outputShape === "BHWC" && inputShape === "BD") {
    return true; // Implicit flatten
  }
  if (outputShape === "BTC" && inputShape === "BD") {
    return true; // Flatten sequence
  }
  if (outputShape === "BD" && inputShape === "BTC") {
    return false; // Can't expand D to TC without explicit reshape
  }

  return false;
}

/**
 * Prüft ob alle Nodes über Kanten erreichbar sind (vom ersten Knoten ohne Eingehende).
 * Gibt isolierte Nodes als WARNING zurück.
 */
export function validateConnectivity(
  nodes: Node[],
  edges: Edge[]
): ValidationError[] {
  if (nodes.length < 2) return [];
  const nodeIds = new Set(nodes.map((n) => n.id));
  const validEdges = edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  const inDegree: Record<string, number> = {};
  nodes.forEach((n) => (inDegree[n.id] = 0));
  validEdges.forEach((e) => inDegree[e.target]++);

  const adj: Record<string, string[]> = {};
  nodes.forEach((n) => (adj[n.id] = []));
  validEdges.forEach((e) => adj[e.source].push(e.target));

  // BFS/Kahn von allen Wurzeln
  const roots = nodes.filter((n) => inDegree[n.id] === 0);
  const reachable = new Set<string>();
  const queue = roots.map((n) => n.id);
  while (queue.length) {
    const id = queue.shift()!;
    reachable.add(id);
    adj[id].forEach((nb) => { if (!reachable.has(nb)) queue.push(nb); });
  }

  const unreachable = nodes.filter((n) => !reachable.has(n.id));
  if (unreachable.length === 0) return [];

  return unreachable.map((n) => ({
    type: ValidationErrorType.INVALID_EDGE,
    severity: "warning" as const,
    sourceNodeId: n.id,
    message: `Node \"${(n.data as any)?._def?.label ?? n.id}\" ist nicht mit dem Graphen verbunden`,
    suggestion: "Node mit einem Eingangs- oder Ausgangs-Node verbinden",
  }));
}

/**
 * Validiere den gesamten Graph auf Shape-Konsistenz
 */
export function validateFullGraph(
  nodes: Node[],
  edges: Edge[],
  nodeDefs: NodeDefinition[]
): { valid: boolean; errors: ValidationError[] } {
  const errors: ValidationError[] = [];
  const nodeIds = new Set(nodes.map((n) => n.id));
  // Dangling edges global herausfiltern — verhindert false-positive Fehler und Crashes
  const validEdges = edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  // VALIDATION 1: Check all edges
  validEdges.forEach((edge) => {
    const { valid, errors: edgeErrors } = validateEdgeConnection(
      edge.source,
      edge.target,
      nodes,
      nodeDefs
    );
    if (!valid) {
      errors.push(...edgeErrors);
    }
  });

  // VALIDATION 2: Check for cycles
  const cycles = detectCycles(nodes, validEdges);
  cycles.forEach((cycle) => {
    errors.push({
      type: ValidationErrorType.CYCLE_DETECTED,
      severity: "warning",
      sourceNodeId: cycle[0],
      message: `Zyklus erkannt: ${cycle.join(" → ")}. Graphen sollten azyklisch sein.`,
      suggestion: "Eine Kante entfernen um den Zyklus aufzubrechen",
    });
  });

  // VALIDATION 3: Check layer parameters
  nodes.forEach((node) => {
    const nodeType = getSynapseNodeType(node);
    const params = (node.data as Record<string, unknown>)?.params as Record<string, unknown> || {};
    const paramErrors = validateNodeParameters(nodeType, params);
    paramErrors.forEach((err) => {
      errors.push({
        type: ValidationErrorType.MISSING_PARAMETER,
        severity: "warning",
        sourceNodeId: node.id,
        message: err,
      });
    });
  });

  // VALIDATION 4: Numeric parameter flow (nur gültige Kanten)
  errors.push(...validateParameterFlow(nodes, validEdges));

  // VALIDATION 5: Nicht verbundene Nodes (Connectivity)
  errors.push(...validateConnectivity(nodes, edges));

  return {
    valid: errors.filter((e) => e.severity === "error").length === 0,
    errors,
  };
}

function nodeParams(node: Node): Record<string, unknown> {
  return ((node.data as Record<string, unknown>)?.params ?? {}) as Record<string, unknown>;
}

function outputFeatureSize(node: Node): number | undefined {
  const type = getSynapseNodeType(node);
  const params = nodeParams(node);
  if (type === "dense") return Number(params.outputSize);
  if (type === "conv2d") return Number(params.outChannels);
  if (type === "layernorm") return Number(params.normalizedShape);
  if (type === "embedding") return Number(params.embeddingDim);
  if (type === "lstm") return Number(params.hiddenSize);
  if (type === "attention" || type === "transformer_block") return Number(params.embedDim);
  return undefined;
}

/**
 * Prüft ob numerische Parameter entlang Kanten zusammenpassen.
 */
export function validateParameterFlow(nodes: Node[], edges: Edge[]): ValidationError[] {
  const errors: ValidationError[] = [];
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  for (const edge of edges) {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) continue;

    const st = getSynapseNodeType(source);
    const tt = getSynapseNodeType(target);
    const sp = nodeParams(source);
    const tp = nodeParams(target);
    const srcOut = outputFeatureSize(source);

    if (st === "dense" && tt === "dense") {
      const sOut = Number(sp.outputSize);
      const tIn = Number(tp.inputSize);
      if (Number.isFinite(sOut) && Number.isFinite(tIn) && sOut !== tIn) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `Dense „${edge.source}": outputSize=${sOut} ≠ Dense „${edge.target}": inputSize=${tIn}`,
          suggestion: `Im Property Panel oder set_param: ${edge.target}.inputSize = ${sOut}`,
          details: { paramKey: "inputSize", expectedValue: sOut, currentValue: tIn },
        });
      }
    }

    if (tt === "conv2d" && srcOut !== undefined) {
      const tIn = Number(tp.inChannels);
      if (Number.isFinite(srcOut) && Number.isFinite(tIn) && srcOut !== tIn) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `Conv2d „${edge.target}": inChannels=${tIn} ≠ Ausgabe von „${edge.source}" (${srcOut})`,
          suggestion: `set_param ${edge.target} inChannels=${srcOut}`,
          details: { paramKey: "inChannels", expectedValue: srcOut, currentValue: tIn },
        });
      }
    }

    if (st === "conv2d" && tt === "dense") {
      const sOut = Number(sp.outChannels);
      const tIn = Number(tp.inputSize);
      if (Number.isFinite(sOut) && Number.isFinite(tIn) && sOut !== tIn) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `Conv2d „${edge.source}": outChannels=${sOut} ≠ Dense „${edge.target}": inputSize=${tIn}`,
          suggestion: `set_param ${edge.target} inputSize=${sOut} (oder Reshape/Flatten dazwischen)`,
          details: { paramKey: "inputSize", expectedValue: sOut, currentValue: tIn },
        });
      }
    }

    if (tt === "layernorm" && srcOut !== undefined) {
      const tNorm = Number(tp.normalizedShape);
      if (Number.isFinite(srcOut) && Number.isFinite(tNorm) && srcOut !== tNorm) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `LayerNorm „${edge.target}": normalizedShape=${tNorm} ≠ Features von „${edge.source}" (${srcOut})`,
          suggestion: `set_param ${edge.target} normalizedShape=${srcOut}`,
          details: { paramKey: "normalizedShape", expectedValue: srcOut, currentValue: tNorm },
        });
      }
    }

    if ((tt === "attention" || tt === "transformer_block") && srcOut !== undefined) {
      const tEmbed = Number(tp.embedDim);
      if (Number.isFinite(srcOut) && Number.isFinite(tEmbed) && srcOut !== tEmbed) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `${tt} „${edge.target}": embedDim=${tEmbed} ≠ Features von „${edge.source}" (${srcOut})`,
          suggestion: `set_param ${edge.target} embedDim=${srcOut}`,
          details: { paramKey: "embedDim", expectedValue: srcOut, currentValue: tEmbed },
        });
      }
    }

    if (st === "dense" && tt === "layernorm") {
      const sOut = Number(sp.outputSize);
      const tNorm = Number(tp.normalizedShape);
      if (Number.isFinite(sOut) && Number.isFinite(tNorm) && sOut !== tNorm) {
        errors.push({
          type: ValidationErrorType.DIMENSION_ERROR,
          severity: "error",
          sourceNodeId: edge.source,
          targetNodeId: edge.target,
          message: `LayerNorm „${edge.target}": normalizedShape=${tNorm} ≠ Dense „${edge.source}": outputSize=${sOut}`,
          suggestion: `set_param ${edge.target} normalizedShape=${sOut}`,
          details: { paramKey: "normalizedShape", expectedValue: sOut, currentValue: tNorm },
        });
      }
    }
  }

  return errors;
}

/**
 * Detect cycles using DFS
 */
function detectCycles(nodes: Node[], edges: Edge[]): string[][] {
  const cycles: string[][] = [];
  const visited = new Set<string>();
  const recursionStack = new Set<string>();

  const adjacency: Record<string, string[]> = {};
  nodes.forEach((n) => { adjacency[n.id] = []; });
  // edges sind hier bereits gefiltert (validEdges aus validateFullGraph)
  edges.forEach((e) => {
    if (adjacency[e.source]) adjacency[e.source].push(e.target);
  });

  const dfs = (nodeId: string, path: string[]): boolean => {
    visited.add(nodeId);
    recursionStack.add(nodeId);
    path.push(nodeId);

    for (const neighbor of adjacency[nodeId]) {
      if (!visited.has(neighbor)) {
        if (dfs(neighbor, [...path])) {
          return true;
        }
      } else if (recursionStack.has(neighbor)) {
        // Cycle found
        const cycleStart = path.indexOf(neighbor);
        cycles.push(path.slice(cycleStart).concat(neighbor));
        return true;
      }
    }

    recursionStack.delete(nodeId);
    return false;
  };

  nodes.forEach((node) => {
    if (!visited.has(node.id)) {
      dfs(node.id, []);
    }
  });

  return cycles;
}

/**
 * Validate node-specific parameters
 */
function validateNodeParameters(nodeType: string, params: Record<string, any>): string[] {
  const errors: string[] = [];

  const requiredParams: Record<string, string[]> = {
    dense: ["inputSize", "outputSize"],
    conv2d: ["inChannels", "outChannels", "kernelSize"],
    lstm: ["inputSize", "hiddenSize"],
    embedding: ["vocabSize", "embeddingDim"],
    attention: ["embedDim", "numHeads"],
    transformer_block: ["embedDim", "numHeads"],
  };

  const required = requiredParams[nodeType] || [];
  required.forEach((param) => {
    if (!params[param] || params[param] === undefined || params[param] === null) {
      errors.push(`Missing parameter: ${param}`);
    }
  });

  return errors;
}

/**
 * Human-readable error message
 */
export function formatValidationError(error: ValidationError): string {
  return `[${error.severity.toUpperCase()}] ${error.message}${
    error.suggestion ? `\n💡 Suggestion: ${error.suggestion}` : ""
  }`;
}

/**
 * DEBUG: Print validation report
 */
export function printValidationReport(
  nodes: Node[],
  edges: Edge[],
  nodeDefs: NodeDefinition[]
): string {
  const { valid, errors } = validateFullGraph(nodes, edges, nodeDefs);

  let report = `
╔════════════════════════════════════════════════════════════╗
║ GRAPH VALIDATION REPORT (Phase 3)                          ║
╚════════════════════════════════════════════════════════════╝

Nodes: ${nodes.length}
Edges: ${edges.length}
Valid: ${valid ? "✅ YES" : "❌ NO"}

${
  errors.length === 0
    ? "✓ No errors found!"
    : `Errors Found: ${errors.length}\n\n${errors
        .map((e) => `  [${e.type}] ${formatValidationError(e)}`)
        .join("\n\n")}`
}
`;

  return report;
}
