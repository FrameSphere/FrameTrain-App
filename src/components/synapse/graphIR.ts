/**
 * Canvas Graph IR — Intermediate Representation for runtime training.
 * Sent as JSON to train_engine via config.canvas_graph (not as Python script).
 */

import type { Node, Edge } from "@xyflow/react";
import { computeExecutionOrder } from "./graph-to-dynamic-forward";
import { getSynapseNodeType } from "./graph-shape-validation";
import type { TrainingConfig } from "./TrainingConsole";

export const CANVAS_IR_VERSION = 1 as const;

export interface IRNode {
  id: string;
  type: string;
  category: string;
  params: Record<string, unknown>;
  position?: { x: number; y: number };
}

export interface IREdge {
  id?: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface IRTrainingSpec {
  epochs: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  optimizer: string;
  /** Gradient-Clipping (max grad norm); 0 = aus */
  clipGrad: number;
  loss: string;
  lossReduction: string;
  labelSmoothing: number;
  scheduler: string;
  warmupSteps: number;
  minLr: number;
  numClasses: number;
  taskType: string;
  precision: "fp32" | "fp16" | "bf16";
  gradAccum: number;
  gpu: string;
}

export interface IRDataSpec {
  type: string;
  params: Record<string, unknown>;
}

export interface CanvasGraphIR {
  version: typeof CANVAS_IR_VERSION;
  nodes: IRNode[];
  edges: IREdge[];
  execution_order: string[];
  training: IRTrainingSpec;
  data: IRDataSpec | null;
  metadata: {
    node_count: number;
    edge_count: number;
    built_at: string;
  };
}

function getCategory(n: Node): string {
  const d = n.data as Record<string, unknown>;
  const def = d?._def as { category?: string } | undefined;
  return (d?.category as string) ?? def?.category ?? "";
}

function getParams(n: Node): Record<string, unknown> {
  return ((n.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};
}

/** Build IR from current React Flow canvas state. */
export function buildCanvasGraphIR(
  nodes: Node[],
  edges: Edge[],
  config: TrainingConfig,
  trainingNodes?: { optimizer?: Record<string, unknown>; loss?: Record<string, unknown>; scheduler?: Record<string, unknown>; output?: Record<string, unknown> }
): CanvasGraphIR {
  const irNodes: IRNode[] = nodes.map((n) => ({
    id: n.id,
    type: getSynapseNodeType(n),
    category: getCategory(n),
    params: getParams(n),
    position: n.position ? { x: n.position.x, y: n.position.y } : undefined,
  }));

  const irEdges: IREdge[] = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? undefined,
    targetHandle: e.targetHandle ?? undefined,
  }));

  // silent=true: Zyklus-Warnings bereits in SynapseBuilder gezeigt, hier kein Duplikat
  const execution_order = computeExecutionOrder(nodes, edges, true);

  // Datenquelle: echte Loader-Nodes haben Vorrang vor dem generischen "input"-
  // Platzhalter — sonst würde bei input + image_loader im Graph zufällig
  // "input" als data_type in der Engine landen (→ "wird nicht unterstützt").
  const LOADER_TYPES = new Set(["image_loader", "csv_loader", "parquet_loader"]);
  const dataNode =
    nodes.find((n) => LOADER_TYPES.has(getSynapseNodeType(n)))
    ?? nodes.find((n) => getCategory(n) === "data");
  const trainNodes = nodes.filter((n) => getCategory(n) === "training");

  const optimNode = trainNodes.find((n) => getSynapseNodeType(n) === "optimizer");
  const lossNode = trainNodes.find((n) => getSynapseNodeType(n) === "loss");
  const schedNode = trainNodes.find((n) => getSynapseNodeType(n) === "scheduler");
  const outNode = trainNodes.find((n) => getSynapseNodeType(n) === "output_node");

  const op = trainingNodes?.optimizer ?? (optimNode ? getParams(optimNode) : {});
  const lp = trainingNodes?.loss ?? (lossNode ? getParams(lossNode) : {});
  const sp = trainingNodes?.scheduler ?? (schedNode ? getParams(schedNode) : {});
  const xp = trainingNodes?.output ?? (outNode ? getParams(outNode) : {});

  return {
    version: CANVAS_IR_VERSION,
    nodes: irNodes,
    edges: irEdges,
    execution_order,
    training: {
      epochs: config.epochs,
      batchSize: config.batchSize,
      learningRate: config.learningRate ?? Number(op.lr ?? 0.001),
      weightDecay: Number(op.weightDecay ?? 0.01),
      optimizer: String(op.type ?? "adamw"),
      clipGrad: Number(op.clipGrad ?? 1.0),
      loss: String(lp.type ?? "cross_entropy"),
      lossReduction: String(lp.reduction ?? "mean"),
      labelSmoothing: Number(lp.labelSmoothing ?? 0),
      scheduler: String(sp.type ?? "cosine"),
      warmupSteps: Number(sp.warmupSteps ?? 0),
      minLr: Number(sp.minLr ?? 0),
      numClasses: Number(xp.numClasses ?? 10),
      taskType: String(xp.taskType ?? "classification"),
      precision: config.precision ?? "fp32",
      gradAccum: config.gradAccum ?? 1,
      gpu: config.gpu ?? "cpu",
    },
    data: dataNode
      ? { type: getSynapseNodeType(dataNode), params: getParams(dataNode) }
      : null,
    metadata: {
      node_count: nodes.length,
      edge_count: edges.length,
      built_at: new Date().toISOString(),
    },
  };
}

export function isNonEmptyCanvasGraph(ir: CanvasGraphIR | null | undefined): boolean {
  return !!ir && ir.version === 1 && ir.nodes.length > 0 && ir.execution_order.length > 0;
}
