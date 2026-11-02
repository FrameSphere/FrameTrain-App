/**
 * Live-Prüfung: Ergebnis des Formen-Dienstes auf Kanten und Nodes legen.
 * Reine Funktionen, die React-Hülle sitzt in SynapseBuilder.
 */

import type { Edge, Node } from "@xyflow/react";

export interface LiveNodeResult {
  outputs: Record<string, number[]>;
  params?: number;
}

export interface LiveCheckResult {
  ok: boolean;
  params: number;
  input_shape: number[];
  nodes: Record<string, LiveNodeResult>;
  errors: { node: string; message: string }[];
  edge_sources?: Record<string, [string, string]>;
}

/** Nur was die Formen ändert: Positionen und Auswahl lösen keine Prüfung aus. */
export function structureKey(nodes: Node[], edges: Edge[]): string {
  return JSON.stringify([
    nodes.map((n) => [n.id, (n.data as any)?.nodeType ?? (n.data as any)?._def?.type, (n.data as any)?.params]),
    edges.map((e) => [e.source, e.sourceHandle, e.target, e.targetHandle]),
  ]);
}

export function formatShape(shape: number[]): string {
  return `[${["B", ...shape.slice(1)].join(",")}]`;
}

export function formatParams(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)} Mrd.`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)} Mio.`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)} Tsd.`;
  return String(count);
}

/** Form, die über eine Kante läuft; Datenquellen tragen die Eingabeform. */
function edgeShape(edge: Edge, result: LiveCheckResult): number[] | undefined {
  const outputs = result.nodes[edge.source]?.outputs;
  if (outputs) return outputs[edge.sourceHandle ?? "out"] ?? Object.values(outputs)[0];
  const source = result.edge_sources?.[`${edge.target}:${edge.targetHandle ?? "in"}`];
  return source?.[0] === "__input__" ? result.input_shape : undefined;
}

const EDGE_LABEL = {
  labelStyle: { fill: "#a5b4fc", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" },
  labelBgStyle: { fill: "#0d1117", stroke: "#1e293b" },
  labelBgPadding: [5, 3] as [number, number],
  labelBgBorderRadius: 4,
};

export function decorateEdges(edges: Edge[], result: LiveCheckResult | null): Edge[] {
  if (!result) return edges;
  return edges.map((edge) => {
    const shape = edgeShape(edge, result);
    return shape ? { ...edge, label: formatShape(shape), ...EDGE_LABEL } : edge;
  });
}

export function decorateNodes(nodes: Node[], result: LiveCheckResult | null): Node[] {
  if (!result) return nodes;
  const errors = new Map(result.errors.map((e) => [e.node, e.message]));
  return nodes.map((node) => {
    const liveError = errors.get(node.id);
    const liveParams = result.nodes[node.id]?.params;
    if (!liveError && liveParams === undefined) return node;
    return { ...node, data: { ...node.data, liveError, liveParams } };
  });
}
