/** Auswahl zu einem Block zusammenfassen: eine Kante rein, eine Kante raus. */

import type { Edge, Node } from "@xyflow/react";

export interface GroupResult {
  block: Node;
  nodes: Node[];
  edges: Edge[];
}

const typeOf = (n: Node): string => (n.data as any)?._def?.type ?? (n.data as any)?.nodeType;

export function groupToBlock(nodes: Node[], edges: Edge[], template: Node): GroupResult | string {
  const picked = nodes.filter((n) => n.selected);
  if (picked.length < 2) return "needTwo";
  const ids = new Set(picked.map((n) => n.id));
  const incoming = edges.filter((e) => !ids.has(e.source) && ids.has(e.target));
  const outgoing = edges.filter((e) => ids.has(e.source) && !ids.has(e.target));
  if (incoming.length > 1 || outgoing.length > 1) return "oneEdge";
  const inner = edges.filter((e) => ids.has(e.source) && ids.has(e.target));
  const x = picked.reduce((sum, n) => sum + n.position.x, 0) / picked.length;
  const y = picked.reduce((sum, n) => sum + n.position.y, 0) / picked.length;
  const data = template.data as Record<string, unknown>;
  const block: Node = {
    ...template,
    position: { x, y },
    data: {
      ...data,
      params: {
        ...(data.params as Record<string, unknown>),
        graph: {
          nodes: picked.map((n) => ({ id: n.id, type: typeOf(n), params: (n.data as any).params ?? {} })),
          edges: inner.map((e) => ({
            source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle,
          })),
        },
      },
    },
  };
  const rewired = edges
    .filter((e) => !ids.has(e.source) || !ids.has(e.target))
    .map((e) => (ids.has(e.target)
      ? { ...e, target: block.id, targetHandle: "in" }
      : ids.has(e.source) ? { ...e, source: block.id, sourceHandle: "out" } : e));
  return { block, nodes: [...nodes.filter((n) => !ids.has(n.id)), block], edges: rewired };
}
