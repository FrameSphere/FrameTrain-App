/**
 * Execution-Order für den Canvas-Graph.
 *
 * Topologisches Sorting (Kahn) über beliebige Graphen. Wird von codeGenerator.ts
 * (canvas_model.py) und graphIR.ts (Runtime-IR fürs Training) genutzt.
 *
 * HINWEIS: Frühere „Dynamic Forward"-Codegeneratoren (generateDynamicForward,
 * buildNodeDependencyMap) wurden entfernt — sie waren toter, fehlerhafter Code.
 * Der echte, node-ID-basierte Forward-Generator lebt in codeGenerator.ts und
 * ist konsistent mit dem Backend-Executor (plugins/canvas/executor.py).
 */

import type { Node, Edge } from "@xyflow/react";

/**
 * Konvertiere den Canvas-Graph in eine Execution Order
 * basierend auf Abhängigkeiten (nicht nur sequenziell).
 */
export function computeExecutionOrder(
  nodes: Node[],
  edges: Edge[],
  silent = false
): string[] {
  const nodeIds = new Set(nodes.map((n) => n.id));
  // Dangling edges entfernen – Kanten auf nicht-existente Nodes verursachen
  // künstlich erhöhte inDegree-Werte die nie 0 werden → falscher Zyklen-Alarm
  const validEdges = edges.filter(
    (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
  );

  const adjacencyList: { [key: string]: string[] } = {};
  const inDegree: { [key: string]: number } = {};

  nodes.forEach((node) => {
    adjacencyList[node.id] = [];
    inDegree[node.id] = 0;
  });

  validEdges.forEach((edge) => {
    const { source, target } = edge;
    adjacencyList[source].push(target);
    inDegree[target]++;
  });

  const queue: string[] = [];
  const executionOrder: string[] = [];

  nodes.forEach((node) => {
    if (inDegree[node.id] === 0) {
      queue.push(node.id);
    }
  });

  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    executionOrder.push(nodeId);

    adjacencyList[nodeId].forEach((neighbor) => {
      inDegree[neighbor]--;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    });
  }

  if (executionOrder.length !== nodes.length) {
    const missing = nodes.map((n) => n.id).filter((id) => !executionOrder.includes(id));
    const missingSet = new Set(missing);
    const hasCycle = validEdges.some(
      (e) => missingSet.has(e.source) && missingSet.has(e.target)
    );
    if (!silent) {
      if (hasCycle) {
        console.warn(`[Synapse] Zyklen in ${missing.length} Node(s): ${missing.join(", ")} – werden ans Ende gesetzt.`);
      } else {
        console.warn(`[Synapse] ${missing.length} nicht verbundene Node(s): ${missing.join(", ")} – bitte prüfen ob alle Nodes angeschlossen sind.`);
      }
    }
    executionOrder.push(...missing);
  }

  return executionOrder;
}
