/**
 * Phase 2: Dynamic Forward Engine
 * 
 * Erweitert Phase 1 um echte Graph-Topologie:
 * - Topologisches Sorting von beliebigen Graphen
 * - Multi-Input Node Support (z.B. Attention, Concatenation)
 * - Dynamic Tensor Routing statt sequentieller Ausführung
 * - Shape-Flow Tracking für Validierung (Phase 3 vorbereitung)
 * 
 * INPUT: Canvas-Graph (Nodes + Edges)
 * OUTPUT: Python-Code mit dynamischer Forward-Engine
 * 
 * BEISPIEL-TRANSFORMATION:
 * Canvas Graph:
 *   Input → Dense → Attention → Dense → Output
 *                  ↑────────────↑ (Multi-Input!)
 * 
 * Generierter Forward Code:
 *   def forward(self, x):
 *       activations = {}
 *       
 *       # Layer 0: Dense (Input)
 *       activations["layer_0"] = self.layers["layer_0"](x)
 *       
 *       # Layer 1: Attention (Multi-Input: Query + Key/Value)
 *       q = activations["layer_0"]
 *       kv = activations["layer_0"]
 *       attn_out, _ = self.layers["layer_1"](q, kv, kv)
 *       activations["layer_1"] = attn_out
 *       
 *       # Layer 2: Dense
 *       activations["layer_2"] = self.layers["layer_2"](activations["layer_1"])
 *       
 *       return activations["layer_2"]
 */

import type { Node, Edge } from "@xyflow/react";
import type { ModelGraphConfig } from "./graphToModel";

interface NodeDependencies {
  [nodeId: string]: {
    layerId: string;
    type: string;
    inputNodeIds: string[];
    canBeSkipped: boolean;
  };
}

/**
 * Konvertiere den Canvas-Graph in eine Execution Order
 * basierend auf Abhängigkeiten (nicht nur sequenziell)
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

/**
 * Baue eine Map von Node-ID → Layer-ID für Tracking
 */
export function buildNodeDependencyMap(
  nodes: Node[],
  edges: Edge[],
  executionOrder: string[]
): NodeDependencies {
  const map: NodeDependencies = {};

  // Finde Layer-Index für jeden Node
  const layerIndexByNodeId: { [nodeId: string]: number } = {};
  let layerCounter = 0;
  
  nodes.forEach((node) => {
    const nodeType = (node.data as any)?.nodeType;
    // Input/Output Nodes zählen nicht als Layer
    if (nodeType !== "input" && nodeType !== "output") {
      layerIndexByNodeId[node.id] = layerCounter++;
    }
  });

  // Baue für jeden Node die Input-Dependencies
  executionOrder.forEach((nodeId) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const nodeType = (node.data as any)?.nodeType;
    const layerId = layerIndexByNodeId[nodeId];
    
    // Finde alle eingehenden Kanten
    const incomingEdges = edges.filter((e) => e.target === nodeId);
    const inputNodeIds = incomingEdges.map((e) => e.source);

    map[nodeId] = {
      layerId: `layer_${layerId}`,
      type: nodeType || "unknown",
      inputNodeIds: inputNodeIds,
      canBeSkipped: nodeType === "input" || nodeType === "output",
    };
  });

  return map;
}

/**
 * Generiere Dynamic Forward Code mit Tensor Routing
 * 
 * WICHTIG: Das ist die Kern-Engine für beliebige Graphen!
 * Sie muss Multi-Input Nodes, Skip Connections, etc. unterstützen.
 */
export function generateDynamicForward(
  nodes: Node[],
  edges: Edge[],
  modelConfig: ModelGraphConfig
): string {
  const executionOrder = computeExecutionOrder(nodes, edges);
  const dependencies = buildNodeDependencyMap(nodes, edges, executionOrder);

  let code = `    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamic Forward Pass: Topologie-basierte Ausführung
        
        Der Forward-Pass traversiert den Graph in Dependency-Order
        und routet Tensoren dynamisch zwischen Nodes.
        Dies ermöglicht beliebige Graphen, nicht nur sequenziell.
        """
        activations = {}  # Speichere Output von jedem Layer
        
`;

  // DEBUG: Execution Order
  code += `        # Debug: Execution Order\n`;
  code += `        # ${executionOrder.map((id, i) => `[${i}] ${id}`).join(" → ")}\n\n`;

  // Iteriere durch Execution Order und generiere Layer-Aufrufe
  executionOrder.forEach((nodeId, orderIndex) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const nodeType = (node.data as any)?.nodeType;
    const nodeData = node.data as any;
    const dep = dependencies[nodeId];

    // Skip Input/Output Nodes
    if (nodeType === "input" || nodeType === "output") {
      if (nodeType === "input") {
        code += `        # Input Node\n`;
        code += `        activations["input"] = x\n\n`;
      }
      return;
    }

    const layerId = dep.layerId;
    code += `        # Step ${orderIndex}: ${nodeType} (Node: ${nodeId})\n`;

    // Generiere Input-Gathering basierend auf Node-Typ
    if (dep.inputNodeIds.length === 0) {
      // Kein Input → nutze x
      code += `        ${layerId}_input = x\n`;
    } else if (dep.inputNodeIds.length === 1) {
      // Single Input
      const inputNodeId = dep.inputNodeIds[0];
      const inputNodeDef = nodes.find((n) => n.id === inputNodeId);
      const inputNodeType = (inputNodeDef?.data as any)?.nodeType;

      if (inputNodeType === "input") {
        code += `        ${layerId}_input = activations["input"]\n`;
      } else {
        const inputLayerIdx = Array.from(executionOrder).indexOf(inputNodeId);
        code += `        ${layerId}_input = activations["layer_${inputLayerIdx}"]\n`;
      }
    } else {
      // Multi Input → sammle und verarbeite basierend auf Node-Typ
      if (nodeType === "attention") {
        // Attention: Query, Key, Value
        code += `        # Multi-Input Attention: Sammle Query, Key, Value\n`;
        const qInput = dep.inputNodeIds[0] || "input";
        const kvInput = dep.inputNodeIds.length > 1 ? dep.inputNodeIds[1] : qInput;

        code += `        query = activations.get("${qInput}", x)\n`;
        code += `        key = activations.get("${kvInput}", x)\n`;
        code += `        value = activations.get("${kvInput}", x)\n`;
        code += `        ${layerId}_output, ${layerId}_weights = self.layers["${layerId}"](query, key, value)\n`;
        code += `        activations["${layerId}"] = ${layerId}_output\n\n`;
        return;
      } else if (nodeType === "concatenate") {
        // Concatenation: Multiple inputs concat
        code += `        # Multi-Input Concatenation\n`;
        code += `        concat_inputs = [\n`;
        dep.inputNodeIds.forEach((inputId) => {
          code += `            activations.get("${inputId}", x),\n`;
        });
        code += `        ]\n`;
        code += `        ${layerId}_input = torch.cat(concat_inputs, dim=-1)\n`;
      } else {
        // Default: nutze ersten Input (warnung!)
        code += `        # WARNING: Multi-Input für ${nodeType} - nutze ersten Input\n`;
        code += `        ${layerId}_input = activations.get("${dep.inputNodeIds[0]}", x)\n`;
      }
    }

    // Generiere Layer-Call
    if (nodeType === "attention") {
      // Wurde oben schon gemacht
    } else if (nodeType === "lstm") {
      code += `        ${layerId}_output, (h, c) = self.layers["${layerId}"](${layerId}_input)\n`;
      code += `        activations["${layerId}"] = ${layerId}_output\n`;
    } else if (nodeType === "transformer_block") {
      code += `        ${layerId}_output = self.layers["${layerId}"](${layerId}_input)\n`;
      code += `        activations["${layerId}"] = ${layerId}_output\n`;
    } else {
      // Standard Layer (Dense, Conv2D, etc.)
      code += `        activations["${layerId}"] = self.layers["${layerId}"](${layerId}_input)\n`;
    }

    code += `\n`;
  });

  // Output: Letzter Layer oder Output Node
  const lastNonOutputNode = executionOrder.reverse().find((id) => {
    const n = nodes.find((node) => node.id === id);
    return (n?.data as any)?.nodeType !== "output";
  });

  if (lastNonOutputNode) {
    const lastDep = dependencies[lastNonOutputNode];
    code += `        # Return: Final Output\n`;
    code += `        return activations["${lastDep.layerId}"]\n`;
  } else {
    code += `        return x  # Kein Output\n`;
  }

  return code;
}

/**
 * DEBUG: Gebe Graph-Topologie aus
 */
export function debugGraphTopology(nodes: Node[], edges: Edge[]): string {
  let debug = "";
  
  debug += "=== Graph Topologie ===\n";
  debug += `Nodes: ${nodes.length}\n`;
  debug += `Edges: ${edges.length}\n\n`;

  debug += "Node-Liste:\n";
  nodes.forEach((n) => {
    const nodeType = (n.data as any)?.nodeType;
    debug += `  ${n.id}: ${nodeType}\n`;
  });

  debug += "\nEdges:\n";
  edges.forEach((e) => {
    debug += `  ${e.source} → ${e.target}\n`;
  });

  return debug;
}
