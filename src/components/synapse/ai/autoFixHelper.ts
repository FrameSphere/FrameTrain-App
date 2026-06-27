/**
 * Auto-Fix Helper für Canvas-Graph Probleme
 * 
 * Wendet automatisch Fixes vom AI Coach auf den Graph an
 */

import { Node, Edge } from "@xyflow/react";
import { FixSuggestion } from "./SynapseAICoach";

export interface AutoFixResult {
  success: boolean;
  message: string;
  nodesAdded?: Node[];
  nodesRemoved?: string[];
  edgesAdded?: Edge[];
  edgesRemoved?: string[];
}

/**
 * Führt einen Fix auf dem Canvas-Graph aus
 * 
 * ⚠️ WICHTIG: "remove_node" wird NICHT automatisch angewendet!
 * Das würde User-Modelle ohne Bestätigung zerstören.
 * Stattdessen: WARNUNG anzeigen, User fragen.
 */
export function applyAutoFix(
  fix: FixSuggestion,
  nodes: Node[],
  edges: Edge[],
  setNodes: (nodes: Node[]) => void,
  setEdges: (edges: Edge[]) => void
): AutoFixResult {
  switch (fix.action) {
    case "remove_node":
      // ❌ NICHT automatisch löschen!
      // Nur Warnung zeigen
      return {
        success: false,
        message: `⚠️ Node-Deletion erfordert User-Bestätigung. Bitte manuell entfernen oder Auto-Fix aktualisieren.`,
      };

    case "adjust_params":
      return adjustParamsFix(fix, nodes, setNodes);

    case "insert_bridge":
      return insertBridgeFix(fix, nodes, edges, setNodes, setEdges);

    case "reorder_nodes":
      return reorderNodesFix(fix, nodes, edges, setNodes, setEdges);
    
    case "inspect_only":
      return {
        success: true,
        message: `ℹ️ ${fix.description}`,
      };

    default:
      return {
        success: false,
        message: `Unbekannter Fix-Typ: ${fix.action}`,
      };
  }
}

/**
 * Entfernt einen problematischen Node
 */
function removeNodeFix(
  fix: FixSuggestion,
  nodes: Node[],
  edges: Edge[],
  setNodes: (nodes: Node[]) => void,
  setEdges: (edges: Edge[]) => void
): AutoFixResult {
  const nodeId = fix.targetNodeId;
  const node = nodes.find((n) => n.id === nodeId);

  if (!node) {
    return {
      success: false,
      message: `Node nicht gefunden: ${nodeId}`,
    };
  }

  // Finde incoming und outgoing edges
  const incomingEdges = edges.filter((e) => e.target === nodeId);
  const outgoingEdges = edges.filter((e) => e.source === nodeId);

  // Erstelle Skip-Connections (bypass den removed node)
  const newEdges: Edge[] = [];
  incomingEdges.forEach((inEdge) => {
    outgoingEdges.forEach((outEdge) => {
      newEdges.push({
        id: `bypass_${inEdge.source}_${outEdge.target}`,
        source: inEdge.source,
        target: outEdge.target,
      });
    });
  });

  // Entferne Node und seine Edges
  const newNodes = nodes.filter((n) => n.id !== nodeId);
  const remainingEdges = edges.filter(
    (e) => e.source !== nodeId && e.target !== nodeId
  );

  setNodes(newNodes);
  setEdges([...remainingEdges, ...newEdges]);

  return {
    success: true,
    message: `✅ Node ${node.data?.label || nodeId} entfernt`,
    nodesRemoved: [nodeId],
    edgesAdded: newEdges,
    edgesRemoved: [...incomingEdges, ...outgoingEdges].map((e) => e.id),
  };
}

/**
 * Passt die Parameter eines Nodes an
 */
function adjustParamsFix(
  fix: FixSuggestion,
  nodes: Node[],
  setNodes: (nodes: Node[]) => void
): AutoFixResult {
  const nodeId = fix.targetNodeId;
  const node = nodes.find((n) => n.id === nodeId);

  if (!node) {
    return {
      success: false,
      message: `Node nicht gefunden: ${nodeId}`,
    };
  }

  const newNode = {
    ...node,
    data: {
      ...node.data,
      params: {
        ...(typeof node.data?.params === 'object' ? node.data.params : {}),
        ...(typeof fix.params === 'object' ? fix.params : {}),
      },
    },
  };

  setNodes(nodes.map((n) => (n.id === nodeId ? newNode : n)));

  return {
    success: true,
    message: `✅ Parameter angepasst: ${Object.keys(fix.params || {}).join(", ")}`,
  };
}

/**
 * Fügt eine Bridge-Layer ein um Shape-Mismatch zu beheben
 */
function insertBridgeFix(
  fix: FixSuggestion,
  nodes: Node[],
  edges: Edge[],
  setNodes: (nodes: Node[]) => void,
  setEdges: (edges: Edge[]) => void
): AutoFixResult {
  const beforeNodeId = fix.targetNodeId;
  const params = fix.params || {};

  // Erstelle neue Bridge-Node
  const bridgeId = `bridge_${Date.now()}`;
  const bridgeNode: Node = {
    id: bridgeId,
    data: {
      label: "Dense Bridge",
      nodeType: "dense",
      params: {
        inputSize: params.inputSize || 256,
        outputSize: params.outputSize || 512,
        bias: true,
        initializer: "xavier_uniform",
      },
    },
    position: {
      x: Math.random() * 500 - 250,
      y: Math.random() * 500 - 250,
    },
  };

  // Finde Incoming Edges
  const incomingEdges = edges.filter((e) => e.target === beforeNodeId);

  // Erstelle neue Edges
  const newEdges: Edge[] = [];

  // Redirect incoming edges zur Bridge
  incomingEdges.forEach((edge) => {
    newEdges.push({
      ...edge,
      id: `${edge.source}-${bridgeId}`,
      target: bridgeId,
    });
  });

  // Verbinde Bridge mit Target
  newEdges.push({
    id: `${bridgeId}-${beforeNodeId}`,
    source: bridgeId,
    target: beforeNodeId,
  });

  // Entferne alte Edges und add neue
  const remainingEdges = edges.filter(
    (e) => !incomingEdges.find((ie) => ie.id === e.id)
  );

  setNodes([...nodes, bridgeNode]);
  setEdges([...remainingEdges, ...newEdges]);

  return {
    success: true,
    message: `✅ Bridge-Layer eingefügt: ${params.inputSize || 256} → ${params.outputSize || 512}`,
    nodesAdded: [bridgeNode],
    edgesAdded: newEdges,
    edgesRemoved: incomingEdges.map((e) => e.id),
  };
}

/**
 * Ordnet Nodes neu an basierend auf Topologie
 */
function reorderNodesFix(
  fix: FixSuggestion,
  nodes: Node[],
  edges: Edge[],
  setNodes: (nodes: Node[]) => void,
  setEdges: (edges: Edge[]) => void
): AutoFixResult {
  // TODO: Implementiere topologisches Sorting der Nodes
  // Basierend auf Edge-Verbindungen
  return {
    success: false,
    message: "Reordering ist noch nicht implementiert",
  };
}

/**
 * Zeige Auto-Fix Result als Toast/Notification
 */
export function showAutoFixNotification(result: AutoFixResult) {
  if (result.success) {
    console.log(`✅ ${result.message}`);
    // TODO: Toast anzeigen
  } else {
    console.error(`❌ ${result.message}`);
    // TODO: Error Toast anzeigen
  }
}
