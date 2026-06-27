/**
 * Converts a Node/Edge Graph into a trainable Model Configuration
 * This bridges visual nodes to actual ML model definitions
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeDefinition } from "./nodeTypes";
import { getSynapseNodeType } from "./graph-shape-validation";

export interface LayerConfig {
  type: string;
  name: string;
  params: Record<string, any>;
}

export interface ModelGraphConfig {
  id: string;
  name: string;
  layers: LayerConfig[];
  inputShape: string;
  outputSize: number;
  datasetType?: string;
  graphMetadata: {
    nodeCount: number;
    edgeCount: number;
    generatedAt: number;
  };
}

/**
 * Generate a model configuration from the visual node graph.
 * Traverses the graph, collects layer definitions, and orders them by connectivity.
 */
export function generateModelConfigFromGraph(
  nodes: Node[],
  edges: Edge[],
  nodeDefs: NodeDefinition[],
  graphName: string = "Canvas Network"
): ModelGraphConfig | null {
  console.log("[generateModelConfigFromGraph] Starting with", nodes.length, "nodes and", nodeDefs.length, "definitions");
  
  if (nodes.length === 0) {
    console.warn("[generateModelConfigFromGraph] No nodes provided");
    return null;
  }

  // Build reverse edge map to find root nodes (nodes with NO incoming edges)
  const incomingEdges = new Set<string>();
  edges.forEach((e) => {
    incomingEdges.add(e.target);
  });

  // Find input node: either type="input" or a root node (no incoming edges)
  let inputNode = nodes.find((n) => {
    const nodeType = getSynapseNodeType(n);
    const def = nodeDefs.find((d) => d.type === nodeType);
    const isInput = def?.type === "input" || (def?.category as any) === "input";
    console.log(`[Node ${n.id}] type=${nodeType}, hasDefinition=${!!def}, isInput=${isInput}, hasIncoming=${incomingEdges.has(n.id)}`);
    return isInput;
  });

  // Fallback: use first root node (no incoming edges) if no explicit input node
  if (!inputNode) {
    inputNode = nodes.find((n) => !incomingEdges.has(n.id));
    if (inputNode) {
      const nodeType = getSynapseNodeType(inputNode);
      // Keine Warning wenn ein bekannter Data-Node als Root fungiert
      const isKnownDataRoot = ["image_loader","csv_loader","input","augmentation"]
        .includes(nodeType);
      if (!isKnownDataRoot) {
        console.warn("[generateModelConfigFromGraph] No explicit input node found. Looking for root nodes...");
      }
      console.log("[generateModelConfigFromGraph] Found root node:", inputNode.id, "type=", nodeType);
    }
  }

  if (!inputNode) {
    console.warn("[generateModelConfigFromGraph] No input node and no root node found. Available nodes:");
    nodes.forEach((n) => {
      const nodeType = getSynapseNodeType(n);
      const def = nodeDefs.find((d) => d.type === nodeType);
      console.warn(`  - ${n.id}: type=${nodeType}, hasDef=${!!def}, label=${(n.data as any)?.label}, hasIncoming=${incomingEdges.has(n.id)}`);
    });
    console.warn("[generateModelConfigFromGraph] Available definitions:");
    nodeDefs.forEach((d) => {
      console.warn(`  - ${d.type} (${d.category})`);
    });
    return null;
  }

  console.log("[generateModelConfigFromGraph] Found input/root node:", inputNode.id);

  const outputNode = nodes.find((n) => {
    const nodeType = getSynapseNodeType(n);
    const def = nodeDefs.find((d) => d.type === nodeType);
    const label = (n.data as any)?.label || "";
    return def?.category === "layer" && typeof label === "string" && label.includes("Output");
  });

  // Build edge map for traversal
  const edgeMap = new Map<string, string[]>();
  edges.forEach((e) => {
    if (!edgeMap.has(e.source)) edgeMap.set(e.source, []);
    edgeMap.get(e.source)?.push(e.target);
  });

  // Traverse graph from input to collect layers in order
  const layers: LayerConfig[] = [];
  const visited = new Set<string>();
  const queue: string[] = [inputNode.id];

  // Extract input shape from input node
  let inputShape = "-1, 3, 224, 224";
  if ((inputNode.data as any)?.params?.shape) {
    const shape = (inputNode.data as any).params.shape;
    if (typeof shape === "string") {
      inputShape = shape;
    }
  }

  let processedCount = 0;
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    processedCount++;

    const node = nodes.find((n) => n.id === nodeId);
    if (!node) continue;

    const nodeType = getSynapseNodeType(node);
    const nodeDef = nodeDefs.find((d) => d.type === nodeType);

    console.log(`[Process ${processedCount}] Node ${nodeId}: type=${nodeType}, hasDef=${!!nodeDef}, category=${nodeDef?.category}`);

    // Skip only explicit input nodes - they're metadata
    // But TRAVERSE THROUGH them to find actual layer nodes
    if (nodeDef?.type === "input") {
      console.log(`  → Skipping input metadata node, but continuing traversal`);
      const nextNodes = edgeMap.get(nodeId) || [];
      queue.push(...nextNodes.filter((n) => !visited.has(n)));
      continue;
    }

    // Collect ONLY layer and activation nodes
    if (nodeDef?.category === "layer" || nodeDef?.category === "activation") {
      const params = { ...(node.data as any)?.params };
      
      // Validate params
      if (!params || Object.keys(params).length === 0) {
        console.warn(`[WARNING] Node ${nodeId} (${nodeType}) has empty params`, node.data);
      }
      
      layers.push({
        type: nodeType,
        name: (node.data as any)?.label || nodeDef?.label,
        params,
      });
      console.log(`  → Added Layer (${nodeDef?.category}): ${nodeType} with params:`, Object.keys(params));
    } else {
      console.log(`  → Skipping non-layer node (${nodeDef?.category}): ${nodeType}`);
    }

    // Continue traversal through ALL nodes (data, layers, training)
    const nextNodes = edgeMap.get(nodeId) || [];
    queue.push(...nextNodes.filter((n) => !visited.has(n)));
  }

  console.log(`[generateModelConfigFromGraph] Collected ${layers.length} layers from ${processedCount} nodes`);

  if (layers.length === 0) {
    console.warn("[generateModelConfigFromGraph] No valid layers found!");
    return null;
  }

  // Infer output size from final layer
  let outputSize = 10; // default
  if (layers.length > 0) {
    const lastLayer = layers[layers.length - 1];
    if (lastLayer.params?.outputSize) {
      outputSize = lastLayer.params.outputSize;
    } else if (lastLayer.params?.units) {
      outputSize = lastLayer.params.units;
    }
  }

  const config = {
    id: `graph_${Date.now()}`,
    name: graphName,
    layers,
    inputShape,
    outputSize,
    graphMetadata: {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      generatedAt: Date.now(),
    },
  };

  console.log("[generateModelConfigFromGraph] SUCCESS:", config);
  return config;
}

/**
 * Generate Python code for this model (PyTorch style)
 * This can be used for code export or as reference
 */
export function generatePyTorchCode(config: ModelGraphConfig): string {
  let code = `import torch
import torch.nn as nn

class ${config.name.replace(/\s+/g, "")}(nn.Module):
    def __init__(self):
        super().__init__()
        
`;

  // Add layers
  config.layers.forEach((layer, idx) => {
    switch (layer.type) {
      case "dense":
        code += `        self.layer${idx} = nn.Linear(${layer.params.inputSize || "input_size"}, ${layer.params.outputSize || 128})\n`;
        break;
      case "conv2d":
        code += `        self.layer${idx} = nn.Conv2d(${layer.params.inChannels || 3}, ${layer.params.outChannels || 64}, kernel_size=${layer.params.kernelSize || 3}, stride=${layer.params.stride || 1}, padding=${layer.params.padding || 1})\n`;
        break;
      case "lstm":
        code += `        self.layer${idx} = nn.LSTM(${layer.params.inputSize || "input_size"}, ${layer.params.hiddenSize || 256}, num_layers=${layer.params.numLayers || 1}, batch_first=True)\n`;
        break;
      case "relu":
      case "activation":
        code += `        self.layer${idx} = nn.ReLU()\n`;
        break;
      case "dropout":
        code += `        self.layer${idx} = nn.Dropout(${layer.params.rate || 0.1})\n`;
        break;
      default:
        code += `        # TODO: Implement ${layer.type}\n`;
    }
  });

  code += `
    def forward(self, x):
`;

  config.layers.forEach((_, idx) => {
    code += `        x = self.layer${idx}(x)\n`;
  });

  code += `        return x
`;

  return code;
}
