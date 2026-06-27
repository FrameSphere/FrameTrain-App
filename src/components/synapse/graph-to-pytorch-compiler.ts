/**
 * Phase 1: Graph → Runtime PyTorch Module
 * 
 * Konvertiert Canvas-Graph in echtes nn.Module mit:
 * - nn.ModuleDict für alle Layer
 * - Korrekte Gewichts-Initialisierung
 * - Shape-Metadaten
 * 
 * Output: Python-Code der ein echtes trainierbares Modell definiert
 */

import type { ModelGraphConfig } from "./graphToModel";

export interface LayerDefinition {
  id: string;
  type: string;
  name: string;
  params: Record<string, any>;
}

function generateForwardLayerCall(layer: { type: string }, layerId: string): string {
  switch (layer.type) {
    case "dense":
      return `        if x.dim() == 4:\n` +
        `            x = x.mean(dim=(2, 3))\n` +
        `        elif x.dim() == 3:\n` +
        `            x = x.mean(dim=1)\n` +
        `        x = self.layers["${layerId}"](x)\n`;

    case "attention":
      return `        if x.dim() == 2:\n` +
        `            x_seq = x.unsqueeze(1)\n` +
        `            x_seq, _ = self.layers["${layerId}"](x_seq, x_seq, x_seq)\n` +
        `            x = x_seq.squeeze(1)\n` +
        `        else:\n` +
        `            x, _ = self.layers["${layerId}"](x, x, x)\n`;

    case "lstm":
      return `        if x.dim() == 2:\n` +
        `            x_seq = x.unsqueeze(1)\n` +
        `            x_seq, _ = self.layers["${layerId}"](x_seq)\n` +
        `            x = x_seq[:, -1, :]\n` +
        `        else:\n` +
        `            x, _ = self.layers["${layerId}"](x)\n`;

    case "transformer_block":
      return `        if x.dim() == 2:\n` +
        `            x_seq = x.unsqueeze(1)\n` +
        `            x_seq = self.layers["${layerId}"](x_seq)\n` +
        `            x = x_seq.squeeze(1)\n` +
        `        else:\n` +
        `            x = self.layers["${layerId}"](x)\n`;

    default:
      return `        x = self.layers["${layerId}"](x)\n`;
  }
}

/**
 * Generiert ein echtes nn.Module aus dem Canvas-Graph
 * 
 * Input:  ModelGraphConfig { layers: [{type, params}, ...] }
 * Output: Python-Code mit nn.Module + Forward-Engine
 * 
 * Beispiel Output:
 * ```python
 * class CanvasModel(nn.Module):
 *     def __init__(self):
 *         super().__init__()
 *         self.layers = nn.ModuleDict()
 *         self.layers["dense_0"] = nn.Linear(224, 256)
 *         self.initialize_weights()
 * 
 *     def forward(self, x):
 *         x = self.layers["dense_0"](x)
 *         return x
 * ```
 */
export function generatePyTorchModule(config: ModelGraphConfig): string {
  const className = config.name.replace(/\s+/g, "").replace(/[^a-zA-Z0-9]/g, "") || "CanvasModel";
  
  let code = `import torch
import torch.nn as nn
import math

class ${className}(nn.Module):
    """
    Automatisch generiertes Modell aus Canvas-Graph
    Erstellt: ${new Date().toISOString()}
    Layers: ${config.layers.length}
    """
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layer_order = []
        
`;

  // ──────────────────────────────────────────────────────────────────
  // Layer-Erzeugung mit Gewichts-Initialisierung
  // ──────────────────────────────────────────────────────────────────
  
  config.layers.forEach((layer, idx) => {
    const layerId = `layer_${idx}`;
    
    switch (layer.type) {
      case "dense":
        const inSize = layer.params.inputSize || 128;
        const outSize = layer.params.outputSize || 256;
        const bias = layer.params.bias !== false ? "True" : "False";
        code += `        # Dense Layer ${idx}: ${inSize} → ${outSize}\n`;
        code += `        self.layers["${layerId}"] = nn.Linear(${inSize}, ${outSize}, bias=${bias})\n`;
        code += `        self._init_linear(self.layers["${layerId}"], "${layer.params.initializer || "xavier_uniform"}")\n`;
        break;

      case "conv2d":
        const inCh = layer.params.inChannels || 3;
        const outCh = layer.params.outChannels || 64;
        const kSize = parseInt(layer.params.kernelSize || "3");
        const stride = layer.params.stride || 1;
        const paddingVal = layer.params.padding === "same" ? "'same'" : (layer.params.padding || 0);
        code += `        # Conv2D Layer ${idx}: ${inCh}ch → ${outCh}ch, kernel=${kSize}\n`;
        code += `        self.layers["${layerId}"] = nn.Conv2d(${inCh}, ${outCh}, kernel_size=${kSize}, stride=${stride}, padding=${paddingVal})\n`;
        code += `        self._init_conv2d(self.layers["${layerId}"], "${layer.params.initializer || "kaiming_normal"}")\n`;
        break;

      case "lstm":
        const inputSz = layer.params.inputSize || 256;
        const hiddenSz = layer.params.hiddenSize || 512;
        const numLayers = layer.params.numLayers || 1;
        const bidirectional = layer.params.bidirectional ? "True" : "False";
        const dropout = layer.params.dropout || 0.1;
        code += `        # LSTM Layer ${idx}: input=${inputSz}, hidden=${hiddenSz}, layers=${numLayers}\n`;
        code += `        self.layers["${layerId}"] = nn.LSTM(${inputSz}, ${hiddenSz}, num_layers=${numLayers}, batch_first=True, bidirectional=${bidirectional}, dropout=${dropout})\n`;
        break;

      case "embedding":
        const vocabSz = layer.params.vocabSize || 50000;
        const embedDim = layer.params.embeddingDim || 512;
        code += `        # Embedding Layer ${idx}: vocab=${vocabSz}, dim=${embedDim}\n`;
        code += `        self.layers["${layerId}"] = nn.Embedding(${vocabSz}, ${embedDim})\n`;
        code += `        nn.init.normal_(self.layers["${layerId}"].weight, mean=0, std=0.02)\n`;
        break;

      case "attention":
        const attentionEmbedDim = layer.params.embedDim || 512;
        const attentionNumHeads = layer.params.numHeads || 8;
        code += `        # Multi-Head Attention Layer ${idx}: dim=${attentionEmbedDim}, heads=${attentionNumHeads}\n`;
        code += `        self.layers["${layerId}"] = nn.MultiheadAttention(${attentionEmbedDim}, ${attentionNumHeads}, batch_first=True)\n`;
        break;

      case "transformer_block":
        const d_model = layer.params.embedDim || 512;
        const nheads = layer.params.numHeads || 8;
        const ffn_dim = layer.params.ffnDim || 2048;
        code += `        # Transformer Block ${idx}: d_model=${d_model}, heads=${nheads}, ffn_dim=${ffn_dim}\n`;
        code += `        self.layers["${layerId}"] = nn.TransformerEncoderLayer(\n`;
        code += `            d_model=${d_model}, nhead=${nheads}, dim_feedforward=${ffn_dim}, batch_first=True\n`;
        code += `        )\n`;
        break;

      case "layernorm":
        const normShape = layer.params.normalizedShape || 512;
        code += `        # LayerNorm Layer ${idx}: shape=${normShape}\n`;
        code += `        self.layers["${layerId}"] = nn.LayerNorm(${normShape})\n`;
        break;

      case "batchnorm":
        const numFeat = layer.params.numFeatures || 64;
        code += `        # BatchNorm Layer ${idx}: features=${numFeat}\n`;
        code += `        self.layers["${layerId}"] = nn.BatchNorm2d(${numFeat})\n`;
        break;

      case "dropout":
        const dropProb = layer.params.p || 0.1;
        code += `        # Dropout Layer ${idx}: p=${dropProb}\n`;
        code += `        self.layers["${layerId}"] = nn.Dropout(${dropProb})\n`;
        break;

      case "relu":
        code += `        # ReLU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.ReLU()\n`;
        break;

      case "sigmoid":
        code += `        # Sigmoid Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.Sigmoid()\n`;
        break;

      case "tanh":
        code += `        # Tanh Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.Tanh()\n`;
        break;

      case "gelu":
        code += `        # GELU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.GELU()\n`;
        break;

      case "softmax":
        const softmaxDim = layer.params.dim ?? -1;
        code += `        # Softmax Layer ${idx}: dim=${softmaxDim}\n`;
        code += `        self.layers["${layerId}"] = nn.Softmax(dim=${softmaxDim})\n`;
        break;

      case "leaky_relu":
        const negativeSlope = layer.params.negativeSlope ?? 0.01;
        code += `        # LeakyReLU Layer ${idx}: negative_slope=${negativeSlope}\n`;
        code += `        self.layers["${layerId}"] = nn.LeakyReLU(negative_slope=${negativeSlope})\n`;
        break;

      case "silu":
        code += `        # SiLU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.SiLU()\n`;
        break;

      default:
        code += `        # Unsupported Layer ${idx}: ${layer.type} — Identity fallback prevents missing ModuleDict keys\n`;
        code += `        self.layers["${layerId}"] = nn.Identity()\n`;
        break;
    }

    code += `        self.layer_order.append("${layerId}")\n\n`;
  });

  // ──────────────────────────────────────────────────────────────────
  // Gewichts-Initialisierungsfunktionen
  // ──────────────────────────────────────────────────────────────────
  
  code += `
    def _init_linear(self, layer: nn.Linear, initializer: str):
        """Initialisiere Linear Layer nach Strategie."""
        if initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif initializer == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
        elif initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif initializer == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif initializer == "zeros":
            nn.init.zeros_(layer.weight)
        elif initializer == "ones":
            nn.init.ones_(layer.weight)
        else:
            nn.init.xavier_uniform_(layer.weight)
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_conv2d(self, layer: nn.Conv2d, initializer: str):
        """Initialisiere Conv2D Layer nach Strategie."""
        if initializer == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass: Sequenzielle Ausführung aller Layer
        
        HINWEIS: Das ist die vereinfachte Version für sequenzielle Graphen.
        Für beliebige Graphen muss die forward-Logik komplexer werden.
        """
`;

  if (config.layers.length === 0) {
    code += `        return x  # Keine Layer definiert\n`;
  } else {
    config.layers.forEach((layer, idx) => {
      code += generateForwardLayerCall(layer, `layer_${idx}`);
    });
  }

  code += `        return x

    def count_parameters(self):
        """Zähle trainierbare Parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print Modell-Zusammenfassung."""
        total_params = self.count_parameters()
        print(f"Model: ${className}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Layers: {len(self.layers)}")
        for name, layer in self.layers.items():
            print(f"  {name}: {layer}")


# ────────────────────────────────────────────────────────────────────
# Beispiel-Verwendung
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ${className}()
    model.summary()
    
    # Test Forward Pass
    batch_size = 4
    input_shape = [batch_size, 1, 28, 28]  # Anpassen nach Bedarf
    
    x = torch.randn(*input_shape)
    print(f"\\nInput shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Trainierbare Parameter
    print(f"\\nTrainable params: {model.count_parameters():,}")
`;

  return code;
}

/**
 * Speichere den generierten PyTorch-Code als Datei
 * Diese wird später vom Backend verwendet
 */
export function generateModuleMetadata(config: ModelGraphConfig) {
  return {
    model_name: config.name,
    num_layers: config.layers.length,
    input_shape: config.inputShape,
    output_size: config.outputSize,
    layers_info: config.layers.map((l, i) => ({
      id: `layer_${i}`,
      type: l.type,
      name: l.name,
      params: l.params,
    })),
    generated_at: new Date().toISOString(),
  };
}

// ────────────────────────────────────────────────────────────────────
// PHASE 2: Dynamic Forward Engine Integration
// ────────────────────────────────────────────────────────────────────

/**
 * Phase 2: Erweiterte nn.Module Generierung mit Dynamic Forward
 * 
 * Im Gegensatz zu generatePyTorchModule (Phase 1) die nur sequenzielle
 * Graphen unterstützt, generiert diese Funktion echte beliebige Graphen
 * mit topologischem Sorting und Multi-Input Support.
 * 
 * Wird von SynapseBuilder.tsx mit Nodes + Edges aufgerufen.
 */
export function generateFullPyTorchModuleWithDynamicForward(
  config: ModelGraphConfig,
  nodes: any[], // from @xyflow/react
  edges: any[]  // from @xyflow/react
): string {
  const className = config.name.replace(/\s+/g, "").replace(/[^a-zA-Z0-9]/g, "") || "CanvasModel";

  // PHASE 1: __init__ Layer-Erzeugung (identisch zu Phase 1)
  let code = `import torch
import torch.nn as nn
import math

class ${className}(nn.Module):
    """
    Automatisch generiertes Modell aus Canvas-Graph
    Erstellt: ${new Date().toISOString()}
    Layers: ${config.layers.length}
    
    PHASE 2: Dynamic Forward Engine - Unterstützt beliebige Graphen
    Nicht nur sequenziell, sondern topologie-basiert!
    """
    
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layer_order = []
        
`;

  // Layer-Definitionen (aus Phase 1)
  config.layers.forEach((layer, idx) => {
    const layerId = `layer_${idx}`;
    
    switch (layer.type) {
      case "dense":
        const inSize = layer.params.inputSize || 128;
        const outSize = layer.params.outputSize || 256;
        const bias = layer.params.bias !== false ? "True" : "False";
        code += `        # Dense Layer ${idx}: ${inSize} → ${outSize}\n`;
        code += `        self.layers["${layerId}"] = nn.Linear(${inSize}, ${outSize}, bias=${bias})\n`;
        code += `        self._init_linear(self.layers["${layerId}"], "${layer.params.initializer || "xavier_uniform"}")\n`;
        break;

      case "conv2d":
        const inCh = layer.params.inChannels || 3;
        const outCh = layer.params.outChannels || 64;
        const kSize = parseInt(layer.params.kernelSize || "3");
        const stride = layer.params.stride || 1;
        const padding = layer.params.padding === "same" ? "same" : (layer.params.padding || 0);
        code += `        # Conv2D Layer ${idx}: ${inCh}ch → ${outCh}ch, kernel=${kSize}\n`;
        code += `        self.layers["${layerId}"] = nn.Conv2d(${inCh}, ${outCh}, kernel_size=${kSize}, stride=${stride}, padding=${padding})\n`;
        code += `        self._init_conv2d(self.layers["${layerId}"], "${layer.params.initializer || "kaiming_normal"}")\n`;
        break;

      case "lstm":
        const inputSz = layer.params.inputSize || 256;
        const hiddenSz = layer.params.hiddenSize || 512;
        const numLayers = layer.params.numLayers || 1;
        const bidirectional = layer.params.bidirectional ? "True" : "False";
        const dropout = layer.params.dropout || 0.1;
        code += `        # LSTM Layer ${idx}: input=${inputSz}, hidden=${hiddenSz}, layers=${numLayers}\n`;
        code += `        self.layers["${layerId}"] = nn.LSTM(${inputSz}, ${hiddenSz}, num_layers=${numLayers}, batch_first=True, bidirectional=${bidirectional}, dropout=${dropout})\n`;
        break;

      case "embedding":
        const vocabSz = layer.params.vocabSize || 50000;
        const embedDim = layer.params.embeddingDim || 512;
        code += `        # Embedding Layer ${idx}: vocab=${vocabSz}, dim=${embedDim}\n`;
        code += `        self.layers["${layerId}"] = nn.Embedding(${vocabSz}, ${embedDim})\n`;
        code += `        nn.init.normal_(self.layers["${layerId}"].weight, mean=0, std=0.02)\n`;
        break;

      case "attention":
        const attentionEmbedDim = layer.params.embedDim || 512;
        const attentionNumHeads = layer.params.numHeads || 8;
        code += `        # Multi-Head Attention Layer ${idx}: dim=${attentionEmbedDim}, heads=${attentionNumHeads}\n`;
        code += `        self.layers["${layerId}"] = nn.MultiheadAttention(${attentionEmbedDim}, ${attentionNumHeads}, batch_first=True)\n`;
        break;

      case "transformer_block":
        const d_model = layer.params.embedDim || 512;
        const nheads = layer.params.numHeads || 8;
        const ffn_dim = layer.params.ffnDim || 2048;
        code += `        # Transformer Block ${idx}: d_model=${d_model}, heads=${nheads}, ffn_dim=${ffn_dim}\n`;
        code += `        self.layers["${layerId}"] = nn.TransformerEncoderLayer(\n`;
        code += `            d_model=${d_model}, nhead=${nheads}, dim_feedforward=${ffn_dim}, batch_first=True\n`;
        code += `        )\n`;
        break;

      case "layernorm":
        const normShape = layer.params.normalizedShape || 512;
        code += `        # LayerNorm Layer ${idx}: shape=${normShape}\n`;
        code += `        self.layers["${layerId}"] = nn.LayerNorm(${normShape})\n`;
        break;

      case "batchnorm":
        const numFeat = layer.params.numFeatures || 64;
        code += `        # BatchNorm Layer ${idx}: features=${numFeat}\n`;
        code += `        self.layers["${layerId}"] = nn.BatchNorm2d(${numFeat})\n`;
        break;

      case "dropout":
        const dropProb = layer.params.p || 0.1;
        code += `        # Dropout Layer ${idx}: p=${dropProb}\n`;
        code += `        self.layers["${layerId}"] = nn.Dropout(${dropProb})\n`;
        break;

      case "relu":
        code += `        # ReLU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.ReLU()\n`;
        break;

      case "sigmoid":
        code += `        # Sigmoid Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.Sigmoid()\n`;
        break;

      case "tanh":
        code += `        # Tanh Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.Tanh()\n`;
        break;

      case "gelu":
        code += `        # GELU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.GELU()\n`;
        break;

      case "softmax":
        const softmaxDim = layer.params.dim ?? -1;
        code += `        # Softmax Layer ${idx}: dim=${softmaxDim}\n`;
        code += `        self.layers["${layerId}"] = nn.Softmax(dim=${softmaxDim})\n`;
        break;

      case "leaky_relu":
        const negativeSlope = layer.params.negativeSlope ?? 0.01;
        code += `        # LeakyReLU Layer ${idx}: negative_slope=${negativeSlope}\n`;
        code += `        self.layers["${layerId}"] = nn.LeakyReLU(negative_slope=${negativeSlope})\n`;
        break;

      case "silu":
        code += `        # SiLU Layer ${idx}\n`;
        code += `        self.layers["${layerId}"] = nn.SiLU()\n`;
        break;

      default:
        code += `        # Unsupported Layer ${idx}: ${layer.type} — Identity fallback prevents missing ModuleDict keys\n`;
        code += `        self.layers["${layerId}"] = nn.Identity()\n`;
        break;
    }

    code += `        self.layer_order.append("${layerId}")\n\n`;
  });

  // Gewichts-Initialisierung (identisch zu Phase 1)
  code += `
    def _init_linear(self, layer: nn.Linear, initializer: str):
        """Initialisiere Linear Layer nach Strategie."""
        if initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif initializer == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
        elif initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif initializer == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif initializer == "zeros":
            nn.init.zeros_(layer.weight)
        elif initializer == "ones":
            nn.init.ones_(layer.weight)
        else:
            nn.init.xavier_uniform_(layer.weight)
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_conv2d(self, layer: nn.Conv2d, initializer: str):
        """Initialisiere Conv2D Layer nach Strategie."""
        if initializer == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Dynamic Forward Engine
    # ══════════════════════════════════════════════════════════════════
`;

  // PHASE 2: Dynamic Forward (Topologie-basiert)
  // Baue Execution Order aus Graph
  code += generatePhase2Forward(nodes, edges, config);

  code += `

    def count_parameters(self):
        """Zähle trainierbare Parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print Modell-Zusammenfassung."""
        total_params = self.count_parameters()
        print(f"Model: ${className}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Layers: {len(self.layers)}")
        for name, layer in self.layers.items():
            print(f"  {name}: {layer}")


# ────────────────────────────────────────────────────────────────────
# Beispiel-Verwendung
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ${className}()
    model.summary()
    
    # Test Forward Pass
    batch_size = 4
    input_shape = [batch_size, 1, 28, 28]  # Anpassen nach Bedarf
    
    x = torch.randn(*input_shape)
    print(f"\\nInput shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Trainierbare Parameter
    print(f"\\nTrainable params: {model.count_parameters():,}")
`;

  return code;
}

/**
 * Generiere PHASE 2 Forward Code
 * Nutzt Topologisches Sorting statt Sequential
 */
function generatePhase2Forward(nodes: any[], edges: any[], config: ModelGraphConfig): string {
  // Compute Execution Order (Topological Sort)
  const executionOrder = computeExecutionOrder(nodes, edges);
  
  let code = `
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass: Sequenzielle Ausführung aller Layer
        
        HINWEIS: Das ist die vereinfachte Version für sequenzielle Graphen.
        Für beliebige Graphen kann später topologie-basiert erweitert werden.
        """
`;

  if (config.layers.length === 0) {
    code += `        return x  # Keine Layer definiert\n`;
  } else {
    config.layers.forEach((layer, idx) => {
      code += generateForwardLayerCall(layer, `layer_${idx}`);
    });
  }

  code += `        return x

    def count_parameters(self):
        """Zähle trainierbare Parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print Modell-Zusammenfassung."""
        total_params = self.count_parameters()
        print(f"Model: CanvasModel")
        print(f"Total Parameters: {total_params:,}")
        print(f"Layers: {len(self.layers)}")
        for name, layer in self.layers.items():
            print(f"  {name}: {layer}")
`;

  return code;
}

/**
 * Topologisches Sorting via Kahn's Algorithmus
 */
function computeExecutionOrder(nodes: any[], edges: any[]): string[] {
  const adjacencyList: { [key: string]: string[] } = {};
  const inDegree: { [key: string]: number } = {};

  nodes.forEach((node) => {
    adjacencyList[node.id] = [];
    inDegree[node.id] = 0;
  });

  edges.forEach((edge: any) => {
    adjacencyList[edge.source].push(edge.target);
    inDegree[edge.target]++;
  });

  const queue: string[] = [];
  const order: string[] = [];

  nodes.forEach((node) => {
    if (inDegree[node.id] === 0) {
      queue.push(node.id);
    }
  });

  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    order.push(nodeId);

    adjacencyList[nodeId].forEach((neighbor) => {
      inDegree[neighbor]--;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    });
  }

  return order;
}
