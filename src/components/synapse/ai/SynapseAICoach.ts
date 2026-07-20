/**
 * SynapseAICoach - Intelligenter Training-Debugger & Graph-Optimizer
 * 
 * Features:
 * - Training-Ergebnis Analyse
 * - Error-Diagnostik (Shape, Memory, etc.)
 * - Automatische Problem-Identification
 * - Vorschläge zur Graph-Optimierung
 * - Direkte Node/Layer Fixes
 */

import { Node, Edge } from "@xyflow/react";

export interface TrainingResult {
  success: boolean;
  jobId: string;
  duration: number;
  epochs: number;
  finalLoss?: number;
  error?: string;
  errorType?: "shape" | "memory" | "nan" | "runtime" | "unknown";
  errorDetails?: string;
  timestamp: number;
}

export interface GraphDiagnosis {
  isHealthy: boolean;
  issues: DiagnosticIssue[];
  warnings: string[];
  suggestions: FixSuggestion[];
}

export interface DiagnosticIssue {
  type: "shape_mismatch" | "dimension_incompatible" | "layer_config";
  severity: "error" | "warning";
  nodeId: string;
  nodeType: string;
  description: string;
  affectedEdges: string[]; // edge IDs that are problematic
}

export interface FixSuggestion {
  id: string;
  title: string;
  description: string;
  action: "remove_node" | "adjust_params" | "reorder_nodes" | "insert_bridge" | "inspect_only";
  targetNodeId: string;
  params?: Record<string, any>;
  priority: "high" | "medium" | "low";
  requiresUserConfirmation?: boolean;
}

/**
 * Parser für Training-Fehler aus Python Backend
 */
export class ErrorParser {
  static parseTrainingError(errorMsg: string): {
    type: string;
    root: string;
    details: string;
  } {
    const msg = errorMsg.toLowerCase();

    if (
      msg.includes("image_loader") || msg.includes("csv_loader") ||
      msg.includes("parquet_loader") || msg.includes("found no valid file") ||
      msg.includes("kein dataset")
    ) {
      return {
        type: "dataset_error",
        root: "Dataset-Problem: Pfad, Auswahl oder Format prüfen",
        details: errorMsg,
      };
    }
    if (msg.includes("shape")) {
      return {
        type: "shape_mismatch",
        root: this.extractShapeInfo(errorMsg),
        details: errorMsg,
      };
    }
    if (msg.includes("out of memory")) {
      return {
        type: "memory_error",
        root: "GPU/CPU Memory insuffizient",
        details: errorMsg,
      };
    }
    if (/\bnan\b/.test(msg)) {
      return {
        type: "nan_error",
        root: "NaN in loss - Training instabil",
        details: errorMsg,
      };
    }
    if (msg.includes("dimension")) {
      return {
        type: "dimension_error",
        root: this.extractDimensionInfo(errorMsg),
        details: errorMsg,
      };
    }

    return {
      type: "unknown_error",
      root: "Unbekannter Fehler",
      details: errorMsg,
    };
  }

  static extractShapeInfo(errorMsg: string): string {
    // Extrahiere Shape-Informationen aus Error-Message
    const shapeMatch = errorMsg.match(/shape.*?\[.*?\]/gi);
    const expectedMatch = errorMsg.match(/expected.*?\d+/gi);

    let info = "";
    if (shapeMatch) info += shapeMatch[0];
    if (expectedMatch) info += ` but got ${expectedMatch[0]}`;

    return info || "Shape mismatch detected";
  }

  static extractDimensionInfo(errorMsg: string): string {
    const dimMatch = errorMsg.match(/dimension.*?\d+/gi);
    return dimMatch ? dimMatch[0] : "Dimension error";
  }
}

/**
 * Analysiert Canvas-Graph auf Kompatibilitätsprobleme
 */
export class GraphAnalyzer {
  static analyzeGraph(
    nodes: Node[],
    edges: Edge[],
    layerConfig: any[]
  ): GraphDiagnosis {
    const issues: DiagnosticIssue[] = [];
    const warnings: string[] = [];
    const suggestions: FixSuggestion[] = [];

    // Die Heuristik unten nimmt eine LINEARE Kette an (Layer i → i+1 in
    // Array-Reihenfolge). Bei verzweigten Graphen (Fan-in/Fan-out) wären
    // die Meldungen frei erfunden — dann keine Struktur-Diagnose liefern;
    // maßgeblich bleibt die echte Graph-Validierung (validateFullGraph).
    const inDeg = new Map<string, number>();
    const outDeg = new Map<string, number>();
    edges.forEach((e) => {
      outDeg.set(e.source, (outDeg.get(e.source) ?? 0) + 1);
      inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
    });
    const isLinearChain = nodes.every(
      (n) => (inDeg.get(n.id) ?? 0) <= 1 && (outDeg.get(n.id) ?? 0) <= 1
    );
    if (!isLinearChain) {
      return { isHealthy: true, issues, warnings, suggestions };
    }

    // ─────────────────────────────────────────────────────────────────
    // 1. Shape Compatibility Check
    // ─────────────────────────────────────────────────────────────────

    const shapeMap = new Map<string, number>();

    // Initialisiere erste Layer mit Input-Size
    layerConfig.forEach((layer, idx) => {
      const nodeId = `layer_${idx}`;
      const outputShape =
        this.getLayerOutputShape(layer.type, layer.params) || 256;
      shapeMap.set(nodeId, outputShape);
    });

    // Prüfe Layer-Sequenz auf Shape-Kompatibilität
    for (let i = 0; i < layerConfig.length - 1; i++) {
      const currentLayer = layerConfig[i];
      const nextLayer = layerConfig[i + 1];
      const currentOutput = shapeMap.get(`layer_${i}`) || 256;
      const nextInput =
        this.getLayerInputShape(nextLayer.type, nextLayer.params) || 256;

      if (currentOutput !== nextInput) {
        const nodeId = `layer_${i + 1}`;
        const issue: DiagnosticIssue = {
          type: "shape_mismatch",
          severity: "error",
          nodeId: nodeId,
          nodeType: nextLayer.type,
          description: `Layer ${i + 1} (${nextLayer.type}) erwartet Input-Size ${nextInput}, bekommt aber ${currentOutput} von Layer ${i}`,
          affectedEdges: edges
            .filter((e) => e.target === nodeId)
            .map((e) => e.id),
        };
        issues.push(issue);

        // Generiere Fix-Suggestions (aber NICHT automatisch löschen!)
        
        // Suggestion 1: Bridge-Layer (sicher & präserviert Model)
        suggestions.push({
          id: `fix_bridge_${i}_${i + 1}`,
          title: `Bridge-Layer zwischen Layer ${i} und ${i + 1}`,
          description: `Füge eine Dense-Bridge Layer hinzu um von ${currentOutput} auf ${nextInput} zu transformieren`,
          action: "insert_bridge",
          targetNodeId: nodeId,
          params: {
            inputSize: currentOutput,
            outputSize: nextInput,
            insertBefore: i + 1,
          },
          priority: "high",
        });
        
        // Suggestion 2: Nur Warnung (nicht automatisch löschen)
        suggestions.push({
          id: `inspect_${i + 1}`,
          title: `⚠ Layer ${i + 1} überprüfen`,
          description: `Diese Layer könnte der Verursacher sein. Überprüfe manuell oder nutze die Bridge-Option.`,
          action: "inspect_only",
          targetNodeId: nodeId,
          priority: "medium",
          requiresUserConfirmation: false,
        });
      }
    }

    // ─────────────────────────────────────────────────────────────────
    // 2. Layer Configuration Checks
    // ─────────────────────────────────────────────────────────────────

    layerConfig.forEach((layer, idx) => {
      const nodeId = `layer_${idx}`;

      // Check LayerNorm
      if (layer.type === "layernorm") {
        const shape = layer.params.normalizedShape;
        const prevOutput = shapeMap.get(`layer_${idx - 1}`) || shape;
        if (shape !== prevOutput) {
          warnings.push(
            `LayerNorm ${idx}: normalized_shape=${shape} stimmt nicht mit Input=${prevOutput} überein`
          );
        }
      }

      // Check Attention Input
      if (layer.type === "attention") {
        const embedDim = layer.params.embedDim || 512;
        const prevOutput = shapeMap.get(`layer_${idx - 1}`) || 512;
        if (embedDim !== prevOutput) {
          warnings.push(
            `Attention ${idx}: embedDim=${embedDim} passt nicht zu Input=${prevOutput}`
          );
        }
      }

      // Check LSTM Input
      if (layer.type === "lstm") {
        const inputSize = layer.params.inputSize || 256;
        const prevOutput = shapeMap.get(`layer_${idx - 1}`) || 256;
        if (inputSize !== prevOutput) {
          warnings.push(
            `LSTM ${idx}: inputSize=${inputSize} passt nicht zu Input=${prevOutput}`
          );
        }
      }
    });

    return {
      isHealthy: issues.length === 0,
      issues,
      warnings,
      suggestions,
    };
  }

  static getLayerOutputShape(
    type: string,
    params: Record<string, any>
  ): number {
    switch (type) {
      case "dense":
        return params.outputSize || 256;
      case "conv2d":
        return params.outChannels || 64;
      case "lstm":
        return params.hiddenSize || 512;
      case "attention":
        return params.embedDim || 512;
      case "layernorm":
        return params.normalizedShape || 512;
      case "transformer_block":
        return params.embedDim || 512;
      default:
        return 256;
    }
  }

  static getLayerInputShape(
    type: string,
    params: Record<string, any>
  ): number {
    switch (type) {
      case "dense":
        return params.inputSize || 128;
      case "conv2d":
        return params.inChannels || 3;
      case "lstm":
        return params.inputSize || 256;
      case "attention":
        return params.embedDim || 512;
      case "layernorm":
        return params.normalizedShape || 512;
      case "transformer_block":
        return params.embedDim || 512;
      default:
        return 256;
    }
  }
}

/**
 * Auto-Fixer für Graph-Probleme
 */
export class GraphAutoFixer {
  static applyFix(
    fix: FixSuggestion,
    nodes: Node[],
    edges: Edge[],
    onNodeUpdate: (node: Node) => void,
    onEdgeCreate?: (edge: Edge) => void
  ): boolean {
    switch (fix.action) {
      case "remove_node":
        return this.removeNode(fix.targetNodeId, nodes, edges, onNodeUpdate);

      case "adjust_params":
        return this.adjustNodeParams(
          fix.targetNodeId,
          fix.params || {},
          nodes,
          onNodeUpdate
        );

      case "insert_bridge":
        return this.insertBridgeLayer(
          fix.targetNodeId,
          fix.params || {},
          nodes,
          edges,
          onNodeUpdate,
          onEdgeCreate
        );

      case "reorder_nodes":
        return this.reorderNodes(fix.params || {}, nodes, edges, onNodeUpdate);

      default:
        return false;
    }
  }

  private static removeNode(
    nodeId: string,
    nodes: Node[],
    edges: Edge[],
    onNodeUpdate: (node: Node) => void
  ): boolean {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return false;

    // Mark node as removed
    node.hidden = true;
    onNodeUpdate(node);
    return true;
  }

  private static adjustNodeParams(
    nodeId: string,
    params: Record<string, any>,
    nodes: Node[],
    onNodeUpdate: (node: Node) => void
  ): boolean {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return false;

    // Update node params
    if (node.data) {
      node.data.params = { 
        ...(typeof node.data.params === 'object' ? node.data.params : {}), 
        ...params 
      };
      onNodeUpdate(node);
      return true;
    }
    return false;
  }

  private static insertBridgeLayer(
    beforeNodeId: string,
    params: Record<string, any>,
    nodes: Node[],
    edges: Edge[],
    onNodeUpdate: (node: Node) => void,
    onEdgeCreate?: (edge: Edge) => void
  ): boolean {
    // Erstelle neue Bridge-Layer
    const bridgeNode: Node = {
      id: `bridge_${Date.now()}`,
      data: {
        label: "Dense Bridge",
        nodeType: "dense",
        params: {
          inputSize: params.inputSize || 256,
          outputSize: params.outputSize || 256,
          bias: true,
          initializer: "xavier_uniform",
        },
      },
      position: { x: 0, y: 0 },
    };

    onNodeUpdate(bridgeNode);

    // Erstelle Kanten
    const incomingEdges = edges.filter((e) => e.target === beforeNodeId);
    incomingEdges.forEach((edge) => {
      const newEdge: Edge = {
        ...edge,
        id: `${edge.source}-bridge`,
        target: bridgeNode.id,
      };
      onEdgeCreate?.(newEdge);
    });

    // Verbinde Bridge mit Target-Node
    const bridgeToTargetEdge: Edge = {
      id: `bridge-${beforeNodeId}`,
      source: bridgeNode.id,
      target: beforeNodeId,
    };
    onEdgeCreate?.(bridgeToTargetEdge);

    return true;
  }

  private static reorderNodes(
    params: Record<string, any>,
    nodes: Node[],
    edges: Edge[],
    onNodeUpdate: (node: Node) => void
  ): boolean {
    // TODO: Implement node reordering
    return false;
  }
}

/**
 * Training Result Analyzer - Komplett Token-Optimiert
 */
export class TrainingAnalyzer {
  /**
   * Kompakte Analyse der Training-Ergebnisse
   * Gibt nur relevante Infos zurück
   */
  static analyzeResult(result: TrainingResult): {
    status: "success" | "shape_error" | "memory_error" | "failed";
    message: string;
    diagnosis?: GraphDiagnosis;
    fixes?: FixSuggestion[];
  } {
    if (result.success) {
      return {
        status: "success",
        message: `✓ Training erfolgreich! (${result.epochs} Epochs in ${Math.round(result.duration / 1000)}s, Loss: ${result.finalLoss?.toFixed(4)})`,
      };
    }

    const parsed = ErrorParser.parseTrainingError(result.error || "");

    if (parsed.type === "shape_mismatch") {
      return {
        status: "shape_error",
        message: `✗ Shape-Fehler: ${parsed.root}`,
      };
    }

    if (parsed.type === "memory_error") {
      return {
        status: "memory_error",
        message: `✗ Memory-Fehler: Reduziere batch_size oder modell_complexity`,
      };
    }

    return {
      status: "failed",
      message: `✗ Training fehlgeschlagen: ${parsed.root}`,
    };
  }

  /**
   * Extrahiere nur die wichtigsten Informationen aus JSON-Logs
   */
  static extractKeyMetrics(logJson: string): any {
    try {
      const logs = JSON.parse(logJson);
      if (!Array.isArray(logs)) return null;

      // Sammle nur relevante Events
      const summary = {
        startTime: null as string | null,
        endTime: null as string | null,
        status: [] as string[],
        errors: [] as string[],
        finalMetrics: {} as Record<string, any>,
      };

      logs.forEach((log: any) => {
        if (log.timestamp && !summary.startTime) summary.startTime = log.timestamp;
        if (log.timestamp) summary.endTime = log.timestamp;

        if (log.type === "error") {
          summary.errors.push(log.data?.error || "Unknown error");
        }
        if (log.type === "status" && log.data?.message) {
          // Nur wichtige Status-Messages
          if (
            log.data.message.includes("✓") ||
            log.data.message.includes("✅") ||
            log.data.message.includes("Epoch")
          ) {
            summary.status.push(log.data.message);
          }
        }
        if (log.type === "complete") {
          summary.finalMetrics = log.data?.final_metrics || {};
        }
      });

      return summary;
    } catch (e) {
      return null;
    }
  }
}

/**
 * AI Coach - Hauptschnittstelle für Interaktion
 */
export class SynapseAICoach {
  private lastTrainingResult: TrainingResult | null = null;
  private lastDiagnosis: GraphDiagnosis | null = null;

  setLastTrainingResult(result: TrainingResult) {
    this.lastTrainingResult = result;
  }

  /**
   * Antworte auf User-Fragen zum Training
   * Token-effizient: Gibt nur relevante Infos zurück
   */
  respondToQuestion(question: string): {
    answer: string;
    suggestFixes?: FixSuggestion[];
    needsUserConfirmation?: boolean;
  } {
    const q = question.toLowerCase();
    const result = this.lastTrainingResult;
    const analysis = result ? TrainingAnalyzer.analyzeResult(result) : null;
    const errorType = result?.error
      ? ErrorParser.parseTrainingError(result.error).type
      : null;

    // Fehlertyp-spezifische Handlungsempfehlung (statt "verstehe nicht")
    const guidance = (): string => {
      switch (errorType) {
        case "dataset_error":
          return SynapseAICoach.datasetStructureHelp();
        case "memory_error":
          return "Speicher-Problem: Batch-Size halbieren, Bildgröße reduzieren oder andere Apps schließen.";
        case "shape_mismatch":
        case "dimension_error":
          return "Shape-Problem: Nutze im Fehler-Banner »Mit AI beheben« — die Synapse-AI kennt die betroffenen Nodes und Parameter.";
        case "nan_error":
          return "NaN im Loss: Learning Rate deutlich senken (z.B. /10) oder Gradient Clipping aktivieren.";
        default:
          return "Details stehen im Fehlertext der Trainings-Konsole. Über »An FrameTrain senden« kannst du den Fehler auch ans Team melden.";
      }
    };

    // "Wie muss das Dataset aussehen?" / "How should the dataset be structured?"
    if (q.includes("dataset") || q.includes("aufgebaut") || q.includes("structured")) {
      return { answer: SynapseAICoach.datasetStructureHelp() };
    }

    // "Was sind die Probleme?" / "What are the issues?"
    if (q.includes("problem") || q.includes("issue") || q.includes("was sind")) {
      const issues = this.lastDiagnosis?.issues ?? [];
      if (issues.length > 0) {
        return {
          answer: `Gefundene Probleme:\n${issues.map((i) => `• ${i.description}`).join("\n")}`,
          suggestFixes: this.lastDiagnosis?.suggestions,
        };
      }
      if (result && !result.success && analysis) {
        return { answer: `${analysis.message}\n\n${guidance()}` };
      }
      return { answer: "✓ Keine Probleme gefunden!" };
    }

    // "Fixe Shape-Fehler" / "Fix shape errors" / "Was soll ich ändern?" / "What should I change?"
    if (
      q.includes("fix") || q.includes("behe") || q.includes("änder") || q.includes("change")
    ) {
      const suggestions = this.lastDiagnosis?.suggestions ?? [];
      if (suggestions.length > 0) {
        const topFix = suggestions[0];
        return {
          answer: `Vorschlag: ${topFix.title}\n${topFix.description}`,
          suggestFixes: [topFix],
          needsUserConfirmation: true,
        };
      }
      if (result && !result.success) {
        return { answer: guidance() };
      }
      return { answer: "✓ Nichts zu fixen — das Training war erfolgreich." };
    }

    // "War das Training erfolgreich?" / "Was training successful?"
    if (q.includes("erfolgreich") || q.includes("success") || q.includes("training")) {
      if (!result || !analysis) {
        return { answer: "Ich habe keine Training-Informationen. Führe erst ein Training durch." };
      }
      return {
        answer: analysis.message,
        suggestFixes:
          analysis.status === "shape_error" ? this.lastDiagnosis?.suggestions : undefined,
      };
    }

    // Fallback: statt "verstehe nicht" den Fehler + passende Hilfe zeigen
    if (result && !result.success && analysis) {
      return { answer: `${analysis.message}\n\n${guidance()}` };
    }
    return {
      answer: "Frag mich z.B.: Was sind die Probleme? · Was soll ich ändern? · War das Training erfolgreich?",
    };
  }

  /** Erklärt die erwartete Dataset-Struktur für den Canvas-image_loader. */
  static datasetStructureHelp(): string {
    return [
      "Der image_loader trainiert Bild-KLASSIFIKATION und erwartet:",
      "",
      "dataset/",
      "  hund/    bild1.jpg …",
      "  katze/   bild2.jpg …",
      "",
      "oder mit fertigen Splits:",
      "",
      "dataset/",
      "  train/hund/…   train/katze/…",
      "  val/hund/…     val/katze/…",
      "",
      "⚠ YOLO-Datasets (train/images/ + train/labels/ + dataset.yaml) sind",
      "Objekterkennungs-Daten — dafür das YOLO-Training im Training-Panel",
      "nutzen, nicht den Canvas-image_loader.",
      "Parquet-/CSV-Datasets brauchen den parquet_loader- bzw. csv_loader-Node.",
    ].join("\n");
  }

  analyzeDiagnosis(diagnosis: GraphDiagnosis) {
    this.lastDiagnosis = diagnosis;
  }
}
