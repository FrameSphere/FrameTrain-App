import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  Connection,
  BackgroundVariant,
  ReactFlowInstance,
  Panel,
  ReactFlowProvider,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import { NodeLibrary } from "./NodeLibrary";
import { PropertyPanel } from "./PropertyPanel";
import { TrainingConsole, TrainingConfig, TrainingMetrics, TrainingStatus } from "./TrainingConsole";
import { SessionLibrary } from "./SessionLibrary";
import type { SavedSession } from "./SessionLibrary";
import { NODE_DEFINITIONS, NodeDefinition } from "./nodeTypes";
import { SynapseNodeComponent } from "./nodes/SynapseNodeComponent";
import { dragState } from "./dragState";
import { generateTrainingScript, extractCanvasModelClass } from "./codeGenerator";
import { buildCanvasGraphIR } from "./graphIR";
import { generateModelConfigFromGraph } from "./graphToModel";
import {
  validateEdgeConnection,
  validateFullGraph,
  formatValidationError,
  type ValidationError,
  ValidationErrorType,
} from "./graph-shape-validation";
import { ShapeErrorBanner } from "./ShapeErrorBanner";
import {
  buildShapeAgentPrompt,
  buildShapeUserGuide,
  applyShapeHighlightsToNodes,
  applyShapeHighlightsToEdges,
  clearShapeHighlights,
  getAffectedNodes,
  validationErrorsFromRuntimeDiag,
  collectAffectedNodeIds,
  type ShapeIssueSource,
} from "./ai/synapseShapeDiagnostics";
import { ModelLibrary, writeModelDesign } from "./ModelLibrary";
import { SynapseAICoachPanel } from "./ai/SynapseAICoachPanel";
import type { TrainingResult, FixSuggestion } from "./ai/SynapseAICoach";
import { applyAutoFix } from "./ai/autoFixHelper";
import { exportCanvasNetworkToModelLibrary, updateCanvasNetworkModel } from "./canvasModelBridge";
import { runSynapseAgent, stripToolCallTags } from "./ai/synapseAgent";
import type { AgentStep, AgentResumeState } from "./ai/synapseAgent";
import { createToolExecutor, type GraphMutationEvent } from "./ai/synapseAgentTools";
import { buildSynapseGraphContext } from "./ai/synapseGraphContext";
import { SynapseAIPanel } from "./ai/SynapseAIPanel";
import "./ai/synapseAIPanel.css";
import { useAISettings } from "../../contexts/AISettingsContext";
import { useLanguage } from "../../contexts/LanguageContext";
import type { ChatMessage } from "../../ai/aiClient";

// ─── Constants ────────────────────────────────────────────────────────────────
const nodeTypes = { synapseNode: SynapseNodeComponent };

// ─── Build a React-Flow node from a NodeDefinition ────────────────────────────
function makeNode(def: NodeDefinition, position: { x: number; y: number }): Node {
  const params: Record<string, unknown> = {};
  def.params.forEach((p) => (params[p.key] = p.default));
  return {
    id: `${def.type}-${Date.now()}`,
    type: "synapseNode",
    position,
    data: {
      _def: def,
      label: def.label,
      category: def.category,
      icon: def.icon,
      color: def.color,
      inputs: def.inputs ?? [],
      outputs: def.outputs ?? [],
      paramDefs: def.params ?? [],
      params,
    },
  };
}

// ─── Metric line parser  "[Metric] epoch=1 loss=0.52..." ─────────────────────
function parseMetricLine(line: string): TrainingMetrics | null {
  if (!line.startsWith("[Metric]")) return null;
  try {
    const kv: Record<string, string> = {};
    for (const part of line.replace("[Metric]", "").trim().split(/\s+/)) {
      const [k, v] = part.split("=");
      if (k && v) kv[k] = v;
    }
    if (!kv.epoch) return null;
    return {
      epoch:    parseInt(kv.epoch,  10),
      loss:     parseFloat(kv.loss     ?? "0"),
      valLoss:  kv.val_loss  ? parseFloat(kv.val_loss)  : undefined,
      accuracy: kv.accuracy  ? parseFloat(kv.accuracy)  : undefined,
      lr:       parseFloat(kv.lr ?? "0"),
    };
  } catch {
    return null;
  }
}

// ─── React Flow dark-theme CSS overrides ──────────────────────────────────────
const RF_CSS = `
  .react-flow__controls{background:#0d1117!important;border:1px solid #1e293b!important;border-radius:8px!important;box-shadow:none!important}
  .react-flow__controls-button{background:#0d1117!important;border:none!important;border-bottom:1px solid #1e293b!important;fill:#475569!important;color:#475569!important;width:26px!important;height:26px!important}
  .react-flow__controls-button:last-child{border-bottom:none!important;border-radius:0 0 7px 7px!important}
  .react-flow__controls-button:first-child{border-radius:7px 7px 0 0!important}
  .react-flow__controls-button:hover{background:#1e293b!important;fill:#94a3b8!important;color:#94a3b8!important}
  .react-flow__controls-button svg{fill:inherit!important}
  .react-flow__minimap{background:#0a0e17!important;border:1px solid #1e293b!important;border-radius:8px!important}
`;

interface SynapseBuilderProps {
  userId: string;
}

// Training path: Canvas graph → Graph IR (JSON) → start_training → train_engine canvas runtime.
// generateTrainingScript is export-only (not used for training).

// ─────────────────────────────────────────────────────────────────────────────
// Inner component — needs ReactFlowProvider context
// ─────────────────────────────────────────────────────────────────────────────
const SynapseBuilderInner: React.FC<SynapseBuilderProps> = ({ userId }) => {
  const { t, language } = useLanguage();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [fullscreen, setFullscreen]         = useState(false);
  const [showLibrary, setShowLibrary]         = useState(false);
  const [libraryTab, setLibraryTab]           = useState<"sessions" | "models">("sessions");
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [completedVersionId, setCompletedVersionId] = useState<string | null>(null);
  const [showAICoach, setShowAICoach]         = useState(false);
  const [trainingResult, setTrainingResult]   = useState<TrainingResult | null>(null);
  const [trainStartedAt, setTrainStartedAt]   = useState<number | null>(null);
  const trainStartedAtRef = useRef<number | null>(null);

  // AI panel state
  const { settings: aiSettings } = useAISettings();
  const [showAiPanel, setShowAiPanel]       = useState(false);
  const [aiMessages, setAiMessages]         = useState<ChatMessage[]>([]);
  const [aiInput, setAiInput]               = useState("");
  const [aiLoading, setAiLoading]           = useState(false);
  const [aiError, setAiError]               = useState<string | null>(null);
  const [aiSteps, setAiSteps]               = useState<AgentStep[]>([]);
  const [aiResumeState, setAiResumeState]   = useState<AgentResumeState | null>(null);
  const [shapeErrors, setShapeErrors]       = useState<ValidationError[]>([]);
  const [shapeIssueSource, setShapeIssueSource] = useState<ShapeIssueSource | null>(null);
  const [shapeBannerVisible, setShapeBannerVisible] = useState(false);
  const aiAbortRef                          = useRef<AbortController | null>(null);

  // ── Export-Erfolg Modal ─────────────────────────────────────────────
  const [exportModal, setExportModal] = useState<{ modelId: string; name: string } | null>(null);
  const [activeCanvasModelId, setActiveCanvasModelId] = useState<string | null>(null);
  const [activeModelName,     setActiveModelName]     = useState<string | null>(null);
  // Training state
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("idle");
  const [metrics, setMetrics]               = useState<TrainingMetrics[]>([]);
  const [logLines, setLogLines]             = useState<string[]>([]);
  const [outputDir, setOutputDir]           = useState<string>("/tmp/synapse_output");

  const rfRef          = useRef<ReactFlowInstance | null>(null);
  const cancelRef      = useRef(false);
  const fitTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unlistenersRef = useRef<Array<() => void>>([]);

  // ── Resolve output dir from Tauri once ────────────────────────────────
  useEffect(() => {
    // Try @tauri-apps/api/path first, fallback to a sensible temp dir
    import("@tauri-apps/api/path")
      .then(({ appDataDir }) => appDataDir())
      .then((dir) => setOutputDir(`${dir}/synapse_output`))
      .catch(() =>
        invoke<string>("get_app_data_dir")
          .then((dir) => setOutputDir(`${dir}/synapse_output`))
          .catch(() => setOutputDir("/tmp/synapse_output"))
      );
  }, []);

  // ── Track unsaved changes ─────────────────────────────────────────────
  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) setHasUnsavedChanges(true);
  }, [nodes, edges]);
  useEffect(() => {
    if (nodes.length === 0) return;
    if (fitTimerRef.current) clearTimeout(fitTimerRef.current);
    fitTimerRef.current = setTimeout(() => {
      rfRef.current?.fitView({ duration: 350, padding: 0.18, maxZoom: 1.1 });
    }, 120);
    return () => { if (fitTimerRef.current) clearTimeout(fitTimerRef.current); };
  }, [nodes.length]);

  // ── Cleanup event listeners ────────────────────────────────────────────
  const cleanupListeners = useCallback(() => {
    unlistenersRef.current.forEach((u) => u());
    unlistenersRef.current = [];
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────
  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? null;
  const selectedDef  = selectedNode
    ? ((selectedNode.data as any)._def as NodeDefinition) ?? null
    : null;

  const clearShapeIssues = useCallback(() => {
    setShapeErrors([]);
    setShapeIssueSource(null);
    setShapeBannerVisible(false);
    setNodes((nds) => clearShapeHighlights(nds, edges).nodes);
    setEdges((eds) => clearShapeHighlights(nodes, eds).edges);
  }, [nodes, edges, setNodes, setEdges]);

  const focusShapeNodesByErrors = useCallback((errs: ValidationError[]) => {
    const ids = collectAffectedNodeIds(errs);
    if (ids.size === 0 || !rfRef.current) return;
    rfRef.current.fitView({
      nodes: [...ids].map((id) => ({ id })),
      duration: 500,
      padding: 0.35,
      maxZoom: 1.15,
    });
  }, []);

  const reportShapeIssues = useCallback(
    (
      errors: ValidationError[],
      source: ShapeIssueSource,
      currentNodes: Node[],
      currentEdges: Edge[],
      opts?: { openPanel?: boolean; extraContext?: string }
    ) => {
      const errOnly = errors.filter((e) => e.severity === "error");
      if (errOnly.length === 0) return;

      setShapeErrors(errOnly);
      setShapeIssueSource(source);
      setShapeBannerVisible(true);
      setNodes((nds) => applyShapeHighlightsToNodes(nds, errOnly));
      setEdges((eds) => applyShapeHighlightsToEdges(eds, errOnly));

      const msgs = errOnly.map(formatValidationError).join("\n");
      setLogLines((p) => [...p, `[Shape] ${msgs}`]);

      if (opts?.openPanel !== false) {
        setShowAiPanel(true);
        setAiInput(
          buildShapeAgentPrompt(
            errOnly,
            currentNodes,
            currentEdges,
            source,
            opts?.extraContext
          )
        );
      }

      const focusId = errOnly[0]?.targetNodeId ?? errOnly[0]?.sourceNodeId;
      if (focusId) setSelectedNodeId(focusId);

      setTimeout(() => focusShapeNodesByErrors(errOnly), 80);
    },
    [setNodes, setEdges, focusShapeNodesByErrors]
  );

  // ── Connections (Phase 3: shape validation) ─────────────────────────────
  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target) return;
      const { valid, errors } = validateEdgeConnection(
        c.source, c.target, nodes, NODE_DEFINITIONS
      );
      if (!valid) {
        reportShapeIssues(errors, "connection", nodes, edges, { openPanel: false });
        return;
      }
      setEdges((eds) =>
        addEdge({ ...c, animated: true, style: { stroke: "#a78bfa", strokeWidth: 1.5 } }, eds)
      );
    },
    [setEdges, nodes, edges, reportShapeIssues]
  );

  // ── Add node via click in library ──────────────────────────────────────
  const handleAddNode = useCallback(
    (def: NodeDefinition, pos?: { x: number; y: number }) => {
      let position = pos ?? { x: 300, y: 200 };
      if (rfRef.current && !pos) {
        const { x, y, zoom } = rfRef.current.getViewport();
        position = {
          x: (-x + window.innerWidth  / 2) / zoom + (Math.random() - 0.5) * 120,
          y: (-y + window.innerHeight / 2) / zoom + (Math.random() - 0.5) * 80,
        };
      }
      setNodes((nds) => [...nds, makeNode(def, position)]);
    },
    [setNodes]
  );

  // ── Drag & Drop ───────────────────────────────────────────────────────
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const nodeType = dragState.nodeType ?? e.dataTransfer.getData("text/plain");
      dragState.nodeType = null;
      if (!nodeType) return;
      const def = NODE_DEFINITIONS.find((d) => d.type === nodeType);
      if (!def || !rfRef.current) return;
      const position = rfRef.current.screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setNodes((nds) => [...nds, makeNode(def, position)]);
    },
    [setNodes]
  );

  // ── Select ────────────────────────────────────────────────────────────
  const onNodeClick  = useCallback((_: React.MouseEvent, node: Node) => setSelectedNodeId(node.id), []);
  const onPaneClick  = useCallback(() => setSelectedNodeId(null), []);

  // ── Param change ──────────────────────────────────────────────────────
  const onParamChange = useCallback(
    (nodeId: string, key: string, value: unknown) => {
      setNodes((nds) => {
        const next = nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...((n.data as any).params ?? {}), [key]: value } } }
            : n
        );
        const check = validateFullGraph(next, edges, NODE_DEFINITIONS);
        if (check.valid) {
          queueMicrotask(() => clearShapeIssues());
        } else if (shapeErrors.length > 0) {
          queueMicrotask(() =>
            reportShapeIssues(check.errors, shapeIssueSource ?? "graph", next, edges, {
              openPanel: false,
            })
          );
        }
        return next;
      });
    },
    [setNodes, edges, clearShapeIssues, reportShapeIssues, shapeErrors.length, shapeIssueSource]
  );

  // ── Auto-fix shape error from DIAGNOSTIC_JSON ─────────────────────────
  // Finds the Dense node whose inputSize matches expected_input_features
  // and corrects it to actual_output_features. No AI needed for this case.
  const autoFixShapeError = useCallback(
    (diag: Record<string, any>): boolean => {
      const actual   = diag.actual_output_features;
      const expected = diag.expected_input_features;
      if (!actual || !expected || actual === expected) return false;

      let fixed = false;
      setNodes((nds) =>
        nds.map((n) => {
          if (fixed) return n;
          const d = n.data as any;
          const t = d._def?.type ?? d.nodeType;
          if (t === "dense" && Number(d.params?.inputSize) === Number(expected)) {
            fixed = true;
            return {
              ...n,
              data: {
                ...n.data,
                params: { ...d.params, inputSize: actual },
              },
            };
          }
          return n;
        })
      );

      if (fixed) {
        clearShapeIssues();
        setLogLines((p) => [
          ...p,
          `[Synapse] ✅ Auto-Fix: Dense inputSize ${expected} → ${actual}`,
        ]);
        setTrainingStatus("idle");
      }
      return fixed;
    },
    [setNodes, clearShapeIssues]
  );

  const resolveDatasetPath = useCallback(async (datasetId?: string): Promise<string> => {
    if (!datasetId) return "";
    try {
      return await invoke<string>("get_dataset_path", { datasetId });
    } catch (e) {
      console.warn("[Synapse] get_dataset_path:", e);
      return "";
    }
  }, []);

  const registerTrainingVersion = useCallback(async (modelName: string, outDir: string) => {
    try {
      const res = await invoke<{ version_id: string; model_id: string }>(
        "register_synapse_training_version",
        { modelName, outputDir: outDir }
      );
      setCompletedVersionId(res.version_id);
      setLogLines((p) => [
        ...p,
        `[Synapse] ✓ In Modellbibliothek registriert (Version: ${res.version_id})`,
      ]);
    } catch (e) {
      console.warn("[Synapse] register_synapse_training_version:", e);
      setLogLines((p) => [...p, `[Synapse] Hinweis: Version nicht in DB registriert: ${e}`]);
    }
  }, []);

  const parseStatusLogLine = useCallback((message: string) => {
    setLogLines((p) => [...p, message]);
    const metric = parseMetricLine(message);
    if (metric) setMetrics((m) => [...m, metric]);
    const diagMatch = message.match(/\[DIAGNOSTIC_JSON\]\s*(\{[\s\S]*?\})\s*\[\/DIAGNOSTIC_JSON\]/);
    if (diagMatch) {
      try {
        const diag = JSON.parse(diagMatch[1]);
        if (diag.error_type === "shape_mismatch") {
          const fixed = autoFixShapeError(diag);
          if (!fixed) {
            const errs = validationErrorsFromRuntimeDiag(diag, nodes);
            reportShapeIssues(errs, "runtime", nodes, edges, {
              extraContext: String(diag.raw_error ?? ""),
              openPanel: false,
            });
          }
        }
      } catch { /* ignore */ }
    }
  }, [autoFixShapeError, nodes, edges, reportShapeIssues]);

  // ── Runtime training: Canvas → Graph IR → start_training ───────────────
  const handleStartTraining = useCallback(
    async (config: TrainingConfig & { selectedDatasetId?: string; selectedDatasetName?: string }) => {
      if (nodes.length === 0) {
        setLogLines(["[Error] Keine Nodes im Graph. Füge Nodes hinzu und verbinde sie."]);
        setTrainingStatus("error");
        return;
      }

      const graphCheck = validateFullGraph(nodes, edges, NODE_DEFINITIONS);
      if (!graphCheck.valid) {
        const msgs = graphCheck.errors
          .filter((e) => e.severity === "error")
          .map(formatValidationError);
        setLogLines(["[Shape] Graph ungültig — Training abgebrochen:", ...msgs]);
        setTrainingStatus("error");
        reportShapeIssues(graphCheck.errors, "graph", nodes, edges, { openPanel: false });
        return;
      }
      clearShapeIssues();

      const datasetPath = await resolveDatasetPath(config.selectedDatasetId);
      if (config.selectedDatasetId && !datasetPath) {
        setLogLines((p) => [
          ...p,
          `[Warnung] Dataset "${config.selectedDatasetName ?? config.selectedDatasetId}" — Pfad nicht gefunden, nutze Dummy-Daten.`,
        ]);
      }

      const canvasGraph = buildCanvasGraphIR(nodes, edges, config);

      setCompletedVersionId(null);
      setShowAICoach(false);
      const started = Date.now();
      setTrainStartedAt(started);
      trainStartedAtRef.current = started;
      const jobIdRef = { current: "" as string };
      const trainOutputRef = { current: "" as string };

      cancelRef.current = false;
      cleanupListeners();
      setTrainingStatus("running");
      setMetrics([]);
      setLogLines([
        "[Synapse] Graph-IR erstellt...",
        `[Synapse] ${canvasGraph.nodes.length} nodes · ${canvasGraph.edges.length} edges`,
        "[Synapse] Starte Runtime-Training (train_engine / canvas)...",
      ]);

      try {
        const [ul1, ul2, ul3, ul4] = await Promise.all([
          listen<{ data?: { epoch?: number; train_loss?: number; val_loss?: number; learning_rate?: number } }>(
            "training-progress",
            (e) => {
              const d = e.payload?.data;
              if (!d) return;
              if (d.train_loss != null && d.epoch != null) {
                setMetrics((m) => [
                  ...m,
                  {
                    epoch: d.epoch ?? m.length + 1,
                    loss: d.train_loss ?? 0,
                    valLoss: d.val_loss ?? undefined,
                    lr: d.learning_rate ?? 0,
                  },
                ]);
              }
            }
          ),

          listen<{ data?: { status?: string; message?: string } }>("training-status", (e) => {
            const msg = e.payload?.data?.message;
            if (msg) parseStatusLogLine(msg);
          }),

          listen<{ new_version_id?: string; data?: { output_path?: string } }>("training-complete", (e) => {
            setTrainingStatus("done");
            const duration = trainStartedAtRef.current ? Date.now() - trainStartedAtRef.current : 0;
            setTrainingResult({
              success: true,
              jobId: jobIdRef.current,
              duration,
              epochs: config.epochs,
              timestamp: Date.now(),
            });
            const outPath = trainOutputRef.current || outputDir;
            setLogLines((p) => [
              ...p,
              `[Synapse] Training abgeschlossen!`,
              `[Synapse] Outputs: ${outPath}`,
              `[Synapse] model.pt · metrics.json (Runtime IR)`,
            ]);
            if (e.payload?.new_version_id) {
              setCompletedVersionId(e.payload.new_version_id);
            } else {
              registerTrainingVersion("Synapse Canvas Model", outPath).catch(() => {});
            }
            invoke("disable_prevent_sleep").catch(() => {});
            cleanupListeners();
          }),

          listen<{ data?: { error?: string; details?: string } }>("training-error", (e) => {
            const err = e.payload?.data?.error ?? "Unbekannter Fehler";
            const det = e.payload?.data?.details ?? "";
            const full = det ? `${err}\n${det}` : err;
            setLogLines((p) => [...p, `[ERROR] ${err}`, ...(det ? [`[Details] ${det}`] : [])]);
            setTrainingStatus("error");
            const isShape = /shape|dimension|multiplied|mismatch|mat1|mat2/i.test(full);
            if (isShape) {
              const graphCheck = validateFullGraph(nodes, edges, NODE_DEFINITIONS);
              const errs =
                graphCheck.errors.filter((x) => x.severity === "error").length > 0
                  ? graphCheck.errors
                  : [
                      {
                        type: ValidationErrorType.DIMENSION_ERROR,
                        severity: "error" as const,
                        sourceNodeId: nodes[0]?.id ?? "graph",
                        message: full.slice(0, 500),
                        suggestion:
                          "Parameter mit set_param anpassen (inputSize, inChannels, normalizedShape, embedDim)",
                      } satisfies ValidationError,
                    ];
              reportShapeIssues(errs, "training", nodes, edges, { extraContext: full, openPanel: false });
            } else {
              setShowAICoach(true);
              setTrainingResult({
                success: false,
                jobId: jobIdRef.current,
                duration: trainStartedAtRef.current ? Date.now() - trainStartedAtRef.current : 0,
                epochs: config.epochs,
                error: full,
                errorType: "runtime",
                timestamp: Date.now(),
              });
            }
            invoke("disable_prevent_sleep").catch(() => {});
            cleanupListeners();
          }),
        ]);

        unlistenersRef.current = [ul1, ul2, ul3, ul4];

        const job = await invoke<{
          id: string;
          output_path?: string | null;
        }>("start_training", {
          modelId: "",
          modelName: "Synapse Canvas Model",
          datasetId: config.selectedDatasetId ?? "",
          datasetName: config.selectedDatasetName ?? "",
          versionId: null,
          config: {
            task_type: "canvas",
            canvas_graph: canvasGraph,
            epochs: config.epochs,
            batch_size: config.batchSize,
            learning_rate: config.learningRate,
            fp16: config.precision === "fp16",
            bf16: config.precision === "bf16",
            dataset_path: datasetPath,
            optimizer: canvasGraph.training.optimizer,
            scheduler: canvasGraph.training.scheduler,
          },
        });

        jobIdRef.current = job.id;
        if (job.output_path) {
          trainOutputRef.current = `${job.output_path.replace(/\/$/, "")}/final_model`;
        }
        invoke("enable_prevent_sleep").catch(() => {});
      } catch (err: unknown) {
        const msg = String(err);
        setLogLines((p) => [
          ...p,
          `[ERROR] Training konnte nicht gestartet werden: ${msg}`,
        ]);
        setTrainingStatus("error");
        cleanupListeners();
      }
    },
    [
      nodes,
      edges,
      outputDir,
      cleanupListeners,
      reportShapeIssues,
      clearShapeIssues,
      resolveDatasetPath,
      registerTrainingVersion,
      parseStatusLogLine,
    ]
  );

  const handleApplyCoachFix = useCallback(
    (fix: FixSuggestion) => {
      applyAutoFix(fix, nodes, edges, setNodes, setEdges);
      setLogLines((p) => [...p, `[AI Coach] Fix angewendet: ${fix.title}`]);
    },
    [nodes, edges, setNodes, setEdges]
  );

  const handleSaveToModelLibrary = useCallback(async () => {
    const graphConfig = generateModelConfigFromGraph(nodes, edges, NODE_DEFINITIONS);
    if (!graphConfig) {
      setLogLines((p) => [...p, "[Error] Kein gültiger Graph zum Speichern."]);
      return;
    }
    const script = generateTrainingScript(nodes, edges, {
      epochs: 1, batchSize: 32, learningRate: 0.001, gpu: "cpu", precision: "fp32", gradAccum: 1,
    });
    const pythonCode = extractCanvasModelClass(script);
    const graphIR = buildCanvasGraphIR(nodes, edges, {
      epochs: 10, batchSize: 32, learningRate: 0.001, gpu: "cpu", precision: "fp32", gradAccum: 1,
    });
    const design = {
      nodes,
      edges,
      viewport: rfRef.current?.getViewport(),
      graphConfig,
      pythonCode,
      schemaVersion: 1,
    };

    // ── UPDATE: bestehendes Modell überschreiben ────────────────────────────────
    if (activeCanvasModelId) {
      try {
        await updateCanvasNetworkModel(activeCanvasModelId, graphConfig, pythonCode, graphIR);
        await writeModelDesign(activeCanvasModelId, design);
        setHasUnsavedChanges(false);
        setLogLines((p) => [...p, `[Synapse] ✓ Modell „${activeModelName}“ aktualisiert`]);
        return;
      } catch (e) {
        setLogLines((p) => [...p, `[Error] Update fehlgeschlagen: ${e} — lege neues Modell an…`]);
      }
    }

    // ── NEU: neues Modell anlegen ───────────────────────────────────────────────
    const name = `Synapse ${new Date().toLocaleDateString("de-DE")}`;
    try {
      const result = await exportCanvasNetworkToModelLibrary(graphConfig, pythonCode, name, graphIR);
      await writeModelDesign(result.modelId, design);
      setHasUnsavedChanges(false);
      setActiveCanvasModelId(result.modelId);
      setActiveModelName(name);
      setExportModal({ modelId: result.modelId, name });
      setLogLines((p) => [...p, `[Synapse] ✓ Modell „${name}“ exportiert (ID: ${result.modelId.slice(0,14)})`]);
    } catch (e) {
      setLogLines((p) => [...p, `[Error] Speichern fehlgeschlagen: ${e}`]);
    }
  }, [nodes, edges, activeCanvasModelId, activeModelName]);

  const handleStopTraining = useCallback(() => {
    cancelRef.current = true;
    invoke("stop_training").catch(() => {});
    cleanupListeners();
    setTrainingStatus("idle");
    setLogLines((p) => [...p, "[Synapse] Training gestoppt."]);
  }, [cleanupListeners]);

  const handleExport = useCallback(
    (format: string) => {
      setLogLines((p) => [...p, `[Export] Exportiere als ${format}…`]);
      // Open the output dir in Finder/Explorer so user can grab the files
      invoke("open_path_in_finder", { path: outputDir }).catch(() => {
        setLogLines((pp) => [...pp, `[Export] Output-Verzeichnis: ${outputDir}`]);
      });
    },
    [outputDir]
  );

  // ── Fullscreen (CSS-only — requestFullscreen not in Tauri WebView) ─────
  const toggleFullscreen = useCallback(() => setFullscreen((f) => !f), []);

  const handleGraphMutation = useCallback((ev: GraphMutationEvent) => {
    if (ev.type === "node" && ev.nodeId && rfRef.current) {
      requestAnimationFrame(() => {
        rfRef.current?.fitView({
          nodes: [{ id: ev.nodeId! }],
          duration: 480,
          padding: 0.4,
          maxZoom: 1.12,
        });
      });
    }
  }, []);

  // ── AI Send ───────────────────────────────────────────────────────────────
  const aiSend = useCallback(async (msg?: string, resume?: AgentResumeState) => {
    const text = (msg ?? aiInput).trim();
    if (!text && !resume) return;
    if (aiLoading) { aiAbortRef.current?.abort(); return; }

    aiAbortRef.current = new AbortController();
    setAiLoading(true);
    setAiError(null);
    setAiSteps([]);
    setAiResumeState(null);

    const userMsg: ChatMessage = { role: 'user', content: text || 'Resume' };
    const nextMessages = [...aiMessages, userMsg];
    setAiMessages(nextMessages);
    setAiInput("");

    const graphCtx = buildSynapseGraphContext(nodes, edges, selectedNodeId, NODE_DEFINITIONS);
    const executor = createToolExecutor({
      nodes,
      edges,
      setNodes,
      setEdges,
      nodeDefinitions: NODE_DEFINITIONS,
      onGraphMutation: handleGraphMutation,
    });

    try {
      const result = await runSynapseAgent({
        userMessage: text || 'Resume',
        chatHistory: nextMessages,
        aiSettings,
        responseLanguage: language,
        graphContextStr: graphCtx,
        executorHandle: executor,
        getGraphContext: () => buildSynapseGraphContext(
          executor.getState().nodes, executor.getState().edges, selectedNodeId, NODE_DEFINITIONS
        ),
        onStepsUpdate: setAiSteps,
        signal: aiAbortRef.current.signal,
        resumeState: resume,
      });

      // Sync final graph state
      const finalState = executor.getState();
      setNodes([...finalState.nodes]);
      setEdges([...finalState.edges]);

      const cleanSummary = stripToolCallTags(result.summary || "");
      const assistantContent = result.steps.length > 0
        ? `${t('synapseBuilder.aiSummary.executed').replace('{count}', String(result.steps.length))}${cleanSummary ? "\n" + cleanSummary : ""}`
        : cleanSummary || t('synapseBuilder.aiSummary.noChanges');

      setAiMessages(prev => [...prev, { role: 'assistant', content: assistantContent }]);

      if (result.error) setAiError(result.error);
      if (result.canResume && result.resumeState) setAiResumeState(result.resumeState);

      const graphCheck = validateFullGraph(
        finalState.nodes,
        finalState.edges,
        NODE_DEFINITIONS
      );
      if (!result.error && graphCheck.valid) {
        clearShapeIssues();
      }
    } catch (e: any) {
      setAiError(String(e?.message ?? e));
    } finally {
      setAiLoading(false);
    }
  }, [
    aiInput,
    aiLoading,
    aiMessages,
    nodes,
    edges,
    selectedNodeId,
    aiSettings,
    setNodes,
    setEdges,
    handleGraphMutation,
    clearShapeIssues,
  ]);

  // ── Session load ───────────────────────────────────────────────────────
  const handleLoadSession = useCallback(
    (session: SavedSession) => {
      setNodes(session.nodes);
      setEdges(session.edges);
      setSelectedNodeId(null);
      setTimeout(() => {
        if (session.viewport && rfRef.current) {
          rfRef.current.setViewport(session.viewport, { duration: 300 });
        } else {
          rfRef.current?.fitView({ duration: 400, padding: 0.18 });
        }
      }, 50);
    },
    [setNodes, setEdges]
  );

  // ── Delete node with keyboard ──────────────────────────────────────────
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        selectedNodeId &&
        !(e.target as HTMLElement).closest("input,textarea,select")
      ) {
        setNodes((nds) => nds.filter((n) => n.id !== selectedNodeId));
        setEdges((eds) =>
          eds.filter((ed) => ed.source !== selectedNodeId && ed.target !== selectedNodeId)
        );
        setSelectedNodeId(null);
      }
    },
    [selectedNodeId, setNodes, setEdges]
  );

  // ─── Render ─────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        display: "flex", flexDirection: "column",
        height: fullscreen ? "100vh" : "100%",
        width:  fullscreen ? "100vw" : "100%",
        background: "#080c12", overflow: "hidden",
        fontFamily: "'JetBrains Mono','Fira Code',monospace",
        position: fullscreen ? "fixed" : "relative",
        inset:    fullscreen ? 0 : "auto",
        zIndex:   fullscreen ? 9999 : "auto",
      }}
      onKeyDown={onKeyDown}
      tabIndex={0}
    >
      <style>{RF_CSS}</style>

      {/* ── Header ── */}
      <div style={{
        display: "flex", alignItems: "center",
        padding: "0 16px", height: 42,
        borderBottom: "1px solid #1e293b",
        background: "#0a0e17",
        flexShrink: 0, gap: 16,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="2.5" fill="#a78bfa"/>
            <circle cx="2" cy="4"  r="1.5" fill="#64748b"/>
            <circle cx="14" cy="4" r="1.5" fill="#64748b"/>
            <circle cx="2" cy="12" r="1.5" fill="#64748b"/>
            <circle cx="14" cy="12" r="1.5" fill="#64748b"/>
            <line x1="2"  y1="4"  x2="8" y2="8" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.5"/>
            <line x1="14" y1="4"  x2="8" y2="8" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.5"/>
            <line x1="2"  y1="12" x2="8" y2="8" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.5"/>
            <line x1="14" y1="12" x2="8" y2="8" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.5"/>
          </svg>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", letterSpacing: "0.04em" }}>
            Synapse Builder
          </span>
        </div>

        <div style={{ fontSize: 11, color: "#334155", marginLeft: 8 }}>
          {nodes.length} nodes · {edges.length} connections
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          {/* Model Library */}
          <button
            onClick={handleSaveToModelLibrary}
            style={{
              ...btnStyle,
              borderColor: hasUnsavedChanges ? "#eab308" : activeCanvasModelId ? "#10b981" : "#6366f1",
              color:       hasUnsavedChanges ? "#eab308" : activeCanvasModelId ? "#34d399" : "#818cf8",
              background:  hasUnsavedChanges ? "rgba(234,179,8,0.08)" : activeCanvasModelId ? "rgba(16,185,129,0.08)" : "rgba(99,102,241,0.08)",
              boxShadow:   hasUnsavedChanges ? "0 0 8px rgba(234,179,8,0.25)" : "none",
              maxWidth: 160,
            }}
            title={activeCanvasModelId
              ? `Änderungen an „${activeModelName}“ speichern`
              : t('synapseBuilder.exportGraphTooltip')}
          >
            {hasUnsavedChanges
              ? activeCanvasModelId ? `● ↻ ${(activeModelName ?? "Modell").slice(0, 12)}` : t('synapseBuilder.modelsButton')
              : activeCanvasModelId ? `↻ ${(activeModelName ?? "Modell").slice(0, 14)}`      : t('synapseBuilder.modelsButton')}
          </button>
          <button onClick={() => { setLibraryTab("sessions"); setShowLibrary(true); }} style={btnStyle} title={t('synapseBuilder.sessionsModelsButton')}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <rect x="1"   y="1"   width="4.5" height="4.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
              <rect x="6.5" y="1"   width="4.5" height="4.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
              <rect x="1"   y="6.5" width="4.5" height="4.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
              <rect x="6.5" y="6.5" width="4.5" height="4.5" rx="1" stroke="currentColor" strokeWidth="1.3"/>
            </svg>
            {t('synapseBuilder.sessionsModelsButton')}
          </button>

          {/* AI Assistant */}
          <button
            onClick={() => setShowAiPanel((v) => !v)}
            style={{
              ...btnStyle,
              borderColor: showAiPanel ? "#a78bfa" : "#1e293b",
              color: showAiPanel ? "#a78bfa" : "#475569",
              background: showAiPanel ? "rgba(167,139,250,0.08)" : "transparent",
            }}
            title={t('synapseBuilder.aiAssistantButton')}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <circle cx="6" cy="5" r="3" stroke="currentColor" strokeWidth="1.3"/>
              <path d="M4 8.5c0 1.1.9 2 2 2s2-.9 2-2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
              <line x1="6" y1="2" x2="6" y2="1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
            </svg>
            AI
          </button>

          {/* Clear */}
          <button
            onClick={() => { setNodes([]); setEdges([]); setSelectedNodeId(null); setActiveCanvasModelId(null); setActiveModelName(null); }}
            style={btnStyle}
          >
            Clear
          </button>

          {/* Fullscreen */}
          <button onClick={toggleFullscreen} style={btnStyle} title={fullscreen ? "Exit fullscreen" : "Fullscreen"}>
            {fullscreen ? (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M1 4.5H4.5V1M7.5 1V4.5H11M11 7.5H7.5V11M4.5 11V7.5H1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M1 1H4.5M1 1V4.5M11 1H7.5M11 1V4.5M1 11H4.5M1 11V7.5M11 11H7.5M11 11V7.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* ── Main layout ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>
        <NodeLibrary onAddNode={handleAddNode} />

        {/* Canvas — onDrop/onDragOver on <ReactFlow> directly (v12 requirement) */}
        <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onInit={(inst) => { rfRef.current = inst; }}
            nodeTypes={nodeTypes}
            onDrop={onDrop}
            onDragOver={onDragOver}
            fitView
            minZoom={0.05}
            maxZoom={2}
            deleteKeyCode={null}
            proOptions={{ hideAttribution: true }}
            style={{ background: "#080c12" }}
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: "#a78bfa80", strokeWidth: 1.5 },
            }}
          >
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#1e293b"/>
            <Controls/>
            <MiniMap
              nodeColor={(node) => ((node.data as any).color as string) ?? "#a78bfa"}
              nodeStrokeWidth={0}
              maskColor="rgba(8,12,18,0.75)"
              style={{ borderRadius: 8 }}
            />
            {shapeBannerVisible && shapeErrors.length > 0 && (
              <Panel position="top-center">
                <ShapeErrorBanner
                  title={
                    shapeIssueSource === "connection"
                      ? "Verbindung wegen Shape-Konflikt blockiert"
                      : shapeIssueSource === "graph"
                        ? "Graph hat Shape-Fehler — Training nicht möglich"
                        : shapeIssueSource === "training"
                          ? "Training: Shape-Fehler im Netzwerk"
                          : "Runtime Shape-Fehler"
                  }
                  affected={getAffectedNodes(shapeErrors, nodes)}
                  onDismiss={clearShapeIssues}
                  onOpenAI={() => {
                    if (!shapeIssueSource) return;
                    // Neuer Chat — vorherige Nachrichten löschen
                    setAiMessages([]);
                    // Panel öffnen + Fix-Prompt einfüllen
                    reportShapeIssues(shapeErrors, shapeIssueSource, nodes, edges, { openPanel: true });
                  }}
                  onFocusNodes={() => focusShapeNodesByErrors(shapeErrors)}
                />
              </Panel>
            )}
            {nodes.length === 0 && !shapeBannerVisible && (
              <Panel position="top-center">
                <div style={{ marginTop: 80, textAlign: "center", pointerEvents: "none" }}>
                  <div style={{ fontSize: 40, opacity: 0.05, marginBottom: 10 }}>◈</div>
                  <div style={{ fontSize: 12, color: "#1e2d3d", lineHeight: 1.9 }}>
                    Drag nodes from the library<br/>or click a node to place it
                  </div>
                </div>
              </Panel>
            )}
          </ReactFlow>
        </div>

        <PropertyPanel
          selectedNode={selectedNode}
          definition={selectedDef}
          onParamChange={onParamChange}
        />
      </div>

      <TrainingConsole
        onStartTraining={handleStartTraining}
        onStopTraining={handleStopTraining}
        onExport={handleExport}
        status={trainingStatus}
        metrics={metrics}
        logLines={logLines}
        completedVersionId={completedVersionId}
        outputDir={trainingStatus === "done" ? outputDir : undefined}
        userId={userId}
      />

      {showLibrary && libraryTab === "sessions" && (
        <SessionLibrary
          isOpen={showLibrary}
          onClose={() => setShowLibrary(false)}
          currentNodes={nodes}
          currentEdges={edges}
          currentViewport={rfRef.current?.getViewport()}
          onLoad={handleLoadSession}
          onSwitchToModels={() => setLibraryTab("models")}
        />
      )}

      {showLibrary && libraryTab === "models" && (
        <ModelLibrary
          isOpen={showLibrary}
          onClose={() => setShowLibrary(false)}
          userId={userId}
          onLoad={(model) => {
            setNodes(model.nodes);
            setEdges(model.edges);
            setSelectedNodeId(null);
            setActiveCanvasModelId(model.canvasModelId ?? null);
            setActiveModelName(model.name);
            setHasUnsavedChanges(false);
            setTimeout(() => {
              if (model.viewport && rfRef.current) {
                rfRef.current.setViewport(model.viewport, { duration: 300 });
              } else {
                rfRef.current?.fitView({ duration: 400, padding: 0.18 });
              }
            }, 50);
          }}
          onSwitchToSessions={() => setLibraryTab("sessions")}
        />
      )}

      {showAICoach && trainingResult && (
        <div style={{ position: "absolute", right: showAiPanel ? 420 : 0, bottom: 200, zIndex: 600 }}>
          <SynapseAICoachPanel
            trainingResult={trainingResult}
            nodes={nodes}
            edges={edges}
            layerConfig={nodes.map((n) => ({
              type: (n.data as any)?._def?.type,
              params: (n.data as any)?.params,
            }))}
            onApplyFix={handleApplyCoachFix}
            onClose={() => setShowAICoach(false)}
          />
        </div>
      )}

      <SynapseAIPanel
        open={showAiPanel}
        onClose={() => setShowAiPanel(false)}
        userId={userId}
        messages={aiMessages}
        onMessagesChange={setAiMessages}
        input={aiInput}
        onInputChange={setAiInput}
        onSend={aiSend}
        loading={aiLoading}
        error={aiError}
        steps={aiSteps}
        resumeState={aiResumeState}
        onAbort={() => aiAbortRef.current?.abort()}
        shapeMode={shapeErrors.length > 0}
        shapeUserGuide={
          shapeIssueSource
            ? buildShapeUserGuide(shapeErrors, nodes, shapeIssueSource)
            : undefined
        }
        affectedNodes={getAffectedNodes(shapeErrors, nodes)}
      />

      {/* ── Export-Erfolg-Modal ───────────────────────────────────────── */}
      {exportModal && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9000,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
        }} onClick={() => setExportModal(null)}>
          <div style={{
            background: "#0d1117", border: "1px solid #6366f1",
            borderRadius: 12, padding: "28px 32px", maxWidth: 400, width: "90%",
            boxShadow: "0 0 40px rgba(99,102,241,0.3)",
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ fontSize: 28, marginBottom: 8, textAlign: "center" }}>✅</div>
            <h3 style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 700, textAlign: "center", marginBottom: 6 }}>
              {t('synapseBuilder.exportModal.title')}
            </h3>
            <p style={{ color: "#64748b", fontSize: 12, textAlign: "center", marginBottom: 16, lineHeight: 1.7 }}>
              <strong style={{ color: "#a78bfa" }}>{exportModal.name}</strong> {t('synapseBuilder.exportModal.description')}
            </p>
            <div style={{ background: "#080c12", borderRadius: 8, padding: "10px 14px", marginBottom: 16, border: "1px solid #1e293b" }}>
              <div style={{ color: "#475569", fontSize: 10, marginBottom: 4 }}>{t('synapseBuilder.exportModal.nextStepsTitle')}</div>
              <div style={{ color: "#94a3b8", fontSize: 11, lineHeight: 1.9 }}>
                {t('synapseBuilder.exportModal.step1')}<br/>
                {t('synapseBuilder.exportModal.step2').replace('{name}', exportModal.name)}<br/>
                {t('synapseBuilder.exportModal.step3')}<br/>
                {t('synapseBuilder.exportModal.step4')}
              </div>
            </div>
            <p style={{ color: "#334155", fontSize: 10, textAlign: "center", marginBottom: 16 }}>
              ID: <code style={{ color: "#475569" }}>{exportModal.modelId}</code>
            </p>
            <button
              onClick={() => setExportModal(null)}
              style={{
                width: "100%", padding: "8px 0", borderRadius: 6,
                background: "rgba(99,102,241,0.15)", border: "1px solid #6366f1",
                color: "#818cf8", fontSize: 12, cursor: "pointer", fontFamily: "inherit",
              }}
            >
              {t('synapseBuilder.exportModal.confirmButton')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Public export wrapped in ReactFlowProvider ────────────────────────────────
export const SynapseBuilder: React.FC<SynapseBuilderProps> = ({ userId }) => (
  <ReactFlowProvider>
    <SynapseBuilderInner userId={userId} />
  </ReactFlowProvider>
);

const btnStyle: React.CSSProperties = {
  display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
  padding: "4px 10px", background: "transparent",
  border: "1px solid #1e293b", borderRadius: 5,
  color: "#475569", fontSize: 11, cursor: "pointer",
  fontFamily: "'JetBrains Mono',monospace", height: 26, minWidth: 26,
};

export default SynapseBuilder;
