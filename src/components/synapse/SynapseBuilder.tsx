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
import { Save as SaveIcon, Sparkles as SparklesIcon, FolderOpen as LibraryIcon, Trash2 as TrashIcon } from "lucide-react";
import { useContextMenuActions } from "../../ui/contextMenuRegistry";

import { NodeLibrary } from "./NodeLibrary";
import { PropertyPanel } from "./PropertyPanel";
import { TrainingConsole, TrainingConfig, TrainingMetrics, TrainingStatus } from "./TrainingConsole";
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
import { autoLayoutNodes } from "./synapseLayout";
import { SynapseAIPanel } from "./ai/SynapseAIPanel";
import "./ai/synapseAIPanel.css";
import { useAISettings } from "../../contexts/AISettingsContext";
import { useLanguage } from "../../contexts/LanguageContext";
import { usePageContext } from "../../contexts/PageContext";
import { buildPageContext, kv } from "../../ai/coachContext";
import { callAI } from "../../ai/aiClient";
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
  const [shapeExtraContext, setShapeExtraContext] = useState<string | null>(null);
  const aiAbortRef                          = useRef<AbortController | null>(null);
  const lastAiRequestRef                    = useRef<{ text: string; displayText?: string } | null>(null);
  // Rollierende Chat-Komprimierung: Zusammenfassung älterer Nachrichten
  // + Index, bis wohin der Verlauf bereits komprimiert wurde.
  const aiChatSummaryRef                    = useRef<string | null>(null);
  const aiSummaryCoveredRef                 = useRef(0);

  // ── Export-Erfolg Modal ─────────────────────────────────────────────
  const [exportModal, setExportModal] = useState<{ modelId: string; name: string } | null>(null);
  // Namensdialog beim ersten Speichern (statt Auto-Name "Synapse <Datum>")
  const [saveDialogName, setSaveDialogName] = useState<string | null>(null);
  // Bestätigung vor "Clear" bei ungespeicherten Änderungen
  const [confirmClearOpen, setConfirmClearOpen] = useState(false);
  const [activeCanvasModelId, setActiveCanvasModelId] = useState<string | null>(null);
  const [activeModelName,     setActiveModelName]     = useState<string | null>(null);
  // Training state
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("idle");
  const [metrics, setMetrics]               = useState<TrainingMetrics[]>([]);
  const [logLines, setLogLines]             = useState<string[]>([]);

  // ── Seiten-Kontext für den globalen Floating AI Coach ──────────────────
  const { setCurrentPageContent } = usePageContext();
  useEffect(() => {
    setCurrentPageContent(buildPageContext({
      pageId: 'synapse',
      language,
      title: 'Synapse Builder',
      purpose: 'Visueller Node-Builder: ML-Pipelines aus Bausteinen zusammenstecken. Synapse hat einen eigenen, spezialisierten Coach im Panel.',
      state: [
        kv('Nodes', String(nodes.length)),
        kv('Verbindungen', String(edges.length)),
        kv('Ausgewählter Node', selectedNodeId ?? '—'),
        kv('Modell', activeModelName ?? '—'),
        kv('Ungespeicherte Änderungen', hasUnsavedChanges ? 'ja' : 'nein'),
        kv('Training', trainingStatus),
      ],
      actions: [
        'Nodes aus der Bibliothek ziehen und verbinden',
        'Node anklicken → Parameter im Inspector rechts bearbeiten',
        'Speichern / Exportieren über die Kopfleiste',
        'Für tiefe Node-Graph-Fragen den Synapse-Coach im Panel nutzen',
      ],
    }), 'synapse');
  }, [nodes.length, edges.length, selectedNodeId, activeModelName, hasUnsavedChanges, trainingStatus, language, setCurrentPageContent]);
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
  // skipDirtyRef: programmatische Änderungen (Modell/Session laden, Autosave-
  // Restore, Shape-Markierungen) sind KEINE User-Änderungen und dürfen den
  // "Ungespeichert"-Status nicht setzen.
  const skipDirtyRef = useRef(false);
  useEffect(() => {
    if (skipDirtyRef.current) { skipDirtyRef.current = false; return; }
    if (nodes.length > 0 || edges.length > 0) setHasUnsavedChanges(true);
  }, [nodes, edges]);

  // ── Autosave: Arbeitsstand überlebt App-Neustart ──────────────────────
  const autosaveKey = `synapse_autosave_${userId}`;
  useEffect(() => {
    if (nodes.length === 0 && edges.length === 0) return;
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(autosaveKey, JSON.stringify({
          nodes, edges,
          viewport: rfRef.current?.getViewport(),
          activeCanvasModelId, activeModelName,
          dirty: hasUnsavedChanges,
          savedAt: Date.now(),
        }));
      } catch { /* quota */ }
    }, 1000);
    return () => clearTimeout(timer);
  }, [nodes, edges, activeCanvasModelId, activeModelName, hasUnsavedChanges, autosaveKey]);

  // Beim Öffnen: letzten Arbeitsstand wiederherstellen (Canvas ist beim Mount leer)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(autosaveKey);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (!Array.isArray(saved?.nodes) || saved.nodes.length === 0) return;
      skipDirtyRef.current = true;
      setNodes(saved.nodes);
      setEdges(saved.edges ?? []);
      setActiveCanvasModelId(saved.activeCanvasModelId ?? null);
      setActiveModelName(saved.activeModelName ?? null);
      setHasUnsavedChanges(saved.dirty ?? true);
      setLogLines((p) => [
        ...p,
        t('synapseBuilder.autosaveRestored').replace('{count}', String(saved.nodes.length)),
      ]);
      setTimeout(() => {
        if (saved.viewport) rfRef.current?.setViewport(saved.viewport, { duration: 0 });
      }, 120);
    } catch { /* defektes Autosave ignorieren */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
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
    setShapeExtraContext(null);
    setShapeBannerVisible(false);
    skipDirtyRef.current = true; // Markierungen entfernen ist keine User-Änderung
    setNodes((nds) => clearShapeHighlights(nds, edges).nodes);
    setEdges((eds) => clearShapeHighlights(nodes, eds).edges);
  }, [nodes, edges, setNodes, setEdges]);

  // "Zoom to Nodes": Canvas auf die fehlerhaften Nodes zoomen und die erste
  // betroffene Node selektieren, damit das Property Panel sie direkt zeigt.
  const focusShapeNodesByErrors = useCallback((errs: ValidationError[]) => {
    const ids = collectAffectedNodeIds(errs);
    if (ids.size === 0 || !rfRef.current) return;
    const first = errs[0]?.targetNodeId ?? errs[0]?.sourceNodeId;
    if (first && ids.has(first)) setSelectedNodeId(first);
    rfRef.current.fitView({
      nodes: [...ids].map((id) => ({ id })),
      duration: 500,
      padding: 0.35,
      maxZoom: 1.15,
    });
  }, []);

  // Shape-Fehler melden: Nodes/Edges markieren + Banner zeigen. Mehr nicht —
  // AI-Chat und Zoom passieren erst auf expliziten Klick im Banner.
  const reportShapeIssues = useCallback(
    (
      errors: ValidationError[],
      source: ShapeIssueSource,
      opts?: { extraContext?: string }
    ) => {
      const errOnly = errors.filter((e) => e.severity === "error");
      if (errOnly.length === 0) return;

      setShapeErrors(errOnly);
      setShapeIssueSource(source);
      setShapeExtraContext(opts?.extraContext ?? null);
      setShapeBannerVisible(true);
      skipDirtyRef.current = true; // Fehler-Markierungen sind keine User-Änderung
      setNodes((nds) => applyShapeHighlightsToNodes(nds, errOnly));
      setEdges((eds) => applyShapeHighlightsToEdges(eds, errOnly));

      const msgs = errOnly.map(formatValidationError).join("\n");
      setLogLines((p) => [...p, `[Shape] ${msgs}`]);
    },
    [setNodes, setEdges]
  );

  // ── Connections (Phase 3: shape validation) ─────────────────────────────
  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target) return;
      const { valid, errors } = validateEdgeConnection(
        c.source, c.target, nodes, NODE_DEFINITIONS
      );
      if (!valid) {
        reportShapeIssues(errors, "connection");
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
            reportShapeIssues(check.errors, shapeIssueSource ?? "graph")
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
          `[Synapse] ✓ Auto-Fix: Dense inputSize ${expected} → ${actual}`,
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
            reportShapeIssues(errs, "runtime", {
              extraContext: String(diag.raw_error ?? ""),
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
        reportShapeIssues(graphCheck.errors, "graph");
        return;
      }
      clearShapeIssues();

      const datasetPath = await resolveDatasetPath(config.selectedDatasetId);

      // Graph mit Daten-Loader braucht zwingend ein ausgewähltes Dataset —
      // sonst früher klarer Abbruch statt kryptischem Engine-Fehler.
      const loaderNode = nodes.find((n) => {
        const t = (n.data as any)?._def?.type ?? (n.data as any)?.nodeType;
        return t === "image_loader" || t === "csv_loader" || t === "parquet_loader";
      });
      if (loaderNode && !datasetPath) {
        const loaderType = (loaderNode.data as any)?._def?.type ?? "data";
        setLogLines((p) => [
          ...p,
          config.selectedDatasetId
            ? `[Error] Dataset "${config.selectedDatasetName ?? config.selectedDatasetId}" — Pfad konnte nicht aufgelöst werden.`
            : `[Error] Der Graph enthält einen ${loaderType}-Node, aber es ist kein Dataset ausgewählt.`,
          `[Error] Bitte unten in der Trainingsleiste ein passendes Dataset wählen (${loaderType === "image_loader" ? "Bilder in Klassen-Ordnern" : loaderType === "csv_loader" ? "CSV-Datei" : "Parquet-Dateien"}) und erneut starten.`,
        ]);
        setTrainingStatus("error");
        return;
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
                        suggestion: /outputs BD.*expects BTC/i.test(full)
                          ? "Rang-Problem (2D → 3D): set_param hilft NICHT — reshape-Node einfügen (shape \"1, <features>\") oder Attention/Transformer entfernen"
                          : "Parameter mit set_param anpassen (inputSize, inChannels, normalizedShape, embedDim)",
                      } satisfies ValidationError,
                    ];
              reportShapeIssues(errs, "training", { extraContext: full });
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

        if (!activeCanvasModelId) {
          setLogLines((p) => [...p,
            "[Synapse] Hinweis: Graph ist nicht gespeichert — das trainierte Modell " +
            "steht danach nicht im Inferenz-Tab zur Verfuegung. Mit \u201eSpeichern\u201c " +
            "sichern und erneut trainieren."]);
        }

        const job = await invoke<{
          id: string;
          output_path?: string | null;
        }>("start_training", {
          // Ohne die Modell-ID war der Lauf mit keinem gespeicherten Canvas-Modell
          // verknuepft: die Gewichte landeten nur im Trainings-Output und der
          // Inferenz-Tab meldete dauerhaft "(kein model.pt)".
          modelId: activeCanvasModelId ?? "",
          modelName: activeModelName || "Synapse Canvas Model",
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

  // Eigentliches Speichern: Update des aktiven Modells oder Neuanlage mit Name.
  const performSave = useCallback(async (newModelName?: string) => {
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
    if (activeCanvasModelId && !newModelName) {
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
    const name = (newModelName ?? "").trim() || `Synapse ${new Date().toLocaleDateString("de-DE")}`;
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

  // Klick auf Speichern: aktives Modell → direkt updaten;
  // sonst Namensdialog öffnen (vorbefüllt, Enter = speichern).
  const handleSaveToModelLibrary = useCallback(() => {
    if (nodes.length === 0) {
      setLogLines((p) => [...p, "[Error] Kein Graph zum Speichern — füge zuerst Nodes hinzu."]);
      return;
    }
    if (activeCanvasModelId) {
      void performSave();
    } else {
      setSaveDialogName(`Synapse ${new Date().toLocaleDateString("de-DE")}`);
    }
  }, [nodes.length, activeCanvasModelId, performSave]);

  // "Clear" mit Guard: bei ungespeicherten Änderungen erst bestätigen
  const performClear = useCallback(() => {
    skipDirtyRef.current = true;
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    setActiveCanvasModelId(null);
    setActiveModelName(null);
    setHasUnsavedChanges(false);
    setConfirmClearOpen(false);
    clearShapeIssues();
    try { localStorage.removeItem(autosaveKey); } catch { /* ignore */ }
  }, [setNodes, setEdges, clearShapeIssues, autosaveKey]);

  const requestClear = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) return;
    if (hasUnsavedChanges) setConfirmClearOpen(true);
    else performClear();
  }, [nodes.length, edges.length, hasUnsavedChanges, performClear]);

  // ── Auto-Layout: Knoten übersichtlich in Ebenen anordnen ──────────────────
  // Ordnet den Canvas per Layering-Algorithmus (links→rechts, gestapelte
  // Knoten werden entzerrt), sodass Kanten sichtbar werden — kein manuelles
  // Auseinanderziehen mehr nötig.
  const handleAutoLayout = useCallback(() => {
    if (nodes.length === 0) return;
    setNodes((nds) => autoLayoutNodes(nds, edges));
    setTimeout(() => rfRef.current?.fitView({ duration: 400, padding: 0.18, maxZoom: 1.1 }), 60);
  }, [nodes.length, edges, setNodes]);

  // ── Rechtsklick-Menü: Synapse-Aktionen ────────────────────────────────────
  useContextMenuActions(() => [
    {
      id: 'synapse-save', group: 'Synapse',
      label: t('synapseBuilder.saveButton'), icon: SaveIcon,
      disabled: nodes.length === 0,
      onSelect: () => handleSaveToModelLibrary(),
    },
    {
      id: 'synapse-ai', group: 'Synapse',
      label: t('synapseBuilder.aiAssistantButton'), icon: SparklesIcon,
      onSelect: () => setShowAiPanel(true),
    },
    {
      id: 'synapse-lib', group: 'Synapse',
      label: t('synapseBuilder.sessionsModelsButton'), icon: LibraryIcon,
      onSelect: () => setShowLibrary(true),
    },
    {
      id: 'synapse-clear', group: 'Synapse',
      label: t('synapse.contextClear'), icon: TrashIcon,
      disabled: nodes.length === 0 && edges.length === 0,
      onSelect: () => requestClear(),
    },
  ]);

  const handleStopTraining = useCallback(() => {
    cancelRef.current = true;
    invoke("stop_training").catch(() => {});
    cleanupListeners();
    setTrainingStatus("idle");
    setLogLines((p) => [...p, "[Synapse] Training gestoppt."]);
  }, [cleanupListeners]);

  const handleExport = useCallback(
    (format: string) => {
      // Datei-Export (.txt/.csv/.json) läuft direkt in der TrainingConsole via
      // Blob-Download — hier nur protokollieren.
      if (format.startsWith("export_file:")) {
        setLogLines((p) => [...p, `[Export] Datei gespeichert: ${format.slice("export_file:".length)}`]);
        return;
      }
      if (format.startsWith("export_error:")) {
        setLogLines((p) => [...p, `[Export] Fehler: ${format.slice("export_error:".length)}`]);
        return;
      }
      if (format === "model_export") {
        setLogLines((p) => [...p, `[Export] Trainiertes Modell in Downloads exportiert.`]);
        return;
      }
      // "open_folder" (oder Unbekanntes): Output-Ordner im Finder/Explorer öffnen
      setLogLines((p) => [...p, `[Export] Öffne Output-Verzeichnis…`]);
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

  // ── Rollierende Chat-Komprimierung ────────────────────────────────────────
  // Ab SUMMARY_TRIGGER Nachrichten werden ältere Nachrichten (alles außer den
  // letzten KEEP_VERBATIM) in eine kompakte Fakten-Zusammenfassung komprimiert.
  // Die läuft inkrementell mit ("immer wieder"): bestehende Zusammenfassung +
  // neue alte Nachrichten → neue Zusammenfassung. Läuft im Hintergrund nach
  // jeder Antwort und blockiert nie das Senden; Fehler sind still (alte
  // Zusammenfassung bleibt dann einfach gültig).
  const SUMMARY_TRIGGER = 8;
  const KEEP_VERBATIM   = 4;

  const maybeUpdateChatSummary = useCallback(async (msgs: ChatMessage[]) => {
    if (!aiSettings.enabled) return;
    if (msgs.length < SUMMARY_TRIGGER) return;
    const upTo = msgs.length - KEEP_VERBATIM;
    if (upTo <= aiSummaryCoveredRef.current) return; // nichts Neues zu komprimieren

    const fresh = msgs
      .slice(aiSummaryCoveredRef.current, upTo)
      .map((m) => `${m.role}: ${m.content.slice(0, 400)}`)
      .join('\n');
    const prev = aiChatSummaryRef.current;

    const en = (language ?? '').toLowerCase().startsWith('en');
    const system = en
      ? `You compress the history of a chat between a user and a canvas AI assistant that builds neural networks.
Produce a compact factual summary (max ~150 words, bullet points). ALWAYS keep:
- the user's goals and decisions that were made
- concrete node IDs, node types and parameter values that were discussed or changed
- open problems / unresolved issues
No filler, facts only.`
      : `Du komprimierst den Verlauf eines Chats zwischen User und einem Canvas-AI-Assistenten, der neuronale Netze baut.
Erstelle eine kompakte Fakten-Zusammenfassung (max. ~150 Wörter, Stichpunkte). Behalte IMMER:
- Ziele des Users und getroffene Entscheidungen
- konkrete Node-IDs, Node-Typen und Parameterwerte, die besprochen oder geändert wurden
- offene Probleme / Ungeklärtes
Keine Floskeln, nur Fakten.`;

    const content = prev
      ? `${en ? 'Existing summary' : 'Bisherige Zusammenfassung'}:\n${prev}\n\n${en ? 'New messages to merge in' : 'Neue Nachrichten zum Einarbeiten'}:\n${fresh}`
      : fresh;

    try {
      const summary = await callAI(aiSettings, {
        system,
        messages: [{ role: 'user', content }],
        maxTokens: 300,
        temperature: 0.2,
        responseLanguage: language,
      });
      if (summary.trim()) {
        aiChatSummaryRef.current = summary.trim();
        aiSummaryCoveredRef.current = upTo;
      }
    } catch (e) {
      console.warn('[Synapse] Chat-Komprimierung übersprungen:', e);
    }
  }, [aiSettings, language]);

  // Panel ersetzt den Verlauf (Chat-Wechsel / neuer Chat) → Zusammenfassung
  // gehört zum alten Chat und wird verworfen.
  const handlePanelMessagesChange = useCallback((msgs: ChatMessage[]) => {
    aiChatSummaryRef.current = null;
    aiSummaryCoveredRef.current = 0;
    setAiMessages(msgs);
  }, []);

  // ── AI Send ───────────────────────────────────────────────────────────────
  // opts.displayText: kompakte Nachricht für die Chat-Anzeige, während der
  //   volle Text (z. B. Shape-Diagnose-Prompt) an den Agenten geht.
  // opts.isRetry: Nachricht steht bereits im Chat — nicht erneut anhängen.
  const aiSend = useCallback(async (
    msg?: string,
    resume?: AgentResumeState,
    opts?: { displayText?: string; isRetry?: boolean }
  ) => {
    const text = (msg ?? aiInput).trim();
    if (!text && !resume) return;
    if (aiLoading) { aiAbortRef.current?.abort(); return; }

    aiAbortRef.current = new AbortController();
    setAiLoading(true);
    setAiError(null);
    setAiSteps([]);
    setAiResumeState(null);
    if (!resume) {
      lastAiRequestRef.current = { text: text || 'Resume', displayText: opts?.displayText };
    }

    const userMsg: ChatMessage = { role: 'user', content: opts?.displayText ?? (text || 'Resume') };
    const nextMessages = opts?.isRetry ? [...aiMessages] : [...aiMessages, userMsg];
    setAiMessages(nextMessages);
    setAiInput("");

    // Aktive Shape-Fehler: vollen Diagnose-Prompt unsichtbar an den Agenten
    // anhängen — im Chat bleibt nur die kompakte User-Nachricht sichtbar.
    const shapeDiagnostic = !resume && shapeErrors.length > 0
      ? buildShapeAgentPrompt(
          shapeErrors, nodes, edges, shapeIssueSource ?? "graph", shapeExtraContext ?? undefined
        )
      : null;
    const agentText = shapeDiagnostic && text
      ? `${text}\n\n${shapeDiagnostic}`
      : (text || 'Resume');

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
        userMessage: agentText,
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
        chatSummary: aiChatSummaryRef.current ?? undefined,
        // Review-Runde: nur echte Shape-/Dimensions-Fehler zurückmelden —
        // ein bewusst unvollständiger Graph (User baut inkrementell) löst
        // keine ungewollten "Reparaturen" aus.
        getValidationReport: () => {
          const st = executor.getState();
          const check = validateFullGraph(st.nodes, st.edges, NODE_DEFINITIONS);
          const errs = check.errors.filter(
            (e) =>
              e.severity === "error" &&
              (e.type === ValidationErrorType.SHAPE_MISMATCH ||
               e.type === ValidationErrorType.DIMENSION_ERROR)
          );
          return {
            valid: errs.length === 0,
            report: errs.map(formatValidationError).join("\n"),
          };
        },
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

      // Verlauf im Hintergrund komprimieren (blockiert nichts, Fehler still)
      void maybeUpdateChatSummary([...nextMessages, { role: 'assistant', content: assistantContent }]);

      if (result.error) setAiError(result.error);
      if (result.canResume && result.resumeState) setAiResumeState(result.resumeState);

      const graphCheck = validateFullGraph(
        finalState.nodes,
        finalState.edges,
        NODE_DEFINITIONS
      );
      if (!result.error) {
        if (graphCheck.valid) {
          clearShapeIssues();
        } else if (shapeErrors.length > 0) {
          // Agent hat den Graph verändert, aber Fehler bleiben →
          // Markierungen/Chip auf den aktuellen Stand bringen
          reportShapeIssues(graphCheck.errors, shapeIssueSource ?? "graph");
        }
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
    reportShapeIssues,
    shapeErrors,
    shapeIssueSource,
    shapeExtraContext,
    maybeUpdateChatSummary,
  ]);

  // ── Fehlgeschlagene AI-Anfrage erneut senden ──────────────────────────────
  const aiRetry = useCallback(() => {
    const last = lastAiRequestRef.current;
    if (!last || aiLoading) return;
    aiSend(last.text, undefined, { displayText: last.displayText, isRetry: true });
  }, [aiSend, aiLoading]);

  // ── "Fix with AI": Panel öffnen und Fix-Vorlage in die Chat-Eingabe legen ──
  // Der User schickt sie mit Enter ab; der volle Diagnose-Kontext wird beim
  // Senden automatisch angehängt (siehe aiSend). Über einen Ref, damit der
  // verzögerte Aufruf (nach Panel-Initialisierung) den aktuellen State sieht.
  const shapeFixSenderRef = useRef<() => void>(() => {});
  useEffect(() => {
    shapeFixSenderRef.current = () => {
      if (shapeErrors.length === 0) return;
      const affected = getAffectedNodes(shapeErrors, nodes)
        .map((a) => a.label || a.id)
        .join(", ");
      setAiInput(
        t('synapseAI.panel.shapeFixRequestMessage').replace('{nodes}', affected || '—')
      );
    };
  });

  const requestShapeFixWithAI = useCallback(() => {
    setShowAiPanel(true);
    // Kurz warten, bis das Panel seine Chat-Auswahl initialisiert hat
    // (die setzt beim Öffnen u. a. die Eingabe zurück)
    window.setTimeout(() => shapeFixSenderRef.current(), 150);
  }, []);

  // ── Keyboard: Delete = Node löschen, Cmd/Ctrl+S = Speichern ────────────
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        handleSaveToModelLibrary();
        return;
      }
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
    [selectedNodeId, setNodes, setEdges, handleSaveToModelLibrary]
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
              ? `${t('synapseBuilder.saveTooltipUpdate').replace('{name}', activeModelName ?? '')} (⌘S)`
              : `${t('synapseBuilder.exportGraphTooltip')} (⌘S)`}
          >
            {activeCanvasModelId ? (
              `${hasUnsavedChanges ? "● " : "✓ "}${(activeModelName ?? "Modell").slice(0, 13)}`
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M6 1.5v5.5M3.5 5L6 7.5 8.5 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M1.5 8.5v1a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1v-1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                </svg>
                {t('synapseBuilder.saveButton')}
              </>
            )}
          </button>
          <button onClick={() => setShowLibrary(true)} style={btnStyle} title={t('synapseBuilder.sessionsModelsButton')}>
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

          {/* Auto-Layout — Knoten automatisch anordnen */}
          <button onClick={handleAutoLayout} disabled={nodes.length === 0} style={btnStyle} title={t('synapseBuilder.autoLayoutHint')}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <rect x="1"   y="1"   width="3" height="10" rx="1" stroke="currentColor" strokeWidth="1.3"/>
              <rect x="5"   y="3"   width="3" height="6"  rx="1" stroke="currentColor" strokeWidth="1.3"/>
              <rect x="9"   y="1.5" width="2" height="9"  rx="1" stroke="currentColor" strokeWidth="1.3"/>
            </svg>
            {t('synapseBuilder.autoLayout')}
          </button>

          {/* Clear — mit Bestätigung bei ungespeicherten Änderungen */}
          <button onClick={requestClear} style={btnStyle}>
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
                  onOpenAI={requestShapeFixWithAI}
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

      {showLibrary && (
        <ModelLibrary
          isOpen={showLibrary}
          onClose={() => setShowLibrary(false)}
          userId={userId}
          onLoad={(model) => {
            skipDirtyRef.current = true; // Laden ist keine User-Änderung
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
        />
      )}

      {showAICoach && trainingResult && (
        <div style={{ position: "absolute", right: showAiPanel ? 434 : 16, bottom: 216, zIndex: 600 }}>
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
        onMessagesChange={handlePanelMessagesChange}
        input={aiInput}
        onInputChange={setAiInput}
        onSend={aiSend}
        loading={aiLoading}
        error={aiError}
        steps={aiSteps}
        resumeState={aiResumeState}
        onAbort={() => aiAbortRef.current?.abort()}
        onRetry={aiRetry}
        onShapeFix={requestShapeFixWithAI}
        activeCanvasModelId={activeCanvasModelId}
        canvasHasNodes={nodes.length > 0}
        shapeMode={shapeErrors.length > 0}
        affectedNodes={getAffectedNodes(shapeErrors, nodes)}
      />

      {/* ── Namensdialog beim ersten Speichern ────────────────────────── */}
      {saveDialogName !== null && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9000,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
        }} onClick={() => setSaveDialogName(null)}>
          <div style={{
            background: "#0d1117", border: "1px solid #6366f1",
            borderRadius: 12, padding: "24px 28px", maxWidth: 380, width: "90%",
            boxShadow: "0 0 40px rgba(99,102,241,0.3)",
          }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 700, marginBottom: 6 }}>
              {t('synapseBuilder.saveDialog.title')}
            </h3>
            <p style={{ color: "#64748b", fontSize: 11, marginBottom: 14, lineHeight: 1.6 }}>
              {t('synapseBuilder.saveDialog.description')}
            </p>
            <input
              autoFocus
              value={saveDialogName}
              onChange={(e) => setSaveDialogName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const n = saveDialogName;
                  setSaveDialogName(null);
                  void performSave(n);
                }
                if (e.key === "Escape") setSaveDialogName(null);
              }}
              onFocus={(e) => e.target.select()}
              style={{
                width: "100%", boxSizing: "border-box",
                background: "#111827", border: "1px solid #334155",
                borderRadius: 8, color: "#e2e8f0", fontSize: 13,
                padding: "9px 12px", outline: "none", fontFamily: "inherit",
                marginBottom: 14,
              }}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => setSaveDialogName(null)}
                style={{
                  flex: 1, padding: "8px 0", borderRadius: 6,
                  background: "transparent", border: "1px solid #334155",
                  color: "#94a3b8", fontSize: 12, cursor: "pointer", fontFamily: "inherit",
                }}
              >
                {t('synapseBuilder.saveDialog.cancel')}
              </button>
              <button
                onClick={() => {
                  const n = saveDialogName;
                  setSaveDialogName(null);
                  void performSave(n);
                }}
                style={{
                  flex: 1, padding: "8px 0", borderRadius: 6,
                  background: "linear-gradient(135deg, #6366f1, #a78bfa)", border: "none",
                  color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                }}
              >
                {t('synapseBuilder.saveDialog.save')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Clear-Bestätigung bei ungespeicherten Änderungen ──────────── */}
      {confirmClearOpen && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9000,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
        }} onClick={() => setConfirmClearOpen(false)}>
          <div style={{
            background: "#0d1117", border: "1px solid #7f1d1d",
            borderRadius: 12, padding: "24px 28px", maxWidth: 380, width: "90%",
            boxShadow: "0 0 40px rgba(239,68,68,0.2)",
          }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ color: "#fecaca", fontSize: 14, fontWeight: 700, marginBottom: 6 }}>
              ⚠ {t('synapseBuilder.confirmClear.title')}
            </h3>
            <p style={{ color: "#94a3b8", fontSize: 12, marginBottom: 16, lineHeight: 1.6 }}>
              {t('synapseBuilder.confirmClear.body')}
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => setConfirmClearOpen(false)}
                style={{
                  flex: 1, padding: "8px 0", borderRadius: 6,
                  background: "transparent", border: "1px solid #334155",
                  color: "#94a3b8", fontSize: 12, cursor: "pointer", fontFamily: "inherit",
                }}
              >
                {t('synapseBuilder.confirmClear.cancel')}
              </button>
              <button
                onClick={performClear}
                style={{
                  flex: 1, padding: "8px 0", borderRadius: 6,
                  background: "rgba(239,68,68,0.18)", border: "1px solid #7f1d1d",
                  color: "#fca5a5", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                }}
              >
                {t('synapseBuilder.confirmClear.discard')}
              </button>
            </div>
          </div>
        </div>
      )}

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
            <div style={{ fontSize: 26, marginBottom: 8, textAlign: "center", color: "#34d399" }}>✓</div>
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
