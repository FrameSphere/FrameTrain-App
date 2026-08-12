import React, { useState, useRef, useEffect, useCallback } from "react";
import { CanvasInferenceTab } from "./CanvasInferenceTab";
import { invoke } from "@tauri-apps/api/core";
import { useLanguage } from "../../contexts/LanguageContext";

// ─── Types ────────────────────────────────────────────────────────────────────
export type TrainingStatus = "idle" | "running" | "paused" | "done" | "error";

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  gpu: string;
  precision: "fp32" | "fp16" | "bf16";
  gradAccum: number;
  // PHASE 4: Canvas integration
  canvasModelCode?: string;  // Generated nn.Module code from canvas graph
  canvasGraphMetadata?: any; // Graph topology info for debugging
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  valLoss?: number;
  accuracy?: number;
  lr: number;
  gpuMemMB?: number;
}

// Dataset selection for training (model comes from canvas graph)
export interface DatasetOption { id: string; name: string; }

interface TrainingConsoleProps {
  onStartTraining: (config: TrainingConfig) => void;
  onStopTraining: () => void;
  onExport: (format: string) => void;
  status: TrainingStatus;
  metrics: TrainingMetrics[];
  logLines: string[];
  // Real training
  completedVersionId?: string | null;
  userId: string;
  /** Output directory from SynapseBuilder (model.pt, metrics.json, canvas_model.py) */
  outputDir?: string;
}

// ─── Mini Sparkline ───────────────────────────────────────────────────────────
const Sparkline: React.FC<{ data: number[]; color: string; label: string }> = ({
  data, color, label,
}) => {
  if (data.length < 2) return null;
  const w = 120, h = 36;
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");
  const last = data[data.length - 1];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <div style={{ fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: "0.07em" }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
        <svg width={w} height={h} style={{ overflow: "visible" }}>
          <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" opacity={0.8} />
          <polygon points={`0,${h} ${pts} ${w},${h}`} fill={color} opacity={0.07} />
          <circle cx={w} cy={h - ((last - min) / range) * h} r={2.5} fill={color} />
        </svg>
        <span style={{ fontSize: 13, fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace", whiteSpace: "nowrap" }}>
          {last < 1 ? last.toFixed(4) : last.toFixed(2)}
        </span>
      </div>
    </div>
  );
};

// ─── Config Field ─────────────────────────────────────────────────────────────
const ConfigField: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
    <label style={{ fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: "0.07em" }}>
      {label}
    </label>
    {children}
  </div>
);

const inputStyle: React.CSSProperties = {
  background: "#111827", border: "1px solid #1e293b", borderRadius: 5,
  color: "#e2e8f0", fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
  padding: "5px 8px", outline: "none", width: "100%", boxSizing: "border-box",
};

// ─── Status badge ─────────────────────────────────────────────────────────────
const StatusBadge: React.FC<{ status: TrainingStatus }> = ({ status }) => {
  const map: Record<TrainingStatus, { label: string; color: string; pulse?: boolean }> = {
    idle:    { label: "IDLE",     color: "#475569" },
    running: { label: "TRAINING", color: "#34d399", pulse: true },
    paused:  { label: "PAUSED",   color: "#facc15" },
    done:    { label: "DONE",     color: "#38bdf8" },
    error:   { label: "ERROR",    color: "#f87171" },
  };
  const s = map[status];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em" }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: s.color, boxShadow: s.pulse ? `0 0 8px ${s.color}` : "none", animation: s.pulse ? "pulse 1.2s infinite" : "none" }} />
      {s.label}
      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
};

// ─── Model/Dataset Selector Modal ─────────────────────────────────────────────
const SelectorModal: React.FC<{
  title: string;
  items: { id: string; name: string }[];
  onSelect: (item: { id: string; name: string }) => void;
  onClose: () => void;
  loading: boolean;
}> = ({ title, items, onSelect, onClose, loading }) => {
  const { t } = useLanguage();
  return (
  <>
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 2000 }} />
    <div style={{
      position: "fixed", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
      width: 400, maxHeight: 400, background: "#0d1117", border: "1px solid #1e293b",
      borderRadius: 10, display: "flex", flexDirection: "column", overflow: "hidden",
      zIndex: 2001, boxShadow: "0 20px 60px rgba(0,0,0,0.8)",
    }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", fontFamily: "'JetBrains Mono',monospace" }}>{title}</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: 14 }}>✕</button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: 8 }}>
        {loading ? (
          <div style={{ padding: "20px", textAlign: "center", color: "#334155", fontSize: 12, fontFamily: "'JetBrains Mono',monospace" }}>{t('synapse.trainingConsole.datasetSelector.loading')}</div>
        ) : items.length === 0 ? (
          <div style={{ padding: "20px", textAlign: "center", color: "#334155", fontSize: 12, fontFamily: "'JetBrains Mono',monospace" }}>
            {t('synapse.trainingConsole.datasetSelector.empty')}
          </div>
        ) : items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item)}
            style={{
              display: "block", width: "100%", padding: "10px 12px", textAlign: "left",
              background: "transparent", border: "1px solid #1e293b", borderRadius: 6,
              color: "#cbd5e1", fontSize: 12, cursor: "pointer", marginBottom: 4,
              fontFamily: "'JetBrains Mono',monospace",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#4f46e5")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "#1e293b")}
          >
            {item.name}
          </button>
        ))}
      </div>
    </div>
  </>
  );
};

// ─── Export Modal ─────────────────────────────────────────────────────────────
const ExportModal: React.FC<{
  completedVersionId: string | null;
  onExportFormat: (fmt: string) => void;
  onExportVersion: () => void;
  onClose: () => void;
  exporting: boolean;
  exportPath: string | null;
}> = ({ completedVersionId, onExportFormat, onExportVersion, onClose, exporting, exportPath }) => {
  const { t } = useLanguage();
  return (
  <>
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 2000 }} />
    <div style={{
      position: "fixed", top: "50%", left: "50%", transform: "translate(-50%,-50%)",
      width: 360, background: "#0d1117", border: "1px solid #1e293b",
      borderRadius: 10, display: "flex", flexDirection: "column", overflow: "hidden",
      zIndex: 2001, boxShadow: "0 20px 60px rgba(0,0,0,0.8)",
    }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", fontFamily: "'JetBrains Mono',monospace" }}>{t('synapse.trainingConsole.exportModal.title')}</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: 14 }}>✕</button>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
        {/* Export trainiertes Modell in Downloads */}
        {completedVersionId && (
          <div style={{ padding: "10px 12px", background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.2)", borderRadius: 8 }}>
            <div style={{ fontSize: 11, color: "#34d399", fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", marginBottom: 6 }}>
              {t('synapse.trainingConsole.exportModal.trainedModelTitle')}
            </div>
            <div style={{ fontSize: 10, color: "#475569", fontFamily: "'JetBrains Mono',monospace", marginBottom: 8 }}>
              {t('synapse.trainingConsole.exportModal.trainedModelDesc')}
            </div>
            {exportPath ? (
              <div style={{ fontSize: 10, color: "#34d399", fontFamily: "'JetBrains Mono',monospace", wordBreak: "break-all" }}>
                {t('synapse.trainingConsole.exportModal.exportedLabel', { path: exportPath ?? '' })}
              </div>
            ) : (
              <button
                onClick={onExportVersion}
                disabled={exporting}
                style={{
                  padding: "7px 14px", background: exporting ? "rgba(52,211,153,0.05)" : "rgba(52,211,153,0.15)",
                  border: "1px solid rgba(52,211,153,0.4)", borderRadius: 6, color: "#34d399",
                  fontSize: 11, fontWeight: 600, cursor: exporting ? "not-allowed" : "pointer",
                  fontFamily: "'JetBrains Mono',monospace", width: "100%",
                }}
              >
                {exporting ? t('synapse.trainingConsole.exportModal.exportingButton') : t('synapse.trainingConsole.exportModal.exportButton')}
              </button>
            )}
          </div>
        )}

        {/* Log-Export-Formate */}
        <div style={{ fontSize: 10, color: "#334155", fontFamily: "'JetBrains Mono',monospace", paddingTop: completedVersionId ? 4 : 0 }}>
          {t('synapse.trainingConsole.exportModal.logMetricsLabel')}
        </div>
        {([
          { id: "txt",  labelKey: "synapse.trainingConsole.exportModal.formatTxt" },
          { id: "csv",  labelKey: "synapse.trainingConsole.exportModal.formatCsv" },
          { id: "json", labelKey: "synapse.trainingConsole.exportModal.formatJson" },
        ] as const).map(({ id, labelKey }) => (
          <button
            key={id}
            onClick={() => onExportFormat(id)}
            style={{
              padding: "8px 12px", background: "transparent", border: "1px solid #1e293b",
              borderRadius: 6, color: "#94a3b8", fontSize: 12, cursor: "pointer", textAlign: "left",
              fontFamily: "'JetBrains Mono',monospace",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#1e293b")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            {t(labelKey)}
          </button>
        ))}
      </div>
    </div>
  </>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────
export const TrainingConsole: React.FC<TrainingConsoleProps> = ({
  onStartTraining,
  onStopTraining,
  onExport,
  status,
  metrics,
  logLines,
  completedVersionId,
  userId,
  outputDir,
}) => {
  const { t } = useLanguage();
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    gpu: "cuda:0",
    precision: "fp32",
    gradAccum: 1,
  });

  const [activeTab, setActiveTab] = useState<"config" | "metrics" | "log" | "inference">("config");
  const [showExportModal, setShowExportModal] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportPath, setExportPath] = useState<string | null>(null);

  // Dataset selection (architecture = current canvas graph)
  const [showDatasetSelector, setShowDatasetSelector] = useState(false);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption | null>(null);

  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log
  useEffect(() => {
    if (activeTab === "log" && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines, activeTab]);

  // Auto-switch to log tab when training starts
  useEffect(() => {
    if (status === "running") setActiveTab("log");
  }, [status]);

  // Auto-switch to metrics tab when done
  useEffect(() => {
    if (status === "done") setActiveTab("metrics");
  }, [status]);

  // Reset export path when new training starts
  useEffect(() => {
    if (status === "running") setExportPath(null);
  }, [status]);

  const losses    = React.useMemo(() => metrics.map((m) => m.loss), [metrics]);
  const valLosses = React.useMemo(() => metrics.filter((m) => m.valLoss != null).map((m) => m.valLoss!), [metrics]);
  const accs      = React.useMemo(() => metrics.filter((m) => m.accuracy != null).map((m) => m.accuracy!), [metrics]);

  const isRunning = status === "running";
  const canStart  = status === "idle" || status === "done" || status === "error";

  const loadDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    try {
      const result = await invoke<{ id: string; name: string }[]>("list_datasets");
      setDatasets(result ?? []);
    } catch (e) {
      console.error("[TrainingConsole] list_datasets:", e);
      setDatasets([]);
    } finally {
      setLoadingDatasets(false);
    }
  }, []);

  // ── Export trainiertes Modell ─────────────────────────────────────────────
  const handleExportVersion = useCallback(async () => {
    if (!completedVersionId) return;
    setExporting(true);
    try {
      const path = await invoke<string>("export_model_version", { versionId: completedVersionId });
      setExportPath(path);
      onExport("model_export");
    } catch (e: any) {
      console.error("[TrainingConsole] export_model_version:", e);
      onExport(`export_error: ${e}`);
    } finally {
      setExporting(false);
    }
  }, [completedVersionId, onExport]);

  // ── Log / Metriken / Config als echte Datei exportieren ───────────────────
  // Schreibt tatsächlich .txt/.csv/.json statt nur den Ordner zu öffnen.
  const handleExportFormat = useCallback((fmt: string) => {
    const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    let filename = "", content = "", mime = "text/plain;charset=utf-8";

    if (fmt === "csv") {
      filename = `metrics_${ts}.csv`;
      mime = "text/csv;charset=utf-8";
      const header = "epoch,loss,val_loss,accuracy,lr,gpu_mem_mb";
      const rows = metrics.map((m) =>
        [m.epoch, m.loss, m.valLoss ?? "", m.accuracy ?? "", m.lr, m.gpuMemMB ?? ""].join(",")
      );
      content = [header, ...rows].join("\n");
    } else if (fmt === "json") {
      filename = `config_${ts}.json`;
      mime = "application/json";
      content = JSON.stringify(
        { config, epochsTrained: metrics.length, finalMetrics: metrics[metrics.length - 1] ?? null },
        null, 2
      );
    } else {
      filename = `training_log_${ts}.txt`;
      content = logLines.join("\n") || "(kein Log vorhanden)";
    }

    try {
      const blob = new Blob([content], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
      onExport(`export_file:${filename}`);
    } catch (e) {
      onExport(`export_error:${e}`);
    }
    setShowExportModal(false);
  }, [metrics, config, logLines, onExport]);

  // ── Train-Click ───────────────────────────────────────────────────────────
  const handleTrainClick = useCallback(() => {
    onStartTraining({
      ...config,
      selectedDatasetId: selectedDataset?.id,
      selectedDatasetName: selectedDataset?.name,
    } as TrainingConfig & { selectedDatasetId?: string; selectedDatasetName?: string });
  }, [config, selectedDataset, onStartTraining]);

  return (
    <div style={{ height: 200, background: "#0a0e17", borderTop: "1px solid #1e293b", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Top bar */}
      <div style={{ display: "flex", alignItems: "center", padding: "0 14px", gap: 12, height: 38, borderBottom: "1px solid #1e293b", flexShrink: 0 }}>
        <StatusBadge status={status} />

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2, marginLeft: 8 }}>
          {(["config", "metrics", "log", "inference"] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: "4px 10px", background: activeTab === tab ? "#1e293b" : "transparent",
              border: "none", borderRadius: 4, color: activeTab === tab ? "#e2e8f0" : "#475569",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
              letterSpacing: "0.04em",
            }}>
              {t(`synapse.trainingConsole.tabs.${tab}`)}
              {tab === "log" && logLines.length > 0 && (
                <span style={{ marginLeft: 4, fontSize: 9, color: "#334155" }}>{logLines.length}</span>
              )}
            </button>
          ))}
        </div>

        {/* Epoch progress */}
        {metrics.length > 0 && (
          <div style={{ marginLeft: "auto", fontSize: 11, color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
            {t('synapse.trainingConsole.epochLabel', { current: metrics[metrics.length - 1].epoch, total: config.epochs })}
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 6, marginLeft: metrics.length > 0 ? 0 : "auto" }}>
          {/* Export */}
          <button
            onClick={() => setShowExportModal(true)}
            style={{
              padding: "5px 12px", background: "transparent",
              border: `1px solid ${completedVersionId ? "rgba(52,211,153,0.4)" : "#1e293b"}`,
              borderRadius: 5, color: completedVersionId ? "#34d399" : "#94a3b8",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
            }}
          >
            {t('synapse.trainingConsole.exportButton')}
          </button>

          {/* Start / Stop */}
          {isRunning ? (
            <button onClick={onStopTraining} style={{
              padding: "5px 14px", background: "#f8717120", border: "1px solid #f87171",
              borderRadius: 5, color: "#f87171", fontSize: 11, fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
            }}>
              {t('synapse.trainingConsole.stopButton')}
            </button>
          ) : (
            <button disabled={!canStart} onClick={handleTrainClick} style={{
              padding: "5px 14px", background: canStart ? "#34d39920" : "transparent",
              border: `1px solid ${canStart ? "#34d399" : "#1e293b"}`,
              borderRadius: 5, color: canStart ? "#34d399" : "#334155",
              fontSize: 11, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
              cursor: canStart ? "pointer" : "not-allowed",
            }}>
              {t('synapse.trainingConsole.trainButton')}
            </button>
          )}
        </div>
      </div>

      {/* ── Done-Banner: model.pt ready ───────────────────────────────────────── */}
      {status === "done" && outputDir && (
        <div style={{
          position: "absolute", top: 38, left: 0, right: 0,
          background: "linear-gradient(90deg, rgba(52,211,153,0.12), rgba(52,211,153,0.06))",
          borderBottom: "1px solid rgba(52,211,153,0.25)",
          padding: "7px 14px",
          display: "flex", alignItems: "center", gap: 12,
          zIndex: 10, backdropFilter: "blur(4px)",
        }}>
          <span style={{ fontSize: 12, color: "#34d399", fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>
            {t('synapse.trainingConsole.doneBannerTitle')}
          </span>
          <span style={{ fontSize: 11, color: "#475569", fontFamily: "'JetBrains Mono',monospace", flex: 1 }}>
            model.pt · metrics.json · canvas_model.py
          </span>
          <button
            onClick={() => onExport("open_folder")}
            style={{
              padding: "4px 12px",
              background: "rgba(52,211,153,0.15)",
              border: "1px solid rgba(52,211,153,0.4)",
              borderRadius: 5, color: "#34d399",
              fontSize: 11, fontWeight: 600,
              fontFamily: "'JetBrains Mono',monospace",
              cursor: "pointer",
            }}
          >
            {t('synapse.trainingConsole.openFolderButton')}
          </button>
        </div>
      )}

      {/* Panel body */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex" }}>

        {/* CONFIG TAB */}
        {activeTab === "config" && (
          <div style={{ display: "flex", gap: 16, padding: "10px 14px", overflowX: "auto", alignItems: "flex-start", width: "100%" }}>

            <ConfigField label={t('synapse.trainingConsole.architectureLabel')}>
              <div style={{ ...inputStyle, width: 120, color: "#64748b", fontSize: 10 }}>
                Canvas Graph
              </div>
            </ConfigField>

            <ConfigField label={t('synapse.trainingConsole.datasetLabel')}>
              <button
                onClick={() => { loadDatasets(); setShowDatasetSelector(true); }}
                style={{
                  ...inputStyle, cursor: "pointer", textAlign: "left", width: 120,
                  color: selectedDataset ? "#e2e8f0" : "#334155",
                  borderColor: selectedDataset ? "#4f46e5" : "#1e293b",
                }}
              >
                {selectedDataset ? selectedDataset.name.slice(0, 14) + (selectedDataset.name.length > 14 ? "…" : "") : t('synapse.trainingConsole.datasetPlaceholder')}
              </button>
            </ConfigField>

            <div style={{ width: 1, background: "#1e293b", alignSelf: "stretch", flexShrink: 0 }} />

            <ConfigField label={t('synapse.trainingConsole.epochsLabel')}>
              <input type="number" value={config.epochs} min={1} onChange={(e) => setConfig((c) => ({ ...c, epochs: Number(e.target.value) }))} style={{ ...inputStyle, width: 70 }} />
            </ConfigField>
            <ConfigField label={t('synapse.trainingConsole.batchSizeLabel')}>
              <input type="number" value={config.batchSize} min={1} onChange={(e) => setConfig((c) => ({ ...c, batchSize: Number(e.target.value) }))} style={{ ...inputStyle, width: 70 }} />
            </ConfigField>
            <ConfigField label={t('synapse.trainingConsole.learningRateLabel')}>
              <input type="number" value={config.learningRate} step={0.0001} min={0} onChange={(e) => setConfig((c) => ({ ...c, learningRate: Number(e.target.value) }))} style={{ ...inputStyle, width: 90 }} />
            </ConfigField>
            <ConfigField label={t('synapse.trainingConsole.gpuLabel')}>
              <select
                value={config.gpu}
                onChange={(e) => {
                  const gpu = e.target.value;
                  // fp16/bf16 sind CUDA-only → auf Nicht-CUDA automatisch fp32 erzwingen
                  setConfig((c) => ({
                    ...c,
                    gpu,
                    precision: gpu.startsWith("cuda") ? c.precision : "fp32",
                  }));
                }}
                style={{ ...inputStyle, width: 100 }}
              >
                {["cpu", "cuda:0", "cuda:1", "mps"].map((g) => (
                  <option key={g} value={g} style={{ background: "#111827" }}>{g}</option>
                ))}
              </select>
            </ConfigField>
            <ConfigField label={t('synapse.trainingConsole.precisionLabel')}>
              <select
                value={config.precision}
                onChange={(e) => setConfig((c) => ({ ...c, precision: e.target.value as TrainingConfig["precision"] }))}
                style={{ ...inputStyle, width: 80 }}
                title={config.gpu.startsWith("cuda") ? undefined : t('synapse.trainingConsole.precisionCudaHint')}
              >
                {["fp32", "fp16", "bf16"].map((p) => (
                  <option
                    key={p}
                    value={p}
                    disabled={p !== "fp32" && !config.gpu.startsWith("cuda")}
                    style={{ background: "#111827" }}
                  >
                    {p}{p !== "fp32" && !config.gpu.startsWith("cuda") ? " (CUDA)" : ""}
                  </option>
                ))}
              </select>
            </ConfigField>
            <ConfigField label={t('synapse.trainingConsole.gradAccumLabel')}>
              <input type="number" value={config.gradAccum} min={1} onChange={(e) => setConfig((c) => ({ ...c, gradAccum: Number(e.target.value) }))} style={{ ...inputStyle, width: 60 }} />
            </ConfigField>
          </div>
        )}

        {/* METRICS TAB */}
        {activeTab === "metrics" && (
          <div style={{ display: "flex", gap: 32, padding: "12px 18px", alignItems: "flex-start", overflowX: "auto", width: "100%" }}>
            {losses.length > 1 ? (
              <>
                <Sparkline data={losses} color="#f87171" label={t('synapse.trainingConsole.metricLoss')} />
                {valLosses.length > 1 && <Sparkline data={valLosses} color="#fb923c" label={t('synapse.trainingConsole.metricValLoss')} />}
                {accs.length > 1 && <Sparkline data={accs} color="#34d399" label={t('synapse.trainingConsole.metricAccuracy')} />}
                <div style={{ marginLeft: "auto", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 20px", alignSelf: "center" }}>
                  {[
                    { label: t('synapse.trainingConsole.metricLoss'), val: losses[losses.length - 1]?.toFixed(4) },
                    { label: t('synapse.trainingConsole.metricValLoss'), val: valLosses[valLosses.length - 1]?.toFixed(4) ?? "—" },
                    { label: t('synapse.trainingConsole.metricAccuracy'), val: accs.length ? (accs[accs.length - 1] * 100).toFixed(1) + "%" : "—" },
                    { label: t('synapse.trainingConsole.metricLr'), val: metrics[metrics.length - 1]?.lr?.toExponential(2) ?? "—" },
                  ].map(({ label, val }) => (
                    <div key={label}>
                      <div style={{ fontSize: 9, color: "#475569", fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", letterSpacing: "0.07em" }}>{label}</div>
                      <div style={{ fontSize: 13, color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{val}</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ fontSize: 12, color: "#334155", fontFamily: "'JetBrains Mono', monospace", alignSelf: "center" }}>
                {t('synapse.trainingConsole.noMetrics')}
              </div>
            )}
          </div>
        )}

        

          {/* INFERENCE TAB */}
          {activeTab === "inference" && (
            <CanvasInferenceTab userId={userId ?? ""} />
          )}

          {/* LOG TAB */}
          {activeTab === "log" && (
          <div ref={logRef} style={{
            flex: 1, overflowY: "auto", padding: "8px 14px",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b",
            lineHeight: 1.7, scrollbarWidth: "thin", scrollbarColor: "#1e293b transparent",
          }}>
            {logLines.length === 0 ? (
              <span style={{ color: "#334155" }}>{t('synapse.trainingConsole.logWaiting')}</span>
            ) : (
              logLines.map((line, i) => {
                const isError   = line.toLowerCase().includes("error");
                const isWarning = line.toLowerCase().includes("warn");
                const isSuccess = line.toLowerCase().includes("complete") || line.toLowerCase().includes("✓");
                return (
                  <div key={i} style={{ color: isError ? "#f87171" : isWarning ? "#facc15" : isSuccess ? "#34d399" : "#64748b" }}>
                    <span style={{ color: "#1e293b", marginRight: 8 }}>{String(i + 1).padStart(3, "0")}</span>
                    {line}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>

      {/* ── Export Modal ──────────────────────────────────────────────────────── */}
      {showExportModal && (
        <ExportModal
          completedVersionId={completedVersionId ?? null}
          onExportFormat={handleExportFormat}
          onExportVersion={handleExportVersion}
          onClose={() => setShowExportModal(false)}
          exporting={exporting}
          exportPath={exportPath}
        />
      )}

      {/* ── Dataset Selector ────────────────────────────────────────────────── */}
      {showDatasetSelector && (
        <SelectorModal
          title={t('synapse.trainingConsole.datasetSelector.title')}
          items={datasets}
          onSelect={(d) => { setSelectedDataset(d); setShowDatasetSelector(false); }}
          onClose={() => setShowDatasetSelector(false)}
          loading={loadingDatasets}
        />
      )}
    </div>
  );
};

export default TrainingConsole;
