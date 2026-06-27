/**
 * CanvasInferenceTab.tsx
 * ========================
 * Tab zum Testen trainierter Canvas/Synapse-Modelle.
 * Lädt gespeicherte Modelle via list_canvas_models_with_pt,
 * nimmt Tensor-Input als kommagetrennte Zahlen entgegen,
 * schickt via run_canvas_inference und zeigt Predictions.
 *
 * Benötigt: model.pt + graph_metadata.json im Modell-Ordner.
 * Keine Canvas-Session oder localStorage nötig.
 */

import React, { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useLanguage } from "../../contexts/LanguageContext";

// ─── Types ────────────────────────────────────────────────────────────────────

interface CanvasModelInfo {
  model_id: string;
  name: string;
  has_weights: boolean;
  model_pt_path: string;
  metadata_path: string;
  task_type: string;
  num_classes: number;
}

interface CanvasInferenceResult {
  predicted_class: number | null;
  confidence: number | null;
  predicted_value: unknown | null;
  top_predictions: Array<{ class_idx: number; score: number }> | null;
  all_probs: number[] | null;
  inference_ms: number;
  task_type: string;
  error: string | null;
}

interface Props {
  userId: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function parseInputString(raw: string): number[] | null {
  try {
    const nums = raw
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number);
    if (nums.some(isNaN)) return null;
    return nums;
  } catch {
    return null;
  }
}

function pct(v: number) {
  return (v * 100).toFixed(1) + "%";
}

// ─── Component ────────────────────────────────────────────────────────────────

export const CanvasInferenceTab: React.FC<Props> = ({ userId }) => {
  const { t } = useLanguage();
  const [models, setModels] = useState<CanvasModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [inputRaw, setInputRaw] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<CanvasInferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── Modelle laden ─────────────────────────────────────────────────────────
  const fetchModels = useCallback(async () => {
    setLoadingModels(true);
    try {
      const list = await invoke<CanvasModelInfo[]>("list_canvas_models_with_pt", { userId });
      setModels(list);
      const withPt = list.find((m) => m.has_weights);
      if (withPt) setSelectedModelId((prev) => prev ?? withPt.model_id);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoadingModels(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const selected = models.find((m) => m.model_id === selectedModelId);

  // ── Inference ausführen ───────────────────────────────────────────────────
  const runInference = useCallback(async () => {
    if (!selectedModelId) return;
    const nums = parseInputString(inputRaw);
    if (!nums || nums.length === 0) {
      setError(t('canvasInferenceTab.invalidInput'));
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await invoke<CanvasInferenceResult>("run_canvas_inference", {
        modelId: selectedModelId,
        input: nums,
      });
      setResult(res);
      if (res.error) setError(res.error);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }, [selectedModelId, inputRaw]);

  // ─── Styles ───────────────────────────────────────────────────────────────
  const S = {
    root: {
      display: "flex",
      flexDirection: "column" as const,
      gap: 16,
      padding: "16px 20px",
      height: "100%",
      overflowY: "auto" as const,
      fontFamily: "'JetBrains Mono', monospace",
      color: "#94a3b8",
    } as React.CSSProperties,
    section: {
      background: "rgba(15,23,36,0.7)",
      border: "1px solid #1e293b",
      borderRadius: 8,
      padding: "14px 16px",
    } as React.CSSProperties,
    label: {
      fontSize: 10,
      color: "#475569",
      textTransform: "uppercase" as const,
      letterSpacing: "0.08em",
      marginBottom: 6,
    } as React.CSSProperties,
    select: {
      width: "100%",
      background: "#0d1117",
      border: "1px solid #1e293b",
      borderRadius: 5,
      color: "#e2e8f0",
      fontSize: 11,
      padding: "6px 8px",
      cursor: "pointer",
      outline: "none",
    } as React.CSSProperties,
    textarea: {
      width: "100%",
      background: "#0d1117",
      border: "1px solid #1e293b",
      borderRadius: 5,
      color: "#e2e8f0",
      fontSize: 11,
      padding: "8px 10px",
      resize: "vertical" as const,
      minHeight: 60,
      outline: "none",
      fontFamily: "inherit",
      boxSizing: "border-box" as const,
    } as React.CSSProperties,
    btn: {
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      padding: "7px 16px",
      background: "rgba(99,102,241,0.12)",
      border: "1px solid #6366f1",
      borderRadius: 6,
      color: "#818cf8",
      fontSize: 11,
      cursor: "pointer",
      fontFamily: "inherit",
    } as React.CSSProperties,
    btnDisabled: {
      opacity: 0.4,
      cursor: "not-allowed",
      pointerEvents: "none" as const,
    } as React.CSSProperties,
    tag: (color: string) =>
      ({
        display: "inline-block",
        padding: "1px 7px",
        borderRadius: 3,
        fontSize: 9,
        background: `${color}20`,
        border: `1px solid ${color}60`,
        color,
        letterSpacing: "0.06em",
        textTransform: "uppercase" as const,
      } as React.CSSProperties),
    barFill: (score: number) =>
      ({
        height: "100%",
        width: `${Math.min(score * 100, 100)}%`,
        background: score > 0.6 ? "#22c55e" : score > 0.3 ? "#f59e0b" : "#6366f1",
        borderRadius: 2,
        transition: "width 0.3s ease",
      } as React.CSSProperties),
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={S.root}>

      {/* ── Modell-Auswahl ─────────────────────────────────────────────────── */}
      <div style={S.section}>
        <div style={S.label}>Modell auswählen</div>
        {loadingModels ? (
          <div style={{ fontSize: 11, color: "#475569" }}>Lade Modelle…</div>
        ) : models.length === 0 ? (
          <div style={{ fontSize: 11, color: "#475569" }}>
            Keine Canvas-Modelle gefunden. Erst im Builder ein Modell exportieren und trainieren.
          </div>
        ) : (
          <select
            style={S.select}
            value={selectedModelId ?? ""}
            onChange={(e) => {
              setSelectedModelId(e.target.value || null);
              setResult(null);
              setError(null);
            }}
          >
            <option value="">{t('canvasInferenceTab.modelSelectPlaceholder')}</option>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id} disabled={!m.has_weights}>
                {m.name}
                {!m.has_weights ? " (kein model.pt)" : ""}
              </option>
            ))}
          </select>
        )}

        {/* Modell-Infos */}
        {selected && (
          <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" as const }}>
            <span style={S.tag("#6366f1")}>{selected.task_type}</span>
            <span style={S.tag("#0ea5e9")}>{selected.num_classes} Klassen</span>
            {selected.has_weights ? (
              <span style={S.tag("#22c55e")}>model.pt ✓</span>
            ) : (
              <span style={S.tag("#ef4444")}>Kein model.pt</span>
            )}
          </div>
        )}
      </div>

      {/* ── Input ──────────────────────────────────────────────────────────── */}
      <div style={S.section}>
        <div style={S.label}>Tensor-Input</div>
        <div style={{ fontSize: 10, color: "#334155", marginBottom: 6 }}>
          Kommagetrennte Zahlen, z.B.:{" "}
          <code style={{ color: "#475569" }}>0.1, 0.5, 0.3, 0.8</code>
        </div>
        <textarea
          style={S.textarea}
          placeholder={t('canvasInferenceTab.inputPlaceholder')}
          value={inputRaw}
          onChange={(e) => setInputRaw(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runInference();
          }}
        />
        <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
          <button
            style={{
              ...S.btn,
              ...(!selected || !selected.has_weights || running ? S.btnDisabled : {}),
            }}
            onClick={runInference}
            disabled={!selected || !selected.has_weights || running}
          >
            {running ? (
              <>
                <span
                  style={{ animation: "spin 0.8s linear infinite", display: "inline-block" }}
                >
                  ⟳
                </span>
                Inferenz läuft…
              </>
            ) : (
              <>▶ Inferenz starten</>
            )}
          </button>
          <span style={{ fontSize: 10, color: "#334155" }}>⌘↵ oder Ctrl+↵</span>
        </div>
      </div>

      {/* ── Fehler ─────────────────────────────────────────────────────────── */}
      {error && (
        <div
          style={{
            ...S.section,
            border: "1px solid #ef444460",
            background: "rgba(239,68,68,0.06)",
          }}
        >
          <div style={{ fontSize: 10, color: "#ef4444", fontWeight: 700, marginBottom: 4 }}>
            Fehler
          </div>
          <div style={{ fontSize: 11, color: "#fca5a5", whiteSpace: "pre-wrap" as const }}>
            {error}
          </div>
        </div>
      )}

      {/* ── Ergebnis ───────────────────────────────────────────────────────── */}
      {result && !result.error && (
        <div style={S.section}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
            }}
          >
            <div style={S.label}>Ergebnis</div>
            <span style={{ fontSize: 10, color: "#334155" }}>
              {result.inference_ms.toFixed(1)} ms
            </span>
          </div>

          {result.task_type === "regression" ? (
            // ── Regression ──
            <div style={{ fontSize: 18, color: "#e2e8f0", fontWeight: 700 }}>
              {typeof result.predicted_value === "number"
                ? result.predicted_value.toFixed(4)
                : JSON.stringify(result.predicted_value)}
            </div>
          ) : (
            // ── Klassifikation ──
            <>
              {/* Haupt-Prediction */}
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 10,
                  marginBottom: 12,
                  paddingBottom: 12,
                  borderBottom: "1px solid #1e293b",
                }}
              >
                <span style={{ fontSize: 22, color: "#e2e8f0", fontWeight: 700 }}>
                  Klasse {result.predicted_class ?? "?"}
                </span>
                <span style={{ fontSize: 13, color: "#22c55e", fontWeight: 600 }}>
                  {result.confidence != null ? pct(result.confidence) : ""}
                </span>
              </div>

              {/* Top-Predictions */}
              {result.top_predictions && result.top_predictions.length > 0 && (
                <div>
                  <div style={{ ...S.label, marginBottom: 8 }}>Top Predictions</div>
                  {result.top_predictions.map((tp) => (
                    <div key={tp.class_idx} style={{ marginBottom: 6 }}>
                      <div
                        style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}
                      >
                        <span
                          style={{
                            color:
                              tp.class_idx === result.predicted_class ? "#a78bfa" : "#64748b",
                          }}
                        >
                          Klasse {tp.class_idx}
                        </span>
                        <span style={{ color: "#94a3b8" }}>{pct(tp.score)}</span>
                      </div>
                      <div
                        style={{
                          height: 4,
                          borderRadius: 2,
                          background: "#1e293b",
                          marginTop: 3,
                          overflow: "hidden",
                        }}
                      >
                        <div style={S.barFill(tp.score)} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── Hinweis kein model.pt ───────────────────────────────────────────── */}
      {selected && !selected.has_weights && (
        <div
          style={{
            ...S.section,
            border: "1px solid #f59e0b60",
            background: "rgba(245,158,11,0.06)",
            fontSize: 11,
            color: "#fbbf24",
          }}
        >
          ⚠️ Dieses Modell hat noch kein <code>model.pt</code> – erst Training durchführen.
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default CanvasInferenceTab;
