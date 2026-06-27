import React, { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

// ─── Types ────────────────────────────────────────────────────────────────────
export interface SavedModel {
  id: string;
  name: string;
  savedAt: number;
  nodeCount: number;
  edgeCount: number;
  nodes: any[];
  edges: any[];
  viewport?: { x: number; y: number; zoom: number };
  graphConfig?: any;
  pythonCode?: string;
  moduleMetadata?: any;
  /** Backend canvas_XXXX ID — erforderlich für SQLite-Persistenz */
  canvasModelId?: string;
}

// ─── CanvasDesign: was in SQLite gespeichert wird ─────────────────────────────
export interface CanvasDesign {
  nodes: any[];
  edges: any[];
  viewport?: { x: number; y: number; zoom: number };
  graphConfig?: any;
  pythonCode?: string;
  schemaVersion: number;
}

interface ModelLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  onLoad: (model: SavedModel) => void;
  userId: string; // required — kein anonymes Speichern
  onSwitchToSessions?: () => void;
}

// ─── SQLite-backed storage helpers (exportiert für SynapseBuilder) ────────────
// W3: localStorage wird nicht mehr für Canvas-Designs verwendet.
// SQLite ist die einzige Source of Truth.

/**
 * Liest alle gespeicherten Canvas-Modelle aus der models-Tabelle (SQLite).
 * Gibt ein leeres Array zurück wenn keine Modelle vorhanden oder Fehler aufgetreten.
 * KEIN localStorage-Zugriff.
 */
export async function readModels(userId: string): Promise<SavedModel[]> {
  try {
    const raw = await invoke<{ model_id: string; name: string; has_weights: boolean; model_pt_path: string; metadata_path: string; task_type: string; num_classes: number }[]>(
      "list_canvas_models_with_pt",
      { userId }
    );
    // Nur Metadaten — nodes/edges werden lazy via loadModelDesign geladen
    return raw.map((m) => ({
      id: m.model_id,
      name: m.name,
      savedAt: 0,
      nodeCount: 0,
      edgeCount: 0,
      nodes: [],
      edges: [],
      canvasModelId: m.model_id,
    }));
  } catch {
    return [];
  }
}

/**
 * Speichert das Canvas-Design eines Modells in SQLite.
 * model_id muss eine bekannte canvas_XXXX ID sein.
 * KEIN localStorage-Zugriff.
 */
export async function writeModelDesign(
  modelId: string,
  design: CanvasDesign
): Promise<void> {
  await invoke("save_canvas_model_design", {
    modelId,
    designJson: JSON.stringify(design),
  });
}

/**
 * Laedt das Canvas-Design eines Modells aus SQLite.
 * Gibt null zurueck wenn kein Design gespeichert.
 */
export async function loadModelDesign(modelId: string): Promise<CanvasDesign | null> {
  try {
    const raw = await invoke<string | null>("load_canvas_model_design", { modelId });
    if (!raw) return null;
    return JSON.parse(raw) as CanvasDesign;
  } catch {
    return null;
  }
}

/**
 * Löscht das Canvas-Design eines Modells aus SQLite.
 */
export async function deleteModelDesign(modelId: string): Promise<void> {
  await invoke("delete_canvas_model_design", { modelId });
}

// ─── Legacy sync helpers — NUR für SynapseBuilder Kompatibilität ──────────────
// readModels/writeModels bleiben als synchrone Wrapper für Orte die
// noch keinen async-Umbau erhalten haben. Sie greifen NICHT auf localStorage zu.
/** @deprecated Verwende readModels (async) direkt. Diese Variante gibt leeres Array zurück. */
export function readModelsSync(_userId: string): SavedModel[] {
  // Synchroner Zugriff auf SQLite ist nicht möglich.
  // SynapseBuilder wurde auf async umgebaut — diese Funktion ist nur Stub.
  return [];
}

function formatDate(ts: number): string {
  const d = new Date(ts);
  return (
    d.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit" }) +
    " · " +
    d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })
  );
}

// ─── Component ────────────────────────────────────────────────────────────────
export const ModelLibrary: React.FC<ModelLibraryProps> = ({ isOpen, onClose, onLoad, userId, onSwitchToSessions }) => {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    readModels(userId)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [isOpen, userId]);

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteModelDesign(id).catch(() => {});
      setModels((prev) => prev.filter((m) => m.id !== id));
    },
    []
  );

  const handleLoad = useCallback(
    async (model: SavedModel) => {
      const canvasId = model.canvasModelId ?? model.id;
      const design = await loadModelDesign(canvasId).catch(() => null);
      const loaded: SavedModel = design
        ? {
            ...model,
            nodes: design.nodes,
            edges: design.edges,
            viewport: design.viewport,
            graphConfig: design.graphConfig,
            pythonCode: design.pythonCode,
            nodeCount: design.nodes.length,
            edgeCount: design.edges.length,
          }
        : model;
      onLoad(loaded);
      onClose();
    },
    [onLoad, onClose]
  );

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(3px)",
          zIndex: 1000,
        }}
      />

      {/* Modal */}
      <div
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 520,
          maxHeight: "75vh",
          background: "#0d1117",
          border: "1px solid #1e293b",
          borderRadius: 12,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          zIndex: 1001,
          boxShadow: "0 24px 64px rgba(0,0,0,0.7)",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            padding: "14px 18px",
            borderBottom: "1px solid #1e293b",
            flexShrink: 0,
          }}
        >
          <span
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
              flex: 1,
            }}
          >
            ◈ Gespeicherte Modelle
          </span>
          {onSwitchToSessions && (
            <button
              onClick={onSwitchToSessions}
              style={{
                fontSize: 10, color: "#a78bfa", background: "none",
                border: "1px solid #334155", borderRadius: 4, padding: "2px 8px",
                cursor: "pointer", marginRight: 8, fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              ← Sessions
            </button>
          )}
          <span
            style={{
              fontSize: 11,
              color: "#334155",
              fontFamily: "'JetBrains Mono', monospace",
              marginRight: 12,
            }}
          >
            {models.length} Modell{models.length !== 1 ? "e" : ""}
          </span>
          <button onClick={onClose} style={iconBtn}>
            ✕
          </button>
        </div>

        {/* Model list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
          {loading ? (
            <div style={{ textAlign: "center", padding: "40px 20px", color: "#334155", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
              Lade Modelle…
            </div>
          ) : models.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                padding: "40px 20px",
                color: "#1e293b",
                fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace",
                lineHeight: 2,
              }}
            >
              Noch keine Modelle gespeichert.
              <br />
              Baue einen Graph und klicke „Speichern".
            </div>
          ) : (
            models.map((model) => (
              <div
                key={model.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 12px",
                  borderRadius: 8,
                  border: "1px solid #1e293b",
                  background: "#0a0e17",
                  marginBottom: 6,
                  transition: "border-color 0.12s",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#334155")}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = "#1e293b")}
              >
                {/* Icon */}
                <div
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: 7,
                    background: "rgba(167,139,250,0.12)",
                    border: "1px solid rgba(167,139,250,0.2)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 14,
                    flexShrink: 0,
                  }}
                >
                  ◈
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: "#e2e8f0",
                      fontFamily: "'JetBrains Mono', monospace",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {model.name}
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color: "#334155",
                      fontFamily: "'JetBrains Mono', monospace",
                      marginTop: 2,
                    }}
                  >
                    {formatDate(model.savedAt)} · {model.nodeCount} nodes · {model.edgeCount} edges
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 5, flexShrink: 0 }}>
                  <button onClick={() => handleLoad(model)} style={loadBtn}>
                    Laden
                  </button>
                  <button
                    onClick={() => handleDelete(model.id)}
                    style={deleteBtn}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.color = "#f87171";
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "#7f1d1d";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.color = "#475569";
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "#1e293b";
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        {models.length > 0 && (
          <div
            style={{
              padding: "8px 18px",
              borderTop: "1px solid #0f172a",
              fontSize: 10,
              color: "#1e293b",
              fontFamily: "'JetBrains Mono', monospace",
              flexShrink: 0,
            }}
          >
            Modelle werden in SQLite gespeichert – nur für deinen Account sichtbar.
          </div>
        )}
      </div>
    </>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const iconBtn: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 24,
  height: 24,
  background: "transparent",
  border: "none",
  cursor: "pointer",
  fontSize: 13,
  borderRadius: 4,
  color: "#475569",
};

const loadBtn: React.CSSProperties = {
  padding: "4px 10px",
  borderRadius: 5,
  border: "1px solid #4f46e5",
  background: "rgba(79,70,229,0.12)",
  color: "#818cf8",
  fontSize: 11,
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', monospace",
  fontWeight: 600,
};

const deleteBtn: React.CSSProperties = {
  padding: "4px 8px",
  borderRadius: 5,
  border: "1px solid #1e293b",
  background: "transparent",
  color: "#475569",
  fontSize: 11,
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', monospace",
};

export default ModelLibrary;
