import React, { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { exportCanvasNetworkToModelLibrary } from "./canvasModelBridge";
import { buildCanvasGraphIR } from "./graphIR";

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
  /** model.pt vorhanden — Modell wurde bereits trainiert */
  hasWeights?: boolean;
  taskType?: string;
  numClasses?: number;
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
      hasWeights: m.has_weights,
      taskType: m.task_type,
      numClasses: m.num_classes,
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
export const ModelLibrary: React.FC<ModelLibraryProps> = ({ isOpen, onClose, onLoad, userId }) => {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [loading, setLoading] = useState(false);
  // Zwei-Klick-Löschen: erster Klick auf ✕ merkt die ID, zweiter löscht wirklich
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  // Inline-Umbenennen
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  useEffect(() => {
    if (!isOpen) return;
    setConfirmDeleteId(null);
    setRenameId(null);
    setLoading(true);
    readModels(userId)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [isOpen, userId]);

  // Löscht das Modell WIRKLICH (Backend: Ordner + DB + Metadata) und das Design.
  // Vorher wurde nur das Design gelöscht — das Modell tauchte danach wieder auf.
  const handleDelete = useCallback(
    async (id: string) => {
      if (confirmDeleteId !== id) {
        setConfirmDeleteId(id);
        return;
      }
      setConfirmDeleteId(null);
      await deleteModelDesign(id).catch(() => {});
      await invoke("delete_model", { modelId: id }).catch(() => {});
      setModels((prev) => prev.filter((m) => m.id !== id));
    },
    [confirmDeleteId]
  );

  const startRename = useCallback((m: SavedModel) => {
    setRenameId(m.id);
    setRenameValue(m.name);
  }, []);

  // Kopie anlegen — für Experimente am Design, ohne das Original anzufassen
  const [duplicatingId, setDuplicatingId] = useState<string | null>(null);
  const handleDuplicate = useCallback(async (m: SavedModel) => {
    if (duplicatingId) return;
    setDuplicatingId(m.id);
    try {
      const design = await loadModelDesign(m.canvasModelId ?? m.id);
      if (!design) return;
      const graphIR = buildCanvasGraphIR(design.nodes as any, design.edges as any, {
        epochs: 10, batchSize: 32, learningRate: 0.001, gpu: "cpu", precision: "fp32", gradAccum: 1,
      });
      const name = `${m.name} (Kopie)`;
      const result = await exportCanvasNetworkToModelLibrary(
        design.graphConfig, design.pythonCode ?? "", name, graphIR
      );
      await writeModelDesign(result.modelId, design);
      const fresh = await readModels(userId);
      setModels(fresh);
    } catch { /* Kopie fehlgeschlagen — Liste unverändert */ }
    finally { setDuplicatingId(null); }
  }, [duplicatingId, userId]);

  const commitRename = useCallback(async () => {
    const id = renameId;
    const name = renameValue.trim();
    setRenameId(null);
    if (!id || !name) return;
    try {
      await invoke("rename_model", { modelId: id, newName: name });
      setModels((prev) => prev.map((m) => (m.id === id ? { ...m, name } : m)));
    } catch { /* Name bleibt unverändert */ }
  }, [renameId, renameValue]);

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
            ◈ Modell-Bibliothek
          </span>
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
                  {renameId === model.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") setRenameId(null);
                      }}
                      onBlur={commitRename}
                      style={{
                        width: "100%", background: "#111827", border: "1px solid #4f46e5",
                        borderRadius: 5, color: "#e2e8f0", fontSize: 12, padding: "3px 8px",
                        fontFamily: "'JetBrains Mono', monospace", outline: "none",
                      }}
                    />
                  ) : (
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
                      title={model.name}
                    >
                      {model.name}
                    </div>
                  )}
                  <div
                    style={{
                      fontSize: 10,
                      color: "#334155",
                      fontFamily: "'JetBrains Mono', monospace",
                      marginTop: 2,
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    {model.hasWeights && (
                      <span style={{ color: "#34d399", border: "1px solid rgba(16,185,129,0.35)", borderRadius: 4, padding: "0 5px" }}>
                        ✓ trainiert
                      </span>
                    )}
                    {model.taskType && <span>{model.taskType}</span>}
                    {model.numClasses != null && model.numClasses > 0 && <span>· {model.numClasses} Klassen</span>}
                    {model.savedAt > 0 && <span>· {formatDate(model.savedAt)}</span>}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 5, flexShrink: 0 }}>
                  <button onClick={() => handleLoad(model)} style={loadBtn}>
                    Laden
                  </button>
                  <button
                    onClick={() => startRename(model)}
                    style={deleteBtn}
                    title="Umbenennen"
                  >
                    ✎
                  </button>
                  <button
                    onClick={() => handleDuplicate(model)}
                    style={{ ...deleteBtn, ...(duplicatingId === model.id ? { opacity: 0.5 } : {}) }}
                    disabled={duplicatingId !== null}
                    title="Duplizieren — Kopie zum Experimentieren"
                  >
                    ⧉
                  </button>
                  <button
                    onClick={() => handleDelete(model.id)}
                    style={{
                      ...deleteBtn,
                      ...(confirmDeleteId === model.id
                        ? { color: "#f87171", borderColor: "#7f1d1d", background: "rgba(127,29,29,0.2)" }
                        : {}),
                    }}
                    title={confirmDeleteId === model.id ? "Nochmal klicken zum endgültigen Löschen" : "Modell löschen"}
                    onMouseLeave={() => setConfirmDeleteId((c) => (c === model.id ? null : c))}
                  >
                    {confirmDeleteId === model.id ? "Löschen?" : "✕"}
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
            Modelle sind an deinen Account gebunden und im Training-Panel trainierbar.
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
