import React, { useState, useEffect, useCallback } from "react";
import { Node, Edge } from "@xyflow/react";

// ─── Types ────────────────────────────────────────────────────────────────────
export interface SavedSession {
  id: string;
  name: string;
  savedAt: number;
  nodeCount: number;
  edgeCount: number;
  nodes: Node[];
  edges: Edge[];
  viewport?: { x: number; y: number; zoom: number };
}

interface SessionLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  currentNodes: Node[];
  currentEdges: Edge[];
  currentViewport?: { x: number; y: number; zoom: number };
  onLoad: (session: SavedSession) => void;
  onSwitchToModels?: () => void;
}

// ─── localStorage helpers ─────────────────────────────────────────────────────
const KEY = "synapse_sessions_v1";

function readSessions(): SavedSession[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? "[]"); }
  catch { return []; }
}
function writeSessions(s: SavedSession[]) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* quota */ }
}
function fmt(ts: number) {
  const d = new Date(ts);
  return d.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "2-digit" })
    + " · " + d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
}

// ─── Component ────────────────────────────────────────────────────────────────
export const SessionLibrary: React.FC<SessionLibraryProps> = ({
  isOpen, onClose, currentNodes, currentEdges, currentViewport, onLoad, onSwitchToModels,
}) => {
  const [sessions, setSessions] = useState<SavedSession[]>([]);
  const [name, setName]         = useState("");
  const [saved, setSaved]       = useState(false);

  useEffect(() => { if (isOpen) setSessions(readSessions()); }, [isOpen]);

  const handleSave = useCallback(() => {
    const label = name.trim() || `Modell ${new Date().toLocaleDateString("de-DE")}`;
    const s: SavedSession = {
      id: `session-${Date.now()}`,
      name: label, savedAt: Date.now(),
      nodeCount: currentNodes.length, edgeCount: currentEdges.length,
      nodes: currentNodes, edges: currentEdges, viewport: currentViewport,
    };
    const next = [s, ...sessions];
    writeSessions(next); setSessions(next); setName("");
    setSaved(true); setTimeout(() => setSaved(false), 1800);
  }, [name, sessions, currentNodes, currentEdges, currentViewport]);

  const handleDelete = useCallback((id: string) => {
    const next = sessions.filter((s) => s.id !== id);
    writeSessions(next); setSessions(next);
  }, [sessions]);

  const handleLoad = useCallback((s: SavedSession) => {
    onLoad(s); onClose();
  }, [onLoad, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div onClick={onClose} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.55)", backdropFilter:"blur(3px)", zIndex:1000 }} />

      {/* Modal */}
      <div style={{
        position:"fixed", top:"50%", left:"50%", transform:"translate(-50%,-50%)",
        width:520, maxHeight:"75vh", background:"#0d1117",
        border:"1px solid #1e293b", borderRadius:12,
        display:"flex", flexDirection:"column", overflow:"hidden",
        zIndex:1001, boxShadow:"0 24px 64px rgba(0,0,0,0.7)",
        fontFamily:"'JetBrains Mono',monospace",
      }}>
        {/* Header */}
        <div style={{ display:"flex", alignItems:"center", padding:"14px 18px", borderBottom:"1px solid #1e293b", flexShrink:0 }}>
          <span style={{ fontSize:13, fontWeight:700, color:"#e2e8f0", flex:1 }}>◈ Sessions</span>
          {onSwitchToModels && (
            <button onClick={onSwitchToModels} style={{ fontSize:10, color:"#a78bfa", background:"none", border:"1px solid #334155", borderRadius:4, padding:"2px 8px", cursor:"pointer", marginRight:8 }}>
              Modelle →
            </button>
          )}
          <span style={{ fontSize:11, color:"#334155", marginRight:12 }}>{sessions.length} gespeichert</span>
          <button onClick={onClose} style={{ background:"none", border:"none", color:"#475569", cursor:"pointer", fontSize:14 }}>✕</button>
        </div>

        {/* Save row */}
        <div style={{ display:"flex", gap:8, padding:"12px 18px", borderBottom:"1px solid #0f172a", flexShrink:0, background:"#090d13" }}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
            placeholder={`Aktuellen Graph speichern… (${currentNodes.length} Nodes)`}
            style={{ flex:1, background:"#111827", border:"1px solid #1e293b", borderRadius:6, color:"#e2e8f0", fontSize:12, padding:"6px 10px", fontFamily:"'JetBrains Mono',monospace", outline:"none" }}
          />
          <button
            onClick={handleSave}
            disabled={currentNodes.length === 0}
            style={{
              padding:"6px 14px", borderRadius:6, border:"none",
              background: saved ? "#059669" : "#4f46e5",
              color:"#fff", fontSize:11, fontWeight:600,
              cursor: currentNodes.length === 0 ? "not-allowed" : "pointer",
              fontFamily:"'JetBrains Mono',monospace",
              opacity: currentNodes.length === 0 ? 0.4 : 1,
              transition:"background 0.2s", whiteSpace:"nowrap",
            }}
          >
            {saved ? "✓ Gespeichert" : "Speichern"}
          </button>
        </div>

        {/* List */}
        <div style={{ flex:1, overflowY:"auto", padding:"8px 10px", scrollbarWidth:"thin", scrollbarColor:"#1e293b transparent" }}>
          {sessions.length === 0 ? (
            <div style={{ textAlign:"center", padding:"40px 20px", color:"#1e293b", fontSize:12, lineHeight:2 }}>
              Noch keine Modelle gespeichert.<br/>Baue einen Graph und klicke "Speichern".
            </div>
          ) : sessions.map((s) => (
            <div
              key={s.id}
              style={{ display:"flex", alignItems:"center", gap:10, padding:"10px 12px", borderRadius:8, border:"1px solid #1e293b", background:"#0a0e17", marginBottom:6, transition:"border-color 0.12s" }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#334155")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "#1e293b")}
            >
              <div style={{ width:34, height:34, borderRadius:7, background:"rgba(167,139,250,0.12)", border:"1px solid rgba(167,139,250,0.2)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14, flexShrink:0 }}>◈</div>
              <div style={{ flex:1, minWidth:0 }}>
                <div style={{ fontSize:12, fontWeight:600, color:"#e2e8f0", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{s.name}</div>
                <div style={{ fontSize:10, color:"#334155", marginTop:2 }}>{fmt(s.savedAt)} · {s.nodeCount} nodes · {s.edgeCount} edges</div>
              </div>
              <div style={{ display:"flex", gap:5, flexShrink:0 }}>
                <button onClick={() => handleLoad(s)} style={{ padding:"4px 10px", borderRadius:5, border:"1px solid #4f46e5", background:"rgba(79,70,229,0.12)", color:"#818cf8", fontSize:11, cursor:"pointer", fontWeight:600 }}>Laden</button>
                <button
                  onClick={() => handleDelete(s.id)}
                  style={{ padding:"4px 8px", borderRadius:5, border:"1px solid #1e293b", background:"transparent", color:"#475569", fontSize:11, cursor:"pointer" }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color="#f87171"; (e.currentTarget as HTMLButtonElement).style.borderColor="#7f1d1d"; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color="#475569"; (e.currentTarget as HTMLButtonElement).style.borderColor="#1e293b"; }}
                >✕</button>
              </div>
            </div>
          ))}
        </div>

        {sessions.length > 0 && (
          <div style={{ padding:"8px 18px", borderTop:"1px solid #0f172a", fontSize:10, color:"#1e293b", flexShrink:0 }}>
            Modelle werden lokal gespeichert.
          </div>
        )}
      </div>
    </>
  );
};

export default SessionLibrary;
