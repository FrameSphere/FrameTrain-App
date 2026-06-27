import React, { useState, useCallback } from "react";
import { NODE_CATEGORIES, NodeDefinition } from "./nodeTypes";
import { dragState } from "./dragState";

// ─── Icons & Colors ───────────────────────────────────────────────────────────
const icons: Record<string, string> = {
  Data:        "⬡",
  Layers:      "◈",
  Activations: "⚡",
  Training:    "⚙",
  Logic:       "⬦",
  Math:        "∑",
  Advanced:    "✦",
};

const categoryColors: Record<string, string> = {
  Data:        "#38bdf8",
  Layers:      "#a78bfa",
  Activations: "#fb923c",
  Training:    "#34d399",
  Logic:       "#f472b6",
  Math:        "#facc15",
  Advanced:    "#f87171",
};

// ─── Props ────────────────────────────────────────────────────────────────────
interface NodeLibraryProps {
  onAddNode: (definition: NodeDefinition, position?: { x: number; y: number }) => void;
}

// ─── NodeCard ─────────────────────────────────────────────────────────────────
const NodeCard: React.FC<{
  node: NodeDefinition;
  color: string;
  onAdd: () => void;
}> = ({ node, color, onAdd }) => {
  const [hovered, setHovered] = useState(false);

  // Sets dragState.nodeType (shared module var) — reliable in Tauri WKWebView.
  // Also writes text/plain as a standard fallback.
  // Ghost element stays in DOM until dragend to avoid WebKit capture issues.
  const handleDragStart = useCallback(
    (e: React.DragEvent) => {
      // Primary: module-level shared state (bypasses dataTransfer MIME issues)
      dragState.nodeType = node.type;

      // Fallback: text/plain is universally supported
      e.dataTransfer.setData("text/plain", node.type);
      e.dataTransfer.effectAllowed = "copy";

      // Ghost drag image — keep in DOM until dragend (not setTimeout 0)
      const ghost = document.createElement("div");
      ghost.innerText = node.label;
      ghost.style.cssText = [
        "position:fixed",
        "top:-200px",
        "left:-200px",
        "background:#1e293b",
        "color:#a78bfa",
        "padding:4px 12px",
        "border-radius:5px",
        "font-size:12px",
        "font-family:monospace",
        "border:1px solid #a78bfa40",
        "white-space:nowrap",
        "pointer-events:none",
      ].join(";");
      document.body.appendChild(ghost);
      e.dataTransfer.setDragImage(ghost, 0, 0);
      // Remove only after drag ends, not immediately
      e.currentTarget.addEventListener("dragend", () => {
        if (document.body.contains(ghost)) document.body.removeChild(ghost);
        dragState.nodeType = null;
      }, { once: true });
    },
    [node]
  );

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onAdd}
      title={node.description}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 10px",
        borderRadius: 6,
        cursor: "grab",
        background: hovered ? `${color}18` : "transparent",
        border: `1px solid ${hovered ? color + "55" : "transparent"}`,
        transition: "all 0.12s ease",
        userSelect: "none",
      }}
    >
      <div
        style={{
          width: 7,
          height: 7,
          borderRadius: 2,
          background: color,
          flexShrink: 0,
          boxShadow: hovered ? `0 0 6px ${color}` : "none",
          transition: "box-shadow 0.12s",
        }}
      />
      <span
        style={{
          fontSize: 12,
          color: hovered ? "#e2e8f0" : "#94a3b8",
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          transition: "color 0.12s",
          flex: 1,
        }}
      >
        {node.label}
      </span>
      {hovered && (
        <span style={{ fontSize: 11, color: color, opacity: 0.8, lineHeight: 1 }}>+</span>
      )}
    </div>
  );
};

// ─── CategorySection ──────────────────────────────────────────────────────────
const CategorySection: React.FC<{
  category: string;
  nodes: NodeDefinition[];
  onAddNode: (def: NodeDefinition) => void;
}> = ({ category, nodes, onAddNode }) => {
  const [open, setOpen] = useState(true);
  const color = categoryColors[category] ?? "#64748b";

  return (
    <div style={{ marginBottom: 2 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "7px 10px",
          background: "transparent",
          border: "none",
          cursor: "pointer",
          borderRadius: 5,
        }}
      >
        <span style={{ fontSize: 13, color, lineHeight: 1 }}>{icons[category] ?? "◆"}</span>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: "#94a3b8",
            letterSpacing: "0.09em",
            textTransform: "uppercase",
            flex: 1,
            textAlign: "left",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {category}
        </span>
        <span
          style={{
            fontSize: 9,
            color: "#334155",
            display: "inline-block",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s",
          }}
        >
          ▶
        </span>
      </button>

      {open && (
        <div style={{ paddingLeft: 2 }}>
          {nodes.map((node) => (
            <NodeCard
              key={node.type}
              node={node}
              color={color}
              onAdd={() => onAddNode(node)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────
export const NodeLibrary: React.FC<NodeLibraryProps> = ({ onAddNode }) => {
  const [search, setSearch] = useState("");

  const handleAdd = useCallback(
    (def: NodeDefinition) => onAddNode(def),
    [onAddNode]
  );

  const filtered = search.trim()
    ? Object.entries(NODE_CATEGORIES).reduce<Record<string, NodeDefinition[]>>(
        (acc, [cat, nodes]) => {
          const matches = nodes.filter(
            (n) =>
              n.label.toLowerCase().includes(search.toLowerCase()) ||
              n.type.toLowerCase().includes(search.toLowerCase())
          );
          if (matches.length) acc[cat] = matches;
          return acc;
        },
        {}
      )
    : NODE_CATEGORIES;

  return (
    <div
      style={{
        width: 220,
        minWidth: 220,
        // ✅ height:100% + flex column so scroll works inside bounded space
        height: "100%",
        background: "#0d1117",
        borderRight: "1px solid #1e293b",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",   // outer container clips
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 12px 10px",
          borderBottom: "1px solid #1e293b",
          flexShrink: 0,   // ✅ never shrinks
        }}
      >
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.12em",
            color: "#334155",
            textTransform: "uppercase",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 9,
          }}
        >
          Node Library
        </div>

        {/* Search */}
        <div style={{ position: "relative" }}>
          <span
            style={{
              position: "absolute",
              left: 9,
              top: "50%",
              transform: "translateY(-50%)",
              fontSize: 12,
              color: "#334155",
              pointerEvents: "none",
              lineHeight: 1,
            }}
          >
            ⌕
          </span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search nodes…"
            style={{
              width: "100%",
              padding: "5px 8px 5px 24px",
              background: "#111827",
              border: "1px solid #1e293b",
              borderRadius: 5,
              color: "#e2e8f0",
              fontSize: 12,
              fontFamily: "'JetBrains Mono', monospace",
              outline: "none",
              boxSizing: "border-box",
            }}
          />
        </div>
      </div>

      {/* ✅ Scrollable node list */}
      <div
        className="synapse-library"
        style={{
          flex: 1,
          overflowY: "auto",   // scrolls when content overflows
          overflowX: "hidden",
          padding: "6px 4px",
          // custom scrollbar (webkit)
          scrollbarWidth: "thin",
          scrollbarColor: "#1e293b transparent",
        }}
      >
        <style>{`
          .synapse-library::-webkit-scrollbar { width: 4px; }
          .synapse-library::-webkit-scrollbar-track { background: transparent; }
          .synapse-library::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
        `}</style>

        {Object.keys(filtered).length === 0 ? (
          <div
            style={{
              textAlign: "center",
              padding: "20px 12px",
              color: "#334155",
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            No nodes found
          </div>
        ) : (
          Object.entries(filtered).map(([category, nodes]) => (
            <CategorySection
              key={category}
              category={category}
              nodes={nodes}
              onAddNode={handleAdd}
            />
          ))
        )}
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "7px 12px",
          borderTop: "1px solid #1e293b",
          fontSize: 10,
          color: "#1e293b",
          fontFamily: "'JetBrains Mono', monospace",
          textAlign: "center",
          flexShrink: 0,
        }}
      >
        drag or click to add
      </div>
    </div>
  );
};

export default NodeLibrary;
