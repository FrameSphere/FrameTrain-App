import React, { useCallback } from "react";
import { Node } from "@xyflow/react";
import { NodeDefinition, ParamDefinition } from "./nodeTypes";
import { useLanguage } from "../../contexts/LanguageContext";

interface PropertyPanelProps {
  selectedNode: Node | null;
  definition: NodeDefinition | null;
  onParamChange: (nodeId: string, paramKey: string, value: unknown) => void;
}

const categoryColors: Record<string, string> = {
  Data:        "#38bdf8",
  Layers:      "#a78bfa",
  Activations: "#fb923c",
  Training:    "#34d399",
  Logic:       "#f472b6",
  Math:        "#facc15",
  Advanced:    "#f87171",
};

// ─── Field base style ─────────────────────────────────────────────────────────
const inputBase: React.CSSProperties = {
  width: "100%",
  padding: "6px 9px",
  background: "#111827",
  border: "1px solid #1e293b",
  borderRadius: 5,
  color: "#e2e8f0",
  fontSize: 12,
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  outline: "none",
  boxSizing: "border-box",
  transition: "border-color 0.12s",
};

// ─── Field components ─────────────────────────────────────────────────────────
const FieldNumber: React.FC<{
  param: ParamDefinition; value: number; accent: string; onChange: (v: number) => void;
}> = ({ param, value, accent, onChange }) => (
  <input
    type="number"
    value={value}
    min={param.min}
    max={param.max}
    step={param.step ?? 1}
    onChange={(e) => onChange(Number(e.target.value))}
    onFocus={(e) => (e.target.style.borderColor = accent)}
    onBlur={(e)  => (e.target.style.borderColor = "#1e293b")}
    style={inputBase}
  />
);

const FieldText: React.FC<{
  value: string; accent: string; onChange: (v: string) => void;
}> = ({ value, accent, onChange }) => (
  <input
    type="text"
    value={value}
    onChange={(e) => onChange(e.target.value)}
    onFocus={(e) => (e.target.style.borderColor = accent)}
    onBlur={(e)  => (e.target.style.borderColor = "#1e293b")}
    style={inputBase}
  />
);

const FieldSelect: React.FC<{
  param: ParamDefinition; value: string; accent: string; onChange: (v: string) => void;
}> = ({ param, value, accent, onChange }) => (
  <select
    value={value}
    onChange={(e) => onChange(e.target.value)}
    onFocus={(e) => (e.target.style.borderColor = accent)}
    onBlur={(e)  => (e.target.style.borderColor = "#1e293b")}
    style={{ ...inputBase, cursor: "pointer" }}
  >
    {(param.options ?? []).map((opt) => (
      <option key={opt} value={opt} style={{ background: "#111827" }}>
        {opt}
      </option>
    ))}
  </select>
);

const FieldBool: React.FC<{
  label: string; value: boolean; accent: string; onChange: (v: boolean) => void;
}> = ({ label, value, accent, onChange }) => (
  <button
    onClick={() => onChange(!value)}
    style={{
      display: "flex", alignItems: "center", gap: 8,
      background: "transparent", border: "none", cursor: "pointer", padding: 0,
    }}
  >
    <div
      style={{
        width: 30, height: 16, borderRadius: 8,
        background: value ? accent : "#1e293b",
        position: "relative", transition: "background 0.15s", flexShrink: 0,
      }}
    >
      <div
        style={{
          position: "absolute", top: 2, left: value ? 15 : 2,
          width: 12, height: 12, borderRadius: "50%",
          background: "#fff", transition: "left 0.15s",
        }}
      />
    </div>
    <span style={{ fontSize: 12, color: value ? "#e2e8f0" : "#64748b" }}>{label}</span>
  </button>
);

const FieldCode: React.FC<{
  value: string; accent: string; onChange: (v: string) => void;
}> = ({ value, accent, onChange }) => (
  <textarea
    value={value}
    onChange={(e) => onChange(e.target.value)}
    onFocus={(e) => (e.target.style.borderColor = accent)}
    onBlur={(e)  => (e.target.style.borderColor = "#1e293b")}
    rows={5}
    spellCheck={false}
    style={{ ...inputBase, resize: "vertical", lineHeight: 1.5 }}
  />
);

// ─── Param row ────────────────────────────────────────────────────────────────
const ParamRow: React.FC<{
  param: ParamDefinition; value: unknown; accent: string; onChange: (v: unknown) => void;
}> = ({ param, value, accent, onChange }) => {
  const renderInput = () => {
    switch (param.type) {
      case "number":  return <FieldNumber param={param} value={value as number} accent={accent} onChange={onChange} />;
      case "select":  return <FieldSelect param={param} value={value as string} accent={accent} onChange={onChange} />;
      case "boolean": return <FieldBool label={param.label} value={value as boolean} accent={accent} onChange={onChange} />;
      case "code":    return <FieldCode value={value as string} accent={accent} onChange={onChange} />;
      default:        return <FieldText value={value as string} accent={accent} onChange={onChange} />;
    }
  };

  return (
    <div style={{ marginBottom: 13 }}>
      {param.type !== "boolean" && (
        <label
          style={{
            display: "block", fontSize: 10, fontWeight: 600,
            color: "#475569", letterSpacing: "0.08em",
            textTransform: "uppercase",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 5,
          }}
        >
          {param.label}
          {param.description && (
            <span title={param.description} style={{ marginLeft: 4, opacity: 0.5, cursor: "help" }}>?</span>
          )}
        </label>
      )}
      {renderInput()}
    </div>
  );
};

// ─── Main ─────────────────────────────────────────────────────────────────────
export const PropertyPanel: React.FC<PropertyPanelProps> = ({
  selectedNode,
  definition,
  onParamChange,
}) => {
  const { t } = useLanguage();
  const accent = categoryColors[definition?.category ?? ""] ?? "#64748b";

  const handleChange = useCallback(
    (key: string, val: unknown) => {
      if (!selectedNode) return;
      onParamChange(selectedNode.id, key, val);
    },
    [selectedNode, onParamChange]
  );

  return (
    <div
      style={{
        // ✅ Fixed width + proper height containment — no more cut-off
        width: 240,
        minWidth: 240,
        maxWidth: 240,
        height: "100%",
        background: "#0d1117",
        borderLeft: "1px solid #1e293b",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        flexShrink: 0,   // ✅ never squished by canvas
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 14px 10px",
          borderBottom: "1px solid #1e293b",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontSize: 10, fontWeight: 700, letterSpacing: "0.12em",
            color: "#334155", textTransform: "uppercase",
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {t('synapse.propertyPanel.title')}
        </div>
      </div>

      {/* ✅ Scrollable body */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          overflowX: "hidden",
          scrollbarWidth: "thin",
          scrollbarColor: "#1e293b transparent",
        }}
      >
        {!selectedNode || !definition ? (
          /* Empty state */
          <div
            style={{
              display: "flex", flexDirection: "column", alignItems: "center",
              justifyContent: "center", height: "100%", padding: 24,
              gap: 10, textAlign: "center",
            }}
          >
            <div style={{ fontSize: 24, opacity: 0.08 }}>◈</div>
            <div
              style={{
                fontSize: 11, color: "#334155",
                fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.7,
              }}
            >
              {t('synapse.propertyPanel.emptyHint')}
            </div>
          </div>
        ) : (
          <div style={{ padding: "14px 14px" }}>
            {/* Node header */}
            <div style={{ marginBottom: 16 }}>
              <div
                style={{
                  fontSize: 13, fontWeight: 600, color: accent,
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 5,
                }}
              >
                {definition.label}
              </div>
              <div
                style={{
                  display: "inline-block", fontSize: 10, color: accent,
                  background: `${accent}18`, border: `1px solid ${accent}30`,
                  borderRadius: 4, padding: "2px 7px",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                {definition.category}
              </div>
              {definition.description && (
                <div
                  style={{
                    marginTop: 8, fontSize: 11, color: "#475569",
                    fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.6,
                  }}
                >
                  {definition.description}
                </div>
              )}
            </div>

            <div style={{ height: 1, background: "#1e293b", marginBottom: 14 }} />

            {/* Node ID */}
            <div style={{ marginBottom: 14 }}>
              <label
                style={{
                  display: "block", fontSize: 10, fontWeight: 600,
                  color: "#334155", letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace", marginBottom: 4,
                }}
              >
                {t('synapse.propertyPanel.nodeIdLabel')}
              </label>
              <div
                style={{
                  fontSize: 10, color: "#334155",
                  fontFamily: "'JetBrains Mono', monospace",
                  background: "#0a0e17", border: "1px solid #1e293b",
                  borderRadius: 5, padding: "5px 9px",
                  wordBreak: "break-all",
                }}
              >
                {selectedNode.id}
              </div>
            </div>

            <div style={{ height: 1, background: "#1e293b", marginBottom: 14 }} />

            {/* Params */}
            {definition.params.length === 0 ? (
              <div
                style={{
                  fontSize: 11, color: "#334155",
                  fontFamily: "'JetBrains Mono', monospace",
                  textAlign: "center", padding: "10px 0",
                }}
              >
                {t('synapse.propertyPanel.noParams')}
              </div>
            ) : (
              definition.params.map((param) => {
                const paramsMap = ((selectedNode.data as Record<string, unknown>)['params'] as Record<string, unknown>) ?? {};
                const value = paramsMap[param.key] ?? param.default;
                return (
                  <ParamRow
                    key={param.key}
                    param={param}
                    value={value}
                    accent={accent}
                    onChange={(v) => handleChange(param.key, v)}
                  />
                );
              })
            )}

            {/* Tensor shapes */}
            {(selectedNode.data as Record<string, unknown>)["inputShape"] && (
              <>
                <div style={{ height: 1, background: "#1e293b", margin: "6px 0 14px" }} />
                <label
                  style={{
                    display: "block", fontSize: 10, fontWeight: 600,
                    color: "#334155", letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    fontFamily: "'JetBrains Mono', monospace", marginBottom: 7,
                  }}
                >
                  {t('synapse.propertyPanel.tensorShapeLabel')}
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                  <span style={{ color: "#38bdf8" }}>
                    {String((selectedNode.data as Record<string, unknown>)["inputShape"])}
                  </span>
                  <span style={{ color: "#334155" }}>→</span>
                  <span style={{ color: "#34d399" }}>
                    {String((selectedNode.data as Record<string, unknown>)["outputShape"] ?? "?")}
                  </span>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PropertyPanel;
