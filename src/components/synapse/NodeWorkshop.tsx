/**
 * Node-Werkstatt: eigenen Node anlegen oder aendern.
 * Links Aussehen, Mitte Code (bei Bloecken der Inhalt), rechts die Vorschau mit
 * den Ports und Einstellungen, die der Formen-Dienst aus dem Code liest.
 */

import React, { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Check, X } from "lucide-react";
import ModalPortal from "../ui/ModalPortal";
import { useLanguage } from "../../contexts/LanguageContext";
import {
  CUSTOM_COLORS, CUSTOM_ICONS, DEFAULT_CODE, customNodeApi, paramsFromDescription, portsFromDescription,
  type CodeDescription, type CustomNodeInput,
} from "./customNodes";

interface Props {
  draft: CustomNodeInput;
  /** Aus einer .ftnode-Datei: Code stammt nicht vom Nutzer selbst. */
  foreign?: boolean;
  onSave: (input: CustomNodeInput) => Promise<void>;
  onClose: () => void;
}

const mono = "'JetBrains Mono', 'Fira Code', monospace";
const field: React.CSSProperties = {
  width: "100%", padding: "6px 9px", background: "#111827", border: "1px solid #1e293b", borderRadius: 5,
  color: "#e2e8f0", fontSize: 12, fontFamily: mono, outline: "none", boxSizing: "border-box",
};
const label: React.CSSProperties = {
  display: "block", fontSize: 10, fontWeight: 600, color: "#64748b", letterSpacing: "0.08em",
  textTransform: "uppercase", fontFamily: mono, margin: "12px 0 5px",
};

function parseShape(text: string): number[] | null {
  const dims = text.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  return dims.length >= 1 && dims.every((d) => Number.isInteger(d) && d > 0) ? dims : null;
}

export const NodeWorkshop: React.FC<Props> = ({ draft, foreign, onSave, onClose }) => {
  const { t } = useLanguage();
  const isCode = draft.kind === "code";
  const [name, setName] = useState(draft.name);
  const [icon, setIcon] = useState(draft.definition.icon || (isCode ? "sparkles" : "box"));
  const [color, setColor] = useState(draft.definition.color || CUSTOM_COLORS[0]);
  const [description, setDescription] = useState(draft.definition.description ?? "");
  const [code, setCode] = useState(draft.definition.code ?? DEFAULT_CODE);
  const [shapeText, setShapeText] = useState("1, 64");
  const [desc, setDesc] = useState<CodeDescription | null>(null);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  const shape = parseShape(shapeText);
  useEffect(() => {
    if (!isCode || !shape) return;
    const id = ++seq.current;
    setChecking(true);
    const timer = setTimeout(() => {
      customNodeApi.describe(code, draft.definition.values ?? {}, shape)
        .then((d) => { if (id === seq.current) setDesc(d); })
        .catch((e) => { if (id === seq.current) setDesc({ ok: false, error: String(e) }); })
        .finally(() => { if (id === seq.current) setChecking(false); });
    }, 450);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code, shapeText, isCode]);

  const ports = useMemo(() => (desc?.ok ? portsFromDescription(desc) : null), [desc]);
  const params = useMemo(() => (desc?.ok ? paramsFromDescription(desc) : []), [desc]);
  const blockSize = !isCode ? ((draft.definition.graph as any)?.nodes?.length ?? 0) : 0;
  const Icon = CUSTOM_ICONS[icon] ?? CUSTOM_ICONS.sparkles;

  const canSave = name.trim().length > 0 && (!isCode || Boolean(desc?.ok)) && !saving;
  const save = async () => {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    const look = { icon, color, description: description.trim() };
    const definition = isCode
      ? {
          ...look, code, inputs: ports!.inputs, outputs: ports!.outputs, params,
          values: Object.fromEntries(params.map((p) => [p.key, draft.definition.values?.[p.key] ?? p.default])),
        }
      : { ...look, graph: draft.definition.graph, repeat: draft.definition.repeat ?? 1 };
    try {
      await onSave({ id: draft.id ?? null, name: name.trim(), kind: draft.kind, definition });
    } catch (e) {
      setError(String(e));
      setSaving(false);
    }
  };

  return (
    <ModalPortal>
      <div
        className="fixed inset-0 z-[9999] flex items-center justify-center"
        style={{ background: "rgba(0,0,0,0.7)" }}
        onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
      >
        <div
          role="dialog"
          aria-label={t("synapseBuilder.custom.workshopTitle")}
          style={{
            width: "min(980px, 94vw)", maxHeight: "90vh", display: "flex", flexDirection: "column",
            background: "#0d1117", border: "1px solid #1e293b", borderRadius: 12, overflow: "hidden",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", padding: "12px 16px", borderBottom: "1px solid #1e293b" }}>
            <div style={{ flex: 1, fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>
              {draft.id ? t("synapseBuilder.custom.editTitle") : t("synapseBuilder.custom.workshopTitle")}
            </div>
            <button onClick={onClose} aria-label={t("common.close", "Close")}
              style={{ background: "transparent", border: "none", color: "#64748b", cursor: "pointer" }}>
              <X size={16} />
            </button>
          </div>

          {foreign && (
            <div style={{ display: "flex", gap: 8, alignItems: "flex-start", margin: "12px 16px 0", padding: "8px 10px",
              background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 6,
              fontSize: 11.5, color: "#fbbf24", lineHeight: 1.5 }}>
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
              {t("synapseBuilder.custom.foreignWarning")}
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "190px minmax(0,1fr) 220px", gap: 16, padding: 16,
            overflow: "auto", minHeight: 0 }}>
            {/* Aussehen */}
            <div>
              <label style={{ ...label, marginTop: 0 }}>{t("synapseBuilder.custom.name")}</label>
              <input value={name} onChange={(e) => setName(e.target.value)} style={field} autoFocus
                placeholder={isCode ? "Residual MLP" : "Encoder"} />
              <label style={label}>{t("synapseBuilder.custom.icon")}</label>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 4 }}>
                {Object.entries(CUSTOM_ICONS).map(([key, I]) => (
                  <button key={key} type="button" onClick={() => setIcon(key)} aria-label={key} title={key}
                    style={{ height: 28, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer",
                      background: icon === key ? `${color}22` : "transparent", borderRadius: 5,
                      border: `1px solid ${icon === key ? color : "#1e293b"}`, color: icon === key ? color : "#64748b" }}>
                    <I size={14} />
                  </button>
                ))}
              </div>
              <label style={label}>{t("synapseBuilder.custom.color")}</label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {CUSTOM_COLORS.map((c) => (
                  <button key={c} type="button" onClick={() => setColor(c)} aria-label={c}
                    style={{ width: 20, height: 20, borderRadius: "50%", background: c, cursor: "pointer",
                      border: color === c ? "2px solid #e2e8f0" : "2px solid transparent" }} />
                ))}
              </div>
              <label style={label}>{t("synapseBuilder.custom.description")}</label>
              <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3}
                style={{ ...field, resize: "vertical", lineHeight: 1.5 }} />
            </div>

            {/* Inhalt */}
            <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
              {isCode ? (
                <>
                  <label style={{ ...label, marginTop: 0 }}>Python</label>
                  <textarea value={code} onChange={(e) => setCode(e.target.value)} spellCheck={false}
                    aria-label="Python"
                    style={{ ...field, flex: 1, minHeight: 340, resize: "vertical", lineHeight: 1.55, tabSize: 4 }}
                    onKeyDown={(e) => {
                      if (e.key !== "Tab") return;
                      e.preventDefault();
                      const el = e.currentTarget;
                      const { selectionStart: s, selectionEnd: end } = el;
                      setCode(code.slice(0, s) + "    " + code.slice(end));
                      requestAnimationFrame(() => { el.selectionStart = el.selectionEnd = s + 4; });
                    }} />
                  <div style={{ fontSize: 11, color: "#64748b", marginTop: 8, lineHeight: 1.6 }}>
                    {t("synapseBuilder.custom.codeHint")}
                  </div>
                </>
              ) : (
                <>
                  <label style={{ ...label, marginTop: 0 }}>{t("synapseBuilder.custom.blockContent")}</label>
                  <div style={{ ...field, lineHeight: 1.7, color: "#94a3b8" }}>
                    {t("synapseBuilder.custom.blockSummary", { count: blockSize, repeat: draft.definition.repeat ?? 1 })}
                    <div style={{ marginTop: 6 }}>
                      {((draft.definition.graph as any)?.nodes ?? []).map((n: any) => (
                        <div key={n.id} style={{ color: "#cbd5e1" }}>{n.type}</div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Vorschau */}
            <div>
              <label style={{ ...label, marginTop: 0 }}>{t("synapseBuilder.custom.preview")}</label>
              <div style={{ border: `1.5px solid ${color}`, borderRadius: 10, overflow: "hidden", background: "rgba(12,12,22,0.97)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 7, padding: "8px 10px",
                  background: `${color}1f`, borderBottom: `1px solid ${color}59` }}>
                  <Icon size={14} color={color} />
                  <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: "#f1f5f9", overflow: "hidden",
                    textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name || "…"}</span>
                  <span style={{ fontSize: 9, color, border: `1px solid ${color}55`, borderRadius: 4, padding: "1px 5px" }}>
                    {t("synapseBuilder.custom.badge")}
                  </span>
                </div>
                <div style={{ padding: "6px 0", fontSize: 11, color: "#94a3b8" }}>
                  {(isCode ? ports?.inputs ?? [] : [{ id: "in", label: "Input" }]).map((p) => (
                    <div key={p.id} style={{ padding: "2px 12px" }}>● {p.label}</div>
                  ))}
                  {params.length > 0 && (
                    <div style={{ margin: "6px 10px", padding: "4px 8px", background: "rgba(255,255,255,0.035)", borderRadius: 6 }}>
                      {params.map((p) => (
                        <div key={p.key} style={{ display: "flex", justifyContent: "space-between" }}>
                          <span>{p.key}</span><span style={{ color }}>{String(p.default)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {(isCode ? ports?.outputs ?? [] : [{ id: "out", label: "Output" }]).map((p) => (
                    <div key={p.id} style={{ padding: "2px 12px", textAlign: "right" }}>{p.label} ●</div>
                  ))}
                </div>
              </div>

              {isCode && (
                <>
                  <label style={label}>{t("synapseBuilder.custom.probeShape")}</label>
                  <input value={shapeText} onChange={(e) => setShapeText(e.target.value)} style={field}
                    aria-label={t("synapseBuilder.custom.probeShape")} />
                  <div role="status" style={{ fontSize: 11.5, marginTop: 8, lineHeight: 1.5, fontFamily: mono, wordBreak: "break-word" }}>
                    {!shape ? (
                      <span style={{ color: "#f87171" }}>{t("synapseBuilder.custom.badShape")}</span>
                    ) : checking && !desc ? (
                      <span style={{ color: "#64748b" }}>{t("synapseBuilder.custom.checking")}</span>
                    ) : desc && !desc.ok ? (
                      <span style={{ color: "#f87171" }}>{desc.error}</span>
                    ) : desc?.probeError ? (
                      <span style={{ color: "#f87171" }}>{desc.probeError}</span>
                    ) : desc?.probe ? (
                      <span style={{ color: "#4ade80" }}>
                        <Check size={12} style={{ verticalAlign: -2 }} />{" "}
                        {Object.entries(desc.probe).map(([k, v]) => `${k}: ${v.join(", ")}`).join(" · ")}
                        <span style={{ color: "#64748b" }}> · {t("synapseBuilder.custom.paramCount", { count: desc.parameters ?? 0 })}</span>
                      </span>
                    ) : null}
                  </div>
                </>
              )}
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", borderTop: "1px solid #1e293b" }}>
            <div style={{ flex: 1, fontSize: 11.5, color: "#f87171" }}>{error}</div>
            <button onClick={onClose} style={{ padding: "6px 14px", background: "transparent", color: "#94a3b8",
              border: "1px solid #1e293b", borderRadius: 6, cursor: "pointer", fontSize: 12 }}>
              {t("common.cancel", "Cancel")}
            </button>
            <button onClick={save} disabled={!canSave}
              title={!name.trim() ? t("synapseBuilder.custom.needName") : undefined}
              style={{ padding: "6px 14px", background: canSave ? color : "#1e293b", color: canSave ? "#0b0b14" : "#475569",
                border: "none", borderRadius: 6, cursor: canSave ? "pointer" : "default", fontSize: 12, fontWeight: 600 }}>
              {draft.id ? t("synapseBuilder.custom.saveVersion") : t("synapseBuilder.custom.saveToLibrary")}
            </button>
          </div>
        </div>
      </div>
    </ModalPortal>
  );
};

export default NodeWorkshop;
