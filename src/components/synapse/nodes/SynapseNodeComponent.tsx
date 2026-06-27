import { memo, CSSProperties } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import {
  CATEGORY_META,
  LAYOUT,
  calcInputHandleTop,
  calcOutputHandleTop,
  NodeDef,
} from '../nodeTypes';

// ─────────────────────────────────────────────────────────────────────────────
// Data shape that lives inside each React-Flow node's `data` field
// ─────────────────────────────────────────────────────────────────────────────
export interface SynapseNodeData {
  label: string;
  nodeType: string;
  category: string;
  icon: string;
  color: string;
  inputs: NodeDef['inputs'];
  outputs: NodeDef['outputs'];
  paramDefs: NodeDef['params'];
  params: Record<string, any>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared handle style factory
// ─────────────────────────────────────────────────────────────────────────────
function handleStyle(color: string, top: number): CSSProperties {
  return {
    width: 11,
    height: 11,
    background: color,
    border: '2px solid rgba(0,0,0,0.6)',
    borderRadius: '50%',
    top,
    cursor: 'crosshair',
    boxShadow: `0 0 6px ${color}88`,
    transition: 'box-shadow 0.15s ease',
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main node component
// ─────────────────────────────────────────────────────────────────────────────
export const SynapseNodeComponent = memo(function SynapseNodeComponent({
  data,
  selected,
}: NodeProps) {
  const d = data as unknown as SynapseNodeData & {
    _sparkle?: number;
    _shapeErrorRole?: "source" | "target" | "both";
  };
  const meta = CATEGORY_META[d.category as keyof typeof CATEGORY_META];
  if (!meta) return null;

  const sparkleActive = Boolean(d._sparkle && Date.now() - d._sparkle < 1500);
  const shapeRole = d._shapeErrorRole;
  const shapeClass = shapeRole
    ? `synapse-node-shape-error-${shapeRole}`
    : undefined;

  const numInputs       = d.inputs.length;
  const numOutputs      = d.outputs.length;
  const previewParams   = d.paramDefs.slice(0, LAYOUT.MAX_PREVIEW_PARAMS);
  const numPreviewParams = previewParams.length;

  // ── Calculate body section heights ────────────────────────────────────────
  const inputsH  = numInputs  * LAYOUT.ROW_H;
  const outputsH = numOutputs * LAYOUT.ROW_H;
  const paramsH  = numPreviewParams > 0
    ? LAYOUT.GAP + numPreviewParams * LAYOUT.PARAM_H + LAYOUT.GAP
    : 0;
  const inputOutputGap = (numInputs > 0 && numOutputs > 0) ? LAYOUT.GAP : 0;

  const totalBodyH = LAYOUT.BODY_PAD_T + inputsH + paramsH + inputOutputGap + outputsH + LAYOUT.BODY_PAD_B;
  const totalH     = LAYOUT.HEADER_H + totalBodyH;

  // ── Node outer container style ─────────────────────────────────────────────
  const shapeBorder =
    shapeRole === "both"
      ? "#dc2626"
      : shapeRole === "target"
        ? "#ef4444"
        : shapeRole === "source"
          ? "#f97316"
          : null;

  const containerStyle: CSSProperties = {
    background: 'rgba(12, 12, 22, 0.97)',
    border: `1.5px solid ${shapeBorder ?? (selected ? meta.color : meta.border)}`,
    borderRadius: 12,
    minWidth: 195,
    maxWidth: 230,
    height: totalH,
    boxShadow: selected
      ? `0 0 0 2px ${meta.color}55, 0 8px 32px rgba(0,0,0,0.6), 0 0 24px ${meta.glow}`
      : '0 4px 24px rgba(0,0,0,0.55)',
    transition: 'border-color 0.18s ease, box-shadow 0.18s ease',
    overflow: 'hidden',
    position: 'relative',
    fontFamily: '"Inter", system-ui, sans-serif',
  };

  // ── Header style ───────────────────────────────────────────────────────────
  const headerStyle: CSSProperties = {
    background: `linear-gradient(135deg, ${meta.color}22 0%, ${meta.color}0a 100%)`,
    borderBottom: `1px solid ${meta.border}`,
    height: LAYOUT.HEADER_H,
    padding: '0 12px',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexShrink: 0,
  };

  // ── Body container ─────────────────────────────────────────────────────────
  const bodyStyle: CSSProperties = {
    padding: `${LAYOUT.BODY_PAD_T}px 0 ${LAYOUT.BODY_PAD_B}px 0`,
  };

  // ── Shared row styles ──────────────────────────────────────────────────────
  const rowStyle = (align: 'left' | 'right'): CSSProperties => ({
    height: LAYOUT.ROW_H,
    display: 'flex',
    alignItems: 'center',
    justifyContent: align === 'right' ? 'flex-end' : 'flex-start',
    padding: align === 'left' ? '0 12px 0 20px' : '0 20px 0 12px',
    fontSize: 10.5,
    color: '#9ca3af',
    gap: 6,
    userSelect: 'none',
  });

  return (
    <div
      className={[sparkleActive ? 'synapse-node-sparkle' : '', shapeClass].filter(Boolean).join(' ') || undefined}
      style={containerStyle}
    >
      {/* ── Input handles (absolute, left edge) ─────────────────────────── */}
      {d.inputs.map((inp, i) => (
        <Handle
          key={inp.id}
          type="target"
          position={Position.Left}
          id={inp.id}
          style={handleStyle(meta.color, calcInputHandleTop(i))}
          title={`${inp.label} (${inp.portType})`}
        />
      ))}

      {/* ── Output handles (absolute, right edge) ────────────────────────── */}
      {d.outputs.map((out, i) => (
        <Handle
          key={out.id}
          type="source"
          position={Position.Right}
          id={out.id}
          style={handleStyle(meta.color, calcOutputHandleTop(i, numInputs, numPreviewParams))}
          title={`${out.label} (${out.portType})`}
        />
      ))}

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div style={headerStyle}>
        <span style={{ fontSize: 15, lineHeight: 1, flexShrink: 0 }}>{d.icon}</span>
        <span style={{
          color: '#f1f5f9',
          fontSize: 12,
          fontWeight: 600,
          flex: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          letterSpacing: '0.01em',
        }}>
          {d.label}
        </span>
        <span style={{
          background: `${meta.color}28`,
          color: meta.color,
          fontSize: 8.5,
          fontWeight: 700,
          padding: '2px 7px',
          borderRadius: 5,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          flexShrink: 0,
          border: `1px solid ${meta.color}44`,
        }}>
          {meta.label}
        </span>
      </div>

      {/* ── Body ──────────────────────────────────────────────────────────── */}
      <div style={bodyStyle}>

        {/* Input port labels */}
        {d.inputs.map((inp, _i) => (
          <div key={inp.id} style={rowStyle('left')}>
            <span style={{
              width: 5, height: 5,
              borderRadius: '50%',
              background: meta.color,
              flexShrink: 0,
              opacity: 0.7,
            }} />
            <span style={{ color: '#6b7280', fontSize: 10 }}>{inp.label}</span>
          </div>
        ))}

        {/* Params preview section */}
        {numPreviewParams > 0 && (
          <div style={{
            margin: `${LAYOUT.GAP}px 10px ${LAYOUT.GAP}px 10px`,
            padding: '5px 8px',
            background: 'rgba(255,255,255,0.035)',
            borderRadius: 7,
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
            {previewParams.map(p => (
              <div key={p.key} style={{
                height: LAYOUT.PARAM_H,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: 10,
              }}>
                <span style={{ color: '#52525b', fontWeight: 500 }}>{p.label}</span>
                <span style={{
                  color: meta.color,
                  fontWeight: 700,
                  fontSize: 10.5,
                  maxWidth: 80,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {String(d.params[p.key] ?? p.default)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Gap between inputs+params and outputs */}
        {numInputs > 0 && numOutputs > 0 && numPreviewParams === 0 && (
          <div style={{ height: LAYOUT.GAP }} />
        )}

        {/* Output port labels */}
        {d.outputs.map((out, _i) => (
          <div key={out.id} style={rowStyle('right')}>
            <span style={{ color: '#6b7280', fontSize: 10 }}>{out.label}</span>
            <span style={{
              width: 5, height: 5,
              borderRadius: '50%',
              background: meta.color,
              flexShrink: 0,
              opacity: 0.7,
            }} />
          </div>
        ))}
      </div>

      {/* ── Selected indicator bar ───────────────────────────────────────── */}
      {selected && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: 2,
          background: `linear-gradient(90deg, transparent, ${meta.color}, transparent)`,
          borderRadius: '12px 12px 0 0',
        }} />
      )}
    </div>
  );
});
