import React from "react";
import { useLanguage } from "../../contexts/LanguageContext";
import type { AffectedNodeInfo } from "./ai/synapseShapeDiagnostics";

interface ShapeErrorBannerProps {
  title: string;
  affected: AffectedNodeInfo[];
  onDismiss: () => void;
  onOpenAI: () => void;
  onFocusNodes: () => void;
}

export const ShapeErrorBanner: React.FC<ShapeErrorBannerProps> = ({
  title,
  affected,
  onDismiss,
  onOpenAI,
  onFocusNodes,
}) => {
  const { t } = useLanguage();
  return (
  <div className="synapse-shape-banner">
    <div className="synapse-shape-banner-header">
      <span className="synapse-shape-banner-icon">⚠</span>
      <span className="synapse-shape-banner-title">{title}</span>
      <button type="button" className="synapse-shape-banner-close" onClick={onDismiss} title={t('synapse.shapeErrorBanner.closeTooltip')}>
        ✕
      </button>
    </div>
    {affected.length > 0 && (
      <div className="synapse-shape-banner-nodes">
        {affected.map((a) => (
          <span
            key={a.id}
            className={`synapse-shape-node-chip role-${a.role}`}
            title={`${a.id} (${a.type})`}
          >
            {a.role === "source" && "▶ "}
            {a.role === "target" && "◀ "}
            {a.role === "both" && "◆ "}
            {a.label}
            <span className="synapse-shape-node-id">{a.id}</span>
          </span>
        ))}
      </div>
    )}
    <div className="synapse-shape-banner-actions">
      <button type="button" className="synapse-shape-btn primary" onClick={onOpenAI}>
        {t('synapse.shapeErrorBanner.fixWithAIButton')}
      </button>
      <button type="button" className="synapse-shape-btn" onClick={onFocusNodes}>
        {t('synapse.shapeErrorBanner.focusNodesButton')}
      </button>
    </div>
  </div>
  );
};

export default ShapeErrorBanner;
