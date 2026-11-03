import React, { useMemo, useState } from "react";
import { useLanguage } from "../../contexts/LanguageContext";
import blueprints from "./blueprints.json";

export interface Blueprint {
  id: string;
  goal: string;
  label: string;
  description: string;
  label_en?: string;
  description_en?: string;
  nodes: { id: string; type: string; params: Record<string, unknown> }[];
  edges: (string | undefined)[][];
}

const GOALS = ["all", "image", "text", "audio", "video", "table"] as const;

/** Startauswahl auf der leeren Fläche: Ziel wählen, Vorlage übernehmen. */
export const BlueprintPicker: React.FC<{ onPick: (b: Blueprint) => void }> = ({ onPick }) => {
  const { t, language } = useLanguage();
  const en = language === "en";
  const [goal, setGoal] = useState<(typeof GOALS)[number]>("all");
  const shown = useMemo(
    () => (blueprints as Blueprint[]).filter((b) => goal === "all" || b.goal === goal),
    [goal],
  );
  return (
    <div style={{
      width: 520, maxWidth: "calc(100vw - 48px)", padding: 16, borderRadius: 12,
      background: "#0d1117", border: "1px solid #1e293b",
    }}>
      <div style={{ fontSize: 13, color: "#e6e9f0", marginBottom: 10 }}>{t('synapseBuilder.blueprints.title')}</div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 10 }}>
        {GOALS.map((g) => (
          <button key={g} onClick={() => setGoal(g)} style={{
            padding: "3px 10px", borderRadius: 999, border: "none", cursor: "pointer", fontSize: 11,
            background: goal === g ? "rgba(129,140,248,0.18)" : "transparent",
            color: goal === g ? "#c7d2fe" : "#94a0b8",
          }}>
            {t(`synapseBuilder.blueprints.${g}`)}
          </button>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
        {shown.map((b) => (
          <button key={b.id} onClick={() => onPick(b)} style={{
            padding: "8px 10px", textAlign: "left", cursor: "pointer", borderRadius: 8,
            background: "#111827", border: "1px solid #334155",
          }}>
            <div style={{ fontSize: 12, color: "#e6e9f0" }}>{en ? b.label_en ?? b.label : b.label}</div>
            <div style={{ fontSize: 11, color: "#7a869e" }}>{en ? b.description_en ?? b.description : b.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
};
