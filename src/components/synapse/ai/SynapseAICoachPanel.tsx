/**
 * SynapseAICoachPanel - AI Training-Debugger
 *
 * Öffnet sich nach fehlgeschlagenem Training. Jede Frage (Quick Question oder
 * frei eingetippt) geht mit vollem Fehler- und Graph-Kontext an den
 * konfigurierten AI-Provider. Die regelbasierten Antworten aus SynapseAICoach
 * dienen als Sofort-Fallback (AI deaktiviert / Call fehlgeschlagen) — es gibt
 * in keinem Pfad ein "Ich konnte deine Frage nicht verstehen".
 */

import React, { useEffect, useRef, useState } from "react";
import { useLanguage } from "../../../contexts/LanguageContext";
import { useAISettings } from "../../../contexts/AISettingsContext";
import { callAI } from "../../../ai/aiClient";
import type { ChatMessage } from "../../../ai/aiClient";
import GradientChatInput from "../../ui/GradientChatInput";
import { buildSynapseGraphContext } from "./synapseGraphContext";
import { NODE_DEFINITIONS } from "../nodeTypes";
import {
  SynapseAICoach,
  TrainingResult,
  TrainingAnalyzer,
  GraphAnalyzer,
  FixSuggestion,
} from "./SynapseAICoach";
import { Node, Edge } from "@xyflow/react";
import { Bot } from "lucide-react";

interface SynapseAICoachPanelProps {
  trainingResult: TrainingResult | null;
  nodes: Node[];
  edges: Edge[];
  layerConfig: any[];
  onApplyFix?: (fix: FixSuggestion) => void;
  onClose?: () => void;
}

export const SynapseAICoachPanel: React.FC<SynapseAICoachPanelProps> = ({
  trainingResult,
  nodes,
  edges,
  layerConfig,
  onApplyFix,
  onClose,
}) => {
  const { t, language } = useLanguage();
  const { settings: aiSettings } = useAISettings();
  const [coach] = useState(() => new SynapseAICoach());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [retryQuestion, setRetryQuestion] = useState<string | null>(null);
  const [fixes, setFixes] = useState<FixSuggestion[]>([]);
  const [appliedFixes, setAppliedFixes] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (!trainingResult) {
    return null;
  }

  // Analysiere Training-Ergebnis
  const analysis = TrainingAnalyzer.analyzeResult(trainingResult);
  // Die Struktur-Diagnose ist eine Shape-Heuristik — bei Nicht-Shape-Fehlern
  // (Dataset, Speicher, Pakete …) würde sie irreführende "Fehler" anzeigen.
  const isShapeRelated = /shape|dimension|mismatch|mat1|mat2/i.test(trainingResult.error ?? "");
  const isDatasetError = /image_loader|csv_loader|parquet_loader|found no valid file|dataset/i.test(trainingResult.error ?? "");
  const diagnosis = isShapeRelated
    ? GraphAnalyzer.analyzeGraph(nodes, edges, layerConfig)
    : { isHealthy: true, issues: [], warnings: [], suggestions: [] };

  coach.setLastTrainingResult(trainingResult);
  coach.analyzeDiagnosis(diagnosis);

  // ── System-Prompt: Fehler + Graph + FrameTrain-Wissen ────────────────────
  const buildCoachSystem = (): string => {
    const errText = (trainingResult.error ?? "").slice(0, 1600);
    const graphCtx = buildSynapseGraphContext(nodes, edges, null, NODE_DEFINITIONS).slice(0, 2500);
    const issueLines = diagnosis.issues.map((i) => `- ${i.description}`).join("\n");

    return `Du bist der "Synapse AI Coach" in FrameTrain, einer Desktop-App mit visuellem Neural-Network-Builder (Canvas).
Der letzte Trainingslauf ist fehlgeschlagen. Beantworte JEDE Frage des Users hilfreich, kurz (max. ~10 Sätze) und mit konkreten, umsetzbaren Schritten in der App. Weiche nie aus und lehne nie ab.

## Trainingsstatus
${analysis.message}

## Fehlertext (Runtime)
${errText || "(kein Fehlertext)"}
${issueLines ? `\n## Erkannte Graph-Probleme\n${issueLines}` : ""}

## Canvas-Graph (Ground Truth)
${graphCtx}

## FrameTrain-Wissen (nutzen wenn relevant)
- Datasets wählt man unten in der Synapse-Trainingsleiste aus.
- image_loader = Bild-KLASSIFIKATION: braucht einen Ordner pro Klasse (train/<klasse>/*.jpg, optional val/<klasse>/).
- YOLO-Datasets (images/ + labels/ + dataset.yaml) sind OBJEKTERKENNUNG → dafür das YOLO-Training im Training-Panel nutzen, nicht den Canvas.
- Parquet-/CSV-Datasets → parquet_loader- bzw. csv_loader-Node im Canvas verwenden.
- Shape-Fehler: Im Fehler-Banner "Mit AI beheben" nutzen — die Synapse-AI kann Nodes und Parameter direkt ändern.
- Speicher-Fehler (OOM): Batch-Size halbieren, Bildgröße reduzieren, andere Apps schließen.
- Fehler ans Team melden: Button "An FrameTrain senden" im Fehler-Dialog.`;
  };

  // ── Frage stellen: AI zuerst, regelbasierte Antwort als Fallback ─────────
  const askCoach = async (question: string, isRetry = false) => {
    const text = question.trim();
    if (!text || loading) return;

    // Regelbasierte Antwort immer vorbereiten (Fallback + Fix-Vorschläge)
    const ruleBased = coach.respondToQuestion(text);
    if (ruleBased.suggestFixes && ruleBased.suggestFixes.length > 0) {
      setFixes(ruleBased.suggestFixes);
    }
    setRetryQuestion(null);

    // Retry: Fallback-Antwort entfernen — die User-Frage ist schon im Verlauf
    const base = isRetry && messages[messages.length - 1]?.role === "assistant"
      ? messages.slice(0, -1)
      : messages;
    const withUser = isRetry ? base : [...base, { role: "user" as const, content: text }];
    setMessages(withUser);
    setInput("");

    if (!aiSettings.enabled) {
      setMessages([...withUser, { role: "assistant", content: ruleBased.answer }]);
      return;
    }

    setLoading(true);
    try {
      const answer = await callAI(aiSettings, {
        system: buildCoachSystem(),
        messages: withUser.slice(-8),
        maxTokens: 700,
        temperature: 0.4,
        responseLanguage: language,
      });
      setMessages([...withUser, { role: "assistant", content: answer.trim() || ruleBased.answer }]);
    } catch {
      // AI nicht erreichbar → regelbasierte Antwort + Retry anbieten
      setMessages([
        ...withUser,
        { role: "assistant", content: `${ruleBased.answer}\n\n${t('synapseAI.coach.aiUnavailableNote')}` },
      ]);
      setRetryQuestion(text);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyFix = (fix: FixSuggestion) => {
    if (onApplyFix) {
      onApplyFix(fix);
      setAppliedFixes([...appliedFixes, fix.id]);
    }
  };

  const quickQuestions = [
    ...(isDatasetError ? [t('synapseAI.coach.question4')] : []),
    t('synapseAI.coach.question1'),
    ...(isShapeRelated ? [t('synapseAI.coach.question2')] : []),
    t('synapseAI.coach.question3'),
  ];

  return (
    <div className="w-96 max-h-[540px] bg-gray-900 border border-purple-500/60 shadow-2xl rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-800 p-3 flex justify-between items-center flex-shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-white" />
          <h3 className="font-bold text-white">{t('synapseAI.coach.title')}</h3>
        </div>
        <button
          onClick={onClose}
          className="text-gray-300 hover:text-white text-xl"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {/* Status */}
        <div className={`p-3 rounded text-sm font-mono ${
          analysis.status === "success"
            ? "bg-green-900 text-green-100"
            : "bg-red-900 text-red-100"
        }`}>
          {analysis.message}
        </div>

        {/* Issues Summary */}
        {diagnosis.issues.length > 0 && (
          <div className="bg-red-900/30 border border-red-500 rounded p-3">
            <h4 className="text-red-400 font-bold mb-2">
              ⚠ {diagnosis.issues.length} Fehler gefunden:
            </h4>
            <ul className="text-xs text-red-200 space-y-1">
              {diagnosis.issues.map((issue) => (
                <li key={issue.nodeId}>
                  • {issue.description}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Warnings */}
        {diagnosis.warnings.length > 0 && (
          <div className="bg-yellow-900/30 border border-yellow-500 rounded p-3">
            <h4 className="text-yellow-400 font-bold mb-2">
              ⚠ {diagnosis.warnings.length} Warnungen:
            </h4>
            <ul className="text-xs text-yellow-200 space-y-1">
              {diagnosis.warnings.map((warn, idx) => (
                <li key={idx}>• {warn}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Quick Questions — passend zum Fehlertyp */}
        {analysis.status !== "success" && messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-gray-400 font-bold">{t('synapseAI.coach.quickQuestionsLabel')}</p>
            {quickQuestions.map((q) => (
              <button
                key={q}
                onClick={() => askCoach(q)}
                disabled={loading}
                className="w-full text-left p-2 text-xs rounded bg-gray-800 hover:bg-purple-900 transition disabled:opacity-50"
              >
                › {q}
              </button>
            ))}
          </div>
        )}

        {/* Chat-Verlauf */}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-2.5 rounded-lg text-xs whitespace-pre-wrap leading-relaxed ${
              m.role === "user"
                ? "bg-purple-900/40 border border-purple-500/30 text-purple-100 ml-6"
                : "bg-gray-800 border border-gray-700 text-gray-100 mr-2"
            }`}
          >
            {m.content}
          </div>
        ))}

        {/* AI denkt nach */}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-purple-300 p-2">
            <span className="animate-pulse">✦</span>
            <span>{t('synapseAI.panel.thinkingLabel')}…</span>
          </div>
        )}

        {/* Retry nach AI-Fehler */}
        {retryQuestion && !loading && (
          <button
            onClick={() => askCoach(retryQuestion, true)}
            className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-200 text-xs font-medium transition-all"
          >
            {t('aiCoach.retryButton')}
          </button>
        )}

        {/* Fix-Vorschläge aus der Diagnose */}
        {fixes.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-purple-400 font-bold">
              {t('synapseAI.coach.suggestedActionsLabel')}
            </p>
            {fixes.map((fix) => (
              <div
                key={fix.id}
                className={`bg-gray-700/50 border-l-2 p-2 rounded text-xs ${
                  fix.action === "inspect_only" ? "border-yellow-400" : "border-purple-400"
                } ${appliedFixes.includes(fix.id) ? "opacity-50" : ""}`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className={`font-bold ${
                      fix.action === "inspect_only" ? "text-yellow-300" : "text-purple-300"
                    }`}>
                      {fix.title}
                    </p>
                    <p className="text-gray-300 text-xs mt-1">
                      {fix.description}
                    </p>
                    {fix.action === "inspect_only" && (
                      <p className="text-yellow-200 text-xs mt-2">
                        {t('synapseAI.coach.inspectOnlyNote')}
                      </p>
                    )}
                  </div>
                  {fix.action !== "inspect_only" && !appliedFixes.includes(fix.id) && (
                    <button
                      onClick={() => handleApplyFix(fix)}
                      className="text-xs bg-purple-600 hover:bg-purple-500 px-2 py-1 rounded whitespace-nowrap ml-2"
                    >
                      {t('synapseAI.coach.applyButton')}
                    </button>
                  )}
                  {appliedFixes.includes(fix.id) && (
                    <span className="text-xs text-green-400 ml-2">✓</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Eingabe: jede freie Frage ist erlaubt */}
      <div className="p-3 border-t border-gray-700 flex-shrink-0">
        <GradientChatInput
          value={input}
          onChange={setInput}
          onSend={() => askCoach(input)}
          loading={loading}
          size="sm"
          placeholder={t('synapseAI.coach.inputPlaceholder')}
        />
      </div>

      {/* Footer */}
      <div className="bg-gray-800 border-t border-gray-700 px-3 py-2 text-[10px] text-gray-400 flex-shrink-0">
        {t('synapseAI.coach.footerHint')}
      </div>
    </div>
  );
};
