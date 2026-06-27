/**
 * SynapseAICoachPanel - UI für AI Training-Debugger
 * 
 * Wird nach jedem Training angezeigt und hilft Fehler zu beheben
 */

import React, { useState } from "react";
import { useLanguage } from "../../../contexts/LanguageContext";
import {
  SynapseAICoach,
  TrainingResult,
  TrainingAnalyzer,
  GraphAnalyzer,
  GraphAutoFixer,
  FixSuggestion,
} from "./SynapseAICoach";
import { Node, Edge } from "@xyflow/react";

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
  const { t } = useLanguage();
  const [coach] = useState(() => new SynapseAICoach());
  const [selectedQuestion, setSelectedQuestion] = useState<string>("");
  const [response, setResponse] = useState<any>(null);
  const [appliedFixes, setAppliedFixes] = useState<string[]>([]);

  if (!trainingResult) {
    return null;
  }

  // Analysiere Training-Ergebnis
  const analysis = TrainingAnalyzer.analyzeResult(trainingResult);
  const diagnosis = GraphAnalyzer.analyzeGraph(nodes, edges, layerConfig);

  coach.setLastTrainingResult(trainingResult);
  coach.analyzeDiagnosis(diagnosis);

  const handleQuestion = (question: string) => {
    setSelectedQuestion(question);
    const resp = coach.respondToQuestion(question);
    setResponse(resp);
  };

  const handleApplyFix = (fix: FixSuggestion) => {
    if (onApplyFix) {
      onApplyFix(fix);
      setAppliedFixes([...appliedFixes, fix.id]);
    }
  };

  return (
    <div className="fixed bottom-0 right-0 w-96 max-h-96 bg-gray-900 border-l border-t border-purple-500 shadow-2xl rounded-tl-lg overflow-hidden flex flex-col z-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-800 p-3 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
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
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
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
              🚨 {diagnosis.issues.length} Fehler gefunden:
            </h4>
            <ul className="text-xs text-red-200 space-y-1">
              {diagnosis.issues.map((issue) => (
                <li key={issue.nodeId}>
                  • Layer {issue.nodeId}: {issue.description}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Warnings */}
        {diagnosis.warnings.length > 0 && (
          <div className="bg-yellow-900/30 border border-yellow-500 rounded p-3">
            <h4 className="text-yellow-400 font-bold mb-2">
              ⚠️ {diagnosis.warnings.length} Warnungen:
            </h4>
            <ul className="text-xs text-yellow-200 space-y-1">
              {diagnosis.warnings.map((warn, idx) => (
                <li key={idx}>• {warn}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Quick Questions */}
        {analysis.status !== "success" && (
          <div className="space-y-2">
            <p className="text-xs text-gray-400 font-bold">{t('synapseAI.coach.quickQuestionsLabel')}</p>
            {[
              t('synapseAI.coach.question1'),
              t('synapseAI.coach.question2'),
              t('synapseAI.coach.question3'),
            ].map((q) => (
              <button
                key={q}
                onClick={() => handleQuestion(q)}
                className={`w-full text-left p-2 text-xs rounded bg-gray-800 hover:bg-purple-900 transition ${
                  selectedQuestion === q ? "bg-purple-900 border-l-2 border-purple-400" : ""
                }`}
              >
                💬 {q}
              </button>
            ))}
          </div>
        )}

        {/* Response */}
        {response && (
          <div className="bg-gray-800 border border-purple-500 rounded p-3 space-y-3">
            <p className="text-sm text-gray-100 whitespace-pre-wrap">
              {response.answer}
            </p>

            {/* Suggested Fixes */}
            {response.suggestFixes && response.suggestFixes.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-purple-400 font-bold">
                  {t('synapseAI.coach.suggestedActionsLabel')}
                </p>
                {response.suggestFixes.map((fix: FixSuggestion) => (
                  <div
                    key={fix.id}
                    className={`bg-gray-700/50 border-l-2 p-2 rounded text-xs ${
                      fix.action === "inspect_only"
                        ? "border-yellow-400"  // Warnungen = gelb
                        : "border-purple-400"  // Auto-Fixes = lila
                    } ${appliedFixes.includes(fix.id) ? "opacity-50" : ""}`}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <p className={`font-bold ${
                          fix.action === "inspect_only"
                            ? "text-yellow-300"
                            : "text-purple-300"
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
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-gray-800 border-t border-gray-700 p-3 text-xs text-gray-400">
        {t('synapseAI.coach.footerHint')}
      </div>
    </div>
  );
};
