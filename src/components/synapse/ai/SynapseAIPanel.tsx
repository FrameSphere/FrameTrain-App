/**
 * Synapse AI Assistant — floating glass panel with chat history & live steps UI.
 * Design-only layer; AI logic stays in synapseAgent.ts.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useLanguage } from "../../../contexts/LanguageContext";
import type { ChatMessage } from "../../../ai/aiClient";
import type { AgentResumeState } from "./synapseAgent";
import type { AgentStep } from "./synapseAgentTools";
import {
  freshSynapseChat,
  loadSynapseAIChats,
  saveSynapseAIChats,
  type SynapseAIChat,
  type SynapseAIMessage,
} from "./synapseAIStorage";
import type { AffectedNodeInfo } from "./synapseShapeDiagnostics";
import "./synapseAIPanel.css";

function formatChatDate(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit" }) +
    " · " + d.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
}

function toolLabel(tool: string, t: (key: string, fallback?: string) => string): string {
  const map: Record<string, string> = {
    add_node: t('synapseAI.toolLabels.add_node', 'Add Node'),
    add_edge: t('synapseAI.toolLabels.add_edge', 'Connect'),
    remove_node: t('synapseAI.toolLabels.remove_node', 'Remove Node'),
    remove_edge: t('synapseAI.toolLabels.remove_edge', 'Remove Edge'),
    set_param: t('synapseAI.toolLabels.set_param', 'Parameter'),
    set_label: t('synapseAI.toolLabels.set_label', 'Rename'),
    move_node: t('synapseAI.toolLabels.move_node', 'Move'),
    done: t('synapseAI.toolLabels.done', 'Done'),
  };
  return map[tool] ?? tool;
}

export interface SynapseAIPanelProps {
  open: boolean;
  onClose: () => void;
  userId: string;
  messages: ChatMessage[];
  onMessagesChange: (messages: ChatMessage[]) => void;
  input: string;
  onInputChange: (v: string) => void;
  onSend: (msg?: string, resume?: AgentResumeState) => void;
  loading: boolean;
  error: string | null;
  steps: AgentStep[];
  resumeState: AgentResumeState | null;
  onAbort: () => void;
  /** Shape-Fix Modus: Vorlage im Input, Kurzinfo im Panel */
  shapeMode?: boolean;
  shapeUserGuide?: string;
  affectedNodes?: AffectedNodeInfo[];
}

export const SynapseAIPanel: React.FC<SynapseAIPanelProps> = ({
  open,
  onClose,
  userId,
  messages,
  onMessagesChange,
  input,
  onInputChange,
  onSend,
  loading,
  error,
  steps,
  resumeState,
  onAbort,
  shapeMode = false,
  shapeUserGuide,
  affectedNodes = [],
}) => {
  const { t } = useLanguage();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [chats, setChats] = useState<SynapseAIChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const persistChat = useCallback(
    (chatId: string, msgs: ChatMessage[], title?: string) => {
      setChats((prev) => {
        const now = Date.now();
        const existing = prev.find((c) => c.id === chatId);
        const firstUser = msgs.find((m) => m.role === "user");
        const autoTitle =
          title ??
          (firstUser?.content.slice(0, 42).trim() || "Synapse Chat") +
            (firstUser && firstUser.content.length > 42 ? "…" : "");

        const stored: SynapseAIMessage[] = msgs.map((m, i) => ({
          id: `m_${chatId}_${i}`,
          role: m.role as "user" | "assistant",
          content: m.content,
          ts: now,
        }));

        const nextChat: SynapseAIChat = existing
          ? {
              ...existing,
              title: autoTitle,
              messages: stored,
              updatedAt: now,
            }
          : {
              ...freshSynapseChat(),
              id: chatId,
              title: autoTitle,
              messages: stored,
              createdAt: now,
              updatedAt: now,
            };

        const rest = prev.filter((c) => c.id !== chatId);
        const next = [nextChat, ...rest].slice(0, 40);
        saveSynapseAIChats(next, userId);
        return next;
      });
    },
    [userId]
  );

  useEffect(() => {
    if (!open) return;
    const loaded = loadSynapseAIChats(userId);
    setChats(loaded);
    if (loaded.length > 0 && !activeChatId) {
      const c = loaded[0];
      setActiveChatId(c.id);
      onMessagesChange(
        c.messages.map((m) => ({ role: m.role, content: m.content }))
      );
    } else if (loaded.length === 0) {
      const c = freshSynapseChat();
      setActiveChatId(c.id);
      setChats([c]);
      saveSynapseAIChats([c], userId);
    }
  }, [open, userId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!open || !activeChatId) return;
    persistChat(activeChatId, messages);
  }, [messages, activeChatId, open, persistChat]);

  useEffect(() => {
    if (open) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, steps, loading, open]);

  const handleNewChat = useCallback(() => {
    const c = freshSynapseChat();
    setChats((prev) => {
      const next = [c, ...prev];
      saveSynapseAIChats(next, userId);
      return next;
    });
    setActiveChatId(c.id);
    onMessagesChange([]);
    onInputChange("");
  }, [userId, onMessagesChange, onInputChange]);

  const handleSelectChat = useCallback(
    (c: SynapseAIChat) => {
      setActiveChatId(c.id);
      onMessagesChange(c.messages.map((m) => ({ role: m.role, content: m.content })));
      setHistoryOpen(false);
    },
    [onMessagesChange]
  );

  const handleDeleteChat = useCallback(
    (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setChats((prev) => {
        const next = prev.filter((c) => c.id !== id);
        saveSynapseAIChats(next, userId);
        if (activeChatId === id) {
          if (next.length > 0) {
            handleSelectChat(next[0]);
          } else {
            handleNewChat();
          }
        }
        return next;
      });
    },
    [userId, activeChatId, handleSelectChat, handleNewChat]
  );

  if (!open) return null;

  const doneCount = steps.filter((s) => s.status === "success").length;
  const runningStep = steps.find((s) => s.status === "running");

  return (
    <>
      <div className="synapse-ai-canvas-hint">
        {t('synapseAI.panel.canvasHint')}
      </div>

      <div className="synapse-ai-panel-wrap">
        {historyOpen && (
          <div className="synapse-ai-history-float">
            <div style={{ padding: "10px 10px 8px", borderBottom: "1px solid rgba(51,65,85,0.5)" }}>
              <div style={{ fontSize: 10, color: "#64748b", marginBottom: 8, letterSpacing: "0.06em" }}>
                CHAT-VERLAUF
              </div>
              <button
                type="button"
                onClick={handleNewChat}
                style={{
                  width: "100%",
                  padding: "8px",
                  borderRadius: 8,
                  border: "1px dashed rgba(167,139,250,0.35)",
                  background: "rgba(167,139,250,0.06)",
                  color: "#a78bfa",
                  fontSize: 11,
                  cursor: "pointer",
                }}
              >
                {t('aiCoach.newChat')}
              </button>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {chats.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={`synapse-ai-history-item ${c.id === activeChatId ? "active" : ""}`}
                  onClick={() => handleSelectChat(c)}
                >
                  <div style={{ fontWeight: 600, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {c.title}
                  </div>
                  <div style={{ fontSize: 9, color: "#475569" }}>{formatChatDate(c.updatedAt)}</div>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => handleDeleteChat(c.id, e)}
                    onKeyDown={(e) => e.key === "Enter" && handleDeleteChat(c.id, e as unknown as React.MouseEvent)}
                    style={{ float: "right", fontSize: 10, color: "#64748b", marginTop: -14 }}
                    title={t('aiCoach.deleteChat')}
                  >
                    ✕
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="synapse-ai-main">
          <header className="synapse-ai-header">
            <div className="synapse-ai-orb">
              <div className="synapse-ai-orb-core" />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.02em" }}>
                {t('synapseAI.panel.title')}
              </div>
              <div style={{ fontSize: 10, color: "#64748b" }}>
                {shapeMode
                  ? t('synapseAI.panel.subtitleShapeMode')
                  : loading
                    ? runningStep
                      ? `${toolLabel(runningStep.tool, t)}…`
                      : t('synapseAI.panel.stepCount').replace('{done}', String(doneCount)).replace('{total}', String(steps.length))
                    : t('synapseAI.panel.subtitleIdle')}
              </div>
            </div>
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              title={t('aiCoach.chatHistory')}
              style={iconBtnStyle(historyOpen)}
            >
              ☰
            </button>
            <button type="button" onClick={handleNewChat} title={t('aiCoach.newChat')} style={iconBtnStyle(false)}>
              +
            </button>
            <button type="button" onClick={onClose} title={t('trainingDashboard.header.minimizeTooltip')} style={iconBtnStyle(false)}>
              ›
            </button>
          </header>

          <div className="synapse-ai-messages">
            {shapeMode && shapeUserGuide && (
              <>
                <div className="synapse-ai-shape-card">{shapeUserGuide}</div>
                {affectedNodes.length > 0 && (
                  <div style={{ fontSize: 10, color: "#94a3b8", padding: "0 4px" }}>
                    {t('synapseAI.panel.affectedLabel')} {affectedNodes.map((a) => a.id).join(", ")}
                  </div>
                )}
                <button
                  type="button"
                  className="synapse-ai-shape-fix-btn"
                  onClick={() => !loading && onSend()}
                  disabled={loading || !input.trim()}
                >
                  {t('synapseAI.panel.shapeSendButton')}
                </button>
              </>
            )}

            {messages.length === 0 && !loading && !shapeMode && (
              <div style={{ textAlign: "center", padding: "28px 12px", color: "#475569" }}>
                <div style={{ fontSize: 28, marginBottom: 12, opacity: 0.4 }}>✦</div>
                <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.7 }}>
                  {t('synapseAI.panel.emptyHint')}
                </div>
                <div style={{ marginTop: 14, fontSize: 10, color: "#334155" }}>
                  {t('synapseAI.panel.emptyExample')}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={`msg-${i}`} className={`synapse-ai-msg ${m.role}`} style={{ animationDelay: `${i * 0.05}s` }}>
                {m.content}
              </div>
            ))}

            {loading && steps.length > 0 && (
              <div className="synapse-ai-steps">
                <div style={{ fontSize: 9, color: "#64748b", marginBottom: 6, letterSpacing: "0.08em" }}>
                  {t('synapseAI.panel.liveCanvasLabel')}
                </div>
                {steps.map((s, i) => (
                  <div
                    key={s.id}
                    className={`synapse-ai-step-row ${s.status}`}
                    style={{ animationDelay: `${i * 0.06}s` }}
                  >
                    <span className="synapse-ai-step-dot" />
                    <span style={{ flex: 1 }}>{toolLabel(s.tool, t)}</span>
                    {s.status === "success" && <span>✓</span>}
                    {s.status === "error" && <span>✕</span>}
                  </div>
                ))}
              </div>
            )}

            {loading && steps.length === 0 && (
              <div className="synapse-ai-thinking">
                <span>{t('synapseAI.panel.thinkingLabel')}</span>
                <span className="synapse-ai-thinking-dots">
                  <span /><span /><span />
                </span>
              </div>
            )}

            {error && (
              <div
                style={{
                  fontSize: 11,
                  color: "#fca5a5",
                  padding: "10px 12px",
                  background: "rgba(127,29,29,0.2)",
                  borderRadius: 10,
                  border: "1px solid rgba(248,113,113,0.25)",
                }}
              >
                {error}
                {resumeState && (
                  <button
                    type="button"
                    onClick={() => onSend(undefined, resumeState)}
                    style={{
                      display: "block",
                      marginTop: 8,
                      padding: "6px 12px",
                      background: "linear-gradient(135deg, #6366f1, #a78bfa)",
                      border: "none",
                      borderRadius: 6,
                      color: "#fff",
                      fontSize: 11,
                      cursor: "pointer",
                    }}
                  >
                    Fortsetzen
                  </button>
                )}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="synapse-ai-input-area">
            <div className="synapse-ai-input-row">
              <textarea
                className="synapse-ai-textarea"
                value={input}
                onChange={(e) => onInputChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (!loading) onSend();
                  }
                }}
                placeholder={shapeMode ? t('synapseAI.panel.inputPlaceholderShapeMode') : t('synapseAI.panel.inputPlaceholder')}
                disabled={loading}
                rows={2}
              />
              <button
                type="button"
                className={`synapse-ai-send ${loading ? "stop" : ""}`}
                onClick={() => (loading ? onAbort() : onSend())}
                title={loading ? t('synapseAI.panel.abortTooltip') : t('synapseAI.panel.sendTooltip')}
              >
                {loading ? t('synapseAI.panel.abortButton') : t('synapseAI.panel.sendButton')}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

function iconBtnStyle(active: boolean): React.CSSProperties {
  return {
    width: 28,
    height: 28,
    borderRadius: 8,
    border: `1px solid ${active ? "rgba(167,139,250,0.4)" : "rgba(51,65,85,0.6)"}`,
    background: active ? "rgba(167,139,250,0.12)" : "transparent",
    color: active ? "#a78bfa" : "#64748b",
    fontSize: 12,
    cursor: "pointer",
    flexShrink: 0,
  };
}

export default SynapseAIPanel;
