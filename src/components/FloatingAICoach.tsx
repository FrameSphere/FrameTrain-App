import { useState, useRef, useEffect, useCallback } from 'react';
import {
  X, Send, Loader2, AlertCircle, CheckCircle, Maximize2, Minimize2,
  MessageSquare, Plus, Trash2, ChevronDown, ChevronRight, Brain,
  FileSearch, Cpu, Sparkles, ArrowLeft, ArrowRight, ExternalLink, HelpCircle, Sliders,
  Play, Square, Zap, Bug, Gauge, Wrench
} from 'lucide-react';
import { useAISettings, TOKEN_BUDGET_CONFIG } from '../contexts/AISettingsContext';
import { useTheme } from '../contexts/ThemeContext';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';
import { callAI as callAIClient } from '../ai/aiClient';
import GradientChatInput from './ui/GradientChatInput';
import { PROVIDER_META } from '../ai/providerMeta';
import { onOpenAICoach } from '../ai/aiCoachEvents';
import {
  buildCoachSystemPrompt, parseCoachActions, navTargetLabel, linkTargetLabel,
  commandLabel, hasPageKnowledge, type CoachAction, type PageId,
} from '../ai/coachContext';
import { navigateTo } from '../ui/navigationEvents';
import { applyCoachConfig, runCoachCommand } from '../ai/coachToolEvents';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import type { Language } from '../contexts/LanguageContext';

// ============ Types ============

interface ThinkingStep {
  id: string;
  label: string;
  detail?: string;
  icon: 'search' | 'brain' | 'cpu' | 'sparkles' | 'check' | 'error';
  status: 'pending' | 'active' | 'done' | 'error';
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  thinkingSteps?: ThinkingStep[];
  thinkingCollapsed?: boolean;
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

// PROVIDER_META kommt aus src/ai/providerMeta.ts (zentral für die App)

const STORAGE_KEY = 'ft_ai_chats_v2';
const MAX_CHATS = 50;

// ============ Helpers ============

function createFreshChat(title: string): Chat {
  return {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title,
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function generateTitle(firstMessage: string): string {
  return firstMessage.slice(0, 40) + (firstMessage.length > 40 ? '…' : '');
}

type TFn = (key: string, fallback?: string) => string;
function formatRelativeTime(ts: number, t: TFn): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (mins < 1) return t('aiCoach.justNow');
  if (mins < 60) return t('aiCoach.minutesAgo').replace('{n}', String(mins));
  if (hours < 24) return t('aiCoach.hoursAgo').replace('{n}', String(hours));
  return (days === 1 ? t('aiCoach.daysAgo') : t('aiCoach.daysAgoPlural')).replace('{n}', String(days));
}

function formatPageContextTitle(pageContent: string): string {
  const firstLine = pageContent.split('\n').find(line => line.trim())?.trim() ?? '';
  const clean = firstLine
    .replace(/^=+\s*/, '')
    .replace(/\s*=+$/, '')
    .replace(/^FrameTrain\s+/i, '')
    .trim();
  return clean || '–';
}

// ============ Markdown Renderer ============

function renderInline(str: string, key?: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match;
  let i = 0;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`t${i++}`}>{str.slice(lastIndex, match.index)}</span>);
    }
    if (match[0].startsWith('**')) {
      parts.push(<strong key={`b${i++}`} className="font-semibold text-white">{match[2]}</strong>);
    } else if (match[0].startsWith('*')) {
      parts.push(<em key={`em${i++}`} className="italic">{match[3]}</em>);
    } else if (match[0].startsWith('`')) {
      parts.push(
        <code key={`c${i++}`} className="px-1.5 py-0.5 bg-white/10 rounded text-[11px] font-mono text-purple-300">
          {match[4]}
        </code>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < str.length) {
    parts.push(<span key={`t${i++}`}>{str.slice(lastIndex)}</span>);
  }
  return parts.length > 0 ? parts : str;
}

function MarkdownText({ text, className = '' }: { text: string; className?: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // ── Code Block (```...```) ──────────────────────────────────────────────
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim(); // z.B. "python", "bash", ""
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // schließendes ``` überspringen
      elements.push(
        <div key={`cb-${i}`} className="my-2 rounded-xl overflow-hidden border border-white/10">
          {lang && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.06] border-b border-white/10">
              <span className="text-[10px] font-mono text-purple-300 font-medium">{lang}</span>
            </div>
          )}
          <pre className="px-3 py-2.5 bg-black/40 overflow-x-auto">
            <code className="text-[11px] font-mono text-emerald-300 leading-relaxed whitespace-pre">
              {codeLines.join('\n')}
            </code>
          </pre>
        </div>
      );
      continue;
    }

    if (!trimmed) {
      elements.push(<div key={i} className="h-1.5" />);
      i++;
      continue;
    }

    if (trimmed.match(/^[-*•]\s/)) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^[-*•]\s/)) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ''));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 mt-0.5 flex-shrink-0 text-xs">•</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (trimmed.match(/^\d+\.\s/)) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^\d+\.\s/)) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 flex-shrink-0 font-medium text-xs w-4">{j + 1}.</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    if (trimmed.startsWith('###')) {
      elements.push(
        <div key={i} className="font-semibold text-white mt-2 mb-1 text-sm">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else if (trimmed.startsWith('##')) {
      elements.push(
        <div key={i} className="font-bold text-white mt-2 mb-1">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else {
      elements.push(
        <p key={i} className="leading-relaxed">
          {renderInline(line)}
        </p>
      );
    }
    i++;
  }

  return <div className={`text-sm space-y-0.5 ${className}`}>{elements}</div>;
}

// ============ Light-Color Detection (Fix für Monochrome / Arctic White) ============

function hexLuminance(hex: string): number {
  try {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const toL = (c: number) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    return 0.2126 * toL(r) + 0.7152 * toL(g) + 0.0722 * toL(b);
  } catch { return 0.5; }
}

function darkenHex(hex: string, amount: number): string {
  try {
    const r = Math.max(0, parseInt(hex.slice(1, 3), 16) - amount);
    const g = Math.max(0, parseInt(hex.slice(3, 5), 16) - amount);
    const b = Math.max(0, parseInt(hex.slice(5, 7), 16) - amount);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  } catch { return hex; }
}

const COACH_MAGIC_CSS = `
  @keyframes ftCoachButtonFloat { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-3px) scale(1.02); } }
  @keyframes ftCoachPanelIn { from { opacity:0; transform:translate3d(10px,14px,0) scale(.975); filter:blur(8px); } to { opacity:1; transform:none; filter:blur(0); } }
  @keyframes ftCoachAuraSweep { 0%,100% { opacity:.28; transform:translateX(-32%); } 50% { opacity:.82; transform:translateX(32%); } }
  @keyframes ftCoachMessageIn { from { opacity:0; transform:translateY(8px) scale(.985); } to { opacity:1; transform:none; } }
  @keyframes ftCoachThinkingSweep { from { transform:translateX(-80%); opacity:.18; } 50% { opacity:.72; } to { transform:translateX(180%); opacity:.18; } }
  @keyframes ftCoachSoftPulse { 0%,100% { box-shadow:0 0 0 rgba(255,255,255,0); } 50% { box-shadow:0 0 26px rgba(255,255,255,.12); } }
  .ft-coach-button { animation:ftCoachButtonFloat 4.5s ease-in-out infinite; }
  .ft-coach-shell {
    animation:ftCoachPanelIn .26s cubic-bezier(.2,.85,.22,1);
    backdrop-filter:blur(22px) saturate(1.28);
    -webkit-backdrop-filter:blur(22px) saturate(1.28);
  }
  .ft-coach-shell::before {
    content:"";
    position:absolute;
    inset:0;
    pointer-events:none;
    border-radius:16px;
    background:
      radial-gradient(circle at 18% 4%, rgba(255,255,255,.10), transparent 28%),
      radial-gradient(circle at 88% 10%, rgba(255,255,255,.08), transparent 30%);
    opacity:.75;
  }
  .ft-coach-shell::after {
    content:"";
    position:absolute;
    top:0;
    left:14px;
    right:14px;
    height:1px;
    pointer-events:none;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
    animation:ftCoachAuraSweep 5.2s ease-in-out infinite;
  }
  .ft-coach-message { animation:ftCoachMessageIn .22s ease both; }
  .ft-coach-scroll::-webkit-scrollbar { width:6px; }
  .ft-coach-scroll::-webkit-scrollbar-track { background:transparent; }
  .ft-coach-scroll::-webkit-scrollbar-thumb { background:rgba(148,163,184,.32); border-radius:999px; }
  .ft-coach-thinking-card { position:relative; overflow:hidden; }
  .ft-coach-thinking-card::before {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent);
    animation:ftCoachThinkingSweep 1.7s ease-in-out infinite;
  }
`;

// ============ Thinking Block ============

function ThinkingBlock({
  steps,
  isActive,
  collapsed,
  onToggle,
}: {
  steps: ThinkingStep[];
  isActive: boolean;
  collapsed: boolean;
  onToggle: () => void;
}) {
  // Fix 4: Theme-Farben im ThinkingBlock
  const { currentTheme } = useTheme();
  const { t } = useLanguage();
  const tPrimary = hexLuminance(currentTheme.colors.primary) > 0.5
    ? darkenHex(currentTheme.colors.primary, 100)
    : currentTheme.colors.primary;
  const borderColor = `${tPrimary}33`;
  const bgColor     = `${tPrimary}1a`;
  const bgHover     = `${tPrimary}26`;
  const textColor   = `${tPrimary}cc`;
  const chevronColor = `${tPrimary}99`;

  const iconMap: Record<ThinkingStep['icon'], React.ReactNode> = {
    search: <FileSearch className="w-3 h-3" />,
    brain: <Brain className="w-3 h-3" />,
    cpu: <Cpu className="w-3 h-3" />,
    sparkles: <Sparkles className="w-3 h-3" />,
    check: <CheckCircle className="w-3 h-3 text-green-400" />,
    error: <AlertCircle className="w-3 h-3 text-red-400" />,
  };

  const activeStep = steps.find(s => s.status === 'active');
  const doneCount = steps.filter(s => s.status === 'done').length;

  if (isActive) {
    return (
      <div className="ft-coach-thinking-card mb-3 rounded-2xl border border-white/10 bg-white/[0.045] p-2.5">
        <div className="flex items-center gap-1.5 text-xs mb-1.5" style={{ color: textColor }}>
          <Loader2 className="w-3 h-3 animate-spin" style={{ color: tPrimary }} />
          <span>{activeStep?.label || t('aiCoach.thinking')}</span>
        </div>
        <div className="pl-1 space-y-1 ml-1.5" style={{ borderLeft: `2px solid ${borderColor}` }}>
          {steps.map(step => (
            <div
              key={step.id}
              className={`flex items-start gap-2 py-0.5 transition-all ${
                step.status === 'active' ? 'opacity-100' : 'opacity-60'
              }`}
            >
              <div className={`flex-shrink-0 mt-0.5 ${
                step.status === 'done' ? 'text-green-400' : ''
              }`} style={step.status === 'active' ? { color: tPrimary } : undefined}>
                {step.status === 'active'
                  ? <Loader2 className="w-3 h-3 animate-spin" />
                  : iconMap[step.icon]
                }
              </div>
              <div className="min-w-0">
                <div className={`text-xs ${
                  step.status === 'active' ? 'text-white' : 'text-gray-400'
                }`}>
                  {step.label}
                </div>
                {step.detail && (
                  <div className="text-[10px] text-gray-600 mt-0.5 break-words">{step.detail}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl transition-all group hover:scale-[1.015]"
        style={{ background: bgColor, border: `1px solid ${borderColor}` }}
        onMouseEnter={e => (e.currentTarget.style.background = bgHover)}
        onMouseLeave={e => (e.currentTarget.style.background = bgColor)}
      >
        <Brain className="w-3 h-3 flex-shrink-0" style={{ color: tPrimary }} />
        <span className="text-[11px] font-medium" style={{ color: textColor }}>
          {t('aiCoach.hadThought')}
        </span>
        <span className="text-[10px] ml-0.5" style={{ color: chevronColor }}>· {doneCount} {t('aiCoach.steps')}</span>
        <ChevronDown
          className={`w-3 h-3 ml-auto transition-transform ${collapsed ? '' : 'rotate-180'}`}
          style={{ color: chevronColor }}
        />
      </button>

      {!collapsed && (
        <div className="mt-1.5 pl-1 space-y-1 ml-1.5 pb-1" style={{ borderLeft: `2px solid ${borderColor}` }}>
          {steps.map(step => (
            <div key={step.id} className="flex items-start gap-2 py-0.5 opacity-60">
              <div className="flex-shrink-0 mt-0.5 text-green-400">
                {iconMap[step.icon]}
              </div>
              <div className="min-w-0">
                <div className="text-xs text-gray-400">{step.label}</div>
                {step.detail && (
                  <div className="text-[10px] text-gray-600 mt-0.5 break-words">{step.detail}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============ Coach Action Chips (Tools) ============

// Set-Chip: übernimmt eine empfohlene Trainings-Config; zeigt nach Klick "Übernommen".
function SetConfigChip({
  summary,
  onApply,
}: {
  summary: string;
  onApply: () => void;
}) {
  const { t } = useLanguage();
  const [applied, setApplied] = useState(false);
  return (
    <button
      onClick={() => { if (!applied) { onApply(); setApplied(true); } }}
      disabled={applied}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-[11px] font-medium transition-all ${
        applied
          ? 'bg-green-500/15 border border-green-500/30 text-green-300'
          : 'bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 hover:scale-[1.02]'
      }`}
      title={summary}
    >
      {applied ? <CheckCircle className="w-3 h-3 flex-shrink-0" /> : <Sliders className="w-3 h-3 flex-shrink-0" />}
      <span className="truncate max-w-[220px]">
        {applied ? t('aiCoach.configApplied') : `${t('aiCoach.applyConfig')}: ${summary}`}
      </span>
    </button>
  );
}

// Zwei-Stufen-Bestätigungs-Chip (für sensible Aktionen wie Training starten).
function ConfirmChip({
  label,
  confirmLabel,
  icon,
  danger,
  onConfirm,
}: {
  label: string;
  confirmLabel: string;
  icon: React.ReactNode;
  danger?: boolean;
  onConfirm: () => void;
}) {
  const [armed, setArmed] = useState(false);
  const tone = danger
    ? 'bg-red-500/15 hover:bg-red-500/25 border-red-500/30 text-red-200'
    : 'bg-amber-500/15 hover:bg-amber-500/25 border-amber-500/30 text-amber-200';
  return (
    <button
      onClick={() => { if (armed) { onConfirm(); setArmed(false); } else { setArmed(true); } }}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border text-[11px] font-medium transition-all hover:scale-[1.02] ${tone}`}
      title={armed ? confirmLabel : label}
    >
      {icon}
      <span className="truncate max-w-[200px]">{armed ? confirmLabel : label}</span>
    </button>
  );
}

function CoachActionChips({
  actions,
  language,
  gradient,
  automation,
  onNavigate,
  onAsk,
  onExplain,
  onEstimate,
  onCommand,
  onTrain,
}: {
  actions: CoachAction[];
  language: Language;
  gradient: string;
  automation: boolean;
  onNavigate: () => void;
  onAsk: (text: string) => void;
  onExplain: (topic: 'error' | 'log') => void;
  onEstimate: () => void;
  onCommand: (action: Extract<CoachAction, { type: 'command' }>) => void;
  onTrain: (op: 'start' | 'stop') => void;
}) {
  const { t } = useLanguage();
  if (actions.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-1">
      {actions.map((a, i) => {
        // ── Fehler/Log erklären ──
        if (a.type === 'explain') {
          return (
            <button
              key={`explain-${i}`}
              onClick={() => onExplain(a.topic)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-white/[0.08] hover:bg-white/[0.14] border border-white/10 text-gray-200 text-[11px] font-medium transition-all hover:scale-[1.02]"
              title={t('aiCoach.explainError')}
            >
              <Bug className="w-3 h-3 opacity-80" />
              {t('aiCoach.explainError')}
            </button>
          );
        }
        // ── RAM schätzen ──
        if (a.type === 'estimate') {
          return (
            <button
              key={`est-${i}`}
              onClick={onEstimate}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-white/[0.08] hover:bg-white/[0.14] border border-white/10 text-gray-200 text-[11px] font-medium transition-all hover:scale-[1.02]"
              title={t('aiCoach.estimateRam')}
            >
              <Gauge className="w-3 h-3 opacity-80" />
              {t('aiCoach.estimateRam')}
            </button>
          );
        }
        // ── Seitenspezifisches Kommando (open/hf/split/apply) ──
        if (a.type === 'command') {
          return (
            <button
              key={`cmd-${i}`}
              onClick={() => onCommand(a)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-white text-[11px] font-medium shadow-lg shadow-black/20 transition-all hover:scale-[1.03]"
              style={{ background: gradient, boxShadow: 'inset 0 1px 0 rgba(255,255,255,.16)' }}
              title={commandLabel(a, language)}
            >
              <Wrench className="w-3 h-3 opacity-90" />
              <span className="truncate max-w-[200px]">{commandLabel(a, language)}</span>
            </button>
          );
        }
        // ── Training starten/stoppen (nur bei Automation, mit Bestätigung) ──
        if (a.type === 'train') {
          if (!automation) return null;
          const start = a.op === 'start';
          return (
            <ConfirmChip
              key={`train-${i}`}
              label={start ? t('aiCoach.trainStart') : t('aiCoach.trainStop')}
              confirmLabel={t('aiCoach.confirm')}
              danger={!start}
              icon={start ? <Play className="w-3 h-3" /> : <Square className="w-3 h-3" />}
              onConfirm={() => onTrain(a.op)}
            />
          );
        }
        // ── Navigation (primär, Theme-Gradient) ──
        if (a.type === 'navigate') {
          const label = t('aiCoach.openPage').replace('{page}', navTargetLabel(a.view, language));
          return (
            <button
              key={`nav-${a.view}-${i}`}
              onClick={() => { navigateTo(a.view); onNavigate(); }}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-white text-[11px] font-medium shadow-lg shadow-black/20 transition-all hover:scale-[1.03]"
              style={{ background: gradient, boxShadow: 'inset 0 1px 0 rgba(255,255,255,.16)' }}
              title={label}
            >
              <ArrowRight className="w-3 h-3 opacity-90" />
              {label}
            </button>
          );
        }
        // ── Externer Hilfe-Link (sekundär) ──
        if (a.type === 'link') {
          const label = linkTargetLabel(a.key, language);
          return (
            <button
              key={`link-${a.key}-${i}`}
              onClick={() => { openUrl(a.url).catch(() => {}); }}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-white/[0.08] hover:bg-white/[0.14] border border-white/10 text-gray-200 text-[11px] font-medium transition-all hover:scale-[1.03]"
              title={a.url}
            >
              <ExternalLink className="w-3 h-3 opacity-80" />
              {label}
            </button>
          );
        }
        // ── Trainings-Config übernehmen ──
        if (a.type === 'set') {
          return (
            <SetConfigChip
              key={`set-${i}`}
              summary={a.summary}
              onApply={() => applyCoachConfig(a.patch)}
            />
          );
        }
        // ── Anschlussfrage (Ein-Klick weiterfragen) ──
        return (
          <button
            key={`ask-${i}`}
            onClick={() => onAsk(a.text)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] border border-dashed border-white/15 text-gray-300 text-[11px] font-medium transition-all hover:scale-[1.02] text-left"
            title={a.text}
          >
            <HelpCircle className="w-3 h-3 flex-shrink-0 opacity-70" />
            <span className="truncate max-w-[200px]">{a.text}</span>
          </button>
        );
      })}
    </div>
  );
}

// ============ Main Component ============

interface FloatingAICoachProps {
  currentPageContent?: string;
  userId?: string;
}

function getChatStorageKey(userId?: string) {
  if (!userId) return STORAGE_KEY;
  return `${STORAGE_KEY}_${userId}`;
}

function loadChatsForUser(userId?: string): Chat[] {
  // Migration: legacy (global) -> user scoped (only if userId present and scoped is empty)
  const scopedKey = getChatStorageKey(userId);
  try {
    const scopedRaw = localStorage.getItem(scopedKey);
    if (scopedRaw) return JSON.parse(scopedRaw);
  } catch { /* ignore */ }

  if (!userId) return [];

  try {
    const legacyRaw = localStorage.getItem(STORAGE_KEY);
    if (!legacyRaw) return [];
    const parsed = JSON.parse(legacyRaw) as Chat[];
    // Only migrate if scoped key is missing/empty
    localStorage.setItem(scopedKey, JSON.stringify(parsed));
    localStorage.removeItem(STORAGE_KEY);
    return parsed;
  } catch {
    return [];
  }
}

function saveChatsForUser(chats: Chat[], userId?: string): void {
  try {
    const limited = chats.slice(0, MAX_CHATS);
    localStorage.setItem(getChatStorageKey(userId), JSON.stringify(limited));
  } catch { /* ignore */ }
}

export default function FloatingAICoach({ currentPageContent, userId }: FloatingAICoachProps) {
  const { settings } = useAISettings();
  const budgetCfg = TOKEN_BUDGET_CONFIG[settings.tokenBudget ?? 'balanced'];
  const { language } = useLanguage();
  const { currentTheme } = useTheme();
  const { currentPageContent: ctxPageContent, currentPageId } = usePageContext();
  const { t } = useLanguage();

  // Theme-Farben mit Light-Color-Detection (Fix für Monochrome / Arctic White)
  const safePrimary   = hexLuminance(currentTheme.colors.primary)   > 0.5 ? darkenHex(currentTheme.colors.primary,   100) : currentTheme.colors.primary;
  const safeSecondary = hexLuminance(currentTheme.colors.secondary) > 0.5 ? darkenHex(currentTheme.colors.secondary, 100) : currentTheme.colors.secondary;
  const themeGradient       = `linear-gradient(135deg, ${safePrimary}, ${safeSecondary})`;
  const themeGradientSubtle = `linear-gradient(to right, ${safePrimary}1a, ${safeSecondary}0d)`;

  const pageContent = currentPageContent || ctxPageContent || '';

  // Modal state
  const [isOpen, setIsOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [view, setView] = useState<'chat' | 'chatList'>('chat');
  // Automation-Modus: schaltet Training-Start/Stop-Tools frei (persistiert)
  const [automation, setAutomation] = useState<boolean>(() => {
    try { return localStorage.getItem('ft_coach_automation') === '1'; } catch { return false; }
  });
  const toggleAutomation = useCallback(() => {
    setAutomation(v => {
      const nv = !v;
      try { localStorage.setItem('ft_coach_automation', nv ? '1' : '0'); } catch { /* ignore */ }
      return nv;
    });
  }, []);

  // ── Chat state ──
  // `chats` = only persisted chats (have at least one message)
  // `currentChat` = the currently displayed chat (may be unsaved/empty)
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  // Track if currentChat is already in `chats` (persisted)
  const currentChatPersistedRef = useRef(false);

  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  // Letzte fehlgeschlagene Nachricht — für "Erneut senden" nach Fehler
  const lastFailedTextRef = useRef<string | null>(null);

  // Draggable/resizable state
  const [position, setPosition] = useState({ x: window.innerWidth - 390, y: window.innerHeight - 560 });
  const [size, setSize] = useState({ width: 370, height: 520 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, w: 0, h: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const prevMsgCountRef = useRef(0);
  // Letzter Seitenkontext der dem Modell bekannt ist
  const lastSentPageContentRef = useRef<string>('');
  // Seiten, deren tiefes Wissen im aktuellen Chat bereits ans Modell ging
  // (verhindert erneutes Senden bei jeder Nachricht auf derselben Seite)
  const knowledgeSentRef = useRef<Set<PageId>>(new Set());

  // Load chats from localStorage on mount (only persisted ones)
  useEffect(() => {
    const loaded = loadChatsForUser(userId);
    setChats(loaded);
  }, [userId]);

  // Fix 3: Scroll nur wenn neue Nachricht hinzukommt, nicht bei Collapse-Toggle
  useEffect(() => {
    const count = currentChat?.messages.length ?? 0;
    if (count > prevMsgCountRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    prevMsgCountRef.current = count;
  }, [currentChat?.messages]);

  // ── Drag / resize ──
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newX = Math.max(0, Math.min(window.innerWidth - size.width, e.clientX - dragOffset.x));
        const newY = Math.max(0, Math.min(window.innerHeight - size.height, e.clientY - dragOffset.y));
        setPosition({ x: newX, y: newY });
      } else if (isResizing) {
        const dx = e.clientX - resizeStart.x;
        const dy = e.clientY - resizeStart.y;
        setSize({ width: Math.max(300, resizeStart.w + dx), height: Math.max(320, resizeStart.h + dy) });
      }
    };
    const handleMouseUp = () => { setIsDragging(false); setIsResizing(false); };
    if (isDragging || isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, isResizing, dragOffset, resizeStart, size]);

  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    setIsDragging(true);
    setDragOffset({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, w: size.width, h: size.height });
  };

  // ── Opens modal with a fresh empty chat (never saved until message sent) ──
  const openModal = useCallback(() => {
    const fresh = createFreshChat(t('aiCoach.newChat'));
    setCurrentChat(fresh);
    currentChatPersistedRef.current = false;
    knowledgeSentRef.current = new Set();
    lastSentPageContentRef.current = '';
    setView('chat');
    setError('');
    setInputText('');
    setThinkingSteps([]);
    setIsOpen(true);
  }, []);

  // ── Start a new chat (from chat list or header button) ──
  // Creates a fresh unsaved chat and switches to it, without touching `chats`
  const handleNewChat = useCallback(() => {
    const fresh = createFreshChat(t('aiCoach.newChat'));
    setCurrentChat(fresh);
    currentChatPersistedRef.current = false;
    knowledgeSentRef.current = new Set();
    lastSentPageContentRef.current = '';
    setView('chat');
    setError('');
    setInputText('');
    setThinkingSteps([]);
  }, []);

  // External open (z.B. aus Analysis/Training/Dev Panels)
  useEffect(() => {
    return onOpenAICoach(({ prefill, newChat }) => {
      if (!isOpen) {
        openModal();
      } else if (newChat) {
        handleNewChat();
      }
      if (typeof prefill === 'string' && prefill.trim()) {
        setInputText(prefill);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    });
  }, [handleNewChat, isOpen, openModal]);

  // ── Switch to an existing persisted chat ──
  const switchToChat = useCallback((chatId: string) => {
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentChat(chat);
      currentChatPersistedRef.current = true;
      // Neuer Chat-Kontext → Seiten-Wissen im nächsten Turn frisch mitschicken
      knowledgeSentRef.current = new Set();
      lastSentPageContentRef.current = '';
    }
    setView('chat');
    setError('');
  }, [chats]);

  // ── Delete a persisted chat ──
  const deleteChat = useCallback((chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = chats.filter(c => c.id !== chatId);
    setChats(updated);
    saveChatsForUser(updated, userId);
    // If we're currently viewing this chat, open a fresh one
      if (currentChat?.id === chatId) {
      const fresh = createFreshChat(t('aiCoach.newChat'));
      setCurrentChat(fresh);
      currentChatPersistedRef.current = false;
    }
  }, [chats, currentChat, t]);

  // ── Update currentChat and sync to `chats` if already persisted ──
  const applyToCurrentChat = useCallback((updater: (c: Chat) => Chat) => {
    setCurrentChat(prev => {
      if (!prev) return prev;
      const updated = updater(prev);
      // Also update in chats list if this chat is persisted
      if (currentChatPersistedRef.current) {
        setChats(prevChats => {
          const newChats = prevChats.map(c => c.id === updated.id ? updated : c);
          saveChatsForUser(newChats, userId);
          return newChats;
        });
      }
      return updated;
    });
  }, []);

  // ── Ensure currentChat is persisted (call before first message) ──
  const ensurePersisted = useCallback((chat: Chat) => {
    if (!currentChatPersistedRef.current) {
      currentChatPersistedRef.current = true;
      setChats(prev => {
        const newChats = [chat, ...prev];
        saveChatsForUser(newChats, userId);
        return newChats;
      });
    }
  }, []);

  // ── System Prompt ───────────────────────────────────────────────────────────
  // Einheitliches System via src/ai/coachContext.ts (lazy per-page knowledge):
  //   - 1. Message: Persona + Kurzüberblick + Skills + Tools + Wissen der AKTUELLEN
  //                 Seite + deren Live-Zustand.
  //   - Seitenwechsel: Wissen der NEUEN Seite (falls neu) + neuer Live-Zustand.
  //   - sonst: nur die Persona.
  // Sprache folgt der App-Sprache (DE/EN).
  const buildSystemPrompt = (isFirstMessage: boolean): string => {
    const pageChanged = !isFirstMessage &&
      !!pageContent &&
      pageContent !== lastSentPageContentRef.current;

    // Tiefes Seiten-Wissen nur senden, wenn für diese Seite in diesem Chat neu.
    const pid = currentPageId;
    const includePageKnowledge =
      !!pid &&
      hasPageKnowledge(pid) &&
      !knowledgeSentRef.current.has(pid) &&
      (isFirstMessage || pageChanged);

    if (includePageKnowledge && pid) knowledgeSentRef.current.add(pid);
    if (isFirstMessage || pageChanged) lastSentPageContentRef.current = pageContent;

    return buildCoachSystemPrompt({
      language,
      pageId: pid,
      pageContent,
      isFirstMessage,
      pageChanged,
      includePageKnowledge,
      automation,
    });
  };

  // ── Token-Budget für History ──────────────────────────────────────────────────
  // Begrenzt die History auf ~2000 Tokens (neueste zuerst), statt blind slice(-10)
  const buildTokenBudgetedHistory = (messages: Message[]): { role: 'user' | 'assistant'; content: string }[] => {
    const MAX_TOKENS = budgetCfg.historyTokenBudget;
    let budget = 0;
    const selected: Message[] = [];
    for (let i = messages.length - 1; i >= 0; i--) {
      const estimated = Math.ceil(messages[i].content.length / 4);
      if (budget + estimated > MAX_TOKENS && selected.length >= 2) break;
      selected.unshift(messages[i]);
      budget += estimated;
    }
    return selected.map(m => ({ role: m.role, content: m.content }));
  };

  // ── Thinking Steps (erweiterter Flow) ──────────────────────────────────────
  // Gibt die finalen done-Steps zurück für die Message-Persistenz.
  const runThinkingAnimation = async ({
    isFirstMessage,
    hasPageContext,
    historyLength,
    userMessage,
  }: {
    isFirstMessage: boolean;
    hasPageContext: boolean;
    historyLength: number;
    userMessage: string;
  }): Promise<ThinkingStep[]> => {

    // Wir bauen die Steps dynamisch basierend auf dem tatsächlichen Kontext
    type StepDef = { id: string; label: string; detail?: string; icon: ThinkingStep['icon']; ms: number };
    const defs: StepDef[] = [];

    // Schritt 1: Kontext lesen — nur beim ersten Message ist das der Seiteninhalt
    if (isFirstMessage && hasPageContext) {
      defs.push({
        id: 's_ctx',
        label: t('aiCoach.thinking.analyzePage'),
        detail: pageContent.split('\n').find(l => l.trim())?.trim()?.slice(0, 70) + '...',
        icon: 'search',
        ms: 420,
      });
    } else if (!isFirstMessage && pageContent && pageContent !== lastSentPageContentRef.current) {
      // Seitenwechsel erkannt
      defs.push({
        id: 's_ctx',
        label: t('aiCoach.thinking.pageChanged'),
        detail: pageContent.split('\n').find(l => l.trim())?.trim()?.slice(0, 70) + '...',
        icon: 'search',
        ms: 380,
      });
    } else if (historyLength > 0) {
      defs.push({
        id: 's_ctx',
        label: t('aiCoach.thinking.readingHistory').replace('{n}', String(historyLength)),
        icon: 'cpu',
        ms: 280,
      });
    }

    // Schritt 2: Intent klassifizieren — was will der User?
    const isErrorMsg     = /fehler|error|crash|oom|failed|kaputt|problem|nicht|can't|cannot/i.test(userMessage);
    const isHowToMsg     = /wie|how|what|was|erkl|explain|versteh|help/i.test(userMessage);
    const isConfigMsg    = /batch|lr|learning|epoch|lora|config|param|setting|einstellung/i.test(userMessage);
    const intentLabel = isErrorMsg  ? t('aiCoach.thinking.intentError')
                      : isConfigMsg ? t('aiCoach.thinking.intentConfig')
                      : isHowToMsg  ? t('aiCoach.thinking.intentExplain')
                      : t('aiCoach.thinking.intentGeneral');
    defs.push({
      id: 's_intent',
      label: intentLabel,
      icon: 'brain',
      ms: 320,
    });

    // Schritt 3: Denken / Formulieren
    defs.push({
      id: 's_think',
      label: t('aiCoach.thinking.formulating'),
      icon: 'sparkles',
      ms: 0, // läuft bis API antwortet — kein timeout hier
    });

    // Animation starten
    const allDone: ThinkingStep[] = defs.map(d => ({
      id: d.id, label: d.label, detail: d.detail, icon: d.icon, status: 'done' as const
    }));

    for (let i = 0; i < defs.length; i++) {
      const def = defs[i];
      // Vorherige als done markieren, aktuellen als active
      setThinkingSteps([
        ...defs.slice(0, i).map(d => ({ id: d.id, label: d.label, detail: d.detail, icon: d.icon, status: 'done' as const })),
        { id: def.id, label: def.label, detail: def.detail, icon: def.icon, status: 'active' as const },
      ]);
      if (def.ms > 0) await new Promise(r => setTimeout(r, def.ms));
    }

    return allDone;
  };

  // ── AI-Titel generierung (feuert im Hintergrund nach erster Antwort) ────────────────
  const generateAITitle = useCallback(async (chatId: string, userMessage: string, assistantResponse: string) => {
    try {
      const titlePrompt = `Generate a SHORT chat title (3-6 words max) for this conversation.
User asked: "${userMessage.slice(0, 200)}"
Assistant answered about: "${assistantResponse.slice(0, 200)}"

Rules:
- 3 to 6 words only
- No punctuation at the end
- No quotes
- Capture the core topic
- Same language as the user message
- Examples: "YOLO Training Konfiguration", "Batch Size Fehler beheben", "LoRA vs Full Fine-Tuning"

Reply with ONLY the title, nothing else.`;

      const title = await callAIClient(settings, {
        system: 'You generate concise chat titles. Reply with ONLY the title, no explanation, no quotes, no punctuation at end.',
        messages: [{ role: 'user', content: titlePrompt }],
        maxTokens: 20,
        temperature: 0.4,
      });

      const cleanTitle = title.trim().replace(/^["']|["']$/g, '').replace(/\.$/, '').slice(0, 50);
      if (!cleanTitle) return;

      // Titel in chats und currentChat aktualisieren
      setChats(prev => {
        const updated = prev.map(c => c.id === chatId ? { ...c, title: cleanTitle } : c);
        saveChatsForUser(updated, userId);
        return updated;
      });
      setCurrentChat(prev => prev?.id === chatId ? { ...prev, title: cleanTitle } : prev);
    } catch {
      // Titel-Generierung ist optional — Fehler still ignorieren
    }
  }, [settings, userId]);

  const sendMessage = async (overrideText?: string, isRetry = false) => {
    const text = (overrideText ?? inputText).trim();
    if (!text || isLoading || !currentChat) return;

    if (!settings.enabled) {
      setError(t('aiCoach.disabledError'));
      return;
    }

    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const chatSnapshot = currentChat;
    if (!isRetry) setInputText('');
    setError('');
    lastFailedTextRef.current = text;

    // Retry: die User-Nachricht steht bereits im Chat — nicht doppelt anhängen
    const lastMsg = chatSnapshot.messages[chatSnapshot.messages.length - 1];
    const alreadyInChat = isRetry && lastMsg?.role === 'user' && lastMsg?.content === text;

    const isFirstMessage = alreadyInChat
      ? chatSnapshot.messages.length === 1
      : chatSnapshot.messages.length === 0;
    const chatWithUserMsg: Chat = alreadyInChat
      ? chatSnapshot
      : {
          ...chatSnapshot,
          messages: [...chatSnapshot.messages, userMsg],
          title: isFirstMessage ? generateTitle(text) : chatSnapshot.title,
          updatedAt: Date.now(),
        };

    if (isFirstMessage) {
      ensurePersisted(chatWithUserMsg);
    }

    setCurrentChat(chatWithUserMsg);
    if (!isFirstMessage && currentChatPersistedRef.current) {
      setChats(prev => {
        const updated = prev.map(c => c.id === chatWithUserMsg.id ? chatWithUserMsg : c);
        saveChatsForUser(updated, userId);
        return updated;
      });
    }

    setIsLoading(true);

    // Thinking Animation starten (parallel zum API-Call)
    const thinkingPromise = runThinkingAnimation({
      isFirstMessage,
      hasPageContext: !!pageContent,
      historyLength: chatSnapshot.messages.length,
      userMessage: text,
    });

    try {
      const meta = PROVIDER_META[settings.provider];
      if (meta.needsKey && !settings.apiKey) {
        throw new Error(t('aiCoach.apiKeyMissing'));
      }

      // Token-budgetierte History
      const history = buildTokenBudgetedHistory(chatWithUserMsg.messages);
      // System Prompt: Seitenkontext nur beim ersten Message
      const systemPrompt = buildSystemPrompt(isFirstMessage);

      // API-Call und Thinking-Animation parallel — API-Call bestimmt wann Step 3 endet
      const [responseText, finalThinkingSteps] = await Promise.all([
        callAIClient(settings, {
          system: systemPrompt,
          messages: history,
          maxTokens: budgetCfg.maxTokens,
          temperature: 0.7,
          responseLanguage: language,
        }),
        thinkingPromise,
      ]);

      setThinkingSteps([]);

      const assistantMsg: Message = {
        id: `msg-${Date.now()}-ai`,
        role: 'assistant',
        content: responseText,
        timestamp: Date.now(),
        thinkingSteps: finalThinkingSteps,
        thinkingCollapsed: true,
      };

      setCurrentChat(prev => {
        if (!prev) return prev;
        const updated: Chat = { ...prev, messages: [...prev.messages, assistantMsg], updatedAt: Date.now() };
        if (currentChatPersistedRef.current) {
          setChats(prevChats => {
            const newChats = prevChats.map(c => c.id === updated.id ? updated : c);
            saveChatsForUser(newChats, userId);
            return newChats;
          });
        }
        return updated;
      });

      // AI-Titel im Hintergrund generieren (nur nach erster Antwort)
      if (isFirstMessage) {
        generateAITitle(chatSnapshot.id, text, responseText);
      }
      lastFailedTextRef.current = null;

    } catch (e: any) {
      setThinkingSteps([]);
      setError(e?.message || t('aiCoach.unknownError'));
    } finally {
      setIsLoading(false);
    }
  };

  const retryLastMessage = () => {
    const text = lastFailedTextRef.current;
    if (!text || isLoading) return;
    sendMessage(text, true);
  };

  // ── Tool-Handler ─────────────────────────────────────────────────────────
  const en = language === 'en';

  // Fehler/Log erklären: frischen Seitenkontext (mit aktuellem Log) erzwingen + fragen
  const handleExplain = (topic: 'error' | 'log') => {
    if (isLoading) return;
    lastSentPageContentRef.current = ''; // erzwingt Re-Injektion des aktuellen Kontexts
    const prompt = topic === 'log'
      ? (en ? 'Explain the current log output and what it means.' : 'Erkläre die aktuelle Log-Ausgabe und was sie bedeutet.')
      : (en ? 'Explain the current error in detail and give me a concrete fix.' : 'Erkläre den aktuellen Fehler im Detail und gib mir einen konkreten Fix.');
    sendMessage(prompt);
  };

  // RAM schätzen: frischen Kontext (mit Modellgröße + Schätzung) + fragen
  const handleEstimate = () => {
    if (isLoading) return;
    lastSentPageContentRef.current = '';
    sendMessage(en
      ? 'Estimate RAM/VRAM and rough time for my current model and config, and how to reduce it if needed.'
      : 'Schätze RAM/VRAM und grob die Zeit für mein aktuelles Modell + Config — und wie ich es bei Bedarf senken kann.');
  };

  // Seitenspezifisches Kommando: ggf. zur Zielseite navigieren, dann ausführen
  const handleCommand = (action: Extract<CoachAction, { type: 'command' }>) => {
    navigateTo(action.page);
    runCoachCommand(action.command);
    setIsOpen(false); // Coach schließen, damit der User den Dialog/das Ergebnis sieht
  };

  // Training starten/stoppen (nur nach ConfirmChip-Bestätigung)
  const handleTrain = (op: 'start' | 'stop') => {
    navigateTo('training');
    runCoachCommand(op === 'start' ? { kind: 'startTraining' } : { kind: 'stopTraining' });
    setIsOpen(false);
  };

  // ── Toggle thinking collapse for a specific message ──
  const toggleMessageThinking = (msgId: string) => {
    applyToCurrentChat(c => ({
      ...c,
      messages: c.messages.map(m =>
        m.id === msgId ? { ...m, thinkingCollapsed: !m.thinkingCollapsed } : m
      ),
    }));
  };

  if (!settings.enabled) return null;

  // ── Closed: floating button ──
  if (!isOpen) {
    return (
      <>
        <style>{COACH_MAGIC_CSS}</style>
        <button
          onClick={openModal}
          className="ft-coach-button fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-2xl hover:shadow-white/20 hover:scale-110 transition-all flex items-center justify-center z-40"
          style={{
            background: themeGradient,
            boxShadow: `0 18px 48px ${safePrimary}33, 0 0 0 1px rgba(255,255,255,.14), inset 0 1px 0 rgba(255,255,255,.28)`,
          }}
          title={t('aiCoach.openTitle')}
        >
          <Sparkles className="absolute w-3 h-3 text-white/70 -top-0.5 right-1.5" />
          <Brain className="w-6 h-6 text-white" />
        </button>
      </>
    );
  }

  // ── Chat list view ──
  const renderChatListView = () => (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-white/10 flex-shrink-0 select-none cursor-move"
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <MessageSquare className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-semibold text-white">{t('aiCoach.chatHistory')}</span>
          <span className="text-xs text-gray-500">({chats.length})</span>
        </div>
        <div className="pointer-events-auto flex items-center gap-1">
          <button
            onClick={handleNewChat}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all"
            title={t('aiCoach.newChat')}
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={() => setView('chat')}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title={t('aiCoach.back')}
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="ft-coach-scroll flex-1 overflow-y-auto p-2 space-y-1">
        {chats.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center p-4">
            <MessageSquare className="w-8 h-8 text-gray-600" />
            <p className="text-gray-500 text-sm">{t('aiCoach.noChats')}</p>
            <p className="text-gray-600 text-xs">{t('aiCoach.noChatsHint')}</p>
            <button
              onClick={handleNewChat}
              className="px-3 py-2 text-xs font-medium bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg border border-purple-500/30 transition-all flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('aiCoach.newChatStart')}
            </button>
          </div>
        ) : (
          <>
            <button
              onClick={handleNewChat}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-white/10 hover:border-purple-500/30 hover:bg-purple-500/5 text-gray-400 hover:text-purple-300 transition-all text-xs font-medium"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('aiCoach.newChat')}
            </button>
            {chats.map(chat => (
              <div key={chat.id} className="ft-coach-message flex items-center gap-1.5 group">
                <button
                  onClick={() => switchToChat(chat.id)}
                  className={`flex-1 flex items-start justify-between gap-2 px-3 py-2.5 rounded-xl border text-left transition-all ${
                    chat.id === currentChat?.id
                      ? 'bg-white/[0.10] border-white/20 text-white shadow-lg shadow-black/20'
                      : 'bg-white/[0.035] border-white/5 hover:bg-white/[0.07] hover:border-white/10 text-gray-300 hover:translate-x-0.5'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium truncate">{chat.title}</div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[10px] text-gray-600">{formatRelativeTime(chat.updatedAt, t)}</span>
                      {chat.messages.length > 0 && (
                        <span className="text-[10px] text-gray-600">{chat.messages.length} {t('aiCoach.messagesCount')}</span>
                      )}
                    </div>
                  </div>
                </button>
                <button
                  onClick={(e) => deleteChat(chat.id, e)}
                  className="p-1.5 rounded-lg hover:bg-red-500/20 text-gray-600 hover:text-red-400 transition-all flex-shrink-0 hover:scale-110"
                  title={t('aiCoach.deleteChat')}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );

  // ── Chat view ──
  const renderChatView = () => (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 flex-shrink-0 select-none cursor-move"
        style={{ background: themeGradientSubtle }}
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none min-w-0">
          <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: themeGradient }}>
            <Brain className="w-3.5 h-3.5 text-white" />
          </div>
          <div className="min-w-0">
            <div className="text-xs font-bold text-white truncate max-w-[160px]">
              {currentChat?.title || t('aiCoach.openTitle')}
            </div>
            {pageContent && (
              <div className="text-[10px] text-gray-500 truncate max-w-[160px]">
                {formatPageContextTitle(pageContent).slice(0, 35)}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 pointer-events-auto flex-shrink-0">
          <button
            onClick={toggleAutomation}
            className={`p-1.5 rounded-lg transition-all ${automation ? 'bg-amber-500/20 text-amber-300' : 'hover:bg-white/10 text-gray-400 hover:text-amber-300'}`}
            title={automation ? t('aiCoach.automationOn') : t('aiCoach.automationOff')}
          >
            <Zap className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setView('chatList')} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all" title={t('aiCoach.chatHistory')}>
            <MessageSquare className="w-3.5 h-3.5" />
          </button>
          <button onClick={handleNewChat} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all" title={t('aiCoach.newChat')}>
            <Plus className="w-3.5 h-3.5" />
          </button>
          {!isMaximized && (
            <button onClick={() => setIsMaximized(true)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
          {isMaximized && (
            <button onClick={() => setIsMaximized(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
              <Minimize2 className="w-3.5 h-3.5" />
            </button>
          )}
          <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="ft-coach-scroll flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {(!currentChat || currentChat.messages.length === 0) && !isLoading && (
          <div className="ft-coach-message flex flex-col items-center justify-center h-full text-center gap-3 py-8">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center"
              style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}33, ${currentTheme.colors.secondary}1a)`, boxShadow: `0 14px 34px ${safePrimary}22, inset 0 1px 0 rgba(255,255,255,.12)` }}>
              <Brain className="w-6 h-6" style={{ color: safePrimary }} />
            </div>
            <div>
              <p className="text-gray-300 text-sm font-medium">{t('aiCoach.greeting')}</p>
              <p className="text-gray-600 text-xs mt-1">
                {pageContent
                  ? t('aiCoach.greetingWithContext')
                  : t('aiCoach.greetingNoContext')}
              </p>
            </div>
            {pageContent && (
              <div className="flex justify-center w-full">
                <div className="max-w-[220px] px-3 py-2 bg-white/[0.03] border border-white/5 rounded-xl space-y-1">
                  <div className="flex items-center gap-1 leading-none">
                    <FileSearch className="w-3 h-3 flex-shrink-0" style={{ color: safePrimary }} />
                    <span className="text-[10px] font-medium" style={{ color: `${safePrimary}cc` }}>{t('aiCoach.loadedContext')}</span>
                  </div>
                  <p className="text-[10px] text-gray-600 leading-relaxed truncate">
                    {formatPageContextTitle(pageContent)}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {currentChat?.messages.map(msg => (
          <div key={msg.id} className={`ft-coach-message flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' ? (() => {
              const { cleanedText, actions } = parseCoachActions(msg.content);
              return (
              <div className="max-w-[88%] space-y-1">
                {/* Thinking block — shown above the message when finished */}
                {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                  <ThinkingBlock
                    steps={msg.thinkingSteps}
                    isActive={false}
                    collapsed={msg.thinkingCollapsed ?? true}
                    onToggle={() => toggleMessageThinking(msg.id)}
                  />
                )}
                <div className="px-3 py-2.5 rounded-2xl rounded-tl-sm bg-white/[0.065] border border-white/[0.10] text-gray-200 shadow-lg shadow-black/20">
                  <MarkdownText text={cleanedText} />
                </div>
                <CoachActionChips
                  actions={actions}
                  language={language}
                  gradient={themeGradient}
                  automation={automation}
                  onNavigate={() => setIsOpen(false)}
                  onAsk={(text) => { if (!isLoading) sendMessage(text); }}
                  onExplain={handleExplain}
                  onEstimate={handleEstimate}
                  onCommand={handleCommand}
                  onTrain={handleTrain}
                />
              </div>
              );
            })() : (
              <div className="max-w-[85%] px-3 py-2.5 rounded-2xl rounded-tr-sm text-white text-sm leading-relaxed shadow-lg shadow-black/20"
                style={{ background: themeGradient, boxShadow: `0 12px 28px ${safePrimary}20, inset 0 1px 0 rgba(255,255,255,.16)` }}>
                {msg.content}
              </div>
            )}
          </div>
        ))}

        {/* Active thinking steps (while loading) */}
        {isLoading && thinkingSteps.length > 0 && (
          <div className="ft-coach-message flex justify-start">
            <div className="max-w-[88%]">
              <ThinkingBlock
                steps={thinkingSteps}
                isActive={true}
                collapsed={false}
                onToggle={() => {}}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="ft-coach-message flex justify-start">
            <div className="max-w-[88%] px-3 py-2.5 rounded-2xl bg-red-500/10 border border-red-500/20 flex flex-col gap-2">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                <span className="text-red-300 text-xs leading-relaxed break-words">{error}</span>
              </div>
              {lastFailedTextRef.current && !isLoading && (
                <button
                  onClick={retryLastMessage}
                  className="self-start px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-200 text-xs font-medium transition-all"
                >
                  {t('aiCoach.retryButton')}
                </button>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 pb-3 pt-2 flex-shrink-0 border-t border-white/5 bg-black/10">
        <GradientChatInput
          ref={inputRef}
          value={inputText}
          onChange={setInputText}
          onSend={sendMessage}
          loading={isLoading}
          placeholder={t('aiCoach.inputPlaceholder')}
          size="sm"
          gradient={themeGradient}
          primaryColor={safePrimary}
          sendTitle={t('aiCoach.inputPlaceholder')}
        />
      </div>
    </div>
  );

  const content = view === 'chatList' ? renderChatListView() : renderChatView();

  // ── Maximized overlay ──
  if (isMaximized) {
    return (
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
        <style>{COACH_MAGIC_CSS}</style>
        <div className="ft-coach-shell relative bg-slate-900/90 rounded-2xl border border-white/10 w-full max-w-2xl h-[85vh] overflow-hidden flex flex-col shadow-2xl">
          {content}
        </div>
      </div>
    );
  }

  // ── Floating window ──
  return (
    <div
      className="ft-coach-shell fixed bg-slate-900/90 rounded-2xl border border-white/[0.10] shadow-2xl overflow-hidden flex flex-col z-50"
      style={{
        width: `${size.width}px`,
        height: `${size.height}px`,
        left: `${position.x}px`,
        top: `${position.y}px`,
        boxShadow: `0 25px 70px rgba(0,0,0,0.52), 0 0 0 1px rgba(255,255,255,0.06), 0 0 48px ${safePrimary}14`,
      }}
    >
      <style>{COACH_MAGIC_CSS}</style>
      {content}
      <div
        className="absolute bottom-0 right-0 w-5 h-5 cursor-se-resize"
        onMouseDown={handleResizeMouseDown}
        style={{
          background: `linear-gradient(135deg, transparent 50%, ${currentTheme.colors.primary}4d 100%)`,
          borderRadius: '0 0 16px 0',
        }}
      />
    </div>
  );
}
