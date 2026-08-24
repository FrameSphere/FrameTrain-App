// DevTrainPanel.tsx – Dev Train Mode (ausgekoppelt aus TrainingPanel)
// KI-Assistent kann den Code direkt bearbeiten (EDIT-Protokoll)

import { useCallback, useMemo, useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Loader2, Terminal, FolderOpen, FileCode,
  FolderClosed, Bot, Send, Maximize2, Minimize2, X, Minus, Plus,
  AlertCircle, CheckCircle, TrendingDown, BarChart3, Zap,
  Save, FileText, Trash2, Pencil, Check, Wand2, Sparkles, Copy,
  History, MessageSquarePlus, Globe,
  MemoryStick, Database, Package, Bug, HelpCircle,
} from 'lucide-react';
import OpenLibraryModal from './OpenLibraryModal';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';
import type { TrainingJob, TrainingProgress, LossPoint, ModelInfo, DatasetInfo } from './TrainingPanel';
import { callAI, LossChart } from './TrainingPanel';
import TrainingDashboard from './TrainingDashboard';
import { classifyError, type ErrorCategory } from '../utils/errorClassify';
import { useEscapeKey } from '../hooks/useEscapeKey';
import { parseEdits, applyEdit, applyAllEdits, removeEditBlocks, extractFullPythonCode, type CodeEdit } from '../ai/codeEdits';
import { buildAutoSystemPrompt, parseAutoAction, type AutoAction } from '../ai/autoModeProtocol';
import { sendAppErrorReport } from '../utils/errorReport';
import { migrateLegacyDevScripts } from '../utils/devScriptStorage';
import { detectScriptModality, type ScriptModality } from './scriptModality';
import DiffViewer from './DiffViewer';
import { dateLocale } from '../utils/dateLocale';

// ── Script Library ────────────────────────────────────────────────────────

interface SavedScript { id: string; name: string; script: string; savedAt: string; }

const getScriptsKey = (userId?: string) => userId ? `ft_saved_scripts_${userId}` : 'ft_saved_scripts';
const loadScripts  = (userId?: string): SavedScript[] => { try { const parsed = JSON.parse(localStorage.getItem(getScriptsKey(userId)) ?? '[]'); return Array.isArray(parsed) ? parsed.filter((s: unknown): s is SavedScript => !!s && typeof (s as SavedScript).script === 'string') : []; } catch { return []; } };
const saveScript   = (name: string, script: string, userId?: string) => { const key = getScriptsKey(userId); const all = loadScripts(userId); all.unshift({ id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString() }); localStorage.setItem(key, JSON.stringify(all.slice(0, 50))); };
const deleteScript = (id: string, userId?: string) => { const key = getScriptsKey(userId); localStorage.setItem(key, JSON.stringify(loadScripts(userId).filter(s => s.id !== id))); };
const updateScript = (id: string, script: string, userId?: string) => { const key = getScriptsKey(userId); const all = loadScripts(userId); const idx = all.findIndex(s => s.id === id); if (idx >= 0) { all[idx] = { ...all[idx], script, savedAt: new Date().toISOString() }; localStorage.setItem(key, JSON.stringify(all)); } };

// ── Edit Parsing ──────────────────────────────────────────────────────────
// Zentralisiert in src/ai/codeEdits.ts

// ── Line Highlighting Utilities ────────────────────────────────────────────

interface HighlightedLine { lineNum: number; type: 'added' | 'removed' | 'modified'; }

function calculateAffectedLines(script: string, edit: CodeEdit): HighlightedLine[] {
  const findLines = edit.find.split('\n');
  const replaceLines = edit.replace.split('\n');
  const affected: HighlightedLine[] = [];

  // Find where the edit text appears in the script (case-sensitive exact match)
  const findStart = script.indexOf(edit.find);
  if (findStart === -1) return affected; // Not found

  // Calculate the starting line number (1-indexed)
  // Count all newlines before the findStart position
  const linesBeforeFindCount = (script.slice(0, findStart).match(/\n/g) || []).length;
  const startLineNum = linesBeforeFindCount + 1;
  
  // Lines being REMOVED (show in RED) - the find block
  for (let i = 0; i < findLines.length; i++) {
    affected.push({ lineNum: startLineNum + i, type: 'removed' });
  }

  // Lines being ADDED (show in GREEN) - the replace block, appears right after removed lines
  for (let i = 0; i < replaceLines.length; i++) {
    affected.push({ lineNum: startLineNum + findLines.length + i, type: 'added' });
  }

  return affected;
}

// ── Error Categorization ──────────────────────────────────────────────────

// Kategorie aus der gemeinsamen classifyError-Quelle — dieselbe Einordnung wie
// im Trainings-Dashboard. Vorher hatte diese Datei eine eigene, abweichende
// Reihenfolge (nan/inf/loss VOR Code-Fehlern) und stufte denselben Fehler
// anders ein als das Dashboard.
function analyzeError(errorMsg: string): { category: ErrorCategory } {
  return { category: classifyError(errorMsg) };
}

// Kein Emoji im Fehler-Modal — lucide-Icon je Fehlerkategorie.
const ERROR_CATEGORY_ICON: Record<ErrorCategory, React.ReactNode> = {
  memory:       <MemoryStick className="w-7 h-7 text-red-400" />,
  cuda:         <Zap className="w-7 h-7 text-amber-400" />,
  dataset:      <Database className="w-7 h-7 text-blue-400" />,
  packages:     <Package className="w-7 h-7 text-purple-400" />,
  config:       <BarChart3 className="w-7 h-7 text-orange-400" />,
  code:         <Bug className="w-7 h-7 text-pink-400" />,
  labels:       <Database className="w-7 h-7 text-cyan-400" />,
  architecture: <Package className="w-7 h-7 text-violet-400" />,
  network:      <Globe className="w-7 h-7 text-sky-400" />,
  unknown:      <HelpCircle className="w-7 h-7 text-gray-400" />,
};

// ── Error Modal (Dev Train) ───────────────────────────────────────────────

interface DevTrainErrorModalProps {
  isOpen: boolean;
  errorTitle: string;
  errorMessage: string;
  errorDetails: string;
  script: string;
  output: string;
  onClose: () => void;
  onSendToFrameTrain: () => void;
  onSendToAI: (errorContext: string) => void;
  isSending?: boolean;
}

function DevTrainErrorModal({
  isOpen,
  errorTitle,
  errorMessage,
  errorDetails,
  script,
  output,
  onClose,
  onSendToFrameTrain,
  onSendToAI,
  isSending,
}: DevTrainErrorModalProps) {
  const { t, language } = useLanguage();
  useEscapeKey(onClose, isOpen);
  const analysis = analyzeError(errorMessage);
  const [copied, setCopied] = useState(false);
  const [sent, setSent] = useState(false);

  if (!isOpen) return null;

  const categoryKey = `devTrainPanel.errorModal.errorCategories.${analysis.category}`;
  const analysisTitle = t(`${categoryKey}.title`);
  const analysisHint = t(`${categoryKey}.hint`);

  const errorContext = `[Dev Train Fehler]\n\nTitel: ${errorTitle}\n\nFehler: ${errorMessage}\n\nDetails: ${errorDetails}\n\nSkript:\n${script}\n\nAusgabe/Logs:\n${output}`;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-9999 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-red-500/10 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0">{ERROR_CATEGORY_ICON[analysis.category]}</div>
            <div>
              <h2 className="text-lg font-bold text-white">{t('devTrainPanel.errorModal.title')}</h2>
              <p className="text-sm text-red-300">{analysisTitle}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Hint */}
          <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
            <p className="text-sm text-blue-300"><strong>{t('devTrainPanel.errorModal.hintLabel')}</strong> {analysisHint}</p>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-2">{t('devTrainPanel.errorModal.errorLabel')}</p>
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg overflow-x-auto max-h-24">
                <pre className="text-xs text-red-300 font-mono whitespace-pre-wrap break-words">{errorMessage}</pre>
              </div>
            </div>
          )}

          {/* Details */}
          {errorDetails && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-2">{t('devTrainPanel.errorModal.detailsLabel')}</p>
              <div className="p-3 bg-white/5 border border-white/10 rounded-lg overflow-x-auto max-h-24">
                <pre className="text-xs text-gray-400 font-mono whitespace-pre-wrap break-words">{errorDetails}</pre>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="px-6 py-4 border-t border-white/10 flex gap-3 flex-shrink-0">
          <button
            onClick={() => {
              navigator.clipboard.writeText(errorContext);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            {copied ? t('devTrainPanel.errorModal.copied') : t('devTrainPanel.errorModal.copyButton')}
          </button>

          <button
            onClick={onSendToFrameTrain}
            disabled={isSending || sent}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm border transition-all ${
              sent 
                ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                : 'bg-red-500/20 hover:bg-red-500/30 border-red-500/30 text-red-300 disabled:opacity-50'
            }`}
          >
            {isSending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {t('devTrainPanel.errorModal.sendingButton')}
              </>
            ) : sent ? (
              <>
                <Check className="w-4 h-4" />
                {t('devTrainPanel.errorModal.sentButton')}
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4" />
                {t('devTrainPanel.errorModal.sendToFrameTrainButton')}
              </>
            )}
          </button>

          <button
            onClick={() => onSendToAI(errorContext)}
            className="flex items-center gap-2 px-4 py-2 bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 rounded-lg text-sm text-violet-300 transition-all"
          >
            <Sparkles className="w-4 h-4" />
            {t('devTrainPanel.errorModal.sendToAIButton')}
          </button>

          <button
            onClick={onClose}
            className="ml-auto px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all"
          >
            {t('devTrainPanel.errorModal.closeButton')}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Save Name Dialog Modal ────────────────────────────────────────────────

function SaveNameDialog({ isOpen, defaultName, onSave, onClose }: { isOpen: boolean; defaultName: string; onSave: (name: string) => void; onClose: () => void; }) {
  const { t, language } = useLanguage();
  useEscapeKey(onClose, isOpen);
  const [name, setName] = useState(defaultName);

  useEffect(() => {
    setName(defaultName);
  }, [defaultName]);

  if (!isOpen) return null;

  const handleSave = () => {
    if (!name.trim()) return;
    onSave(name.trim());
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
          <div className="flex items-center gap-2">
            <Save className="w-5 h-5 text-amber-400" />
            <h2 className="text-lg font-bold text-white">{t('devTrainPanel.saveDialog.title')}</h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-300 text-sm">{t('devTrainPanel.saveDialog.description')}</p>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSave()}
            placeholder={t('devTrainPanel.saveDialog.placeholder')}
            autoFocus
            className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40"
          />
        </div>
        <div className="px-6 pb-6 flex gap-2">
          <button
            onClick={handleSave}
            disabled={!name.trim()}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40 transition-all"
          >
            <Save className="w-4 h-4" /> {t('devTrainPanel.saveDialog.saveButton')}
          </button>
          <button
            onClick={onClose}
            className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white text-sm font-medium transition-all"
          >
            {t('devTrainPanel.saveDialog.cancelButton')}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Script Library Modal ──────────────────────────────────────────────────

function ScriptLibraryModal({ currentScript, onLoad, onClose, userId }: { currentScript: string; onLoad: (s: SavedScript) => void; onClose: () => void; userId?: string; }) {
  const { t, language } = useLanguage();
  useEscapeKey(onClose);
  const [scripts, setScripts]       = useState<SavedScript[]>([]);
  const [saveName, setSaveName]     = useState('');
  const [showSaveForm, setShowForm] = useState(false);
  const { success } = useNotification();

  useEffect(() => { setScripts(loadScripts(userId)); }, [userId]);

  const handleSave = () => {
    if (!saveName.trim()) return;
    saveScript(saveName.trim(), currentScript, userId);
    setScripts(loadScripts(userId));
    setSaveName(''); setShowForm(false);
    success(t('devTrainPanel.library.savedTitle'), t('devTrainPanel.library.savedDetail').replace('{name}', saveName));
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FolderClosed className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">{t('devTrainPanel.library.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {scripts.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FileText className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">{t('devTrainPanel.library.empty')}</p>
            </div>
          ) : scripts.map(s => (
            <div key={s.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium text-sm truncate">{s.name}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{new Date(s.savedAt).toLocaleDateString(dateLocale(language), { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' })}</p>
                  <pre className="text-gray-600 text-[10px] mt-1.5 font-mono truncate">{s.script.split('\n').slice(0, 2).join(' · ')}</pre>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                  <button onClick={() => { deleteScript(s.id, userId); setScripts(loadScripts(userId)); }} className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>
                  <button onClick={() => { onLoad(s); onClose(); }} className="px-3 py-1.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-xs font-medium transition-all">{t('devTrainPanel.library.loadButton')}</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 pb-5 border-t border-white/10 pt-4 flex-shrink-0">
          {showSaveForm ? (
            <div className="flex gap-2">
              <input value={saveName} onChange={e => setSaveName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSave()} placeholder={t('devTrainPanel.library.namePlaceholder')} autoFocus
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40" />
              <button onClick={handleSave} disabled={!saveName.trim()} className="px-4 py-2 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40"><Save className="w-4 h-4" /></button>
              <button onClick={() => setShowForm(false)} className="px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-400 text-sm"><X className="w-4 h-4" /></button>
            </div>
          ) : (
            <button onClick={() => setShowForm(true)} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-300 text-sm font-medium transition-all">
              <Save className="w-4 h-4" /> {t('devTrainPanel.library.saveCurrentButton')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Code AI Sidebar (mit Edit-Skill) ─────────────────────────────────────

interface AiMessage { role: 'user' | 'assistant'; content: string; edits?: CodeEdit[]; action?: AutoAction | null; }
interface AppliedEditInfo { messageId: number; editId: string; originalScript: string; }
interface ChatSession {
  id: string;
  title: string;
  messages: AiMessage[];
  createdAt: string;
  updatedAt: string;
}

const DEVTRAIN_SESSIONS_KEY = 'ft_devtrain_sessions';
const MAX_DEVTRAIN_SESSIONS = 15;
const MAX_SESSION_MESSAGES  = 30;
const SESSION_MAX_AGE_MS    = 12 * 60 * 60 * 1000; // 12h

function loadChatSessions(): ChatSession[] {
  try { return JSON.parse(localStorage.getItem(DEVTRAIN_SESSIONS_KEY) ?? '[]'); } catch { return []; }
}
function saveChatSessions(sessions: ChatSession[]) {
  localStorage.setItem(DEVTRAIN_SESSIONS_KEY, JSON.stringify(sessions.slice(0, MAX_DEVTRAIN_SESSIONS)));
}
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60_000)          return 'gerade eben';
  if (diff < 3_600_000)       return `vor ${Math.floor(diff / 60_000)} Min`;
  if (diff < 86_400_000)      return `vor ${Math.floor(diff / 3_600_000)} Std`;
  if (diff < 7 * 86_400_000)  return `vor ${Math.floor(diff / 86_400_000)} Tagen`;
  return new Date(iso).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit' });
}
function makeSessionTitle(firstUserMsg: string): string {
  return firstUserMsg.trim().slice(0, 42) + (firstUserMsg.trim().length > 42 ? '…' : '');
}

function escapeHtml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function highlightPythonToHtml(code: string) {
  // Lightweight highlighter: strings + comments first, then keywords/numbers in remaining code.
  const KEYWORDS = new Set([
    'False','None','True','and','as','assert','async','await','break','class','continue','def','del','elif','else','except','finally',
    'for','from','global','if','import','in','is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield',
  ]);
  const BUILTINS = new Set([
    'print','len','range','enumerate','zip','map','filter','list','dict','set','tuple','str','int','float','bool','open','sum','min','max',
    'sorted','any','all','isinstance','type','super','dir','vars','getattr','setattr','hasattr','Exception','ValueError','TypeError',
  ]);

  type Seg = { t: 'code' | 'str' | 'cmt'; v: string };
  const segs: Seg[] = [];
  let i = 0;
  let cur = '';
  let state: 'code' | 'str' | 'cmt' = 'code';
  let quote: "'" | '"' | "'''" | '"""' | null = null;

  const flush = () => {
    if (!cur) return;
    segs.push({ t: state, v: cur });
    cur = '';
  };

  while (i < code.length) {
    const ch = code[i];
    const next3 = code.slice(i, i + 3);

    if (state === 'code') {
      if (next3 === "'''" || next3 === '"""') {
        flush();
        state = 'str';
        quote = next3 as "'''" | '"""';
        cur += next3;
        i += 3;
        continue;
      }
      if (ch === "'" || ch === '"') {
        flush();
        state = 'str';
        quote = ch as "'" | '"';
        cur += ch;
        i += 1;
        continue;
      }
      if (ch === '#') {
        flush();
        state = 'cmt';
        quote = null;
        cur += ch;
        i += 1;
        continue;
      }
      cur += ch;
      i += 1;
      continue;
    }

    if (state === 'cmt') {
      cur += ch;
      i += 1;
      if (ch === '\n') {
        flush();
        state = 'code';
      }
      continue;
    }

    // string state
    cur += ch;
    i += 1;
    if (quote === "'" || quote === '"') {
      if (ch === '\\' && i < code.length) {
        cur += code[i];
        i += 1;
        continue;
      }
      if (ch === quote) {
        flush();
        state = 'code';
        quote = null;
      }
      continue;
    }
    if (quote === "'''" || quote === '"""') {
      if (code.slice(i - 1, i - 1 + 3) === quote) {
        cur += quote.slice(1);
        i += 2;
        flush();
        state = 'code';
        quote = null;
      }
    }
  }
  flush();

  /**
   * Wendet ein Replace nur auf Text AUSSERHALB bereits eingefügter HTML-Tags an.
   *
   * Ohne diesen Schutz lief der Identifier-Durchlauf über das eigene Markup:
   * In `<span class="tok-num">` steckt das Python-Keyword `class`, das prompt
   * ein zweites Mal umschlossen wurde. Das Ergebnis war kaputtes HTML, das im
   * Editor als literaler Text `class="tok-num">2` auftauchte und den Code
   * praktisch unlesbar machte.
   */
  const replaceOutsideTags = (
    input: string,
    pattern: RegExp,
    replacer: (...args: any[]) => string,
  ): string =>
    input
      .split(/(<[^>]*>)/g)
      .map(part => (part.startsWith('<') && part.endsWith('>') ? part : part.replace(pattern, replacer as any)))
      .join('');

  const highlightCode = (s: string) => {
    let out = escapeHtml(s);
    out = replaceOutsideTags(out, /\b\d+(\.\d+)?\b/g, (m: string) => `<span class="tok-num">${m}</span>`);
    out = replaceOutsideTags(out, /\b(def)\s+([A-Za-z_][A-Za-z0-9_]*)/g,
      (_m: string, kw: string, name: string) => `<span class="tok-kw">${kw}</span> <span class="tok-fn">${name}</span>`);
    out = replaceOutsideTags(out, /\b(class)\s+([A-Za-z_][A-Za-z0-9_]*)/g,
      (_m: string, kw: string, name: string) => `<span class="tok-kw">${kw}</span> <span class="tok-cl">${name}</span>`);
    out = replaceOutsideTags(out, /(^|\n)(\s*)(@[\w.]+)/g,
      (_m: string, pre: string, ws: string, dec: string) => `${pre}${ws}<span class="tok-de">${dec}</span>`);
    out = replaceOutsideTags(out, /\b([A-Za-z_][A-Za-z0-9_]*)\b/g, (_m: string, w: string) => {
      if (KEYWORDS.has(w)) return `<span class="tok-kw">${w}</span>`;
      if (BUILTINS.has(w)) return `<span class="tok-bi">${w}</span>`;
      return w;
    });
    return out;
  };

  const html = segs.map(seg => {
    if (seg.t === 'str') return `<span class="tok-str">${escapeHtml(seg.v)}</span>`;
    if (seg.t === 'cmt') return `<span class="tok-cmt">${escapeHtml(seg.v)}</span>`;
    return highlightCode(seg.v);
  }).join('');

  return html.endsWith('\n') ? html + ' ' : html;
}

function CodeAISidebar({ script, modelInfo, datasets, outputPath, onApplyEdit, onReplaceScript, onClose, initialInput, onHighlightLines, onClearHighlights }: {
  script: string;
  modelInfo: ModelInfo | null;
  datasets: DatasetInfo[];
  outputPath: string;
  onApplyEdit: (editedScript: string) => void;
  onReplaceScript: (code: string) => void;
  onClose: () => void;
  initialInput?: string;
  onHighlightLines?: (edits: CodeEdit[]) => void;
  onClearHighlights?: () => void;
}) {
  const { t, language } = useLanguage();
  const { settings: aiSettings } = useAISettings();
  const [messages, setMessages]  = useState<AiMessage[]>([]);
  const [input, setInput]        = useState('');
  const [loading, setLoading]    = useState(false);
  const [showDiffModal, setShowDiffModal] = useState(false);
  const [currentMessageWithEdits, setCurrentMessageWithEdits] = useState<AiMessage | null>(null);
  const [isApplyingEdits, setIsApplyingEdits] = useState(false);
  const [appliedEdits, setAppliedEdits] = useState<AppliedEditInfo[]>([]);
  // Text der letzten fehlgeschlagenen Anfrage — für "Erneut senden"
  const [retryText, setRetryText] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const lastPrefillRef = useRef<string>('');

  // ── Session State ──────────────────────────────────────────────
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showHistory, setShowHistory]           = useState(false);
  useEscapeKey(() => setShowHistory(false), showHistory);
  const [isReadonly, setIsReadonly]             = useState(false);
  const [sessionTitle, setSessionTitle]         = useState('');
  const currentSessionIdRef = useRef<string | null>(null);
  currentSessionIdRef.current = currentSessionId;

  // Init: lade oder erstelle Session beim Mount
  useEffect(() => {
    const sessions = loadChatSessions();
    const last = sessions[0];
    const tooOld  = last ? (Date.now() - new Date(last.updatedAt).getTime()) > SESSION_MAX_AGE_MS : false;
    const tooLong = last ? last.messages.length >= MAX_SESSION_MESSAGES : false;
    if (!last || tooOld || tooLong) {
      // Neue Session starten
      const id = `s_${Date.now()}`;
      const newSession: ChatSession = { id, title: t('devTrainPanel.aiSidebar.newChatButton'), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
      saveChatSessions([newSession, ...sessions]);
      setCurrentSessionId(id);
      setSessionTitle(t('devTrainPanel.aiSidebar.newChatButton'));
      setMessages([]);
    } else {
      setCurrentSessionId(last.id);
      setSessionTitle(last.title);
      setMessages(last.messages);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync: messages → aktuelle Session speichern
  useEffect(() => {
    const id = currentSessionIdRef.current;
    if (!id) return;
    const sessions = loadChatSessions();
    const idx = sessions.findIndex(s => s.id === id);
    if (idx < 0) return;
    if (messages.length === 0) {
      // Leere Session nicht im Verlauf behalten
      saveChatSessions(sessions.filter(s => s.id !== id));
      return;
    }
    sessions[idx] = { ...sessions[idx], messages, updatedAt: new Date().toISOString() };
    saveChatSessions(sessions);
  }, [messages]);

  const startNewSession = () => {
    const id = `s_${Date.now()}`;
      const newSession: ChatSession = { id, title: t('devTrainPanel.aiSidebar.newChatButton'), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    // Leere Sessions aufräumen bevor neue hinzugefügt wird
    const sessions = loadChatSessions().filter(s => s.messages.length > 0);
    saveChatSessions([newSession, ...sessions]);
    setCurrentSessionId(id);
      setSessionTitle(t('devTrainPanel.aiSidebar.newChatButton'));
    setMessages([]);
    setAppliedEdits([]);
    setCurrentMessageWithEdits(null);
    setIsReadonly(false);
    setShowHistory(false);
    setRetryText(null);
    onClearHighlights?.();
  };

  const switchToSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setSessionTitle(session.title);
    setMessages(session.messages);
    setAppliedEdits([]);
    setCurrentMessageWithEdits(null);
    setIsReadonly(true);
    setShowHistory(false);
    setRetryText(null);
    onClearHighlights?.();
  };

  const continueFromSession = (session: ChatSession) => {
    const id = `s_${Date.now()}`;
    const title = session.title + t('devTrainPanel.aiSidebar.continuedSuffix');
    const newSession: ChatSession = { id, title, messages: session.messages, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    const sessions = loadChatSessions();
    saveChatSessions([newSession, ...sessions]);
    setCurrentSessionId(id);
    setSessionTitle(title);
    setMessages(session.messages);
    setAppliedEdits([]);
    setCurrentMessageWithEdits(null);
    setIsReadonly(false);
    setShowHistory(false);
    onClearHighlights?.();
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const sessions = loadChatSessions().filter(s => s.id !== id);
    saveChatSessions(sessions);
    // Wenn aktuelle Session gelöscht wird → neue starten
    if (id === currentSessionIdRef.current) startNewSession();
    // History-Panel neu rendern durch force-update trick
    setShowHistory(false);
    setTimeout(() => setShowHistory(true), 0);
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => {
    if (!initialInput) return;
    if (initialInput === lastPrefillRef.current) return;
    lastPrefillRef.current = initialInput;
    setInput(initialInput);
  }, [initialInput]);

  // Highlight lines when a message with edits is added
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === 'assistant' && lastMsg?.edits && lastMsg.edits.length > 0) {
      onHighlightLines?.(lastMsg.edits);
    }
  }, [messages, onHighlightLines]);

  const modelPath   = modelInfo?.local_path || modelInfo?.source_path || modelInfo?.name || 'MODELL_PFAD';
  const dsRefs      = datasets.map((d, i) => `${i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`} = "${d.storage_path || d.name}" (${d.name})`);

  const baseSystemPrompt = `Du bist ein professioneller Code-Side-Assistant in FrameTrain (Dev Train).

ZIEL: Hilf dem User, das Skript schnell, korrekt und robust zu fixen/verbessern.

KONTEXT (lokal):
- MODEL_PATH = "${modelPath}"
${dsRefs.map(r => `- ${r}`).join('\n')}
- OUTPUT_PATH = "${outputPath}"

INSTALLIERTE PAKETE: torch, transformers, datasets, scikit-learn, numpy, accelerate, peft, bitsandbytes

AKTUELLER SCRIPT-INHALT:
\`\`\`python
${script}
\`\`\`

ANFORDERUNGEN:
- Antworte kurz, technisch präzise, ohne Floskeln.
- Wenn möglich: nutze mode="edit" mit ##EDIT_START## Blöcken.
- Wenn ein kompletter Rewrite klar besser ist: mode="rewrite" + kompletter \`\`\`python\`\`\` Block.
- Stelle Rückfragen nur wenn absolut nötig.`;

  const systemPrompt = buildAutoSystemPrompt(baseSystemPrompt);

  const suggestions = [
    t('devTrainPanel.aiSidebar.suggestions.fixError'),
    t('devTrainPanel.aiSidebar.suggestions.paths'),
    t('devTrainPanel.aiSidebar.suggestions.performance'),
  ];

  const send = async (retryText?: string) => {
    const isRetry = typeof retryText === 'string';
    const text = (isRetry ? retryText : input).trim();
    if (!text || loading || isReadonly) return;
    const userMsg: AiMessage = { role: 'user', content: text };
    // Session-Titel beim ersten User-Message setzen
    if (!isRetry && messages.length === 0) {
      const title = makeSessionTitle(text);
      setSessionTitle(title);
      const sessions = loadChatSessions();
      const idx = sessions.findIndex(s => s.id === currentSessionIdRef.current);
      if (idx >= 0) { sessions[idx].title = title; saveChatSessions(sessions); }
    }
    // Retry: letzte (Fehler-)Antwort entfernen — User-Nachricht ist schon im Verlauf
    const base = isRetry && messages[messages.length - 1]?.role === 'assistant'
      ? messages.slice(0, -1)
      : messages;
    const withUser = isRetry ? base : [...base, userMsg];
    setMessages(withUser);
    if (!isRetry) setInput('');
    setRetryText(null);
    setLoading(true);

    try {
      const history = withUser.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));
      const last = history.pop()!;
      const response = await callAI(aiSettings, systemPrompt, last.content, history, language);

      const { action, cleaned } = parseAutoAction(response);
      const inferredEdit = (action?.mode === 'edit') || cleaned.includes('##EDIT_START##');
      const edits = inferredEdit ? parseEdits(response) : [];
      const code = action?.mode === 'rewrite' ? (extractFullPythonCode(response) ?? null) : null;
      const finalContent = code ? [cleaned, '```python', code, '```'].join('\n') : cleaned;
      setMessages(m => [...m, { role: 'assistant', content: finalContent, edits, action }]);
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', content: `Fehler: ${String(err)}` }]);
      setRetryText(text);
    } finally { setLoading(false); }
  };

  const handleApplyEdit = (editId: string, updatedEdits: CodeEdit[]) => {
    if (!currentMessageWithEdits?.edits) return;
    const editIdx = currentMessageWithEdits.edits.findIndex(e => e.id === editId);
    if (editIdx < 0) return;
    
    const edit = updatedEdits[editIdx];
    const { result, success, strategy, confidence } = applyEdit(script, edit);
    
    if (success) {
      onApplyEdit(result);
      onClearHighlights?.();
      const messageIdx = messages.findIndex(m => m === currentMessageWithEdits);
      setAppliedEdits(prev => [...prev, { messageId: messageIdx, editId, originalScript: script }]);
      setMessages(m => m.map(mm => 
        mm === currentMessageWithEdits 
          ? { 
              ...mm,
              edits: updatedEdits.map((e, i) => 
                i === editIdx ? { ...e, applied: true, failed: false, strategy, confidence } : e
              ),
            } 
          : mm
      ));
      setCurrentMessageWithEdits(m => m ? { ...m, edits: updatedEdits } : null);
      
      // Close modal after 500ms if all edits applied
      setTimeout(() => {
        if (updatedEdits.every(e => e.applied || e.failed)) {
          setShowDiffModal(false);
          setCurrentMessageWithEdits(null);
          onClearHighlights?.();
        }
      }, 500);
    } else {
      setMessages(m => m.map(mm => 
        mm === currentMessageWithEdits 
          ? {
              ...mm,
              edits: updatedEdits.map((e, i) => i === editIdx ? { ...e, failed: true } : e),
            } 
          : mm
      ));
      setCurrentMessageWithEdits(m => m ? { ...m, edits: updatedEdits } : null);
    }
  };

  const handleApplyAllEdits = (updatedEdits: CodeEdit[], msgOverride?: AiMessage) => {
    const targetMsg = msgOverride ?? currentMessageWithEdits;
    if (!targetMsg?.edits) return;
    if (msgOverride) setCurrentMessageWithEdits(msgOverride);
    setIsApplyingEdits(true);
    try {
      const applied = applyAllEdits(script, updatedEdits);
      onApplyEdit(applied.result);
      onClearHighlights?.();
      const messageIdx = messages.findIndex(m => m === targetMsg);
      setAppliedEdits(prev => [...prev, ...updatedEdits.map(e => ({ messageId: messageIdx, editId: e.id, originalScript: script }))]);
      setMessages(m => m.map(mm =>
        mm === targetMsg
          ? {
              ...mm,
              edits: updatedEdits.map((e, i) => ({
                ...e,
                applied: applied.results[i]?.success,
                failed: !applied.results[i]?.success,
                strategy: applied.results[i]?.strategy,
                confidence: applied.results[i]?.confidence
              }))
            }
          : mm
      ));
      setCurrentMessageWithEdits(m => m ? { ...m, edits: updatedEdits } : null);

      setTimeout(() => {
        setShowDiffModal(false);
        setCurrentMessageWithEdits(null);
        onClearHighlights?.();
      }, 500);
    } finally {
      setIsApplyingEdits(false);
    }
  };

  const handleUndoEdit = (messageId: number, editId: string) => {
    const appliedEdit = appliedEdits.find(ae => ae.messageId === messageId && ae.editId === editId);
    if (appliedEdit) {
      onApplyEdit(appliedEdit.originalScript);
      onClearHighlights?.();
      setAppliedEdits(prev => prev.filter(ae => !(ae.messageId === messageId && ae.editId === editId)));

      setMessages(m => m.map((mm, idx) => 
        idx === messageId && mm.edits
          ? { ...mm, edits: mm.edits.map(e => e.id === editId ? { ...e, applied: false } : e) }
          : mm
      ));
    }
  };

  return (
    <>
      <div className="flex flex-col h-full bg-slate-950 overflow-hidden relative">
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <Bot className="w-4 h-4 text-violet-400 flex-shrink-0" />
            <span className="text-sm font-medium text-white">{t('devTrainPanel.aiSidebar.title')}</span>
            <span className="ml-1 px-2 py-0.5 rounded-md bg-purple-500/15 border border-purple-500/25 text-purple-200 text-[10px] font-medium flex-shrink-0">{t('devTrainPanel.aiSidebar.autoBadge')}</span>
          </div>
          <div className="flex items-center gap-0.5 flex-shrink-0">
            <button
              onClick={() => setShowHistory(v => !v)}
              title={t('devTrainPanel.aiSidebar.historyTooltip')}
              className={`p-1.5 rounded-lg transition-all ${
                showHistory
                  ? 'bg-violet-500/20 text-violet-300'
                  : 'hover:bg-white/5 text-gray-500 hover:text-white'
              }`}
            >
              <History className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={startNewSession}
              title={t('devTrainPanel.aiSidebar.newChatTooltip')}
              className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all"
            >
              <MessageSquarePlus className="w-3.5 h-3.5" />
            </button>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all ml-0.5">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Session-Titel Chip */}
        {sessionTitle && sessionTitle !== t('devTrainPanel.aiSidebar.newChatButton') && (
          <div className="px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.01] flex items-center gap-1.5">
            <span className="text-[9px] text-gray-600">↳</span>
            <span className="text-[10px] text-gray-500 truncate">{sessionTitle}</span>
            {isReadonly && (
              <span className="ml-auto flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400/80">{t('devTrainPanel.aiSidebar.readonlyBadge')}</span>
            )}
          </div>
        )}

        {/* History Panel */}
        {showHistory && (
          <div className="absolute inset-x-0 top-[41px] z-10 bg-slate-950 border-b border-white/10 flex flex-col shadow-xl" style={{ maxHeight: '60%', overflowY: 'auto' }}>
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
              <span className="text-[10px] font-medium text-gray-400">{t('devTrainPanel.aiSidebar.historyTitle')}</span>
              <button onClick={startNewSession} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 text-violet-300 text-[10px] transition-all">
                <MessageSquarePlus className="w-3 h-3" /> {t('devTrainPanel.aiSidebar.newChatButton')}
              </button>
            </div>
            <div className="overflow-y-auto flex-1">
              {loadChatSessions().length === 0 ? (
                <p className="text-center text-gray-600 text-[10px] py-6">{t('devTrainPanel.aiSidebar.emptyHistory')}</p>
              ) : loadChatSessions().map(session => {
                const isActive = session.id === currentSessionId;
                return (
                  <div
                    key={session.id}
                    onClick={() => switchToSession(session)}
                    className={`group flex items-start gap-2 px-3 py-2.5 border-b border-white/[0.04] cursor-pointer transition-all ${
                      isActive ? 'bg-violet-500/10' : 'hover:bg-white/[0.04]'
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <p className={`text-[11px] truncate font-medium ${
                        isActive ? 'text-violet-200' : 'text-gray-300'
                      }`}>
                        {session.title}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[9px] text-gray-600">{relativeTime(session.updatedAt)}</span>
                        <span className="text-[9px] text-gray-700">· {t('devTrainPanel.aiSidebar.messagesCount').replace('{count}', String(session.messages.length))}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                      {!isActive && session.messages.length > 0 && (
                        <button
                          onClick={e => { e.stopPropagation(); continueFromSession(session); }}
                          className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-all"
                        >
                          {t('devTrainPanel.aiSidebar.continueButton')}
                        </button>
                      )}
                      <button
                        onClick={e => deleteSession(session.id, e)}
                        className="p-0.5 rounded hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {messages.length === 0 && (
            <div className="py-6 space-y-3">
              <p className="text-gray-400 text-xs">{t('devTrainPanel.aiSidebar.emptySidebarHint')}</p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.map(s => (
                  <button key={s} onClick={() => setInput(s)}
                    className={`px-2.5 py-1 rounded-lg border text-[10px] transition-all ${
                      'bg-purple-500/10 border-purple-500/20 text-purple-200 hover:bg-purple-500/15'
                    }`}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-xs ${
                m.role === 'user'
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : (m.action?.mode === 'edit')
                    ? 'bg-amber-500/20 text-amber-400'
                    : (m.action?.mode === 'rewrite')
                      ? 'bg-purple-500/20 text-purple-300'
                      : 'bg-violet-500/20 text-violet-400'
              }`}>
                {m.role === 'user'
                  ? 'U'
                  : (m.action?.mode === 'edit')
                    ? <Wand2 className="w-3 h-3" />
                    : (m.action?.mode === 'rewrite')
                      ? <Sparkles className="w-3 h-3" />
                      : <Bot className="w-3 h-3" />
                }
              </div>
              <div className={`flex-1 max-w-[90%] flex flex-col gap-1.5 ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                {/* Render message parts */}
                {removeEditBlocks(m.content).split(/(```python[\s\S]*?```)/g).map((part, pi) => {
                  if (part.startsWith('```python')) {
                    const code = extractFullPythonCode(part) ?? part;
                    return (
                      <div key={pi} className="w-full rounded-xl overflow-hidden border border-white/10">
                        <div className="flex items-center justify-between px-3 py-1.5 bg-white/[0.03] border-b border-white/10">
                          <span className="text-[10px] text-gray-500 font-mono">{t('trainingPanel.requirements.python')}</span>
                          <button onClick={() => onReplaceScript(code)} className="text-[10px] px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-all">
                            {t('devTrainPanel.aiSidebar.replaceCodeButton')}
                          </button>
                        </div>
                        <pre className="p-3 text-[10px] font-mono text-gray-300 overflow-x-auto max-h-48 leading-relaxed">{code}</pre>
                      </div>
                    );
                  }
                  return part.trim() ? (
                    <div
                      key={pi}
                      className={`px-3 py-2 rounded-xl text-[11px] leading-relaxed whitespace-pre-wrap break-words ${
                        m.role === 'user'
                          ? 'bg-emerald-500/10 text-gray-200 border border-emerald-500/20'
                          : (m.action?.mode === 'edit')
                            ? 'bg-amber-500/[0.06] text-gray-300 border border-amber-500/15'
                            : (m.action?.mode === 'rewrite')
                              ? 'bg-purple-500/[0.08] text-gray-200 border border-purple-500/20'
                              : 'bg-white/[0.05] text-gray-300 border border-white/10'
                      }`}
                    >
                      {part.trim()}
                    </div>
                  ) : null;
                })}

                {/* Simplified Edit Indicator with Undo Support */}
                {m.edits && m.edits.length > 0 && (
                  <div className="w-full space-y-1.5">
                    {m.edits.map((edit, editIdx) => {
                      const messageIdx = messages.indexOf(m);
                      const isApplied = appliedEdits.some(ae => ae.messageId === messageIdx && ae.editId === edit.id);
                      return (
                        <button
                          key={edit.id}
                          onClick={() => {
                            if (!isApplied) {
                              setCurrentMessageWithEdits(m);
                              setShowDiffModal(true);
                            }
                          }}
                          className={`w-full text-left px-3 py-2 rounded-xl transition-all text-[11px] ${
                            isApplied
                              ? 'bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/15'
                              : 'bg-amber-500/10 border border-amber-500/20 hover:bg-amber-500/15'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className={`font-medium flex items-center gap-2 ${isApplied ? 'text-emerald-300' : 'text-amber-300'}`}>
                              {isApplied ? <Check className="w-3.5 h-3.5" /> : <Pencil className="w-3.5 h-3.5" />}
                              {t('devTrainPanel.aiSidebar.editChange').replace('{n}', String(editIdx + 1))}
                            </span>
                            {isApplied ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleUndoEdit(messageIdx, edit.id);
                                }}
                                className="text-emerald-400/70 hover:text-emerald-300 text-xs flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/[0.15] hover:bg-emerald-500/25 transition-all"
                              >
                                <span>{t('devTrainPanel.aiSidebar.undoButton')}</span>
                              </button>
                            ) : (
                              <span className="text-amber-400/70 text-xs">{t('devTrainPanel.aiSidebar.viewDiffLink')}</span>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 bg-purple-500/20 text-purple-300">
                <Sparkles className="w-3 h-3" />
              </div>
              <div className="px-3 py-2 rounded-xl bg-white/5 border border-white/10">
                <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
              </div>
            </div>
          )}
          {retryText && !loading && (
            <div className="pl-8">
              <button
                onClick={() => send(retryText)}
                className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-200 text-[10px] font-medium transition-all"
              >
                {t('aiCoach.retryButton')}
              </button>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <div className="p-3 border-t border-white/10 flex-shrink-0">
          {/* Edit Summary Bar */}
          {(() => {
            const latestEditMsg = [...messages].reverse().find(m => m.edits && m.edits.length > 0);
            const hasUnapplied = latestEditMsg?.edits?.some(e => !e.applied && !e.failed);
            if (!hasUnapplied) return null;
            const addedLines = latestEditMsg.edits.reduce((sum, e) => sum + (!e.applied && !e.failed ? e.replace.split('\n').length : 0), 0);
            const removedLines = latestEditMsg.edits.reduce((sum, e) => sum + (!e.applied && !e.failed ? e.find.split('\n').length : 0), 0);
            return (
              <div className="mb-3 rounded-xl bg-amber-500/10 border border-amber-500/20 overflow-hidden">
                <div className="flex items-center gap-2 px-3 pt-2 pb-1.5">
                  <span className="text-[10px] text-gray-500 shrink-0">{t('devTrainPanel.aiSidebar.editSummaryReady')}</span>
                  <span className="text-[10px] font-medium text-emerald-400 flex items-center gap-0.5">
                    <Plus className="w-3 h-3" />{addedLines}
                  </span>
                  <span className="text-[10px] text-gray-600">/</span>
                  <span className="text-[10px] font-medium text-red-400 flex items-center gap-0.5">
                    <Minus className="w-3 h-3" />{removedLines}
                  </span>
                </div>
                <div className="flex gap-1.5 px-3 pb-2">
                  <button
                    onClick={() => handleApplyAllEdits(latestEditMsg.edits, latestEditMsg)}
                    className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-[10px] font-medium transition-all"
                  >
                    <Check className="w-3 h-3" /> {t('devTrainPanel.aiSidebar.applyButton')}
                  </button>
                  <button
                    onClick={() => {
                      setCurrentMessageWithEdits(latestEditMsg);
                      setShowDiffModal(true);
                    }}
                    className="flex items-center justify-center px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-[10px] font-medium transition-all"
                  >
                    {t('devTrainPanel.aiSidebar.detailsButton')}
                  </button>
                </div>
              </div>
            );
          })()}
          
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-gray-600">
              {isReadonly ? t('devTrainPanel.aiSidebar.readonlyHint') : t('devTrainPanel.aiSidebar.sendHint')}
            </span>
            <span className="text-[10px] text-purple-300/70">{t('devTrainPanel.aiSidebar.autoBadge')}</span>
          </div>
          <div className="flex gap-2 items-end">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={isReadonly ? t('devTrainPanel.aiSidebar.readonlyPlaceholder') : t('devTrainPanel.aiSidebar.inputPlaceholder')}
              rows={2}
              disabled={isReadonly}
              className={`flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none transition-opacity ${
                isReadonly ? 'opacity-40 cursor-not-allowed' : ''
              }`}
            />
            <button onClick={() => send()} disabled={!input.trim() || loading || isReadonly}
              className="p-2.5 rounded-xl border transition-all disabled:opacity-40 bg-purple-500/20 hover:bg-purple-500/30 border-purple-500/30 text-purple-200">
              <Send className="w-4 h-4" />
            </button>
          </div>
          {isReadonly && (
            <button
              onClick={startNewSession}
              className="mt-2 w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 text-violet-300 text-[10px] font-medium transition-all"
            >
              <MessageSquarePlus className="w-3 h-3" /> {t('devTrainPanel.aiSidebar.startNewChatButton')}
            </button>
          )}
        </div>
      </div>

      {/* Diff Viewer Modal */}
      {showDiffModal && currentMessageWithEdits?.edits && (
        <DiffViewer
          edits={currentMessageWithEdits.edits}
          onApply={handleApplyEdit}
          onApplyAll={handleApplyAllEdits}
          onClose={() => {
            setShowDiffModal(false);
            setCurrentMessageWithEdits(null);
            onClearHighlights?.();
          }}
          isApplying={isApplyingEdits}
          onEditChange={(updatedEdits) => {
            setCurrentMessageWithEdits(m => m ? { ...m, edits: updatedEdits } : null);
          }}
        />
      )}
    </>
  );
}

// ── Dev Train Panel ───────────────────────────────────────────────────────

interface DevTrainPanelProps {
  modelInfo: ModelInfo | null;
  selectedVersionPath: string;
  datasets: DatasetInfo[];
  /** Im Dropdown gewähltes Dataset — bestimmt, was als DATASET_PATH ankommt. */
  selectedDatasetId?: string | null;
  onNavigateToAnalysis: (vid: string) => void;
  userData?: { userId: string; email: string; apiKey: string; password: string };
}

export default function DevTrainPanel({ modelInfo, selectedVersionPath, datasets: allDatasets, selectedDatasetId, onNavigateToAnalysis, userData }: DevTrainPanelProps) {
  // Das ausgewählte Dataset muss an Position 0 stehen: DATASET_PATH (ohne
  // Suffix) ist der Pfad, den Skripte und Template verwenden. Vorher wurde
  // stumm immer das erste Dataset der Liste übergeben — wer im Dropdown ein
  // anderes wählte, trainierte unbemerkt auf fremden Daten.
  const datasets = useMemo(() => {
    if (!selectedDatasetId) return allDatasets;
    const idx = allDatasets.findIndex(d => d.id === selectedDatasetId);
    if (idx <= 0) return allDatasets;
    return [allDatasets[idx], ...allDatasets.filter((_, i) => i !== idx)];
  }, [allDatasets, selectedDatasetId]);

  // Globale Legacy-Scripts einmalig in den User-Key übernehmen
  useEffect(() => { migrateLegacyDevScripts(userData?.userId); }, [userData?.userId]);
  const { currentTheme } = useTheme();
  const { success, error }      = useNotification();
  const { settings: aiSettings } = useAISettings();
  const { setCurrentPageContent } = usePageContext();
  const { t, language } = useLanguage();

  const [fileOpen, setFileOpen]   = useState(false);
  const [tlHovered, setTlHovered] = useState(false);
  const [script, setScript]       = useState('');
  const [savedScript, setSavedScript] = useState('');
  const [isDirty, setIsDirty]     = useState(false);
  const [currentScriptId, setCurrentScriptId] = useState<string | null>(null);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveName, setSaveName]   = useState('');
  const [showAI, setShowAI]       = useState(false);
  const [showLibrary, setShowLib] = useState(false);
  const [showOpenLib, setShowOpenLib] = useState(false);
  const [running, setRunning]     = useState(false);
  const [output, setOutput]       = useState('');
  const [lossPoints, setLoss]     = useState<LossPoint[]>([]);
  const [currentJob, setJob]      = useState<TrainingJob | null>(null);
  const [showDashboard, setShowDashboard]     = useState(false);
  const [isDashMinimized, setIsDashMinimized] = useState(false);
  const devSessionIdRef   = useRef<string>('');
  const devStartedAtRef   = useRef<number>(0);
  const [editorH, setEditorH]     = useState(500);
  const [expanded, setExpanded]   = useState(false);
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem('devTrainBannerDismissed') === 'true';
    } catch {
      return false;
    }
  });
  const [showPathsModal, setShowPathsModal] = useState(false);
  const [outputPath, setOutputPath] = useState('[AppData]/training_outputs/dev_<job_id>');
  const outputRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const editorPreRef = useRef<HTMLPreElement>(null);
  const gutterInnerRef = useRef<HTMLDivElement>(null);
  const [activeLine, setActiveLine] = useState(1);
  const [editorScrollTop, setEditorScrollTop] = useState(0);
  const [editorScrollLeft, setEditorScrollLeft] = useState(0);
  const [editorLineHeightPx, setEditorLineHeightPx] = useState(28);
  const [editorPadTopPx, setEditorPadTopPx] = useState(16);
  const [editorPadLeftPx, setEditorPadLeftPx] = useState(16);
  const [cursorX, setCursorX] = useState(0);
  const [cursorY, setCursorY] = useState(0);
  const [showCursorBlink, setShowCursorBlink] = useState(true);

  const [findOpen, setFindOpen] = useState(false);
  const [findQuery, setFindQuery] = useState('');
  const [findStatus, setFindStatus] = useState<{ current: number; total: number } | null>(null);
  const findInputRef = useRef<HTMLInputElement>(null);

  const [highlightedLines, setHighlightedLines] = useState<HighlightedLine[]>([]);

  // Error Modal States
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorTitle, setErrorTitle] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [errorDetails, setErrorDetails] = useState('');
  const [isSendingError, setIsSendingError] = useState(false);
  const [aiPrefill, setAiPrefill] = useState('');

  const lineCount = useMemo(() => Math.max(1, (script || '').split('\n').length), [script]);
  const highlightedHtml = useMemo(() => highlightPythonToHtml(script || ''), [script]);

  // ── AI Coach Page Context ──────────────────────────────────────────────────
  useEffect(() => {
    const dsRefs = datasets.map((d, i) => ({
      key:   i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`,
      value: d.storage_path || '',
      name:  d.name,
    }));

    const lines: string[] = [
      t('devTrainPanel.pageContext.title'),
      '',
      t('devTrainPanel.pageContext.purposeBody'),
      '',
      t('devTrainPanel.pageContext.currentStateTitle'),
      `${t('devTrainPanel.pageContext.statusLabel')}: ${running ? t('devTrainPanel.pageContext.statusRunning') : isDirty ? t('devTrainPanel.pageContext.statusDirty') : t('devTrainPanel.pageContext.statusReady')}`,
      `${t('devTrainPanel.pageContext.modelLabel')}: ${modelInfo?.name || t('devTrainPanel.pageContext.modelNotLoaded')}`,
      `${t('devTrainPanel.pageContext.scriptSizeLabel').replace('{lines}', String(lineCount)).replace('{chars}', String(script.length))}`,
      running ? t('devTrainPanel.pageContext.runtimeActive').replace('{lines}', String(output.split('\n').length)) : output ? t('devTrainPanel.pageContext.lastRun').replace('{lines}', String(output.split('\n').length)) : t('devTrainPanel.pageContext.noOutput'),
      currentJob ? t('devTrainPanel.pageContext.jobStatus').replace('{status}', currentJob.status).replace('{epoch}', String(currentJob.progress?.epoch ?? 0)).replace('{total}', String(currentJob.progress?.total_epochs ?? 0)) : '',
      '',
      t('devTrainPanel.pageContext.scriptStateTitle'),
      isDirty ? t('devTrainPanel.pageContext.unsaved') : t('devTrainPanel.pageContext.saved'),
      currentScriptId ? t('devTrainPanel.pageContext.scriptLoaded').replace('{id}', currentScriptId) : t('devTrainPanel.pageContext.newUnsaved'),
      running ? t('devTrainPanel.pageContext.running') : showDashboard ? t('devTrainPanel.pageContext.dashboardOpen') : t('devTrainPanel.pageContext.idle').replace('{count}', String(lossPoints.length)),
      '',
      t('devTrainPanel.pageContext.layoutTitle'),
      t('devTrainPanel.pageContext.topSection'),
      t('devTrainPanel.pageContext.topModel').replace('{name}', modelInfo?.name || t('devTrainPanel.pageContext.none')),
      t('devTrainPanel.pageContext.topVersionPath').replace('{status}', selectedVersionPath ? t('devTrainPanel.pageContext.loaded') : t('devTrainPanel.pageContext.notSet')),
      t('devTrainPanel.pageContext.topSave'),
      t('devTrainPanel.pageContext.topLibrary'),
      '',
      t('devTrainPanel.pageContext.leftSection'),
      t('devTrainPanel.pageContext.leftEditor'),
      t('devTrainPanel.pageContext.leftGutter'),
      t('devTrainPanel.pageContext.leftStart'),
      '',
      t('devTrainPanel.pageContext.rightSection'),
      t('devTrainPanel.pageContext.rightOutput'),
      running ? t('devTrainPanel.pageContext.rightLive') : t('devTrainPanel.pageContext.rightLast'),
      t('devTrainPanel.pageContext.rightButtons'),
      t('devTrainPanel.pageContext.rightDashboard'),
      '',
      t('devTrainPanel.pageContext.bottomSection'),
      t('devTrainPanel.pageContext.bottomChat'),
      t('devTrainPanel.pageContext.bottomDashboard'),
      '',
      t('devTrainPanel.pageContext.availableActionsTitle'),
      !script.trim() ? t('devTrainPanel.pageContext.step1Empty') : t('devTrainPanel.pageContext.step1Filled'),
      t('devTrainPanel.pageContext.step2'),
      t('devTrainPanel.pageContext.step3'),
      running ? t('devTrainPanel.pageContext.step4Running') : output ? t('devTrainPanel.pageContext.step4Error') : '',
      t('devTrainPanel.pageContext.step5'),
      '',
      t('devTrainPanel.pageContext.contextTitle'),
      t('devTrainPanel.pageContext.contextModel').replace('{name}', modelInfo?.name || t('devTrainPanel.pageContext.none')),
      t('devTrainPanel.pageContext.contextDatasets').replace('{count}', String(datasets.length)).replace('{names}', datasets.length > 0 ? datasets.map(d => d.name).join(', ') : t('devTrainPanel.pageContext.none')),
      t('devTrainPanel.pageContext.contextDatasetPaths').replace('{paths}', dsRefs.map(d => d.key).join(', ')),
      t('devTrainPanel.pageContext.contextOutput').replace('{path}', outputPath || '[AppData]/training_outputs/dev_<job_id>'),
      lossPoints.length > 0 ? t('devTrainPanel.pageContext.contextLoss').replace('{count}', String(lossPoints.length)) : '',
    ];

    setCurrentPageContent(lines.join('\n'), 'training-dev');
  }, [script, lineCount, running, isDirty, output, modelInfo, datasets, currentScriptId, outputPath, selectedVersionPath, currentJob, lossPoints, showDashboard, setCurrentPageContent]);

  const syncEditorScroll = () => {
    const ta = editorRef.current;
    if (!ta) return;
    setEditorScrollTop(ta.scrollTop);
    setEditorScrollLeft(ta.scrollLeft);
    if (editorPreRef.current) {
      editorPreRef.current.scrollTop = ta.scrollTop;
      editorPreRef.current.scrollLeft = ta.scrollLeft;
    }
    if (gutterInnerRef.current) {
      gutterInnerRef.current.style.transform = `translateY(-${ta.scrollTop}px)`;
    }
    updateActiveLine();
  };

  const updateActiveLine = () => {
    const ta = editorRef.current;
    if (!ta) return;
    const caret = ta.selectionStart ?? 0;
    const line = (ta.value.slice(0, caret).match(/\n/g)?.length ?? 0) + 1;
    setActiveLine(line);
    
    // Calculate custom cursor position
    // Get text before caret on current line
    const textBeforeCaret = ta.value.slice(0, caret);
    const lastNewlineIdx = textBeforeCaret.lastIndexOf('\n');
    const textOnLine = lastNewlineIdx === -1 ? textBeforeCaret : textBeforeCaret.slice(lastNewlineIdx + 1);
    
    // Measure text width with monospace font
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.font = `${parseFloat(window.getComputedStyle(ta).fontSize)}px JetBrains Mono, Fira Code, Cascadia Code, Courier New, monospace`;
      const metrics = ctx.measureText(textOnLine);
      setCursorX(editorPadLeftPx + metrics.width);
    }
    
    // Cursor Y position (use ta.scrollTop directly, not state)
    const cursorLine = (ta.value.slice(0, caret).match(/\n/g)?.length ?? 0) + 1;
    setCursorY(editorPadTopPx + (cursorLine - 1) * editorLineHeightPx - ta.scrollTop);
  };

  useEffect(() => {
    syncEditorScroll();
    updateActiveLine();
  }, [fileOpen, expanded, script]);

  useEffect(() => {
    const ta = editorRef.current;
    if (!ta) return;
    const cs = window.getComputedStyle(ta);
    const pt = parseFloat(cs.paddingTop || '0');
    const pl = parseFloat(cs.paddingLeft || '0');
    if (Number.isFinite(pt) && pt >= 0) setEditorPadTopPx(pt);
    if (Number.isFinite(pl) && pl >= 0) setEditorPadLeftPx(pl);

    // Keep highlight math and caret in sync by using the textarea's resolved line-height (numeric).
    // If line-height is "normal" (rare here), we keep the default 28px.
    const lh = parseFloat(cs.lineHeight || '');
    if (Number.isFinite(lh) && lh > 0) setEditorLineHeightPx(lh);
  }, [fileOpen, expanded]);

  const updateFindStatus = useCallback((query: string, cursorStart: number) => {
    const ta = editorRef.current;
    if (!ta || !query) { setFindStatus(null); return; }
    const text = ta.value;
    let total = 0;
    let idx = 0;
    while (true) {
      const at = text.indexOf(query, idx);
      if (at === -1) break;
      total += 1;
      idx = at + Math.max(1, query.length);
    }
    if (total === 0) { setFindStatus({ current: 0, total: 0 }); return; }
    // current = first match at/after cursorStart
    let current = 1;
    idx = 0;
    while (true) {
      const at = text.indexOf(query, idx);
      if (at === -1) break;
      if (at >= cursorStart) break;
      current += 1;
      idx = at + Math.max(1, query.length);
    }
    if (current > total) current = total;
    setFindStatus({ current, total });
  }, []);

  const findNext = useCallback((dir: 1 | -1) => {
    const ta = editorRef.current;
    if (!ta) return;
    const q = findQuery;
    if (!q) { setFindStatus(null); return; }
    const text = ta.value;
    const start = ta.selectionStart ?? 0;
    const end = ta.selectionEnd ?? start;
    let at = -1;
    if (dir === 1) {
      at = text.indexOf(q, end);
      if (at === -1) at = text.indexOf(q, 0);
    } else {
      at = text.lastIndexOf(q, Math.max(0, start - 1));
      if (at === -1) at = text.lastIndexOf(q);
    }
    if (at === -1) { setFindStatus({ current: 0, total: 0 }); return; }
    ta.focus();
    ta.setSelectionRange(at, at + q.length);
    updateActiveLine();
    // scroll into view (approx)
    const line = (text.slice(0, at).match(/\n/g)?.length ?? 0) + 1;
    const targetTop = (line - 1) * editorLineHeightPx;
    const pad = editorPadTopPx;
    if (ta.scrollTop > targetTop) ta.scrollTop = Math.max(0, targetTop - pad);
    else if (ta.scrollTop + ta.clientHeight < targetTop + editorLineHeightPx + pad) ta.scrollTop = Math.max(0, targetTop - pad);
    syncEditorScroll();
    updateFindStatus(q, at);
  }, [findQuery, editorLineHeightPx, editorPadTopPx, syncEditorScroll, updateActiveLine, updateFindStatus]);

  // Echte Pfade
  const modelPath  = selectedVersionPath || modelInfo?.local_path || modelInfo?.source_path || modelInfo?.name || '';
  const dsRefs     = datasets.map((d, i) => ({
    key:   i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`,
    value: d.storage_path || '',
    name:  d.name,
  }));

  // Output-Pfad aus App-Data-Dir laden
  useEffect(() => {
    invoke<string>('get_app_data_dir')
      .then(dir => setOutputPath(`${dir}/training_outputs/dev_<job_id>`))
      .catch(() => setOutputPath('[AppData]/training_outputs/dev_<job_id>'));
  }, []);

  // Neue Datei erstellen
  const handleNewFile = () => {
    setScript('');
    setSavedScript('');
    setCurrentScriptId(null);
    setIsDirty(false);
    setFileOpen(true);
  };

  // Datei schließen (nur wenn gespeichert)
  const handleCloseFile = () => {
    if (isDirty) {
      error('Ungespeicherte Änderungen', 'Bitte erst speichern (⌘S) bevor du die Datei schließt.');
      return;
    }
    setFileOpen(false);
    setScript('');
    setSavedScript('');
    setCurrentScriptId(null);
    setExpanded(false);
  };

  // Template generieren
  const generateTemplate = () => {
    if (!modelInfo || !outputPath) return;
    const template = generateDefaultScript(modelInfo, datasets, outputPath);
    setScript(template);
    setIsDirty(true);
  };

  // Save-Funktion
  const handleSave = () => {
    if (currentScriptId) {
      // Skript existiert bereits in der Bibliothek → Update
      updateScript(currentScriptId, script, userData?.userId);
      setSavedScript(script);
      setIsDirty(false);
      success(t('devTrainPanel.notifications.savedUpdated'), t('devTrainPanel.notifications.savedUpdatedDetail'));
    } else {
      // Skript ist neu → Dialog für Namen zeigen
      setSaveName('Mein Trainings-Skript');
      setShowSaveDialog(true);
    }
  };

  const handleSaveWithName = (name: string) => {
    if (!name.trim()) return;
    saveScript(name.trim(), script, userData?.userId);
    const allScripts = loadScripts(userData?.userId);
    const newScript = allScripts[0];
    if (newScript) {
      setCurrentScriptId(newScript.id);
    }
    setSavedScript(script);
    setIsDirty(false);
    setShowSaveDialog(false);
    setSaveName('');
    success(t('devTrainPanel.notifications.savedTitle'), t('devTrainPanel.notifications.savedDetail').replace('{name}', name));
  };

  // Keyboard shortcuts: Cmd+S / Ctrl+S zum Speichern
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'f') {
        const ta = editorRef.current;
        const isEditorFocused = !!ta && document.activeElement === ta;
        if (fileOpen && isEditorFocused) {
          e.preventDefault();
          setFindOpen(true);
          setTimeout(() => findInputRef.current?.focus(), 0);
          const cursor = ta.selectionStart ?? 0;
          updateFindStatus(findQuery, cursor);
        }
      }
      if (e.key === 'Escape' && findOpen) {
        setFindOpen(false);
        setFindStatus(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fileOpen, findOpen, findQuery, savedScript, script, currentScriptId, updateFindStatus]);

  // Warnung bei ungespeicherten Änderungen beim Verlassen
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  // Training-Events
  useEffect(() => {
    let u1: (() => void) | undefined, u2: (() => void) | undefined, u3: (() => void) | undefined, u4: (() => void) | undefined;

    // Watchdog löschen sobald das Backend lebt — sonst würde ein Training,
    // das länger als 5 Minuten läuft, fälschlich als "failed" markiert.
    const clearWatchdog = () => {
      if ((window as any).__devTrainingTimeout) {
        clearTimeout((window as any).__devTrainingTimeout);
        delete (window as any).__devTrainingTimeout;
      }
    };

    // Normale Trainings (TrainingPanel) emittieren dieselben Event-Namen —
    // hier nur auf die eigenen Dev-Jobs (job_id "dev_…") reagieren.
    const isDevJob = (jobId?: string) => jobId?.startsWith('dev_') ?? false;

    listen<{ job_id?: string; data: Partial<TrainingProgress> }>('training-progress', e => {
      if (!isDevJob(e.payload.job_id)) return;
      clearWatchdog();
      const d = e.payload.data;
      if (d.train_loss != null) setLoss(pts => [...pts, { step: d.step ?? pts.length, epoch: d.epoch ?? 0, train_loss: d.train_loss!, val_loss: d.val_loss ?? undefined }]);
      setJob(j => j ? { ...j, status: 'running', progress: { ...j.progress, ...d } as TrainingProgress } : null);
    }).then(fn => { u1 = fn; });

    listen<{ line: string }>('dev-training-output', e => {
      clearWatchdog();
      setOutput(o => o + e.payload.line + '\n');
      setTimeout(() => outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight }), 50);
    }).then(fn => { u2 = fn; });

    listen<{ job_id?: string }>('training-complete', e => {
      if (!isDevJob(e.payload?.job_id)) return;
      // Timeout bereinigen
      if ((window as any).__devTrainingTimeout) {
        clearTimeout((window as any).__devTrainingTimeout);
        delete (window as any).__devTrainingTimeout;
      }
      setRunning(false);
      setJob(j => j ? { ...j, status: 'completed' } : null);
      setOutput(o => o + '\n' + t('devTrainPanel.progress.trainingComplete'));
      invoke('disable_prevent_sleep').catch(() => {});
    }).then(fn => { u3 = fn; });

    listen<{ job_id?: string; data?: { error?: string; details?: string } }>('training-error', e => {
      if (!isDevJob(e.payload.job_id)) return;
      const d = e.payload.data;
      // Timeout bereinigen
      if ((window as any).__devTrainingTimeout) {
        clearTimeout((window as any).__devTrainingTimeout);
        delete (window as any).__devTrainingTimeout;
      }
      setRunning(false);
      setJob(j => j ? { ...j, status: 'failed', error: d?.error ?? 'Fehler' } : null);
      setOutput(o => o + '\n[Fehler] ' + (d?.error ?? 'Fehler') + (d?.details ? '\n' + d.details : ''));
      invoke('disable_prevent_sleep').catch(() => {});
      
      // Error-Modal öffnen
      setErrorTitle(d?.error ?? 'Training Fehler');
      setErrorMessage(d?.error ?? 'Unbekannter Fehler');
      setErrorDetails(d?.details ?? '');
      setShowErrorModal(true);
    }).then(fn => { u4 = fn; });

    return () => { u1?.(); u2?.(); u3?.(); u4?.(); };
  }, []);

  // Persist banner dismissed state in sessionStorage
  useEffect(() => {
    try {
      sessionStorage.setItem('devTrainBannerDismissed', String(dismissed));
    } catch {
      // sessionStorage not available, ignore
    }
  }, [dismissed]);

  const handleStart = async () => {
    if (!script.trim() || !modelInfo) { error('Fehler', 'Kein Modell ausgewählt oder Skript leer.'); return; }

    // Ungespeicherte Änderungen automatisch sichern, statt den Start
    // abzulehnen. Die alte Variante warf nur einen Toast — wer ihn verpasste,
    // sah scheinbar gar keine Reaktion und rätselte, warum sich nichts tut.
    if (isDirty) {
      if (currentScriptId) {
        updateScript(currentScriptId, script, userData?.userId);
      } else {
        const autoName = `Dev Train ${new Date().toLocaleString(language === 'de' ? 'de-DE' : 'en-US')}`;
        saveScript(autoName, script, userData?.userId);
        const newest = loadScripts(userData?.userId)[0];
        if (newest) setCurrentScriptId(newest.id);
      }
      setSavedScript(script);
      setIsDirty(false);
      success(t('devTrainPanel.notifications.autoSavedTitle'), t('devTrainPanel.notifications.autoSavedDetail'));
    }

    setRunning(true); setOutput(''); setLoss([]);

    const refs: Record<string, string> = {
      MODEL_PATH: modelPath,
      ...Object.fromEntries(dsRefs.map(r => [r.key, r.value])),
    };

    try {
      const job = await invoke<TrainingJob>('start_dev_training', {
        script,
        modelId:     modelInfo!.id,
        modelName:   modelInfo!.name,
        datasetId:   datasets[0]?.id ?? '',
        datasetName: datasets[0]?.name ?? '',
        refs,
      });
      setJob(job);
      setOutput(`${t('devTrainPanel.progress.trainingStarted').replace('{id}', job.id)}\n`);
      invoke('enable_prevent_sleep').catch(() => {});
      devSessionIdRef.current = job.id;
      devStartedAtRef.current = Date.now();
      setShowDashboard(true);
      setIsDashMinimized(false);
      success(t('devTrainPanel.notifications.startSuccess'), t('devTrainPanel.notifications.startSuccessDetail'));
      
      // Timeout: Wenn nach 5 Minuten kein Event kommt, als fehlgeschlagen markieren
      const timeoutId = setTimeout(() => {
        setJob(j => {
          if (j && (j.status === 'pending' || j.status === 'running')) {
            setRunning(false);
            error(t('devTrainPanel.notifications.timeout'), t('devTrainPanel.notifications.timeoutDetail'));
            return { ...j, status: 'failed', error: 'Timeout: Kein Response vom Backend' };
          }
          return j;
        });
      }, 5 * 60 * 1000);

      // Cleanup-Funktion speichern (optional)
      (window as any).__devTrainingTimeout = timeoutId;
    } catch (err: unknown) {
      setOutput(`${t('common.error')}: ${String(err)}`);
      setRunning(false);
      setJob(null);
      error(t('common.error'), String(err));
    }
  };

  const handleStop = async () => {
    try {
      // Dev-Training hat einen eigenen Stop-Command (stop_training kennt den Dev-Prozess nicht)
      await invoke('stop_dev_training');
    } catch { /* ignore */ }
    invoke('disable_prevent_sleep').catch(() => {});
    setRunning(false);
    // Status "stopped" statt null — so zeigt das Dashboard sauber "Gestoppt"
    // an, statt in einen leeren Zustand zu springen
    setJob(j => (j ? { ...j, status: 'stopped' } : null));
    setOutput(o => o + '\n' + t('devTrainPanel.progress.trainingStopped'));
    
    // Clear timeout wenn vorhanden
    if ((window as any).__devTrainingTimeout) {
      clearTimeout((window as any).__devTrainingTimeout);
      delete (window as any).__devTrainingTimeout;
    }
  };

  // Error Report Funktionen
  const handleSendToFrameTrain = async () => {
    setIsSendingError(true);
    try {
      const analysis = analyzeError(errorMessage);
      const categoryKey = `devTrainPanel.errorModal.errorCategories.${analysis.category}`;
      const ok = await sendAppErrorReport({
        error_type: `devtrain:${analysis.category}`,
        title: errorTitle || 'Dev Train Fehler',
        message: errorMessage,
        details: errorDetails,
        logs: output,           // Alle Output-Logs
        script_full: script,    // Komplettes Skript
        error_analysis: analysis.category,
        error_category: t(`${categoryKey}.title`),
      });

      if (ok) {
        success('Gesendet!', 'Fehler wurde an FrameTrain Team gesendet. Danke!');
        // Modal bleibt offen - Nutzer kann weiter Optionen nutzen
      } else {
        error('Fehler', 'Konnte Fehler nicht senden. Prüfe deine Internetverbindung.');
      }
    } catch (err) {
      error('Fehler', 'Netzwerkfehler: ' + String(err));
    } finally {
      setIsSendingError(false);
    }
  };

  const handleSendToAI = (errorContext: string) => {
    setShowErrorModal(false);
    setShowAI(true);
    setAiPrefill(
      `[Dev Train Fehler]\n\n` +
      `Bitte hilf mir, meinen Dev-Train Run zu reparieren.\n\n` +
      `FEHLER:\n${errorContext}\n\n` +
      `WICHTIG: DATASET_PATH ist ein lokaler Ordner von FrameTrain. ` +
      `Falls ich fälschlich load_dataset(DATASET_PATH) benutze, schlage eine robuste Alternative vor (z.B. load_from_disk oder passende load_dataset(..., data_files=...)).\n\n` +
      `Du darfst Edits vorschlagen oder den Code neu schreiben – wähle selbst die beste Vorgehensweise.`
    );
  };

  const isRunning = running || currentJob?.status === 'running' || currentJob?.status === 'pending';
  const progress  = currentJob?.progress;

  return (
    <div className={`flex gap-0 ${expanded ? 'fixed inset-0 z-40 bg-slate-950 p-4' : ''}`}>
      <div className={`flex-1 space-y-4 ${expanded ? 'overflow-y-auto pr-2' : ''}`}>

        {/* Quick actions */}
        {/* Copy paths button removed - copy buttons are now on individual paths */}

        {/* Info Banner */}
        {!dismissed && (
          <div className="p-4 rounded-2xl border border-blue-500/30 bg-blue-500/10">
            <div className="flex items-start justify-between gap-2 mb-1">
              <div className="flex items-center gap-2"><Terminal className="w-4 h-4 text-blue-400" /><span className="text-blue-300 font-semibold text-sm">{t('devTrainPanel.banner.title')}</span></div>
              <button onClick={() => { setDismissed(true); localStorage.setItem('devTrainBannerDismissed', 'true'); }} className="p-1 rounded-lg hover:bg-white/10 text-blue-400/60 hover:text-white transition-all"><X className="w-3.5 h-3.5" /></button>
            </div>
            <p className="text-gray-400 text-xs">{t('devTrainPanel.banner.description')}</p>
          </div>
        )}

        {/* Model + Dataset Row */}
        <button
          onClick={() => setShowPathsModal(!showPathsModal)}
          className="w-full px-4 py-3 rounded-2xl border border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/15 transition-all flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-blue-300">{t('devTrainPanel.paths.toggleLabel')}</span>
          </div>
          <div className={`transform transition-transform ${showPathsModal ? 'rotate-180' : ''}`}>
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </button>

        {/* Paths Modal — Collapsible */}
        {showPathsModal && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-6">
            {/* Model Block */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-white">{t('devTrainPanel.paths.modelTitle')}</span>
              </div>
              <RefRow color="text-emerald-400" label={t('devTrainPanel.paths.modelPathLabel')} value={modelPath} hint={modelInfo?.name} />
              
              <div className="flex items-center gap-2 mb-1 mt-3">
                <span className="text-sm font-medium text-white">{t('devTrainPanel.paths.versionTitle')}</span>
              </div>
              <div className="text-[11px] font-mono text-gray-400">
                {modelInfo?.name ? `${modelInfo.name}` : t('devTrainPanel.paths.versionFallback')}
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-white/10" />

            {/* Dataset Block */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-white">{t('devTrainPanel.paths.datasetTitle')}</span>
              </div>
              {dsRefs.map((r, idx) => (
                <RefRow
                  key={r.key}
                  color="text-blue-400"
                  label={t('devTrainPanel.paths.datasetPathLabel') + (datasets.length > 1 ? ` ${idx + 1}` : '')}
                  value={r.value}
                  hint={r.name}
                />
              ))}
            </div>

            {/* Divider */}
            <div className="border-t border-white/10" />

            {/* Output Path */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">{t('devTrainPanel.paths.outputTitle')}</span>
              </div>
              <RefRow
                color="text-purple-400"
                label={t('devTrainPanel.paths.outputPathLabel')}
                value={outputPath.replace('<job_id>', 'dev_train')}
              />
            </div>
          </div>
        )}

        {/* Code Editor */}
        <div className={`rounded-2xl border border-white/10 overflow-hidden ${expanded ? 'flex-1 flex flex-col' : ''}`}>
          {/* Toolbar — always visible */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-slate-900">
            <div className="flex items-center gap-3 min-w-0 shrink">
              <div
                className="flex gap-1.5"
                onMouseEnter={() => setTlHovered(true)}
                onMouseLeave={() => setTlHovered(false)}
              >
                {/* Rot: Datei schließen */}
                <button
                  onClick={fileOpen ? handleCloseFile : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${
                    fileOpen ? 'bg-red-500 cursor-pointer hover:bg-red-400' : 'bg-red-500/40 cursor-default'
                  }`}
                  title={fileOpen ? (isDirty ? t('devTrainPanel.toolbar.unsavedChangesTooltip') : t('devTrainPanel.toolbar.closeFileTooltip')) : ''}
                >
                  {tlHovered && fileOpen && (
                    <X className="w-[7px] h-[7px] text-red-900 stroke-[3]" />
                  )}
                  {!tlHovered && isDirty && fileOpen && (
                    <div className="w-[5px] h-[5px] rounded-full bg-red-900" />
                  )}
                </button>

                {/* Gelb: Speichern */}
                <button
                  onClick={fileOpen && isDirty ? handleSave : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${
                    fileOpen && isDirty ? 'bg-amber-400 cursor-pointer hover:bg-amber-300' : 'bg-amber-500/40 cursor-default'
                  }`}
                  title={fileOpen && isDirty ? t('devTrainPanel.toolbar.saveTooltip') : ''}
                >
                  {tlHovered && fileOpen && isDirty && (
                    <Minus className="w-[7px] h-[7px] text-amber-900 stroke-[3]" />
                  )}
                </button>

                {/* Grün: Vergrößern / Verkleinern */}
                <button
                  onClick={fileOpen ? () => setExpanded(v => !v) : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${
                    fileOpen ? 'bg-emerald-500 cursor-pointer hover:bg-emerald-400' : 'bg-emerald-500/40 cursor-default'
                  }`}
                  title={fileOpen ? (expanded ? t('devTrainPanel.toolbar.minimizeTooltip') : t('devTrainPanel.toolbar.maximizeTooltip')) : ''}
                >
                  {tlHovered && fileOpen && (
                    expanded
                      ? <Minimize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" />
                      : <Maximize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" />
                  )}
                </button>
              </div>
              <div className="flex items-center gap-2">
                <FileCode className={`w-4 h-4 ${fileOpen ? 'text-emerald-400' : 'text-gray-600'}`} />
                <span className={`text-sm font-medium truncate max-w-[160px] ${fileOpen ? 'text-gray-300' : 'text-gray-600'}`}>
                  {fileOpen ? t('devTrainPanel.editor.fileName') : t('devTrainPanel.editor.noDocument')}
                </span>
              </div>
            </div>
            {fileOpen && findOpen && (
              <div className="flex items-center gap-2 px-2 py-1 rounded-xl bg-white/5 border border-white/10">
                <input
                  ref={findInputRef}
                  value={findQuery}
                  onChange={e => {
                    const q = e.target.value;
                    setFindQuery(q);
                    const ta = editorRef.current;
                    updateFindStatus(q, ta?.selectionStart ?? 0);
                  }}
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      findNext(e.shiftKey ? -1 : 1);
                    }
                    if (e.key === 'Escape') {
                      setFindOpen(false);
                      setFindStatus(null);
                      editorRef.current?.focus();
                    }
                  }}
                  placeholder={t('devTrainPanel.editor.searchPlaceholder')}
                  className="w-44 px-2 py-1 bg-transparent text-gray-200 text-xs focus:outline-none placeholder:text-gray-600"
                />
                <span className="text-[10px] text-gray-500 font-mono w-12 text-right">
                  {findStatus ? `${findStatus.current}/${findStatus.total}` : ''}
                </span>
                <button
                  onClick={() => findNext(-1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title={t('devTrainPanel.editor.prevSearchTooltip')}
                >
                  ↑
                </button>
                <button
                  onClick={() => findNext(1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title={t('devTrainPanel.editor.nextSearchTooltip')}
                >
                  ↓
                </button>
                <button
                  onClick={() => { setFindOpen(false); setFindStatus(null); editorRef.current?.focus(); }}
                  className="p-1 rounded-lg hover:bg-white/10 text-gray-500 hover:text-white transition-all"
                  title={t('devTrainPanel.editor.closeSearchTooltip')}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
            <div className="flex items-center gap-2 flex-shrink-0">
              {!fileOpen ? (
                // Welcome-Modus: nur Neue Datei + Datei laden
                <>
                  <button onClick={handleNewFile}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/25 text-emerald-400 text-xs font-medium transition-all">
                    <FileCode className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.newFileButton')}
                  </button>
                  <button onClick={() => setShowLib(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/25 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.loadFileButton')}
                  </button>
                  <button onClick={() => setShowOpenLib(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/25 text-violet-400 text-xs font-medium transition-all">
                    <Globe className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.openLibraryButton')}
                  </button>
                </>
              ) : (
                // Editor-Modus: volle Toolbar
                <>
                  {isDirty && (
                    <button onClick={handleSave}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-xs font-medium transition-all">
                      <Save className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.saveButton')}
                    </button>
                  )}
                  <button onClick={() => setShowLib(true)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.libraryButton')}
                  </button>
                  <button onClick={() => setShowOpenLib(true)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 text-violet-400 text-xs font-medium transition-all">
                    <Globe className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.openLibraryButton')}
                  </button>
                  <button
                    onClick={() => {
                      if (!aiSettings.enabled) {
                        error(t('devTrainPanel.editor.aiNotEnabled'), t('devTrainPanel.editor.aiNotEnabledDetail'));
                        return;
                      }
                      setShowAI(v => !v);
                    }}
                    title={!aiSettings.enabled ? t('devTrainPanel.editor.aiNotEnabledDetail') : ''}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : !aiSettings.enabled ? 'bg-white/5 text-gray-500 border-white/10 opacity-60' : 'bg-white/5 text-gray-400 hover:text-white border-white/10'}`}>
                    <Bot className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.aiButton')}
                  </button>
                  <button
                    onClick={() => {
                      setFindOpen(true);
                      setTimeout(() => findInputRef.current?.focus(), 0);
                      const ta = editorRef.current;
                      updateFindStatus(findQuery, ta?.selectionStart ?? 0);
                    }}
                    className="px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs font-medium transition-all"
                    title={t('devTrainPanel.editor.searchButton')}
                  >
                    ⌘F
                  </button>
                  <button onClick={() => setExpanded(v => !v)}
                    className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all">
                    {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                  {isRunning ? (
                    <button onClick={handleStop}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 text-xs font-medium transition-all">
                      <Square className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.stopButton')}
                    </button>
                  ) : (
                    <button onClick={handleStart} disabled={!script.trim() || !modelInfo}
                      className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-xs font-semibold hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}>
                      <Play className="w-3.5 h-3.5" /> {t('devTrainPanel.editor.runButton')}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {!fileOpen ? (
            // Welcome-Body: dunkle Fläche mit zentrierten Buttons
            <div
              className="flex flex-col items-center justify-center bg-slate-950 text-center"
              style={{ height: `${editorH}px` }}
            >
              <FileCode className="w-12 h-12 text-gray-700 mb-6" />
              <p className="text-gray-500 text-sm mb-8">{t('devTrainPanel.emptyState.description')}</p>
              <div className="flex gap-4">
                <button
                  onClick={handleNewFile}
                  className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-emerald-500/20 bg-emerald-500/8 hover:bg-emerald-500/15 hover:border-emerald-500/40 transition-all group"
                >
                  <FileCode className="w-7 h-7 text-emerald-500 group-hover:text-emerald-400" />
                  <div>
                    <p className="font-semibold text-white text-sm">{t('devTrainPanel.emptyState.newFile')}</p>
                    <p className="text-xs text-gray-500 mt-1">{t('devTrainPanel.emptyState.newFileHint')}</p>
                  </div>
                </button>
                <button
                  onClick={() => setShowLib(true)}
                  className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-amber-500/20 bg-amber-500/8 hover:bg-amber-500/15 hover:border-amber-500/40 transition-all group"
                >
                  <FolderClosed className="w-7 h-7 text-amber-500 group-hover:text-amber-400" />
                  <div>
                    <p className="font-semibold text-white text-sm">{t('devTrainPanel.emptyState.loadFile')}</p>
                    <p className="text-xs text-gray-500 mt-1">{t('devTrainPanel.emptyState.loadFileHint')}</p>
                  </div>
                </button>
                <button
                  onClick={() => setShowOpenLib(true)}
                  className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-violet-500/20 bg-violet-500/8 hover:bg-violet-500/15 hover:border-violet-500/40 transition-all group"
                >
                  <Globe className="w-7 h-7 text-violet-500 group-hover:text-violet-400" />
                  <div>
                    <p className="font-semibold text-white text-sm">{t('devTrainPanel.emptyState.openLib')}</p>
                    <p className="text-xs text-gray-500 mt-1">{t('devTrainPanel.emptyState.openLibHint')}</p>
                  </div>
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Editor + optional AI sidebar */}
              <div className="flex" style={{ height: expanded ? 'calc(100vh - 280px)' : `${editorH}px` }}>
                <div className="flex flex-1 min-w-0 overflow-hidden bg-slate-950">
                  <div className="relative flex-shrink-0 w-[56px] bg-slate-950 border-r border-white/5 select-none">
                    <div ref={gutterInnerRef} className="pt-4 px-3 text-right font-mono">
                      {Array.from({ length: lineCount }).map((_, i) => {
                        const n = i + 1;
                        const isActive = n === activeLine;
                        return (
                          <div
                            key={n}
                            className={`text-[10px] ${isActive ? 'text-violet-300' : 'text-gray-700'}`}
                            style={{ lineHeight: `${editorLineHeightPx}px` }}
                          >
                            {n}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="relative flex-1 min-w-0">
                    {/* Line highlights for edits */}
                    {highlightedLines.map((hl) => (
                      <div
                        key={`hl-${hl.lineNum}`}
                        aria-hidden
                        className="absolute left-0 right-0 pointer-events-none"
                        style={{
                          top: Math.round(editorPadTopPx + (hl.lineNum - 1) * editorLineHeightPx - editorScrollTop),
                          height: Math.ceil(editorLineHeightPx),
                          background:
                            hl.type === 'added'
                              ? 'rgba(34,197,94,0.25)'
                              : hl.type === 'removed'
                                ? 'rgba(239,68,68,0.25)'
                                : 'rgba(234,179,8,0.20)',
                          borderLeft:
                            hl.type === 'added'
                              ? '3px solid rgba(34,197,94,0.8)'
                              : hl.type === 'removed'
                                ? '3px solid rgba(239,68,68,0.8)'
                                : '3px solid rgba(234,179,8,0.8)',
                          borderRight:
                            hl.type === 'added'
                              ? '1px solid rgba(34,197,94,0.3)'
                              : hl.type === 'removed'
                                ? '1px solid rgba(239,68,68,0.3)'
                                : '1px solid rgba(234,179,8,0.3)',
                        }}
                      />
                    ))}

                    {/* Active line highlight */}
                    <div
                      aria-hidden
                      className="absolute left-0 right-0 pointer-events-none"
                      style={{
                        top: Math.round(editorPadTopPx + (activeLine - 1) * editorLineHeightPx - editorScrollTop),
                        height: Math.ceil(editorLineHeightPx),
                        background: 'rgba(168,85,247,0.06)',
                        borderLeft: '2px solid rgba(168,85,247,0.28)',
                      }}
                    />
                    <pre
                      ref={editorPreRef}
                      aria-hidden
                      className="absolute inset-0 text-xs font-mono overflow-hidden pointer-events-none text-gray-200 whitespace-pre"
                      style={{ 
                        fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code','Courier New',monospace", 
                        tabSize: 2 as any, 
                        lineHeight: `${editorLineHeightPx}px`,
                        padding: `${editorPadTopPx}px ${editorPadLeftPx}px ${editorPadTopPx}px ${editorPadLeftPx}px`,
                        boxSizing: 'border-box' as const,
                      }}
                      dangerouslySetInnerHTML={{ __html: highlightedHtml }}
                    />
                    <textarea
                      ref={editorRef}
                      autoFocus
                      value={script}
                      wrap="off"
                      placeholder={t('devTrainPanel.editor.placeholder')}
                      onChange={e => {
                        const newVal = e.target.value;
                        if (newVal === '! ') {
                          if (!modelInfo) {
                            error(t('devTrainPanel.editor.noModelForTemplate'), t('devTrainPanel.editor.noModelForTemplateDetail'));
                            return;
                          }
                          generateTemplate();
                          return;
                        }
                        setScript(newVal);
                        setIsDirty(newVal !== savedScript);
                      }}
                      onScroll={syncEditorScroll}
                      onKeyUp={updateActiveLine}
                      onMouseUp={updateActiveLine}
                      onSelect={updateActiveLine}
                      onFocus={() => { updateActiveLine(); setShowCursorBlink(true); }}
                      onBlur={() => setShowCursorBlink(false)}
                      spellCheck={false}
                      className="absolute inset-0 bg-transparent text-transparent text-xs font-mono focus:outline-none min-w-0 overflow-auto placeholder:text-gray-700 selection:bg-violet-500/25"
                      style={{
                        fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code','Courier New',monospace",
                        resize: 'none',
                        tabSize: 2 as any,
                        lineHeight: `${editorLineHeightPx}px`,
                        padding: `${editorPadTopPx}px ${editorPadLeftPx}px ${editorPadTopPx}px ${editorPadLeftPx}px`,
                        overflow: 'auto',
                        boxSizing: 'border-box' as const,
                        caretColor: 'transparent',
                      }}
                    />

                    {/* Custom Cursor */}
                    {showCursorBlink && (
                      <div
                        className="absolute pointer-events-none"
                        style={{
                          left: `${cursorX}px`,
                          top: `${cursorY + (editorLineHeightPx - 20) / 2}px`,
                          width: '2px',
                          height: '20px',
                          background: 'rgb(229,229,231)',
                          animation: 'blink 1s infinite',
                        }}
                      />
                    )}

                    <style>{`
                      @keyframes blink {
                        0%, 49% { opacity: 1; }
                        50%, 100% { opacity: 0; }
                      }
                    `}</style>

                    <div className="absolute inset-0 pointer-events-none ring-1 ring-inset ring-white/5" />
                  </div>
                </div>
                {showAI && (
                  <div className="w-80 border-l border-white/10 flex-shrink-0 flex flex-col overflow-hidden">
                  <CodeAISidebar
                      script={script}
                      modelInfo={modelInfo}
                      datasets={datasets}
                      outputPath={outputPath.replace('<job_id>', 'dev_XXX')}
                      onApplyEdit={(newScript) => {
                        setScript(newScript);
                        setIsDirty(true);
                      }}
                      onReplaceScript={(newScript) => {
                        setScript(newScript);
                        setIsDirty(true);
                      }}
                      onClose={() => setShowAI(false)}
                      initialInput={aiPrefill}
                      onHighlightLines={(edits) => {
                        const highlighted: HighlightedLine[] = [];
                        edits.forEach(edit => {
                          highlighted.push(...calculateAffectedLines(script, edit));
                        });
                        setHighlightedLines(highlighted);
                      }}
                      onClearHighlights={() => setHighlightedLines([])}
                    />
                  </div>
                )}
              </div>

              {/* Resize handle */}
              {!expanded && (
                <div
                  className="h-2 bg-white/[0.02] hover:bg-violet-500/20 cursor-ns-resize border-t border-white/10 flex items-center justify-center group transition-colors"
                  onMouseDown={e => {
                    e.preventDefault();
                    const startY = e.clientY, startH = editorH;
                    const move = (ev: MouseEvent) => setEditorH(Math.max(300, Math.min(900, startH + ev.clientY - startY)));
                    const up   = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
                    window.addEventListener('mousemove', move);
                    window.addEventListener('mouseup', up);
                  }}
                >
                  <div className="w-8 h-0.5 rounded-full bg-white/20 group-hover:bg-violet-400/60 transition-colors" />
                </div>
              )}
            </>
          )}
        </div>

        {/* Progress / Output */}
        {(isRunning || currentJob) && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isRunning
                  ? <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" />
                  : currentJob?.status === 'completed'
                    ? <CheckCircle className="w-4 h-4 text-emerald-400" />
                    : <AlertCircle className="w-4 h-4 text-red-400" />}
                <span className="text-white font-medium text-sm">
                  {isRunning ? t('devTrainPanel.output.running') : `Status: ${currentJob?.status}`}
                </span>
              </div>
              {progress && <span className="text-gray-400 text-xs">Epoch {progress.epoch}/{progress.total_epochs}</span>}
            </div>

            {progress && (
              <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`}
                  style={{ width: `${devProgressPercent(progress)}%` }} />
              </div>
            )}

            {lossPoints.length > 1 && (
              <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3">
                <p className="text-xs text-gray-500 mb-2">{t('devTrainPanel.output.lossHistory')}</p>
                <LossChart points={lossPoints} />
              </div>
            )}

            {progress && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: t('devTrainPanel.output.trainLossLabel'), value: progress.train_loss?.toFixed(4) ?? '—', icon: <TrendingDown className="w-3.5 h-3.5" /> },
                  { label: t('devTrainPanel.output.valLossLabel'),   value: progress.val_loss?.toFixed(4)   ?? '—', icon: <BarChart3    className="w-3.5 h-3.5" /> },
                  { label: t('devTrainPanel.output.lrLabel'),         value: progress.learning_rate?.toExponential(2) ?? '—', icon: <Zap className="w-3.5 h-3.5" /> },
                ].map(m => (
                  <div key={m.label} className="p-3 rounded-xl bg-white/5 space-y-1">
                    <div className="flex items-center gap-1.5 text-gray-500">{m.icon}<span className="text-xs">{m.label}</span></div>
                    <p className="text-white font-semibold text-sm">{m.value}</p>
                  </div>
                ))}
              </div>
            )}

            {(output || running) && (
              <div className="rounded-xl border border-white/10 bg-black/30 overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10">
                  <Terminal className="w-3.5 h-3.5 text-gray-500" />
                  <span className="text-[10px] text-gray-500">{t('devTrainPanel.output.outputLabel')}</span>
                </div>
                <div ref={outputRef} className="p-3 max-h-48 overflow-y-auto">
                  {isRunning && !output && <p className="text-gray-600 text-[10px] animate-pulse">{t('devTrainPanel.output.waitingForOutput')}</p>}
                  <pre className="text-[10px] font-mono text-gray-300 whitespace-pre-wrap leading-relaxed">{output}</pre>
                </div>
              </div>
            )}

            {currentJob?.error && (
              <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-300 text-xs">{currentJob.error}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {showLibrary && (
        <ScriptLibraryModal 
          currentScript={script} 
          onLoad={s => {
            setScript(s.script);
            setSavedScript(s.script);
            setCurrentScriptId(s.id);
            setIsDirty(false);
            setFileOpen(true);
          }} 
          onClose={() => setShowLib(false)}
          userId={userData?.userId} 
        />
      )}

      {showOpenLib && (
        <OpenLibraryModal
          userData={userData}
          onClose={() => setShowOpenLib(false)}
          onLoadScript={(scriptContent, scriptName) => {
            setScript(scriptContent);
            setSavedScript('');
            setCurrentScriptId(null);
            setIsDirty(true);
            setFileOpen(true);
            setShowOpenLib(false);
          }}
        />
      )}

      <SaveNameDialog
        isOpen={showSaveDialog}
        defaultName={saveName}
        onSave={handleSaveWithName}
        onClose={() => setShowSaveDialog(false)}
      />

      <DevTrainErrorModal
        isOpen={showErrorModal}
        errorTitle={errorTitle}
        errorMessage={errorMessage}
        errorDetails={errorDetails}
        script={script}
        output={output}
        onClose={() => setShowErrorModal(false)}
        onSendToFrameTrain={handleSendToFrameTrain}
        onSendToAI={handleSendToAI}
        isSending={isSendingError}
      />

      <TrainingDashboard
        isOpen={showDashboard}
        isMinimized={isDashMinimized}
        onMinimize={() => setIsDashMinimized(true)}
        onMaximize={() => setIsDashMinimized(false)}
        mode="dev"
        modelName={modelInfo?.name ?? 'Unbekanntes Modell'}
        datasetName={datasets[0]?.name ?? 'Kein Dataset'}
        job={currentJob}
        lossPoints={lossPoints}
        sessionId={devSessionIdRef.current}
        startedAt={devStartedAtRef.current}
        onStop={handleStop}
        onClose={() => { setShowDashboard(false); setJob(null); setLoss([]); }}
        devScript={script}
        onSendCodeToKI={(s, err) => {
          setShowAI(true);
      setAiPrefill(
            `[Dev Train Error]\n\n` +
            `${t('devTrainPanel.aiPrefill.fixScript')}\n\n` +
            `${t('devTrainPanel.aiPrefill.error')}\n${err}\n\n` +
            `${t('devTrainPanel.aiPrefill.script')}\n${s}\n`
          );
        }}
      />
    </div>
  );
}

// ── Helper Components ─────────────────────────────────────────────────────

function RefRow({ color, label, value, hint }: { color: string; label: string; value: string; hint?: string }) {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Silent fail
    }
  };

  return (
    <div className="flex items-start gap-3 py-0.5">
      <span className={`${color} min-w-[140px] flex-shrink-0`}>{label}</span>
      <div className="min-w-0 flex-1">
        <span className={`break-all ${value ? 'text-gray-300' : 'text-gray-600 italic'}`}>
          {value || t('devTrainPanel.paths.notSet')}
        </span>
        {hint && <span className="text-gray-600 ml-1.5 text-[10px]">({hint})</span>}
      </div>
      {value && (
        <button
          onClick={handleCopy}
          className={`flex-shrink-0 px-2 py-1 rounded-md text-xs transition-all ${
            copied
              ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-300'
              : 'bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10'
          }`}
          title={useLanguage().t('settings.account.copy')}
        >
          {copied ? (
            <Check className="w-3.5 h-3.5" />
          ) : (
            <Copy className="w-3.5 h-3.5" />
          )}
        </button>
      )}
    </div>
  );
}

/**
 * Fortschritt in Prozent fuer eigene Dev-Train-Scripts.
 *
 * Die meisten Scripts melden nur step/total_steps. Ohne diese Ableitung blieb
 * der Balken bei 0 %, obwohl daneben "Step 30 / 60" stand.
 */
export function devProgressPercent(p: {
  progress_percent?: number; step?: number; total_steps?: number;
  epoch?: number; total_epochs?: number;
} | null | undefined): number {
  if (!p) return 0;
  const given = p.progress_percent ?? 0;
  if (given > 0) return Math.min(100, given);
  const step = p.step ?? 0, total = p.total_steps ?? 0;
  if (total > 0 && step > 0) return Math.min(100, (step / total) * 100);
  const ep = p.epoch ?? 0, eps = p.total_epochs ?? 0;
  return eps > 0 && ep > 0 ? Math.min(100, (ep / eps) * 100) : 0;
}

// ── Default Script Generator ──────────────────────────────────────────────

function generateDefaultScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const modality = detectScriptModality(model);
  if (modality !== 'text') {
    return generateMediaTrainScript(modality, model, datasets, outputPath);
  }
  return generateTextTrainScript(model, datasets, outputPath);
}

/**
 * Trainings-Vorlagen fuer Bild, Audio und Seq2Seq. Sie melden dieselben
 * Events wie die Text-Vorlage (status/progress/complete) und schreiben
 * label_mapping.json, damit Labor und Tests spaeter Klassennamen zeigen.
 */
function generateMediaTrainScript(
  modality: Exclude<ScriptModality, 'text'>,
  model: ModelInfo | null,
  datasets: DatasetInfo[],
  outputPath: string,
): string {
  const ds = datasets[0];
  const modelPathDefault   = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_train').replace('{wird beim Start gesetzt}', 'dev_train');

  const header = `#!/usr/bin/env python3
# FrameTrain - Dev Train Script
#
# Dieses Template laeuft so wie es ist. Aendere es nach Belieben.
# Fortschritt erscheint in FrameTrain, sobald eine JSON-Zeile mit
# {"type": "progress", ...} auf stdout geschrieben wird - genau das macht
# der FrameTrainCallback weiter unten.

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")


def emit(kind: str, **data):
    # Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event).
    print(json.dumps({"type": kind, "data": data}), flush=True)


class FrameTrainCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        emit(
            "progress",
            # state.epoch laeuft von 0.0 hoch - 1-basiert melden wie die Engine
            epoch=min(EPOCHS, int(state.epoch or 0) + 1),
            total_epochs=EPOCHS,
            step=state.global_step,
            total_steps=state.max_steps or 0,
            train_loss=logs.get("loss"),
            val_loss=logs.get("eval_loss"),
            learning_rate=logs.get("learning_rate", LR),
        )
`;

  if (modality === 'seq2seq') {
    return `${header}
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

# -- Hyperparameter -------------------------------------------------------
EPOCHS      = 1
BATCH_SIZE  = 4
LR          = 3e-4
MAX_SOURCE  = 256
MAX_TARGET  = 64
MAX_STEPS   = 60      # -1 = ganzes Dataset
TASK_PREFIX = ""      # z.B. "summarize: " fuer T5
SOURCE_COL  = None    # None = automatisch erkennen
TARGET_COL  = None

EXTS = (".csv", ".tsv", ".json", ".jsonl", ".parquet")


def load_rows(path: str, split: str):
    import csv

    p = Path(path) / split
    if not p.is_dir():
        p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        return []
    f = files[0]
    if f.suffix.lower() == ".parquet":
        import pandas as pd
        return pd.read_parquet(f).to_dict("records")
    if f.suffix.lower() in (".json", ".jsonl"):
        raw = f.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    delim = "\\t" if f.suffix.lower() == ".tsv" else ","
    with open(f, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


emit("status", stage="loading", message="Daten werden geladen ...")
train_rows = load_rows(DATASET_PATH, "train")
eval_rows = load_rows(DATASET_PATH, "val") or load_rows(DATASET_PATH, "validation") or load_rows(DATASET_PATH, "test")
if not train_rows:
    raise RuntimeError(f"Keine Trainingsdaten in {DATASET_PATH} gefunden.")

cols = list(train_rows[0].keys())
source_col = SOURCE_COL or next((c for c in ("source", "input", "text", "article", "document", "de") if c in cols), None)
target_col = TARGET_COL or next((c for c in ("target", "output", "summary", "translation", "en") if c in cols), None)
if source_col is None or target_col is None:
    raise RuntimeError(f"Quell-/Zielspalte nicht gefunden. Vorhanden: {cols}")
print(f"{len(train_rows)} Trainings- / {len(eval_rows)} Eval-Beispiele | '{source_col}' -> '{target_col}'", flush=True)

emit("status", stage="loading", message="Modell wird geladen ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


class Seq2SeqRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc = tokenizer(TASK_PREFIX + str(row[source_col]), truncation=True, max_length=MAX_SOURCE)
        lab = tokenizer(text_target=str(row[target_col]), truncation=True, max_length=MAX_TARGET)
        enc["labels"] = lab["input_ids"]
        return enc


args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=5,
    eval_strategy="no" if MAX_STEPS > 0 else "epoch",
    save_strategy="no",
    report_to=[],
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=Seq2SeqRows(train_rows),
    eval_dataset=Seq2SeqRows(eval_rows) if eval_rows else None,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    callbacks=[FrameTrainCallback()],
)

emit("status", stage="training", message="Training laeuft ...")
trainer.train()

metrics = trainer.evaluate() if eval_rows else {}
print("Eval:", metrics, flush=True)

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Modell gespeichert unter {OUTPUT_PATH}", flush=True)

emit(
    "complete",
    model_path=OUTPUT_PATH,
    final_metrics={
        "accuracy": None,
        "val_loss": metrics.get("eval_loss"),
        "total_epochs": EPOCHS,
    },
)
`;
  }

  const isImage = modality === 'image';
  const kindLabel = isImage ? 'Bild' : 'Audio';
  const exts = isImage
    ? '{".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}'
    : '{".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}';
  const imports = isImage
    ? `from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification`
    : `import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification`;
  const hyper = isImage
    ? `EPOCHS     = 1
BATCH_SIZE = 8
LR         = 5e-5
MAX_STEPS  = 40      # -1 = ganzes Dataset`
    : `EPOCHS      = 1
BATCH_SIZE  = 4
LR          = 3e-5
MAX_STEPS   = 40      # -1 = ganzes Dataset
MAX_SECONDS = 10.0    # laengere Aufnahmen werden gekappt`;
  const loadModel = isImage
    ? `processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH, num_labels=len(classes), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True,
)`
    : `processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
SAMPLING_RATE = int(getattr(processor, "sampling_rate", 16000) or 16000)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_PATH, num_labels=len(classes), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True,
)`;
  const itemBody = isImage
    ? `        with Image.open(path) as im:
            image = im.convert("RGB")
        pixel = processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel, "labels": label}`
    : `        wave, _ = librosa.load(str(path), sr=SAMPLING_RATE, mono=True)
        feats = processor(
            wave, sampling_rate=SAMPLING_RATE, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=int(MAX_SECONDS * SAMPLING_RATE),
        )
        return {"input_values": feats["input_values"][0], "labels": label}`;

  return `${header}
${imports}

# -- Hyperparameter -------------------------------------------------------
${hyper}

EXTS = ${exts}


def collect(root: Path):
    # Dateien samt Klasse (Ordnername) einsammeln.
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    items = []
    for d in class_dirs:
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in EXTS:
                items.append((f, d.name))
    return items


emit("status", stage="loading", message="Daten werden geladen ...")
root = Path(DATASET_PATH)
if not root.exists():
    raise FileNotFoundError(f"DATASET_PATH existiert nicht: {DATASET_PATH}")

train_items = collect(root / "train") if (root / "train").is_dir() else collect(root)
eval_root = next((root / s for s in ("val", "validation", "test") if (root / s).is_dir()), None)
eval_items = collect(eval_root) if eval_root else []
if not train_items:
    raise RuntimeError(f"Keine ${kindLabel}dateien mit Klassenordnern in {DATASET_PATH} gefunden.")

classes = sorted({cls for _, cls in train_items})
id2label = {i: c for i, c in enumerate(classes)}
label2id = {c: i for i, c in id2label.items()}
print(f"{len(train_items)} Trainings- / {len(eval_items)} Eval-Dateien | Klassen: {classes}", flush=True)

emit("status", stage="loading", message="Modell wird geladen ...")
${loadModel}


class FileDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        label = label2id[cls]
${itemBody}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=5,
    eval_strategy="no" if MAX_STEPS > 0 else "epoch",
    save_strategy="no",
    report_to=[],
    disable_tqdm=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=FileDataset(train_items),
    eval_dataset=FileDataset(eval_items) if eval_items else None,
    compute_metrics=compute_metrics,
    callbacks=[FrameTrainCallback()],
)

emit("status", stage="training", message="Training laeuft ...")
trainer.train()

metrics = trainer.evaluate() if eval_items else {}
print("Eval:", metrics, flush=True)

# -- Speichern ------------------------------------------------------------
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_PATH)
processor.save_pretrained(OUTPUT_PATH)
# Klassennamen, damit Labor und Tests spaeter Namen statt 0/1 zeigen
with open(f"{OUTPUT_PATH}/label_mapping.json", "w") as f:
    json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)
print(f"Modell gespeichert unter {OUTPUT_PATH}", flush=True)

emit(
    "complete",
    model_path=OUTPUT_PATH,
    final_metrics={
        "accuracy": metrics.get("eval_accuracy"),
        "val_loss": metrics.get("eval_loss"),
        "total_epochs": EPOCHS,
    },
)
`;
}

function generateTextTrainScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const ds = datasets[0];
  const modelPathDefault  = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_train').replace('{wird beim Start gesetzt}', 'dev_train');

  return `#!/usr/bin/env python3
# FrameTrain - Dev Train Script
#
# Dieses Template laeuft so wie es ist. Aendere es nach Belieben.
# Fortschritt erscheint in FrameTrain, sobald eine JSON-Zeile mit
# {"type": "progress", ...} auf stdout geschrieben wird - genau das macht
# der FrameTrainCallback weiter unten.

import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# -- Pfade (von FrameTrain als ENV-Vars gesetzt) --------------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

# -- Hyperparameter -------------------------------------------------------
EPOCHS      = 1
BATCH_SIZE  = 8
LR          = 2e-5
MAX_LENGTH  = 128
MAX_STEPS   = 60      # -1 = ganzes Dataset. Klein halten beim Ausprobieren.
MAX_EVAL    = 500     # 0 = kompletten Eval-Split auswerten
TEXT_COL    = None    # None = automatisch erkennen
LABEL_COL   = None    # None = automatisch erkennen


def emit(kind: str, **data):
    """Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event)."""
    print(json.dumps({"type": kind, "data": data}), flush=True)


# -- Dataset laden --------------------------------------------------------
def load_frametrain_dataset(path: str):
    """
    FrameTrain liefert DATASET_PATH als lokalen Ordner.
    - save_to_disk-Dataset  -> load_from_disk(path)
    - sonst: JSON/JSONL/CSV/TSV/Parquet im Split-Layout train/ val/ test/
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")

    if p.is_dir() and any((p / m).exists() for m in
                          ("dataset_info.json", "state.json", "dataset_dict.json")):
        return load_from_disk(str(p))

    EXTS = (".json", ".jsonl", ".csv", ".tsv", ".parquet")

    def files_in(sub: str):
        d = p / sub
        if not d.is_dir():
            return []
        return sorted(f for f in d.rglob("*") if f.suffix.lower() in EXTS)

    data_files = {}
    for split, sub_names in (("train", ["train"]),
                             ("validation", ["val", "validation"]),
                             ("test", ["test"])):
        for sub in sub_names:
            found = files_in(sub)
            if found:
                data_files[split] = [str(f) for f in found]
                break

    if not data_files:
        loose = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        if not loose:
            raise RuntimeError(f"Keine Daten-Dateien in {path} gefunden.")
        data_files["train"] = [str(f) for f in loose]

    ext = Path(data_files["train"][0]).suffix.lower()
    if ext in (".json", ".jsonl"):
        return load_dataset("json", data_files=data_files)
    if ext == ".parquet":
        return load_dataset("parquet", data_files=data_files)
    if ext == ".tsv":
        return load_dataset("csv", data_files=data_files, delimiter="\\t")
    return load_dataset("csv", data_files=data_files)


def pick_columns(ds):
    """Text- und Label-Spalte erraten, falls nicht oben festgelegt."""
    cols = list(ds.features.keys())
    text = TEXT_COL or next(
        (c for c in ("text", "sentence", "content", "review", "input") if c in cols),
        None,
    )
    label = LABEL_COL or next(
        (c for c in ("label", "labels", "target", "class") if c in cols), None
    )
    if text is None or label is None:
        raise RuntimeError(
            f"Text-/Label-Spalte nicht gefunden. Vorhanden: {cols}. "
            "Setze TEXT_COL / LABEL_COL oben im Script."
        )
    return text, label


emit("status", stage="loading", message="Dataset wird geladen ...")
dataset = load_frametrain_dataset(DATASET_PATH)
print("Dataset geladen:", dataset, flush=True)

train_ds = dataset["train"]
eval_ds = dataset.get("validation") or dataset.get("test")
if eval_ds is None:
    split = train_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

# Eval-Split deckeln: eine Auswertung ueber 25.000 Beispiele dauert laenger
# als das ganze Training mit MAX_STEPS. Zufaellig ziehen, nicht die ersten N —
# viele Splits sind nach Label sortiert und ergaeben sonst nur eine Klasse.
if MAX_EVAL and len(eval_ds) > MAX_EVAL:
    eval_ds = eval_ds.shuffle(seed=42).select(range(MAX_EVAL))
    print(f"Eval-Split auf {MAX_EVAL} Beispiele begrenzt", flush=True)

text_col, label_col = pick_columns(train_ds)
num_labels = len(set(train_ds[label_col]))
if num_labels < 2:
    raise RuntimeError(
        f"Label-Spalte '{label_col}' hat nur einen Wert - damit laesst sich "
        "keine Klassifikation trainieren."
    )
print(f"Spalten: text='{text_col}', label='{label_col}', Klassen={num_labels}", flush=True)

# -- Modell & Tokenizer ---------------------------------------------------
emit("status", stage="loading", message="Modell wird geladen ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=num_labels
)
print(f"Modell geladen von {MODEL_PATH}", flush=True)


def tokenize(batch):
    return tokenizer(batch[text_col], truncation=True, max_length=MAX_LENGTH)


keep = {label_col}
train_tok = train_ds.map(tokenize, batched=True,
                         remove_columns=[c for c in train_ds.column_names if c not in keep])
eval_tok = eval_ds.map(tokenize, batched=True,
                       remove_columns=[c for c in eval_ds.column_names if c not in keep])
if label_col != "labels":
    train_tok = train_tok.rename_column(label_col, "labels")
    eval_tok = eval_tok.rename_column(label_col, "labels")


# -- Fortschritt an FrameTrain melden -------------------------------------
class FrameTrainCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        emit(
            "progress",
            # state.epoch laeuft von 0.0 hoch - 1-basiert melden wie die Engine
            epoch=min(EPOCHS, int(state.epoch or 0) + 1),
            total_epochs=EPOCHS,
            step=state.global_step,
            total_steps=state.max_steps or 0,
            train_loss=logs.get("loss"),
            val_loss=logs.get("eval_loss"),
            learning_rate=logs.get("learning_rate", LR),
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=5,
    eval_strategy="no" if MAX_STEPS > 0 else "epoch",
    save_strategy="no",
    report_to=[],
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[FrameTrainCallback()],
)

emit("status", stage="training", message="Training laeuft ...")
trainer.train()

metrics = trainer.evaluate()
print("Eval:", metrics, flush=True)

# -- Speichern ------------------------------------------------------------
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Modell gespeichert unter {OUTPUT_PATH}", flush=True)

emit(
    "complete",
    model_path=OUTPUT_PATH,
    final_metrics={
        "accuracy": metrics.get("eval_accuracy"),
        "val_loss": metrics.get("eval_loss"),
        "total_epochs": EPOCHS,
    },
)
`;
}
