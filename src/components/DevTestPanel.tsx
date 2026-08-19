// DevTestPanel.tsx – Dev Test Mode (analog zu DevTrainPanel, aber für Inference/Testing)
// KI-Assistent kann den Code direkt bearbeiten (EDIT-Protokoll)

import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Loader2, Terminal, FolderOpen, FileCode,
  FolderClosed, Bot, Send, Maximize2, Minimize2, X, Minus, Plus,
  AlertCircle, CheckCircle,
  Save, FileText, Trash2, Pencil, Check, Wand2, Sparkles, Copy,
  FlaskConical, ClipboardList,
  History, MessageSquarePlus, Globe,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';
import type { ModelInfo, DatasetInfo } from './TrainingPanel';
import { callAI } from './TrainingPanel';
import { parseEdits, applyEdit, applyAllEdits, removeEditBlocks, extractFullPythonCode, type CodeEdit } from '../ai/codeEdits';
import { buildAutoSystemPrompt, parseAutoAction, type AutoAction } from '../ai/autoModeProtocol';
import { migrateLegacyDevScripts } from '../utils/devScriptStorage';
import { detectPlugin } from '../plugins/registry';
import DiffViewer from './DiffViewer';
import OpenLibraryModal from './OpenLibraryModal';
import { dateLocale } from '../utils/dateLocale';

// ── Script Library ────────────────────────────────────────────────────────

interface SavedScript { id: string; name: string; script: string; savedAt: string; }

// User-getrennt (vorher globaler Key für ALLE Accounts auf dem Gerät!) —
// Legacy-Migration übernimmt utils/devScriptStorage beim ersten Laden.
const getScriptsKey = (userId?: string) => userId ? `ft_saved_test_scripts_${userId}` : 'ft_saved_test_scripts';
const loadScripts  = (userId?: string): SavedScript[] => { try { return JSON.parse(localStorage.getItem(getScriptsKey(userId)) ?? '[]'); } catch { return []; } };
const saveScript   = (name: string, script: string, userId?: string) => { const all = loadScripts(userId); all.unshift({ id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString() }); localStorage.setItem(getScriptsKey(userId), JSON.stringify(all.slice(0, 50))); };
const deleteScript = (id: string, userId?: string) => localStorage.setItem(getScriptsKey(userId), JSON.stringify(loadScripts(userId).filter(s => s.id !== id)));
const updateScript = (id: string, script: string, userId?: string) => { const all = loadScripts(userId); const idx = all.findIndex(s => s.id === id); if (idx >= 0) { all[idx] = { ...all[idx], script, savedAt: new Date().toISOString() }; localStorage.setItem(getScriptsKey(userId), JSON.stringify(all)); } };

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

// ── Python Syntax Highlighting ─────────────────────────────────────────

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function highlightPythonToHtml(code: string) {
  const KEYWORDS = new Set(['False','None','True','and','as','assert','async','await','break','class','continue','def','del','elif','else','except','finally','for','from','global','if','import','in','is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield']);
  const BUILTINS = new Set(['print','len','range','enumerate','zip','map','filter','list','dict','set','tuple','str','int','float','bool','open','sum','min','max','sorted','any','all','isinstance','type','super','dir','vars','getattr','setattr','hasattr','Exception','ValueError','TypeError']);

  type Seg = { t: 'code' | 'str' | 'cmt'; v: string };
  const segs: Seg[] = [];
  let i = 0; let cur = ''; let state: 'code' | 'str' | 'cmt' = 'code'; let quote: string | null = null;
  const flush = () => { if (!cur) return; segs.push({ t: state, v: cur }); cur = ''; };

  while (i < code.length) {
    const ch = code[i]; const next3 = code.slice(i, i + 3);
    if (state === 'code') {
      if (next3 === "'''" || next3 === '"""') { flush(); state = 'str'; quote = next3; cur += next3; i += 3; continue; }
      if (ch === "'" || ch === '"') { flush(); state = 'str'; quote = ch; cur += ch; i += 1; continue; }
      if (ch === '#') { flush(); state = 'cmt'; quote = null; cur += ch; i += 1; continue; }
      cur += ch; i += 1; continue;
    }
    if (state === 'cmt') { cur += ch; i += 1; if (ch === '\n') { flush(); state = 'code'; } continue; }
    cur += ch; i += 1;
    if (quote === "'" || quote === '"') {
      if (ch === '\\' && i < code.length) { cur += code[i]; i += 1; continue; }
      if (ch === quote) { flush(); state = 'code'; quote = null; }
      continue;
    }
    if (quote === "'''" || quote === '"""') {
      if (code.slice(i - 1, i - 1 + 3) === quote) { cur += quote.slice(1); i += 2; flush(); state = 'code'; quote = null; }
    }
  }
  flush();

  const highlightCode = (s: string) => {
    let out = escapeHtml(s);
    out = out.replace(/\b\d+(\.\d+)?\b/g, '<span class="tok-num">$&</span>');
    out = out.replace(/\b(def)\s+([A-Za-z_][A-Za-z0-9_]*)/g, '<span class="tok-kw">$1</span> <span class="tok-fn">$2</span>');
    out = out.replace(/\b(class)\s+([A-Za-z_][A-Za-z0-9_]*)/g, '<span class="tok-kw">$1</span> <span class="tok-cl">$2</span>');
    out = out.replace(/(^|\n)(\s*)(@[\w\.]+)/g, '$1$2<span class="tok-de">$3</span>');
    out = out.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\b/g, (m, w: string) => {
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

// ── Save Name Dialog ───────────────────────────────────────────────────────

function SaveNameDialog({ isOpen, defaultName, onSave, onClose }: { isOpen: boolean; defaultName: string; onSave: (name: string) => void; onClose: () => void; }) {
  const { t } = useLanguage();
  const [name, setName] = useState(defaultName);
  useEffect(() => { setName(defaultName); }, [defaultName]);
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
          <div className="flex items-center gap-2"><Save className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">{t('devTestPanel.saveDialog.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-300 text-sm">{t('devTestPanel.saveDialog.description')}</p>
          <input value={name} onChange={e => setName(e.target.value)} onKeyDown={e => e.key === 'Enter' && name.trim() && onSave(name.trim())} placeholder={t('devTestPanel.saveDialog.placeholder')} autoFocus
            className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40" />
        </div>
        <div className="px-6 pb-6 flex gap-2">
          <button onClick={() => name.trim() && onSave(name.trim())} disabled={!name.trim()}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40 transition-all">
            <Save className="w-4 h-4" /> {t('devTestPanel.saveDialog.saveButton')}
          </button>
          <button onClick={onClose} className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white text-sm font-medium transition-all">{t('devTestPanel.saveDialog.cancelButton')}</button>
        </div>
      </div>
    </div>
  );
}

// ── Script Library Modal ──────────────────────────────────────────────────

function ScriptLibraryModal({ currentScript, onLoad, onClose, userId }: { currentScript: string; onLoad: (s: SavedScript) => void; onClose: () => void; userId?: string; }) {
  const { t, language } = useLanguage();
  const [scripts, setScripts] = useState<SavedScript[]>([]);
  const [saveName, setSaveName] = useState('');
  const [showSaveForm, setShowForm] = useState(false);
  const { success } = useNotification();

  useEffect(() => { setScripts(loadScripts(userId)); }, [userId]);

  const handleSave = () => {
    if (!saveName.trim()) return;
    saveScript(saveName.trim(), currentScript, userId);
    setScripts(loadScripts(userId));
    setSaveName(''); setShowForm(false);
    success(t('devTestPanel.notifications.savedNew'), t('devTestPanel.notifications.savedNewDetail').replace('{name}', saveName));
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FolderClosed className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">{t('devTestPanel.library.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {scripts.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FileText className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">{t('devTestPanel.library.empty')}</p>
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
                  <button onClick={() => { onLoad(s); onClose(); }} className="px-3 py-1.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-xs font-medium transition-all">{t('devTestPanel.library.loadButton')}</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 pb-5 border-t border-white/10 pt-4 flex-shrink-0">
          {showSaveForm ? (
            <div className="flex gap-2">
              <input value={saveName} onChange={e => setSaveName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSave()} placeholder={t('devTestPanel.library.namePlaceholder')} autoFocus
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40" />
              <button onClick={handleSave} disabled={!saveName.trim()} className="px-4 py-2 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40"><Save className="w-4 h-4" /></button>
              <button onClick={() => setShowForm(false)} className="px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-400 text-sm"><X className="w-4 h-4" /></button>
            </div>
          ) : (
            <button onClick={() => setShowForm(true)} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-300 text-sm font-medium transition-all">
              <Save className="w-4 h-4" /> {t('devTestPanel.library.saveCurrentButton')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Code AI Sidebar ───────────────────────────────────────────────────────

interface AiMessage { role: 'user' | 'assistant'; content: string; edits?: CodeEdit[]; action?: AutoAction | null; }
interface AppliedEditInfo { messageId: number; editId: string; originalScript: string; }
interface ChatSession {
  id: string;
  title: string;
  messages: AiMessage[];
  createdAt: string;
  updatedAt: string;
}

const DEVTEST_SESSIONS_KEY = 'ft_devtest_sessions';
const MAX_DEVTEST_SESSIONS = 15;
const MAX_SESSION_MESSAGES = 30;
const SESSION_MAX_AGE_MS   = 12 * 60 * 60 * 1000;

function loadChatSessions(): ChatSession[] {
  try { return JSON.parse(localStorage.getItem(DEVTEST_SESSIONS_KEY) ?? '[]'); } catch { return []; }
}
function saveChatSessions(sessions: ChatSession[]) {
  localStorage.setItem(DEVTEST_SESSIONS_KEY, JSON.stringify(sessions.slice(0, MAX_DEVTEST_SESSIONS)));
}
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60_000)         return 'gerade eben';
  if (diff < 3_600_000)      return `vor ${Math.floor(diff / 60_000)} Min`;
  if (diff < 86_400_000)     return `vor ${Math.floor(diff / 3_600_000)} Std`;
  if (diff < 7 * 86_400_000) return `vor ${Math.floor(diff / 86_400_000)} Tagen`;
  return new Date(iso).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit' });
}
function makeSessionTitle(msg: string): string {
  return msg.trim().slice(0, 42) + (msg.trim().length > 42 ? '…' : '');
}

function CodeAISidebar({ script, modelInfo, datasets, outputPath, onApplyEdit, onReplaceScript, onClose, initialInput, modelPathOverride, onHighlightLines, onClearHighlights }: {
  script: string; modelInfo: ModelInfo | null; datasets: DatasetInfo[]; outputPath: string;
  onApplyEdit: (s: string) => void; onReplaceScript: (s: string) => void; onClose: () => void;
  initialInput?: string; modelPathOverride?: string; onHighlightLines?: (edits: CodeEdit[]) => void; onClearHighlights?: () => void;
}) {
  const { settings: aiSettings } = useAISettings();
  const { t, language } = useLanguage();
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
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
  const [isReadonly, setIsReadonly]             = useState(false);
  const [sessionTitle, setSessionTitle]         = useState('');
  const currentSessionIdRef = useRef<string | null>(null);
  currentSessionIdRef.current = currentSessionId;

  useEffect(() => {
    const sessions = loadChatSessions();
    const last = sessions[0];
    const tooOld  = last ? (Date.now() - new Date(last.updatedAt).getTime()) > SESSION_MAX_AGE_MS : false;
    const tooLong = last ? last.messages.length >= MAX_SESSION_MESSAGES : false;
    if (!last || tooOld || tooLong) {
      const id = `s_${Date.now()}`;
      const ns: ChatSession = { id, title: t('devTestPanel.chat.newChatTitle'), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
      saveChatSessions([ns, ...sessions]);
      setCurrentSessionId(id); setSessionTitle(t('devTestPanel.chat.newChatTitle')); setMessages([]);
    } else {
      setCurrentSessionId(last.id); setSessionTitle(last.title); setMessages(last.messages);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const id = currentSessionIdRef.current;
    if (!id) return;
    const sessions = loadChatSessions();
    const idx = sessions.findIndex(s => s.id === id);
    if (idx < 0) return;
    if (messages.length === 0) { saveChatSessions(sessions.filter(s => s.id !== id)); return; }
    sessions[idx] = { ...sessions[idx], messages, updatedAt: new Date().toISOString() };
    saveChatSessions(sessions);
  }, [messages]);

  const startNewSession = () => {
    const id = `s_${Date.now()}`;
    const ns: ChatSession = { id, title: t('devTestPanel.chat.newChatTitle'), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    const sessions = loadChatSessions().filter(s => s.messages.length > 0);
    saveChatSessions([ns, ...sessions]);
    setCurrentSessionId(id); setSessionTitle(t('devTestPanel.chat.newChatTitle')); setMessages([]);
    setAppliedEdits([]); setCurrentMessageWithEdits(null); setIsReadonly(false); setShowHistory(false);
    setRetryText(null);
    onClearHighlights?.();
  };

  const switchToSession = (session: ChatSession) => {
    setCurrentSessionId(session.id); setSessionTitle(session.title); setMessages(session.messages);
    setAppliedEdits([]); setCurrentMessageWithEdits(null); setIsReadonly(true); setShowHistory(false);
    setRetryText(null);
    onClearHighlights?.();
  };

  const continueFromSession = (session: ChatSession) => {
    const id = `s_${Date.now()}`;
    const title = session.title + t('devTestPanel.chat.continuedSuffix');
    const ns: ChatSession = { id, title, messages: session.messages, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    const sessions = loadChatSessions();
    saveChatSessions([ns, ...sessions]);
    setCurrentSessionId(id); setSessionTitle(title); setMessages(session.messages);
    setAppliedEdits([]); setCurrentMessageWithEdits(null); setIsReadonly(false); setShowHistory(false);
    onClearHighlights?.();
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const sessions = loadChatSessions().filter(s => s.id !== id);
    saveChatSessions(sessions);
    if (id === currentSessionIdRef.current) startNewSession();
    setShowHistory(false);
    setTimeout(() => setShowHistory(true), 0);
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => {
    if (!initialInput || initialInput === lastPrefillRef.current) return;
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

  const modelPath = modelPathOverride || modelInfo?.local_path || modelInfo?.source_path || modelInfo?.name || 'MODELL_PFAD';
  const dsRefs    = datasets.map((d, i) => `${i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`} = "${d.storage_path || d.name}" (${d.name})`);

  const baseSystemPrompt = `Du bist ein professioneller Code-Side-Assistant in FrameTrain (Dev Test).

ZIEL: Hilf dem User, das Test-Skript schnell, korrekt und robust zu fixen/verbessern.

KONTEXT (lokal):
- MODEL_PATH = "${modelPath}"
${dsRefs.map(r => `- ${r}`).join('\n')}
- OUTPUT_PATH = "${outputPath}"

INSTALLIERTE PAKETE: torch, transformers, datasets, scikit-learn, numpy, accelerate, peft

AKTUELLER SCRIPT-INHALT:
\`\`\`python
${script}
\`\`\`

ANFORDERUNGEN:
- Antworte kurz, technisch präzise.
- Wenn möglich: mode="edit" mit ##EDIT_START## Blöcken.
- Wenn ein kompletter Rewrite klar besser ist: mode="rewrite" + kompletter \`\`\`python\`\`\` Block.
- Rückfragen nur wenn absolut nötig.`;

  const systemPrompt = buildAutoSystemPrompt(baseSystemPrompt);

  const suggestions = [
    t('devTestPanel.aiSidebar.suggestions.fixError'),
    t('devTestPanel.aiSidebar.suggestions.addMetric'),
    t('devTestPanel.aiSidebar.suggestions.batchInference'),
  ];

  const send = async (retryTextArg?: string) => {
    const isRetry = typeof retryTextArg === 'string';
    const text = (isRetry ? retryTextArg : input).trim();
    if (!text || loading || isReadonly) return;
    const userMsg: AiMessage = { role: 'user', content: text };
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
              ) 
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
          ? { ...mm, edits: updatedEdits.map((e, i) => i === editIdx ? { ...e, failed: true } : e) } 
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
        <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <Bot className="w-4 h-4 text-violet-400 flex-shrink-0" />
            <span className="text-sm font-medium text-white">{t('devTestPanel.chat.title')}</span>
          </div>
          <div className="flex items-center gap-0.5 flex-shrink-0">
            <span className="px-2 py-0.5 rounded-md bg-purple-500/15 border border-purple-500/25 text-purple-200 text-[10px] font-medium">{t('devTestPanel.chat.autoBadge')}</span>
            <button
              onClick={() => setShowHistory(v => !v)}
              title={t('devTestPanel.chat.historyTooltip')}
              className={`p-1.5 rounded-lg transition-all ${
                showHistory ? 'bg-violet-500/20 text-violet-300' : 'hover:bg-white/5 text-gray-500 hover:text-white'
              }`}
            >
              <History className="w-3.5 h-3.5" />
            </button>
            <button onClick={startNewSession} title={t('devTestPanel.chat.newChatTooltip')} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all">
              <MessageSquarePlus className="w-3.5 h-3.5" />
            </button>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all ml-0.5"><X className="w-3.5 h-3.5" /></button>
          </div>
        </div>

        {sessionTitle && sessionTitle !== t('devTestPanel.chat.newChatTitle') && (
          <div className="px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.01] flex items-center gap-1.5">
            <span className="text-[9px] text-gray-600">↳</span>
            <span className="text-[10px] text-gray-500 truncate">{sessionTitle}</span>
            {isReadonly && <span className="ml-auto flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400/80">{t('devTestPanel.chat.readonlyBadge')}</span>}
          </div>
        )}

        {showHistory && (
          <div className="absolute inset-x-0 top-[41px] z-10 bg-slate-950 border-b border-white/10 flex flex-col shadow-xl" style={{ maxHeight: '60%', overflowY: 'auto' }}>
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
              <span className="text-[10px] font-medium text-gray-400">{t('devTestPanel.chat.historyTitle')}</span>
              <button onClick={startNewSession} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 text-violet-300 text-[10px] transition-all">
                <MessageSquarePlus className="w-3 h-3" /> {t('devTestPanel.chat.newChatButton')}
              </button>
            </div>
            <div className="overflow-y-auto flex-1">
              {loadChatSessions().length === 0 ? (
                <p className="text-center text-gray-600 text-[10px] py-6">{t('devTestPanel.chat.emptyHistory')}</p>
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
                      <p className={`text-[11px] truncate font-medium ${isActive ? 'text-violet-200' : 'text-gray-300'}`}>{session.title}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[9px] text-gray-600">{relativeTime(session.updatedAt)}</span>
                        <span className="text-[9px] text-gray-700">· {t('devTestPanel.chat.messagesCount').replace('{count}', String(session.messages.length))}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                      {!isActive && session.messages.length > 0 && (
                        <button onClick={e => { e.stopPropagation(); continueFromSession(session); }}
                          className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-all">
                          {t('devTestPanel.chat.continueButton')}
                        </button>
                      )}
                      <button onClick={e => deleteSession(session.id, e)}
                        className="p-0.5 rounded hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all">
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {messages.length === 0 && (
            <div className="py-6 space-y-3">
              <p className="text-gray-400 text-xs">{t('devTestPanel.chat.emptySidebarHint')}</p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.map(s => (
                  <button key={s} onClick={() => setInput(s)} className="px-2.5 py-1 rounded-lg border text-[10px] transition-all bg-purple-500/10 border-purple-500/20 text-purple-200 hover:bg-purple-500/15">{s}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-xs ${
                m.role === 'user'
                  ? 'bg-amber-500/20 text-amber-400'
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
                {removeEditBlocks(m.content).split(/(```python[\s\S]*?```)/g).map((part, pi) => {
                  if (part.startsWith('```python')) {
                    const code = extractFullPythonCode(part) ?? part;
                    return (
                      <div key={pi} className="w-full rounded-xl overflow-hidden border border-white/10">
                        <div className="flex items-center justify-between px-3 py-1.5 bg-white/[0.03] border-b border-white/10">
                          <span className="text-[10px] text-gray-500 font-mono">{t('trainingPanel.requirements.python')}</span>
                          <button onClick={() => onReplaceScript(code)} className="text-[10px] px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-all">{t('devTestPanel.aiSidebar.replaceCodeButton')}</button>
                        </div>
                        <pre className="p-3 text-[10px] font-mono text-gray-300 overflow-x-auto max-h-48 leading-relaxed">{code}</pre>
                      </div>
                    );
                  }
                  return part.trim() ? (
                    <div key={pi} className={`px-3 py-2 rounded-xl text-[11px] leading-relaxed whitespace-pre-wrap break-words ${
                      m.role === 'user'
                        ? 'bg-amber-500/10 text-gray-200 border border-amber-500/20'
                        : (m.action?.mode === 'edit')
                          ? 'bg-amber-500/[0.06] text-gray-300 border border-amber-500/15'
                          : (m.action?.mode === 'rewrite')
                            ? 'bg-purple-500/[0.08] text-gray-200 border border-purple-500/20'
                            : 'bg-white/[0.05] text-gray-300 border border-white/10'
                    }`}>{part.trim()}</div>
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
                              {t('devTestPanel.chat.changeLabel').replace('{n}', String(editIdx + 1))}
                            </span>
                            {isApplied ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleUndoEdit(messageIdx, edit.id);
                                }}
                                className="text-emerald-400/70 hover:text-emerald-300 text-xs flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/[0.15] hover:bg-emerald-500/25 transition-all"
                              >
                                <span>{t('devTestPanel.chat.undoButton')}</span>
                              </button>
                            ) : (
                              <span className="text-amber-400/70 text-xs">{t('devTestPanel.chat.viewDiffLink')}</span>
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
              <div className="px-3 py-2 rounded-xl bg-white/5 border border-white/10"><Loader2 className="w-4 h-4 text-violet-400 animate-spin" /></div>
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
                  <span className="text-[10px] text-gray-500 shrink-0">{t('devTestPanel.chat.editSummaryReady')}</span>
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
                    <Check className="w-3 h-3" /> {t('devTestPanel.chat.applyButton')}
                  </button>
                  <button
                    onClick={() => {
                      setCurrentMessageWithEdits(latestEditMsg);
                      setShowDiffModal(true);
                    }}
                    className="flex items-center justify-center px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-[10px] font-medium transition-all"
                  >
                    {t('devTestPanel.chat.detailsButton')}
                  </button>
                </div>
              </div>
            );
          })()}
          
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-gray-600">
              {isReadonly ? t('devTestPanel.chat.readonlyHint') : t('devTestPanel.chat.sendHint')}
            </span>
            <span className="text-[10px] text-purple-300/70">{t('devTestPanel.chat.autoBadgeLabel')}</span>
          </div>
          <div className="flex gap-2 items-end">
            <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={isReadonly ? t('devTestPanel.chat.readonlyPlaceholder') : t('devTestPanel.chat.inputPlaceholder')} rows={2}
              disabled={isReadonly}
              className={`flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none transition-opacity ${
                isReadonly ? 'opacity-40 cursor-not-allowed' : ''
              }`} />
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
              <MessageSquarePlus className="w-3 h-3" /> {t('devTestPanel.chat.startNewChatButton')}
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

// ── Error Modal ────────────────────────────────────────────────────────────

function DevTestErrorModal({ isOpen, errorTitle, errorMessage, errorDetails, script, output, onClose, onSendToAI, isSending }: {
  isOpen: boolean; errorTitle: string; errorMessage: string; errorDetails: string;
  script: string; output: string; onClose: () => void; onSendToAI: (ctx: string) => void; isSending?: boolean;
}) {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);
  if (!isOpen) return null;
  const ctx = `[Dev Test Fehler]\n\nTitel: ${errorTitle}\n\nFehler: ${errorMessage}\n\nDetails: ${errorDetails}\n\nSkript:\n${script}\n\nAusgabe:\n${output}`;
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-red-500/10 flex-shrink-0">
          <div className="flex items-center gap-3"><span className="text-3xl">❌</span><div><h2 className="text-lg font-bold text-white">{t('devTestPanel.errorModal.title')}</h2><p className="text-sm text-red-300">{errorTitle}</p></div></div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {errorMessage && <div><p className="text-xs text-gray-500 font-medium mb-2">{t('devTestPanel.errorModal.errorLabel')}</p><div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg max-h-24 overflow-auto"><pre className="text-xs text-red-300 font-mono whitespace-pre-wrap">{errorMessage}</pre></div></div>}
          {errorDetails && <div><p className="text-xs text-gray-500 font-medium mb-2">{t('devTestPanel.errorModal.detailsLabel')}</p><div className="p-3 bg-white/5 border border-white/10 rounded-lg max-h-24 overflow-auto"><pre className="text-xs text-gray-400 font-mono whitespace-pre-wrap">{errorDetails}</pre></div></div>}
        </div>
        <div className="px-6 py-4 border-t border-white/10 flex gap-3 flex-shrink-0">
          <button onClick={() => { navigator.clipboard.writeText(ctx); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all">
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}{copied ? t('devTestPanel.errorModal.copied') : t('devTestPanel.errorModal.copyButton')}
          </button>
          <button onClick={() => onSendToAI(ctx)} disabled={isSending}
            className="flex items-center gap-2 px-4 py-2 bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 rounded-lg text-sm text-violet-300 transition-all disabled:opacity-50">
            <Sparkles className="w-4 h-4" /> {t('devTestPanel.errorModal.sendToAIButton')}
          </button>
          <button onClick={onClose} className="ml-auto px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all">{t('devTestPanel.errorModal.closeButton')}</button>
        </div>
      </div>
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
    <div className="flex items-start gap-3 py-0.5 text-[11px] font-mono">
      <span className={`${color} min-w-[140px] flex-shrink-0`}>{label}</span>
      <div className="min-w-0 flex-1">
        <span className={`break-all ${value ? 'text-gray-300' : 'text-gray-600 italic'}`}>{value || t('devTestPanel.paths.notSet')}</span>
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
          title={t('devTestPanel.paths.copyTooltip')}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      )}
    </div>
  );
}

// ── Default Script Generator ──────────────────────────────────────────────

type ScriptModality = 'text' | 'image' | 'audio' | 'seq2seq';

/** Welche Art Code braucht dieses Modell? Gleiche Erkennung wie im Training. */
function detectScriptModality(model: ModelInfo | null): ScriptModality {
  if (!model) return 'text';
  const r = detectPlugin(
    model.source_path || model.local_path || model.name,
    model.model_type ? { model_type: model.model_type } : undefined,
  );
  if (!r.supported) return 'text';
  switch (r.plugin.taskType) {
    case 'hf_image_classification':
    case 'image_classification':
      return 'image';
    case 'audio_classification':
      return 'audio';
    case 'seq2seq':
      return 'seq2seq';
    default:
      return 'text';
  }
}

function generateDefaultTestScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const modality = detectScriptModality(model);
  if (modality !== 'text') {
    return generateMediaTestScript(modality, model, datasets, outputPath);
  }
  return generateTextTestScript(model, datasets, outputPath);
}

/**
 * Bild-, Audio- und Seq2Seq-Skripte. Sie laden dieselben Daten wie die
 * Test-Engine-Plugins: Bild/Audio als Ordner pro Klasse, Seq2Seq als
 * Tabellendatei mit Quell- und Zielspalte.
 */
function generateMediaTestScript(
  modality: Exclude<ScriptModality, 'text'>,
  model: ModelInfo | null,
  datasets: DatasetInfo[],
  outputPath: string,
): string {
  const ds = datasets[0];
  const modelPathDefault   = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_test').replace('{wird beim Start gesetzt}', 'dev_test');

  if (modality === 'seq2seq') {
    return `#!/usr/bin/env python3
# FrameTrain - Dev Test Script (Seq2Seq)
#
# Laedt Modell + Test-Split, erzeugt Text und vergleicht ihn mit der Zielspalte.

import csv
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

MAX_SAMPLES    = 100
MAX_NEW_TOKENS = 64
TASK_PREFIX    = ""     # z.B. "summarize: " fuer T5
SOURCE_COL     = None   # None = automatisch erkennen
TARGET_COL     = None

EXTS = (".csv", ".tsv", ".json", ".jsonl", ".parquet")


def load_rows(path: str):
    # Erste Tabellendatei aus test/, sonst val/ bzw. train/.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")

    files = []
    for sub in ("test", "val", "validation", "train", ""):
        d = p / sub if sub else p
        if d.is_dir():
            files = sorted(f for f in d.rglob("*") if f.suffix.lower() in EXTS)
        if files:
            break
    if not files:
        raise RuntimeError(f"Keine Tabellendatei in {path} gefunden.")

    f = files[0]
    print(f"Datei: {f}", flush=True)
    if f.suffix.lower() == ".parquet":
        import pandas as pd
        return pd.read_parquet(f).to_dict("records")
    if f.suffix.lower() in (".json", ".jsonl"):
        raw = f.read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            return json.loads(raw)
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    delim = "\\t" if f.suffix.lower() == ".tsv" else ","
    with open(f, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


rows = load_rows(DATASET_PATH)
if MAX_SAMPLES:
    rows = rows[:MAX_SAMPLES]
cols = list(rows[0].keys()) if rows else []
source_col = SOURCE_COL or next((c for c in ("source", "input", "text", "article", "document", "de") if c in cols), None)
target_col = TARGET_COL or next((c for c in ("target", "output", "summary", "translation", "en") if c in cols), None)
if source_col is None:
    raise RuntimeError(f"Quellspalte nicht gefunden. Vorhanden: {cols}. Setze SOURCE_COL oben.")
print(f"{len(rows)} Beispiele | Quelle='{source_col}' | Ziel='{target_col}'", flush=True)

print(f"Lade Modell aus: {MODEL_PATH}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    else "cpu"
)
model.to(device)
print(f"Geraet: {device}", flush=True)

predictions, exact = [], 0
for i, row in enumerate(rows, 1):
    src = TASK_PREFIX + str(row[source_col])
    inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    tgt = str(row[target_col]).strip() if target_col else None
    if tgt is not None and pred == tgt:
        exact += 1
    predictions.append({"source": str(row[source_col]), "predicted": pred, "target": tgt})
    if i <= 5:
        print(f"  [{i}] {pred}   (erwartet: {tgt})", flush=True)
    if i % 20 == 0:
        print(f"  {i}/{len(rows)} ausgewertet", flush=True)

results = {"predictions": predictions, "n": len(predictions), "device": device}
if target_col:
    results["exact_match"] = exact / len(predictions) if predictions else 0.0
    print(f"\\nExakte Treffer: {results['exact_match']:.4f}", flush=True)
else:
    print("Keine Zielspalte gefunden - nur Vorhersagen, keine Metrik.", flush=True)

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
with open(f"{OUTPUT_PATH}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Ergebnisse gespeichert: {OUTPUT_PATH}/results.json", flush=True)
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
  const loadModel = isImage
    ? `processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)`
    : `processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
SAMPLING_RATE = int(getattr(processor, "sampling_rate", 16000) or 16000)`;
  const prepareInputs = isImage
    ? `    with Image.open(path) as im:
        image = im.convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)`
    : `    wave, _ = librosa.load(str(path), sr=SAMPLING_RATE, mono=True)
    wave = wave[:int(MAX_SECONDS * SAMPLING_RATE)]
    inputs = processor([wave], sampling_rate=SAMPLING_RATE,
                       return_tensors="pt", padding=True).to(device)`;
  const extraConfig = isImage ? '' : 'MAX_SECONDS = 10.0   # laengere Aufnahmen werden gekappt\n';

  return `#!/usr/bin/env python3
# FrameTrain - Dev Test Script (${kindLabel}klassifikation)
#
# Erwartet den Trainings-Aufbau: ein Ordner pro Klasse, optional unter
# train/ val/ test/. Schreibt einen Bericht nach OUTPUT_PATH/results.json.

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
${imports}

MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

MAX_SAMPLES = 200    # None = alles auswerten
${extraConfig}EXTS = ${exts}


def collect_files(root: Path):
    # Dateien samt erwarteter Klasse (Ordnername). Bevorzugt test/, dann val/, train/.
    for sub in ("test", "val", "validation", "train"):
        if (root / sub).is_dir():
            root = root / sub
            break
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    files = []
    if class_dirs:
        for d in class_dirs:
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EXTS:
                    files.append((f, d.name))
    else:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in EXTS:
                files.append((f, None))
    return files


ds_path = Path(DATASET_PATH)
if not ds_path.exists():
    raise FileNotFoundError(f"DATASET_PATH existiert nicht: {DATASET_PATH}")

files = collect_files(ds_path)
if not files:
    raise RuntimeError(f"Keine ${kindLabel}dateien in {DATASET_PATH} gefunden.")
if MAX_SAMPLES:
    files = files[:MAX_SAMPLES]
print(f"{len(files)} ${kindLabel}dateien gefunden", flush=True)

print(f"Lade Modell aus: {MODEL_PATH}", flush=True)
${loadModel}
model.eval()
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    else "cpu"
)
model.to(device)
print(f"Geraet: {device}", flush=True)

# Klassennamen: label_mapping.json des Trainings, sonst config.json
id2label = {}
mapping = Path(MODEL_PATH) / "label_mapping.json"
if mapping.exists():
    lm = json.loads(mapping.read_text(encoding="utf-8"))
    if isinstance(lm.get("id2label"), dict):
        id2label = {int(k): str(v) for k, v in lm["id2label"].items()}
    elif isinstance(lm.get("classes"), list):
        id2label = {i: str(c) for i, c in enumerate(lm["classes"])}
if not id2label:
    id2label = {int(k): str(v) for k, v in (getattr(model.config, "id2label", {}) or {}).items()}
label2id = {v: k for k, v in id2label.items()}
print(f"Klassen: {list(id2label.values()) or '?'}", flush=True)

preds, targets, rows = [], [], []
for i, (path, cls) in enumerate(files, 1):
${prepareInputs}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(torch.argmax(probs))
    preds.append(pred_id)
    if cls is not None and cls in label2id:
        targets.append(label2id[cls])
    rows.append({
        "file": str(path),
        "expected": cls,
        "predicted": id2label.get(pred_id, str(pred_id)),
        "confidence": float(probs[pred_id]),
    })
    if i % 20 == 0 or i == len(files):
        print(f"  {i}/{len(files)} ausgewertet", flush=True)

results = {"predictions": rows, "n": len(rows), "device": device}

if len(targets) == len(preds) and targets:
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    names = [id2label.get(i, str(i)) for i in sorted(id2label)]
    print(f"\\nAccuracy: {acc:.4f}\\n", flush=True)
    print(classification_report(targets, preds, target_names=names, zero_division=0), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(targets, preds), flush=True)
    results["accuracy"] = acc
    results["report"] = classification_report(
        targets, preds, target_names=names, zero_division=0, output_dict=True)
else:
    print("Keine Klassenordner erkannt - nur Vorhersagen, keine Metriken.", flush=True)

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
with open(f"{OUTPUT_PATH}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Ergebnisse gespeichert: {OUTPUT_PATH}/results.json", flush=True)
`;
}

function generateTextTestScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const ds = datasets[0];
  const modelPathDefault   = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_test').replace('{wird beim Start gesetzt}', 'dev_test');

  return `#!/usr/bin/env python3
# FrameTrain - Dev Test Script
#
# Laeuft so wie es ist: laedt Modell + Test-Split, macht Inference und
# schreibt einen Bericht nach OUTPUT_PATH/results.json.

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -- Pfade (von FrameTrain als ENV-Vars gesetzt) --------------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

BATCH_SIZE = 16
MAX_LENGTH = 128
MAX_SAMPLES = 500   # None = alles auswerten
TEXT_COL   = None   # None = automatisch erkennen
LABEL_COL  = None   # None = automatisch erkennen


def load_frametrain_dataset(path: str):
    """Laedt ein FrameTrain-Dataset (save_to_disk oder Split-Ordner)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")

    if p.is_dir() and any((p / m).exists() for m in
                          ("dataset_info.json", "state.json", "dataset_dict.json")):
        return load_from_disk(str(p))

    EXTS = (".json", ".jsonl", ".csv", ".tsv", ".parquet")

    def files_in(sub: str):
        d = p / sub
        return sorted(f for f in d.rglob("*") if f.suffix.lower() in EXTS) if d.is_dir() else []

    data_files = {}
    for split, subs in (("train", ["train"]),
                        ("validation", ["val", "validation"]),
                        ("test", ["test"])):
        for sub in subs:
            found = files_in(sub)
            if found:
                data_files[split] = [str(f) for f in found]
                break

    if not data_files:
        loose = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        if not loose:
            raise RuntimeError(f"Keine Daten-Dateien in {path} gefunden.")
        data_files["train"] = [str(f) for f in loose]

    ext = Path(next(iter(data_files.values()))[0]).suffix.lower()
    if ext in (".json", ".jsonl"):
        return load_dataset("json", data_files=data_files)
    if ext == ".parquet":
        return load_dataset("parquet", data_files=data_files)
    if ext == ".tsv":
        return load_dataset("csv", data_files=data_files, delimiter="\\t")
    return load_dataset("csv", data_files=data_files)


print(f"Lade Dataset aus: {DATASET_PATH}", flush=True)
dataset = load_frametrain_dataset(DATASET_PATH)

# Bevorzugt der Test-Split, sonst validation, sonst train.
eval_ds = dataset.get("test") or dataset.get("validation") or dataset["train"]

cols = list(eval_ds.features.keys())
text_col = TEXT_COL or next(
    (c for c in ("text", "sentence", "content", "review", "input") if c in cols), None)
label_col = LABEL_COL or next(
    (c for c in ("label", "labels", "target", "class") if c in cols), None)
if text_col is None:
    raise RuntimeError(f"Text-Spalte nicht gefunden. Vorhanden: {cols}. Setze TEXT_COL oben.")

if MAX_SAMPLES and len(eval_ds) > MAX_SAMPLES:
    eval_ds = eval_ds.select(range(MAX_SAMPLES))
print(f"Auswertung auf {len(eval_ds)} Beispielen (Spalte '{text_col}')", flush=True)

print(f"Lade Modell aus: {MODEL_PATH}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    else "cpu"
)
model.to(device)
print(f"Geraet: {device}", flush=True)

# -- Inference ------------------------------------------------------------
all_preds = []
texts = eval_ds[text_col]
for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)} ausgewertet", flush=True)

results = {"predictions": all_preds, "n": len(all_preds), "device": device}

# -- Bericht, falls Labels vorhanden --------------------------------------
if label_col is not None:
    all_labels = list(eval_ds[label_col])
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\\nAccuracy: {acc:.4f}\\n", flush=True)
    print(classification_report(all_labels, all_preds, zero_division=0), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(all_labels, all_preds), flush=True)
    results["labels"] = all_labels
    results["accuracy"] = acc
    results["report"] = classification_report(
        all_labels, all_preds, zero_division=0, output_dict=True)
else:
    print("Keine Label-Spalte gefunden - nur Vorhersagen, keine Metriken.", flush=True)

# -- Speichern ------------------------------------------------------------
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
with open(f"{OUTPUT_PATH}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Ergebnisse gespeichert: {OUTPUT_PATH}/results.json", flush=True)
`;
}

// ── DevTestPanel ──────────────────────────────────────────────────────────

interface DevTestPanelProps {
  modelInfo: ModelInfo | null;
  selectedVersionPath?: string;
  datasets: DatasetInfo[];
  userData?: { userId: string; email: string; apiKey: string; password: string };
}

export default function DevTestPanel({ modelInfo, selectedVersionPath, datasets, userData }: DevTestPanelProps) {
  // Globale Legacy-Scripts einmalig in den User-Key übernehmen
  useEffect(() => { migrateLegacyDevScripts(userData?.userId); }, [userData?.userId]);
  const { currentTheme } = useTheme();
  const { success, error } = useNotification();
  const { settings: aiSettings } = useAISettings();
  const { setCurrentPageContent } = usePageContext();
  const { t } = useLanguage();

  const [fileOpen, setFileOpen]       = useState(false);
  const [tlHovered, setTlHovered]     = useState(false);
  const [script, setScript]           = useState('');
  const [savedScript, setSavedScript] = useState('');
  const [isDirty, setIsDirty]         = useState(false);
  const [currentScriptId, setCurrentScriptId] = useState<string | null>(null);
  const [showSaveDialog, setShowSaveDialog]    = useState(false);
  const [saveName, setSaveName]               = useState('');
  const [showAI, setShowAI]                   = useState(false);
  const [showLibrary, setShowLib]             = useState(false);
  const [showOpenLib, setShowOpenLib]         = useState(false);
  const [running, setRunning]                 = useState(false);
  const [output, setOutput]                   = useState('');
  const [exitCode, setExitCode]               = useState<number | null>(null);
  const [dismissed, setDismissed]             = useState(() => {
    try {
      return localStorage.getItem('devTestBannerDismissed') === 'true';
    } catch {
      return false;
    }
  });
  const [expanded, setExpanded]               = useState(false);
  const [editorH, setEditorH]                 = useState(500);
  const [outputPath, setOutputPath]           = useState('');
  const [aiPrefill, setAiPrefill]             = useState('');
  const [showPathsModal, setShowPathsModal]   = useState(false);

  // Error Modal States
  const [showErrorModal, setShowErrorModal]   = useState(false);
  const [errorTitle, setErrorTitle]           = useState('');
  const [errorMessage, setErrorMessage]       = useState('');
  const [errorDetails, setErrorDetails]       = useState('');

  const outputRef     = useRef<HTMLDivElement>(null);
  const editorRef     = useRef<HTMLTextAreaElement>(null);
  const editorPreRef  = useRef<HTMLPreElement>(null);
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

  const lineCount     = useMemo(() => Math.max(1, (script || '').split('\n').length), [script]);
  const highlightedHtml = useMemo(() => highlightPythonToHtml(script || ''), [script]);

  const modelPath = selectedVersionPath || modelInfo?.local_path || modelInfo?.source_path || modelInfo?.name || '';
  const dsRefs    = datasets.map((d, i) => ({
    key:   i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`,
    value: d.storage_path || '',
    name:  d.name,
  }));

  // ── AI Coach Page Context ──────────────────────────────────────────────────
  useEffect(() => {
    const lines: string[] = [
      t('devTestPanel.pageContext.title'),
      '',
      t('devTestPanel.pageContext.purposeBody'),
      '',
      t('devTestPanel.pageContext.currentStateTitle'),
      `Status: ${running ? '🔄 Skript läuft' : isDirty ? '✏️ Editor: Änderungen' : '✓ Bereit'}`,
      `Modell: ${modelInfo?.name || '(nicht geladen)'}`,
      `Skript-Größe: ${lineCount} Zeilen, ${script.length} Zeichen`,
      running ? `Output: ${output.split('\n').length} Zeilen` : output ? `Letzter Output: ${output.split('\n').length} Zeilen` : 'Kein Output',
      exitCode !== null ? `Exit Code: ${exitCode}` : '',
      '',
      t('devTestPanel.pageContext.scriptStateTitle'),
      isDirty ? '⚠️ Unsaved changes in editor' : '✓ Script gespeichert',
      currentScriptId ? `📂 Loaded: ${currentScriptId}` : '📋 Neu/Unsaved',
      running ? '🔄 Execution läuft' : `⏸️ Idle${output ? ' (mit Output)' : ''}`,
      '',
      t('devTestPanel.pageContext.layoutTitle'),
      '**OBEN:**',
      `  • [Modell Badge] (${modelInfo?.name || 'keine'})`,
      '  • [💾 Speichern Button] (grün wenn dirty)',
      '  • [📁 Bibliothek] (Saved scripts)',
      '  • [▼ Pfade] (DATASET_PATH, OUTPUT_PATH)',
      '',
      '**LINKS:**',
      '  • Python-Editor mit Syntax-Highlighting',
      '  • Zeilennummern + Gutter',
      '  • [🧪 Ausführen Button] (Ctrl+Enter)',
      '',
      '**RECHTS:**',
      '  • Output/Logs Panel mit Scrollbar',
      `  • ${running ? 'Live Output' : 'Letzter Output'}`,
      '  • [📋 Copy] [🗑️ Clear] Buttons',
      '',
      '**UNTEN:**',
      '  • [💬 Chat mit KI] (Code + Output als Context)',
      '',
      t('devTestPanel.pageContext.availableActionsTitle'),
      !script.trim() ? '1. Schreib Python-Code in Editor links' : '1. Passe Code an (optional)',
      '2. Klick 🧪 [Ausführen] oder Ctrl+Enter',
      '3. Überwache Output rechts',
      output && !running ? '4. Bei Fehler: Klick 💬 [Chat mit KI] → AI liefert Fixes' : '',
      '5. Speichere Skript: 💾 [Speichern] oder Neue Bibliothek anlegen',
      '',
      t('devTestPanel.pageContext.contextTitle'),
      `Modell: ${modelInfo?.name || '(keine)'}`,
      `Datasets: ${datasets.length} verfügbar (${datasets.length > 0 ? datasets.map(d => d.name).join(', ') : 'keine'})`,
      `Output-Pfad: ${outputPath || '[AppData]/test_outputs'}`,
      `Verfügbare Paths: ${dsRefs.map(d => d.key).join(', ')}`,
    ];

    setCurrentPageContent(lines.join('\n'), 'tests-dev');
  }, [script, lineCount, running, isDirty, output, exitCode, modelInfo, datasets, currentScriptId, outputPath, dsRefs, setCurrentPageContent]);

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

    // Custom cursor position
    const textBeforeCaret = ta.value.slice(0, caret);
    const lastNewlineIdx = textBeforeCaret.lastIndexOf('\n');
    const textOnLine = lastNewlineIdx === -1 ? textBeforeCaret : textBeforeCaret.slice(lastNewlineIdx + 1);

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.font = `${parseFloat(window.getComputedStyle(ta).fontSize)}px JetBrains Mono, Fira Code, Cascadia Code, Courier New, monospace`;
      const metrics = ctx.measureText(textOnLine);
      setCursorX(editorPadLeftPx + metrics.width);
    }
    setCursorY(editorPadTopPx + (line - 1) * editorLineHeightPx - ta.scrollTop);
  };

  useEffect(() => { syncEditorScroll(); updateActiveLine(); }, [fileOpen, expanded, script]);

  useEffect(() => {
    const ta = editorRef.current;
    if (!ta) return;
    const cs = window.getComputedStyle(ta);
    const pt = parseFloat(cs.paddingTop || '0');
    const pl = parseFloat(cs.paddingLeft || '0');
    if (Number.isFinite(pt) && pt >= 0) setEditorPadTopPx(pt);
    if (Number.isFinite(pl) && pl >= 0) setEditorPadLeftPx(pl);
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
    const line = (text.slice(0, at).match(/\n/g)?.length ?? 0) + 1;
    const targetTop = (line - 1) * editorLineHeightPx;
    const pad = editorPadTopPx;
    if (ta.scrollTop > targetTop) ta.scrollTop = Math.max(0, targetTop - pad);
    else if (ta.scrollTop + ta.clientHeight < targetTop + editorLineHeightPx + pad) ta.scrollTop = Math.max(0, targetTop - pad);
    syncEditorScroll();
    updateFindStatus(q, at);
  }, [findQuery, editorLineHeightPx, editorPadTopPx, updateFindStatus]);

  useEffect(() => {
    invoke<string>('get_app_data_dir')
      .then(dir => setOutputPath(`${dir}/test_outputs/dev_<job_id>`))
      .catch(() => setOutputPath('[AppData]/test_outputs/dev_<job_id>'));
  }, []);

  // ── Event Listener: dev-test-output ──────────────────────────────────────

  useEffect(() => {
    let u1: (() => void) | undefined;
    let u2: (() => void) | undefined;

    listen<{ line: string }>('dev-test-output', e => {
      setOutput(o => o + e.payload.line + '\n');
      setTimeout(() => outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight }), 50);
    }).then(fn => { u1 = fn; });

    listen<{ data?: { error?: string; details?: string }; exit_code?: number }>('dev-test-complete', e => {
      setRunning(false);
      const code = e.payload?.exit_code ?? 0;
      setExitCode(code);
      if (code === 0) {
        setOutput(o => o + `\n${t('devTestPanel.output.complete')}`);
        invoke('disable_prevent_sleep').catch(() => {});
      } else {
        const msg = e.payload?.data?.error ?? `Prozess beendet mit Exit-Code ${code}`;
        const details = e.payload?.data?.details ?? '';
        setOutput(o => o + `\n${t('devTestPanel.output.errorPrefix')} ${msg}${details ? '\n' + details : ''}`);
        invoke('disable_prevent_sleep').catch(() => {});
      setErrorTitle(t('devTestPanel.errorModal.title'));
        setErrorMessage(msg);
        setErrorDetails(details);
        setShowErrorModal(true);
      }
    }).then(fn => { u2 = fn; });

    return () => { u1?.(); u2?.(); };
  }, []);

  // ── Datei-Aktionen ─────────────────────────────────────────────────────

  const handleNewFile = () => {
    setScript(''); setSavedScript(''); setCurrentScriptId(null); setIsDirty(false);
    setOutput(''); setExitCode(null); setFileOpen(true);
  };

  const handleCloseFile = () => {
    if (isDirty) { error(t('devTestPanel.notifications.unsavedTitle'), t('devTestPanel.notifications.unsavedDetail')); return; }
    setFileOpen(false); setScript(''); setSavedScript(''); setCurrentScriptId(null); setExpanded(false);
  };

  const generateTemplate = () => {
    if (!modelInfo || !outputPath) return;
    setScript(generateDefaultTestScript(modelInfo, datasets, outputPath));
    setIsDirty(true);
  };

  const handleSave = () => {
    if (currentScriptId) {
      updateScript(currentScriptId, script, userData?.userId);
      setSavedScript(script); setIsDirty(false);
    success(t('devTestPanel.notifications.updatedTitle'), t('devTestPanel.notifications.updatedDetail'));
    } else {
      setSaveName('Mein Test-Skript');
      setShowSaveDialog(true);
    }
  };

  const handleSaveWithName = (name: string) => {
    if (!name.trim()) return;
    saveScript(name.trim(), script, userData?.userId);
    const newScript = loadScripts(userData?.userId)[0];
    if (newScript) setCurrentScriptId(newScript.id);
    setSavedScript(script); setIsDirty(false); setShowSaveDialog(false); setSaveName('');
    success(t('devTestPanel.notifications.savedTitle'), t('devTestPanel.notifications.savedDetail').replace('{name}', name));
  };

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

  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => { if (isDirty) { e.preventDefault(); e.returnValue = ''; return ''; } };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isDirty]);

  // ── Test starten / stoppen ─────────────────────────────────────────────

  const handleStart = async () => {
    if (isDirty) { error(t('devTestPanel.notifications.unsavedTitle'), t('devTestPanel.notifications.unsavedDetail')); return; }
    if (!script.trim() || !modelInfo) { error(t('devTestPanel.notifications.noModelOrScriptTitle'), t('devTestPanel.notifications.noModelOrScriptDetail')); return; }

    setRunning(true); setOutput(''); setExitCode(null);

    const refs: Record<string, string> = {
      MODEL_PATH: modelPath,
      ...Object.fromEntries(dsRefs.map(r => [r.key, r.value])),
    };

    try {
      await invoke('start_dev_test', {
        script,
        modelId:     modelInfo!.id,
        modelName:   modelInfo!.name,
        datasetId:   datasets[0]?.id ?? '',
        datasetName: datasets[0]?.name ?? '',
        refs,
      });
      setOutput(`${t('devTestPanel.output.started')}\n`);
      invoke('enable_prevent_sleep').catch(() => {});
      success(t('devTestPanel.notifications.startedTitle'), t('devTestPanel.notifications.startedDetail'));
    } catch (err: unknown) {
      setOutput(`${t('devTestPanel.output.errorPrefix')} ${String(err)}`);
      setRunning(false);
      error(t('common.error'), String(err));
    }
  };

  const handleStop = async () => {
    try { await invoke('stop_dev_test'); } catch { /* ignore */ }
    invoke('disable_prevent_sleep').catch(() => {});
    setRunning(false);
    setOutput(o => o + `\n${t('devTestPanel.output.stopped')}`);
  };

  const handleSendToAI = (errorContext: string) => {
    setShowErrorModal(false);
    setShowAI(true);
    setAiPrefill(
      `[Dev Test Fehler]\n\nBitte hilf mir, meinen Dev-Test Run zu reparieren.\n\nFEHLER:\n${errorContext}\n\nDu darfst Edits vorschlagen oder den Code neu schreiben – wähle selbst die beste Vorgehensweise.`
    );
  };

  const isRunning = running;

  return (
    <div className={`flex gap-0 ${expanded ? 'fixed inset-0 z-40 bg-slate-950 p-4' : ''}`}>
      <div className={`flex-1 space-y-4 ${expanded ? 'overflow-y-auto pr-2' : ''}`}>

        {/* Info Banner */}
        {!dismissed && (
          <div className="p-4 rounded-2xl border border-amber-500/30 bg-amber-500/10">
            <div className="flex items-start justify-between gap-2 mb-1">
              <div className="flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-amber-400" />
                <span className="text-amber-300 font-semibold text-sm">{t('devTestPanel.banner.title')}</span>
              </div>
              <button
                onClick={() => {
                  setDismissed(true);
                  try { localStorage.setItem('devTestBannerDismissed', 'true'); } catch { /* ignore */ }
                }}
                className="p-1 rounded-lg hover:bg-white/10 text-amber-400/60 hover:text-white transition-all"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <p className="text-gray-400 text-xs">{t('devTestPanel.banner.description')}</p>
          </div>
        )}

        {/* Paths — Collapsible (wie DevTrain) */}
        <button
          onClick={() => setShowPathsModal(v => !v)}
          className="w-full px-4 py-3 rounded-2xl border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/15 transition-all flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-300">{t('devTestPanel.paths.toggleLabel')}</span>
          </div>
          <div className={`transform transition-transform ${showPathsModal ? 'rotate-180' : ''}`}>
            <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </button>

        {showPathsModal && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-6">
            {/* Model */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-white">{t('devTestPanel.paths.modelTitle')}</span>
              </div>
              <RefRow color="text-emerald-400" label="MODEL_PATH" value={modelPath} hint={modelInfo?.name} />
            </div>

            <div className="border-t border-white/10" />

            {/* Dataset */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-white">{t('devTestPanel.paths.datasetTitle')}</span>
              </div>
              {dsRefs.map(r => <RefRow key={r.key} color="text-blue-400" label={r.key} value={r.value} hint={r.name} />)}
            </div>

            <div className="border-t border-white/10" />

            {/* Output */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">{t('devTestPanel.paths.outputTitle')}</span>
              </div>
              <RefRow color="text-purple-400" label="OUTPUT_PATH" value={outputPath.replace('<job_id>', '{wird beim Start gesetzt}')} />
            </div>
          </div>
        )}

        {/* Code Editor */}
        <div className={`rounded-2xl border border-white/10 overflow-hidden ${expanded ? 'flex-1 flex flex-col' : ''}`}>
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-slate-900">
            <div className="flex items-center gap-3">
              <div className="flex gap-1.5" onMouseEnter={() => setTlHovered(true)} onMouseLeave={() => setTlHovered(false)}>
                {/* Rot: Schließen */}
                <button onClick={fileOpen ? handleCloseFile : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${fileOpen ? 'bg-red-500 cursor-pointer hover:bg-red-400' : 'bg-red-500/40 cursor-default'}`}
                  title={fileOpen ? (isDirty ? t('devTestPanel.toolbar.unsavedChangesTooltip') : t('devTestPanel.toolbar.closeFileTooltip')) : ''}>
                  {tlHovered && fileOpen && <X className="w-[7px] h-[7px] text-red-900 stroke-[3]" />}
                  {!tlHovered && isDirty && fileOpen && <div className="w-[5px] h-[5px] rounded-full bg-red-900" />}
                </button>
                {/* Gelb: Speichern */}
                <button onClick={fileOpen && isDirty ? handleSave : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${fileOpen && isDirty ? 'bg-amber-400 cursor-pointer hover:bg-amber-300' : 'bg-amber-500/40 cursor-default'}`}
                  title={fileOpen && isDirty ? t('devTestPanel.toolbar.saveTooltip') : ''}>
                  {tlHovered && fileOpen && isDirty && <Minus className="w-[7px] h-[7px] text-amber-900 stroke-[3]" />}
                </button>
                {/* Grün: Vollbild */}
                <button onClick={fileOpen ? () => setExpanded(v => !v) : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${fileOpen ? 'bg-emerald-500 cursor-pointer hover:bg-emerald-400' : 'bg-emerald-500/40 cursor-default'}`}
                  title={fileOpen ? (expanded ? t('devTestPanel.toolbar.minimizeTooltip') : t('devTestPanel.toolbar.maximizeTooltip')) : ''}>
                  {tlHovered && fileOpen && (expanded ? <Minimize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" /> : <Maximize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" />)}
                </button>
              </div>
              <div className="flex items-center gap-2">
                <FileCode className={`w-4 h-4 ${fileOpen ? 'text-amber-400' : 'text-gray-600'}`} />
                <span className={`text-sm font-medium ${fileOpen ? 'text-gray-300' : 'text-gray-600'}`}>
                  {fileOpen ? t('devTestPanel.editor.fileName') : t('devTestPanel.toolbar.noDocument')}
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
                  placeholder={t('devTestPanel.toolbar.searchPlaceholder')}
                  className="w-44 px-2 py-1 bg-transparent text-gray-200 text-xs focus:outline-none placeholder:text-gray-600"
                />
                <span className="text-[10px] text-gray-500 font-mono w-12 text-right">
                  {findStatus ? `${findStatus.current}/${findStatus.total}` : ''}
                </span>
                <button
                  onClick={() => findNext(-1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title={t('devTestPanel.toolbar.prevSearchTooltip')}
                >
                  ↑
                </button>
                <button
                  onClick={() => findNext(1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title={t('devTestPanel.toolbar.nextSearchTooltip')}
                >
                  ↓
                </button>
                <button
                  onClick={() => { setFindOpen(false); setFindStatus(null); editorRef.current?.focus(); }}
                  className="p-1 rounded-lg hover:bg-white/10 text-gray-500 hover:text-white transition-all"
                  title={t('devTestPanel.toolbar.closeSearchTooltip')}
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            <div className="flex items-center gap-2">
              {!fileOpen ? (
                <>
                  <button onClick={handleNewFile} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/25 text-amber-400 text-xs font-medium transition-all">
                    <FileCode className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.newFileButton')}
                  </button>
                  <button onClick={() => setShowLib(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.loadFileButton')}
                  </button>
                  <button onClick={() => setShowOpenLib(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs font-medium transition-all">
                    <Globe className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.openLibraryButton')}
                  </button>
                </>
              ) : (
                <>
                  {isDirty && (
                    <button onClick={handleSave} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-xs font-medium transition-all">
                      <Save className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.saveButton')}
                    </button>
                  )}
                  <button onClick={() => setShowLib(true)} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.libraryButton')}
                  </button>
                  <button
                    onClick={() => setShowOpenLib(true)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs font-medium transition-all"
                  >
                    <Globe className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.openLibraryButton')}
                  </button>
                  <button
                    onClick={() => {
                      if (!aiSettings.enabled) {
                        error(t('devTestPanel.toolbar.aiDisabledTitle'), t('devTestPanel.toolbar.aiDisabledDetail'));
                        return;
                      }
                      setShowAI(v => !v);
                    }}
                    title={!aiSettings.enabled ? t('devTestPanel.toolbar.aiDisabledTooltip') : ''}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : !aiSettings.enabled ? 'bg-white/5 text-gray-500 border-white/10 opacity-60' : 'bg-white/5 text-gray-400 hover:text-white border-white/10'}`}>
                    <Bot className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.aiButton')}
                  </button>
                  <button
                    onClick={() => {
                      setFindOpen(true);
                      setTimeout(() => findInputRef.current?.focus(), 0);
                      const ta = editorRef.current;
                      updateFindStatus(findQuery, ta?.selectionStart ?? 0);
                    }}
                    className="px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs font-medium transition-all"
                    title={t('devTestPanel.toolbar.findTooltip')}
                  >
                    ⌘F
                  </button>
                  <button onClick={() => setExpanded(v => !v)} className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all">
                    {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                  {isRunning ? (
                    <button onClick={handleStop} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 text-xs font-medium transition-all">
                      <Square className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.stopButton')}
                    </button>
                  ) : (
                    <button onClick={handleStart} disabled={!script.trim() || !modelInfo}
                      className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-white text-xs font-semibold hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}>
                      <Play className="w-3.5 h-3.5" /> {t('devTestPanel.toolbar.startButton')}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Body */}
          {!fileOpen ? (
            <div className="flex flex-col items-center justify-center bg-slate-950 text-center" style={{ height: `${editorH}px` }}>
              <FlaskConical className="w-12 h-12 text-gray-700 mb-6" />
              <p className="text-gray-500 text-sm mb-8">{t('devTestPanel.emptyState.description')}</p>
              <div className="flex gap-4">
                <button onClick={handleNewFile} className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-amber-500/20 bg-amber-500/8 hover:bg-amber-500/15 hover:border-amber-500/40 transition-all group">
                  <FileCode className="w-7 h-7 text-amber-500 group-hover:text-amber-400" />
                  <div><p className="font-semibold text-white text-sm">{t('devTestPanel.emptyState.newFileTitle')}</p><p className="text-xs text-gray-500 mt-1">{t('devTestPanel.emptyState.newFileHint')}</p></div>
                </button>
                <button onClick={() => setShowLib(true)} className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20 transition-all group">
                  <FolderClosed className="w-7 h-7 text-gray-500 group-hover:text-gray-300" />
                  <div><p className="font-semibold text-white text-sm">{t('devTestPanel.emptyState.loadFileTitle')}</p><p className="text-xs text-gray-500 mt-1">{t('devTestPanel.emptyState.loadFileHint')}</p></div>
                </button>
                <button onClick={() => setShowOpenLib(true)} className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-violet-500/20 bg-violet-500/8 hover:bg-violet-500/15 hover:border-violet-500/40 transition-all group">
                  <Globe className="w-7 h-7 text-violet-500 group-hover:text-violet-400" />
                  <div><p className="font-semibold text-white text-sm">{t('devTestPanel.emptyState.openLibraryTitle')}</p><p className="text-xs text-gray-500 mt-1">{t('devTestPanel.emptyState.openLibraryHint')}</p></div>
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex" style={{ height: expanded ? 'calc(100vh - 280px)' : `${editorH}px` }}>
                <div className="flex flex-1 min-w-0 overflow-hidden bg-slate-950">
                  {/* Zeilennummern */}
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
                  {/* Editor */}
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
                    <pre ref={editorPreRef} aria-hidden
                      className="absolute inset-0 text-xs font-mono overflow-hidden pointer-events-none text-gray-200 whitespace-pre"
                      style={{
                        fontFamily: "'JetBrains Mono','Fira Code','Cascadia Code','Courier New',monospace",
                        tabSize: 2 as any,
                        lineHeight: `${editorLineHeightPx}px`,
                        padding: `${editorPadTopPx}px ${editorPadLeftPx}px ${editorPadTopPx}px ${editorPadLeftPx}px`,
                        boxSizing: 'border-box' as const,
                      }}
                      dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
                    <textarea ref={editorRef} autoFocus value={script} wrap="off"
                      placeholder={t('devTestPanel.editor.placeholder')}
                      onChange={e => {
                        const v = e.target.value;
                        if (v === '! ') { generateTemplate(); return; }
                        setScript(v); setIsDirty(v !== savedScript);
                      }}
                      onScroll={syncEditorScroll} onKeyUp={updateActiveLine} onMouseUp={updateActiveLine}
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

                {/* AI Sidebar */}
                {showAI && (
                  <div className="w-80 border-l border-white/10 flex-shrink-0 flex flex-col overflow-hidden">
                    <CodeAISidebar
                      script={script} modelInfo={modelInfo} datasets={datasets}
                      outputPath={outputPath.replace('<job_id>', 'dev_test')}
                      modelPathOverride={modelPath}
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

              {/* Resize Handle */}
              {!expanded && (
                <div className="h-2 bg-white/[0.02] hover:bg-amber-500/20 cursor-ns-resize border-t border-white/10 flex items-center justify-center group transition-colors"
                  onMouseDown={e => {
                    e.preventDefault();
                    const startY = e.clientY, startH = editorH;
                    const move = (ev: MouseEvent) => setEditorH(Math.max(300, Math.min(900, startH + ev.clientY - startY)));
                    const up   = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
                    window.addEventListener('mousemove', move);
                    window.addEventListener('mouseup', up);
                  }}>
                  <div className="w-8 h-0.5 rounded-full bg-white/20 group-hover:bg-amber-400/60 transition-colors" />
                </div>
              )}
            </>
          )}
        </div>

        {/* Output Panel */}
        {(isRunning || output) && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isRunning
                  ? <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
                  : exitCode === 0
                    ? <CheckCircle className="w-4 h-4 text-emerald-400" />
                    : exitCode !== null
                      ? <AlertCircle className="w-4 h-4 text-red-400" />
                      : <ClipboardList className="w-4 h-4 text-gray-400" />}
                <span className="text-white font-medium text-sm">
                  {isRunning ? t('devTestPanel.output.running') : exitCode === 0 ? t('devTestPanel.output.success') : exitCode !== null ? t('devTestPanel.output.failed') : t('devTestPanel.output.label')}
                </span>
              </div>
              {!isRunning && output && (
                <button onClick={() => setOutput('')} className="text-xs text-gray-500 hover:text-white px-2 py-1 rounded-lg bg-white/5 transition-all">{t('devTestPanel.output.clearButton')}</button>
              )}
            </div>

            <div className="rounded-xl border border-white/10 bg-black/30 overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10">
                <Terminal className="w-3.5 h-3.5 text-gray-500" />
                <span className="text-[10px] text-gray-500">{t('devTestPanel.output.panelLabel')}</span>
              </div>
              <div ref={outputRef} className="p-3 max-h-64 overflow-y-auto">
                {isRunning && !output && <p className="text-gray-600 text-[10px] animate-pulse">{t('devTestPanel.output.waiting')}</p>}
                <pre className="text-[10px] font-mono text-gray-300 whitespace-pre-wrap leading-relaxed">{output}</pre>
              </div>
            </div>
          </div>
        )}
      </div>

      {showLibrary && (
        <ScriptLibraryModal currentScript={script} userId={userData?.userId} onLoad={s => {
          setScript(s.script); setSavedScript(s.script); setCurrentScriptId(s.id);
          setIsDirty(false); setFileOpen(true);
        }} onClose={() => setShowLib(false)} />
      )}

      <SaveNameDialog isOpen={showSaveDialog} defaultName={saveName} onSave={handleSaveWithName} onClose={() => setShowSaveDialog(false)} />

      <DevTestErrorModal
        isOpen={showErrorModal} errorTitle={errorTitle} errorMessage={errorMessage}
        errorDetails={errorDetails} script={script} output={output}
        onClose={() => setShowErrorModal(false)} onSendToAI={handleSendToAI}
      />

      {showOpenLib && (
        <OpenLibraryModal
          userData={userData}
          mode="test"
          onClose={() => setShowOpenLib(false)}
          onLoadScript={(scriptContent, scriptName) => {
            setScript(scriptContent);
            setSavedScript(scriptContent);
            setCurrentScriptId(null);
            setIsDirty(true);
            setFileOpen(true);
            setShowOpenLib(false);
            success('Geladen!', `„${scriptName}“ wurde in den Editor geladen.`);
          }}
        />
      )}
    </div>
  );
}
