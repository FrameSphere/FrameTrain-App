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
  History, MessageSquarePlus,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import type { ModelInfo, DatasetInfo } from './TrainingPanel';
import { callAI } from './TrainingPanel';
import { parseEdits, applyEdit, applyAllEdits, removeEditBlocks, extractFullPythonCode, type CodeEdit } from '../ai/codeEdits';
import { buildAutoSystemPrompt, parseAutoAction, type AutoAction } from '../ai/autoModeProtocol';
import DiffViewer from './DiffViewer';

// ── Script Library ────────────────────────────────────────────────────────

interface SavedScript { id: string; name: string; script: string; savedAt: string; }

const SCRIPTS_KEY = 'ft_saved_test_scripts';
const loadScripts  = (): SavedScript[] => { try { return JSON.parse(localStorage.getItem(SCRIPTS_KEY) ?? '[]'); } catch { return []; } };
const saveScript   = (name: string, script: string) => { const all = loadScripts(); all.unshift({ id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString() }); localStorage.setItem(SCRIPTS_KEY, JSON.stringify(all.slice(0, 50))); };
const deleteScript = (id: string) => localStorage.setItem(SCRIPTS_KEY, JSON.stringify(loadScripts().filter(s => s.id !== id)));
const updateScript = (id: string, script: string) => { const all = loadScripts(); const idx = all.findIndex(s => s.id === id); if (idx >= 0) { all[idx] = { ...all[idx], script, savedAt: new Date().toISOString() }; localStorage.setItem(SCRIPTS_KEY, JSON.stringify(all)); } };

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
  const [name, setName] = useState(defaultName);
  useEffect(() => { setName(defaultName); }, [defaultName]);
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
          <div className="flex items-center gap-2"><Save className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">Skript speichern</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-300 text-sm">Gib einen Namen für dein Test-Skript ein.</p>
          <input value={name} onChange={e => setName(e.target.value)} onKeyDown={e => e.key === 'Enter' && name.trim() && onSave(name.trim())} placeholder="z.B. Mein Test-Skript" autoFocus
            className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40" />
        </div>
        <div className="px-6 pb-6 flex gap-2">
          <button onClick={() => name.trim() && onSave(name.trim())} disabled={!name.trim()}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40 transition-all">
            <Save className="w-4 h-4" /> Speichern
          </button>
          <button onClick={onClose} className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white text-sm font-medium transition-all">Abbrechen</button>
        </div>
      </div>
    </div>
  );
}

// ── Script Library Modal ──────────────────────────────────────────────────

function ScriptLibraryModal({ currentScript, onLoad, onClose }: { currentScript: string; onLoad: (s: SavedScript) => void; onClose: () => void; }) {
  const [scripts, setScripts] = useState<SavedScript[]>([]);
  const [saveName, setSaveName] = useState('');
  const [showSaveForm, setShowForm] = useState(false);
  const { success } = useNotification();

  useEffect(() => { setScripts(loadScripts()); }, []);

  const handleSave = () => {
    if (!saveName.trim()) return;
    saveScript(saveName.trim(), currentScript);
    setScripts(loadScripts());
    setSaveName(''); setShowForm(false);
    success('Gespeichert', `Test-Skript "${saveName}" gespeichert.`);
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FolderClosed className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">Test-Skript Bibliothek</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {scripts.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FileText className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">Noch keine Test-Skripte gespeichert.</p>
            </div>
          ) : scripts.map(s => (
            <div key={s.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium text-sm truncate">{s.name}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{new Date(s.savedAt).toLocaleDateString('de-DE', { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' })}</p>
                  <pre className="text-gray-600 text-[10px] mt-1.5 font-mono truncate">{s.script.split('\n').slice(0, 2).join(' · ')}</pre>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                  <button onClick={() => { deleteScript(s.id); setScripts(loadScripts()); }} className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>
                  <button onClick={() => { onLoad(s); onClose(); }} className="px-3 py-1.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-xs font-medium transition-all">Laden</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 pb-5 border-t border-white/10 pt-4 flex-shrink-0">
          {showSaveForm ? (
            <div className="flex gap-2">
              <input value={saveName} onChange={e => setSaveName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSave()} placeholder="Skript-Name…" autoFocus
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-amber-500/40" />
              <button onClick={handleSave} disabled={!saveName.trim()} className="px-4 py-2 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-300 text-sm font-medium disabled:opacity-40"><Save className="w-4 h-4" /></button>
              <button onClick={() => setShowForm(false)} className="px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-400 text-sm"><X className="w-4 h-4" /></button>
            </div>
          ) : (
            <button onClick={() => setShowForm(true)} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-300 text-sm font-medium transition-all">
              <Save className="w-4 h-4" /> Aktuelles Skript speichern
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
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showDiffModal, setShowDiffModal] = useState(false);
  const [currentMessageWithEdits, setCurrentMessageWithEdits] = useState<AiMessage | null>(null);
  const [isApplyingEdits, setIsApplyingEdits] = useState(false);
  const [appliedEdits, setAppliedEdits] = useState<AppliedEditInfo[]>([]);
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
      const ns: ChatSession = { id, title: 'Neuer Chat', messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
      saveChatSessions([ns, ...sessions]);
      setCurrentSessionId(id); setSessionTitle('Neuer Chat'); setMessages([]);
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
    const ns: ChatSession = { id, title: 'Neuer Chat', messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    const sessions = loadChatSessions().filter(s => s.messages.length > 0);
    saveChatSessions([ns, ...sessions]);
    setCurrentSessionId(id); setSessionTitle('Neuer Chat'); setMessages([]);
    setAppliedEdits([]); setCurrentMessageWithEdits(null); setIsReadonly(false); setShowHistory(false);
    onClearHighlights?.();
  };

  const switchToSession = (session: ChatSession) => {
    setCurrentSessionId(session.id); setSessionTitle(session.title); setMessages(session.messages);
    setAppliedEdits([]); setCurrentMessageWithEdits(null); setIsReadonly(true); setShowHistory(false);
    onClearHighlights?.();
  };

  const continueFromSession = (session: ChatSession) => {
    const id = `s_${Date.now()}`;
    const title = session.title + ' (Fortgesetzt)';
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
    'Fehler beheben (Stacktrace)',
    'Eval/Metric hinzufügen (F1/Accuracy)',
    'Batch-Inference robust machen',
  ];

  const send = async () => {
    if (!input.trim() || loading || isReadonly) return;
    const userMsg: AiMessage = { role: 'user', content: input.trim() };
    if (messages.length === 0) {
      const title = makeSessionTitle(input.trim());
      setSessionTitle(title);
      const sessions = loadChatSessions();
      const idx = sessions.findIndex(s => s.id === currentSessionIdRef.current);
      if (idx >= 0) { sessions[idx].title = title; saveChatSessions(sessions); }
    }
    setMessages(m => [...m, userMsg]); setInput(''); setLoading(true);
    try {
      const history = [...messages, userMsg].map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));
      const last = history.pop()!;
      const response = await callAI(aiSettings, systemPrompt, last.content, history);
      const { action, cleaned } = parseAutoAction(response);
      const inferredEdit = (action?.mode === 'edit') || cleaned.includes('##EDIT_START##');
      const edits = inferredEdit ? parseEdits(response) : [];
      const code = action?.mode === 'rewrite' ? (extractFullPythonCode(response) ?? null) : null;
      const finalContent = code ? [cleaned, '```python', code, '```'].join('\n') : cleaned;
      setMessages(m => [...m, { role: 'assistant', content: finalContent, edits, action }]);
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', content: `Fehler: ${String(err)}` }]);
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
            <span className="text-sm font-medium text-white">KI-Assistent</span>
          </div>
          <div className="flex items-center gap-0.5 flex-shrink-0">
            <span className="px-2 py-0.5 rounded-md bg-purple-500/15 border border-purple-500/25 text-purple-200 text-[10px] font-medium">Auto</span>
            <button
              onClick={() => setShowHistory(v => !v)}
              title="Chat-Verlauf"
              className={`p-1.5 rounded-lg transition-all ${
                showHistory ? 'bg-violet-500/20 text-violet-300' : 'hover:bg-white/5 text-gray-500 hover:text-white'
              }`}
            >
              <History className="w-3.5 h-3.5" />
            </button>
            <button onClick={startNewSession} title="Neuer Chat" className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all">
              <MessageSquarePlus className="w-3.5 h-3.5" />
            </button>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all ml-0.5"><X className="w-3.5 h-3.5" /></button>
          </div>
        </div>

        {sessionTitle && sessionTitle !== 'Neuer Chat' && (
          <div className="px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.01] flex items-center gap-1.5">
            <span className="text-[9px] text-gray-600">↳</span>
            <span className="text-[10px] text-gray-500 truncate">{sessionTitle}</span>
            {isReadonly && <span className="ml-auto flex-shrink-0 text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400/80">Lesemodus</span>}
          </div>
        )}

        {showHistory && (
          <div className="absolute inset-x-0 top-[41px] z-10 bg-slate-950 border-b border-white/10 flex flex-col shadow-xl" style={{ maxHeight: '60%', overflowY: 'auto' }}>
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
              <span className="text-[10px] font-medium text-gray-400">Chat-Verlauf</span>
              <button onClick={startNewSession} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 text-violet-300 text-[10px] transition-all">
                <MessageSquarePlus className="w-3 h-3" /> Neuer Chat
              </button>
            </div>
            <div className="overflow-y-auto flex-1">
              {loadChatSessions().length === 0 ? (
                <p className="text-center text-gray-600 text-[10px] py-6">Noch keine gespeicherten Chats.</p>
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
                        <span className="text-[9px] text-gray-700">· {session.messages.length} Nachrichten</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                      {!isActive && session.messages.length > 0 && (
                        <button onClick={e => { e.stopPropagation(); continueFromSession(session); }}
                          className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 transition-all">
                          Fortsetzen
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
              <p className="text-gray-400 text-xs">Beschreibe Ziel/Problem — ich liefere direkt Edits oder einen Rewrite.</p>
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
                          <span className="text-[10px] text-gray-500 font-mono">Python</span>
                          <button onClick={() => onReplaceScript(code)} className="text-[10px] px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-all">Ersetzen</button>
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
                              Änderung {editIdx + 1}
                            </span>
                            {isApplied ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleUndoEdit(messageIdx, edit.id);
                                }}
                                className="text-emerald-400/70 hover:text-emerald-300 text-xs flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/[0.15] hover:bg-emerald-500/25 transition-all"
                              >
                                <span>Rückgängig</span>
                              </button>
                            ) : (
                              <span className="text-amber-400/70 text-xs">→ Diff ansehen</span>
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
                  <span className="text-[10px] text-gray-500 shrink-0">Bereit:</span>
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
                    <Check className="w-3 h-3" /> Übernehmen
                  </button>
                  <button
                    onClick={() => {
                      setCurrentMessageWithEdits(latestEditMsg);
                      setShowDiffModal(true);
                    }}
                    className="flex items-center justify-center px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-[10px] font-medium transition-all"
                  >
                    Details
                  </button>
                </div>
              </div>
            );
          })()}
          
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-gray-600">
              {isReadonly ? 'Lesemodus – Chat nicht aktiv' : 'Enter = senden · Shift+Enter = neue Zeile'}
            </span>
            <span className="text-[10px] text-purple-300/70">Auto</span>
          </div>
          <div className="flex gap-2 items-end">
            <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={isReadonly ? 'Zum Schreiben neuen Chat starten → □ oben rechts' : 'Ziel / Problem / gewünschte Änderung…'} rows={2}
              disabled={isReadonly}
              className={`flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none transition-opacity ${
                isReadonly ? 'opacity-40 cursor-not-allowed' : ''
              }`} />
            <button onClick={send} disabled={!input.trim() || loading || isReadonly}
              className="p-2.5 rounded-xl border transition-all disabled:opacity-40 bg-purple-500/20 hover:bg-purple-500/30 border-purple-500/30 text-purple-200">
              <Send className="w-4 h-4" />
            </button>
          </div>
          {isReadonly && (
            <button
              onClick={startNewSession}
              className="mt-2 w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 text-violet-300 text-[10px] font-medium transition-all"
            >
              <MessageSquarePlus className="w-3 h-3" /> Neuen Chat starten
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
  const [copied, setCopied] = useState(false);
  if (!isOpen) return null;
  const ctx = `[Dev Test Fehler]\n\nTitel: ${errorTitle}\n\nFehler: ${errorMessage}\n\nDetails: ${errorDetails}\n\nSkript:\n${script}\n\nAusgabe:\n${output}`;
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-red-500/10 flex-shrink-0">
          <div className="flex items-center gap-3"><span className="text-3xl">❌</span><div><h2 className="text-lg font-bold text-white">Test fehlgeschlagen</h2><p className="text-sm text-red-300">{errorTitle}</p></div></div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {errorMessage && <div><p className="text-xs text-gray-500 font-medium mb-2">Fehler-Meldung:</p><div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg max-h-24 overflow-auto"><pre className="text-xs text-red-300 font-mono whitespace-pre-wrap">{errorMessage}</pre></div></div>}
          {errorDetails && <div><p className="text-xs text-gray-500 font-medium mb-2">Details:</p><div className="p-3 bg-white/5 border border-white/10 rounded-lg max-h-24 overflow-auto"><pre className="text-xs text-gray-400 font-mono whitespace-pre-wrap">{errorDetails}</pre></div></div>}
        </div>
        <div className="px-6 py-4 border-t border-white/10 flex gap-3 flex-shrink-0">
          <button onClick={() => { navigator.clipboard.writeText(ctx); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all">
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}{copied ? 'Kopiert!' : 'Fehler kopieren'}
          </button>
          <button onClick={() => onSendToAI(ctx)} disabled={isSending}
            className="flex items-center gap-2 px-4 py-2 bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 rounded-lg text-sm text-violet-300 transition-all disabled:opacity-50">
            <Sparkles className="w-4 h-4" /> An KI schicken
          </button>
          <button onClick={onClose} className="ml-auto px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all">Schließen</button>
        </div>
      </div>
    </div>
  );
}

// ── Helper Components ─────────────────────────────────────────────────────

function RefRow({ color, label, value, hint }: { color: string; label: string; value: string; hint?: string }) {
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
        <span className={`break-all ${value ? 'text-gray-300' : 'text-gray-600 italic'}`}>{value || 'nicht gesetzt'}</span>
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
          title="Kopieren"
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      )}
    </div>
  );
}

// ── Default Script Generator ──────────────────────────────────────────────

function generateDefaultTestScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const ds = datasets[0];
  const modelPathDefault   = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_test').replace('{wird beim Start gesetzt}', 'dev_test');

  return `#!/usr/bin/env python3
# FrameTrain – Dev Test Script
# Eigenes Inference- / Evaluierungs-Skript

import os
import json
from pathlib import Path

# ── Pfade (von FrameTrain als ENV-Vars gesetzt) ─────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

# ── Imports ───────────────────────────────────────────────────────────────
# TODO: Importiere Bibliotheken nach Bedarf
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_from_disk, load_dataset
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

# ── Modell & Tokenizer laden ──────────────────────────────────────────────
print(f"✅ Lade Modell aus: {MODEL_PATH}")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# ── Dataset laden ─────────────────────────────────────────────────────────
print(f"✅ Lade Dataset aus: {DATASET_PATH}")
# dataset = load_from_disk(DATASET_PATH)  # oder load_dataset(...)
# test_data = dataset["test"]  # oder dataset["validation"]

# ── Inference ────────────────────────────────────────────────────────────
# Beispiel: einzelner Text
# texts = ["Das ist ein Testtext.", "Noch ein Beispiel."]
# inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
# with torch.no_grad():
#     outputs = model(**inputs)
# predictions = outputs.logits.argmax(dim=-1).tolist()
# print("Predictions:", predictions)

# ── Batch-Evaluation ─────────────────────────────────────────────────────
# all_preds = []
# all_labels = []
# for example in test_data:
#     inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     pred = outputs.logits.argmax(dim=-1).item()
#     all_preds.append(pred)
#     all_labels.append(example["label"])
#
# print(classification_report(all_labels, all_preds))

# ── Ergebnisse speichern ──────────────────────────────────────────────────
# Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
# results = {"predictions": all_preds, "labels": all_labels}
# with open(f"{OUTPUT_PATH}/results.json", "w") as f:
#     json.dump(results, f, indent=2)
# print(f"✅ Ergebnisse gespeichert: {OUTPUT_PATH}/results.json")

print("✅ Test-Skript abgeschlossen!")
`;
}

// ── DevTestPanel ──────────────────────────────────────────────────────────

interface DevTestPanelProps {
  modelInfo: ModelInfo | null;
  selectedVersionPath?: string;
  datasets: DatasetInfo[];
}

export default function DevTestPanel({ modelInfo, selectedVersionPath, datasets }: DevTestPanelProps) {
  const { currentTheme } = useTheme();
  const { success, error } = useNotification();
  const { settings: aiSettings } = useAISettings();

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
        setOutput(o => o + '\n✅ Test abgeschlossen!');
        invoke('disable_prevent_sleep').catch(() => {});
      } else {
        const msg = e.payload?.data?.error ?? `Prozess beendet mit Exit-Code ${code}`;
        const details = e.payload?.data?.details ?? '';
        setOutput(o => o + `\n❌ ${msg}${details ? '\n' + details : ''}`);
        invoke('disable_prevent_sleep').catch(() => {});
        setErrorTitle('Test fehlgeschlagen');
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
    if (isDirty) { error('Ungespeicherte Änderungen', 'Bitte erst speichern (⌘S).'); return; }
    setFileOpen(false); setScript(''); setSavedScript(''); setCurrentScriptId(null); setExpanded(false);
  };

  const generateTemplate = () => {
    if (!modelInfo || !outputPath) return;
    setScript(generateDefaultTestScript(modelInfo, datasets, outputPath));
    setIsDirty(true);
  };

  const handleSave = () => {
    if (currentScriptId) {
      updateScript(currentScriptId, script);
      setSavedScript(script); setIsDirty(false);
      success('Aktualisiert', 'Test-Skript aktualisiert!');
    } else {
      setSaveName('Mein Test-Skript');
      setShowSaveDialog(true);
    }
  };

  const handleSaveWithName = (name: string) => {
    if (!name.trim()) return;
    saveScript(name.trim(), script);
    const newScript = loadScripts()[0];
    if (newScript) setCurrentScriptId(newScript.id);
    setSavedScript(script); setIsDirty(false); setShowSaveDialog(false); setSaveName('');
    success('Gespeichert', `Test-Skript "${name}" gespeichert!`);
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
    if (isDirty) { error('Ungespeicherte Änderungen', 'Bitte erst speichern (⌘S).'); return; }
    if (!script.trim() || !modelInfo) { error('Fehler', 'Kein Modell ausgewählt oder Skript leer.'); return; }

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
      setOutput(`🚀 Test gestartet…\n`);
      invoke('enable_prevent_sleep').catch(() => {});
      success('Gestartet!', 'Dev Test läuft…');
    } catch (err: unknown) {
      setOutput(`❌ ${String(err)}`);
      setRunning(false);
      error('Fehler', String(err));
    }
  };

  const handleStop = async () => {
    try { await invoke('stop_dev_test'); } catch { /* ignore */ }
    invoke('disable_prevent_sleep').catch(() => {});
    setRunning(false);
    setOutput(o => o + '\n⏹️  Test gestoppt.');
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
                <span className="text-amber-300 font-semibold text-sm">Dev Test Mode</span>
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
            <p className="text-gray-400 text-xs">Eigenes Python-Skript für Inference, Evaluation und Testing. Voller Zugriff auf Modell- und Dataset-Pfade.</p>
          </div>
        )}

        {/* Paths — Collapsible (wie DevTrain) */}
        <button
          onClick={() => setShowPathsModal(v => !v)}
          className="w-full px-4 py-3 rounded-2xl border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/15 transition-all flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-300">Pfade konfigurieren</span>
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
                <span className="text-sm font-medium text-white">Modell</span>
              </div>
              <RefRow color="text-emerald-400" label="MODEL_PATH" value={modelPath} hint={modelInfo?.name} />
            </div>

            <div className="border-t border-white/10" />

            {/* Dataset */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-white">Dataset</span>
              </div>
              {dsRefs.map(r => <RefRow key={r.key} color="text-blue-400" label={r.key} value={r.value} hint={r.name} />)}
            </div>

            <div className="border-t border-white/10" />

            {/* Output */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">Output</span>
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
                  title={fileOpen ? (isDirty ? 'Ungespeicherte Änderungen' : 'Datei schließen') : ''}>
                  {tlHovered && fileOpen && <X className="w-[7px] h-[7px] text-red-900 stroke-[3]" />}
                  {!tlHovered && isDirty && fileOpen && <div className="w-[5px] h-[5px] rounded-full bg-red-900" />}
                </button>
                {/* Gelb: Speichern */}
                <button onClick={fileOpen && isDirty ? handleSave : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${fileOpen && isDirty ? 'bg-amber-400 cursor-pointer hover:bg-amber-300' : 'bg-amber-500/40 cursor-default'}`}
                  title={fileOpen && isDirty ? 'Speichern' : ''}>
                  {tlHovered && fileOpen && isDirty && <Minus className="w-[7px] h-[7px] text-amber-900 stroke-[3]" />}
                </button>
                {/* Grün: Vollbild */}
                <button onClick={fileOpen ? () => setExpanded(v => !v) : undefined}
                  className={`relative w-3 h-3 rounded-full flex items-center justify-center transition-all ${fileOpen ? 'bg-emerald-500 cursor-pointer hover:bg-emerald-400' : 'bg-emerald-500/40 cursor-default'}`}
                  title={fileOpen ? (expanded ? 'Verkleinern' : 'Vollbild') : ''}>
                  {tlHovered && fileOpen && (expanded ? <Minimize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" /> : <Maximize2 className="w-[7px] h-[7px] text-emerald-900 stroke-[3]" />)}
                </button>
              </div>
              <div className="flex items-center gap-2">
                <FileCode className={`w-4 h-4 ${fileOpen ? 'text-amber-400' : 'text-gray-600'}`} />
                <span className={`text-sm font-medium ${fileOpen ? 'text-gray-300' : 'text-gray-600'}`}>
                  {fileOpen ? 'test.py' : 'Kein Dokument'}
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
                  placeholder="Suchen…"
                  className="w-44 px-2 py-1 bg-transparent text-gray-200 text-xs focus:outline-none placeholder:text-gray-600"
                />
                <span className="text-[10px] text-gray-500 font-mono w-12 text-right">
                  {findStatus ? `${findStatus.current}/${findStatus.total}` : ''}
                </span>
                <button
                  onClick={() => findNext(-1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title="Vorheriges (Shift+Enter)"
                >
                  ↑
                </button>
                <button
                  onClick={() => findNext(1)}
                  className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[10px] transition-all"
                  title="Nächstes (Enter)"
                >
                  ↓
                </button>
                <button
                  onClick={() => { setFindOpen(false); setFindStatus(null); editorRef.current?.focus(); }}
                  className="p-1 rounded-lg hover:bg-white/10 text-gray-500 hover:text-white transition-all"
                  title="Schließen (Esc)"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            <div className="flex items-center gap-2">
              {!fileOpen ? (
                <>
                  <button onClick={handleNewFile} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/25 text-amber-400 text-xs font-medium transition-all">
                    <FileCode className="w-3.5 h-3.5" /> Neue Datei
                  </button>
                  <button onClick={() => setShowLib(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> Datei laden
                  </button>
                </>
              ) : (
                <>
                  {isDirty && (
                    <button onClick={handleSave} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-xs font-medium transition-all">
                      <Save className="w-3.5 h-3.5" /> Speichern (⌘S)
                    </button>
                  )}
                  <button onClick={() => setShowLib(true)} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> Bibliothek
                  </button>
                  {aiSettings.enabled && (
                    <button onClick={() => setShowAI(v => !v)} className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : 'bg-white/5 text-gray-400 hover:text-white border-white/10'}`}>
                      <Bot className="w-3.5 h-3.5" /> KI
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setFindOpen(true);
                      setTimeout(() => findInputRef.current?.focus(), 0);
                      const ta = editorRef.current;
                      updateFindStatus(findQuery, ta?.selectionStart ?? 0);
                    }}
                    className="px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs font-medium transition-all"
                    title="Suchen (Cmd/Ctrl+F)"
                  >
                    ⌘F
                  </button>
                  <button onClick={() => setExpanded(v => !v)} className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all">
                    {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                  {isRunning ? (
                    <button onClick={handleStop} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 text-xs font-medium transition-all">
                      <Square className="w-3.5 h-3.5" /> Stopp
                    </button>
                  ) : (
                    <button onClick={handleStart} disabled={!script.trim() || !modelInfo}
                      className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-white text-xs font-semibold hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}>
                      <Play className="w-3.5 h-3.5" /> Test starten
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
              <p className="text-gray-500 text-sm mb-8">Öffne oder erstelle eine Datei um zu starten</p>
              <div className="flex gap-4">
                <button onClick={handleNewFile} className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-amber-500/20 bg-amber-500/8 hover:bg-amber-500/15 hover:border-amber-500/40 transition-all group">
                  <FileCode className="w-7 h-7 text-amber-500 group-hover:text-amber-400" />
                  <div><p className="font-semibold text-white text-sm">Neue Datei</p><p className="text-xs text-gray-500 mt-1">Tippe <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-400 font-mono text-[10px]">!</kbd> + Leertaste für Template</p></div>
                </button>
                <button onClick={() => setShowLib(true)} className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20 transition-all group">
                  <FolderClosed className="w-7 h-7 text-gray-500 group-hover:text-gray-300" />
                  <div><p className="font-semibold text-white text-sm">Datei laden</p><p className="text-xs text-gray-500 mt-1">Aus deiner Bibliothek</p></div>
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
                      placeholder={"# Fange an zu tippen…\n# Tippe '! ' + Leertaste um das Template zu laden"}
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
                  {isRunning ? 'Test läuft…' : exitCode === 0 ? 'Test erfolgreich' : exitCode !== null ? 'Test fehlgeschlagen' : 'Ausgabe'}
                </span>
              </div>
              {!isRunning && output && (
                <button onClick={() => setOutput('')} className="text-xs text-gray-500 hover:text-white px-2 py-1 rounded-lg bg-white/5 transition-all">Löschen</button>
              )}
            </div>

            <div className="rounded-xl border border-white/10 bg-black/30 overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10">
                <Terminal className="w-3.5 h-3.5 text-gray-500" />
                <span className="text-[10px] text-gray-500">Ausgabe</span>
              </div>
              <div ref={outputRef} className="p-3 max-h-64 overflow-y-auto">
                {isRunning && !output && <p className="text-gray-600 text-[10px] animate-pulse">Warte auf Python-Output…</p>}
                <pre className="text-[10px] font-mono text-gray-300 whitespace-pre-wrap leading-relaxed">{output}</pre>
              </div>
            </div>
          </div>
        )}
      </div>

      {showLibrary && (
        <ScriptLibraryModal currentScript={script} onLoad={s => {
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
    </div>
  );
}
