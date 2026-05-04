// DevTrainPanel.tsx – Dev Train Mode (ausgekoppelt aus TrainingPanel)
// KI-Assistent kann den Code direkt bearbeiten (EDIT-Protokoll)

import { useCallback, useMemo, useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Loader2, Terminal, FolderOpen, FileCode,
  FolderClosed, Bot, Send, Maximize2, Minimize2, X, Minus,
  AlertCircle, CheckCircle, TrendingDown, BarChart3, Zap,
  Save, FileText, Trash2, Pencil, Check, Wand2, Sparkles, Copy, Package,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import type { TrainingJob, TrainingProgress, LossPoint, ModelInfo, DatasetInfo } from './TrainingPanel';
import { callAI, LossChart } from './TrainingPanel';
import TrainingDashboard from './TrainingDashboard';

// ── Script Library ────────────────────────────────────────────────────────

interface SavedScript { id: string; name: string; script: string; savedAt: string; }

const SCRIPTS_KEY = 'ft_saved_scripts';
const loadScripts  = (): SavedScript[] => { try { return JSON.parse(localStorage.getItem(SCRIPTS_KEY) ?? '[]'); } catch { return []; } };
const saveScript   = (name: string, script: string) => { const all = loadScripts(); all.unshift({ id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString() }); localStorage.setItem(SCRIPTS_KEY, JSON.stringify(all.slice(0, 50))); };
const deleteScript = (id: string) => localStorage.setItem(SCRIPTS_KEY, JSON.stringify(loadScripts().filter(s => s.id !== id)));
const updateScript = (id: string, script: string) => { const all = loadScripts(); const idx = all.findIndex(s => s.id === id); if (idx >= 0) { all[idx] = { ...all[idx], script, savedAt: new Date().toISOString() }; localStorage.setItem(SCRIPTS_KEY, JSON.stringify(all)); } };

// ── Edit Parsing ──────────────────────────────────────────────────────────

interface CodeEdit { id: string; find: string; replace: string; applied?: boolean; failed?: boolean; }

function parseEdits(text: string): CodeEdit[] {
  const edits: CodeEdit[] = [];
  
  // Format 1: ##EDIT_START##...##EDIT_END##
  const regex = /##EDIT_START##\s*FIND:\s*([\s\S]*?)\s*REPLACE:\s*([\s\S]*?)\s*##EDIT_END##/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    // Nur trimmen von außen, aber Whitespace-Struktur beibehalten
    let find = match[1].trim();
    let replace = match[2].trim();
    
    // WICHTIG: Entferne ```python und ``` Marker falls vorhanden
    find = find.replace(/^```python\n?/, '').replace(/\n?```$/, '');
    replace = replace.replace(/^```python\n?/, '').replace(/\n?```$/, '');
    
    edits.push({ id: `edit_${Date.now()}_${edits.length}`, find, replace });
    
    console.log(`📌 Parsed Edit #${edits.length}:`, {
      findLines: find.split('\n').length,
      replaceLines: replace.split('\n').length,
      findPreview: find.substring(0, 60).replace(/\n/g, '⏎'),
      replacePreview: replace.substring(0, 60).replace(/\n/g, '⏎'),
    });
  }
  
  if (edits.length === 0) {
    console.warn('⚠️  Keine Edits gefunden in KI-Response!');
    console.log('Response Länge:', text.length);
    console.log('Enthält ##EDIT_START##?', text.includes('##EDIT_START##'));
    console.log('Enthält ##EDIT_END##?', text.includes('##EDIT_END##'));
  }
  
  return edits;
}

function normalizeWhitespace(str: string): string {
  // Normalisiere Tabs zu Spaces
  return str.replace(/\t/g, '  ');
}

function applyEdit(script: string, edit: CodeEdit): { result: string; success: boolean } {
  // Versuch 1: Exaktes Match (mit allen Whitespaces wie gegeben)
  if (script.includes(edit.find)) {
    console.log('✅ Tier 1: Exaktes Match gefunden!');
    return { result: script.replace(edit.find, edit.replace), success: true };
  }
  console.log('❌ Tier 1: Kein exaktes Match');
  
  // Detailliertes Debugging: Zeige die ersten 3 Zeilen des FIND-Textes
  const findFirstLines = edit.find.split('\n').slice(0, 3);
  console.log('🔍 FIND Text (erste 3 Zeilen):');
  findFirstLines.forEach((line, i) => {
    console.log(`  Zeile ${i}: "${line}" (Length: ${line.length})`);
  });
  
  // Suche nach partiellen Matches
  console.log('🔎 Suche nach partiellen Matches im Script...');
  const scriptLines = script.split('\n');
  const findLines = edit.find.split('\n');
  let partialMatches: number[] = [];
  
  for (let i = 0; i < scriptLines.length; i++) {
    // Prüfe ob irgendeine der Zeile ähnlich ist
    const scriptLine = scriptLines[i];
    const firstFindLine = findLines[0];
    
    if (firstFindLine && scriptLine.includes(firstFindLine.trim().substring(0, 20))) {
      partialMatches.push(i);
      console.log(`  📍 Potentieller Match bei Zeile ${i}: "${scriptLine.substring(0, 60)}..."`);
    }
  }
  
  if (partialMatches.length === 0) {
    console.log('  ❌ Keine partiellen Matches gefunden!');
  }
  
  // Versuch 2: Line-by-Line mit Whitespace-Normalisierung (Tabs → Spaces)
  console.log(`\nTier 2: Versuche Line-by-Line Matching mit ${findLines.length} Zeilen gegen ${scriptLines.length} Skript-Zeilen`);
  
  const replaceLines = edit.replace.split('\n');
  
  // Versuche zu matchen mit normalisiertem Whitespace
  for (let i = 0; i <= scriptLines.length - findLines.length; i++) {
    const segment = scriptLines.slice(i, i + findLines.length);
    
    // Vergleiche mit Whitespace-Normalisierung
    const segmentNormalized = segment.map(normalizeWhitespace);
    const findNormalized = findLines.map(normalizeWhitespace);
    
    // Exakter Line-by-Line Vergleich nach Normalisierung
    if (segmentNormalized.every((line, idx) => line === findNormalized[idx])) {
      console.log(`✅ Tier 2: Match gefunden bei Zeile ${i}!`);
      const before = scriptLines.slice(0, i).join('\n');
      const after = scriptLines.slice(i + findLines.length).join('\n');
      const newContent = [...(before ? [before] : []), ...replaceLines, ...(after ? [after] : [])].join('\n');
      return { result: newContent, success: true };
    }
  }
  console.log('❌ Tier 2: Kein normalisiertes Match');
  
  // Versuch 3: Noch aggressivere Matching (trimEnd jede Zeile)
  console.log('Tier 3: Versuche aggressive trimEnd Matching...');
  for (let i = 0; i <= scriptLines.length - findLines.length; i++) {
    const segmentTrimmed = scriptLines.slice(i, i + findLines.length).map(l => l.trimEnd());
    const findTrimmed = findLines.map(l => l.trimEnd());
    
    if (segmentTrimmed.every((line, idx) => line === findTrimmed[idx])) {
      console.log(`✅ Tier 3: Match gefunden bei Zeile ${i}!`);
      const before = scriptLines.slice(0, i).join('\n');
      const after = scriptLines.slice(i + findLines.length).join('\n');
      const newContent = [...(before ? [before] : []), ...replaceLines, ...(after ? [after] : [])].join('\n');
      return { result: newContent, success: true };
    }
  }
  console.log('❌ Tier 3: Kein trimEnd Match gefunden');
  
  return { result: script, success: false };
}

// ── Error Categorization ──────────────────────────────────────────────────

type ErrorCategory = 'memory' | 'dataset' | 'packages' | 'cuda' | 'code' | 'config' | 'unknown';

function analyzeError(errorMsg: string): { 
  category: ErrorCategory; 
  title: string; 
  hint: string; 
  icon: string;
} {
  const e = (errorMsg ?? '').toLowerCase();
  if (e.includes('cuda out of memory') || e.includes('out of memory') || e.includes('oom'))
    return { category: 'memory', title: '🧠 RAM/GPU Überlauf', hint: 'Batch-Größe reduzieren, FP16 oder LoRA aktivieren, Gradient Checkpointing einschalten.', icon: '🧠' };
  if (e.includes('cuda') || e.includes('mps') || e.includes('device'))
    return { category: 'cuda', title: '⚙️ Hardware/Device Fehler', hint: 'Gerät nicht verfügbar oder inkompatibel. Versuche mit CPU oder anderen Geräte-Einstellungen.', icon: '⚙️' };
  if (e.includes('dataset') || e.includes('file not found') || e.includes('no such file') || e.includes('path'))
    return { category: 'dataset', title: '📁 Dataset/Pfad Fehler', hint: 'Dataset-Pfad prüfen, Dateirechte überprüfen, Dataset existiert noch?', icon: '📁' };
  if (e.includes('modulenotfounderror') || e.includes('importerror') || e.includes('no module named'))
    return { category: 'packages', title: '📦 Fehlende Python-Pakete', hint: 'Fehlende Dependencies. Prüfe im Skript welche Module importiert werden.', icon: '📦' };
  if (e.includes('nan') || e.includes('inf') || e.includes('gradient') || e.includes('loss'))
    return { category: 'config', title: '📊 Numerischer Fehler (NaN/Inf)', hint: 'Learning Rate zu hoch? Gradient Clipping (max_grad_norm) aktivieren, LR reduzieren.', icon: '📊' };
  if (e.includes('syntaxerror') || e.includes('indentationerror') || e.includes('typeerror') || e.includes('attributeerror') || e.includes('nameerror'))
    return { category: 'code', title: '🐛 Code-Fehler', hint: 'Syntax-, Typ- oder Namensfehler. Der KI-Assistent kann dir helfen.', icon: '🐛' };
  return { category: 'unknown', title: '❓ Unbekannter Fehler', hint: 'Den Fehler an das FrameTrain-Team senden oder den KI-Assistenten zur Analyse nutzen.', icon: '❓' };
}

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
  const analysis = analyzeError(errorMessage);
  const [copied, setCopied] = useState(false);
  const [sent, setSent] = useState(false);

  if (!isOpen) return null;

  const errorContext = `[Dev Train Fehler]\n\nTitel: ${errorTitle}\n\nFehler: ${errorMessage}\n\nDetails: ${errorDetails}\n\nSkript:\n${script}\n\nAusgabe/Logs:\n${output}`;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-9999 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-red-500/10 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="text-3xl">{analysis.icon}</div>
            <div>
              <h2 className="text-lg font-bold text-white">Training fehlgeschlagen</h2>
              <p className="text-sm text-red-300">{analysis.title}</p>
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
            <p className="text-sm text-blue-300"><strong>💡 Hinweis:</strong> {analysis.hint}</p>
          </div>

          {/* Error Message */}
          {errorMessage && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-2">Fehler-Meldung:</p>
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg overflow-x-auto max-h-24">
                <pre className="text-xs text-red-300 font-mono whitespace-pre-wrap break-words">{errorMessage}</pre>
              </div>
            </div>
          )}

          {/* Details */}
          {errorDetails && (
            <div>
              <p className="text-xs text-gray-500 font-medium mb-2">Details:</p>
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
            {copied ? 'Kopiert!' : 'Fehler kopieren'}
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
                Wird gesendet…
              </>
            ) : sent ? (
              <>
                <Check className="w-4 h-4" />
                Gesendet!
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4" />
                An FrameTrain senden
              </>
            )}
          </button>

          <button
            onClick={() => onSendToAI(errorContext)}
            className="flex items-center gap-2 px-4 py-2 bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 rounded-lg text-sm text-violet-300 transition-all"
          >
            <Sparkles className="w-4 h-4" />
            An KI schicken
          </button>

          <button
            onClick={onClose}
            className="ml-auto px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 transition-all"
          >
            Schließen
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Save Name Dialog Modal ────────────────────────────────────────────────

function SaveNameDialog({ isOpen, defaultName, onSave, onClose }: { isOpen: boolean; defaultName: string; onSave: (name: string) => void; onClose: () => void; }) {
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
            <h2 className="text-lg font-bold text-white">Skript speichern</h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-300 text-sm">Gib einen Namen für dein Skript ein. Es wird dann in der Bibliothek gespeichert.</p>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSave()}
            placeholder="z.B. Mein Trainings-Skript"
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
            <Save className="w-4 h-4" /> Speichern
          </button>
          <button
            onClick={onClose}
            className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white text-sm font-medium transition-all"
          >
            Abbrechen
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Script Library Modal ──────────────────────────────────────────────────

function ScriptLibraryModal({ currentScript, onLoad, onClose }: { currentScript: string; onLoad: (s: SavedScript) => void; onClose: () => void; }) {
  const [scripts, setScripts]       = useState<SavedScript[]>([]);
  const [saveName, setSaveName]     = useState('');
  const [showSaveForm, setShowForm] = useState(false);
  const { success } = useNotification();

  useEffect(() => { setScripts(loadScripts()); }, []);

  const handleSave = () => {
    if (!saveName.trim()) return;
    saveScript(saveName.trim(), currentScript);
    setScripts(loadScripts());
    setSaveName(''); setShowForm(false);
    success('Gespeichert', `Skript "${saveName}" gespeichert.`);
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FolderClosed className="w-5 h-5 text-amber-400" /><h2 className="text-lg font-bold text-white">Skript-Bibliothek</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {scripts.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FileText className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">Noch keine Skripte gespeichert.</p>
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

// ── Code AI Sidebar (mit Edit-Skill) ─────────────────────────────────────

interface AiMessage { role: 'user' | 'assistant'; content: string; edits?: CodeEdit[]; }

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

function CodeAISidebar({ script, modelInfo, datasets, outputPath, onApplyEdit, onReplaceScript, onClose, initialInput, forceEditMode }: {
  script: string;
  modelInfo: ModelInfo | null;
  datasets: DatasetInfo[];
  outputPath: string;
  onApplyEdit: (editedScript: string) => void;
  onReplaceScript: (code: string) => void;
  onClose: () => void;
  initialInput?: string;
  forceEditMode?: boolean;
}) {
  const { settings: aiSettings } = useAISettings();
  const [messages, setMessages]  = useState<AiMessage[]>([]);
  const [input, setInput]        = useState('');
  const [loading, setLoading]    = useState(false);
  const [editMode, setEditMode]  = useState(false); // false = Chat, true = Code bearbeiten
  const endRef = useRef<HTMLDivElement>(null);
  const lastPrefillRef = useRef<string>('');

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => {
    if (forceEditMode && !editMode) setEditMode(true);
  }, [forceEditMode, editMode]);
  useEffect(() => {
    if (!initialInput) return;
    if (initialInput === lastPrefillRef.current) return;
    lastPrefillRef.current = initialInput;
    setInput(initialInput);
  }, [initialInput]);

  const modelPath   = modelInfo?.local_path || modelInfo?.source_path || modelInfo?.name || 'MODELL_PFAD';
  const dsRefs      = datasets.map((d, i) => `${i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`} = "${d.storage_path || d.name}" (${d.name})`);

  const systemPrompt = editMode
    ? `Du bist ein Code-Assistent für FrameTrain (Desktop-App für KI-Training).

VERFÜGBARE PFADE (lokal auf dem System):
- MODEL_PATH = "${modelPath}"
${dsRefs.map(r => `- ${r}`).join('\n')}
- OUTPUT_PATH = "${outputPath}"

MODELL: ${modelInfo?.name ?? '?'} | DATASETS: ${datasets.map(d => d.name).join(', ')}
INSTALLIERTE PAKETE: torch, transformers, datasets, scikit-learn, numpy, accelerate, peft, bitsandbytes

VERFÜGBARE TRAININGS-METRIKEN & PARAMETER (zum Anpassen):
- epochs: Anzahl der Trainingsdurchläufe
- batch_size: Trainings-Batch Größe
- learning_rate: Lernrate (z.B. 2e-5)
- warmup_ratio: Anteil Warmup-Schritte (0-1)
- weight_decay: L2-Regularisierung
- gradient_accumulation_steps: Gradient Akkumulation
- max_grad_norm: Gradient Clipping
- max_seq_length: Maximale Sequenzlänge
- fp16/bf16: Mixed Precision (FP16 oder BF16)
- gradient_checkpointing: RAM-sparend
- use_lora: LoRA aktivieren
- lora_r, lora_alpha, lora_dropout: LoRA Parameter
- load_in_4bit/load_in_8bit: QLoRA

EDIT-MODUS AKTIV: Du siehst unten den aktuellen Code im Editor.
Du kannst den Code direkt bearbeiten. Nutze IMMER dieses Format für jede Änderung:

##EDIT_START##
FIND:
# Der exakte Text (mehrere Zeilen möglich) den du ERSETZEN willst
torch.save(model.state_dict(), "xlm_roberta_large_trained.pth")
REPLACE:
# Der neue Text als Ersatz
torch.save(model.state_dict(), OUTPUT_PATH + "/xlm_roberta_large_trained.pth")
##EDIT_END##

WICHTIG:
- Nutze ##EDIT_START## und ##EDIT_END## - NICHT \`\`\`python Blöcke!
- Du darfst mehrere Edits hintereinander schreiben
- Das System ignoriert automatisch Unterschiede in Tabs/Spaces/Einrückungen
- Wenn der genaue Text nicht zu finden ist, versuche weniger Kontext zu nutzen

Wenn du den GESAMTEN Code neu schreiben willst, schreib ihn in einem \`\`\`python Block aus.

AKTUELLER SCRIPT-INHALT:
\`\`\`python
${script}
\`\`\`

Antworte immer auf Deutsch. Sei präzise und hilfreich.`
    : `Du bist ein Code-Assistent für FrameTrain (Desktop-App für KI-Training).

VERFÜGBARE PFADE (lokal auf dem System):
- MODEL_PATH = "${modelPath}"
${dsRefs.map(r => `- ${r}`).join('\n')}
- OUTPUT_PATH = "${outputPath}"

MODELL: ${modelInfo?.name ?? '?'} | DATASETS: ${datasets.map(d => d.name).join(', ')}
INSTALLIERTE PAKETE: torch, transformers, datasets, scikit-learn, numpy, accelerate, peft, bitsandbytes

TRAININGS-PARAMETER die der User evtl. anpassen möchte:
epochs, batch_size, learning_rate, warmup_ratio, weight_decay, gradient_accumulation_steps, max_grad_norm, 
max_seq_length, fp16, bf16, gradient_checkpointing, use_lora, lora_r, lora_alpha, lora_dropout, load_in_4bit, load_in_8bit

CHAT-MODUS: Beantworte Fragen und schreibe Python-Code. Code in \`\`\`python Blöcken.

Antworte immer auf Deutsch. Sei präzise und hilfreich.`;

  const suggestions = editMode
    ? ['Metriken anpassen (LR, Batch Size)', 'Pfade korrekt einsetzen', 'Fehlerbehandlung hinzufügen', 'Metriken (F1, Precision) ergänzen', 'Trainings-Logging verbessern', 'LoRA aktivieren für RAM-Ersparnis']
    : ['Trainings-Skript für Textklassifikation', 'Dataset als JSONL laden', 'Wie nutze ich den OUTPUT_PATH?', 'Komplettes XLM-RoBERTa Fine-Tuning', 'Welche Parameter sind wichtig?'];

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg: AiMessage = { role: 'user', content: input.trim() };
    setMessages(m => [...m, userMsg]); setInput(''); setLoading(true);

    try {
      const history = [...messages, userMsg].map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));
      const last = history.pop()!;
      const response = await callAI(aiSettings, systemPrompt, last.content, history);

      // Edits parsen wenn im Edit-Modus
      let edits = editMode ? parseEdits(response) : [];
      
      console.log(`🎯 Parse result: Found ${edits.length} edits in response`);
      
      // Fallback: Wenn keine Edits gefunden wurden, aber es gibt ```python Blöcke,
      // könnte das System versuchen diese zu interpretieren (optional - nur loggen für jetzt)
      if (editMode && edits.length === 0 && response.includes('```python')) {
        console.warn('⚠️  KI hat möglicherweise das falsche Format verwendet (```python statt ##EDIT_START##). Bitte das Format korrigieren.');
      }
      
      setMessages(m => [...m, { role: 'assistant', content: response, edits }]);
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', content: `Fehler: ${String(err)}` }]);
    } finally { setLoading(false); }
  };

  const handleApplyEdit = (msg: AiMessage, edit: CodeEdit, idx: number) => {
    // DEBUG: Detailliertes Logging
    console.log('🔍 Attempting edit:', {
      id: edit.id,
      findLength: edit.find.length,
      replaceLength: edit.replace.length,
      findPreview: edit.find.substring(0, 50),
      findLines: edit.find.split('\n').length,
    });
    
    console.log('📝 Full FIND text:');
    console.log(JSON.stringify(edit.find));
    console.log('📝 Full REPLACE text:');
    console.log(JSON.stringify(edit.replace));
    
    console.log('🔎 Searching in script with', script.split('\n').length, 'lines');
    
    const { result, success } = applyEdit(script, edit);
    
    console.log('✅ Edit result:', { success, resultPreview: result.substring(0, 100) });
    
    if (success) {
      onApplyEdit(result);
      setMessages(m => m.map(mm => mm === msg ? {
        ...mm,
        edits: mm.edits?.map((e, i) => i === idx ? { ...e, applied: true } : e),
      } : mm));
    } else {
      setMessages(m => m.map(mm => mm === msg ? {
        ...mm,
        edits: mm.edits?.map((e, i) => i === idx ? { ...e, failed: true } : e),
      } : mm));
    }
  };

  const handleApplyAllEdits = (msg: AiMessage) => {
    let current = script;
    const results = (msg.edits ?? []).map(edit => {
      const { result, success } = applyEdit(current, edit);
      if (success) current = result;
      return success;
    });
    onApplyEdit(current);
    setMessages(m => m.map(mm => mm === msg ? {
      ...mm,
      edits: mm.edits?.map((e, i) => ({ ...e, applied: results[i], failed: !results[i] })),
    } : mm));
  };

  const extractFullCode = (text: string) => text.match(/```python\n([\s\S]*?)```/)?.[1] ?? null;

  return (
      <div className="flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-violet-400" />
          <span className="text-sm font-medium text-white">KI-Assistent</span>
        </div>
        <div className="flex items-center gap-1">
          {/* Mode Toggle */}
          <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-white/5 border border-white/10">
            <button
              onClick={() => setEditMode(false)}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${!editMode ? 'bg-violet-500/20 text-violet-300' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <Bot className="w-3 h-3" /> Chat
            </button>
            <button
              onClick={() => setEditMode(true)}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${editMode ? 'bg-amber-500/20 text-amber-300' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <Pencil className="w-3 h-3" /> Edit
            </button>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all ml-1">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Edit-Mode Banner */}
      {editMode && (
        <div className="px-3 py-2 bg-amber-500/10 border-b border-amber-500/20 flex-shrink-0">
          <div className="flex items-center gap-1.5">
            <Pencil className="w-3 h-3 text-amber-400" />
            <p className="text-amber-300 text-[10px] font-medium">Edit-Modus: KI sieht deinen Code und kann ihn direkt bearbeiten</p>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {messages.length === 0 && (
          <div className="py-6 space-y-3">
            {editMode ? <Wand2 className="w-7 h-7 text-amber-400 mx-auto" /> : <Bot className="w-7 h-7 text-violet-400 mx-auto" />}
            <p className="text-gray-400 text-xs">
              {editMode ? 'Beschreibe was du am Code ändern möchtest.' : 'Stell eine Frage oder lass Code generieren.'}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map(s => (
                <button key={s} onClick={() => setInput(s)}
                  className={`px-2.5 py-1 rounded-lg border text-[10px] transition-all ${
                    editMode
                      ? 'bg-amber-500/10 border-amber-500/20 text-amber-300 hover:bg-amber-500/15'
                      : 'bg-violet-500/10 border-violet-500/20 text-violet-300 hover:bg-violet-500/15'
                  }`}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-xs ${m.role === 'user' ? 'bg-emerald-500/20 text-emerald-400' : editMode ? 'bg-amber-500/20 text-amber-400' : 'bg-violet-500/20 text-violet-400'}`}>
              {m.role === 'user' ? 'U' : (editMode ? <Wand2 className="w-3 h-3" /> : <Bot className="w-3 h-3" />)}
            </div>
            <div className={`flex-1 max-w-[90%] flex flex-col gap-1.5 ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
              {/* Render message parts */}
              {m.content.replace(/##EDIT_START##[\s\S]*?##EDIT_END##/g, '').split(/(```python[\s\S]*?```)/g).map((part, pi) => {
                if (part.startsWith('```python')) {
                  const code = extractFullCode(part) ?? part;
                  return (
                    <div key={pi} className="w-full rounded-xl overflow-hidden border border-white/10">
                      <div className="flex items-center justify-between px-3 py-1.5 bg-white/[0.03] border-b border-white/10">
                        <span className="text-[10px] text-gray-500 font-mono">Python</span>
                        <button onClick={() => onReplaceScript(code)} className="text-[10px] px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-all">
                          Ersetzen
                        </button>
                      </div>
                      <pre className="p-3 text-[10px] font-mono text-gray-300 overflow-x-auto max-h-48 leading-relaxed">{code}</pre>
                    </div>
                  );
                }
                return part.trim() ? (
                  <div
                    key={pi}
                    className={`px-3 py-2 rounded-xl text-[11px] leading-relaxed ${
                      m.role === 'user'
                        ? 'bg-emerald-500/10 text-gray-200 border border-emerald-500/20'
                        : editMode
                          ? 'bg-amber-500/[0.06] text-gray-300 border border-amber-500/15'
                          : 'bg-white/[0.05] text-gray-300 border border-white/10'
                    }`}
                  >
                    {part.trim()}
                  </div>
                ) : null;
              })}

              {/* Edit blocks */}
              {m.edits && m.edits.length > 0 && (
                <div className="w-full space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] text-amber-400 font-medium flex items-center gap-1">
                      <Pencil className="w-3 h-3" /> {m.edits.length} Änderung{m.edits.length !== 1 ? 'en' : ''} vorgeschlagen
                    </p>
                    {m.edits.length > 1 && (
                      <button onClick={() => handleApplyAllEdits(m)}
                        className="text-[10px] px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30 transition-all">
                        Alle übernehmen
                      </button>
                    )}
                  </div>
                  {m.edits.map((edit, ei) => (
                    <div key={edit.id} className="rounded-xl border border-white/10 overflow-hidden">
                      {/* Find */}
                      <div className="bg-red-500/10 border-b border-white/10 px-3 py-2">
                        <p className="text-[9px] text-red-400 font-medium mb-1">ENTFERNEN:</p>
                        <pre className="text-[10px] font-mono text-red-300/80 overflow-x-auto leading-relaxed line-through max-h-20">{edit.find}</pre>
                      </div>
                      {/* Replace */}
                      <div className="bg-emerald-500/10 border-b border-white/10 px-3 py-2">
                        <p className="text-[9px] text-emerald-400 font-medium mb-1">EINFÜGEN:</p>
                        <pre className="text-[10px] font-mono text-emerald-300/80 overflow-x-auto leading-relaxed max-h-20">{edit.replace}</pre>
                      </div>
                      {/* Action */}
                      <div className="px-3 py-2 bg-white/[0.02] flex items-center justify-between">
                        {edit.applied ? (
                          <span className="text-[10px] text-emerald-400 flex items-center gap-1"><Check className="w-3 h-3" /> Übernommen</span>
                        ) : edit.failed ? (
                          <span className="text-[10px] text-red-400 flex items-center gap-1"><X className="w-3 h-3" /> Text nicht gefunden</span>
                        ) : (
                          <button onClick={() => handleApplyEdit(m, edit, ei)}
                            className="text-[10px] px-3 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 font-medium transition-all">
                            Übernehmen
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-2">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${editMode ? 'bg-amber-500/20' : 'bg-violet-500/20'}`}>
              {editMode ? <Wand2 className="w-3 h-3 text-amber-400" /> : <Bot className="w-3 h-3 text-violet-400" />}
            </div>
            <div className="px-3 py-2 rounded-xl bg-white/5 border border-white/10">
              <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-white/10 flex-shrink-0">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-gray-600">
            {editMode ? 'Enter = senden · Shift+Enter = neue Zeile' : 'Enter = senden · Shift+Enter = neue Zeile'}
          </span>
          {editMode && (
            <span className="text-[10px] text-amber-400/70">Edit-Modus aktiv</span>
          )}
        </div>
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={editMode ? 'Was soll geändert werden?' : 'Frage oder Aufgabe…'}
            rows={2}
            className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none"
          />
          <button onClick={send} disabled={!input.trim() || loading}
            className={`p-2.5 rounded-xl border transition-all disabled:opacity-40 ${editMode ? 'bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/30 text-amber-300' : 'bg-violet-500/20 hover:bg-violet-500/30 border-violet-500/30 text-violet-300'}`}>
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Dev Train Panel ───────────────────────────────────────────────────────

interface DevTrainPanelProps {
  modelInfo: ModelInfo | null;
  selectedVersionPath: string;
  datasets: DatasetInfo[];
  onNavigateToAnalysis: (vid: string) => void;
}

export default function DevTrainPanel({ modelInfo, selectedVersionPath, datasets, onNavigateToAnalysis }: DevTrainPanelProps) {
  const { currentTheme } = useTheme();
  const { success, error }      = useNotification();
  const { settings: aiSettings } = useAISettings();

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
      return sessionStorage.getItem('devTrainBannerDismissed') === 'true';
    } catch {
      return false;
    }
  });
  const [showPathsModal, setShowPathsModal] = useState(false);
  const [outputPath, setOutputPath] = useState('');
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

  // Error Modal States
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorTitle, setErrorTitle] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [errorDetails, setErrorDetails] = useState('');
  const [isSendingError, setIsSendingError] = useState(false);
  const [aiPrefill, setAiPrefill] = useState('');
  const [aiForceEditMode, setAiForceEditMode] = useState(false);

  const lineCount = useMemo(() => Math.max(1, (script || '').split('\n').length), [script]);
  const highlightedHtml = useMemo(() => highlightPythonToHtml(script || ''), [script]);


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
      updateScript(currentScriptId, script);
      setSavedScript(script);
      setIsDirty(false);
      success('Aktualisiert', 'Skript in der Bibliothek aktualisiert!');
    } else {
      // Skript ist neu → Dialog für Namen zeigen
      setSaveName('Mein Trainings-Skript');
      setShowSaveDialog(true);
    }
  };

  const handleSaveWithName = (name: string) => {
    if (!name.trim()) return;
    saveScript(name.trim(), script);
    const allScripts = loadScripts();
    const newScript = allScripts[0];
    if (newScript) {
      setCurrentScriptId(newScript.id);
    }
    setSavedScript(script);
    setIsDirty(false);
    setShowSaveDialog(false);
    setSaveName('');
    success('Gespeichert', `Skript "${name}" in der Bibliothek gespeichert!`);
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

    listen<{ data: Partial<TrainingProgress> }>('training-progress', e => {
      const d = e.payload.data;
      if (d.train_loss != null) setLoss(pts => [...pts, { step: d.step ?? pts.length, epoch: d.epoch ?? 0, train_loss: d.train_loss!, val_loss: d.val_loss ?? undefined }]);
      setJob(j => j ? { ...j, status: 'running', progress: { ...j.progress, ...d } as TrainingProgress } : null);
    }).then(fn => { u1 = fn; });

    listen<{ line: string }>('dev-training-output', e => {
      setOutput(o => o + e.payload.line + '\n');
      setTimeout(() => outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight }), 50);
    }).then(fn => { u2 = fn; });

    listen('training-complete', () => {
      // Timeout bereinigen
      if ((window as any).__devTrainingTimeout) {
        clearTimeout((window as any).__devTrainingTimeout);
        delete (window as any).__devTrainingTimeout;
      }
      setRunning(false);
      setJob(j => j ? { ...j, status: 'completed' } : null);
      setOutput(o => o + '\n✅ Training abgeschlossen!');
      invoke('disable_prevent_sleep').catch(() => {});
    }).then(fn => { u3 = fn; });

    listen<{ data?: { error?: string; details?: string } }>('training-error', e => {
      const d = e.payload.data;
      // Timeout bereinigen
      if ((window as any).__devTrainingTimeout) {
        clearTimeout((window as any).__devTrainingTimeout);
        delete (window as any).__devTrainingTimeout;
      }
      setRunning(false);
      setJob(j => j ? { ...j, status: 'failed', error: d?.error ?? 'Fehler' } : null);
      setOutput(o => o + '\n❌ ' + (d?.error ?? 'Fehler') + (d?.details ? '\n' + d.details : ''));
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
    if (isDirty) {
      error('Ungespeicherte Änderungen', 'Bitte speichere dein Skript zuerst mit Cmd+S oder dem Speichern-Button.');
      return;
    }
    if (!script.trim() || !modelInfo) { error('Fehler', 'Kein Modell ausgewählt oder Skript leer.'); return; }
    
    // DEBUG: Log what we're actually sending
    console.log('🔥 DEBUG: Script das gesendet wird:');
    console.log(script);
    console.log('🔥 DEBUG: Script-Länge:', script.length);
    
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
      setOutput(`🚀 Training gestartet (Job: ${job.id})...\n`);
      invoke('enable_prevent_sleep').catch(() => {});
      devSessionIdRef.current = job.id;
      devStartedAtRef.current = Date.now();
      setShowDashboard(true);
      setIsDashMinimized(false);
      success('Gestartet!', 'Dev Training läuft…');
      
      // Timeout: Wenn nach 5 Minuten kein Event kommt, als fehlgeschlagen markieren
      const timeoutId = setTimeout(() => {
        setJob(j => {
          if (j && (j.status === 'pending' || j.status === 'running')) {
            setRunning(false);
            error('Timeout', 'Training hat nicht innerhalb von 5 Minuten geantwortet.');
            return { ...j, status: 'failed', error: 'Timeout: Kein Response vom Backend' };
          }
          return j;
        });
      }, 5 * 60 * 1000);

      // Cleanup-Funktion speichern (optional)
      (window as any).__devTrainingTimeout = timeoutId;
    } catch (err: unknown) {
      setOutput(`❌ ${String(err)}`);
      setRunning(false);
      setJob(null);
      error('Fehler', String(err));
    }
  };

  const handleStop = async () => {
    try { 
      await invoke('stop_training'); 
    } catch { /* ignore */ }
    invoke('disable_prevent_sleep').catch(() => {});
    setRunning(false); 
    setJob(null);
    setOutput(o => o + '\n⏹️  Training gestoppt.');
    
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
      const response = await fetch(
        'https://webcontrol-hq-api.karol-paschek.workers.dev/api/app-errors',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            site_id: 'frametrain',
            error_type: analysis.category === 'code' ? 'devtrain:code' : `devtrain:${analysis.category}`,
            title: errorTitle || 'Dev Train Fehler',
            message: errorMessage,
            details: errorDetails,
            logs: output,           // Alle Output-Logs
            script_full: script,     // Komplettes Skript
            error_analysis: analysis.category,
            error_category: analysis.title,
            platform: navigator.platform || 'unknown',
            app_version: 'desktop-app',
            timestamp: new Date().toISOString(),
          }),
        }
      );
      
      if (response.ok) {
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
    setAiForceEditMode(true);
    setAiPrefill(
      `[Dev Train Fehler]\n\n` +
      `Bitte repariere mein Dev-Train Skript.\n\n` +
      `FEHLER:\n${errorContext}\n\n` +
      `WICHTIG: DATASET_PATH ist ein lokaler Ordner von FrameTrain. ` +
      `Falls ich fälschlich load_dataset(DATASET_PATH) benutze, schlage eine robuste Alternative vor (z.B. load_from_disk oder passende load_dataset(..., data_files=...)).\n\n` +
      `Gib mir konkrete ##EDIT_START## Blöcke, damit ich die Änderungen direkt übernehmen kann.`
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
              <div className="flex items-center gap-2"><Terminal className="w-4 h-4 text-blue-400" /><span className="text-blue-300 font-semibold text-sm">Dev Train Mode</span></div>
              <button onClick={() => setDismissed(true)} className="p-1 rounded-lg hover:bg-white/10 text-blue-400/60 hover:text-white transition-all"><X className="w-3.5 h-3.5" /></button>
            </div>
            <p className="text-gray-400 text-xs">Eigenes Python-Skript mit vollem Zugriff auf alle Pfade. Gleiche Live-Loss-Kurven wie beim Standard-Training.</p>
          </div>
        )}

        {/* Model + Dataset Row */}
        <button
          onClick={() => setShowPathsModal(!showPathsModal)}
          className="w-full px-4 py-3 rounded-2xl border border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/15 transition-all flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-blue-300">Pfade konfigurieren</span>
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
                <span className="text-sm font-medium text-white">Modell</span>
              </div>
              <RefRow color="text-emerald-400" label="MODEL_PATH" value={modelPath} hint={modelInfo?.name} />
              
              <div className="flex items-center gap-2 mb-1 mt-3">
                <span className="text-sm font-medium text-white">Version</span>
              </div>
              <div className="text-[11px] font-mono text-gray-400">
                {modelInfo?.name ? `${modelInfo.name}` : 'Keine Version'}
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-white/10" />

            {/* Dataset Block */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-white">Dataset</span>
              </div>
              {dsRefs.map(r => <RefRow key={r.key} color="text-blue-400" label={r.key} value={r.value} hint={r.name} />)}
            </div>

            {/* Divider */}
            <div className="border-t border-white/10" />

            {/* Output Path */}
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
          {/* Toolbar — always visible */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-slate-900">
            <div className="flex items-center gap-3">
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
                  title={fileOpen ? (isDirty ? 'Ungespeicherte Änderungen' : 'Datei schließen') : ''}
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
                  title={fileOpen && isDirty ? 'Speichern' : ''}
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
                  title={fileOpen ? (expanded ? 'Verkleinern' : 'Vollbild') : ''}
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
                <span className={`text-sm font-medium ${fileOpen ? 'text-gray-300' : 'text-gray-600'}`}>
                  {fileOpen ? 'train.py' : 'Kein Dokument'}
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
                // Welcome-Modus: nur Neue Datei + Datei laden
                <>
                  <button onClick={handleNewFile}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/25 text-emerald-400 text-xs font-medium transition-all">
                    <FileCode className="w-3.5 h-3.5" /> Neue Datei
                  </button>
                  <button onClick={() => setShowLib(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/25 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> Datei laden
                  </button>
                </>
              ) : (
                // Editor-Modus: volle Toolbar
                <>
                  {isDirty && (
                    <button onClick={handleSave}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-xs font-medium transition-all">
                      <Save className="w-3.5 h-3.5" /> Speichern (⌘S)
                    </button>
                  )}
                  <button onClick={() => setShowLib(true)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 text-amber-400 text-xs font-medium transition-all">
                    <FolderClosed className="w-3.5 h-3.5" /> Bibliothek
                  </button>
                  {aiSettings.enabled && (
                    <button onClick={() => setShowAI(v => !v)}
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : 'bg-white/5 text-gray-400 hover:text-white border-white/10'}`}>
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
                  <button onClick={() => setExpanded(v => !v)}
                    className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all">
                    {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                  {isRunning ? (
                    <button onClick={handleStop}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 text-xs font-medium transition-all">
                      <Square className="w-3.5 h-3.5" /> Stopp
                    </button>
                  ) : (
                    <button onClick={handleStart} disabled={!script.trim() || !modelInfo}
                      className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-xs font-semibold hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed`}>
                      <Play className="w-3.5 h-3.5" /> Training starten
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
              <p className="text-gray-500 text-sm mb-8">Öffne oder erstelle eine Datei um zu starten</p>
              <div className="flex gap-4">
                <button
                  onClick={handleNewFile}
                  className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-emerald-500/20 bg-emerald-500/8 hover:bg-emerald-500/15 hover:border-emerald-500/40 transition-all group"
                >
                  <FileCode className="w-7 h-7 text-emerald-500 group-hover:text-emerald-400" />
                  <div>
                    <p className="font-semibold text-white text-sm">Neue Datei</p>
                    <p className="text-xs text-gray-500 mt-1">Tippe <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-400 font-mono text-[10px]">!</kbd> + Leertaste für Template</p>
                  </div>
                </button>
                <button
                  onClick={() => setShowLib(true)}
                  className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl border border-amber-500/20 bg-amber-500/8 hover:bg-amber-500/15 hover:border-amber-500/40 transition-all group"
                >
                  <FolderClosed className="w-7 h-7 text-amber-500 group-hover:text-amber-400" />
                  <div>
                    <p className="font-semibold text-white text-sm">Datei laden</p>
                    <p className="text-xs text-gray-500 mt-1">Aus deiner Bibliothek</p>
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
                      placeholder={"# Fange an zu tippen…\n# Tippe '! ' + Leertaste um das Template zu laden"}
                      onChange={e => {
                        const newVal = e.target.value;
                        if (newVal === '! ') {
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
                      onApplyEdit={setScript}
                      onReplaceScript={setScript}
                      onClose={() => setShowAI(false)}
                      initialInput={aiPrefill}
                      forceEditMode={aiForceEditMode}
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
                  {isRunning ? 'Training läuft…' : `Status: ${currentJob?.status}`}
                </span>
              </div>
              {progress && <span className="text-gray-400 text-xs">Epoch {progress.epoch}/{progress.total_epochs}</span>}
            </div>

            {progress && (
              <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`}
                  style={{ width: `${progress.progress_percent ?? 0}%` }} />
              </div>
            )}

            {lossPoints.length > 1 && (
              <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3">
                <p className="text-xs text-gray-500 mb-2">Loss-Verlauf</p>
                <LossChart points={lossPoints} />
              </div>
            )}

            {progress && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'Train Loss', value: progress.train_loss?.toFixed(4) ?? '—', icon: <TrendingDown className="w-3.5 h-3.5" /> },
                  { label: 'Val Loss',   value: progress.val_loss?.toFixed(4)   ?? '—', icon: <BarChart3    className="w-3.5 h-3.5" /> },
                  { label: 'LR',         value: progress.learning_rate?.toExponential(2) ?? '—', icon: <Zap className="w-3.5 h-3.5" /> },
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
                  <span className="text-[10px] text-gray-500">Ausgabe</span>
                </div>
                <div ref={outputRef} className="p-3 max-h-48 overflow-y-auto">
                  {isRunning && !output && <p className="text-gray-600 text-[10px] animate-pulse">Warte auf Python-Output…</p>}
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
          setAiForceEditMode(true);
          setAiPrefill(
            `[Dev Train Fehler]\n\n` +
            `Bitte korrigiere mein Skript (Edit-Modus).\n\n` +
            `FEHLER:\n${err}\n\n` +
            `SCRIPT:\n${s}\n`
          );
        }}
      />
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
    <div className="flex items-start gap-3 py-0.5">
      <span className={`${color} min-w-[140px] flex-shrink-0`}>{label}</span>
      <div className="min-w-0 flex-1">
        <span className={`break-all ${value ? 'text-gray-300' : 'text-gray-600 italic'}`}>
          {value || 'nicht gesetzt'}
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
          title="Kopieren"
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

// ── Default Script Generator ──────────────────────────────────────────────

function generateDefaultScript(model: ModelInfo | null, datasets: DatasetInfo[], outputPath: string): string {
  const ds = datasets[0];
  const modelPathDefault  = model?.local_path || model?.source_path || model?.name || '';
  const datasetPathDefault = ds?.storage_path || '';
  const outputPathDefault  = outputPath.replace('<job_id>', 'dev_train').replace('{wird beim Start gesetzt}', 'dev_train');

  return `#!/usr/bin/env python3
# FrameTrain – Dev Train Script
# Passe dieses Template an dein Modell und Dataset an.

import os
import sys
from pathlib import Path

# ── Pfade (von FrameTrain als ENV-Vars gesetzt) ─────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPathDefault}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${datasetPathDefault}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPathDefault}")

# ── Imports ───────────────────────────────────────────────────────────────
# TODO: Importiere torch, transformers, datasets, etc.
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk, load_dataset

# ── Dataset laden ─────────────────────────────────────────────────────────
def load_frametrain_dataset(path: str):
	    """
	    FrameTrain liefert DATASET_PATH meistens als lokalen Ordner.
	    - Wenn das Dataset via datasets.Dataset(Dict).save_to_disk() gespeichert wurde:
	      => load_from_disk(path) benutzen.
	    - Sonst versuchen wir JSON/JSONL/CSV/TSV Dateien im Ordner zu finden und mit load_dataset(...) zu laden.
	    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")

    # HuggingFace save_to_disk Marker
    if p.is_dir() and ((p / "dataset_info.json").exists() or (p / "state.json").exists() or (p / "dataset_dict.json").exists()):
        return load_from_disk(str(p))

    # FrameTrain Split-Ordner Struktur:
    #   <DATASET_PATH>/train/train.json
    #   <DATASET_PATH>/val/validation.json
    #   <DATASET_PATH>/test/test.json
    if p.is_dir():
        def pick_file(*parts: str):
            f = p.joinpath(*parts)
            return f if f.exists() else None

        train_f = pick_file("train", "train.json") or pick_file("train", "train.jsonl") or pick_file("train", "train.csv") or pick_file("train", "train.tsv")
        val_f   = pick_file("val", "validation.json") or pick_file("val", "val.json") or pick_file("val", "validation.jsonl") or pick_file("val", "validation.csv") or pick_file("val", "validation.tsv")
        test_f  = pick_file("test", "test.json") or pick_file("test", "test.jsonl") or pick_file("test", "test.csv") or pick_file("test", "test.tsv")

        if train_f:
            data_files = {"train": str(train_f)}
            if val_f:  data_files["validation"] = str(val_f)
            if test_f: data_files["test"] = str(test_f)

            ext = train_f.suffix.lower()
            if ext in [".json", ".jsonl"]:
                return load_dataset("json", data_files=data_files)
            if ext == ".tsv":
                return load_dataset("csv", data_files=data_files, delimiter="\\t")
            return load_dataset("csv", data_files=data_files)

    # Dateien im Ordner suchen (train/validation/test.*)
    if p.is_dir():
        files = list(p.rglob("*.json")) + list(p.rglob("*.jsonl")) + list(p.rglob("*.csv")) + list(p.rglob("*.tsv"))
        if not files:
            raise RuntimeError(
                "Konnte keine Daten-Dateien im DATASET_PATH finden. "
                "Wenn es ein gespeichertes HF-Dataset ist, stelle sicher dass dataset_info.json/state.json vorhanden ist."
            )

        # Heuristik: train/validation/test nach Dateinamen
        def pick(split: str):
            for f in files:
                name = f.name.lower()
                if split in name:
                    return f
            return None

        train_f = pick("train")
        val_f   = pick("valid") or pick("val") or pick("validation")
        test_f  = pick("test")

        data_files = {}
        if train_f: data_files["train"] = str(train_f)
        if val_f:   data_files["validation"] = str(val_f)
        if test_f:  data_files["test"] = str(test_f)
        if not data_files:
            # Fallback: alles als train
            data_files["train"] = [str(f) for f in files]

        # Loader nach Extension wählen
        ext = (train_f or val_f or test_f or files[0]).suffix.lower()
        if ext in [".json", ".jsonl"]:
            return load_dataset("json", data_files=data_files)
        if ext == ".csv":
            return load_dataset("csv", data_files=data_files)
        if ext == ".tsv":
            return load_dataset("csv", data_files=data_files, delimiter="\\t")

    # Wenn path eine Datei ist
    if p.is_file():
        ext = p.suffix.lower()
        if ext in [".json", ".jsonl"]:
            return load_dataset("json", data_files=str(p))
        if ext == ".csv":
            return load_dataset("csv", data_files=str(p))
        if ext == ".tsv":
            return load_dataset("csv", data_files=str(p), delimiter="\\t")

    raise RuntimeError(f"Unbekanntes Dataset-Format: {path}")

dataset = load_frametrain_dataset(DATASET_PATH)
print("✅ Dataset geladen:", dataset)

# ── Modell & Tokenizer ────────────────────────────────────────────────────
# TODO: Lade Modell und Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

# ── Training ──────────────────────────────────────────────────────────────
# TODO: Implementiere Trainings-Loop
# for epoch in range(num_epochs):
#     for batch in dataset:
#         # forward pass
#         # loss berechnen
#         # backward pass
#         # optimizer step

# ── Metriken ──────────────────────────────────────────────────────────────
# TODO: Berechne Metriken (Loss, Accuracy, etc.)
# train_loss = ...
# val_loss = ...

# ── Speichern ─────────────────────────────────────────────────────────────
# TODO: Speichere Modell und Tokenizer
# model.save_pretrained(OUTPUT_PATH)
# tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Skript abgeschlossen!")
`;
}
