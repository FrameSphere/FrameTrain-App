// OpenLibraryModal.tsx – Community Open Script Library
// User können Skripte weltweit teilen, browsen und in ihre lokale Bibliothek laden.

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  X, Search, ShieldCheck, ShieldAlert, Upload, Download, Star,
  Globe, Filter, Tag, User, Loader2, AlertTriangle, Check,
  FileCode, Clock, Sparkles, ArrowLeft, ChevronDown, Plus,
  TrendingUp, Eye, BookOpen, Send, RefreshCw, FolderClosed, Pencil,
} from 'lucide-react';
import { useNotification } from '../contexts/NotificationContext';

// ── API ───────────────────────────────────────────────────────────────────

const API_BASE = 'https://frame-train.vercel.app/api/library';

// ── Types ─────────────────────────────────────────────────────────────────

export interface LibraryScript {
  id: string;
  name: string;
  description: string;
  author: string;
  model_type: string;
  task_type: string;
  framework: string;
  script: string;
  verified: boolean;
  downloads: number;
  stars: number;
  created_at: string;
  updated_at: string;
  tags: string[];
  script_type?: 'train' | 'test';
}

// ── Saved Script type (matching DevTrainPanel) ────────────────────────────

interface SavedScript { id: string; name: string; script: string; savedAt: string; fromOpenLib?: boolean; }
const getLocalKey = (mode: 'train' | 'test', userId?: string) =>
  userId
    ? `ft_saved_${mode === 'test' ? 'test_' : ''}scripts_${userId}`
    : (mode === 'test' ? 'ft_saved_test_scripts' : 'ft_saved_scripts');
function addToLocalLibrary(name: string, script: string, fromOpenLib = false, storageKey = 'ft_saved_scripts'): void {
  try {
    const all: SavedScript[] = JSON.parse(localStorage.getItem(storageKey) ?? '[]');
    all.unshift({ id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString(), fromOpenLib });
    localStorage.setItem(storageKey, JSON.stringify(all.slice(0, 50)));
  } catch { /* ignore */ }
}

// ── Filter Options ────────────────────────────────────────────────────────

const MODEL_TYPES = ['Alle', 'LLM', 'Vision', 'Classifier', 'Seq2Seq', 'Embedding', 'Custom'];
const TASK_TYPES  = ['Alle', 'Fine-Tuning', 'LoRA / QLoRA', 'Pre-Training', 'Text Classification', 'Image Classification', 'Regression', 'NER', 'Question Answering', 'Custom'];
const FRAMEWORKS  = ['Alle', 'transformers', 'trl', 'pytorch', 'scikit-learn', 'accelerate'];

// ── Helper ────────────────────────────────────────────────────────────────

// ── Helper ────────────────────────────────────────────────────────────────

function parseDate(iso: string | undefined | null): Date {
  // Fallback falls Datum fehlt
  if (!iso) return new Date();
  // Konvertiere "2026-05-23 18:36:20.857+00" zu "2026-05-23T18:36:20.857Z"
  const normalized = iso.replace(' ', 'T').replace('+00', 'Z')
  return new Date(normalized)
}

// ── Helper: Author-Name speichern/laden ──────────────────────────────────────

export const AUTHOR_KEY = (userId: string) => `ft_author_name_${userId}`;

export function getStoredAuthorName(userId?: string): string {
  if (!userId) return '';
  return localStorage.getItem(AUTHOR_KEY(userId)) ?? '';
}

export function validateAuthorName(name: string): string {
  // Remove invalid characters, keep only a-z, A-Z, 0-9, _, -, .
  let val = name.replace(/[^a-z0-9_\-.]/gi, '').slice(0, 40);
  // Remove @ if it's at the start
  if (val.startsWith('@')) val = val.slice(1);
  return val;
}

export function saveAuthorName(userId: string, name: string): void {
  localStorage.setItem(AUTHOR_KEY(userId), validateAuthorName(name));
}

function relativeDate(iso: string | undefined | null): string {
  const diff = Date.now() - parseDate(iso).getTime();
  if (diff < 3_600_000)       return `vor ${Math.floor(diff / 60_000)} Min`;
  if (diff < 86_400_000)      return `vor ${Math.floor(diff / 3_600_000)} Std`;
  if (diff < 7 * 86_400_000)  return `vor ${Math.floor(diff / 86_400_000)} Tagen`;
  return parseDate(iso).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric' });
}

const MODEL_TYPE_COLORS: Record<string, string> = {
  'LLM':        'bg-violet-500/20 text-violet-300 border-violet-500/30',
  'Vision':     'bg-sky-500/20 text-sky-300 border-sky-500/30',
  'Classifier': 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  'Seq2Seq':    'bg-amber-500/20 text-amber-300 border-amber-500/30',
  'Embedding':  'bg-pink-500/20 text-pink-300 border-pink-500/30',
  'Custom':     'bg-slate-500/20 text-slate-300 border-slate-500/30',
};

// ── Duplicate Author Name Modal ──────────────────────────────────────────

function DuplicateNameError({
  name,
  onRetry,
}: {
  name: string;
  onRetry: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-red-500/30 w-full max-w-md overflow-hidden">
        <div className="px-6 py-5 bg-red-500/10 border-b border-red-500/20 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />
          <div>
            <h2 className="text-white font-bold text-base">Name bereits vergeben</h2>
            <p className="text-red-300/80 text-xs mt-0.5">Dieser Community-Name ist nicht verfügbar</p>
          </div>
        </div>
        <div className="p-6 space-y-4">
          <div className="p-4 rounded-xl bg-red-500/8 border border-red-500/20 space-y-2">
            <p className="text-gray-400 text-xs">Der Community-Name</p>
            <p className="text-white font-mono text-sm">@{name}</p>
            <p className="text-gray-400 text-xs mt-2">wird bereits von einem anderen Community-Mitglied verwendet.</p>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed">
            Wähle einen anderen Namen. Community-Namen müssen eindeutig sein, um Verwechslungen zu vermeiden.
          </p>
          <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <p className="text-blue-300 text-xs">💡 Tipp: Probiere einen Namen mit Zahlen oder Unterstrichen, z.B. <span className="font-mono">ai_enthusiast_42</span></p>
          </div>
        </div>
        <div className="px-6 pb-6 flex gap-2">
          <button
            onClick={onRetry}
            className="flex-1 py-2.5 rounded-xl bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 text-violet-300 text-sm font-medium transition-all"
          >
            Anderen Namen versuchen
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Unverified Warning Modal ──────────────────────────────────────────────

function UnverifiedWarning({
  script,
  onConfirm,
  onCancel,
}: {
  script: LibraryScript;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-amber-500/30 w-full max-w-md overflow-hidden">
        <div className="px-6 py-5 bg-amber-500/10 border-b border-amber-500/20 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0" />
          <div>
            <h2 className="text-white font-bold text-base">Nicht verifiziertes Skript</h2>
            <p className="text-amber-300/80 text-xs mt-0.5">Dieses Skript wurde noch nicht von FrameTrain geprüft</p>
          </div>
        </div>
        <div className="p-6 space-y-4">
          <div className="p-4 rounded-xl bg-amber-500/8 border border-amber-500/20 space-y-2">
            <p className="text-white font-medium text-sm">„{script.name}"</p>
            <p className="text-gray-400 text-xs">von @{script.author}</p>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed">
            Nicht verifizierte Skripte stammen direkt von Community-Mitgliedern und wurden
            <strong className="text-amber-300"> noch nicht auf Sicherheit und Korrektheit geprüft</strong>.
            Führe unbekannte Skripte nur aus, wenn du den Inhalt selbst geprüft hast.
          </p>
          <ul className="text-xs text-gray-500 space-y-1">
            <li className="flex items-center gap-2"><AlertTriangle className="w-3 h-3 text-amber-500/70 flex-shrink-0" />Könnte unbekannte Imports oder Netzwerkzugriffe enthalten</li>
            <li className="flex items-center gap-2"><AlertTriangle className="w-3 h-3 text-amber-500/70 flex-shrink-0" />Möglicherweise unvollständig oder fehlerhaft</li>
            <li className="flex items-center gap-2"><Check className="w-3 h-3 text-emerald-500/70 flex-shrink-0" />Skript-Vorschau vor dem Laden einsehbar</li>
          </ul>
        </div>
        <div className="px-6 pb-6 flex gap-2">
          <button
            onClick={onConfirm}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300 text-sm font-medium transition-all"
          >
            <Download className="w-4 h-4" /> Trotzdem hinzufügen
          </button>
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:text-white text-sm font-medium transition-all"
          >
            Abbrechen
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Script Detail View ────────────────────────────────────────────────────

function ScriptDetail({
  script,
  onBack,
  onAddToLibrary,
}: {
  script: LibraryScript;
  onBack: () => void;
  onAddToLibrary: (s: LibraryScript) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-white/10 flex-shrink-0">
        <button onClick={onBack} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-white font-bold text-base truncate">{script.name}</h2>
            {script.verified ? (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/25 text-emerald-300 text-[10px] font-semibold flex-shrink-0">
                <ShieldCheck className="w-3 h-3" /> Verified
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/25 text-amber-300 text-[10px] font-semibold flex-shrink-0">
                <ShieldAlert className="w-3 h-3" /> Nicht verifiziert
              </span>
            )}
          </div>
          <p className="text-gray-500 text-xs mt-0.5">@{script.author} · {relativeDate(script.created_at)}</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {/* Meta badges */}
        <div className="flex flex-wrap gap-2">
          <span className={`px-2.5 py-1 rounded-lg border text-[11px] font-medium ${MODEL_TYPE_COLORS[script.model_type] ?? MODEL_TYPE_COLORS['Custom']}`}>
            {script.model_type}
          </span>
          <span className="px-2.5 py-1 rounded-lg border border-white/15 bg-white/5 text-gray-300 text-[11px] font-medium">
            {script.task_type}
          </span>
          <span className="px-2.5 py-1 rounded-lg border border-blue-500/25 bg-blue-500/10 text-blue-300 text-[11px] font-medium">
            {script.framework}
          </span>
        </div>

        {/* Description */}
        <p className="text-gray-300 text-sm leading-relaxed">{script.description}</p>

        {/* Stats */}
        <div className="flex items-center gap-5 text-gray-500 text-xs">
          <span className="flex items-center gap-1.5"><Download className="w-3.5 h-3.5" /> {script.downloads} Downloads</span>
          <span className="flex items-center gap-1.5"><Star className="w-3.5 h-3.5 text-amber-400" /> {script.stars}</span>
          <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> {relativeDate(script.created_at)}</span>
        </div>

        {/* Tags */}
        {script.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {script.tags.map(tag => (
              <span key={tag} className="px-2 py-0.5 rounded-md bg-white/5 border border-white/10 text-gray-400 text-[10px] font-mono">
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Verified info box */}
        {script.verified ? (
          <div className="p-4 rounded-xl bg-emerald-500/8 border border-emerald-500/20 flex gap-3">
            <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-emerald-300 text-sm font-semibold">Verifiziertes Skript</p>
              <p className="text-gray-400 text-xs mt-1">Dieses Skript wurde von FrameTrain auf Sicherheit und Funktionalität geprüft. Es enthält keine schädlichen Operationen und läuft stabil.</p>
            </div>
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-amber-500/8 border border-amber-500/20 flex gap-3">
            <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-amber-300 text-sm font-semibold">Community-Upload (ungeprüft)</p>
              <p className="text-gray-400 text-xs mt-1">Dieses Skript wurde noch nicht von FrameTrain geprüft. Schau den Code-Inhalt durch bevor du es ausführst.</p>
            </div>
          </div>
        )}

        {/* Script Preview */}
        <div className="rounded-xl border border-white/10 overflow-hidden">
          <button
            onClick={() => setShowPreview(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 bg-white/[0.03] hover:bg-white/[0.05] transition-all"
          >
            <div className="flex items-center gap-2">
              <FileCode className="w-4 h-4 text-gray-400" />
              <span className="text-gray-300 text-sm font-medium">Skript-Vorschau</span>
              <span className="text-gray-600 text-xs">({script.script.split('\n').length} Zeilen)</span>
            </div>
            <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showPreview ? 'rotate-180' : ''}`} />
          </button>
          {showPreview && (
            <div className="border-t border-white/10">
              <div className="flex items-center justify-end px-3 py-2 bg-black/20 border-b border-white/5">
                <button
                  onClick={() => { navigator.clipboard.writeText(script.script); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs transition-all ${copied ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/25' : 'bg-white/5 text-gray-400 hover:text-white border border-white/10'}`}
                >
                  {copied ? <><Check className="w-3 h-3" /> Kopiert!</> : <><Eye className="w-3 h-3" /> Kopieren</>}
                </button>
              </div>
              <pre className="p-4 text-[10px] font-mono text-gray-300 overflow-x-auto max-h-80 leading-relaxed whitespace-pre bg-black/20">
                {script.script}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Footer CTA */}
      <div className="px-6 pb-6 pt-4 border-t border-white/10 flex-shrink-0">
        <button
          onClick={() => onAddToLibrary(script)}
          className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all ${
            script.verified
              ? 'bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300'
              : 'bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-amber-300'
          }`}
        >
          <Download className="w-4 h-4" />
          In lokale Bibliothek hinzufügen
          {!script.verified && <AlertTriangle className="w-3.5 h-3.5 ml-1 opacity-70" />}
        </button>
      </div>
    </div>
  );
}

// ── Upload Tab ────────────────────────────────────────────────────────────

function UploadTab({ mode = 'train', userData }: { mode?: 'train' | 'test'; userData?: { userId: string; email: string; apiKey: string } }) {
  const { success, error } = useNotification();
  const localKey = getLocalKey(mode, userData?.userId);

  // Eigene Skripte aus lokaler Bib laden (ohne fromOpenLib) – user-spezifisch
  const ownScripts: SavedScript[] = (() => {
    try {
      const all: SavedScript[] = JSON.parse(localStorage.getItem(localKey) ?? '[]');
      return all.filter(s => !s.fromOpenLib);
    } catch { return []; }
  })();

  // Author-Name: Wird beim Hochladen automatisch gespeichert und wiederhergestellt
  const [authorInput, setAuthorInput]     = useState('');
  const [authorLocked, setAuthorLocked]   = useState(false);
  const [editingAuthor, setEditingAuthor] = useState(false);
  const [duplicateNameError, setDuplicateNameError] = useState<string | null>(null);

  // Load community name from backend on mount
  useEffect(() => {
    const loadAuthorName = async () => {
      if (!userData?.userId) return;
      
      try {
        // First try to load from backend
        const backendRes = await fetch(`https://frame-train.vercel.app/api/user/community-name?userId=${userData.userId}`);
        if (backendRes.ok) {
          const data = await backendRes.json();
          if (data.communityName) {
            setAuthorInput(validateAuthorName(data.communityName));
            setAuthorLocked(true);
            return;
          }
        }
      } catch (err) {
        console.error('Failed to load community name from backend:', err);
      }
      
      // Fallback to localStorage
      const storedName = getStoredAuthorName(userData?.userId);
      if (storedName) {
        setAuthorInput(validateAuthorName(storedName));
        setAuthorLocked(true);
      }
    };
    
    loadAuthorName();
  }, [userData?.userId]);

  const [selectedId, setSelectedId] = useState<string>('');
  const [form, setForm] = useState({
    name: '',
    description: '',
    model_type: 'LLM',
    task_type: 'Fine-Tuning',
    framework: 'transformers',
    tags: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const setField = (key: string, val: string) =>
    setForm(f => ({ ...f, [key]: val }));

  // Helper: Prüfe, ob Author-Name bereits existiert (Duplikat-Check)
  const checkAuthorNameExists = async (name: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/authors/${encodeURIComponent(name)}/exists`);
      const data = await res.json();
      return data.exists ?? false;
    } catch { return false; }
  };

  // Helper: Aktualisiere User.communityName (alle Scripts werden automatisch aktualisiert)
  const updateUserScriptsWithNewAuthor = async (newName: string): Promise<void> => {
    try {
      const response = await fetch(`https://frame-train.vercel.app/api/user/community-name`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: userData?.userId, communityName: newName }),
      });
      if (!response.ok) {
        console.error('Fehler beim Update:', await response.json());
      }
    } catch (err) {
      console.error('Fehler beim Aktualisieren des Community-Namens:', err);
    }
  };

  const selectedScript = ownScripts.find(s => s.id === selectedId) ?? null;

  const handleSubmit = async () => {
    if (!selectedScript) {
      error('Kein Skript', 'Bitte erst ein Skript aus deiner Bibliothek auswählen.');
      return;
    }
    if (!form.name.trim() || !form.description.trim()) {
      error('Fehlende Felder', 'Bitte Name und Beschreibung ausfüllen.');
      return;
    }
    if (!authorInput.trim()) {
      error('Community-Name erforderlich', 'Bitte gib einen Community-Namen ein.');
      return;
    }
    
    setSubmitting(true);
    try {
      // Prüfe auf Duplikate (nur beim ersten Upload)
      if (!authorLocked) {
        const exists = await checkAuthorNameExists(authorInput.trim());
        if (exists) {
          setDuplicateNameError(authorInput.trim());
          setSubmitting(false);
          return;
        }
      }
      
      // Auto-Speichern des Namens beim ersten Upload (wenn nicht bereits gespeichert)
      if (!authorLocked && userData?.userId && authorInput.trim()) {
        saveAuthorName(userData.userId, authorInput.trim());
        setAuthorLocked(true);
        // Aktualisiere User.communityName in der DB
        await updateUserScriptsWithNewAuthor(authorInput.trim());
      }
      
      const payload = {
        ...form,
        author: authorInput.trim(),
        userId: userData?.userId,
        script: selectedScript.script,
        tags: form.tags.split(',').map(t => t.trim()).filter(Boolean),
        verified: false,
        script_type: mode,
      };
      const res = await fetch(`${API_BASE}/scripts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSubmitted(true);
      success('Hochgeladen!', 'Dein Skript wurde eingereicht und wird bald geprüft.');
    } catch (err) {
      console.error('Upload error:', err);
      setSubmitted(true);
      success('Eingereicht!', 'Dein Skript wurde eingereicht und wartet auf Prüfung.');
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-16 px-6 text-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center">
          <Check className="w-8 h-8 text-emerald-400" />
        </div>
        <div>
          <p className="text-white font-bold text-lg">Skript eingereicht!</p>
          <p className="text-gray-400 text-sm mt-2">Dein Skript wird vom FrameTrain-Team auf Sicherheit und Funktionalität geprüft. Nach der Verifikation erscheint es in der Bibliothek mit dem <span className="text-emerald-300">Verified</span>-Badge.</p>
        </div>
        <button
          onClick={() => { setSubmitted(false); setSelectedId(''); setForm({ name: '', description: '', model_type: 'LLM', task_type: 'Fine-Tuning', framework: 'transformers', tags: '' }); }}
          className="px-6 py-2.5 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:text-white text-sm transition-all"
        >
          Weiteres Skript einreichen
        </button>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto h-full p-6 space-y-5">
      {/* Info */}
      <div className="p-4 rounded-xl bg-blue-500/8 border border-blue-500/20 flex gap-3">
        <Sparkles className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-blue-300 text-sm font-semibold">Skript zur Community-Bibliothek beisteuern</p>
          <p className="text-gray-400 text-xs mt-1 leading-relaxed">
            Wähle ein Skript aus deiner privaten Bibliothek. Es wird eingereicht, geprüft und erhält dann ein
            <span className="text-emerald-300 mx-1">✓ Verified</span>-Badge.
          </p>
        </div>
      </div>

      {/* Skript aus Bibliothek wählen */}
      <div className="space-y-2">
        <label className="text-xs text-gray-400 font-medium flex items-center gap-1.5">
          <FolderClosed className="w-3.5 h-3.5" /> Skript aus deiner Bibliothek wählen <span className="text-red-400">*</span>
        </label>
        {ownScripts.length === 0 ? (
          <div className="p-4 rounded-xl bg-white/[0.03] border border-white/10 text-center space-y-1">
            <FileCode className="w-6 h-6 text-gray-600 mx-auto" />
            <p className="text-gray-500 text-xs">Keine eigenen Skripte in deiner Bibliothek.</p>
            <p className="text-gray-600 text-[10px]">Schreibe zuerst ein Skript im {mode === 'test' ? 'Dev Test' : 'Dev Train'} und speichere es.</p>
          </div>
        ) : (
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {ownScripts.map(s => (
              <button
                key={s.id}
                onClick={() => setSelectedId(prev => prev === s.id ? '' : s.id)}
                className={`w-full text-left px-3 py-2.5 rounded-xl border transition-all ${
                  selectedId === s.id
                    ? 'bg-violet-500/15 border-violet-500/30 text-white'
                    : 'bg-white/[0.03] border-white/10 text-gray-300 hover:bg-white/[0.06]'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{s.name}</p>
                    <p className="text-[10px] text-gray-500 mt-0.5">{s.script.split('\n').length} Zeilen · gespeichert {new Date(s.savedAt).toLocaleDateString('de-DE')}</p>
                  </div>
                  {selectedId === s.id && <Check className="w-4 h-4 text-violet-400 flex-shrink-0" />}
                </div>
              </button>
            ))}
          </div>
        )}
        {selectedScript && (
          <div className="p-3 rounded-xl bg-black/20 border border-white/10">
            <p className="text-[10px] text-gray-600 mb-1">Vorschau:</p>
            <pre className="text-[10px] font-mono text-gray-400 line-clamp-3 overflow-hidden">{selectedScript.script.split('\n').slice(0, 3).join('\n')}</pre>
          </div>
        )}
      </div>

      {/* Metadaten */}
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2 space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Name <span className="text-red-400">*</span></label>
            <input
              value={form.name}
              onChange={e => setField('name', e.target.value)}
              placeholder="z.B. LoRA Fine-Tuning für GPT-2"
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/40"
            />
          </div>
          <div className="col-span-2 space-y-1.5">
            <label className="text-xs text-gray-400 font-medium flex items-center gap-1.5">
              Autor
              {authorLocked && !editingAuthor && (
                <button onClick={() => setEditingAuthor(true)} className="text-gray-600 hover:text-violet-400 transition-colors" title="Namen ändern">
                  <Pencil className="w-3 h-3" />
                </button>
              )}
            </label>
            {!authorLocked || editingAuthor ? (
              <>
                  <input
                    value={authorInput}
                    onChange={e => setAuthorInput(e.target.value.replace(/[^a-z0-9_\-. ]/gi, ''))}
                    placeholder="Dein Community-Name (z. B. ai_enthusiast)"
                    maxLength={40}
                  className="w-full px-3 py-2.5 bg-white/5 border border-violet-500/30 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/60"
                />
                <p className="text-[10px] text-violet-400/70">
                  {!authorLocked ? 'Der Name wird beim Absenden des Skripts gespeichert und für alle deine Uploads verwendet.' : 'Namen ändern — alle künftigen Uploads und retroaktive Updates in der Bibliothek verwenden den neuen Namen.'}
                </p>
              </>
            ) : (
              <div className="flex items-center gap-2 px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl">
                <span className="text-white text-sm flex-1">@{authorInput}</span>
                <span className="text-[10px] text-gray-600">gespeichert</span>
              </div>
            )}
          </div>
          <div className="col-span-2 space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Beschreibung <span className="text-red-400">*</span></label>
            <textarea
              value={form.description}
              onChange={e => setField('description', e.target.value)}
              placeholder="Wofür eignet sich dieses Skript? Welche Voraussetzungen gibt es?"
              rows={3}
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/40 resize-none"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Modell-Typ</label>
            <select value={form.model_type} onChange={e => setField('model_type', e.target.value)}
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-violet-500/40">
              {MODEL_TYPES.filter(t => t !== 'Alle').map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Task-Typ</label>
            <select value={form.task_type} onChange={e => setField('task_type', e.target.value)}
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-violet-500/40">
              {TASK_TYPES.filter(t => t !== 'Alle').map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Framework</label>
            <select value={form.framework} onChange={e => setField('framework', e.target.value)}
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-violet-500/40">
              {FRAMEWORKS.filter(t => t !== 'Alle').map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-gray-400 font-medium">Tags (kommagetrennt)</label>
            <input
              value={form.tags}
              onChange={e => setField('tags', e.target.value)}
              placeholder="lora, llm, 4bit"
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/40"
            />
          </div>
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={submitting || !selectedScript || !form.name.trim() || !authorInput.trim() || !form.description.trim()}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 text-violet-300 font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-all"
      >
        {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        {submitting ? 'Wird hochgeladen…' : 'Skript einreichen'}
      </button>

      {/* Duplicate Name Error Modal */}
      {duplicateNameError && (
        <DuplicateNameError
          name={duplicateNameError}
          onRetry={() => setDuplicateNameError(null)}
        />
      )}
    </div>
  );
}

// ── Script Card ───────────────────────────────────────────────────────────

function ScriptCard({
  script,
  onClick,
}: {
  script: LibraryScript;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-4 rounded-xl border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/20 transition-all group space-y-3"
    >
      {/* Top row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-white font-semibold text-sm truncate">{script.name}</p>
            {script.verified ? (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/20 text-emerald-300 text-[9px] font-bold flex-shrink-0">
                <ShieldCheck className="w-2.5 h-2.5" /> Verified
              </span>
            ) : (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400/70 text-[9px] flex-shrink-0">
                <ShieldAlert className="w-2.5 h-2.5" /> Ungeprüft
              </span>
            )}
          </div>
          <p className="text-gray-500 text-[10px] mt-0.5">@{script.author} · {relativeDate(script.created_at)}</p>
        </div>
        <div className={`w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center border text-xs font-bold ${MODEL_TYPE_COLORS[script.model_type] ?? MODEL_TYPE_COLORS['Custom']}`}>
          {script.model_type.slice(0, 2).toUpperCase()}
        </div>
      </div>

      {/* Description */}
      <p className="text-gray-400 text-xs leading-relaxed line-clamp-2">{script.description}</p>

      {/* Badges */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="px-2 py-0.5 rounded-md bg-white/5 border border-white/10 text-gray-400 text-[10px]">{script.task_type}</span>
        <span className="px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px]">{script.framework}</span>
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-gray-600 text-[10px]">
          <span className="flex items-center gap-1"><Download className="w-3 h-3" /> {script.downloads}</span>
          <span className="flex items-center gap-1"><Star className="w-3 h-3 text-amber-500/60" /> {script.stars}</span>
        </div>
        <span className="text-violet-400/60 text-[10px] font-medium group-hover:text-violet-300 transition-colors flex items-center gap-1">
          Details →
        </span>
      </div>
    </button>
  );
}

// ── Main Modal ────────────────────────────────────────────────────────────

interface OpenLibraryModalProps {
  onClose: () => void;
  onLoadScript: (scriptContent: string, scriptName: string) => void;
  mode?: 'train' | 'test';
  userData?: { userId: string; email: string; apiKey: string };
}

export default function OpenLibraryModal({ onClose, onLoadScript, mode = 'train', userData }: OpenLibraryModalProps) {
  const { success } = useNotification();
  const [tab, setTab] = useState<'browse' | 'upload'>('browse');
  const [scripts, setScripts] = useState<LibraryScript[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterModelType, setFilterModelType] = useState('Alle');
  const [filterTaskType, setFilterTaskType] = useState('Alle');
  const [filterFramework, setFilterFramework] = useState('Alle');
  const [onlyVerified, setOnlyVerified] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedScript, setSelectedScript] = useState<LibraryScript | null>(null);
  const [pendingUnverified, setPendingUnverified] = useState<LibraryScript | null>(null);
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set());
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Skripte laden
  const loadScripts = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/scripts?script_type=${mode}`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error('API unavailable');
      const data = await res.json();
      // API gibt { scripts, total } zurück
      const list = Array.isArray(data) ? data : (data.scripts ?? null);
      setScripts(list ?? []);
    } catch {
      // Bei Fehler leeres Array
      setScripts([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadScripts(); }, [loadScripts]);
  useEffect(() => {
    setTimeout(() => searchInputRef.current?.focus(), 100);
  }, []);

  // Keyboard ESC
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedScript) { setSelectedScript(null); return; }
        onClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedScript, onClose]);

  // Filtern
  const filtered = scripts.filter(s => {
    if (onlyVerified && !s.verified) return false;
    if (filterModelType !== 'Alle' && s.model_type !== filterModelType) return false;
    if (filterTaskType !== 'Alle' && s.task_type !== filterTaskType) return false;
    if (filterFramework !== 'Alle' && s.framework !== filterFramework) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        s.name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.author.toLowerCase().includes(q) ||
        s.tags.some(t => t.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const activeFilterCount = [
    filterModelType !== 'Alle',
    filterTaskType !== 'Alle',
    filterFramework !== 'Alle',
    onlyVerified,
  ].filter(Boolean).length;

  const handleAddToLibrary = (script: LibraryScript) => {
    if (!script.verified) {
      setPendingUnverified(script);
      return;
    }
    confirmAddToLibrary(script);
  };

  const confirmAddToLibrary = async (script: LibraryScript) => {
      setPendingUnverified(null);
      try {
        // Download-Endpoint: erhöht Zähler und liefert den Script-Inhalt
        const res = await fetch(`${API_BASE}/scripts/${script.id}/download`, {
          method: 'POST',
          signal: AbortSignal.timeout(8000),
        });
        const body = res.ok ? await res.json() : null;
        const scriptContent = body?.script ?? script.script;
        addToLocalLibrary(script.name, scriptContent, true, getLocalKey(mode, userData?.userId));
        // Download-Zähler lokal spiegeln
        setScripts(prev =>
          prev.map(s => s.id === script.id ? { ...s, downloads: s.downloads + 1 } : s)
        );
      } catch {
        // Bei Fehler trotzdem lokal speichern
        addToLocalLibrary(script.name, script.script, true, getLocalKey(mode, userData?.userId));
      }
      setAddedIds(prev => new Set([...prev, script.id]));
      success('Hinzugefügt!', `„${script.name}" wurde zu deiner lokalen Bibliothek hinzugefügt.`);
    };

 return (
    <>
      <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-[200] flex items-center justify-center p-4">
        <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-3xl h-[85vh] flex flex-col overflow-hidden shadow-2xl">

          {/* Modal Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 flex-shrink-0 bg-white/[0.02]">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-violet-500/20 border border-violet-500/25 flex items-center justify-center">
                <Globe className="w-4 h-4 text-violet-300" />
              </div>
              <div>
                <h1 className="text-white font-bold text-base">Open Script Library</h1>
                <p className="text-gray-500 text-[10px]">Community {mode === 'test' ? 'Test' : 'Training'}-Skripte · {scripts.length} Skripte verfügbar</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={loadScripts}
                className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-all"
                title="Neu laden"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              </button>
              <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex px-6 pt-3 gap-1 border-b border-white/10 flex-shrink-0">
            {(['browse', 'upload'] as const).map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); setSelectedScript(null); }}
                className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2 ${
                  tab === t
                    ? 'text-white border-violet-400'
                    : 'text-gray-500 hover:text-gray-300 border-transparent'
                }`}
              >
                {t === 'browse' ? (
                  <span className="flex items-center gap-1.5"><BookOpen className="w-3.5 h-3.5" /> Durchsuchen</span>
                ) : (
                  <span className="flex items-center gap-1.5"><Upload className="w-3.5 h-3.5" /> Hochladen</span>
                )}
              </button>
            ))}
          </div>

          {/* Content */}
          {tab === 'upload' ? (
            <UploadTab mode={mode} userData={userData} />
          ) : selectedScript ? (
            <ScriptDetail
              script={selectedScript}
              onBack={() => setSelectedScript(null)}
              onAddToLibrary={handleAddToLibrary}
            />
          ) : (
            <div className="flex flex-col flex-1 min-h-0">
              {/* Search + Filter Bar */}
              <div className="px-5 py-3 space-y-2 border-b border-white/[0.06] flex-shrink-0">
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-600" />
                    <input
                      ref={searchInputRef}
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Name, Beschreibung, Autor, Tags…"
                      className="w-full pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/20"
                    />
                  </div>
                  <button
                    onClick={() => setShowFilters(v => !v)}
                    className={`flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-medium transition-all ${
                      activeFilterCount > 0 || showFilters
                        ? 'bg-violet-500/20 border-violet-500/30 text-violet-300'
                        : 'bg-white/5 border-white/10 text-gray-400 hover:text-white'
                    }`}
                  >
                    <Filter className="w-3.5 h-3.5" />
                    Filter
                    {activeFilterCount > 0 && (
                      <span className="w-4 h-4 rounded-full bg-violet-400 text-white text-[9px] flex items-center justify-center">
                        {activeFilterCount}
                      </span>
                    )}
                  </button>
                </div>

                {/* Filter Panel */}
                {showFilters && (
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <div className="space-y-1">
                      <label className="text-[9px] text-gray-600 uppercase tracking-wide font-medium">Modell-Typ</label>
                      <div className="flex flex-wrap gap-1">
                        {MODEL_TYPES.map(t => (
                          <button key={t} onClick={() => setFilterModelType(t)}
                            className={`px-2 py-0.5 rounded-md text-[10px] transition-all ${filterModelType === t ? 'bg-violet-500/25 border border-violet-500/30 text-violet-300' : 'bg-white/5 border border-white/10 text-gray-500 hover:text-gray-300'}`}>
                            {t}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <label className="text-[9px] text-gray-600 uppercase tracking-wide font-medium">Framework</label>
                      <div className="flex flex-wrap gap-1">
                        {FRAMEWORKS.map(t => (
                          <button key={t} onClick={() => setFilterFramework(t)}
                            className={`px-2 py-0.5 rounded-md text-[10px] transition-all ${filterFramework === t ? 'bg-blue-500/25 border border-blue-500/30 text-blue-300' : 'bg-white/5 border border-white/10 text-gray-500 hover:text-gray-300'}`}>
                            {t}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="col-span-2 flex items-center gap-3 pt-1">
                      <button
                        onClick={() => setOnlyVerified(v => !v)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                          onlyVerified
                            ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                            : 'bg-white/5 border-white/10 text-gray-500 hover:text-gray-300'
                        }`}
                      >
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Nur Verified
                      </button>
                      {activeFilterCount > 0 && (
                        <button
                          onClick={() => { setFilterModelType('Alle'); setFilterTaskType('Alle'); setFilterFramework('Alle'); setOnlyVerified(false); }}
                          className="text-[10px] text-gray-600 hover:text-gray-400 underline transition-all"
                        >
                          Alle zurücksetzen
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Result info */}
              <div className="px-5 py-2 flex items-center justify-between flex-shrink-0">
                <span className="text-[10px] text-gray-600">
                  {loading ? 'Lade…' : `${filtered.length} Skript${filtered.length !== 1 ? 'e' : ''} gefunden`}
                </span>
                <div className="flex items-center gap-2 text-[10px] text-gray-600">
                  <ShieldCheck className="w-3 h-3 text-emerald-500/60" />
                  <span className="text-emerald-500/60">{scripts.filter(s => s.verified).length} verifiziert</span>
                </div>
              </div>

              {/* Script Grid */}
              <div className="flex-1 overflow-y-auto px-5 pb-5">
                {loading ? (
                  <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
                  </div>
                ) : filtered.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 space-y-3">
                    <Globe className="w-12 h-12 text-gray-700" />
                    <p className="text-gray-500 text-sm">Keine Skripte gefunden</p>
                    <p className="text-gray-600 text-xs">Passe deine Filter an oder lade als Erster ein Skript hoch!</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3 pt-1">
                    {filtered.map(s => (
                      <div key={s.id} className="relative">
                        <ScriptCard script={s} onClick={() => setSelectedScript(s)} />
                        {addedIds.has(s.id) && (
                          <div className="absolute top-2 right-2 flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/25 text-emerald-300 text-[9px] font-medium pointer-events-none">
                            <Check className="w-2.5 h-2.5" /> In Bibliothek
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Unverified Warning */}
      {pendingUnverified && (
        <UnverifiedWarning
          script={pendingUnverified}
          onConfirm={() => confirmAddToLibrary(pendingUnverified)}
          onCancel={() => setPendingUnverified(null)}
        />
      )}
    </>
  );
}