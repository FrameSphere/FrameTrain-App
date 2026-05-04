// LaboratoryPanel.tsx – Interaktives Sample-Labor
// Workflow: Datei laden → Samples extrahieren → Einzeln testen → Bewerten → Auswerten

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  FlaskConical, Upload, Play, ChevronRight, ChevronLeft,
  CheckCircle, XCircle, SkipForward, Loader2, AlertCircle,
  BarChart3, X, FileText, Code2, Layers, ChevronDown, ChevronUp,
  Trash2, RotateCcw, Download, Eye, Sparkles, Terminal,
  ThumbsUp, ThumbsDown, Minus, TrendingUp, TrendingDown,
  ClipboardList, Save, FolderOpen, Bot, Send, Pencil,
  Check, Wand2, Copy, Maximize2, Minimize2,
} from 'lucide-react';
import { detectPlugin } from '../plugins/registry';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { callAI } from './TrainingPanel';

// ── Types ─────────────────────────────────────────────────────────────────

interface ModelInfo {
  id: string; name: string; source: string;
  source_path: string | null; local_path: string;
  model_type: string | null; size_bytes?: number;
}

interface VersionTreeItem { id: string; name: string; is_root: boolean; version_number: number; }
interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }

interface LabSample {
  id: string;
  index: number;
  text: string;          // Haupttext für die Inference
  label?: string;        // Erwartetes Label (optional)
  rawData: unknown;      // Original-Daten aus Datei
}

interface TopPred { label: string; score: number; }

interface LabResult {
  sampleId: string;
  sampleIndex: number;
  inputText: string;
  expectedLabel?: string;
  predicted: string;
  confidence?: number;
  topPredictions?: TopPred[];
  inferenceMs: number;
  userRating: 'correct' | 'wrong' | 'skipped';
  userNote: string;
  testedAt: string;
}

interface LabSession {
  id: string;
  name: string;
  modelId: string;
  modelName: string;
  versionId: string;
  versionName: string;
  engineMode: 'engine' | 'dev';
  devScript?: string;
  sourceFileName: string;
  totalSamples: number;
  results: LabResult[];
  createdAt: string;
  updatedAt: string;
}

// ── LocalStorage ──────────────────────────────────────────────────────────

const SESSIONS_KEY = 'ft_lab_sessions';

const loadSessions = (): LabSession[] => {
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) ?? '[]'); } catch { return []; }
};

const saveSession = (s: LabSession) => {
  const all = loadSessions();
  const idx = all.findIndex(x => x.id === s.id);
  if (idx >= 0) all[idx] = s; else all.unshift(s);
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(all.slice(0, 20)));
};

const deleteSession = (id: string) => {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(loadSessions().filter(s => s.id !== id)));
};

// ── Sample Parser ─────────────────────────────────────────────────────────

function extractTextField(obj: unknown): string {
  if (typeof obj === 'string') return obj;
  if (typeof obj !== 'object' || obj === null) return String(obj);
  const o = obj as Record<string, unknown>;
  // Bekannte Text-Keys
  for (const key of ['text', 'input', 'sentence', 'content', 'utterance', 'query', 'sample', 'data', 'value',
                     'abstract', 'body', 'passage', 'document', 'description', 'context', 'premise',
                     'hypothesis', 'review', 'comment', 'message', 'title']) {
    if (typeof o[key] === 'string' && (o[key] as string).length > 0) return o[key] as string;
  }
  // Erster String-Wert als Fallback
  for (const v of Object.values(o)) {
    if (typeof v === 'string' && v.length > 0) return v;
  }
  return JSON.stringify(obj);
}

const TEXT_KEYS  = new Set(['text','input','sentence','content','utterance','query','sample','data','value','abstract','body','passage','document','description','context','premise','hypothesis','review','comment','message','title']);
const LABEL_KEYS = new Set(['label','category','class','target','expected','output','intent']);

function getSideInfo(sample: LabSample): Array<{ key: string; value: string }> {
  const rawObj = (() => {
    if (typeof sample.rawData === 'object' && sample.rawData !== null)
      return sample.rawData as Record<string, unknown>;
    if (typeof sample.rawData === 'string') {
      try { return JSON.parse(sample.rawData) as Record<string, unknown>; } catch { return null; }
    }
    return null;
  })();
  if (!rawObj) return [];
  return Object.entries(rawObj)
    .filter(([k]) => !TEXT_KEYS.has(k) && !LABEL_KEYS.has(k))
    .map(([k, v]) => ({
      key: k,
      value: Array.isArray(v) ? v.join(', ') : String(v),
    }))
    .slice(0, 8);
}

// Immer aus rawData extrahieren (auch fuer bereits geladene Samples)
function getDisplayText(sample: LabSample): string {
  // rawData ist ein Objekt -> direkt extrahieren
  if (typeof sample.rawData === 'object' && sample.rawData !== null) {
    return extractTextField(sample.rawData);
  }
  // rawData ist ein String, der wie JSON aussieht -> parsen, dann extrahieren
  if (typeof sample.rawData === 'string') {
    const trimmed = sample.rawData.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed);
        const extracted = extractTextField(parsed);
        // Nur zurueckgeben wenn es kein JSON-Blob ist
        if (!extracted.trim().startsWith('{') && !extracted.trim().startsWith('[')) {
          return extracted;
        }
      } catch { /* kein gueltiges JSON */ }
    }
    return sample.rawData;
  }
  return sample.text;
}

function extractLabelField(obj: unknown): string | undefined {
  if (typeof obj !== 'object' || obj === null) return undefined;
  const o = obj as Record<string, unknown>;
  for (const key of ['label', 'category', 'class', 'target', 'expected', 'output', 'intent']) {
    if (typeof o[key] === 'string') return o[key] as string;
    if (typeof o[key] === 'number') return String(o[key]);
  }
  return undefined;
}

function parseSamples(content: string, fileName: string): LabSample[] {
  const ext = fileName.split('.').pop()?.toLowerCase() ?? '';
  const raw: unknown[] = [];

  // Auto-Detect: Erkennt JSON/JSONL auch ohne korrekte Erweiterung
  const trimmed = content.trim();
  const looksLikeJsonArray  = trimmed.startsWith('[');
  const looksLikeJsonObject = trimmed.startsWith('{');
  const firstLine = trimmed.split('\n')[0].trim();
  const looksLikeJsonl = firstLine.startsWith('{') || firstLine.startsWith('[');

  const effectiveExt = (() => {
    if (['json', 'jsonl', 'csv', 'tsv', 'txt'].includes(ext)) return ext;
    if (looksLikeJsonArray) return 'json';
    if (looksLikeJsonl && !looksLikeJsonObject) return 'jsonl';
    if (looksLikeJsonObject) return 'jsonl'; // einzelnes Objekt pro Zeile oder ganzes Objekt
    return 'txt';
  })();

  try {
    if (effectiveExt === 'json') {
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed)) raw.push(...parsed);
      else if (typeof parsed === 'object' && parsed !== null) {
        // { samples: [...] } oder { data: [...] } Pattern
        const obj = parsed as Record<string, unknown>;
        const arr = obj['samples'] ?? obj['data'] ?? obj['items'] ?? obj['examples'];
        if (Array.isArray(arr)) raw.push(...arr);
        else raw.push(parsed);
      }
    } else if (effectiveExt === 'jsonl') {
      content.split('\n').filter(l => l.trim()).forEach(l => {
        try { raw.push(JSON.parse(l)); } catch { raw.push(l.trim()); }
      });
    } else if (effectiveExt === 'csv' || effectiveExt === 'tsv') {
      const sep = effectiveExt === 'tsv' ? '\t' : ',';
      const lines = content.split('\n').filter(l => l.trim());
      const headers = lines[0].split(sep).map(h => h.trim().replace(/^"|"$/g, ''));
      for (const line of lines.slice(1)) {
        if (!line.trim()) continue;
        const vals = line.split(sep).map(v => v.trim().replace(/^"|"$/g, ''));
        if (headers.length > 1) {
          const obj: Record<string, string> = {};
          headers.forEach((h, i) => { obj[h] = vals[i] ?? ''; });
          raw.push(obj);
        } else {
          raw.push(vals[0]);
        }
      }
    } else {
      // Plain text: jede nicht-leere Zeile
      content.split('\n').filter(l => l.trim()).forEach(l => raw.push(l.trim()));
    }
  } catch {
    // Fallback: plain text
    content.split('\n').filter(l => l.trim()).forEach(l => raw.push(l.trim()));
  }

  return raw.map((item, i) => ({
    id: `s_${Date.now()}_${i}`,
    index: i,
    text: extractTextField(item),
    label: extractLabelField(item),
    rawData: item,
  }));
}

// ── Confidence Bar ────────────────────────────────────────────────────────

function ConfidenceBar({ value, color = 'amber' }: { value: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  const colorMap: Record<string, string> = {
    amber:   'bg-amber-400',
    emerald: 'bg-emerald-400',
    blue:    'bg-blue-400',
    red:     'bg-red-400',
    violet:  'bg-violet-400',
  };
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className={`h-full rounded-full ${colorMap[color] ?? colorMap.amber} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-400 text-xs font-mono tabular-nums w-10 text-right">{pct.toFixed(1)}%</span>
    </div>
  );
}

// ── Mini SVG Accuracy Donut ───────────────────────────────────────────────

function AccuracyDonut({ correct, wrong, skipped }: { correct: number; wrong: number; skipped: number }) {
  const total = correct + wrong + skipped;
  if (total === 0) return <div className="w-20 h-20 rounded-full bg-white/10 flex items-center justify-center"><span className="text-gray-600 text-xs">–</span></div>;

  const R = 30; const C = 2 * Math.PI * R;
  const correctPct = correct / total;
  const wrongPct   = wrong   / total;

  const correctArc = C * correctPct;
  const wrongArc   = C * wrongPct;
  const skipArc    = C - correctArc - wrongArc;

  let offset = C * 0.25; // Start oben
  const arcs = [
    { arc: correctArc, color: '#10b981', label: 'Korrekt' },
    { arc: wrongArc,   color: '#ef4444', label: 'Falsch' },
    { arc: skipArc,    color: '#374151', label: 'Übersprungen' },
  ];

  return (
    <div className="relative w-20 h-20">
      <svg viewBox="0 0 80 80" className="w-20 h-20 -rotate-90">
        <circle cx="40" cy="40" r={R} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
        {arcs.map((a, i) => {
          const dasharray = `${a.arc} ${C - a.arc}`;
          const dashoffset = offset;
          offset -= a.arc;
          return a.arc > 0 ? (
            <circle key={i} cx="40" cy="40" r={R} fill="none" stroke={a.color} strokeWidth="10"
              strokeDasharray={dasharray} strokeDashoffset={dashoffset} strokeLinecap="butt" />
          ) : null;
        })}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-white font-bold text-sm">{total > 0 ? ((correct / total) * 100).toFixed(0) : 0}%</span>
        <span className="text-gray-500 text-[9px]">Korrekt</span>
      </div>
    </div>
  );
}

// ── Sessions Modal ────────────────────────────────────────────────────────

function SessionsModal({ onLoad, onClose }: { onLoad: (s: LabSession) => void; onClose: () => void }) {
  const [sessions, setSessions] = useState<LabSession[]>([]);
  const { success } = useNotification();

  useEffect(() => { setSessions(loadSessions()); }, []);

  const handleDelete = (id: string) => {
    deleteSession(id);
    setSessions(loadSessions());
    success('Session gelöscht', '');
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FlaskConical className="w-5 h-5 text-pink-400" /><h2 className="text-lg font-bold text-white">Lab-Sessions</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {sessions.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FlaskConical className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">Noch keine Sessions gespeichert.</p>
            </div>
          ) : sessions.map(s => {
            const correct = s.results.filter(r => r.userRating === 'correct').length;
            const wrong   = s.results.filter(r => r.userRating === 'wrong').length;
            const total   = s.results.length;
            return (
              <div key={s.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium text-sm truncate">{s.name}</p>
                    <p className="text-gray-500 text-xs">{s.modelName} · {s.versionName}</p>
                    <div className="flex items-center gap-3 mt-1.5">
                      <span className="text-[10px] text-gray-500">{total}/{s.totalSamples} bewertet</span>
                      {total > 0 && <><span className="text-[10px] text-emerald-400">✅ {correct}</span><span className="text-[10px] text-red-400">❌ {wrong}</span></>}
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-md border ${s.engineMode === 'engine' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20' : 'bg-blue-500/15 text-blue-400 border-blue-500/20'}`}>
                        {s.engineMode === 'engine' ? 'Engine' : 'Dev Script'}
                      </span>
                    </div>
                    <p className="text-gray-600 text-[10px] mt-1">{new Date(s.updatedAt).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit' })}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                    <button onClick={() => handleDelete(s.id)} className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>
                    <button onClick={() => { onLoad(s); onClose(); }} className="px-3 py-1.5 rounded-xl bg-pink-500/20 hover:bg-pink-500/30 border border-pink-500/30 text-pink-300 text-xs font-medium transition-all">Laden</button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Dev Script Editor (Mini) ──────────────────────────────────────────────

function DevScriptEditor({ script, onChange, modelPath, datasets, outputPath }: {
  script: string; onChange: (s: string) => void;
  modelPath: string; datasets: { key: string; value: string; name: string }[];
  outputPath: string;
}) {
  const { settings: aiSettings } = useAISettings();
  const [showAI, setShowAI] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [aiInput, setAiInput] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [aiMessages, setAiMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const aiEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { aiEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [aiMessages]);

  const generateTemplate = () => onChange(`#!/usr/bin/env python3
# FrameTrain – Lab Dev Script
# Das Skript bekommt ein einzelnes Sample via ENV-Variable
# und soll das Ergebnis als JSON-Zeile auf stdout ausgeben.
#
# Pflichtfeld:  {"predicted": "label"}
# Optional:     {"predicted": "label", "confidence": 0.95, "top_predictions": [{"label": "...", "score": 0.95}]}

import os
import json

# ── Pfade (von FrameTrain gesetzt) ────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${modelPath}")
${datasets.map(r => `${r.key}   = os.environ.get("${r.key}", "${r.value}")`).join('\n')}
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${outputPath}")

# ── Sample (wird für jedes Sample neu gesetzt) ────────────────────────────
SAMPLE_INPUT = os.environ.get("LAB_SAMPLE_INPUT", "")

# ── Modell laden (einmalig pro Skript-Start) ──────────────────────────────
# TODO: Lade dein Modell hier
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# ── Inference ─────────────────────────────────────────────────────────────
# TODO: Führe Inference durch
# inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", truncation=True, padding=True)
# with torch.no_grad():
#     outputs = model(**inputs)
# pred_idx = outputs.logits.argmax(-1).item()
# label = model.config.id2label[pred_idx]
# confidence = outputs.logits.softmax(-1).max().item()

# ── Ergebnis ausgeben (PFLICHT: JSON auf stdout) ──────────────────────────
result = {
    "predicted": "TODO_LABEL",       # Vorhergesagtes Label
    # "confidence": 0.95,            # Optional: Konfidenz 0–1
    # "top_predictions": [           # Optional: Alle Labels mit Score
    #     {"label": "TODO", "score": 0.95},
    # ],
}
print(json.dumps(result))
`);

  const askAI = async () => {
    if (!aiInput.trim() || aiLoading) return;
    const userMsg = { role: 'user' as const, content: aiInput.trim() };
    setAiMessages(m => [...m, userMsg]); setAiInput(''); setAiLoading(true);
    try {
      const sys = `Du bist ein Code-Assistent für FrameTrain Lab Dev Scripts.
Das Skript bekommt SAMPLE_INPUT via ENV und soll {"predicted": "...", "confidence": 0.9} auf stdout ausgeben.
MODEL_PATH="${modelPath}", OUTPUT_PATH="${outputPath}".
Antworte auf Deutsch. Code in \`\`\`python Blöcken.`;
      const history = [...aiMessages, userMsg].map(m => ({ role: m.role, content: m.content }));
      const last = history.pop()!;
      const resp = await callAI(aiSettings, sys, last.content, history);
      setAiMessages(m => [...m, { role: 'assistant', content: resp }]);
      // Code-Block als Skript übernehmen?
      const match = resp.match(/```python\n([\s\S]*?)```/);
      if (match) onChange(match[1]);
    } catch (e) {
      setAiMessages(m => [...m, { role: 'assistant', content: `Fehler: ${String(e)}` }]);
    } finally { setAiLoading(false); }
  };

  const h = expanded ? 360 : 220;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-white">Dev Script</span>
        <div className="flex items-center gap-2">
          <button onClick={() => onChange('')} className="text-[10px] text-gray-500 hover:text-red-400 transition-colors">Leeren</button>
          <button onClick={generateTemplate} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-medium hover:bg-blue-500/20 transition-all">
            <Sparkles className="w-3 h-3" /> Template
          </button>
          {aiSettings.enabled && (
            <button onClick={() => setShowAI(v => !v)} className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium border transition-all ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : 'bg-white/5 text-gray-400 border-white/10 hover:text-white'}`}>
              <Bot className="w-3 h-3" /> KI
            </button>
          )}
          <button onClick={() => setExpanded(v => !v)} className="p-1 rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white transition-all">
            {expanded ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
          </button>
        </div>
      </div>

      <div className="flex gap-3" style={{ height: `${h}px` }}>
        <textarea
          value={script}
          onChange={e => onChange(e.target.value)}
          spellCheck={false}
          placeholder={"# Skript hier eingeben oder Template laden…\n# Tipp: Template-Button klicken!"}
          className="flex-1 p-4 bg-slate-950 border border-white/10 rounded-xl text-[11px] font-mono text-gray-200 focus:outline-none focus:border-blue-500/40 resize-none placeholder:text-gray-700 leading-[1.6rem]"
          style={{ fontFamily: "'JetBrains Mono','Fira Code','Courier New',monospace" }}
        />

        {showAI && (
          <div className="w-72 flex flex-col bg-slate-950 border border-white/10 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
              <div className="flex items-center gap-1.5"><Wand2 className="w-3.5 h-3.5 text-violet-400" /><span className="text-xs font-medium text-white">KI-Assistent</span></div>
              <button onClick={() => setShowAI(false)} className="p-1 rounded hover:bg-white/5 text-gray-500"><X className="w-3 h-3" /></button>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {aiMessages.length === 0 && <p className="text-gray-600 text-[10px] text-center py-4">Frag nach dem Skript…</p>}
              {aiMessages.map((m, i) => (
                <div key={i} className={`px-2.5 py-2 rounded-lg text-[10px] leading-relaxed ${m.role === 'user' ? 'bg-violet-500/10 text-gray-200 border border-violet-500/15 ml-4' : 'bg-white/5 text-gray-300 border border-white/10 mr-4'}`}>
                  {m.content.replace(/```python[\s\S]*?```/g, '[Code wurde übernommen ✅]').trim()}
                </div>
              ))}
              {aiLoading && <div className="px-2.5 py-2 rounded-lg bg-white/5 border border-white/10 mr-4"><Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin" /></div>}
              <div ref={aiEndRef} />
            </div>
            <div className="p-2 border-t border-white/10 flex gap-1.5 flex-shrink-0">
              <input value={aiInput} onChange={e => setAiInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && askAI()}
                placeholder="Frage…" className="flex-1 px-2.5 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-[10px] placeholder:text-gray-600 focus:outline-none" />
              <button onClick={askAI} disabled={!aiInput.trim() || aiLoading} className="p-1.5 rounded-lg bg-violet-500/20 border border-violet-500/30 text-violet-300 disabled:opacity-40 transition-all">
                <Send className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </div>
      <p className="text-[10px] text-gray-600">Das Skript läuft für jedes Sample einzeln. Ausgabe muss ein JSON-Objekt mit <code className="text-gray-500">{"\"predicted\""}</code> sein.</p>
    </div>
  );
}

// ── Analysis View ─────────────────────────────────────────────────────────

function AnalysisView({ session, onBack }: { session: LabSession; onBack: () => void }) {
  const [filterRating, setFilterRating] = useState<'all' | 'correct' | 'wrong' | 'skipped'>('all');
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const results = session.results;
  const correct = results.filter(r => r.userRating === 'correct');
  const wrong   = results.filter(r => r.userRating === 'wrong');
  const skipped = results.filter(r => r.userRating === 'skipped');
  const rated   = correct.length + wrong.length;
  const accuracy = rated > 0 ? correct.length / rated : 0;

  // Häufigste Falsch-Predictions
  const wrongPredCounts: Record<string, number> = {};
  wrong.forEach(r => { wrongPredCounts[r.predicted] = (wrongPredCounts[r.predicted] ?? 0) + 1; });
  const topWrongPreds = Object.entries(wrongPredCounts).sort((a, b) => b[1] - a[1]).slice(0, 5);

  // Avg Confidence per rating
  const avgConf = (arr: LabResult[]) => {
    const vals = arr.filter(r => r.confidence != null).map(r => r.confidence!);
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const filtered = filterRating === 'all' ? results
    : results.filter(r => r.userRating === filterRating);

  const exportCSV = () => {
    const rows = [
      ['Index', 'Input', 'Erwartet', 'Vorhergesagt', 'Konfidenz', 'Bewertung', 'Notiz'],
      ...results.map(r => [
        r.sampleIndex + 1, `"${r.inputText.replace(/"/g, '""')}"`,
        r.expectedLabel ?? '', r.predicted,
        r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '',
        r.userRating, `"${r.userNote.replace(/"/g, '""')}"`,
      ]),
    ];
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `lab_${session.name}_results.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white border border-white/10 transition-all"><ChevronLeft className="w-4 h-4" /></button>
          <div><h2 className="text-lg font-bold text-white">{session.name}</h2><p className="text-gray-500 text-xs">{session.modelName} · {session.versionName} · {session.engineMode === 'engine' ? 'Test Engine' : 'Dev Script'}</p></div>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-sm transition-all">
          <Download className="w-4 h-4" /> Export CSV
        </button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Bewertet', value: `${rated}/${session.totalSamples}`, color: 'text-white' },
          { label: 'Korrekt', value: correct.length, color: 'text-emerald-400' },
          { label: 'Falsch', value: wrong.length, color: 'text-red-400' },
          { label: 'Übersprungen', value: skipped.length, color: 'text-gray-400' },
        ].map(m => (
          <div key={m.label} className="rounded-xl border border-white/10 bg-white/5 p-4 text-center">
            <p className="text-gray-500 text-xs mb-1">{m.label}</p>
            <p className={`text-2xl font-bold ${m.color}`}>{m.value}</p>
          </div>
        ))}
      </div>

      {/* Accuracy + Donut */}
      <div className="grid grid-cols-3 gap-4">
        {/* Donut */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 flex flex-col items-center justify-center gap-3">
          <AccuracyDonut correct={correct.length} wrong={wrong.length} skipped={skipped.length} />
          <div className="space-y-1 w-full">
            {[
              { color: 'bg-emerald-400', label: 'Korrekt', n: correct.length },
              { color: 'bg-red-400',     label: 'Falsch',  n: wrong.length },
              { color: 'bg-gray-600',    label: 'Skip',    n: skipped.length },
            ].map(x => (
              <div key={x.label} className="flex items-center gap-2 text-xs">
                <div className={`w-2 h-2 rounded-full ${x.color}`} />
                <span className="text-gray-400 flex-1">{x.label}</span>
                <span className="text-gray-300 tabular-nums">{x.n}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Konfidenz Stats */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
          <p className="text-sm font-medium text-white">Ø Konfidenz</p>
          {[
            { label: 'Korrekte', val: avgConf(correct), color: 'emerald' },
            { label: 'Falsche',  val: avgConf(wrong),   color: 'red' },
          ].map(x => (
            <div key={x.label} className="space-y-1">
              <span className="text-xs text-gray-400">{x.label}</span>
              {x.val != null ? <ConfidenceBar value={x.val} color={x.color} /> : <span className="text-gray-600 text-xs">–</span>}
            </div>
          ))}
          <div className="pt-1 border-t border-white/10">
            <p className="text-xs text-gray-500">Accuracy (bewertet)</p>
            <p className={`text-xl font-bold mt-0.5 ${accuracy > 0.8 ? 'text-emerald-400' : accuracy > 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
              {rated > 0 ? `${(accuracy * 100).toFixed(1)}%` : '–'}
            </p>
          </div>
        </div>

        {/* Häufigste Falsch-Predictions */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
          <p className="text-sm font-medium text-white">Häufigste Fehlklassen</p>
          {topWrongPreds.length === 0
            ? <p className="text-gray-600 text-xs">Keine Fehler vorhanden</p>
            : topWrongPreds.map(([label, count]) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-gray-400 text-xs flex-1 truncate">{label}</span>
                <span className="text-red-400 text-xs font-semibold tabular-nums">{count}×</span>
              </div>
            ))}
        </div>
      </div>

      {/* Filter + Table */}
      <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-white/10">
          <p className="text-white font-medium text-sm">Ergebnisse ({filtered.length})</p>
          <div className="flex items-center gap-1 p-1 rounded-lg bg-white/5 border border-white/10">
            {([['all', 'Alle'], ['correct', '✅'], ['wrong', '❌'], ['skipped', '⏭']] as const).map(([val, label]) => (
              <button key={val} onClick={() => setFilterRating(val as typeof filterRating)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${filterRating === val ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>{label}</button>
            ))}
          </div>
        </div>
        <div className="divide-y divide-white/5 max-h-96 overflow-y-auto">
          {filtered.length === 0
            ? <p className="text-gray-600 text-sm text-center py-8">Keine Einträge.</p>
            : filtered.map((r, i) => (
              <div key={r.sampleId}>
                <button onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  className="w-full flex items-center gap-3 px-5 py-3 hover:bg-white/[0.03] transition-all text-left">
                  <span className="text-gray-600 text-xs tabular-nums w-6 flex-shrink-0">{r.sampleIndex + 1}</span>
                  <span className={`flex-shrink-0 text-base ${r.userRating === 'correct' ? '✅' : r.userRating === 'wrong' ? '❌' : '⏭'}`}>
                    {r.userRating === 'correct' ? '✅' : r.userRating === 'wrong' ? '❌' : '⏭'}
                  </span>
                  <span className="text-gray-300 text-xs flex-1 truncate">{r.inputText}</span>
                  <span className="text-white text-xs font-medium flex-shrink-0">{r.predicted}</span>
                  {r.confidence != null && <span className="text-gray-500 text-xs font-mono tabular-nums flex-shrink-0">{(r.confidence * 100).toFixed(0)}%</span>}
                  {expandedIdx === i ? <ChevronUp className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" /> : <ChevronDown className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />}
                </button>
                {expandedIdx === i && (
                  <div className="px-5 pb-4 space-y-2 bg-white/[0.02] border-t border-white/5">
                    <div className="grid grid-cols-2 gap-3 pt-3 text-xs">
                      <div><span className="text-gray-500">Input:</span><p className="text-gray-300 mt-0.5">{r.inputText}</p></div>
                      <div><span className="text-gray-500">Vorhergesagt:</span><p className="text-white font-semibold mt-0.5">{r.predicted}</p></div>
                      {r.expectedLabel && <div><span className="text-gray-500">Erwartet:</span><p className="text-gray-300 mt-0.5">{r.expectedLabel}</p></div>}
                      {r.userNote && <div className="col-span-2"><span className="text-gray-500">Notiz:</span><p className="text-gray-400 italic mt-0.5">{r.userNote}</p></div>}
                    </div>
                    {r.topPredictions && r.topPredictions.length > 0 && (
                      <div className="space-y-1 pt-1">
                        {r.topPredictions.slice(0, 5).map(p => (
                          <div key={p.label} className="flex items-center gap-2 text-xs">
                            <span className="text-gray-400 w-32 truncate">{p.label}</span>
                            <ConfidenceBar value={p.score} color={p.label === r.predicted ? 'amber' : 'blue'} />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

type LabPhase = 'setup' | 'testing' | 'analysis';

export default function LaboratoryPanel() {
  const { success, error, warning } = useNotification();

  // Models
  const [loadingModels, setLoadingModels] = useState(true);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<{ id: string; name: string; model_id: string; status: string; file_count: number; size_bytes: number; storage_path?: string }[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);

  // Engine
  const [engineMode, setEngineMode] = useState<'engine' | 'dev'>('engine');
  const [devScript, setDevScript] = useState('');

  // Model Server Status
  const [serverStatus, setServerStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [serverVersionId, setServerVersionId] = useState<string | null>(null);
  const serverStatusRef = useRef<'idle' | 'loading' | 'ready' | 'error'>('idle');

  // Samples
  const [samples, setSamples] = useState<LabSample[]>([]);
  const [sourceFileName, setSourceFileName] = useState('');
  const [selectedSampleDatasetId, setSelectedSampleDatasetId] = useState<string | null>(null);
  const [selectedSampleSplit, setSelectedSampleSplit] = useState<'all' | 'train' | 'val' | 'test'>('all');
  const [loadingDatasetSamples, setLoadingDatasetSamples] = useState(false);
  const [currentSampleIdx, setCurrentSampleIdx] = useState(0);

  // Testing state
  const [phase, setPhase] = useState<LabPhase>('setup');
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ predicted: string; confidence?: number; topPredictions?: TopPred[]; inferenceMs: number } | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [userNote, setUserNote] = useState('');
  const [showNote, setShowNote] = useState(false);

  // Session
  const [session, setSession] = useState<LabSession | null>(null);
  const [showSessions, setShowSessions] = useState(false);

  // UI
  const [setupCollapsed, setSetupCollapsed] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);


  const unlistenRef = useRef<(() => void)[]>([]);

  useEffect(() => () => { unlistenRef.current.forEach(fn => fn()); }, []);

  // ── Derived (muss VOR allen useEffects stehen – TDZ-Vermeidung) ──────────

  const selectedModel      = models.find(m => m.id === selectedModelId);
  const selectedModelTree  = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersionTree = selectedModelTree?.versions.find(v => v.id === selectedVersionId);

  const detectedPlugin = useMemo(() => {
    if (!selectedModel) return null;
    const r = detectPlugin(selectedModel.source_path ?? selectedModel.name);
    return r.supported ? r.plugin : null;
  }, [selectedModel]);

  const modelPath  = selectedModel?.local_path || selectedModel?.source_path || selectedModel?.name || '';
  const dsRefs     = datasets.map((d, i) => ({ key: i === 0 ? 'DATASET_PATH' : `DATASET_PATH_${i + 1}`, value: d.storage_path || '', name: d.name }));
  const outputPath = `[AppData]/lab_outputs`;

  // ── Load Models ─────────────────────────────────────────────────────────

  useEffect(() => {
    (async () => {
      setLoadingModels(true);
      try {
        const [list, listWithVersions] = await Promise.all([
          invoke<ModelInfo[]>('list_models'),
          invoke<ModelWithVersionTree[]>('list_models_with_version_tree'),
        ]);
        setModels(list);
        setModelsWithVersions(listWithVersions);
        if (listWithVersions.length > 0) setSelectedModelId(listWithVersions[0].id);
      } catch (e) { console.error('[Lab] initLoad:', e); }
      finally { setLoadingModels(false); }
    })();
  }, []);

  useEffect(() => {
    if (!selectedModelId) { setDatasets([]); setSelectedSampleDatasetId(null); return; }
    invoke<typeof datasets>('list_datasets_for_model', { modelId: selectedModelId })
      .then(list => { setDatasets(list); setSelectedSampleDatasetId(list[0]?.id ?? null); setSamples([]); setSourceFileName(''); })
      .catch(() => { setDatasets([]); setSelectedSampleDatasetId(null); });
  }, [selectedModelId]);

  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); return; }
    const m = modelsWithVersions.find(x => x.id === selectedModelId);
    if (!m?.versions.length) { setSelectedVersionId(null); return; }
    setSelectedVersionId([...m.versions].sort((a, b) => b.version_number - a.version_number)[0].id);
  }, [selectedModelId, modelsWithVersions]);

  // ── Model Server: starten wenn Version wechselt ─────────────────────────

  useEffect(() => {
    if (!selectedVersionId || engineMode !== 'engine' || !detectedPlugin) return;
    if (serverVersionId === selectedVersionId) return; // schon geladen

    setServerStatus('loading');
    serverStatusRef.current = 'loading';
    invoke('lab_start_model_server', { versionId: selectedVersionId }).catch(e => {
      console.error('[Lab] lab_start_model_server:', e);
    });
  }, [selectedVersionId, engineMode, detectedPlugin]);

  useEffect(() => {
    const unlisten = listen<{ status: string; version_id?: string; message?: string }>(
      'lab-server-status',
      e => {
        const { status, version_id, message } = e.payload;
        console.log('[Lab] Server-Status:', status, version_id, message);
        setServerStatus(status as typeof serverStatus);
        serverStatusRef.current = status as typeof serverStatus;
        if (status === 'ready' && version_id) setServerVersionId(version_id);
        if (status === 'error') setServerVersionId(null);
        if (status === 'error' && message) error('Modell-Ladefehler', message);
      }
    );
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Cleanup: Server stoppen wenn Komponente unmountet
  useEffect(() => {
    return () => {
      invoke('lab_stop_model_server').catch(() => {});
    };
  }, []);

  const currentSample = samples[currentSampleIdx] ?? null;
  const results       = session?.results ?? [];
  const testedCount   = results.filter(r => r.userRating !== 'skipped').length;

  // ── Dataset Sample Import ───────────────────────────────────────────────────────

  const handleLoadFromDataset = async () => {
    if (!selectedSampleDatasetId) return;
    setLoadingDatasetSamples(true);
    try {
      const files = await invoke<{ name: string; path: string; size: number; is_dir: boolean; split: string }[]>(
        'get_dataset_files', { datasetId: selectedSampleDatasetId }
      );
      console.log('[Lab] Dataset-Dateien:', files);

      const filtered = files.filter(f => !f.is_dir && (selectedSampleSplit === 'all' || f.split === selectedSampleSplit));
      console.log('[Lab] Gefilterte Dateien:', filtered);

      if (filtered.length === 0) {
        warning('Keine Dateien', `Keine Dateien für Split "${selectedSampleSplit}" gefunden. Verfügbare Splits: ${[...new Set(files.filter(f => !f.is_dir).map(f => f.split))].join(', ') || 'keine'}.`);
        return;
      }

      const allSamples: LabSample[] = [];
      for (const file of filtered) {
        try {
          const content = await invoke<string>('read_dataset_file', { filePath: file.path });
          console.log(`[Lab] Datei gelesen: ${file.name}, Länge: ${content.length}`);
          const parsed = parseSamples(content, file.name);
          console.log(`[Lab] Samples aus ${file.name}:`, parsed.length);
          allSamples.push(...parsed);
        } catch (fileErr) {
          console.warn(`[Lab] Fehler beim Lesen von ${file.name}:`, fileErr);
        }
      }

      if (allSamples.length === 0) {
        warning('Keine Samples', `${filtered.length} Datei(en) geladen, aber keine Samples extrahiert. Prüfe das Format (JSONL, CSV, TXT)`);
        return;
      }

      const reindexed = allSamples.map((s, i) => ({ ...s, id: `s_${Date.now()}_${i}`, index: i }));
      setSamples(reindexed);
      const ds = datasets.find(d => d.id === selectedSampleDatasetId);
      setSourceFileName(ds?.name ?? 'Dataset');
      success('Geladen', `${reindexed.length} Samples aus ${filtered.length} Datei(en) geladen.`);
    } catch (e) {
      console.error('[Lab] handleLoadFromDataset Fehler:', e);
      error('Fehler beim Laden', String(e));
    } finally {
      setLoadingDatasetSamples(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const content = ev.target?.result as string;
      const parsed = parseSamples(content, file.name);
      if (parsed.length === 0) { warning('Leer', 'Keine Samples gefunden.'); return; }
      setSamples(parsed);
      setSourceFileName(file.name);
      success('Geladen', `${parsed.length} Samples aus „${file.name}" geladen.`);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // ── Start Session ────────────────────────────────────────────────────────

  const handleStartSession = () => {
    if (!selectedModel || !selectedVersionId || samples.length === 0) return;
    if (engineMode === 'engine' && !detectedPlugin) { warning('Nicht unterstützt', 'Dieses Modell wird von der Test Engine noch nicht unterstützt. Nutze den Dev Script Modus.'); return; }
    if (engineMode === 'dev' && !devScript.trim()) { warning('Kein Skript', 'Bitte ein Dev Script eingeben.'); return; }

    const newSession: LabSession = {
      id: `lab_${Date.now()}`,
      name: `${selectedModel.name} – ${sourceFileName} (${new Date().toLocaleDateString('de-DE')})`,
      modelId: selectedModel.id,
      modelName: selectedModel.name,
      versionId: selectedVersionId,
      versionName: selectedVersionTree?.name ?? 'v?',
      engineMode,
      devScript: engineMode === 'dev' ? devScript : undefined,
      sourceFileName,
      totalSamples: samples.length,
      results: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setSession(newSession);
    setCurrentSampleIdx(0);
    setTestResult(null);
    setTestError(null);
    setUserNote('');
    setShowNote(false);
    setPhase('testing');
    setSetupCollapsed(true);
  };

  // ── Load Session ─────────────────────────────────────────────────────────

  const handleLoadSession = (s: LabSession) => {
    setSession(s);
    setPhase('analysis');
  };

  // ── Run Test ─────────────────────────────────────────────────────────────

  const handleRunTest = useCallback(async () => {
    if (!currentSample || !selectedVersionId || testing) return;
    setTesting(true);
    setTestResult(null);
    setTestError(null);
    unlistenRef.current.forEach(fn => fn());
    unlistenRef.current = [];

    const start = Date.now();

    try {
      if (engineMode === 'engine') {
        // ── Persistenter Model-Server: direkt invoke, kein Event-Listener nötig ──
        if (serverStatus !== 'ready') {
          setTestError(
            serverStatus === 'loading'
              ? 'Modell wird noch geladen… Bitte kurz warten.'
              : 'Kein Modell geladen. Bitte ein Modell auswählen.'
          );
          setTesting(false);
          return;
        }

        const result = await invoke<{
          predicted: string;
          confidence?: number;
          top_predictions?: TopPred[];
          inference_ms: number;
        }>('lab_infer_sample', { text: currentSample.text });

        setTestResult({
          predicted:      result.predicted,
          confidence:     result.confidence,
          topPredictions: result.top_predictions,
          inferenceMs:    result.inference_ms,
        });
        setTesting(false);
      } else {
        // Dev Script Mode
        const refs: Record<string, string> = {
          MODEL_PATH: modelPath,
          ...Object.fromEntries(dsRefs.map(r => [r.key, r.value])),
          LAB_SAMPLE_INPUT: currentSample.text,
        };

        const u1 = await listen<{ predicted?: string; confidence?: number; top_predictions?: TopPred[]; error?: string }>('lab-script-result', e => {
          if (e.payload.error) {
            setTestError(e.payload.error);
          } else {
            setTestResult({
              predicted: e.payload.predicted ?? '?',
              confidence: e.payload.confidence,
              topPredictions: e.payload.top_predictions,
              inferenceMs: Date.now() - start,
            });
          }
          setTesting(false);
        });
        unlistenRef.current = [u1];

        await invoke('run_lab_script_sample', {
          script: devScript,
          sampleInput: currentSample.text,
          refs,
        });
      }
    } catch (e: unknown) {
      setTestError(String(e));
      setTesting(false);
    }
  }, [currentSample, selectedVersionId, engineMode, devScript, modelPath, dsRefs, testing]);

  // ── Rate Sample ──────────────────────────────────────────────────────────

  const handleRate = useCallback((rating: 'correct' | 'wrong' | 'skipped') => {
    if (!session || !currentSample || !testResult) return;

    const result: LabResult = {
      sampleId: currentSample.id,
      sampleIndex: currentSample.index,
      inputText: currentSample.text,
      expectedLabel: currentSample.label,
      predicted: testResult.predicted,
      confidence: testResult.confidence,
      topPredictions: testResult.topPredictions,
      inferenceMs: testResult.inferenceMs,
      userRating: rating,
      userNote: userNote.trim(),
      testedAt: new Date().toISOString(),
    };

    const updatedSession: LabSession = {
      ...session,
      results: [...session.results.filter(r => r.sampleId !== currentSample.id), result],
      updatedAt: new Date().toISOString(),
    };
    setSession(updatedSession);
    saveSession(updatedSession);

    // Nächstes Sample
    const nextIdx = currentSampleIdx + 1;
    if (nextIdx < samples.length) {
      setCurrentSampleIdx(nextIdx);
      setTestResult(null);
      setTestError(null);
      setUserNote('');
      setShowNote(false);
    } else {
      // Session abgeschlossen
      setPhase('analysis');
    }
  }, [session, currentSample, testResult, userNote, currentSampleIdx, samples]);

  const handleSkipWithoutTest = () => {
    if (!session || !currentSample) return;
    const result: LabResult = {
      sampleId: currentSample.id,
      sampleIndex: currentSample.index,
      inputText: currentSample.text,
      expectedLabel: currentSample.label,
      predicted: '–',
      inferenceMs: 0,
      userRating: 'skipped',
      userNote: '',
      testedAt: new Date().toISOString(),
    };
    const updatedSession: LabSession = {
      ...session,
      results: [...session.results.filter(r => r.sampleId !== currentSample.id), result],
      updatedAt: new Date().toISOString(),
    };
    setSession(updatedSession);
    saveSession(updatedSession);
    const nextIdx = currentSampleIdx + 1;
    if (nextIdx < samples.length) {
      setCurrentSampleIdx(nextIdx);
      setTestResult(null);
      setTestError(null);
      setUserNote('');
    } else {
      setPhase('analysis');
    }
  };

  const alreadyRated = session?.results.find(r => r.sampleId === currentSample?.id);

  // ── Render ────────────────────────────────────────────────────────────────

  if (loadingModels) {
    return <div className="flex items-center justify-center py-24"><Loader2 className="w-8 h-8 text-gray-500 animate-spin" /></div>;
  }

  return (
    <div className="space-y-6">

      {/* ── Page Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Labor</h1>
          <p className="text-gray-400 mt-1">Samples testen, bewerten und Schwachstellen analysieren</p>
        </div>
        <div className="flex items-center gap-2">
          {phase === 'testing' && session && (
            <button onClick={() => setPhase('analysis')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all">
              <BarChart3 className="w-4 h-4" /> Auswertung
            </button>
          )}
          {phase === 'analysis' && session && (
            <button onClick={() => setPhase('testing')} disabled={currentSampleIdx >= samples.length} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all disabled:opacity-40">
              <Play className="w-4 h-4" /> Weiter testen
            </button>
          )}
          <button onClick={() => setShowSessions(true)} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-500/10 hover:bg-pink-500/20 border border-pink-500/20 text-pink-300 text-sm transition-all">
            <FolderOpen className="w-4 h-4" /> Sessions
          </button>
        </div>
      </div>

      {/* ── Analysis View (wenn phase === analysis) ── */}
      {phase === 'analysis' && session && (
        <AnalysisView session={session} onBack={() => setPhase(samples.length > 0 ? 'testing' : 'setup')} />
      )}

      {/* ── Setup + Testing ── */}
      {(phase === 'setup' || phase === 'testing') && (
        <>
          {/* ── Setup Card ── */}
          <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
            <button onClick={() => phase === 'testing' && setSetupCollapsed(v => !v)}
              className={`w-full flex items-center justify-between p-5 ${phase === 'testing' ? 'hover:bg-white/[0.03] cursor-pointer' : ''} transition-all`}>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl bg-pink-500/20 border border-pink-500/30 flex items-center justify-center"><FlaskConical className="w-4 h-4 text-pink-400" /></div>
                <div className="text-left">
                  <span className="text-white font-semibold text-sm">Konfiguration</span>
                  {phase === 'testing' && selectedModel && (
                    <p className="text-gray-500 text-xs">{selectedModel.name} · {engineMode === 'engine' ? 'Test Engine' : 'Dev Script'} · {samples.length} Samples aus „{sourceFileName}"</p>
                  )}
                </div>
              </div>
              {phase === 'testing' && (setupCollapsed ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronUp className="w-4 h-4 text-gray-400" />)}
            </button>

            {!setupCollapsed && (
              <div className="px-5 pb-6 space-y-5 border-t border-white/10 pt-5">

                {/* No Models Warning */}
                {models.length === 0 ? (
                  <div className="rounded-xl border border-white/10 bg-white/5 p-8 text-center space-y-2">
                    <Layers className="w-8 h-8 text-gray-600 mx-auto" />
                    <p className="text-white font-medium">Kein Modell vorhanden</p>
                    <p className="text-gray-500 text-sm">Füge zuerst ein Modell im Model-Manager hinzu.</p>
                  </div>
                ) : (
                  <>
                    {/* Model + Version */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-white">Modell</label>
                        <select value={selectedModelId ?? ''} onChange={e => setSelectedModelId(e.target.value)}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                          {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-900">{m.name}</option>)}
                        </select>
                      </div>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-white">Version</label>
                        <select value={selectedVersionId ?? ''} onChange={e => setSelectedVersionId(e.target.value)}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                          {selectedModelTree?.versions?.length
                            ? [...selectedModelTree.versions].sort((a, b) => b.version_number - a.version_number).map((v, i) => (
                                <option key={v.id} value={v.id} className="bg-slate-900">{v.name}{i === 0 ? ' (neueste)' : ''}</option>
                              ))
                            : <option value="">Keine Versionen</option>}
                        </select>
                      </div>
                    </div>

                    {/* Engine Toggle */}
                    <div className="space-y-3">
                      <label className="block text-sm font-medium text-white">Test Engine</label>
                      <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
                        {([['engine', 'Train Engine', Play, 'amber'], ['dev', 'Dev Script', Code2, 'blue']] as const).map(([val, label, Icon, col]) => (
                          <button key={val} onClick={() => setEngineMode(val as typeof engineMode)}
                            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-medium transition-all ${engineMode === val ? (col === 'amber' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-blue-500/20 text-blue-300 border border-blue-500/30') : 'text-gray-400 hover:text-white'}`}>
                            <Icon className="w-3.5 h-3.5" />{label}
                          </button>
                        ))}
                      </div>

                      {engineMode === 'engine' && selectedModel && !detectedPlugin && (
                        <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                          <span className="text-amber-300 text-xs">Dieses Modell wird noch nicht von der Test Engine unterstützt. Nutze den Dev Script Modus.</span>
                        </div>
                      )}
                      {engineMode === 'engine' && detectedPlugin && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
                            <CheckCircle className="w-3.5 h-3.5 text-amber-400" />
                            <span className="text-amber-300 text-xs">{detectedPlugin.name} – Plugin erkannt</span>
                          </div>
                          {/* Server-Status Badge */}
                          {serverStatus === 'loading' && (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-blue-500/10 border border-blue-500/20">
                              <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin flex-shrink-0" />
                              <span className="text-blue-300 text-xs">Modell wird geladen… Bitte warten.</span>
                            </div>
                          )}
                          {serverStatus === 'ready' && serverVersionId === selectedVersionId && (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                              <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                              <span className="text-emerald-300 text-xs">Modell geladen – bereit für Inferenz</span>
                            </div>
                          )}
                          {serverStatus === 'error' && (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20">
                              <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                              <span className="text-red-300 text-xs">Fehler beim Laden – Version wechseln oder neu starten</span>
                            </div>
                          )}
                        </div>
                      )}

                      {engineMode === 'dev' && (
                        <DevScriptEditor
                          script={devScript} onChange={setDevScript}
                          modelPath={modelPath} datasets={dsRefs} outputPath={outputPath}
                        />
                      )}
                    </div>

                    {/* Dataset Sample-Auswahl */}
                    <div className="space-y-3">
                      <label className="block text-sm font-medium text-white">Sample-Dataset</label>
                      {datasets.length === 0 ? (
                        <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                          <span className="text-amber-300 text-xs">Kein Dataset für dieses Modell vorhanden. Bitte zuerst ein Dataset hochladen.</span>
                        </div>
                      ) : (
                        <select
                          value={selectedSampleDatasetId ?? ''}
                          onChange={e => { setSelectedSampleDatasetId(e.target.value || null); setSamples([]); setSourceFileName(''); }}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none"
                        >
                          <option value="" className="bg-slate-900">– Dataset wählen –</option>
                          {datasets.map(d => (
                            <option key={d.id} value={d.id} className="bg-slate-900">
                              {d.name} ({d.file_count} Dateien{d.status === 'split' ? ' · gesplittet' : ''})
                            </option>
                          ))}
                        </select>
                      )}

                      {selectedSampleDatasetId && (
                        <>
                          {/* Split-Filter */}
                          <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
                            {(['all', 'train', 'val', 'test'] as const).map(split => (
                              <button
                                key={split}
                                onClick={() => setSelectedSampleSplit(split)}
                                className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${
                                  selectedSampleSplit === split
                                    ? 'bg-pink-500/20 text-pink-300 border border-pink-500/30'
                                    : 'text-gray-400 hover:text-white'
                                }`}
                              >
                                {split === 'all' ? 'Alle' : split.charAt(0).toUpperCase() + split.slice(1)}
                              </button>
                            ))}
                          </div>

                          {/* Laden-Button */}
                          <button
                            onClick={handleLoadFromDataset}
                            disabled={loadingDatasetSamples}
                            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-pink-500/15 hover:bg-pink-500/25 border border-pink-500/30 text-pink-300 text-sm font-medium transition-all disabled:opacity-50"
                          >
                            {loadingDatasetSamples
                              ? <><Loader2 className="w-4 h-4 animate-spin" /> Lade Samples…</>
                              : <><FolderOpen className="w-4 h-4" /> Samples laden</>}
                          </button>
                        </>
                      )}

                      {samples.length > 0 && (
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-pink-500/15 border border-pink-500/20 w-full">
                          <CheckCircle className="w-3.5 h-3.5 text-pink-400 flex-shrink-0" />
                          <span className="text-pink-300 text-xs font-medium flex-1">{samples.length} Samples geladen · {sourceFileName}</span>
                          <button onClick={() => { setSamples([]); setSourceFileName(''); }} className="text-gray-500 hover:text-red-400 transition-colors"><X className="w-3.5 h-3.5" /></button>
                        </div>
                      )}
                    </div>

                    {/* Samples Preview */}
                    {samples.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-white">Vorschau ({samples.length} Samples)</p>
                          <button onClick={() => { setSamples([]); setSourceFileName(''); }} className="text-xs text-gray-500 hover:text-red-400 transition-colors">Entfernen</button>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-black/20 divide-y divide-white/5 max-h-40 overflow-y-auto">
                          {samples.slice(0, 8).map((s, i) => (
                            <div key={s.id} className="flex items-center gap-3 px-3 py-2">
                              <span className="text-gray-600 text-[10px] tabular-nums w-4 flex-shrink-0">{i + 1}</span>
                              <span className="text-gray-300 text-xs flex-1 truncate">{s.text}</span>
                              {s.label && <span className="text-gray-500 text-[10px] flex-shrink-0 px-1.5 py-0.5 rounded bg-white/5">{s.label}</span>}
                            </div>
                          ))}
                          {samples.length > 8 && <div className="px-3 py-2 text-gray-600 text-xs">+ {samples.length - 8} weitere…</div>}
                        </div>
                      </div>
                    )}

                    {/* Start Button */}
                    <button
                      onClick={handleStartSession}
                      disabled={!selectedModel || !selectedVersionId || samples.length === 0 || (engineMode === 'engine' && !detectedPlugin) || (engineMode === 'dev' && !devScript.trim())}
                      className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 hover:opacity-90 text-white font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg"
                    >
                      <FlaskConical className="w-4 h-4" /> Lab-Session starten ({samples.length} Samples)
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* ── Testing Workspace ── */}
          {phase === 'testing' && session && currentSample && (
            <>
              {/* Server-Status Banner (nur Engine-Mode, wenn nicht ready) */}
              {engineMode === 'engine' && serverStatus !== 'ready' && (
                <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border text-sm ${
                  serverStatus === 'loading'
                    ? 'bg-blue-500/10 border-blue-500/20 text-blue-300'
                    : serverStatus === 'error'
                    ? 'bg-red-500/10 border-red-500/20 text-red-300'
                    : 'bg-white/5 border-white/10 text-gray-400'
                }`}>
                  {serverStatus === 'loading'
                    ? <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
                    : <AlertCircle className="w-4 h-4 flex-shrink-0" />}
                  <span>
                    {serverStatus === 'loading'
                      ? 'Modell wird geladen… Testen wird gleich möglich.'
                      : serverStatus === 'error'
                      ? 'Modell konnte nicht geladen werden. Bitte Version wechseln.'
                      : 'Kein Modell geladen.'}
                  </span>
                </div>
              )}

              {/* Progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Sample {currentSampleIdx + 1} von {samples.length}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-emerald-400">✅ {results.filter(r => r.userRating === 'correct').length}</span>
                    <span className="text-red-400">❌ {results.filter(r => r.userRating === 'wrong').length}</span>
                    <span className="text-gray-500">⏭ {results.filter(r => r.userRating === 'skipped').length}</span>
                  </div>
                </div>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-pink-500 to-rose-500 transition-all duration-300"
                    style={{ width: `${((currentSampleIdx) / samples.length) * 100}%` }} />
                </div>
              </div>

              {/* Navigation + Satz-Strip */}
              <div className="flex items-center gap-3">
                <button onClick={() => { if (currentSampleIdx > 0) { setCurrentSampleIdx(v => v - 1); setTestResult(null); setTestError(null); setUserNote(''); }}}
                  disabled={currentSampleIdx === 0} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-30 flex-shrink-0">
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <div className="flex-1 min-w-0 px-3 py-2 rounded-xl bg-white/5 border border-white/10">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-gray-400 text-[10px] tabular-nums font-medium">Sample #{currentSample.index + 1} / {samples.length}</span>
                    {currentSample.label && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-pink-500/15 border border-pink-500/20 text-pink-300">
                        {currentSample.label}
                      </span>
                    )}
                  </div>
                  {/* Metadaten-Chips */}
                  {(() => {
                    const info = getSideInfo(currentSample);
                    if (info.length === 0) return null;
                    return (
                      <div className="flex flex-wrap gap-x-3 gap-y-1">
                        {info.map(({ key, value }) => (
                          <span key={key} className="flex items-center gap-1 text-[10px]">
                            <span className="text-gray-600">{key}</span>
                            <span className="text-gray-300 truncate max-w-[180px]">{value}</span>
                          </span>
                        ))}
                      </div>
                    );
                  })()}
                </div>
                <button onClick={() => { if (currentSampleIdx < samples.length - 1) { setCurrentSampleIdx(v => v + 1); setTestResult(null); setTestError(null); setUserNote(''); }}}
                  disabled={currentSampleIdx === samples.length - 1} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-30 flex-shrink-0">
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              {/* Sample Card + Result – zweispaltig */}
              <div className="grid grid-cols-2 gap-4">
                {/* Left: Sample Actions */}
                <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-pink-400" />
                    <span className="text-white font-medium text-sm">Sample #{currentSample.index + 1}</span>
                  </div>

                  <div className="rounded-xl bg-black/30 border border-white/10 p-3 max-h-36 overflow-y-auto">
                    <p className="text-gray-200 text-xs leading-relaxed whitespace-pre-wrap">{getDisplayText(currentSample)}</p>
                  </div>

                  {/* Rohdaten (aufklappbar) */}
                  {typeof currentSample.rawData === 'object' && currentSample.rawData !== null && Object.keys(currentSample.rawData as object).length > 1 && (
                    <details className="group">
                      <summary className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 cursor-pointer list-none">
                        <Eye className="w-3 h-3" /> Rohdaten anzeigen
                      </summary>
                      <pre className="mt-2 text-[10px] text-gray-500 font-mono overflow-x-auto bg-black/20 rounded-lg p-2 max-h-24">{JSON.stringify(currentSample.rawData, null, 2)}</pre>
                    </details>
                  )}

                  {/* Test-Button */}
                  {!alreadyRated ? (
                    <button onClick={handleRunTest} disabled={testing}
                      className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 hover:opacity-90 text-white font-semibold text-sm transition-all disabled:opacity-60">
                      {testing ? <><Loader2 className="w-4 h-4 animate-spin" /> Teste…</> : <><Play className="w-4 h-4" /> Testen</>}
                    </button>
                  ) : (
                    <div className="flex items-center gap-2 justify-center py-2 text-xs text-gray-500">
                      <CheckCircle className="w-3.5 h-3.5" /> Bereits bewertet als <strong className="text-white">{alreadyRated.userRating}</strong>
                    </div>
                  )}

                  <button onClick={handleSkipWithoutTest} className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-500 hover:text-gray-300 text-xs transition-all">
                    <SkipForward className="w-3.5 h-3.5" /> Überspringen
                  </button>
                </div>

                {/* Right: Result + Rating */}
                <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-amber-400" />
                    <span className="text-white font-medium text-sm">Ergebnis</span>
                  </div>

                  {/* Idle State */}
                  {!testResult && !testError && !testing && (
                    <div className="flex flex-col items-center justify-center py-8 gap-3">
                      <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                        <Play className="w-5 h-5 text-gray-600" />
                      </div>
                      <p className="text-gray-600 text-sm">Drücke „Testen" um das Ergebnis zu sehen</p>
                    </div>
                  )}

                  {/* Loading */}
                  {testing && (
                    <div className="flex flex-col items-center justify-center py-8 gap-3">
                      <Loader2 className="w-8 h-8 text-pink-400 animate-spin" />
                      <p className="text-gray-400 text-sm">Inference läuft…</p>
                    </div>
                  )}

                  {/* Error */}
                  {testError && !testing && (
                    <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                      <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                      <p className="text-red-300 text-sm">{testError}</p>
                    </div>
                  )}

                  {/* Result */}
                  {testResult && !testing && (
                    <>
                      {/* Hauptklasse */}
                      <div className="px-4 py-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-between">
                        <span className="text-amber-300 text-lg font-bold">{testResult.predicted}</span>
                        <div className="text-right">
                          {testResult.confidence != null && (
                            <p className="text-amber-400 font-mono text-base font-semibold">{(testResult.confidence * 100).toFixed(1)}%</p>
                          )}
                          <p className="text-gray-600 text-[10px]">{testResult.inferenceMs.toFixed(0)} ms</p>
                        </div>
                      </div>

                      {/* Korrektheits-Indikator falls Label bekannt */}
                      {currentSample.label && (
                        <div className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${testResult.predicted === currentSample.label ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-300' : 'bg-red-500/10 border border-red-500/20 text-red-300'}`}>
                          {testResult.predicted === currentSample.label
                            ? <><CheckCircle className="w-3.5 h-3.5" /> Übereinstimmend mit erwartetem Label</>
                            : <><XCircle className="w-3.5 h-3.5" /> Abweichend — erwartet: <strong>{currentSample.label}</strong></>}
                        </div>
                      )}

                      {/* Top Predictions */}
                      {testResult.topPredictions && testResult.topPredictions.length > 1 && (
                        <div className="space-y-1.5">
                          <p className="text-xs text-gray-500">Alle Klassen</p>
                          {[...testResult.topPredictions].sort((a, b) => b.score - a.score).slice(0, 6).map(p => (
                            <div key={p.label} className="flex items-center gap-2 text-xs">
                              <span className={`w-28 truncate flex-shrink-0 ${p.label === testResult.predicted ? 'text-amber-300 font-medium' : 'text-gray-400'}`}>{p.label}</span>
                              <ConfidenceBar value={p.score} color={p.label === testResult.predicted ? 'amber' : 'blue'} />
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Notiz */}
                      <div className="space-y-2">
                        {showNote ? (
                          <textarea value={userNote} onChange={e => setUserNote(e.target.value)} rows={2} placeholder="Notiz zum Sample…"
                            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none" />
                        ) : (
                          <button onClick={() => setShowNote(true)} className="text-xs text-gray-600 hover:text-gray-400 transition-colors flex items-center gap-1">
                            <Pencil className="w-3 h-3" /> Notiz hinzufügen
                          </button>
                        )}
                      </div>

                      {/* Rating Buttons */}
                      <div className="flex gap-2 pt-1">
                        <button onClick={() => handleRate('correct')} className="flex-1 flex items-center justify-center gap-1.5 py-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 font-semibold text-sm transition-all">
                          <ThumbsUp className="w-4 h-4" /> Korrekt
                        </button>
                        <button onClick={() => handleRate('wrong')} className="flex-1 flex items-center justify-center gap-1.5 py-3 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-sm transition-all">
                          <ThumbsDown className="w-4 h-4" /> Falsch
                        </button>
                        <button onClick={() => handleRate('skipped')} className="px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all">
                          <SkipForward className="w-4 h-4" />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Mini Session Summary */}
              {results.length > 0 && (
                <div className="flex items-center gap-4 px-5 py-3 rounded-2xl bg-white/[0.03] border border-white/10">
                  <AccuracyDonut correct={results.filter(r => r.userRating === 'correct').length} wrong={results.filter(r => r.userRating === 'wrong').length} skipped={results.filter(r => r.userRating === 'skipped').length} />
                  <div className="flex-1 space-y-1">
                    <p className="text-white text-sm font-medium">Bisherige Session</p>
                    <div className="flex items-center gap-4 text-xs">
                      <span className="text-emerald-400">✅ {results.filter(r => r.userRating === 'correct').length} korrekt</span>
                      <span className="text-red-400">❌ {results.filter(r => r.userRating === 'wrong').length} falsch</span>
                      <span className="text-gray-500">⏭ {results.filter(r => r.userRating === 'skipped').length} übersprungen</span>
                    </div>
                  </div>
                  <button onClick={() => setPhase('analysis')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-500/10 hover:bg-pink-500/20 border border-pink-500/20 text-pink-300 text-xs font-medium transition-all">
                    <BarChart3 className="w-3.5 h-3.5" /> Analyse öffnen
                  </button>
                </div>
              )}
            </>
          )}

          {/* ── Abgeschlossen ── */}
          {phase === 'testing' && session && !currentSample && (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-8 text-center space-y-3">
              <CheckCircle className="w-10 h-10 text-emerald-400 mx-auto" />
              <p className="text-white font-semibold">Alle Samples bewertet!</p>
              <button onClick={() => setPhase('analysis')} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-pink-500/20 hover:bg-pink-500/30 border border-pink-500/30 text-pink-300 font-medium text-sm transition-all">
                <BarChart3 className="w-4 h-4" /> Zur Auswertung
              </button>
            </div>
          )}
        </>
      )}

      {/* Sessions Modal */}
      {showSessions && <SessionsModal onLoad={handleLoadSession} onClose={() => setShowSessions(false)} />}
    </div>
  );
}
