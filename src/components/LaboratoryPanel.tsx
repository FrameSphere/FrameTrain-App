// LaboratoryPanel.tsx – Interaktives Sample-Labor
// Workflow: Datei laden → Samples extrahieren → Einzeln testen → Bewerten → Auswerten

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  FlaskConical, Upload, Play, ChevronRight, ChevronLeft,
  CheckCircle, XCircle, SkipForward, Loader2, AlertCircle,
  BarChart3, X, FileText, Code2, Layers, ChevronDown, ChevronUp,
  Trash2, RotateCcw, Download, Eye, Sparkles, Terminal,
  ThumbsUp, ThumbsDown, Minus, TrendingUp, TrendingDown,
  ClipboardList, Save, FolderOpen, Bot, Send, Pencil,
  Check, Wand2, Copy, Maximize2, Minimize2, Zap,
} from 'lucide-react';
import { detectPlugin, pickPreferredModelId } from '../plugins/registry';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { useLanguage } from '../contexts/LanguageContext';
import { usePageContext } from '../contexts/PageContext';
import { callAI } from './TrainingPanel';
import OpenLibraryModal from './OpenLibraryModal';
import { readUserDevScripts } from '../utils/devScriptStorage';
import { useContextMenuActions } from '../ui/contextMenuRegistry';
import { dateLocale } from '../utils/dateLocale';
import { parseDelimitedRows } from './csvRows';

// ── Eigene Dev-Scripts (DevTrain + DevTest) — strikt user-getrennt ───────────
interface LabSavedScript { id: string; name: string; script: string; savedAt: string; source: 'train' | 'test'; }

function loadAllSavedScripts(userId?: string): LabSavedScript[] {
  const { train, test } = readUserDevScripts(userId);
  const seen = new Set<string>();
  return [
    ...train.map(s => ({ ...s, source: 'train' as const })),
    ...test.map(s => ({ ...s, source: 'test' as const })),
  ]
    .filter(s => s && s.id && s.script && !seen.has(s.id) && (seen.add(s.id), true))
    .sort((a, b) => (b.savedAt ?? '').localeCompare(a.savedAt ?? ''));
}

// ── Types ─────────────────────────────────────────────────────────────────

interface ModelInfo {
  id: string; name: string; source: string;
  source_path: string | null; local_path: string;
  model_type: string | null; size_bytes?: number;
}

interface VersionTreeItem { id: string; name: string; is_root: boolean; version_number: number; }
interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }

type LabInputKind = 'text' | 'image' | 'audio' | 'tensor';

/** Zeilen, die aus einer Parquet-Datei als Samples geladen werden (Backend deckelt bei 500). */
const PARQUET_SAMPLE_ROWS = 200;

interface LabSample {
  id: string;
  index: number;
  text: string;          // Haupttext für die Inference
  label?: string;        // Erwartetes Label (optional)
  rawData: unknown;      // Original-Daten aus Datei
  filePath?: string;     // Datei-Sample: absoluter Pfad (Bild-/Audio-Modelle)
  fileKind?: 'image' | 'audio';  // gesetzt → Datei-Inferenz statt Text/Tensor
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

// FIX: Key pro User – verhindert Cross-Account-Leakage
const sessionsKey = (userId?: string) =>
  userId ? `ft_lab_sessions_${userId}` : 'ft_lab_sessions';

const loadSessions = (userId?: string): LabSession[] => {
  try { return JSON.parse(localStorage.getItem(sessionsKey(userId)) ?? '[]'); } catch { return []; }
};

const saveSession = (s: LabSession, userId?: string) => {
  const all = loadSessions(userId);
  const idx = all.findIndex(x => x.id === s.id);
  if (idx >= 0) all[idx] = s; else all.unshift(s);
  localStorage.setItem(sessionsKey(userId), JSON.stringify(all.slice(0, 20)));
};

const deleteSession = (id: string, userId?: string) => {
  localStorage.setItem(sessionsKey(userId), JSON.stringify(loadSessions(userId).filter(s => s.id !== id)));
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

/** Objekt-Zeilen (Parquet-Preview) in Samples umwandeln — gleiche Feld-Erkennung wie parseSamples. */
function samplesFromRows(rows: unknown[]): LabSample[] {
  return rows.map((item, i) => ({
    id: `p_${Date.now()}_${i}`,
    index: i,
    text: extractTextField(item),
    label: extractLabelField(item),
    rawData: item,
  }));
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
      // Anfuehrungszeichen beachten: "Bingen: sonnig, 0 Grad" ist ein Feld.
      raw.push(...parseDelimitedRows(content, effectiveExt === 'tsv' ? '\t' : ','));
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

function AccuracyDonut({
  correct,
  wrong,
  skipped,
  centerLabel,
  labels,
}: {
  correct: number;
  wrong: number;
  skipped: number;
  centerLabel: string;
  labels: { correct: string; wrong: string; skipped: string };
}) {
  const total = correct + wrong + skipped;
  if (total === 0) return <div className="w-20 h-20 rounded-full bg-white/10 flex items-center justify-center"><span className="text-gray-600 text-xs">–</span></div>;

  const R = 30; const C = 2 * Math.PI * R;
  const correctPct = correct / total;
  const wrongPct   = wrong   / total;
  // Die Ringe zeigen die Verteilung aller Samples, die Zahl in der Mitte aber
  // dieselbe Quote wie "Accuracy (bewertet)": uebersprungene Samples sind nicht
  // bewertet und duerfen die Trefferquote nicht druecken.
  const ratedCount = correct + wrong;
  const accuracyPct = ratedCount > 0 ? correct / ratedCount : 0;

  const correctArc = C * correctPct;
  const wrongArc   = C * wrongPct;
  const skipArc    = C - correctArc - wrongArc;

  let offset = C * 0.25; // Start oben
  const arcs = [
    { arc: correctArc, color: '#10b981', label: labels.correct },
    { arc: wrongArc,   color: '#ef4444', label: labels.wrong },
    { arc: skipArc,    color: '#374151', label: labels.skipped },
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
        <span className="text-white font-bold text-sm">{ratedCount > 0 ? `${(accuracyPct * 100).toFixed(0)}%` : '–'}</span>
        <span className="text-gray-500 text-[9px]">{centerLabel}</span>
      </div>
    </div>
  );
}

// ── Sessions Modal ────────────────────────────────────────────────────────

function SessionsModal({ onLoad, onClose, userId }: { onLoad: (s: LabSession) => void; onClose: () => void; userId?: string }) {
  const [sessions, setSessions] = useState<LabSession[]>([]);
  const { success } = useNotification();
  const { t, language } = useLanguage();

  useEffect(() => { setSessions(loadSessions(userId)); }, [userId]);

  const handleDelete = (id: string) => {
    deleteSession(id, userId);
    setSessions(loadSessions(userId));
    success(t('laboratoryPanel.sessionsModal.deleteSuccess'), '');
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><FlaskConical className="w-5 h-5 text-pink-400" /><h2 className="text-lg font-bold text-white">{t('laboratoryPanel.sessionsModal.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {sessions.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <FlaskConical className="w-10 h-10 text-gray-600 mx-auto" />
              <p className="text-gray-500 text-sm">{t('laboratoryPanel.sessionsModal.empty')}</p>
            </div>
          ) : sessions.map(s => {
            const correct = s.results.filter(r => r.userRating === 'correct').length;
            const wrong   = s.results.filter(r => r.userRating === 'wrong').length;
            const total   = s.results.length;
            // "bewertet" meint korrekt + falsch — uebersprungene Samples sind
            // gesehen, aber nicht bewertet.
            const rated   = correct + wrong;
            return (
              <div key={s.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium text-sm truncate">{s.name}</p>
                    <p className="text-gray-500 text-xs">{s.modelName} · {s.versionName}</p>
                    <div className="flex items-center gap-3 mt-1.5">
                      <span className="text-[10px] text-gray-500">{t('laboratoryPanel.sessionsModal.ratedLabel', { tested: rated, total: s.totalSamples })}</span>
                      {total > 0 && (
                        <>
                          <span className="text-[10px] text-emerald-400 inline-flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" />{correct}</span>
                          <span className="text-[10px] text-red-400 inline-flex items-center gap-1"><XCircle className="w-3.5 h-3.5" />{wrong}</span>
                        </>
                      )}
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-md border ${s.engineMode === 'engine' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20' : 'bg-blue-500/15 text-blue-400 border-blue-500/20'}`}>
                        {s.engineMode === 'engine' ? t('laboratoryPanel.sessionsModal.engineBadge') : t('laboratoryPanel.sessionsModal.devScriptBadge')}
                      </span>
                    </div>
                    <p className="text-gray-600 text-[10px] mt-1">{new Date(s.updatedAt).toLocaleDateString(dateLocale(language), { day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit' })}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all">
                    <button onClick={() => handleDelete(s.id)} className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>
                    <button onClick={() => { onLoad(s); onClose(); }} className="px-3 py-1.5 rounded-xl bg-pink-500/20 hover:bg-pink-500/30 border border-pink-500/30 text-pink-300 text-xs font-medium transition-all">{t('laboratoryPanel.sessionsModal.loadButton')}</button>
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
  const { t, language } = useLanguage();
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
      const resp = await callAI(aiSettings, sys, last.content, history, language);
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
        <span className="text-sm font-medium text-white">{t('laboratoryPanel.devScript.title')}</span>
        <div className="flex items-center gap-2">
          <button onClick={() => onChange('')} className="text-[10px] text-gray-500 hover:text-red-400 transition-colors">{t('laboratoryPanel.devScript.clearButton')}</button>
            <button onClick={generateTemplate} className="flex items-center gap-1 px-2 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-medium hover:bg-blue-500/20 transition-all">
            <Sparkles className="w-3 h-3" /> {t('laboratoryPanel.devScript.templateButton')}
          </button>
          {aiSettings.enabled && (
            <button onClick={() => setShowAI(v => !v)} className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium border transition-all ${showAI ? 'bg-violet-500/20 text-violet-300 border-violet-500/30' : 'bg-white/5 text-gray-400 border-white/10 hover:text-white'}`}>
              <Bot className="w-3 h-3" /> {t('laboratoryPanel.devScript.aiSidebar.titleShort')}
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
          placeholder={t('laboratoryPanel.devScript.placeholder')}
          className="flex-1 p-4 bg-slate-950 border border-white/10 rounded-xl text-[11px] font-mono text-gray-200 focus:outline-none focus:border-blue-500/40 resize-none placeholder:text-gray-700 leading-[1.6rem]"
          style={{ fontFamily: "'JetBrains Mono','Fira Code','Courier New',monospace" }}
        />

        {showAI && (
          <div className="w-72 flex flex-col bg-slate-950 border border-white/10 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/10 bg-white/[0.02] flex-shrink-0">
              <div className="flex items-center gap-1.5"><Wand2 className="w-3.5 h-3.5 text-violet-400" /><span className="text-xs font-medium text-white">{t('laboratoryPanel.devScript.aiSidebar.title')}</span></div>
              <button onClick={() => setShowAI(false)} className="p-1 rounded hover:bg-white/5 text-gray-500"><X className="w-3 h-3" /></button>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {aiMessages.length === 0 && <p className="text-gray-600 text-[10px] text-center py-4">{t('laboratoryPanel.devScript.aiSidebar.emptyHint')}</p>}
              {aiMessages.map((m, i) => (
                <div key={i} className={`px-2.5 py-2 rounded-lg text-[10px] leading-relaxed ${m.role === 'user' ? 'bg-violet-500/10 text-gray-200 border border-violet-500/15 ml-4' : 'bg-white/5 text-gray-300 border border-white/10 mr-4'}`}>
                  {m.content.replace(/```python[\s\S]*?```/g, '[Code wurde übernommen]').trim()}
                </div>
              ))}
              {aiLoading && <div className="px-2.5 py-2 rounded-lg bg-white/5 border border-white/10 mr-4"><Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin" /></div>}
              <div ref={aiEndRef} />
            </div>
            <div className="p-2 border-t border-white/10 flex gap-1.5 flex-shrink-0">
              <input value={aiInput} onChange={e => setAiInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && askAI()}
                placeholder={t('laboratoryPanel.devScript.aiSidebar.inputPlaceholder')} className="flex-1 px-2.5 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-[10px] placeholder:text-gray-600 focus:outline-none" />
              <button onClick={askAI} disabled={!aiInput.trim() || aiLoading} className="p-1.5 rounded-lg bg-violet-500/20 border border-violet-500/30 text-violet-300 disabled:opacity-40 transition-all">
                <Send className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </div>
      <p className="text-[10px] text-gray-600">{t('laboratoryPanel.devScript.footerNote')}</p>
    </div>
  );
}

// ── Analysis View ─────────────────────────────────────────────────────────

function AnalysisView({ session, onBack }: { session: LabSession; onBack: () => void }) {
  const { t } = useLanguage();
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
      [t('laboratoryPanel.analysis.csvHeaderIndex'), t('laboratoryPanel.analysis.csvHeaderInput'), t('laboratoryPanel.analysis.csvHeaderExpected'), t('laboratoryPanel.analysis.csvHeaderPredicted'), t('laboratoryPanel.analysis.csvHeaderConfidence'), t('laboratoryPanel.analysis.csvHeaderRating'), t('laboratoryPanel.analysis.csvHeaderNote')],
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
          <div><h2 className="text-lg font-bold text-white">{session.name}</h2><p className="text-gray-500 text-xs">{session.modelName} · {session.versionName} · {session.engineMode === 'engine' ? t('laboratoryPanel.sessionsModal.engineBadge') : t('laboratoryPanel.sessionsModal.devScriptBadge')}</p></div>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-sm transition-all">
          <Download className="w-4 h-4" /> {t('laboratoryPanel.analysis.exportButton')}
        </button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: t('laboratoryPanel.analysis.statsRated'), value: `${rated}/${session.totalSamples}`, color: 'text-white' },
          { label: t('laboratoryPanel.analysis.statsCorrect'), value: correct.length, color: 'text-emerald-400' },
          { label: t('laboratoryPanel.analysis.statsWrong'), value: wrong.length, color: 'text-red-400' },
          { label: t('laboratoryPanel.analysis.statsSkipped'), value: skipped.length, color: 'text-gray-400' },
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
                  <AccuracyDonut
                    correct={correct.length}
                    wrong={wrong.length}
                    skipped={skipped.length}
                    centerLabel={t('laboratoryPanel.analysis.donutCenterLabel')}
                    labels={{
                      correct: t('laboratoryPanel.analysis.donutCorrectLabel'),
                      wrong: t('laboratoryPanel.analysis.donutWrongLabel'),
                      skipped: t('laboratoryPanel.analysis.donutSkipLabel'),
                    }}
                  />
          <div className="space-y-1 w-full">
            {[
              { color: 'bg-emerald-400', label: t('laboratoryPanel.analysis.donutCorrectLabel'), n: correct.length },
              { color: 'bg-red-400',     label: t('laboratoryPanel.analysis.donutWrongLabel'),  n: wrong.length },
              { color: 'bg-gray-600',    label: t('laboratoryPanel.analysis.donutSkipLabel'),    n: skipped.length },
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
          <p className="text-sm font-medium text-white">{t('laboratoryPanel.analysis.avgConfidenceTitle')}</p>
          {[
            { label: t('laboratoryPanel.analysis.avgConfCorrect'), val: avgConf(correct), color: 'emerald' },
            { label: t('laboratoryPanel.analysis.avgConfWrong'),  val: avgConf(wrong),   color: 'red' },
          ].map(x => (
            <div key={x.label} className="space-y-1">
              <span className="text-xs text-gray-400">{x.label}</span>
              {x.val != null ? <ConfidenceBar value={x.val} color={x.color} /> : <span className="text-gray-600 text-xs">–</span>}
            </div>
          ))}
          <div className="pt-1 border-t border-white/10">
            <p className="text-xs text-gray-500">{t('laboratoryPanel.analysis.accuracyLabel')}</p>
            <p className={`text-xl font-bold mt-0.5 ${accuracy > 0.8 ? 'text-emerald-400' : accuracy > 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
              {rated > 0 ? `${(accuracy * 100).toFixed(1)}%` : '–'}
            </p>
          </div>
        </div>

        {/* Häufigste Falsch-Predictions */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-3">
          <p className="text-sm font-medium text-white">{t('laboratoryPanel.analysis.topWrongTitle')}</p>
          {topWrongPreds.length === 0
            ? <p className="text-gray-600 text-xs">{t('laboratoryPanel.analysis.topWrongEmpty')}</p>
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
          <p className="text-white font-medium text-sm">{t('laboratoryPanel.analysis.resultsTitle', { count: filtered.length })}</p>
          <div className="flex items-center gap-1 p-1 rounded-lg bg-white/5 border border-white/10">
            {([
              { val: 'all', label: t('laboratoryPanel.analysis.filterAll'), icon: <ClipboardList className="w-3.5 h-3.5" /> },
              { val: 'correct', label: t('laboratoryPanel.analysis.filterCorrect'), icon: <CheckCircle className="w-3.5 h-3.5" /> },
              { val: 'wrong', label: t('laboratoryPanel.analysis.filterWrong'), icon: <XCircle className="w-3.5 h-3.5" /> },
              { val: 'skipped', label: t('laboratoryPanel.analysis.filterSkipped'), icon: <SkipForward className="w-3.5 h-3.5" /> },
            ] as const).map(({ val, label, icon }) => (
              <button
                key={val}
                onClick={() => setFilterRating(val as typeof filterRating)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${filterRating === val ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}
              >
                <span className="inline-flex items-center gap-1.5">
                  {icon}
                  <span>{label}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="divide-y divide-white/5 max-h-96 overflow-y-auto">
          {filtered.length === 0
            ? <p className="text-gray-600 text-sm text-center py-8">{t('laboratoryPanel.analysis.noEntries')}</p>
            : filtered.map((r, i) => (
              <div key={r.sampleId}>
                <button onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  className="w-full flex items-center gap-3 px-5 py-3 hover:bg-white/[0.03] transition-all text-left">
                  <span className="text-gray-600 text-xs tabular-nums w-6 flex-shrink-0">{r.sampleIndex + 1}</span>
                  <span className="flex-shrink-0">
                    {r.userRating === 'correct'
                      ? <CheckCircle className="w-4 h-4 text-emerald-400" />
                      : r.userRating === 'wrong'
                        ? <XCircle className="w-4 h-4 text-red-400" />
                        : <SkipForward className="w-4 h-4 text-gray-400" />
                    }
                  </span>
                  <span className="text-gray-300 text-xs flex-1 truncate">{r.inputText}</span>
                  <span className="text-white text-xs font-medium flex-shrink-0">{r.predicted}</span>
                  {r.confidence != null && <span className="text-gray-500 text-xs font-mono tabular-nums flex-shrink-0">{(r.confidence * 100).toFixed(0)}%</span>}
                  {expandedIdx === i ? <ChevronUp className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" /> : <ChevronDown className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />}
                </button>
                {expandedIdx === i && (
                  <div className="px-5 pb-4 space-y-2 bg-white/[0.02] border-t border-white/5">
                    <div className="grid grid-cols-2 gap-3 pt-3 text-xs">
                      <div><span className="text-gray-500">{t('laboratoryPanel.analysis.detailInput')}</span><p className="text-gray-300 mt-0.5">{r.inputText}</p></div>
                      <div><span className="text-gray-500">{t('laboratoryPanel.analysis.detailPredicted')}</span><p className="text-white font-semibold mt-0.5">{r.predicted}</p></div>
                      {r.expectedLabel && <div><span className="text-gray-500">{t('laboratoryPanel.analysis.detailExpected')}</span><p className="text-gray-300 mt-0.5">{r.expectedLabel}</p></div>}
                      {r.userNote && <div className="col-span-2"><span className="text-gray-500">{t('laboratoryPanel.analysis.detailNote')}</span><p className="text-gray-400 italic mt-0.5">{r.userNote}</p></div>}
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

export default function LaboratoryPanel({ userId }: { userId?: string }) {
  const { success, error, warning } = useNotification();
  const { t, language } = useLanguage();
  const { setCurrentPageContent } = usePageContext();

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
  // Script-Quellen für den Dev-Modus: eigene Bibliothek + Open Library
  const [showMyScripts, setShowMyScripts] = useState(false);
  const [myScripts, setMyScripts] = useState<LabSavedScript[]>([]);
  const [showOpenLib, setShowOpenLib] = useState(false);

  // Model Server Status
  const [serverStatus, setServerStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  // Konkrete Fehlermeldung vom Server-Start (inline sichtbar, nicht nur als Toast)
  const [serverErrorMsg, setServerErrorMsg] = useState<string | null>(null);
  const [serverVersionId, setServerVersionId] = useState<string | null>(null);
  const [serverInputKind, setServerInputKind] = useState<LabInputKind | null>(null);
  const [serverModality,  setServerModality]  = useState<string | null>(null);
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

  useEffect(() => () => {
    unlistenRef.current.forEach(fn => { try { fn(); } catch { /* listener already removed */ } });
    unlistenRef.current = [];
  }, []);

  // ── Derived (muss VOR allen useEffects stehen – TDZ-Vermeidung) ──────────

  const selectedModel      = models.find(m => m.id === selectedModelId);
  const selectedModelTree  = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersionTree = selectedModelTree?.versions.find(v => v.id === selectedVersionId);

  const detectedPlugin = useMemo(() => {
    if (!selectedModel) return null;
    const r = detectPlugin(
      selectedModel.source_path ?? selectedModel.name,
      selectedModel.model_type ? { model_type: selectedModel.model_type } : undefined,
    );
    return r.supported ? r.plugin : null;
  }, [selectedModel]);

  // Die Lab-Inferenz spricht nur HuggingFace-Formate. Ein YOLO-Modell wurde
  // trotzdem als Plugin gemeldet, liess Samples laden und scheiterte erst beim
  // Modell-Laden an der fehlenden config.json — eine Sackgasse.
  const pluginUnsupportedInLab = detectedPlugin?.id === 'yolo';

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
        const preferred = pickPreferredModelId(listWithVersions, list);
        if (preferred) setSelectedModelId(preferred);
      } catch (e) { console.error('[Lab] initLoad:', e); }
      finally { setLoadingModels(false); }
    })();
  }, [t]);

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

  // ── Model Server: Laden erst nach expliziter Bestätigung durch den User ──
  // (vorher wurde bei jedem Versions-Wechsel sofort automatisch geladen)

  const handleLoadModel = useCallback(() => {
    if (!selectedVersionId) return;
    setServerErrorMsg(null);
    setServerStatus('loading');
    serverStatusRef.current = 'loading';
    invoke('lab_start_model_server', { versionId: selectedVersionId }).catch(e => {
      console.error('[Lab] lab_start_model_server:', e);
    });
  }, [selectedVersionId]);

  // ── Rechtsklick-Menü: Lab-Aktionen ────────────────────────────────────────
  useContextMenuActions(() => [
    {
      id: 'lab-load', group: t('sidebar.nav.laboratory'),
      label: t('laboratoryPanel.setup.loadModelButton'), icon: Zap,
      disabled: !selectedVersionId || serverStatus === 'loading',
      onSelect: () => handleLoadModel(),
    },
    {
      id: 'lab-engine', group: t('sidebar.nav.laboratory'),
      label: engineMode === 'engine'
        ? t('laboratoryPanel.setup.engineDev')
        : t('laboratoryPanel.setup.engineLabel'),
      icon: Code2,
      onSelect: () => setEngineMode(m => m === 'engine' ? 'dev' : 'engine'),
    },
    {
      id: 'lab-sessions', group: t('sidebar.nav.laboratory'),
      label: t('laboratoryPanel.header.sessionsButton'), icon: FolderOpen,
      onSelect: () => setShowSessions(true),
    },
  ]);

  useEffect(() => {
    const unlisten = listen<{ status: string; version_id?: string; message?: string; input_kind?: string; modality?: string }>(
      'lab-server-status',
      e => {
        const { status, version_id, message, input_kind, modality } = e.payload;
        console.log('[Lab] Server-Status:', status, version_id, message, modality);
        setServerStatus(status as typeof serverStatus);
        serverStatusRef.current = status as typeof serverStatus;
        if (status === 'ready' && version_id) setServerVersionId(version_id);
        if (status === 'ready') {
          setServerInputKind((input_kind as LabInputKind) ?? 'text');
          setServerModality(modality ?? 'text');
        }
        if (status === 'loading') { setServerInputKind(null); setServerModality(null); }
        if (status === 'error') {
          setServerVersionId(null);
          setServerInputKind(null);
          setServerModality(null);
          setServerErrorMsg(message ?? null);
          if (message) error(t('laboratoryPanel.setup.notifications.modelLoadError'), message);
        }
      }
    );
    return () => { unlisten.then(fn => fn()).catch(() => { /* listener already removed */ }); };
  }, [t]);

  // Cleanup: Server stoppen wenn Komponente unmountet
  useEffect(() => {
    return () => {
      invoke('lab_stop_model_server').catch(() => {});
    };
  }, []);

  // ── Page context for AI coach ──────────────────────────────────────────────

  useEffect(() => {
    const results = session?.results ?? [];
    const testedCount = results.filter(r => r.userRating !== 'skipped').length;
    const correctCount = results.filter(r => r.userRating === 'correct').length;
    const accuracy = session?.totalSamples ? ((correctCount / testedCount) * 100).toFixed(1) : '0';

    const lines: string[] = [
      '=== FrameTrain Labor (LaboratoryPanel) ===',
      '',
      '--- SEITENZWECK ---',
      'Interaktives Sample-Testlabor für Modell-Inferenzen.',
      'Teste Samples einzeln, sehe Predictions & Confidence, bewerte Ergebnisse & analysiere Performance.',
      '',
      '--- AKTUELLE SETUP ---',
    ];

    if (!selectedModel) {
      lines.push('❌ Modell: (nicht gewählt) → Wähle ein Modell aus dem Dropdown oben');
    } else {
      lines.push(`✓ Modell: ${selectedModel.name} (${selectedModel.model_type ?? 'unbekannter Typ'})`);
      lines.push(`  Size: ${selectedModel.size_bytes}, Path: ${selectedModel.local_path || selectedModel.source_path || 'N/A'}`);
      
      if (selectedVersionTree) {
        lines.push(`✓ Version: ${selectedVersionTree.name} (v${selectedVersionTree.version_number})`);
      } else {
        lines.push('⚠️ Version: (nicht gewählt)');
      }

      lines.push(`  Engine-Mode: ${engineMode === 'engine' ? '⚡ Standard (schnell, pre-loaded)' : '🐍 Dev-Script (custom Python)'}`);
      
      if (engineMode === 'dev' && devScript) {
        lines.push(`  Dev-Script: ${devScript.length} Zeichen`);
      }

      if (detectedPlugin) {
        lines.push(`✓ Plugin: ${detectedPlugin}`);
      } else {
        lines.push('⚠️ Plugin: (keine Unterstützung erkannt)');
      }

      if (serverStatus === 'ready') {
        lines.push(`  Server: 🟢 Ready (${serverVersionId})`);
      } else if (serverStatus === 'loading') {
        lines.push('  Server: 🟡 Loading...');
      } else if (serverStatus === 'error') {
        lines.push('  Server: 🔴 Error');
      }
    }

    lines.push('');
    lines.push('--- DATENLADEN ---');

    if (samples.length === 0) {
      lines.push('❌ Keine Samples geladen → Lade Datei (CSV/JSON/JSONL/TXT) oder wähle Dataset');
    } else {
      lines.push(`✓ Samples geladen: ${samples.length} Samples`);
      lines.push(`  Dateiname: ${sourceFileName}`);
      if (selectedSampleDatasetId) {
        lines.push(`  Dataset: ${datasets.find(d => d.id === selectedSampleDatasetId)?.name || selectedSampleDatasetId}`);
        lines.push(`  Split: ${selectedSampleSplit}`);
      }
    }

    lines.push('');
    lines.push('--- TEST-WORKFLOW FORTSCHRITT ---');

    if (phase === 'setup') {
      lines.push('Phase: 1️⃣ **SETUP** (aktiv)');
      lines.push('  → Wähle Modell & Version, lade Samples');
    } else if (phase === 'testing') {
      lines.push('Phase: 2️⃣ **TESTING** (aktiv)');
      if (currentSampleIdx !== undefined && samples.length > 0) {
        lines.push(`  Sample: ${currentSampleIdx + 1}/${samples.length}`);
        lines.push(`  Tests durchgeführt: ${testedCount}/${samples.length}`);
        if (testResult) {
          lines.push(testResult.confidence != null
            ? `  Letztes Ergebnis: "${testResult.predicted}" (Confidence: ${testResult.confidence.toFixed(2)})`
            : `  Letztes Ergebnis: "${testResult.predicted}" (generierter Text, keine Konfidenz)`);
          if (testError) lines.push(`  ⚠️ Fehler: ${testError}`);
        }
        if (testing) {
          lines.push('  Status: 🔄 Inference läuft...');
        }
      }
    } else if (phase === 'analysis') {
      lines.push('Phase: 3️⃣ **ANALYSIS** (aktiv)');
      lines.push(`  Tests durchgeführt: ${testedCount}/${samples.length}`);
    } else {
      lines.push(`Phase: ${String(phase).toUpperCase()}`);
    }

    lines.push('');
    lines.push('--- STATISTIKEN (LIVE) ---');

    if (session) {
      lines.push(`Session: "${session.name}"`);
      lines.push(`Tests gesamt: ${results.length}/${session.totalSamples}`);
      lines.push(`Correct: ${correctCount}, Wrong: ${results.filter(r => r.userRating === 'wrong').length}, Skipped: ${results.filter(r => r.userRating === 'skipped').length}`);
      if (testedCount > 0) {
        lines.push(`Accuracy: ${accuracy}%`);
      }
    } else {
      lines.push('(keine Session aktiv)');
    }

    lines.push('');
    lines.push('--- VERFÜGBARE AKTIONEN ---');

    if (!selectedModel) {
      lines.push('• Wähle Modell aus der Liste');
    } else if (!samples.length) {
      lines.push('• Lade Daten: Datei hochladen oder Dataset aus Dropdown');
    } else if (phase === 'setup' || phase === 'testing') {
      lines.push('• Teste Sample: Button "Test diesen Sample" (oder Space-Taste)');
      lines.push('• Bewerte Ergebnis: "Correct" / "Wrong" / "Skip" → Optionale Notiz hinzufügen');
      lines.push('• Navigiere: ← → Pfeile zu vor/zurück');
      lines.push('• KI-Coach: Frag KI um Analyse oder Tipps zur Verbesserung');
    } else if (phase === 'analysis') {
      lines.push('• Exportiere Results: CSV-Download');
      lines.push('• Neue Session: Starte weiteren Test');
    }

    lines.push('');
    lines.push('--- UI LAYOUT ---');
    lines.push('**OBEN (Header):**');
    lines.push('  • [Modell Dropdown] (linke Seite)');
    lines.push('  • [Version Dropdown] (daneben)');
    lines.push('  • [Engine Mode Toggle] (rechts: Server/Dev)');
    lines.push('');
    lines.push('**LINKS (Data/Sample Panel):**');
    lines.push('  • ▼ Dataset Auswahl: Split-Dropdown (Train/Val/Test)');
    lines.push('  • Sample Counter: "Sample 5/100"');
    lines.push('  • [◄ Prev Sample] [Sample Input Display] [Next Sample ►]');
    lines.push('  • Raw Input: Text/JSON anzeige des aktuellen Samples');
    lines.push('');
    lines.push('**RECHTS (Results Panel):**');
    lines.push('  • Predicted Class + Confidence %');
    lines.push('  • Expected Class + Match Status (✓/✗)');
    lines.push('  • [👍 Correct] [👎 Wrong] [⏭️ Skip] Buttons');
    lines.push('  • Prediction Details (Top-3 Classes)');
    lines.push('');
    lines.push('**UNTEN (Progress/Stats):**');
    lines.push('  • Progress Bar: "5/100 samples tested"');
    lines.push('  • Live Stats: Accuracy %, Correct/Wrong/Skipped counts');
    lines.push('  • 📊 Accuracy Chart (real-time update)');
    lines.push('  • [Start Testing] [Pause] [Resume] Buttons');
    lines.push('');
    lines.push('--- TIPPS FÜR AI-COACH ---');
    lines.push('• KI kann Dir helfen, falsche Predictions zu verstehen');
    lines.push('• KI kann Patterns in Fehlern erkennen');
    lines.push('• KI kann Modell-Improvements vorschlagen basierend auf Ergebnissen');
    if (!selectedModel) {
      lines.push('• Wähle zuerst ein Modell!');
    }

    setCurrentPageContent(lines.join('\n'), 'laboratory');
  }, [
    selectedModel,
    selectedVersionTree,
    engineMode,
    devScript,
    detectedPlugin,
    serverStatus,
    serverVersionId,
    samples,
    sourceFileName,
    selectedSampleDatasetId,
    selectedSampleSplit,
    datasets,
    phase,
    currentSampleIdx,
    testResult,
    testError,
    testing,
    session,
    setCurrentPageContent,
    loadingDatasetSamples,
  ]);


  const currentSample = samples[currentSampleIdx] ?? null;
  // Erwartet der geladene Server eine andere Eingabeart als das aktuelle Sample?
  const inputMismatch: 'inputMismatchImage' | 'inputMismatchAudio' | 'inputMismatchText' | null =
    !currentSample || !serverInputKind || serverInputKind === 'tensor'
      ? null
      : serverInputKind === 'image' && currentSample.fileKind !== 'image'
      ? 'inputMismatchImage'
      : serverInputKind === 'audio' && currentSample.fileKind !== 'audio'
      ? 'inputMismatchAudio'
      : serverInputKind === 'text' && currentSample.fileKind
      ? 'inputMismatchText'
      : null;
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

      // Bild-Dataset (ImageFolder): Dateien sind Bilder → als Bild-Samples laden,
      // Label = übergeordneter Ordnername. Kein Text-Parsing.
      const IMG_EXT   = /\.(jpe?g|png|bmp|webp|gif|tiff?)$/i;
      const AUDIO_EXT = /\.(wav|mp3|flac|ogg|m4a|aac)$/i;
      const imageFiles = filtered.filter(f => IMG_EXT.test(f.name));
      const audioFiles = filtered.filter(f => AUDIO_EXT.test(f.name));
      const mediaFiles = imageFiles.length >= audioFiles.length ? imageFiles : audioFiles;
      const mediaKind: 'image' | 'audio' = imageFiles.length >= audioFiles.length ? 'image' : 'audio';
      if (mediaFiles.length > 0 && mediaFiles.length >= filtered.length * 0.5) {
        const mediaSamples: LabSample[] = mediaFiles.map((f, i) => {
          const parts = f.path.split(/[/\\]/);
          const folderLabel = parts.length >= 2 ? parts[parts.length - 2] : undefined;
          return {
            id: `${mediaKind}_${Date.now()}_${i}`,
            index: i,
            text: f.name,
            label: folderLabel,
            rawData: { path: f.path, name: f.name },
            filePath: f.path,
            fileKind: mediaKind,
          };
        });
        setSamples(mediaSamples);
        const dsMedia = datasets.find(d => d.id === selectedSampleDatasetId);
        setSourceFileName(dsMedia?.name ?? 'Dataset');
        success(
          t('laboratoryPanel.setup.notifications.loadSuccess'),
          t('laboratoryPanel.setup.notifications.loadSuccessDetail', { count: mediaSamples.length, fileCount: mediaFiles.length }),
        );
        return;
      }

      if (filtered.length === 0) {
        warning(
          t('laboratoryPanel.setup.notifications.noDatasetFiles'),
          t('laboratoryPanel.setup.notifications.noDatasetFilesDetail', {
            split: selectedSampleSplit,
            splits: [...new Set(files.filter(f => !f.is_dir).map(f => f.split))].join(', ') || 'keine',
          }),
        );
        return;
      }

      const allSamples: LabSample[] = [];
      const fileErrors: string[] = [];
      for (const file of filtered) {
        try {
          // Parquet ist binaer: read_dataset_file liefert dafuer nur einen
          // Platzhaltertext. Die Zeilen kommen deshalb ueber den Parquet-Preview.
          if (/\.parquet$/i.test(file.name)) {
            const pq = await invoke<{ rows?: unknown[]; total_rows?: number }>(
              'preview_parquet_file', { filePath: file.path, maxRows: PARQUET_SAMPLE_ROWS }
            );
            const rows = Array.isArray(pq?.rows) ? pq.rows : [];
            console.log(`[Lab] Parquet gelesen: ${file.name}, Zeilen: ${rows.length} von ${pq?.total_rows ?? '?'}`);
            allSamples.push(...samplesFromRows(rows));
            continue;
          }
          const content = await invoke<string>('read_dataset_file', { filePath: file.path });
          console.log(`[Lab] Datei gelesen: ${file.name}, Länge: ${content.length}`);
          const parsed = parseSamples(content, file.name);
          console.log(`[Lab] Samples aus ${file.name}:`, parsed.length);
          allSamples.push(...parsed);
        } catch (fileErr) {
          console.warn(`[Lab] Fehler beim Lesen von ${file.name}:`, fileErr);
          fileErrors.push(`${file.name}: ${String(fileErr)}`);
        }
      }

      if (allSamples.length === 0) {
        warning(
          t('laboratoryPanel.setup.notifications.noSamples'),
          fileErrors.length > 0
            ? fileErrors.join('\n')
            : t('laboratoryPanel.setup.notifications.noSamplesDetail', { count: filtered.length }),
        );
        return;
      }

      const reindexed = allSamples.map((s, i) => ({ ...s, id: `s_${Date.now()}_${i}`, index: i }));
      setSamples(reindexed);
      const ds = datasets.find(d => d.id === selectedSampleDatasetId);
      setSourceFileName(ds?.name ?? 'Dataset');
      success(
        t('laboratoryPanel.setup.notifications.loadSuccess'),
        t('laboratoryPanel.setup.notifications.loadSuccessDetail', { count: reindexed.length, fileCount: filtered.length }),
      );
    } catch (e) {
      console.error('[Lab] handleLoadFromDataset Fehler:', e);
      error(t('laboratoryPanel.setup.notifications.loadError'), String(e));
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
      if (parsed.length === 0) { warning(t('laboratoryPanel.setup.notifications.fileEmpty'), t('laboratoryPanel.setup.notifications.fileEmptyDetail')); return; }
      setSamples(parsed);
      setSourceFileName(file.name);
      success(
        t('laboratoryPanel.setup.notifications.fileLoadSuccess'),
        t('laboratoryPanel.setup.notifications.fileLoadSuccessDetail', { count: parsed.length, name: file.name }),
      );
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // ── Start Session ────────────────────────────────────────────────────────

  const handleStartSession = () => {
    if (!selectedModel || !selectedVersionId || samples.length === 0) return;
    if (engineMode === 'engine' && !detectedPlugin) { warning(t('laboratoryPanel.setup.notifications.engineNotSupported'), t('laboratoryPanel.setup.notifications.engineNotSupportedDetail')); return; }
    if (engineMode === 'dev' && !devScript.trim()) { warning(t('laboratoryPanel.setup.notifications.noScript'), t('laboratoryPanel.setup.notifications.noScriptDetail')); return; }

    const newSession: LabSession = {
      id: `lab_${Date.now()}`,
      name: `${selectedModel.name} – ${sourceFileName} (${new Date().toLocaleDateString(dateLocale(language))})`,
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
    unlistenRef.current.forEach(fn => { try { fn(); } catch { /* listener already removed */ } });
    unlistenRef.current = [];

    const start = Date.now();

    try {
      if (engineMode === 'engine') {
        // ── Persistenter Model-Server: direkt invoke, kein Event-Listener nötig ──
        if (serverStatus !== 'ready') {
            setTestError(
              serverStatus === 'loading'
                ? t('laboratoryPanel.testing.serverBannerLoading')
                : t('laboratoryPanel.testing.serverBannerIdle')
            );
          setTesting(false);
          return;
        }

        const result = await invoke<{
          predicted: string;
          confidence?: number;
          top_predictions?: TopPred[];
          inference_ms: number;
        }>('lab_infer_sample', {
          text: currentSample.text,
          filePath: currentSample.fileKind ? currentSample.filePath ?? null : null,
        });

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
          LAB_IMAGE_PATH: currentSample.fileKind === 'image' ? currentSample.filePath ?? '' : '',
          LAB_FILE_PATH:  currentSample.fileKind ? currentSample.filePath ?? '' : '',
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
  }, [currentSample, selectedVersionId, engineMode, devScript, modelPath, dsRefs, testing, t]);

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
    saveSession(updatedSession, userId);

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
    saveSession(updatedSession, userId);
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
          <h1 className="text-2xl font-bold text-white">{t('laboratoryPanel.title')}</h1>
          <p className="text-gray-400 mt-1">{t('laboratoryPanel.subtitle')}</p>
        </div>
        <div className="flex items-center gap-2">
          {phase === 'testing' && session && (
            <button onClick={() => setPhase('analysis')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all">
              <BarChart3 className="w-4 h-4" /> {t('laboratoryPanel.header.analysisButton')}
            </button>
          )}
          {phase === 'analysis' && session && (
            <button onClick={() => setPhase('testing')} disabled={currentSampleIdx >= samples.length} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all disabled:opacity-40">
              <Play className="w-4 h-4" /> {t('laboratoryPanel.header.continueTestingButton')}
            </button>
          )}
          <button onClick={() => setShowSessions(true)} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-500/10 hover:bg-pink-500/20 border border-pink-500/20 text-pink-300 text-sm transition-all">
            <FolderOpen className="w-4 h-4" /> {t('laboratoryPanel.header.sessionsButton')}
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
                  <span className="text-white font-semibold text-sm">{t('laboratoryPanel.setup.title')}</span>
                  {phase === 'testing' && selectedModel && (
                    <p className="text-gray-500 text-xs">{t('laboratoryPanel.setup.configSummary', {
                      model: selectedModel.name,
                      engine: engineMode === 'engine' ? t('laboratoryPanel.sessionsModal.engineBadge') : t('laboratoryPanel.sessionsModal.devScriptBadge'),
                      count: samples.length,
                      file: sourceFileName,
                    })}</p>
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
                    <p className="text-white font-medium">{t('laboratoryPanel.setup.noModelsTitle')}</p>
                    <p className="text-gray-500 text-sm">{t('laboratoryPanel.setup.noModelsDesc')}</p>
                  </div>
                ) : (
                  <>
                    {/* Model + Version */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-white">{t('laboratoryPanel.setup.modelLabel')}</label>
                        <select value={selectedModelId ?? ''} onChange={e => setSelectedModelId(e.target.value)}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                          {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-900">{m.name}</option>)}
                        </select>
                      </div>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-white">{t('laboratoryPanel.setup.versionLabel')}</label>
                        <select value={selectedVersionId ?? ''} onChange={e => setSelectedVersionId(e.target.value)}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                          {selectedModelTree?.versions?.length
                            ? [...selectedModelTree.versions].sort((a, b) => b.version_number - a.version_number).map((v, i) => (
                                <option key={v.id} value={v.id} className="bg-slate-900">{v.name}{i === 0 ? t('laboratoryPanel.setup.versionLatest') : ''}</option>
                              ))
                            : <option value="">{t('laboratoryPanel.setup.noVersions')}</option>}
                        </select>
                      </div>
                    </div>

                    {/* Engine Toggle */}
                    <div className="space-y-3">
                      <label className="block text-sm font-medium text-white">{t('laboratoryPanel.setup.engineLabel')}</label>
                      <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
                        {([['engine', t('laboratoryPanel.setup.engineStandard'), Play, 'amber'], ['dev', t('laboratoryPanel.setup.engineDev'), Code2, 'blue']] as const).map(([val, label, Icon, col]) => (
                          <button key={val} onClick={() => setEngineMode(val as typeof engineMode)}
                            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-medium transition-all ${engineMode === val ? (col === 'amber' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-blue-500/20 text-blue-300 border border-blue-500/30') : 'text-gray-400 hover:text-white'}`}>
                            <Icon className="w-3.5 h-3.5" />{label}
                          </button>
                        ))}
                      </div>

                      {engineMode === 'engine' && selectedModel && !detectedPlugin && (
                        <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                          <span className="text-amber-300 text-xs">{t('laboratoryPanel.setup.engineNotSupported')}</span>
                        </div>
                      )}
                      {engineMode === 'engine' && detectedPlugin && pluginUnsupportedInLab && (
                        <div className="flex items-start gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                          <span className="text-amber-300 text-xs">
                            {t('laboratoryPanel.setup.pluginNotInLab', { name: detectedPlugin.name })}
                          </span>
                        </div>
                      )}
                      {engineMode === 'engine' && detectedPlugin && !pluginUnsupportedInLab && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
                            <CheckCircle className="w-3.5 h-3.5 text-amber-400" />
                            <span className="text-amber-300 text-xs">{t('laboratoryPanel.setup.pluginDetected', { name: detectedPlugin.name })}</span>
                          </div>
                          {/* Modell laden — erst nach expliziter Bestätigung */}
                          {serverStatus !== 'loading' && !(serverStatus === 'ready' && serverVersionId === selectedVersionId) && selectedVersionId && (
                            <button
                              onClick={handleLoadModel}
                              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-pink-500/15 hover:bg-pink-500/25 border border-pink-500/30 text-pink-300 text-xs font-medium transition-all"
                            >
                              <Zap className="w-3.5 h-3.5" />
                              {serverStatus === 'error'
                                ? t('laboratoryPanel.setup.loadModelRetryButton')
                                : t('laboratoryPanel.setup.loadModelButton')}
                            </button>
                          )}
                          {/* Server-Status Badge */}
                          {serverStatus === 'loading' && (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-blue-500/10 border border-blue-500/20">
                              <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin flex-shrink-0" />
                              <span className="text-blue-300 text-xs">{t('laboratoryPanel.setup.serverLoading')}</span>
                            </div>
                          )}
                          {serverStatus === 'ready' && serverVersionId === selectedVersionId && (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                              <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                              <span className="text-emerald-300 text-xs">{t('laboratoryPanel.setup.serverReady')}</span>
                            </div>
                          )}
                          {serverStatus === 'error' && (
                            <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20">
                              <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                              <span className="text-red-300 text-xs whitespace-pre-wrap">
                                {serverErrorMsg ?? t('laboratoryPanel.setup.serverError')}
                              </span>
                            </div>
                          )}
                          {/* Canvas-Modell: Eingabe-Hinweis */}
                          {selectedModelId?.startsWith('canvas_') && (
                            <p className="text-[10px] text-gray-500 px-1">
                              {t('laboratoryPanel.setup.canvasInputHint')}
                            </p>
                          )}
                        </div>
                      )}

                      {engineMode === 'dev' && (
                        <div className="space-y-2">
                          {/* Script-Quellen: eigene Bibliothek + Open Library */}
                          <div className="flex gap-2">
                            <button
                              onClick={() => { setMyScripts(loadAllSavedScripts(userId)); setShowMyScripts(true); }}
                              className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 hover:text-white text-xs transition-all"
                            >
                              <FolderOpen className="w-3.5 h-3.5" /> {t('laboratoryPanel.devScripts.myScriptsButton')}
                            </button>
                            <button
                              onClick={() => setShowOpenLib(true)}
                              className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 hover:text-white text-xs transition-all"
                            >
                              <Download className="w-3.5 h-3.5" /> {t('laboratoryPanel.devScripts.openLibButton')}
                            </button>
                          </div>
                          <DevScriptEditor
                            script={devScript} onChange={setDevScript}
                            modelPath={modelPath} datasets={dsRefs} outputPath={outputPath}
                          />
                        </div>
                      )}
                    </div>

                    {/* Dataset Sample-Auswahl */}
                    <div className="space-y-3">
                      <label className="block text-sm font-medium text-white">{t('laboratoryPanel.setup.datasetLabel')}</label>
                      {datasets.length === 0 ? (
                        <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                          <span className="text-amber-300 text-xs">{t('laboratoryPanel.setup.noDatasetWarning')}</span>
                        </div>
                      ) : (
                        <select
                          value={selectedSampleDatasetId ?? ''}
                          onChange={e => { setSelectedSampleDatasetId(e.target.value || null); setSamples([]); setSourceFileName(''); }}
                          className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none"
                        >
                          <option value="" className="bg-slate-900">{t('laboratoryPanel.setup.datasetPlaceholder')}</option>
                          {datasets.map(d => (
                            <option key={d.id} value={d.id} className="bg-slate-900">
                              {d.name} ({t('laboratoryPanel.setup.datasetFileCount', { count: d.file_count })}{d.status === 'split' ? t('laboratoryPanel.setup.datasetSplitSuffix') : ''})
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
                                {split === 'all' ? t('laboratoryPanel.setup.splitAll') : split.charAt(0).toUpperCase() + split.slice(1)}
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
                              ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('laboratoryPanel.setup.loadingSamplesButton')}</>
                              : <><FolderOpen className="w-4 h-4" /> {t('laboratoryPanel.setup.loadSamplesButton')}</>}
                          </button>
                        </>
                      )}

                      {samples.length > 0 && (
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-pink-500/15 border border-pink-500/20 w-full">
                          <CheckCircle className="w-3.5 h-3.5 text-pink-400 flex-shrink-0" />
                          <span className="text-pink-300 text-xs font-medium flex-1">{t('laboratoryPanel.setup.samplesLoaded', { count: samples.length, file: sourceFileName })}</span>
                          <button onClick={() => { setSamples([]); setSourceFileName(''); }} className="text-gray-500 hover:text-red-400 transition-colors"><X className="w-3.5 h-3.5" /></button>
                        </div>
                      )}
                    </div>

                    {/* Samples Preview */}
                    {samples.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-white">{t('laboratoryPanel.setup.samplesPreviewTitle', { count: samples.length })}</p>
                          <button onClick={() => { setSamples([]); setSourceFileName(''); }} className="text-xs text-gray-500 hover:text-red-400 transition-colors">{t('laboratoryPanel.setup.samplesRemove')}</button>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-black/20 divide-y divide-white/5 max-h-40 overflow-y-auto">
                          {samples.slice(0, 8).map((s, i) => (
                            <div key={s.id} className="flex items-center gap-3 px-3 py-2">
                              <span className="text-gray-600 text-[10px] tabular-nums w-4 flex-shrink-0">{i + 1}</span>
                              <span className="text-gray-300 text-xs flex-1 truncate">{s.text}</span>
                              {s.label && <span className="text-gray-500 text-[10px] flex-shrink-0 px-1.5 py-0.5 rounded bg-white/5">{s.label}</span>}
                            </div>
                          ))}
                          {samples.length > 8 && <div className="px-3 py-2 text-gray-600 text-xs">{t('laboratoryPanel.setup.samplesPreviewMore', { count: samples.length - 8 })}</div>}
                        </div>
                      </div>
                    )}

                    {/* Start Button */}
                    <button
                      onClick={handleStartSession}
                      disabled={!selectedModel || !selectedVersionId || samples.length === 0 || (engineMode === 'engine' && !detectedPlugin) || (engineMode === 'dev' && !devScript.trim())}
                      className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 hover:opacity-90 text-white font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg"
                    >
                      <FlaskConical className="w-4 h-4" /> {t('laboratoryPanel.setup.startButton', { count: samples.length })}
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
                      ? t('laboratoryPanel.testing.serverBannerLoading')
                      : serverStatus === 'error'
                      ? serverErrorMsg ?? t('laboratoryPanel.testing.serverBannerError')
                      : t('laboratoryPanel.testing.serverBannerIdle')}
                  </span>
                  {(serverStatus === 'idle' || serverStatus === 'error') && selectedVersionId && (
                    <button
                      onClick={handleLoadModel}
                      className="ml-auto flex-shrink-0 px-3 py-1.5 rounded-lg bg-pink-500/15 hover:bg-pink-500/25 border border-pink-500/30 text-pink-300 text-xs font-medium transition-all"
                    >
                      {t('laboratoryPanel.setup.loadModelButton')}
                    </button>
                  )}
                </div>
              )}

              {/* Eingabeart passt nicht zum Modell */}
              {engineMode === 'engine' && serverStatus === 'ready' && inputMismatch && (
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl border text-sm bg-amber-500/10 border-amber-500/20 text-amber-300">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{t(`laboratoryPanel.testing.${inputMismatch}`)}</span>
                </div>
              )}

              {/* Progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t('laboratoryPanel.testing.progressLabel', { current: currentSampleIdx + 1, total: samples.length })}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-emerald-400 inline-flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" />{results.filter(r => r.userRating === 'correct').length}</span>
                    <span className="text-red-400 inline-flex items-center gap-1"><XCircle className="w-3.5 h-3.5" />{results.filter(r => r.userRating === 'wrong').length}</span>
                    <span className="text-gray-500 inline-flex items-center gap-1"><SkipForward className="w-3.5 h-3.5" />{results.filter(r => r.userRating === 'skipped').length}</span>
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
                    <span className="text-gray-400 text-[10px] tabular-nums font-medium">{t('laboratoryPanel.testing.sampleTitle', { index: currentSample.index + 1, total: samples.length })}</span>
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
                    <span className="text-white font-medium text-sm">{t('laboratoryPanel.testing.sampleCardTitle', { index: currentSample.index + 1 })}</span>
                  </div>

                  {currentSample.fileKind && currentSample.filePath ? (
                    <div className="rounded-xl bg-black/30 border border-white/10 p-3 flex flex-col items-center gap-2">
                      {currentSample.fileKind === 'image' ? (
                        <img
                          src={convertFileSrc(currentSample.filePath)}
                          alt={currentSample.text}
                          className="max-h-40 max-w-full rounded-lg object-contain"
                        />
                      ) : (
                        <audio
                          controls
                          src={convertFileSrc(currentSample.filePath)}
                          className="w-full"
                        />
                      )}
                      <span className="text-gray-400 text-[10px] font-mono truncate max-w-full">{currentSample.text}</span>
                    </div>
                  ) : (
                    <div className="rounded-xl bg-black/30 border border-white/10 p-3 max-h-36 overflow-y-auto">
                      <p className="text-gray-200 text-xs leading-relaxed whitespace-pre-wrap">{getDisplayText(currentSample)}</p>
                    </div>
                  )}

                  {/* Rohdaten (aufklappbar) */}
                  {typeof currentSample.rawData === 'object' && currentSample.rawData !== null && Object.keys(currentSample.rawData as object).length > 1 && (
                    <details className="group">
                      <summary className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 cursor-pointer list-none">
                        <Eye className="w-3 h-3" /> {t('laboratoryPanel.testing.rawDataToggle')}
                      </summary>
                      <pre className="mt-2 text-[10px] text-gray-500 font-mono overflow-x-auto bg-black/20 rounded-lg p-2 max-h-24">{JSON.stringify(currentSample.rawData, null, 2)}</pre>
                    </details>
                  )}

                  {/* Test-Button */}
                  {!alreadyRated ? (
                    <button onClick={handleRunTest} disabled={testing}
                      className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 hover:opacity-90 text-white font-semibold text-sm transition-all disabled:opacity-60">
                      {testing ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('laboratoryPanel.testing.testingButton')}</> : <><Play className="w-4 h-4" /> {t('laboratoryPanel.testing.testButton')}</>}
                    </button>
                  ) : (
                    <div className="flex items-center gap-2 justify-center py-2 text-xs text-gray-500">
                      <CheckCircle className="w-3.5 h-3.5" /> {t('laboratoryPanel.testing.alreadyRatedLabel')} <strong className="text-white">{alreadyRated.userRating}</strong>
                    </div>
                  )}

                  <button onClick={handleSkipWithoutTest} className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-500 hover:text-gray-300 text-xs transition-all">
                    <SkipForward className="w-3.5 h-3.5" /> {t('laboratoryPanel.testing.skipButton')}
                  </button>
                </div>

                {/* Right: Result + Rating */}
                <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-amber-400" />
                    <span className="text-white font-medium text-sm">{t('laboratoryPanel.testing.resultTitle')}</span>
                  </div>

                  {/* Idle State */}
                  {!testResult && !testError && !testing && (
                    <div className="flex flex-col items-center justify-center py-8 gap-3">
                      <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center">
                        <Play className="w-5 h-5 text-gray-600" />
                      </div>
                      <p className="text-gray-600 text-sm">{t('laboratoryPanel.testing.idleHint')}</p>
                    </div>
                  )}

                  {/* Loading */}
                  {testing && (
                    <div className="flex flex-col items-center justify-center py-8 gap-3">
                      <Loader2 className="w-8 h-8 text-pink-400 animate-spin" />
                      <p className="text-gray-400 text-sm">{t('laboratoryPanel.testing.inferenceRunning')}</p>
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
                            ? <><CheckCircle className="w-3.5 h-3.5" /> {t('laboratoryPanel.testing.matchLabel')}</>
                            : <><XCircle className="w-3.5 h-3.5" /> {t('laboratoryPanel.testing.mismatchLabel')} <strong>{currentSample.label}</strong></>}
                      </div>
                      )}

                      {/* Top Predictions */}
                      {testResult.topPredictions && testResult.topPredictions.length > 1 && (
                        <div className="space-y-1.5">
                          <p className="text-xs text-gray-500">{t('laboratoryPanel.testing.allClassesLabel')}</p>
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
                          <textarea value={userNote} onChange={e => setUserNote(e.target.value)} rows={2} placeholder={t('laboratoryPanel.testing.notePlaceholder')}
                            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder:text-gray-600 focus:outline-none focus:border-white/20 resize-none" />
                        ) : (
                          <button onClick={() => setShowNote(true)} className="text-xs text-gray-600 hover:text-gray-400 transition-colors flex items-center gap-1">
                            <Pencil className="w-3 h-3" /> {t('laboratoryPanel.testing.addNoteButton')}
                          </button>
                        )}
                      </div>

                      {/* Rating Buttons */}
                      <div className="flex gap-2 pt-1">
                        <button onClick={() => handleRate('correct')} className="flex-1 flex items-center justify-center gap-1.5 py-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 font-semibold text-sm transition-all">
                          <ThumbsUp className="w-4 h-4" /> {t('laboratoryPanel.testing.correctButton')}
                        </button>
                        <button onClick={() => handleRate('wrong')} className="flex-1 flex items-center justify-center gap-1.5 py-3 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-sm transition-all">
                          <ThumbsDown className="w-4 h-4" /> {t('laboratoryPanel.testing.wrongButton')}
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
                  <AccuracyDonut
                    correct={results.filter(r => r.userRating === 'correct').length}
                    wrong={results.filter(r => r.userRating === 'wrong').length}
                    skipped={results.filter(r => r.userRating === 'skipped').length}
                    centerLabel={t('laboratoryPanel.analysis.donutCenterLabel')}
                    labels={{
                      correct: t('laboratoryPanel.analysis.donutCorrectLabel'),
                      wrong: t('laboratoryPanel.analysis.donutWrongLabel'),
                      skipped: t('laboratoryPanel.analysis.donutSkipLabel'),
                    }}
                  />
                  <div className="flex-1 space-y-1">
                    <p className="text-white text-sm font-medium">{t('laboratoryPanel.testing.sessionSummaryTitle')}</p>
                    <div className="flex items-center gap-4 text-xs">
                      <span className="text-emerald-400 inline-flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5" />{t('laboratoryPanel.testing.correctLabel', { count: results.filter(r => r.userRating === 'correct').length })}</span>
                      <span className="text-red-400 inline-flex items-center gap-1"><XCircle className="w-3.5 h-3.5" />{t('laboratoryPanel.testing.wrongLabel', { count: results.filter(r => r.userRating === 'wrong').length })}</span>
                      <span className="text-gray-500 inline-flex items-center gap-1"><SkipForward className="w-3.5 h-3.5" />{t('laboratoryPanel.testing.skippedLabel', { count: results.filter(r => r.userRating === 'skipped').length })}</span>
                    </div>
                  </div>
                  <button onClick={() => setPhase('analysis')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-500/10 hover:bg-pink-500/20 border border-pink-500/20 text-pink-300 text-xs font-medium transition-all">
                    <BarChart3 className="w-3.5 h-3.5" /> {t('laboratoryPanel.testing.openAnalysisButton')}
                  </button>
                </div>
              )}
            </>
          )}

          {/* ── Keine Samples geladen (z.B. nach Modellwechsel) ── */}
          {phase === 'testing' && session && !currentSample && samples.length === 0 && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl border text-sm bg-white/5 border-white/10 text-gray-400">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{t('laboratoryPanel.testing.noSamplesLoaded')}</span>
            </div>
          )}

          {/* ── Abgeschlossen ── */}
          {phase === 'testing' && session && !currentSample && samples.length > 0 && (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-8 text-center space-y-3">
              <CheckCircle className="w-10 h-10 text-emerald-400 mx-auto" />
              <p className="text-white font-semibold">{t('laboratoryPanel.testing.allDoneTitle')}</p>
              <button onClick={() => setPhase('analysis')} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-pink-500/20 hover:bg-pink-500/30 border border-pink-500/30 text-pink-300 font-medium text-sm transition-all">
                <BarChart3 className="w-4 h-4" /> {t('laboratoryPanel.testing.toAnalysisButton')}
              </button>
            </div>
          )}
        </>
      )}

      {/* Sessions Modal */}
      {showSessions && <SessionsModal onLoad={handleLoadSession} onClose={() => setShowSessions(false)} userId={userId} />}

      {/* ── Eigene Dev-Scripts (aus DevTrain/DevTest gespeichert) ── */}
      {showMyScripts && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowMyScripts(false)}>
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[75vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 flex-shrink-0">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-pink-400" />
                <h2 className="text-base font-bold text-white">{t('laboratoryPanel.devScripts.modalTitle')}</h2>
              </div>
              <button onClick={() => setShowMyScripts(false)} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {myScripts.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-10 leading-relaxed">
                  {t('laboratoryPanel.devScripts.empty')}
                </p>
              ) : myScripts.map(s => (
                <div key={s.id} className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 hover:border-white/20 transition-all">
                  <span className={`flex-shrink-0 text-[10px] px-1.5 py-0.5 rounded-md border ${
                    s.source === 'train'
                      ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                      : 'bg-blue-500/10 border-blue-500/25 text-blue-300'
                  }`}>
                    {s.source === 'train' ? 'Train' : 'Test'}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-sm truncate">{s.name}</p>
                    <p className="text-gray-600 text-[10px]">{new Date(s.savedAt).toLocaleString(dateLocale(language))}</p>
                  </div>
                  <button
                    onClick={() => {
                      setDevScript(s.script);
                      setShowMyScripts(false);
                      success(t('laboratoryPanel.devScripts.loadedToast'), s.name);
                    }}
                    className="flex-shrink-0 px-3 py-1.5 rounded-lg bg-pink-500/15 hover:bg-pink-500/25 border border-pink-500/30 text-pink-300 text-xs font-medium transition-all"
                  >
                    {t('laboratoryPanel.devScripts.loadButton')}
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Open Library: Community-Scripts ── */}
      {showOpenLib && (
        <OpenLibraryModal
          mode="test"
          onClose={() => setShowOpenLib(false)}
          onLoadScript={(content, name) => {
            setDevScript(content);
            setShowOpenLib(false);
            success(t('laboratoryPanel.devScripts.loadedToast'), name);
          }}
        />
      )}
    </div>
  );
}
