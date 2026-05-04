// AnalysisPanel.tsx – Vollständige Trainingsanalyse mit KI-Integration
// Nutzt globale AISettings aus dem Context (konfigurierbar in Einstellungen → KI-Assistent)

import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  TrendingDown, Activity, Target, Clock, Layers,
  RefreshCw, Loader2, ChevronDown, ChevronUp, AlertCircle, Download,
  GitBranch, CheckCircle, FileText, Brain, Sparkles, MessageSquare,
  Send, Trash2, RotateCcw, Bot, User, Cpu, Database,
  Save, BookOpen, Zap, Info,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }
interface VersionTreeItem { id: string; name: string; is_root: boolean; version_number: number; }
interface TrainingMetrics {
  id: string; version_id: string; final_train_loss: number; final_val_loss: number | null;
  total_epochs: number; total_steps: number; best_epoch: number | null;
  training_duration_seconds: number | null; created_at: string;
}
interface LogEntry {
  epoch: number; step: number; train_loss: number; val_loss: number | null;
  learning_rate: number; grad_norm?: number | null; elapsed_seconds?: number; timestamp: string;
}
interface EpochSummary {
  epoch: number; avg_train_loss: number | null; min_train_loss: number | null;
  max_train_loss: number | null; val_loss: number | null; duration_seconds: number; steps: number;
}
interface VersionDetails {
  id: string; model_id: string; version_name: string; version_number: number;
  path: string; size_bytes: number; file_count: number; created_at: string;
  is_root: boolean; parent_version_id: string | null;
}
interface FullTrainingData {
  exported_at: string; training_summary: Record<string, any>; config: Record<string, any>;
  hardware: Record<string, any>; model_info: Record<string, any>; dataset_info: Record<string, any>;
  epoch_summaries: EpochSummary[]; step_logs: LogEntry[]; derived_stats: Record<string, any>;
}
interface AIAnalysisReport { version_id: string; report_text: string; provider: string; model: string; generated_at: string; }
interface ChatMessage { role: 'user' | 'assistant'; content: string; }
interface MetricsTemplate { id: string; name: string; description: string; config: Record<string, any>; created_at: string; source: string; }

// ─────────────────────────────────────────────────────────────────────────────
// AI Provider Layer
// ─────────────────────────────────────────────────────────────────────────────

type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';
const PROVIDER_META: Record<AIProvider, { label: string; needsKey: boolean; endpoint: string; authHeader: (k: string) => Record<string, string>; buildBody: (m: string, msgs: ChatMessage[], sys: string) => object; extractText: (d: any) => string; }> = {
  anthropic: { label: 'Claude (Anthropic)', needsKey: true, endpoint: 'https://api.anthropic.com/v1/messages', authHeader: k => ({ 'x-api-key': k, 'anthropic-version': '2023-06-01' }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, system: sys, messages: msgs }), extractText: d => d.content?.[0]?.text || '' },
  openai: { label: 'GPT-4o (OpenAI)', needsKey: true, endpoint: 'https://api.openai.com/v1/chat/completions', authHeader: k => ({ Authorization: `Bearer ${k}` }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.choices?.[0]?.message?.content || '' },
  groq: { label: 'Groq', needsKey: true, endpoint: 'https://api.groq.com/openai/v1/chat/completions', authHeader: k => ({ Authorization: `Bearer ${k}` }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.choices?.[0]?.message?.content || '' },
  ollama: { label: 'Ollama (Lokal)', needsKey: false, endpoint: 'http://localhost:11434/api/chat', authHeader: () => ({}), buildBody: (m, msgs, sys) => ({ model: m, stream: false, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.message?.content || '' },
};
async function callAI(provider: AIProvider, apiKey: string, model: string, messages: ChatMessage[], systemPrompt: string): Promise<string> {
  const meta = PROVIDER_META[provider];
  const resp = await fetch(meta.endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', ...meta.authHeader(apiKey) }, body: JSON.stringify(meta.buildBody(model, messages, systemPrompt)) });
  if (!resp.ok) throw new Error(`API-Fehler (${resp.status}): ${await resp.text()}`);
  const data = await resp.json();
  const text = meta.extractText(data);
  if (!text) throw new Error('Leere Antwort vom KI-Modell');
  return text;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function formatDuration(s: number | null) {
  if (!s) return '-';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function formatDate(d: string) {
  return new Date(d).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function formatBytes(b: number) {
  if (!b) return '0 B';
  const k = 1024, s = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(b) / Math.log(k));
  return parseFloat((b / Math.pow(k, i)).toFixed(2)) + ' ' + s[i];
}
function extractAIRecommendedParams(reportText: string): Record<string, any> | null {
  const match = reportText.match(/```json\s*([\s\S]*?)```/);
  if (!match) return null;
  try { return JSON.parse(match[1].trim()); } catch { return null; }
}
function niceY(min: number, max: number, ticks = 4): number[] {
  const range = max - min || 1; const step = range / ticks; const result: number[] = [];
  for (let i = 0; i <= ticks; i++) result.push(min + i * step);
  return result;
}

// ─── Smoothing: Simple Moving Average ────────────────────────────────────
function smoothArray(arr: number[], windowSize: number = 3): number[] {
  if (windowSize < 1) return arr;
  const result: number[] = [];
  const hw = Math.floor(windowSize / 2);
  for (let i = 0; i < arr.length; i++) {
    const start = Math.max(0, i - hw);
    const end = Math.min(arr.length, i + hw + 1);
    const window = arr.slice(start, end);
    result.push(window.reduce((a, b) => a + b, 0) / window.length);
  }
  return result;
}

// ─── PNG Export ──────────────────────────────────────────────────────────
async function downloadSvgAsPng(svgElement: SVGSVGElement, filename: string = 'chart.png'): Promise<void> {
  return new Promise((resolve, reject) => {
    try {
      const rect = svgElement.getBoundingClientRect();
      const width = Math.ceil(rect.width) || 560;
      const height = Math.ceil(rect.height) || 220;

      const svgClone = svgElement.cloneNode(true) as SVGSVGElement;
      svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      svgClone.setAttribute('width', width.toString());
      svgClone.setAttribute('height', height.toString());
      svgClone.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svgClone.removeAttribute('style');
      svgClone.setAttribute('style', `background-color: #1e293b; width: ${width}px; height: ${height}px;`);

      const svgString = new XMLSerializer().serializeToString(svgClone);
      let finalSvg = svgString;
      if (!finalSvg.includes('xmlns')) {
        finalSvg = finalSvg.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) { reject(new Error('Canvas context nicht verfügbar')); return; }

      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, width, height);

      try {
        const img = new Image();
        const blob = new Blob([finalSvg], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        let loaded = false;
        const timeout = setTimeout(() => {
          if (!loaded) {
            URL.revokeObjectURL(url);
            try {
              const encoded = 'data:image/svg+xml;base64,' + btoa(finalSvg);
              const img2 = new Image();
              img2.onload = () => drawAndSave(img2, ctx, canvas, width, height, filename, resolve, reject);
              img2.onerror = () => reject(new Error('SVG Bild konnte nicht geladen werden'));
              img2.src = encoded;
            } catch (e) { reject(new Error(`Base64 Fehler: ${String(e)}`)); }
          }
        }, 2000);
        img.onload = () => { loaded = true; clearTimeout(timeout); URL.revokeObjectURL(url); drawAndSave(img, ctx, canvas, width, height, filename, resolve, reject); };
        img.onerror = () => { loaded = true; clearTimeout(timeout); URL.revokeObjectURL(url); };
        img.src = url;
      } catch (e) { reject(new Error(`SVG Zeichnung fehlgeschlagen: ${String(e)}`)); }
    } catch (err) { reject(new Error(`Export Error: ${String(err)}`)); }
  });
}

function drawAndSave(
  img: HTMLImageElement,
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  filename: string,
  resolve: (value: void) => void,
  reject: (reason?: any) => void
) {
  try {
    ctx.drawImage(img, 0, 0, width, height);
    canvas.toBlob((blob) => {
      if (!blob) { reject(new Error('PNG Erstellung fehlgeschlagen')); return; }
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url; link.download = filename;
      document.body.appendChild(link); link.click();
      setTimeout(() => { document.body.removeChild(link); URL.revokeObjectURL(url); resolve(); }, 100);
    }, 'image/png', 1.0);
  } catch (err) { reject(new Error(`Canvas Fehler: ${String(err)}`)); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart Components – vollständige SVG-Charts mit Y-Achse, Fills, Epoch-Marker
// ─────────────────────────────────────────────────────────────────────────────

// ── 1. Großer Loss-Verlauf (Step-Level, Train + Val parallel) ────────────────

function BigLossChart({ logs, label, enableSmoothing = false }: { logs: LogEntry[]; label: string; enableSmoothing?: boolean }) {
  const { success, error } = useNotification();
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  if (logs.length < 2) return null;
  const W = 560; const H = 200;
  const PAD = { l: 56, r: 16, t: 18, b: 36 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  let trains = logs.map(l => l.train_loss);
  let valArr = logs.map(l => l.val_loss).filter((v): v is number => v != null);

  if (enableSmoothing && logs.length > 5) {
    trains = smoothArray(trains, Math.max(3, Math.floor(logs.length / 20)));
    if (valArr.length > 5) {
      const valWithNulls = logs.map(l => l.val_loss ?? 0);
      const smoothedVal = smoothArray(valWithNulls, Math.max(3, Math.floor(logs.length / 20)));
      valArr = smoothedVal.filter((_, i) => logs[i].val_loss != null);
    }
  }

  const all = [...trains, ...valArr];
  const minV = Math.min(...all) * 0.97, maxV = Math.max(...all) * 1.03;
  const toX = (i: number) => PAD.l + (i / Math.max(logs.length - 1, 1)) * iW;
  const toY = (v: number) => PAD.t + iH - ((v - minV) / (maxV - minV)) * iH;
  const trainPath = logs.map((l, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(l.train_loss).toFixed(1)}`).join(' ');
  const trainArea = `${trainPath} L${toX(logs.length - 1).toFixed(1)},${(PAD.t + iH).toFixed(1)} L${PAD.l},${(PAD.t + iH).toFixed(1)} Z`;
  const valPts = logs.filter(l => l.val_loss != null);
  const valPath = valPts.map((l, idx) => `${idx === 0 ? 'M' : 'L'}${toX(logs.indexOf(l)).toFixed(1)},${toY(l.val_loss!).toFixed(1)}`).join(' ');
  const epochChanges = logs.map((l, i) => ({ i, epoch: l.epoch })).filter((x, idx) => idx > 0 && x.epoch !== logs[idx - 1].epoch);
  const yTicks = niceY(minV, maxV, 4);
  const last = logs[logs.length - 1];
  const hoveredLog = hoveredStep !== null && hoveredStep >= 0 && hoveredStep < logs.length ? logs[hoveredStep] : null;

  const handleDownload = async () => {
    if (!svgRef.current) { error('Fehler', 'Chart-Element nicht gefunden'); return; }
    try {
      await downloadSvgAsPng(svgRef.current, 'loss-chart.png');
      success('Bild erfolgreich heruntergeladen! 🎉', 'loss-chart.png');
    } catch (err) { error('Download fehlgeschlagen', String(err)); }
  };

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4 col-span-full">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-300">{label}</span>
          {enableSmoothing && <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">🔄 Smoothing ON</span>}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-5 text-xs text-gray-400">
            <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 bg-emerald-400 rounded" /><span>Train Loss</span></div>
            {valArr.length > 0 && <div className="flex items-center gap-1.5"><div className="w-5 border-t border-dashed border-purple-400" /><span>Val Loss</span></div>}
          </div>
          <button onClick={handleDownload} className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all" title="Chart als PNG exportieren">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full relative" style={{ height: 220 }}
        onMouseMove={(e) => {
          const svg = e.currentTarget;
          const rect = svg.getBoundingClientRect();
          const x = (e.clientX - rect.left) / rect.width * W;
          const step = Math.round(Math.max(0, Math.min(logs.length - 1, (x - PAD.l) / iW * (logs.length - 1))));
          setHoveredStep(step);
        }}
        onMouseLeave={() => setHoveredStep(null)}
      >
        <defs>
          <linearGradient id="trainFillBig" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {yTicks.map((v, i) => {
          const y = toY(v);
          return (
            <g key={i}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
              <text x={PAD.l - 6} y={y + 3.5} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="9.5">{v.toFixed(3)}</text>
            </g>
          );
        })}
        {epochChanges.map(({ i, epoch }) => (
          <g key={epoch}>
            <line x1={toX(i)} x2={toX(i)} y1={PAD.t} y2={PAD.t + iH} stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="4,3" />
            <text x={toX(i)} y={H - 6} textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9">E{epoch}</text>
          </g>
        ))}
        <path d={trainArea} fill="url(#trainFillBig)" />
        <path d={trainPath} fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
        {valArr.length > 1 && <path d={valPath} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="6,3" strokeLinejoin="round" strokeLinecap="round" />}
        <circle cx={toX(logs.length - 1)} cy={toY(last.train_loss)} r="4.5" fill="#10b981" stroke="rgba(0,0,0,0.5)" strokeWidth="1.5" />
        {last.val_loss != null && <circle cx={toX(logs.length - 1)} cy={toY(last.val_loss)} r="4" fill="#a855f7" stroke="rgba(0,0,0,0.5)" strokeWidth="1.5" />}
        {hoveredLog && (
          <>
            <circle cx={toX(logs.indexOf(hoveredLog))} cy={toY(hoveredLog.train_loss)} r="5.5" fill="none" stroke="#10b981" strokeWidth="2" opacity="0.6" />
            {hoveredLog.val_loss != null && <circle cx={toX(logs.indexOf(hoveredLog))} cy={toY(hoveredLog.val_loss)} r="5.5" fill="none" stroke="#a855f7" strokeWidth="2" opacity="0.6" />}
          </>
        )}
        <text x={PAD.l} y={H - 6} textAnchor="start" fill="rgba(255,255,255,0.2)" fontSize="9">Step 1</text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" fill="rgba(255,255,255,0.2)" fontSize="9">Step {last.step}</text>
      </svg>
      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-2.5 flex items-center justify-between">
          <span className="text-gray-400">Train</span>
          <span className="text-emerald-400 font-mono font-semibold">{trains[0]?.toFixed(4)} → {trains[trains.length - 1]?.toFixed(4)}</span>
        </div>
        {valArr.length > 0 && (
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-2.5 flex items-center justify-between">
            <span className="text-gray-400">Val</span>
            <span className="text-purple-400 font-mono font-semibold">{valArr[0]?.toFixed(4)} → {valArr[valArr.length - 1]?.toFixed(4)}</span>
          </div>
        )}
      </div>
      {hoveredLog && (
        <div className="mt-3 p-2.5 bg-slate-900/40 border border-slate-700/50 rounded-lg flex items-center gap-3 text-xs">
          <div className="flex-1">
            <div className="text-gray-400 font-semibold">Step {hoveredLog.step} · Epoch {hoveredLog.epoch}</div>
            <div className="text-gray-500 text-xs mt-0.5">Train Loss: <span className="text-emerald-400 font-mono">{hoveredLog.train_loss.toFixed(4)}</span></div>
            {hoveredLog.val_loss != null && <div className="text-gray-500 text-xs">Val Loss: <span className="text-purple-400 font-mono">{hoveredLog.val_loss.toFixed(4)}</span></div>}
            {hoveredLog.learning_rate > 0 && <div className="text-gray-500 text-xs">LR: <span className="text-blue-400 font-mono">{hoveredLog.learning_rate.toExponential(2)}</span></div>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── 2. Epoch-Level: Train + Val Doppellinien mit Dots ────────────────────────

function EpochLossLineChart({ summaries, enableSmoothing = false }: { summaries: EpochSummary[]; enableSmoothing?: boolean }) {
  const { success, error } = useNotification();
  const svgRef = useRef<SVGSVGElement>(null);
  if (!summaries.length) return null;
  const W = 560; const H = 180;
  const PAD = { l: 56, r: 16, t: 18, b: 36 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  let trains = summaries.map(s => s.avg_train_loss ?? 0);
  let valPts = summaries.filter(s => s.val_loss != null);
  let valArr = valPts.map(s => s.val_loss!);

  if (enableSmoothing && summaries.length > 3) {
    trains = smoothArray(trains, Math.max(3, Math.floor(summaries.length / 3)));
    if (valArr.length > 3) {
      valArr = smoothArray(valArr, Math.max(3, Math.floor(valArr.length / 3)));
    }
  }

  const all = [...trains, ...valArr];
  const minV = Math.min(...all) * 0.96, maxV = Math.max(...all) * 1.04;
  const toX = (i: number, total: number) => PAD.l + (i / Math.max(total - 1, 1)) * iW;
  const toY = (v: number) => PAD.t + iH - ((v - minV) / (maxV - minV)) * iH;
  const trainPath = trains.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i, trains.length).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const valPath = valPts.map((s, i) => {
    const xi = summaries.findIndex(x => x.epoch === s.epoch);
    return `${i === 0 ? 'M' : 'L'}${toX(xi, summaries.length).toFixed(1)},${toY(s.val_loss!).toFixed(1)}`;
  }).join(' ');
  const trainArea = `${trainPath} L${toX(trains.length - 1, trains.length).toFixed(1)},${(PAD.t + iH).toFixed(1)} L${PAD.l},${(PAD.t + iH).toFixed(1)} Z`;
  const yTicks = niceY(minV, maxV, 4);

  const handleDownload = async () => {
    if (!svgRef.current) { error('Fehler', 'Chart-Element nicht gefunden'); return; }
    try {
      await downloadSvgAsPng(svgRef.current, 'epoch-loss-chart.png');
      success('Bild erfolgreich heruntergeladen! 🎉', 'epoch-loss-chart.png');
    } catch (err) { error('Download fehlgeschlagen', String(err)); }
  };

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300">Ø Loss pro Epoche</span>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-blue-400 rounded" /><span>Train</span></div>
            {valArr.length > 0 && <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-emerald-400 rounded" /><span>Val</span></div>}
          </div>
          <button onClick={handleDownload} className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all" title="Chart als PNG exportieren">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 200 }}>
        <defs>
          <linearGradient id="epochTrainFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#60a5fa" stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {yTicks.map((v, i) => {
          const y = toY(v);
          return (
            <g key={i}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
              <text x={PAD.l - 6} y={y + 3.5} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="9.5">{v.toFixed(3)}</text>
            </g>
          );
        })}
        {summaries.map((s, i) => (
          <text key={i} x={toX(i, summaries.length)} y={H - 6} textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9">E{s.epoch}</text>
        ))}
        <path d={trainArea} fill="url(#epochTrainFill)" />
        <path d={trainPath} fill="none" stroke="#60a5fa" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
        {valArr.length > 0 && <path d={valPath} fill="none" stroke="#34d399" strokeWidth="2" strokeDasharray="6,3" strokeLinejoin="round" strokeLinecap="round" />}
        {trains.map((v, i) => <circle key={i} cx={toX(i, trains.length)} cy={toY(v)} r="3.5" fill="#60a5fa" stroke="rgba(0,0,0,0.4)" strokeWidth="1" />)}
        {valPts.map((s, i) => {
          const xi = summaries.findIndex(x => x.epoch === s.epoch);
          return <circle key={i} cx={toX(xi, summaries.length)} cy={toY(s.val_loss!)} r="3.5" fill="#34d399" stroke="rgba(0,0,0,0.4)" strokeWidth="1" />;
        })}
      </svg>
    </div>
  );
}

// ── 3. Overfitting-Gap (Val – Train über Epochen) ────────────────────────────

function OverfittingGapChart({ summaries }: { summaries: EpochSummary[] }) {
  const { success, error } = useNotification();
  const svgRef = useRef<SVGSVGElement>(null);
  const paired = summaries.filter(s => s.val_loss != null && s.avg_train_loss != null);
  if (paired.length < 2) return null;
  const gaps = paired.map(s => s.val_loss! - s.avg_train_loss!);
  const maxG = Math.max(...gaps.map(Math.abs), 0.001) * 1.2;
  const W = 400; const H = 140;
  const PAD = { l: 52, r: 12, t: 16, b: 32 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const midY = PAD.t + iH / 2;
  const toX = (i: number) => PAD.l + (i / Math.max(gaps.length - 1, 1)) * iW;
  const toY = (v: number) => midY - (v / maxG) * (iH / 2);
  const linePath = gaps.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${toX(gaps.length - 1).toFixed(1)},${midY} L${PAD.l},${midY} Z`;
  const isOverfitting = gaps[gaps.length - 1] > maxG * 0.35;

  const handleDownload = async () => {
    if (!svgRef.current) { error('Fehler', 'Chart-Element nicht gefunden'); return; }
    try {
      await downloadSvgAsPng(svgRef.current, 'overfitting-gap-chart.png');
      success('Bild erfolgreich heruntergeladen! 🎉', 'overfitting-gap-chart.png');
    } catch (err) { error('Download fehlgeschlagen', String(err)); }
  };

  return (
    <div className={`rounded-xl border p-4 ${isOverfitting ? 'bg-amber-500/8 border-amber-500/25' : 'bg-white/5 border-white/10'}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
          Overfitting-Indikator (Val – Train)
          {isOverfitting && <span className="text-xs px-2 py-0.5 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full">⚠ Gap groß</span>}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Aktuell: <span className={isOverfitting ? 'text-amber-400 font-semibold' : 'text-gray-300'}>+{gaps[gaps.length - 1].toFixed(4)}</span></span>
          <button onClick={handleDownload} className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all" title="Chart als PNG exportieren">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 160 }}>
        <defs>
          <linearGradient id="gapFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={isOverfitting ? '#f59e0b' : '#34d399'} stopOpacity="0.28" />
            <stop offset="100%" stopColor={isOverfitting ? '#f59e0b' : '#34d399'} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <line x1={PAD.l} x2={W - PAD.r} y1={midY} y2={midY} stroke="rgba(255,255,255,0.18)" strokeWidth="1.5" />
        <text x={PAD.l - 6} y={midY + 3.5} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="9">0</text>
        {[0.25, 0.75].map(f => (
          <g key={f}>
            <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + iH * f} y2={PAD.t + iH * f} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <text x={PAD.l - 6} y={PAD.t + iH * f + 3.5} textAnchor="end" fill="rgba(255,255,255,0.2)" fontSize="8.5">
              {f < 0.5 ? `+${(maxG * (1 - f * 2)).toFixed(3)}` : `−${(maxG * (f * 2 - 1)).toFixed(3)}`}
            </text>
          </g>
        ))}
        <path d={areaPath} fill="url(#gapFill)" />
        <path d={linePath} fill="none" stroke={isOverfitting ? '#f59e0b' : '#34d399'} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
        {gaps.map((v, i) => <circle key={i} cx={toX(i)} cy={toY(v)} r="3" fill={isOverfitting ? '#f59e0b' : '#34d399'} stroke="rgba(0,0,0,0.4)" strokeWidth="1" />)}
        {paired.map((s, i) => <text key={i} x={toX(i)} y={H - 5} textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9">E{s.epoch}</text>)}
      </svg>
      <p className="text-xs text-gray-500 mt-2">
        {isOverfitting
          ? '⚠ Der Gap zwischen Val und Train Loss wächst — Overfitting-Zeichen. Mehr Regularisierung oder Early Stopping empfohlen.'
          : '✅ Val–Train Gap bleibt gering — kein starkes Overfitting erkennbar.'}
      </p>
    </div>
  );
}

// ── 4. Loss-Verbesserung pro Epoche (ΔTrain als Balkendiagramm) ──────────────

function EpochImprovementChart({ summaries }: { summaries: EpochSummary[] }) {
  if (summaries.length < 2) return null;
  const improvements = summaries.slice(1).map((s, i) => {
    const prev = summaries[i].avg_train_loss ?? 0, curr = s.avg_train_loss ?? 0;
    return { epoch: s.epoch, delta: prev - curr, pct: prev > 0 ? ((prev - curr) / prev) * 100 : 0 };
  });
  const maxDelta = Math.max(...improvements.map(x => Math.abs(x.delta)), 0.0001);
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <span className="text-sm font-medium text-gray-300 block mb-3">Loss-Verbesserung pro Epoche (ΔTrain)</span>
      <div className="space-y-2">
        {improvements.map(x => {
          const isGood = x.delta > 0;
          const barPct = (Math.abs(x.delta) / maxDelta) * 100;
          return (
            <div key={x.epoch} className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-8 shrink-0">E{x.epoch}</span>
              <div className="flex-1 bg-white/5 rounded-full h-2.5 overflow-hidden">
                <div
                  className={`h-2.5 rounded-full transition-all ${isGood ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-red-500 to-red-400'}`}
                  style={{ width: `${barPct}%` }}
                />
              </div>
              <div className="flex items-center gap-1.5 w-28 justify-end">
                <span className={`text-xs font-mono font-semibold ${isGood ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isGood ? '↓' : '↑'}{Math.abs(x.delta).toFixed(4)}
                </span>
                <span className="text-xs text-gray-600">({Math.abs(x.pct).toFixed(1)}%)</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── 5. Learning Rate Schedule mit Warmup-Erkennung ───────────────────────────

function LrScheduleChart({ logs }: { logs: LogEntry[] }) {
  const { success, error } = useNotification();
  const svgRef = useRef<SVGSVGElement>(null);
  const lrs = logs.filter(l => l.learning_rate > 0).map(l => l.learning_rate);
  if (lrs.length < 2) return null;
  const W = 400; const H = 130;
  const PAD = { l: 56, r: 12, t: 16, b: 30 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const maxLr = Math.max(...lrs), minLr = Math.min(...lrs);
  const toX = (i: number) => PAD.l + (i / Math.max(lrs.length - 1, 1)) * iW;
  const toY = (v: number) => PAD.t + iH - ((v - minLr) / (maxLr - minLr || 1)) * iH;
  const linePath = lrs.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${toX(lrs.length - 1).toFixed(1)},${(PAD.t + iH).toFixed(1)} L${PAD.l},${(PAD.t + iH).toFixed(1)} Z`;
  const peakIdx = lrs.indexOf(maxLr);
  const hasWarmup = peakIdx > 0 && peakIdx < lrs.length * 0.4;

  const handleDownload = async () => {
    if (!svgRef.current) { error('Fehler', 'Chart-Element nicht gefunden'); return; }
    try {
      await downloadSvgAsPng(svgRef.current, 'lr-schedule-chart.png');
      success('Bild erfolgreich heruntergeladen! 🎉', 'lr-schedule-chart.png');
    } catch (err) { error('Download fehlgeschlagen', String(err)); }
  };

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300">Learning Rate Schedule</span>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400 font-mono">{lrs[0]?.toExponential(2)} → {lrs[lrs.length - 1]?.toExponential(2)}</span>
          <button onClick={handleDownload} className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all" title="Chart als PNG exportieren">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 150 }}>
        <defs>
          <linearGradient id="lrFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map(f => (
          <g key={f}>
            <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + iH * f} y2={PAD.t + iH * f} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <text x={PAD.l - 6} y={PAD.t + iH * f + 3.5} textAnchor="end" fill="rgba(255,255,255,0.25)" fontSize="8.5">
              {(minLr + (maxLr - minLr) * (1 - f)).toExponential(1)}
            </text>
          </g>
        ))}
        {hasWarmup && (
          <g>
            <line x1={toX(peakIdx)} x2={toX(peakIdx)} y1={PAD.t} y2={PAD.t + iH} stroke="rgba(251,191,36,0.4)" strokeWidth="1.5" strokeDasharray="4,3" />
            <text x={toX(peakIdx) + 4} y={PAD.t + 11} fill="rgba(251,191,36,0.6)" fontSize="8.5">Warmup Ende</text>
          </g>
        )}
        <path d={areaPath} fill="url(#lrFill)" />
        <path d={linePath} fill="none" stroke="#a78bfa" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
        <text x={PAD.l} y={H - 6} textAnchor="start" fill="rgba(255,255,255,0.2)" fontSize="9">Start</text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" fill="rgba(255,255,255,0.2)" fontSize="9">Ende</text>
      </svg>
    </div>
  );
}

// ── 6. Gradient Norm mit Exploding-Gradient-Warnung ──────────────────────────

function GradNormChart({ logs }: { logs: LogEntry[] }) {
  const { success, error } = useNotification();
  const svgRef = useRef<SVGSVGElement>(null);
  const norms = logs.filter(l => l.grad_norm != null).map(l => l.grad_norm!);
  if (!norms.length) return null;
  const W = 400; const H = 130;
  const PAD = { l: 52, r: 16, t: 16, b: 30 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const maxN = Math.max(...norms, 1) * 1.1;
  const avgN = norms.reduce((a, b) => a + b, 0) / norms.length;
  const DANGER = 5.0;
  const hasSpike = maxN > DANGER;
  const toX = (i: number) => PAD.l + (i / Math.max(norms.length - 1, 1)) * iW;
  const toY = (v: number) => PAD.t + iH - (v / maxN) * iH;
  const linePath = norms.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${toX(norms.length - 1).toFixed(1)},${(PAD.t + iH).toFixed(1)} L${PAD.l},${(PAD.t + iH).toFixed(1)} Z`;
  const dangerY = toY(Math.min(DANGER, maxN));

  const handleDownload = async () => {
    if (!svgRef.current) { error('Fehler', 'Chart-Element nicht gefunden'); return; }
    try {
      await downloadSvgAsPng(svgRef.current, 'grad-norm-chart.png');
      success('Bild erfolgreich heruntergeladen! 🎉', 'grad-norm-chart.png');
    } catch (err) { error('Download fehlgeschlagen', String(err)); }
  };

  return (
    <div className={`rounded-xl border p-4 ${hasSpike ? 'bg-red-500/8 border-red-500/20' : 'bg-white/5 border-white/10'}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
          Gradient Norm
          {hasSpike && <span className="text-xs px-2 py-0.5 bg-red-500/20 text-red-400 border border-red-500/30 rounded-full">⚡ Spike</span>}
        </span>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400">Max: <span className={hasSpike ? 'text-red-400 font-semibold' : 'text-gray-300'}>{Math.max(...norms).toFixed(3)}</span> · Ø: {avgN.toFixed(3)}</span>
          <button onClick={handleDownload} className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-all" title="Chart als PNG exportieren">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 150 }}>
        <defs>
          <linearGradient id="gradFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={hasSpike ? '#ef4444' : '#f59e0b'} stopOpacity="0.25" />
            <stop offset="100%" stopColor={hasSpike ? '#ef4444' : '#f59e0b'} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map(f => (
          <g key={f}>
            <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + iH * f} y2={PAD.t + iH * f} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <text x={PAD.l - 6} y={PAD.t + iH * f + 3.5} textAnchor="end" fill="rgba(255,255,255,0.25)" fontSize="8.5">
              {(maxN * (1 - f)).toFixed(2)}
            </text>
          </g>
        ))}
        {DANGER <= maxN && (
          <g>
            <line x1={PAD.l} x2={W - PAD.r} y1={dangerY} y2={dangerY} stroke="rgba(239,68,68,0.5)" strokeWidth="1.5" strokeDasharray="5,3" />
            <text x={W - PAD.r + 2} y={dangerY + 3} fill="rgba(239,68,68,0.7)" fontSize="8.5">5.0</text>
          </g>
        )}
        <path d={areaPath} fill="url(#gradFill)" />
        <path d={linePath} fill="none" stroke={hasSpike ? '#ef4444' : '#f59e0b'} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      {hasSpike && <p className="text-xs text-red-400/80 mt-2">⚡ Gradient-Spikes über 5.0 erkannt — Gradient Clipping (max_grad_norm ≤ 1.0) empfohlen.</p>}
    </div>
  );
}

// ── 7. Epoch-Dauer als horizontale Balken ────────────────────────────────────

function EpochDurationBar({ summaries }: { summaries: EpochSummary[] }) {
  if (!summaries.length) return null;
  const durations = summaries.map(s => s.duration_seconds);
  const maxDur = Math.max(...durations);
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <span className="text-sm font-medium text-gray-300 block mb-3">Trainingsdauer pro Epoche</span>
      <div className="space-y-1.5">
        {summaries.map(s => (
          <div key={s.epoch} className="flex items-center gap-3">
            <span className="text-xs text-gray-500 w-8 shrink-0">E{s.epoch}</span>
            <div className="flex-1 bg-white/5 rounded-full h-2">
              <div className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all" style={{ width: `${(s.duration_seconds / maxDur) * 100}%` }} />
            </div>
            <span className="text-xs text-gray-400 w-14 text-right">{formatDuration(s.duration_seconds)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Report Renderer
// ─────────────────────────────────────────────────────────────────────────────

function ReportText({ text }: { text: string }) {
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {text.split('\n').map((line, i) => {
        if (line.startsWith('# ')) return <h2 key={i} className="text-lg font-bold text-white mt-5 mb-2">{line.slice(2)}</h2>;
        if (line.startsWith('## ')) return <h3 key={i} className="text-base font-bold text-white mt-4 mb-1">{line.slice(3)}</h3>;
        if (line.startsWith('### ')) return <h4 key={i} className="text-sm font-semibold text-purple-300 mt-3 mb-1">{line.slice(4)}</h4>;
        if (line.startsWith('```')) return null;
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <div key={i} className="flex items-start gap-2">
              <span className="text-purple-400 mt-1 shrink-0">•</span>
              <span className="text-gray-300" dangerouslySetInnerHTML={{ __html: line.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>') }} />
            </div>
          );
        }
        if (!line.trim()) return <div key={i} className="h-2" />;
        return <p key={i} className="text-gray-300" dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>') }} />;
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AI System Prompt
// ─────────────────────────────────────────────────────────────────────────────

const ANALYSIS_SYSTEM_PROMPT = `Du bist ein erfahrener Machine Learning Ingenieur und Modell-Training-Experte.
Analysiere die bereitgestellten Trainingsdaten präzise, konkret und strukturiert auf Deutsch.

Deine Analyse MUSS folgende Abschnitte enthalten:

## 🎯 Gesamtbewertung
Bewerte das Training mit einer Note (1–10) und begründe sie mit konkreten Zahlen.

## ✅ Was lief gut
Beziehe dich auf konkrete Werte.

## ⚠️ Erkannte Probleme & Schwächen
Analysiere: Overfitting, Underfitting, instabile Gradienten, schlechte Konvergenz, fehlendes Val-Set.

## 💡 Detaillierte Verbesserungsvorschläge
Für jedes Problem eine konkrete Lösung mit Begründung.

## 🔧 Empfohlene Parameter für das nächste Training
\`\`\`json
{
  "epochs": ...,
  "batch_size": ...,
  "learning_rate": ...,
  "optimizer": "...",
  "scheduler": "...",
  "warmup_ratio": ...,
  "weight_decay": ...,
  "gradient_accumulation_steps": ...,
  "max_seq_length": ...
}
\`\`\`

## 📊 Prognose
Was erwartest du vom nächsten Training?`;

// ─────────────────────────────────────────────────────────────────────────────
// Props + Main Component
// ─────────────────────────────────────────────────────────────────────────────

interface AnalysisPanelProps { initialVersionId?: string | null; }

export default function AnalysisPanel({ initialVersionId }: AnalysisPanelProps) {
  const { currentTheme } = useTheme();
  const { success, error: notifyError } = useNotification();
  const { settings: aiSettings } = useAISettings();
  const { setCurrentPageContent } = usePageContext();

  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [versionDetails, setVersionDetails] = useState<VersionDetails | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [fullData, setFullData] = useState<FullTrainingData | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  const [report, setReport] = useState<AIAnalysisReport | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [aiRecommendedParams, setAiRecommendedParams] = useState<Record<string, any> | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [templates, setTemplates] = useState<MetricsTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDesc, setTemplateDesc] = useState('');
  const [savingAITemplate, setSavingAITemplate] = useState(false);

  const [showEpochTable, setShowEpochTable] = useState(false);
  const [showLogTable, setShowLogTable] = useState(false);
  const [enableSmoothing, setEnableSmoothing] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: string } | null>(null);

  const aiProvider = aiSettings.provider as AIProvider;
  const aiApiKey = aiSettings.apiKey;
  const aiModel = aiSettings.provider === 'ollama' ? aiSettings.ollamaModel : aiSettings.selectedModel;
  const aiEnabled = aiSettings.enabled;

  // ── Effects ────────────────────────────────────────────────────────────────

  useEffect(() => { loadModels(); loadTemplates(); }, []);

  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); return; }
    const m = modelsWithVersions.find(x => x.id === selectedModelId);
    if (!m?.versions.length) { setSelectedVersionId(null); return; }
    setSelectedVersionId([...m.versions].sort((a, b) => b.version_number - a.version_number)[0].id);
  }, [selectedModelId, modelsWithVersions]);

  useEffect(() => {
    if (selectedVersionId) loadAnalysisData();
    else { setMetrics(null); setVersionDetails(null); setLogs([]); setFullData(null); setReport(null); setChatMessages([]); setAiRecommendedParams(null); }
  }, [selectedVersionId]);

  useEffect(() => {
    if (!initialVersionId || !modelsWithVersions.length) return;
    for (const m of modelsWithVersions) {
      if (m.versions.some(v => v.id === initialVersionId)) { setSelectedModelId(m.id); setSelectedVersionId(initialVersionId); break; }
    }
  }, [initialVersionId, modelsWithVersions]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  useEffect(() => {
    const selectedModel = modelsWithVersions.find(m => m.id === selectedModelId);
    const selectedVersion = selectedModel?.versions.find(v => v.id === selectedVersionId);
    setCurrentPageContent([
      '=== FrameTrain Analyse-Panel ===',
      selectedModel ? `Modell: "${selectedModel.name}"` : 'Kein Modell',
      selectedVersion ? `Version: "${selectedVersion.name}"` : '',
      metrics ? `Train Loss: ${metrics.final_train_loss.toFixed(6)} | Epochen: ${metrics.total_epochs}` : 'Keine Metriken',
      report ? `KI-Bericht vorhanden (${formatDate(report.generated_at)})` : 'Kein KI-Bericht',
    ].join('\n'));
  }, [modelsWithVersions, selectedModelId, selectedVersionId, metrics, report, setCurrentPageContent]);

  // ── Loaders ────────────────────────────────────────────────────────────────

  const loadModels = async () => {
    try { setLoading(true); const list = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree'); setModelsWithVersions(list); if (list.length > 0) setSelectedModelId(list[0].id); }
    catch (e: any) { notifyError('Fehler beim Laden der Modelle', String(e)); }
    finally { setLoading(false); }
  };

  const loadTemplates = async () => {
    try { setTemplates(await invoke<MetricsTemplate[]>('get_metrics_templates')); } catch { setTemplates([]); }
  };

  const loadAnalysisData = async () => {
    if (!selectedVersionId) return;
    setLoadingAnalysis(true);
    try {
      const [metricsRes, detailsRes, logsRes, fullDataRes, reportRes] = await Promise.allSettled([
        invoke<TrainingMetrics>('get_training_metrics', { versionId: selectedVersionId }),
        invoke<VersionDetails>('get_version_details', { versionId: selectedVersionId }),
        invoke<LogEntry[]>('get_training_logs', { versionId: selectedVersionId }),
        invoke<FullTrainingData | null>('get_training_full_data', { versionId: selectedVersionId }),
        invoke<AIAnalysisReport | null>('get_ai_analysis_report', { versionId: selectedVersionId }),
      ]);
      setMetrics(metricsRes.status === 'fulfilled' ? metricsRes.value : null);
      setVersionDetails(detailsRes.status === 'fulfilled' ? detailsRes.value : null);
      setLogs(logsRes.status === 'fulfilled' ? logsRes.value : []);
      setFullData(fullDataRes.status === 'fulfilled' ? fullDataRes.value : null);
      if (reportRes.status === 'fulfilled' && reportRes.value) {
        const r = reportRes.value;
        setReport(r);
        setChatMessages([{ role: 'assistant', content: r.report_text }]);
        setAiRecommendedParams(extractAIRecommendedParams(r.report_text));
      } else { setReport(null); setChatMessages([]); setAiRecommendedParams(null); }
    } finally { setLoadingAnalysis(false); }
  };

  // ── AI Context ─────────────────────────────────────────────────────────────

  function buildFullContext(): string {
    const modelName = modelsWithVersions.find(m => m.id === selectedModelId)?.name || 'Unbekannt';
    const lines: string[] = [`=== TRAININGSANALYSE: ${modelName} (${versionDetails?.version_name || selectedVersionId}) ===`];
    if (fullData) {
      const s = fullData.training_summary, cfg = fullData.config, hw = fullData.hardware, mi = fullData.model_info, ds = fullData.dataset_info, st = fullData.derived_stats || {};
      lines.push(`\nFinal Train Loss: ${s.final_train_loss} | Val: ${s.final_val_loss ?? 'N/A'} | Epochen: ${s.total_epochs} | Steps: ${s.total_steps}`);
      lines.push(`Dauer: ${formatDuration(s.training_duration_seconds)}`);
      if (st.loss_reduction_pct !== undefined) lines.push(`Loss-Reduktion: ${st.loss_reduction_pct}% | Overfitting-Gap: ${st.overfitting_gap_pct ?? 'N/A'}%`);
      if (st.avg_grad_norm !== undefined) lines.push(`Ø Grad Norm: ${st.avg_grad_norm} | Max: ${st.max_grad_norm}`);
      lines.push(`\nConfig: epochs=${cfg.epochs} batch=${cfg.batch_size} lr=${cfg.learning_rate} opt=${cfg.optimizer} sched=${cfg.scheduler}`);
      lines.push(`LoRA: ${cfg.use_lora} | fp16: ${cfg.fp16} | seq_len: ${cfg.max_seq_length}`);
      lines.push(`Hardware: ${hw.device?.toUpperCase()} ${hw.system_ram_gb}GB RAM | Val-Set: ${ds.has_validation ? 'Ja' : 'NEIN'}`);
      if (fullData.epoch_summaries?.length > 0) {
        lines.push('\nEpochen:');
        for (const e of fullData.epoch_summaries) lines.push(`E${e.epoch}: Ø=${e.avg_train_loss?.toFixed(4)} Min=${e.min_train_loss?.toFixed(4)} Val=${e.val_loss?.toFixed(4) ?? 'N/A'}`);
      }
    } else if (metrics) {
      lines.push(`Train Loss: ${metrics.final_train_loss} | Epochen: ${metrics.total_epochs}`);
      if (logs.length > 0) lines.push(`Logs: ${logs[0].train_loss.toFixed(4)} → ${logs[logs.length - 1].train_loss.toFixed(4)}`);
    }
    return lines.join('\n');
  }

  // ── KI-Analyse ─────────────────────────────────────────────────────────────

  const runAIAnalysis = async () => {
    if (!selectedVersionId) return;
    if (!aiEnabled) { notifyError('KI nicht aktiviert', 'Aktiviere den KI-Assistenten in den Einstellungen → KI-Assistent.'); return; }
    const meta = PROVIDER_META[aiProvider];
    if (meta.needsKey && !aiApiKey.trim()) { notifyError('API-Key fehlt', `${meta.label}-Key konfigurieren.`); return; }
    setGeneratingReport(true);
    try {
      const text = await callAI(aiProvider, aiApiKey, aiModel, [{ role: 'user', content: `Analysiere folgendes Training:\n\n${buildFullContext()}` }], ANALYSIS_SYSTEM_PROMPT);
      await invoke('save_ai_analysis_report', { versionId: selectedVersionId, reportText: text, provider: aiProvider, model: aiModel });
      const newReport: AIAnalysisReport = { version_id: selectedVersionId, report_text: text, provider: aiProvider, model: aiModel, generated_at: new Date().toISOString() };
      setReport(newReport); setAiRecommendedParams(extractAIRecommendedParams(text));
      setChatMessages([{ role: 'assistant', content: text }]); setShowChat(true);
      success('Analyse abgeschlossen', `Erstellt mit ${PROVIDER_META[aiProvider].label}`);
    } catch (e: any) { notifyError('Analyse fehlgeschlagen', String(e)); }
    finally { setGeneratingReport(false); }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || chatLoading || !report) return;
    const meta = PROVIDER_META[aiProvider];
    if (meta.needsKey && !aiApiKey.trim()) { notifyError('API-Key fehlt', ''); return; }
    const userMsg: ChatMessage = { role: 'user', content: chatInput.trim() };
    const updated = [...chatMessages, userMsg];
    setChatMessages(updated); setChatInput(''); setChatLoading(true);
    try {
      const sys = `${ANALYSIS_SYSTEM_PROMPT}\n\nVorherige Analyse:\n${report.report_text}\n\nTrainingsdaten:\n${buildFullContext()}`;
      const reply = await callAI(aiProvider, aiApiKey, aiModel, updated, sys);
      setChatMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (e: any) { setChatMessages(prev => [...prev, { role: 'assistant', content: `❌ Fehler: ${String(e)}` }]); }
    finally { setChatLoading(false); }
  };

  // ── Templates ──────────────────────────────────────────────────────────────

  const saveCurrentParamsAsTemplate = async () => {
    if (!templateName.trim() || !fullData?.config) return;
    try {
      const tmpl = await invoke<MetricsTemplate>('save_metrics_template', { name: templateName, description: templateDesc, config: fullData.config, source: 'user' });
      setTemplates(prev => [...prev, tmpl]); setShowSaveTemplate(false); setTemplateName(''); setTemplateDesc('');
      success('Template gespeichert', templateName);
    } catch (e: any) { notifyError('Fehler', String(e)); }
  };

  const saveAIRecommendationAsTemplate = async () => {
    if (!aiRecommendedParams) return;
    setSavingAITemplate(true);
    try {
      const name = `KI-Empfehlung · ${versionDetails?.version_name || selectedVersionId?.slice(0, 8) || 'Analyse'}`;
      const desc = `KI-empfohlene Parameter · ${report ? formatDate(report.generated_at) : 'heute'} · ${PROVIDER_META[aiProvider].label}`;
      const tmpl = await invoke<MetricsTemplate>('save_metrics_template', { name, description: desc, config: aiRecommendedParams, source: 'ai' });
      setTemplates(prev => [...prev, tmpl]);
      success('Template gespeichert', `"${name}" ist jetzt beim Training abrufbar.`);
    } catch (e: any) { notifyError('Fehler', String(e)); }
    finally { setSavingAITemplate(false); }
  };

  const deleteTemplate = async (id: string) => {
    try { await invoke('delete_metrics_template', { templateId: id }); setTemplates(prev => prev.filter(t => t.id !== id)); }
    catch (e: any) { notifyError('Fehler', String(e)); }
  };

  const handleExport = async () => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify({ metrics, logs, fullData, report, exported_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = `training_analyse_${versionDetails?.version_name || selectedVersionId}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    success('Exportiert', '');
  };

  // ── Computed ───────────────────────────────────────────────────────────────

  const epochSummaries = fullData?.epoch_summaries || [];
  const derivedStats   = fullData?.derived_stats;
  const hasVal         = logs.some(l => l.val_loss != null) || epochSummaries.some(s => s.val_loss != null);
  const hasGradNorm    = logs.some(l => l.grad_norm != null);

  // ── Early Returns ──────────────────────────────────────────────────────────

  if (loading) return <div className="flex items-center justify-center py-24"><Loader2 className="w-8 h-8 text-gray-400 animate-spin" /></div>;
  if (!modelsWithVersions.length) return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Trainingsanalyse</h1>
      <div className="bg-white/5 rounded-2xl border border-white/10 p-14 text-center">
        <Layers className="w-12 h-12 text-gray-400 mx-auto mb-4 opacity-40" />
        <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
        <p className="text-gray-400 text-sm">Trainiere zunächst ein Modell.</p>
      </div>
    </div>
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 pb-12">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Trainingsanalyse</h1>
          <p className="text-gray-400 mt-1 text-sm">Metriken · Graphen · KI-Auswertung · Parameter-Templates</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowTemplates(p => !p)} className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 text-sm border border-white/10 transition-all">
            <BookOpen className="w-4 h-4" /><span>Templates ({templates.length})</span>
          </button>
          {metrics && (
            <button onClick={handleExport} className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 text-sm border border-white/10 transition-all">
              <Download className="w-4 h-4" />Export
            </button>
          )}
          <button onClick={loadAnalysisData} disabled={!selectedVersionId || loadingAnalysis} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-40">
            <RefreshCw className={`w-5 h-5 ${loadingAnalysis ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Templates Panel */}
      {showTemplates && (
        <div className="bg-white/5 rounded-xl border border-white/10 p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><BookOpen className="w-4 h-4" />Gespeicherte Parameter-Templates</h3>
          {templates.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-5">Noch keine Templates. Generiere eine KI-Analyse und speichere die empfohlenen Parameter.</p>
          ) : (
            <div className="space-y-2">
              {templates.map(t => (
                <div key={t.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                  <div>
                    <div className="text-white font-medium text-sm">{t.name}</div>
                    {t.description && <div className="text-gray-400 text-xs mt-0.5">{t.description}</div>}
                    <div className="text-gray-600 text-xs mt-0.5">{t.source === 'ai' ? '🤖 KI-Empfehlung' : '👤 Eigenes Template'} · {formatDate(t.created_at)}</div>
                  </div>
                  <button onClick={() => deleteTemplate(t.id)} className="p-1.5 text-gray-500 hover:text-red-400 transition-colors"><Trash2 className="w-3.5 h-3.5" /></button>
                </div>
              ))}
            </div>
          )}
          {fullData?.config && (
            <div className="mt-3 pt-3 border-t border-white/10">
              {!showSaveTemplate ? (
                <button onClick={() => setShowSaveTemplate(true)} className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm hover:opacity-90 transition-all`}>
                  <Save className="w-4 h-4" />Aktuelle Config als Template speichern
                </button>
              ) : (
                <div className="space-y-2">
                  <input value={templateName} onChange={e => setTemplateName(e.target.value)} placeholder="Template-Name" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-white/30" />
                  <input value={templateDesc} onChange={e => setTemplateDesc(e.target.value)} placeholder="Beschreibung (optional)" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-white/30" />
                  <div className="flex gap-2">
                    <button onClick={saveCurrentParamsAsTemplate} disabled={!templateName.trim()} className={`flex-1 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm disabled:opacity-40`}>Speichern</button>
                    <button onClick={() => setShowSaveTemplate(false)} className="px-4 py-2 bg-white/5 rounded-lg text-gray-300 text-sm hover:bg-white/10">Abbrechen</button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Modell & Version */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5">Modell</label>
            <div className="relative">
              <select value={selectedModelId || ''} onChange={e => setSelectedModelId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none focus:border-white/30">
                {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-800">{m.name}</option>)}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1"><GitBranch className="w-3 h-3" />Version</label>
            <div className="relative">
              <select value={selectedVersionId || ''} onChange={e => setSelectedVersionId(e.target.value)} disabled={!modelsWithVersions.find(m => m.id === selectedModelId)?.versions.length} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none focus:border-white/30 disabled:opacity-50">
                {modelsWithVersions.find(m => m.id === selectedModelId)?.versions.map(v => (
                  <option key={v.id} value={v.id} className="bg-slate-800">{v.is_root ? '⭐ ' : ''}{v.name}{v.is_root ? ' (Original)' : ` (v${v.version_number})`}</option>
                )) || <option value="">–</option>}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {loadingAnalysis && <div className="flex items-center justify-center py-14"><Loader2 className="w-10 h-10 text-purple-400 animate-spin" /></div>}

      {!loadingAnalysis && selectedVersionId && !metrics && (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-14 text-center">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-3 opacity-80" />
          <h3 className="text-lg font-semibold text-white mb-1">Keine Trainingsdaten</h3>
          <p className="text-gray-400 text-sm">Diese Version wurde noch nicht trainiert oder die Logs wurden nicht gespeichert.</p>
        </div>
      )}

      {!loadingAnalysis && metrics && (
        <div className="space-y-5">

          {/* ── Metriken-Karten ─────────────────────────────────────────────── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { label: 'Final Train Loss', value: metrics.final_train_loss.toFixed(4), sub: `Val: ${metrics.final_val_loss?.toFixed(4) || 'N/A'}`, icon: <TrendingDown className="w-4 h-4 text-blue-400" />, color: 'text-blue-400' },
              { label: 'Epochen / Steps', value: `${metrics.total_epochs}E`, sub: `${metrics.total_steps.toLocaleString()} Steps`, icon: <Activity className="w-4 h-4 text-purple-400" />, color: 'text-purple-400' },
              { label: 'Dauer', value: formatDuration(metrics.training_duration_seconds), sub: metrics.best_epoch ? `Best: E${metrics.best_epoch}` : '–', icon: <Clock className="w-4 h-4 text-yellow-400" />, color: 'text-yellow-400' },
              { label: 'Status', value: 'Fertig', sub: formatDate(metrics.created_at), icon: <CheckCircle className="w-4 h-4 text-green-400" />, color: 'text-green-400' },
            ].map((c, i) => (
              <div key={i} className="bg-white/5 rounded-xl border border-white/10 p-4">
                <div className="flex items-center justify-between mb-2">{c.icon}<span className="text-xs text-gray-400">{c.label}</span></div>
                <div className={`text-xl font-bold ${c.color}`}>{c.value}</div>
                <div className="text-xs text-gray-500 mt-0.5">{c.sub}</div>
              </div>
            ))}
          </div>

          {/* ── Abgeleitete Statistiken ─────────────────────────────────────── */}
          {derivedStats && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Zap className="w-4 h-4 text-yellow-400" />Abgeleitete Statistiken</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                {derivedStats.loss_reduction_pct !== undefined && (
                  <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3 text-center">
                    <div className="text-gray-400 mb-0.5">Loss-Reduktion</div>
                    <div className="text-emerald-400 font-bold text-lg">{derivedStats.loss_reduction_pct}%</div>
                  </div>
                )}
                {derivedStats.overfitting_gap_pct !== undefined && (
                  <div className={`rounded-lg p-3 text-center ${Math.abs(derivedStats.overfitting_gap_pct) > 20 ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-white/5 border border-white/10'}`}>
                    <div className="text-gray-400 mb-0.5">Overfitting-Gap</div>
                    <div className={`font-bold text-lg ${Math.abs(derivedStats.overfitting_gap_pct) > 20 ? 'text-amber-400' : 'text-white'}`}>{derivedStats.overfitting_gap_pct}%</div>
                  </div>
                )}
                {derivedStats.avg_grad_norm !== undefined && (
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-center">
                    <div className="text-gray-400 mb-0.5">Ø Grad Norm</div>
                    <div className="text-white font-bold text-lg">{derivedStats.avg_grad_norm}</div>
                  </div>
                )}
                {derivedStats.total_log_entries !== undefined && (
                  <div className="bg-white/5 border border-white/10 rounded-lg p-3 text-center">
                    <div className="text-gray-400 mb-0.5">Log-Einträge</div>
                    <div className="text-white font-bold text-lg">{derivedStats.total_log_entries}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Charts ─────────────────────────────────────────────────────── */}
          {(logs.length > 1 || epochSummaries.length > 0) && (
            <div className="space-y-4">
              {/* Smoothing Toggle */}
              <div className="bg-white/5 rounded-xl border border-white/10 p-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-300">Smoothing (Moving Average)</span>
                  <span className="text-xs text-gray-500">Reduziert Noise in Loss-Kurven</span>
                </div>
                <button
                  onClick={() => setEnableSmoothing(!enableSmoothing)}
                  className={`relative w-11 h-6 rounded-full transition-all ${enableSmoothing ? 'bg-blue-500' : 'bg-white/10'}`}
                >
                  <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${enableSmoothing ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>

              {/* 1. Großer Loss-Chart – volle Breite, immer zuerst */}
              {logs.length > 1 && (
                <div className="grid grid-cols-1">
                  <BigLossChart logs={logs} label={`Loss-Verlauf über ${logs.length} Steps${hasVal ? ' (Train + Val)' : ''}`} enableSmoothing={enableSmoothing} />
                </div>
              )}

              {/* 2. Epoch-Level + Overfitting-Gap nebeneinander */}
              {epochSummaries.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <EpochLossLineChart summaries={epochSummaries} enableSmoothing={enableSmoothing} />
                  {hasVal
                    ? <OverfittingGapChart summaries={epochSummaries} />
                    : <EpochDurationBar summaries={epochSummaries} />
                  }
                </div>
              )}

              {/* 3. Loss-Verbesserung + LR-Schedule nebeneinander */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {epochSummaries.length > 1 && <EpochImprovementChart summaries={epochSummaries} />}
                {logs.length > 1 && <LrScheduleChart logs={logs} />}
              </div>

              {/* 4. Gradient Norm + (optional) Epoch-Dauer wenn Val schon woanders ist */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {hasGradNorm && <GradNormChart logs={logs} />}
                {hasVal && epochSummaries.length > 0 && <EpochDurationBar summaries={epochSummaries} />}
              </div>

              {/* 5. Hardware + Dataset Info */}
              {fullData && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                    <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Cpu className="w-4 h-4 text-blue-400" />Hardware</h3>
                    <div className="space-y-1.5 text-xs">
                      {Object.entries(fullData.hardware).map(([k, v]) =>
                        v !== null && v !== undefined ? (
                          <div key={k} className="flex justify-between"><span className="text-gray-400">{k}</span><span className="text-white font-medium">{String(v)}</span></div>
                        ) : null
                      )}
                    </div>
                  </div>
                  <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                    <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Database className="w-4 h-4 text-purple-400" />Dataset & Modell</h3>
                    <div className="space-y-1.5 text-xs">
                      {Object.entries({ ...fullData.dataset_info, ...fullData.model_info }).map(([k, v]) =>
                        v !== null && v !== undefined ? (
                          <div key={k} className="flex justify-between"><span className="text-gray-400">{k}</span><span className="text-white font-medium">{String(v)}</span></div>
                        ) : null
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Epoch-Tabelle ───────────────────────────────────────────────── */}
          {epochSummaries.length > 0 && (
            <div className="bg-white/5 rounded-xl border border-white/10 overflow-hidden">
              <button onClick={() => setShowEpochTable(p => !p)} className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2"><Target className="w-4 h-4" />Epoch-Zusammenfassung ({epochSummaries.length} Epochen)</h3>
                {showEpochTable ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
              </button>
              {showEpochTable && (
                <div className="p-4 pt-0 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-white/10">
                        {['Epoche', 'Ø Train Loss', 'Min Loss', 'Val Loss', 'Dauer', 'Steps'].map(h => <th key={h} className="pb-2 pr-4">{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {epochSummaries.map((e, i) => (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                          <td className="py-2 pr-4 text-white font-medium">{e.epoch}</td>
                          <td className="py-2 pr-4 text-blue-400">{e.avg_train_loss?.toFixed(4) || '–'}</td>
                          <td className="py-2 pr-4 text-emerald-400">{e.min_train_loss?.toFixed(4) || '–'}</td>
                          <td className="py-2 pr-4 text-purple-400">{e.val_loss?.toFixed(4) || 'N/A'}</td>
                          <td className="py-2 pr-4 text-gray-300">{formatDuration(e.duration_seconds)}</td>
                          <td className="py-2 text-gray-400">{e.steps}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* ── KI-Analyse ──────────────────────────────────────────────────── */}
          <div className="bg-gradient-to-br from-purple-500/10 via-blue-500/5 to-transparent rounded-2xl border border-purple-500/20 p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center"><Brain className="w-5 h-5 text-purple-400" /></div>
                <div>
                  <h2 className="text-lg font-bold text-white">KI-Trainingsanalyse</h2>
                  <p className="text-xs text-gray-400">
                    {report ? `${PROVIDER_META[report.provider as AIProvider]?.label || report.provider} · ${report.model} · ${formatDate(report.generated_at)}`
                      : aiEnabled ? `${PROVIDER_META[aiProvider].label} · ${aiModel}` : 'KI nicht aktiviert – Einstellungen → KI-Assistent'}
                  </p>
                </div>
              </div>
              {report && (
                <div className="flex items-center gap-2">
                  <button onClick={runAIAnalysis} disabled={generatingReport || !aiEnabled} className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-300 border border-white/10 transition-all disabled:opacity-40">
                    <RotateCcw className="w-3 h-3" />Neu analysieren
                  </button>
                  <button onClick={async () => {
                    try { await invoke('delete_ai_analysis_report', { versionId: selectedVersionId }); setReport(null); setChatMessages([]); setShowChat(false); setAiRecommendedParams(null); }
                    catch (e: any) { notifyError('Fehler', String(e)); }
                  }} className="p-1.5 bg-white/5 hover:bg-red-500/20 rounded-lg text-gray-400 hover:text-red-400 border border-white/10 transition-all">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {!aiEnabled && !report && (
              <div className="flex items-center gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
                <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />
                <div>
                  <div className="text-amber-300 text-sm font-medium">KI-Assistent nicht aktiviert</div>
                  <div className="text-amber-400/70 text-xs mt-0.5">Aktiviere einen KI-Anbieter unter <strong>Einstellungen → KI-Assistent</strong>.</div>
                </div>
              </div>
            )}

            {!report && !generatingReport && aiEnabled && (
              <button onClick={runAIAnalysis} className={`w-full py-4 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white font-semibold text-base hover:opacity-90 transition-all flex items-center justify-center gap-3 shadow-lg`}>
                <Sparkles className="w-5 h-5" />KI-Analyse starten
              </button>
            )}

            {generatingReport && (
              <div className="flex flex-col items-center justify-center gap-3 py-10 text-gray-300">
                <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
                <span className="text-sm">Analysiere Training mit {PROVIDER_META[aiProvider].label}…</span>
                <span className="text-xs text-gray-500">Das kann 10–30 Sekunden dauern.</span>
              </div>
            )}

            {report && !generatingReport && (
              <div className="space-y-4">
                <div className="bg-black/20 rounded-xl p-5 border border-white/10 max-h-[36rem] overflow-y-auto">
                  <ReportText text={report.report_text} />
                </div>

                {aiRecommendedParams ? (
                  <div className="p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl">
                    <div className="flex items-start gap-3 mb-3">
                      <Sparkles className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
                      <div>
                        <div className="text-sm font-semibold text-white">Empfohlene Parameter für nächstes Training</div>
                        <div className="text-xs text-gray-400 mt-0.5">{Object.keys(aiRecommendedParams).length} Parameter empfohlen</div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1.5 mb-3">
                      {Object.entries(aiRecommendedParams).slice(0, 7).map(([k, v]) => (
                        <span key={k} className="px-2 py-0.5 bg-purple-500/20 rounded text-xs text-purple-200 font-mono">{k}: {String(v)}</span>
                      ))}
                      {Object.keys(aiRecommendedParams).length > 7 && (
                        <span className="px-2 py-0.5 bg-white/10 rounded text-xs text-gray-400">+{Object.keys(aiRecommendedParams).length - 7} weitere</span>
                      )}
                    </div>
                    <button onClick={saveAIRecommendationAsTemplate} disabled={savingAITemplate} className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40`}>
                      {savingAITemplate ? <><Loader2 className="w-4 h-4 animate-spin" />Speichere…</> : <><Save className="w-4 h-4" />Als Training-Template speichern</>}
                    </button>
                  </div>
                ) : fullData?.config && (
                  <div className="text-xs text-gray-500 text-center py-2">
                    <Info className="w-3.5 h-3.5 inline mr-1" />KI hat keinen JSON-Parameterblock generiert · Manuelle Templates können oben gespeichert werden.
                  </div>
                )}

                <button onClick={() => setShowChat(p => !p)} className="w-full flex items-center justify-center gap-2 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-gray-300 hover:text-white transition-all text-sm font-medium">
                  <MessageSquare className="w-4 h-4" />
                  {showChat ? 'Chat ausblenden' : 'Mit KI über das Training chatten'}
                  {showChat ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {showChat && (
                  <div className="bg-black/20 rounded-xl border border-white/10 overflow-hidden">
                    <div className="h-96 overflow-y-auto p-4 space-y-3">
                      {chatMessages.length <= 1 && !chatLoading && (
                        <div className="text-center text-gray-500 text-sm py-10">
                          <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-40" />
                          Stelle Fragen zum Training, zu den Parametern oder zu Optimierungsstrategien.
                        </div>
                      )}
                      {chatMessages.slice(1).map((msg, i) => (
                        <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                          <div className={`w-7 h-7 rounded-full shrink-0 flex items-center justify-center ${msg.role === 'user' ? 'bg-purple-500/30' : 'bg-blue-500/20'}`}>
                            {msg.role === 'user' ? <User className="w-3.5 h-3.5 text-purple-300" /> : <Bot className="w-3.5 h-3.5 text-blue-300" />}
                          </div>
                          <div className={`max-w-[80%] rounded-xl px-3 py-2 text-sm leading-relaxed ${msg.role === 'user' ? 'bg-purple-500/20 text-white' : 'bg-white/5 text-gray-300'}`}>
                            {msg.content}
                          </div>
                        </div>
                      ))}
                      {chatLoading && (
                        <div className="flex gap-3">
                          <div className="w-7 h-7 rounded-full bg-blue-500/20 flex items-center justify-center"><Bot className="w-3.5 h-3.5 text-blue-300" /></div>
                          <div className="bg-white/5 rounded-xl px-3 py-2"><Loader2 className="w-4 h-4 animate-spin text-gray-400" /></div>
                        </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>
                    <div className="border-t border-white/10 p-3 flex gap-2">
                      <input value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); } }} placeholder="Frage zum Training stellen…" className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-white/30" />
                      <button onClick={sendChatMessage} disabled={!chatInput.trim() || chatLoading} className={`p-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white hover:opacity-90 transition-all disabled:opacity-40`}>
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Training Logs ──────────────────────────────────────────────── */}
          {logs.length > 0 && (
            <div className="bg-white/5 rounded-xl border border-white/10 overflow-hidden">
              <button onClick={() => setShowLogTable(p => !p)} className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2"><FileText className="w-4 h-4" />Training Logs ({logs.length} Einträge)</h3>
                {showLogTable ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
              </button>
              {showLogTable && (
                <div className="p-4 pt-0 max-h-72 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-900">
                      <tr className="text-left text-gray-400 border-b border-white/10">
                        {['E', 'Step', 'Train Loss', 'Val Loss', 'LR', 'Grad Norm', 'Zeit'].map(h => <th key={h} className="pb-2 pr-3">{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {logs.map((l, i) => (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                          <td className="py-1.5 pr-3 text-white">{l.epoch}</td>
                          <td className="py-1.5 pr-3 text-gray-300">{l.step}</td>
                          <td className="py-1.5 pr-3 text-blue-400 font-medium">{l.train_loss.toFixed(4)}</td>
                          <td className="py-1.5 pr-3 text-emerald-400">{l.val_loss?.toFixed(4) || '–'}</td>
                          <td className="py-1.5 pr-3 text-amber-400 font-mono">{l.learning_rate.toExponential(2)}</td>
                          <td className="py-1.5 pr-3 text-purple-400">{l.grad_norm?.toFixed(3) || '–'}</td>
                          <td className="py-1.5 text-gray-500">{new Date(l.timestamp).toLocaleTimeString('de-DE')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
