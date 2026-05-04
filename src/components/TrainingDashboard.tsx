// TrainingDashboard.tsx – Live Training Dashboard + Session Logger + Error Recovery

import { useState, useEffect, useRef } from 'react';
import {
  TrendingDown, BarChart3, Zap, Clock, Cpu,
  Square, CheckCircle, AlertCircle, Loader2,
  Minimize2, Maximize2, ChevronDown, ChevronUp,
  X, Sparkles, Send, Copy, Check, Code2, Wrench,
  Database, MemoryStick, Bug,
} from 'lucide-react';
import type { TrainingConfig } from './TrainingPanel';
import type { TrainingJob, LossPoint } from '../contexts/TrainingContext';
import { useTheme } from '../contexts/ThemeContext';

// ── Session Storage ───────────────────────────────────────────────────────

const SESSION_KEY = 'ft_training_sessions';

export interface SessionEvent {
  time: string;
  type: 'start' | 'epoch' | 'checkpoint' | 'complete' | 'error' | 'stop' | 'info';
  message: string;
}

export interface TrainingSession {
  id: string;
  mode: 'standard' | 'dev';
  model_name: string;
  dataset_name: string;
  config?: Partial<TrainingConfig>;
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  loss_points: LossPoint[];
  events: SessionEvent[];
  final_train_loss?: number;
  final_val_loss?: number;
  best_val_loss?: number;
  best_val_step?: number;
  total_steps?: number;
  total_epochs?: number;
}

export function loadSessions(): TrainingSession[] {
  try { return JSON.parse(localStorage.getItem(SESSION_KEY) ?? '[]'); } catch { return []; }
}

export function saveSession(session: TrainingSession) {
  const all = loadSessions();
  const idx = all.findIndex(s => s.id === session.id);
  if (idx >= 0) all[idx] = session;
  else all.unshift(session);
  localStorage.setItem(SESSION_KEY, JSON.stringify(all.slice(0, 100)));
}

export function getSession(id: string): TrainingSession | undefined {
  return loadSessions().find(s => s.id === id);
}

// ── Error Analysis ────────────────────────────────────────────────────────

type ErrorCategory = 'memory' | 'dataset' | 'packages' | 'cuda' | 'config' | 'code' | 'unknown';

function analyzeError(errorMsg: string): { category: ErrorCategory; title: string; hint: string } {
  const e = (errorMsg ?? '').toLowerCase();
  if (e.includes('cuda out of memory') || e.includes('out of memory') || e.includes('oom'))
    return { category: 'memory', title: 'GPU/RAM Überlauf', hint: 'Batch-Größe reduzieren, FP16 oder LoRA aktivieren, Gradient Checkpointing einschalten.' };
  if (e.includes('cuda') || e.includes('mps') || e.includes('device'))
    return { category: 'cuda', title: 'Hardware / Device Fehler', hint: 'Gerät nicht verfügbar oder inkompatibel. FP16 mit BF16 tauschen, oder auf CPU wechseln.' };
  if (e.includes('dataset') || e.includes('file not found') || e.includes('no such file') || e.includes('path'))
    return { category: 'dataset', title: 'Dataset / Pfad Fehler', hint: 'Dataset-Pfad prüfen, Dataset-Split ausführen, Dateirechte prüfen.' };
  if (e.includes('modulenotfounderror') || e.includes('importerror') || e.includes('no module'))
    return { category: 'packages', title: 'Fehlende Python-Pakete', hint: 'pip install torch transformers datasets accelerate ausführen.' };
  if (e.includes('nan') || e.includes('inf') || e.includes('gradient') || e.includes('loss'))
    return { category: 'config', title: 'Numerischer Fehler (NaN/Inf)', hint: 'Learning Rate zu hoch? Gradient Clipping (max_grad_norm) aktivieren, LR auf 1e-5 reduzieren.' };
  if (e.includes('syntaxerror') || e.includes('indentationerror') || e.includes('typeerror') || e.includes('attributeerror'))
    return { category: 'code', title: 'Code-Fehler', hint: 'Syntax- oder Typfehler im Training-Skript. KI-Assistent kann den Fehler analysieren.' };
  return { category: 'unknown', title: 'Unbekannter Fehler', hint: 'Den Fehler an das FrameTrain-Team senden oder den KI-Assistenten zur Analyse nutzen.' };
}

// ── Big Loss Chart ────────────────────────────────────────────────────────

function BigLossChart({ points }: { points: LossPoint[] }) {
  if (points.length < 2) {
    return (
      <div className="h-52 flex flex-col items-center justify-center gap-2">
        <Loader2 className="w-6 h-6 text-gray-600 animate-spin" />
        <p className="text-gray-600 text-sm">Warte auf erste Loss-Werte…</p>
      </div>
    );
  }

  const W = 600; const H = 200;
  const PAD = { l: 52, r: 20, t: 18, b: 40 };
  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;

  const trains = points.map(p => p.train_loss);
  const vals = points.map(p => p.val_loss).filter((v): v is number => v != null);
  const all = [...trains, ...vals];
  const minV = Math.min(...all) * 0.95;
  const maxV = Math.max(...all) * 1.05;

  const toX = (i: number) => PAD.l + (i / (points.length - 1)) * iW;
  const toY = (v: number) => PAD.t + iH - ((v - minV) / (maxV - minV || 1)) * iH;

  const trainPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.train_loss).toFixed(1)}`).join(' ');
  const trainArea = `${trainPath} L${toX(points.length - 1).toFixed(1)},${(PAD.t + iH).toFixed(1)} L${PAD.l},${(PAD.t + iH).toFixed(1)} Z`;

  const valPts = points.filter(p => p.val_loss != null);
  const valPath = valPts.map((p, idx) => {
    const i = points.indexOf(p);
    return `${idx === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.val_loss!).toFixed(1)}`;
  }).join(' ');

  const epochChanges = points
    .map((p, i) => ({ i, epoch: p.epoch }))
    .filter((x, idx) => idx === 0 || x.epoch !== points[idx - 1].epoch);

  const gridYValues = [0, 0.25, 0.5, 0.75, 1];
  const last = points[points.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 220 }}>
      <defs>
        <linearGradient id="trainFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
        </linearGradient>
        <linearGradient id="valFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#a855f7" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#a855f7" stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {gridYValues.map(f => (
        <g key={f}>
          <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + iH * f} y2={PAD.t + iH * f} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
          <text x={PAD.l - 6} y={PAD.t + iH * f + 4} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="10">{(maxV - f * (maxV - minV)).toFixed(3)}</text>
        </g>
      ))}
      {epochChanges.filter(x => x.i > 0).map(({ i, epoch }) => (
        <g key={epoch}>
          <line x1={toX(i)} x2={toX(i)} y1={PAD.t} y2={PAD.t + iH} stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="4,3" />
          <text x={toX(i)} y={H - 8} textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="9">E{epoch}</text>
        </g>
      ))}
      <path d={trainArea} fill="url(#trainFill)" />
      <path d={trainPath} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      {vals.length > 1 && (
        <>
          <path d={`${valPath} L${toX(points.lastIndexOf(valPts[valPts.length - 1]))},${PAD.t + iH} L${toX(points.indexOf(valPts[0]))},${PAD.t + iH} Z`} fill="url(#valFill)" />
          <path d={valPath} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="5,3" strokeLinejoin="round" strokeLinecap="round" />
        </>
      )}
      <circle cx={toX(points.length - 1)} cy={toY(last.train_loss)} r="5" fill="#10b981" stroke="rgba(0,0,0,0.4)" strokeWidth="1.5" />
      <g transform={`translate(${PAD.l}, ${H - 10})`}>
        <circle cx="4" cy="-2" r="4" fill="#10b981" />
        <text x="14" y="2" fill="rgba(255,255,255,0.45)" fontSize="10">Train Loss</text>
        {vals.length > 0 && (
          <>
            <line x1="90" y1="-2" x2="106" y2="-2" stroke="#a855f7" strokeWidth="2" strokeDasharray="4,2" />
            <text x="112" y="2" fill="rgba(255,255,255,0.45)" fontSize="10">Val Loss</text>
          </>
        )}
      </g>
    </svg>
  );
}

// ── Config Summary ────────────────────────────────────────────────────────

function ConfigSummary({ config, mode }: { config?: Partial<TrainingConfig>; mode: 'standard' | 'dev' }) {
  if (mode === 'dev') {
    return <p className="text-gray-600 text-xs italic">Dev Train Mode — kein strukturierter Config.</p>;
  }
  if (!config) {
    return <p className="text-gray-600 text-xs italic">Konfiguration nicht verfügbar.</p>;
  }
  const rows: { label: string; value: string | number | boolean | undefined; color: string }[] = [
    { label: 'Epochen',       value: config.epochs,                            color: 'text-emerald-400' },
    { label: 'Batch Size',    value: config.batch_size,                        color: 'text-blue-400' },
    { label: 'Learning Rate', value: config.learning_rate?.toExponential(2),   color: 'text-purple-400' },
    { label: 'Max Seq Len',   value: config.max_seq_length,                    color: 'text-amber-400' },
    { label: 'Warmup Ratio',  value: config.warmup_ratio,                      color: 'text-cyan-400' },
    { label: 'Grad Accum',    value: config.gradient_accumulation_steps,       color: 'text-pink-400' },
    { label: 'Optimizer',     value: config.optimizer,                         color: 'text-emerald-400' },
    { label: 'Scheduler',     value: config.scheduler,                         color: 'text-blue-400' },
    { label: 'Weight Decay',  value: config.weight_decay,                      color: 'text-purple-400' },
    { label: 'Max Grad Norm', value: config.max_grad_norm,                     color: 'text-amber-400' },
    { label: 'Dropout',       value: config.dropout,                           color: 'text-cyan-400' },
    { label: 'Seed',          value: config.seed,                              color: 'text-gray-300' },
    { label: 'FP16',          value: config.fp16 ? 'Ja' : 'Nein',             color: config.fp16 ? 'text-emerald-400' : 'text-gray-600' },
    { label: 'BF16',          value: config.bf16 ? 'Ja' : 'Nein',             color: config.bf16 ? 'text-emerald-400' : 'text-gray-600' },
    { label: 'LoRA',          value: config.use_lora ? `r=${config.lora_r}` : 'Nein', color: config.use_lora ? 'text-violet-400' : 'text-gray-600' },
    { label: 'QLoRA (4bit)',  value: config.load_in_4bit ? 'Ja' : 'Nein',     color: config.load_in_4bit ? 'text-fuchsia-400' : 'text-gray-600' },
    { label: 'Grad Checkpoint', value: config.gradient_checkpointing ? 'Ja' : 'Nein', color: config.gradient_checkpointing ? 'text-emerald-400' : 'text-gray-600' },
  ];
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
      {rows.filter(r => r.value !== undefined).map(r => (
        <div key={r.label} className="flex items-center justify-between gap-1">
          <span className="text-gray-500 text-[11px] truncate">{r.label}</span>
          <span className={`${r.color} text-[11px] font-mono font-medium flex-shrink-0`}>{String(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

// ── Event Log ─────────────────────────────────────────────────────────────

const EVENT_ICONS: Record<string, string> = {
  start: '🚀', epoch: '📊', checkpoint: '💾',
  complete: '✅', error: '❌', stop: '⏹', info: 'ℹ️',
};
const EVENT_COLORS: Record<string, string> = {
  start: 'text-violet-400', epoch: 'text-blue-400', checkpoint: 'text-amber-400',
  complete: 'text-emerald-400', error: 'text-red-400', stop: 'text-gray-400', info: 'text-gray-400',
};

// ── Error Recovery Panel ──────────────────────────────────────────────────

function ErrorRecoveryPanel({
  mode, errorMsg, config,
  onOpenKIAssistant, onSendCodeToKI, devScript,
}: {
  mode: 'standard' | 'dev';
  errorMsg: string;
  config?: Partial<TrainingConfig>;
  onOpenKIAssistant?: () => void;
  onSendCodeToKI?: (script: string, error: string) => void;
  devScript?: string;
}) {
  const [copied, setCopied] = useState(false);
  const { category, title, hint } = analyzeError(errorMsg);

  const categoryIcon = {
    memory: <MemoryStick className="w-4 h-4 text-red-400" />,
    cuda:   <Zap className="w-4 h-4 text-amber-400" />,
    dataset: <Database className="w-4 h-4 text-blue-400" />,
    packages: <Bug className="w-4 h-4 text-orange-400" />,
    config: <Wrench className="w-4 h-4 text-purple-400" />,
    code:   <Code2 className="w-4 h-4 text-cyan-400" />,
    unknown: <AlertCircle className="w-4 h-4 text-red-400" />,
  }[category];

  const buildDiagReport = () => {
    const lines = [
      '=== FrameTrain Fehlerbericht ===',
      `Modus: ${mode === 'dev' ? 'Dev Train' : 'Standard'}`,
      `Fehler-Kategorie: ${title}`,
      `Fehlermeldung: ${errorMsg}`,
      '',
      '--- Konfiguration ---',
      ...(config ? Object.entries(config).map(([k, v]) => `${k}: ${v}`) : ['Dev Train (kein strukturierter Config)']),
      '',
      '--- Ende Bericht ---',
    ];
    return lines.join('\n');
  };

  const handleCopyReport = async () => {
    try {
      await navigator.clipboard.writeText(buildDiagReport());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
  };

  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/[0.06] overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-red-500/20">
        {categoryIcon}
        <div>
          <p className="text-red-300 font-semibold text-sm">{title}</p>
          <p className="text-gray-500 text-xs mt-0.5">{hint}</p>
        </div>
      </div>

      {/* Error text */}
      <div className="px-4 py-3 bg-black/20 border-b border-red-500/10">
        <pre className="text-red-300/80 text-[10px] font-mono whitespace-pre-wrap line-clamp-3 leading-relaxed">{errorMsg}</pre>
      </div>

      {/* Actions */}
      <div className="p-3 space-y-2">
        <p className="text-gray-500 text-[10px] uppercase tracking-wide font-medium mb-2">Optionen</p>

        {mode === 'standard' && (
          <>
            {/* Metrics KI */}
            {onOpenKIAssistant && (
              <button
                onClick={onOpenKIAssistant}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/25 transition-all group text-left"
              >
                <Sparkles className="w-4 h-4 text-violet-400 flex-shrink-0" />
                <div>
                  <p className="text-violet-300 text-xs font-medium">KI-Metriken anpassen</p>
                  <p className="text-gray-500 text-[10px]">KI-Assistent analysiert die Konfiguration und schlägt Fixes vor</p>
                </div>
              </button>
            )}

            {/* Memory-specific: suggest LoRA */}
            {(category === 'memory' || category === 'cuda') && config && !config.use_lora && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <MemoryStick className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-amber-300 text-xs font-medium">RAM-Tipp</p>
                  <p className="text-gray-400 text-[10px]">LoRA aktivieren (r=8), FP16 oder BF16 einschalten, Batch auf 2–4 reduzieren, Gradient Checkpointing aktivieren.</p>
                </div>
              </div>
            )}

            {/* Dataset warning */}
            {category === 'dataset' && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <Database className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-blue-300 text-xs font-medium">Dataset prüfen</p>
                  <p className="text-gray-400 text-[10px]">Dataset im Dataset-Manager aufteilen (Split ausführen), Pfade und Dateirechte prüfen.</p>
                </div>
              </div>
            )}

            {/* Send to FrameTrain */}
            <button
              onClick={handleCopyReport}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-left"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" /> : <Copy className="w-4 h-4 text-gray-400 flex-shrink-0" />}
              <div>
                <p className={`text-xs font-medium ${copied ? 'text-emerald-300' : 'text-gray-300'}`}>
                  {copied ? 'Fehlerbericht kopiert!' : 'Fehlerbericht kopieren'}
                </p>
                <p className="text-gray-600 text-[10px]">Vollständigen Bericht in Zwischenablage für FrameTrain-Support</p>
              </div>
            </button>

            {/* Engine/unknown: recommend FrameTrain team */}
            {(category === 'unknown' || category === 'packages') && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-white/[0.03] border border-white/10">
                <Send className="w-4 h-4 text-gray-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-gray-300 text-xs font-medium">An FrameTrain senden</p>
                  <p className="text-gray-500 text-[10px]">Fehlerbericht kopieren und an das FrameTrain-Team schicken — der Fehler liegt wahrscheinlich in der Engine und wird behoben.</p>
                </div>
              </div>
            )}
          </>
        )}

        {mode === 'dev' && (
          <>
            {/* Send code + error to KI */}
            {onSendCodeToKI && devScript && (
              <button
                onClick={() => onSendCodeToKI(devScript, errorMsg)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/25 transition-all text-left"
              >
                <Code2 className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                <div>
                  <p className="text-cyan-300 text-xs font-medium">Code mit KI korrigieren</p>
                  <p className="text-gray-500 text-[10px]">Fehler + Python-Code an den KI-Assistenten senden — KI schlägt direkte Code-Fixes vor</p>
                </div>
              </button>
            )}

            {/* Memory hint for dev mode */}
            {(category === 'memory' || category === 'cuda') && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <MemoryStick className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-amber-300 text-xs font-medium">Speicher-Fehler</p>
                  <p className="text-gray-400 text-[10px]">Im Skript Batch-Größe reduzieren, FP16 / bfloat16 verwenden, oder gradient_checkpointing=True setzen.</p>
                </div>
              </div>
            )}

            {/* Copy report */}
            <button
              onClick={handleCopyReport}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-left"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" /> : <Copy className="w-4 h-4 text-gray-400 flex-shrink-0" />}
              <div>
                <p className={`text-xs font-medium ${copied ? 'text-emerald-300' : 'text-gray-300'}`}>
                  {copied ? 'Kopiert!' : 'Fehlerbericht kopieren'}
                </p>
                <p className="text-gray-600 text-[10px]">Für FrameTrain-Support oder GitHub-Issue</p>
              </div>
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────

interface TrainingDashboardProps {
  isOpen: boolean;
  isMinimized: boolean;
  onMinimize: () => void;
  onMaximize: () => void;
  onClose?: () => void;                          // Schließt/verwirft das Dashboard
  mode: 'standard' | 'dev';
  modelName: string;
  datasetName: string;
  config?: TrainingConfig;
  job: TrainingJob | null;
  lossPoints: LossPoint[];
  sessionId: string;
  startedAt: number;
  onStop: () => void;
  completedVersionId?: string | null;            // Version-ID nach erfolgreichem Training
  onNavigateToAnalysis?: (id: string) => void;   // Navigiert zur Analyse-Seite
  // Error recovery
  onOpenKIAssistant?: () => void;                // Standard: öffnet KI-Metriken-Assistent
  devScript?: string;                            // Dev: aktueller Python-Code
  onSendCodeToKI?: (script: string, error: string) => void; // Dev: schickt Code+Fehler an KI
}

export default function TrainingDashboard({
  isOpen, isMinimized, onMinimize, onMaximize, onClose,
  mode, modelName, datasetName, config,
  job, lossPoints, sessionId, startedAt, onStop,
  completedVersionId, onNavigateToAnalysis,
  onOpenKIAssistant, devScript, onSendCodeToKI,
}: TrainingDashboardProps) {
  const { currentTheme } = useTheme();
  const [elapsed, setElapsed] = useState(0);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [showFullConfig, setShowFullConfig] = useState(false);
  const prevEpochRef = useRef(-1);
  const prevStatusRef = useRef('');
  const eventsRef = useRef<SessionEvent[]>([]);

  eventsRef.current = events;

  useEffect(() => {
    if (!isOpen) return;
    const id = setInterval(() => setElapsed(Date.now() - startedAt), 1000);
    return () => clearInterval(id);
  }, [isOpen, startedAt]);

  useEffect(() => {
    if (!job) return;
    const progress = job.progress;
    let newEvents = [...eventsRef.current];
    let changed = false;

    if (job.status !== prevStatusRef.current) {
      prevStatusRef.current = job.status;
      if (job.status === 'running' || job.status === 'pending') {
        const alreadyStarted = newEvents.some(e => e.type === 'start');
        if (!alreadyStarted) {
          newEvents = [{ time: new Date().toISOString(), type: 'start', message: `Training gestartet — ${modelName}` }, ...newEvents];
          changed = true;
        }
      } else if (job.status === 'completed') {
        const lastLoss = lossPoints[lossPoints.length - 1];
        newEvents = [{ time: new Date().toISOString(), type: 'complete', message: `Abgeschlossen! Finaler Loss: ${lastLoss?.train_loss?.toFixed(4) ?? '—'}${lastLoss?.val_loss != null ? ` | Val: ${lastLoss.val_loss.toFixed(4)}` : ''}` }, ...newEvents];
        changed = true;
      } else if (job.status === 'failed') {
        newEvents = [{ time: new Date().toISOString(), type: 'error', message: `Fehler: ${job.error ?? 'Unbekannt'}` }, ...newEvents];
        changed = true;
      } else if (job.status === 'stopped') {
        newEvents = [{ time: new Date().toISOString(), type: 'stop', message: 'Training manuell gestoppt' }, ...newEvents];
        changed = true;
      }
    }

    if (progress && progress.epoch !== prevEpochRef.current && progress.epoch > 0) {
      prevEpochRef.current = progress.epoch;
      const valStr = progress.val_loss != null ? ` · Val: ${progress.val_loss.toFixed(4)}` : '';
      newEvents = [{ time: new Date().toISOString(), type: 'epoch', message: `Epoch ${progress.epoch}/${progress.total_epochs} — Train: ${progress.train_loss?.toFixed(4)}${valStr}` }, ...newEvents];
      changed = true;
    }

    if (changed) setEvents(newEvents);

    const valLosses = lossPoints.map(p => p.val_loss).filter((v): v is number => v != null);
    const session: TrainingSession = {
      id: sessionId,
      mode,
      model_name: modelName,
      dataset_name: datasetName,
      config,
      started_at: new Date(startedAt).toISOString(),
      completed_at: (job.status !== 'running' && job.status !== 'pending') ? new Date().toISOString() : undefined,
      duration_seconds: Math.floor((Date.now() - startedAt) / 1000),
      status: (job.status === 'pending' ? 'running' : job.status) as TrainingSession['status'],
      loss_points: lossPoints,
      events: changed ? newEvents : eventsRef.current,
      final_train_loss: lossPoints[lossPoints.length - 1]?.train_loss,
      final_val_loss: lossPoints[lossPoints.length - 1]?.val_loss ?? undefined,
      best_val_loss: valLosses.length > 0 ? Math.min(...valLosses) : undefined,
      total_steps: progress?.total_steps,
      total_epochs: progress?.total_epochs,
    };
    saveSession(session);
  }, [job?.status, lossPoints.length]);

  if (!isOpen) return null;

  const progress = job?.progress;
  const isRunning  = job?.status === 'running' || job?.status === 'pending';
  const isCompleted = job?.status === 'completed';
  const isFailed   = job?.status === 'failed';
  const isStopped  = job?.status === 'stopped';
  const isDone     = isCompleted || isFailed || isStopped;

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    if (h > 0) return `${h}h ${m % 60}m`;
    if (m > 0) return `${m}m ${s % 60}s`;
    return `${s}s`;
  };

  const eta = (() => {
    if (!progress || !isRunning || progress.progress_percent <= 1) return null;
    const elapsedSec = elapsed / 1000;
    const totalSec = elapsedSec / (progress.progress_percent / 100);
    const remaining = totalSec - elapsedSec;
    if (remaining <= 0) return null;
    return formatDuration(remaining * 1000);
  })();

  const firstLoss = lossPoints[0]?.train_loss;
  const lastLoss  = lossPoints[lossPoints.length - 1]?.train_loss;
  const lossImprovement = firstLoss != null && lastLoss != null && firstLoss !== lastLoss
    ? ((firstLoss - lastLoss) / firstLoss * 100)
    : null;

  // ── Minimized floating bar ──

  if (isMinimized) {
    return (
      <div
        className="fixed bottom-5 right-5 z-50 flex items-center gap-3 px-4 py-3 rounded-2xl bg-slate-900 border border-white/10 shadow-2xl cursor-pointer hover:bg-slate-800 transition-all group"
        onClick={onMaximize}
      >
        {isRunning   && <Loader2      className="w-4 h-4 text-emerald-400 animate-spin flex-shrink-0" />}
        {isCompleted && <CheckCircle  className="w-4 h-4 text-emerald-400 flex-shrink-0" />}
        {isFailed    && <AlertCircle  className="w-4 h-4 text-red-400 flex-shrink-0" />}
        {isStopped   && <Square       className="w-4 h-4 text-gray-400 flex-shrink-0" />}
        <div className="min-w-0">
          <p className="text-white text-xs font-semibold">
            {isRunning ? 'Training läuft…' : isCompleted ? 'Training abgeschlossen ✓' : isFailed ? 'Training fehlgeschlagen' : 'Training gestoppt'}
          </p>
          {progress && (
            <p className="text-gray-500 text-[10px]">
              E{progress.epoch}/{progress.total_epochs} · Loss: {progress.train_loss?.toFixed(4)} · {formatDuration(elapsed)}
            </p>
          )}
        </div>
        {progress && (
          <div className="w-24 h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${progress.progress_percent}%` }} />
          </div>
        )}
        <Maximize2 className="w-3.5 h-3.5 text-gray-500 group-hover:text-white transition-all flex-shrink-0" />
      </div>
    );
  }

  // ── Full Dashboard ──

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4">
      <div className="w-full max-w-5xl bg-slate-900 rounded-2xl border border-white/10 shadow-2xl flex flex-col max-h-[92vh]">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-3">
            {isRunning   && <div className="relative"><Loader2 className="w-5 h-5 text-emerald-400 animate-spin" /></div>}
            {isCompleted && <CheckCircle className="w-5 h-5 text-emerald-400" />}
            {isFailed    && <AlertCircle className="w-5 h-5 text-red-400" />}
            {isStopped   && <Square className="w-5 h-5 text-gray-400" />}
            <div>
              <h2 className="text-white font-bold text-sm">
                {isRunning ? 'Training läuft' : isCompleted ? 'Training abgeschlossen' : isFailed ? 'Training fehlgeschlagen' : isStopped ? 'Training gestoppt' : 'Training Dashboard'}
              </h2>
              <p className="text-gray-500 text-xs">
                {modelName} · {datasetName} · {mode === 'dev' ? 'Dev Train' : 'Standard'}
                {progress && ` · ${Math.round(progress.progress_percent)}%`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isRunning && (
              <button onClick={onStop} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-300 text-xs font-medium transition-all">
                <Square className="w-3.5 h-3.5" /> Stoppen
              </button>
            )}
            <button onClick={onMinimize} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all" title="Minimieren">
              <Minimize2 className="w-4 h-4" />
            </button>
            {/* Close button – nur wenn nicht am Laufen */}
            {isDone && onClose && (
              <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all" title="Schließen">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">

          {/* Metrics strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'Train Loss', value: progress?.train_loss?.toFixed(4) ?? '—', sub: lossImprovement != null ? `${lossImprovement > 0 ? '↓' : '↑'} ${Math.abs(lossImprovement).toFixed(1)}% seit Start` : undefined, icon: <TrendingDown className="w-4 h-4" />, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
              { label: 'Val Loss',   value: progress?.val_loss?.toFixed(4) ?? '—', icon: <BarChart3 className="w-4 h-4" />, color: 'text-purple-400', bg: 'bg-purple-500/10 border-purple-500/20' },
              { label: 'Learning Rate', value: progress?.learning_rate?.toExponential(2) ?? (config?.learning_rate?.toExponential(2) ?? '—'), icon: <Zap className="w-4 h-4" />, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20' },
              { label: 'Laufzeit',  value: formatDuration(elapsed), sub: eta ? `ETA: ~${eta}` : (isCompleted ? 'Abgeschlossen' : isStopped ? 'Gestoppt' : undefined), icon: <Clock className="w-4 h-4" />, color: 'text-blue-400', bg: 'bg-blue-500/10 border-blue-500/20' },
            ].map(m => (
              <div key={m.label} className={`p-4 rounded-xl border ${m.bg} space-y-1`}>
                <div className={`flex items-center gap-1.5 ${m.color}`}>{m.icon}<span className="text-xs">{m.label}</span></div>
                <p className="text-white font-bold text-lg tabular-nums leading-none">{m.value}</p>
                {m.sub && <p className="text-gray-500 text-[10px] leading-tight">{m.sub}</p>}
              </div>
            ))}
          </div>

          {/* Progress bar */}
          {progress && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-gray-400">
                <span>Epoch {progress.epoch} / {progress.total_epochs} · Step {progress.step} / {progress.total_steps}</span>
                <span className="font-mono">{Math.round(progress.progress_percent)}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-white/10 overflow-hidden">
                <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${progress.progress_percent}%` }} />
              </div>
            </div>
          )}

          {/* Error Recovery Panel — nur bei Fehler (nicht bei manuellem Stopp) */}
          {isFailed && job?.error && (
            <ErrorRecoveryPanel
              mode={mode}
              errorMsg={job.error}
              config={config}
              onOpenKIAssistant={onOpenKIAssistant}
              devScript={devScript}
              onSendCodeToKI={onSendCodeToKI}
            />
          )}

          {/* Manuell gestoppt – einfacher Hinweis */}
          {isStopped && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 border border-white/10">
              <Square className="w-4 h-4 text-gray-400 flex-shrink-0" />
              <p className="text-gray-400 text-sm">Training wurde manuell gestoppt. Du kannst es jederzeit neu starten.</p>
            </div>
          )}

          {/* Abgeschlossen – Erfolgsmeldung + Analyse starten */}
          {isCompleted && (
            <div className="space-y-2">
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <p className="text-emerald-300 text-sm font-medium">Training erfolgreich abgeschlossen! 🎉</p>
              </div>
              {completedVersionId && onNavigateToAnalysis && (
                <button
                  onClick={() => onNavigateToAnalysis(completedVersionId)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:opacity-90 text-white font-semibold text-sm transition-all shadow-lg"
                >
                  <BarChart3 className="w-4 h-4" /> Analyse starten →
                </button>
              )}
            </div>
          )}

          {/* Chart + Config */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-gray-400 flex items-center gap-1.5"><TrendingDown className="w-3.5 h-3.5 text-emerald-400" /> Loss-Verlauf</p>
                <span className="text-[10px] text-gray-600">{lossPoints.length} Punkte</span>
              </div>
              <BigLossChart points={lossPoints} />
              {lossPoints.length >= 2 && (
                <div className="flex items-center gap-4 text-[10px] text-gray-500 border-t border-white/8 pt-2">
                  <span>Start: <span className="text-gray-300 font-mono">{firstLoss?.toFixed(4)}</span></span>
                  <span>Aktuell: <span className="text-gray-300 font-mono">{lastLoss?.toFixed(4)}</span></span>
                  {lossImprovement != null && (
                    <span className={lossImprovement > 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {lossImprovement > 0 ? '↓' : '↑'} {Math.abs(lossImprovement).toFixed(1)}%
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
              <button className="w-full flex items-center justify-between text-xs font-medium text-gray-400 hover:text-white transition-all" onClick={() => setShowFullConfig(v => !v)}>
                <span className="flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-blue-400" /> Konfiguration</span>
                {showFullConfig ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
              <ConfigSummary config={config} mode={mode} />
            </div>
          </div>

          {/* Event Log */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/10">
              <Clock className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-xs font-medium text-gray-400">Trainings-Log</span>
              <span className="ml-auto text-[10px] text-gray-600">{events.length} Einträge · wird in ft_training_sessions gespeichert</span>
            </div>
            <div className="max-h-40 overflow-y-auto p-3 space-y-1.5">
              {events.length === 0 ? (
                <p className="text-gray-600 text-xs text-center py-4 italic">Warte auf Events…</p>
              ) : events.map((ev, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px]">
                  <span className="text-gray-600 tabular-nums flex-shrink-0 font-mono text-[10px]">
                    {new Date(ev.time).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  <span className="flex-shrink-0 text-[13px] leading-[1.1]">{EVENT_ICONS[ev.type] ?? 'ℹ️'}</span>
                  <span className={`${EVENT_COLORS[ev.type] ?? 'text-gray-300'} leading-relaxed`}>{ev.message}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
