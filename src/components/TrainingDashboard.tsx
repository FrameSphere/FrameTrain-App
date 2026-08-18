// TrainingDashboard.tsx – Live Training Dashboard + Session Logger + Error Recovery

import { useState, useEffect, useRef } from 'react';
import {
  TrendingDown, BarChart3, Zap, Clock, Cpu,
  Square, CheckCircle, AlertCircle, Loader2,
  Minimize2, Maximize2, ChevronDown, ChevronUp,
  X, Sparkles, Send, Copy, Check, Code2, Wrench,
  Database, MemoryStick, Bug, Rocket, Save, Info, XCircle,
} from 'lucide-react';
import type { TrainingConfig } from './TrainingPanel';
import type { TrainingJob, LossPoint } from '../contexts/TrainingContext';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { sendAppErrorReport } from '../utils/errorReport';
import { dateLocale } from '../utils/dateLocale';

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

type ErrorCategory = 'memory' | 'dataset' | 'labels' | 'architecture' | 'packages' | 'cuda' | 'config' | 'code' | 'unknown';

export function analyzeError(errorMsg: string, t: (key: string) => string): { category: ErrorCategory; title: string; hint: string } {
  const e = (errorMsg ?? '').toLowerCase();
  if (e.includes('cuda out of memory') || e.includes('out of memory') || e.includes('oom'))
    return { category: 'memory', title: t('trainingDashboard.errorRecovery.memoryCategoryTitle'), hint: t('trainingDashboard.errorRecovery.memoryCategoryHint') };

  // Label-/Klassenprobleme VOR der Geräte-Prüfung: PyTorch meldet den
  // Regressions-Fallback bei num_labels=1 als "mse_loss_out_mps: only defined
  // for floating types". Wer nur auf "mps" prüft, verkauft dem Nutzer einen
  // Datenfehler als Hardwareproblem und schickt ihn zu FP16/CPU-Einstellungen.
  if (e.includes('mse_loss') || e.includes('only defined for floating types')
      || e.includes('label-spalte') || e.includes('label column')
      || e.includes('num_labels') || e.includes('nur einen einzigen wert'))
    return { category: 'labels', title: t('trainingDashboard.errorRecovery.labelsCategoryTitle'), hint: t('trainingDashboard.errorRecovery.labelsCategoryHint') };

  if (e.includes('wird noch nicht unterstützt') || e.includes('not yet supported')
      || e.includes('modell-architektur') || e.includes('model architecture')
      || e.includes('unsupported architecture'))
    return { category: 'architecture', title: t('trainingDashboard.errorRecovery.architectureCategoryTitle'), hint: t('trainingDashboard.errorRecovery.architectureCategoryHint') };

  // Geräte-Fehler nur bei echten Geräte-Meldungen, nicht bei jedem Vorkommen
  // von "mps"/"device" irgendwo in einem Traceback.
  if (e.includes('cuda error') || e.includes('cuda unavailable') || e.includes('no cuda')
      || e.includes('cuda is not available') || e.includes('mps not available')
      || e.includes('mps backend') || e.includes('device-side assert')
      || e.includes('no gpu') || e.includes('device not found')
      || /device .*(unavailable|not available|mismatch)/.test(e))
    return { category: 'cuda', title: t('trainingDashboard.errorRecovery.cudaCategoryTitle'), hint: t('trainingDashboard.errorRecovery.cudaCategoryHint') };
  if (e.includes('modulenotfounderror') || e.includes('importerror') || e.includes('no module')
      || e.includes('torchvision') || e.includes('versionskonflikt') || e.includes('version conflict'))
    return { category: 'packages', title: t('trainingDashboard.errorRecovery.packagesCategoryTitle'), hint: t('trainingDashboard.errorRecovery.packagesCategoryHint') };

  // Python-Fehlertypen VOR der Dataset-Pruefung. Ein Traceback aus einem
  // Dev-Train-Script nennt fast immer DATASET_PATH — vorher wurde deshalb
  // jeder NameError als "Dataset / Pfad Fehler" ausgegeben und der Nutzer
  // suchte den Fehler im Dataset statt in seinem Code.
  if (e.includes('syntaxerror') || e.includes('indentationerror') || e.includes('nameerror')
      || e.includes('typeerror') || e.includes('attributeerror') || e.includes('keyerror')
      || e.includes('indexerror') || e.includes('unboundlocalerror')
      || e.includes('zerodivisionerror') || e.includes('recursionerror'))
    return { category: 'code', title: t('trainingDashboard.errorRecovery.codeCategoryTitle'), hint: t('trainingDashboard.errorRecovery.codeCategoryHint') };

  // Nur echte Datei-/Dataset-Meldungen. Ein blosses "path" irgendwo im
  // Traceback reicht nicht — das steht in jedem Python-Stacktrace.
  if (e.includes('filenotfounderror') || e.includes('file not found')
      || e.includes('no such file') || e.includes('dataset')
      || e.includes('existiert nicht') || e.includes('keine daten-dateien')
      || e.includes('permission denied') || e.includes('isadirectoryerror'))
    return { category: 'dataset', title: t('trainingDashboard.errorRecovery.datasetCategoryTitle'), hint: t('trainingDashboard.errorRecovery.datasetCategoryHint') };

  // \b-Grenzen: sonst matchen deutsche Wörter wie "E**inf**ach" oder "Fi**nan**zen"
  if (/\bnan\b|\binf\b/.test(e) || e.includes('gradient') || e.includes('loss'))
    return { category: 'config', title: t('trainingDashboard.errorRecovery.configCategoryTitle'), hint: t('trainingDashboard.errorRecovery.configCategoryHint') };
  return { category: 'unknown', title: t('trainingDashboard.errorRecovery.unknownCategoryTitle'), hint: t('trainingDashboard.errorRecovery.unknownCategoryHint') };
}

// ── Big Loss Chart ────────────────────────────────────────────────────────

function BigLossChart({ points }: { points: LossPoint[] }) {
  const { t } = useLanguage();
  if (points.length < 2) {
    return (
      <div className="h-52 flex flex-col items-center justify-center gap-2">
        <Loader2 className="w-6 h-6 text-gray-600 animate-spin" />
        <p className="text-gray-600 text-sm">{t('trainingDashboard.chart.waitingForData')}</p>
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
        <text x="14" y="2" fill="rgba(255,255,255,0.45)" fontSize="10">{t('trainingDashboard.chart.legendTrain')}</text>
        {vals.length > 0 && (
          <>
            <line x1="90" y1="-2" x2="106" y2="-2" stroke="#a855f7" strokeWidth="2" strokeDasharray="4,2" />
            <text x="112" y="2" fill="rgba(255,255,255,0.45)" fontSize="10">{t('trainingDashboard.chart.legendVal')}</text>
          </>
        )}
      </g>
    </svg>
  );
}

// ── Config Summary ────────────────────────────────────────────────────────

function ConfigSummary({ config, mode }: { config?: Partial<TrainingConfig>; mode: 'standard' | 'dev' }) {
  const { t } = useLanguage();
  if (mode === 'dev') {
    return <p className="text-gray-600 text-xs italic">{t('trainingDashboard.config.devMode')}</p>;
  }
  if (!config) {
    return <p className="text-gray-600 text-xs italic">{t('trainingDashboard.config.notAvailable')}</p>;
  }
  const rows: { label: string; value: string | number | boolean | undefined; color: string }[] = [
    { label: t('trainingDashboard.config.epochs'),       value: config.epochs,                            color: 'text-emerald-400' },
    { label: t('trainingDashboard.config.batchSize'),    value: config.batch_size,                        color: 'text-blue-400' },
    { label: t('trainingDashboard.config.learningRate'), value: config.learning_rate?.toExponential(2),   color: 'text-purple-400' },
    { label: t('trainingDashboard.config.maxSeqLen'),   value: config.max_seq_length,                    color: 'text-amber-400' },
    { label: t('trainingDashboard.config.warmupRatio'),  value: config.warmup_ratio,                      color: 'text-cyan-400' },
    { label: t('trainingDashboard.config.gradAccum'),    value: config.gradient_accumulation_steps,       color: 'text-pink-400' },
    { label: t('trainingDashboard.config.optimizer'),     value: config.optimizer,                         color: 'text-emerald-400' },
    { label: t('trainingDashboard.config.scheduler'),     value: config.scheduler,                         color: 'text-blue-400' },
    { label: t('trainingDashboard.config.weightDecay'),  value: config.weight_decay,                      color: 'text-purple-400' },
    { label: t('trainingDashboard.config.maxGradNorm'), value: config.max_grad_norm,                     color: 'text-amber-400' },
    { label: t('trainingDashboard.config.dropout'),       value: config.dropout,                           color: 'text-cyan-400' },
    { label: t('trainingDashboard.config.seed'),          value: config.seed,                              color: 'text-gray-300' },
    { label: t('trainingDashboard.config.fp16'),          value: config.fp16 ? t('trainingDashboard.config.yes') : t('trainingDashboard.config.no'),             color: config.fp16 ? 'text-emerald-400' : 'text-gray-600' },
    { label: t('trainingDashboard.config.bf16'),          value: config.bf16 ? t('trainingDashboard.config.yes') : t('trainingDashboard.config.no'),             color: config.bf16 ? 'text-emerald-400' : 'text-gray-600' },
    { label: t('trainingDashboard.config.lora'),          value: config.use_lora ? `r=${config.lora_r}` : t('trainingDashboard.config.no'), color: config.use_lora ? 'text-violet-400' : 'text-gray-600' },
    { label: t('trainingDashboard.config.qlora'),  value: config.load_in_4bit ? t('trainingDashboard.config.yes') : t('trainingDashboard.config.no'),     color: config.load_in_4bit ? 'text-fuchsia-400' : 'text-gray-600' },
    { label: t('trainingDashboard.config.gradCheckpoint'), value: config.gradient_checkpointing ? t('trainingDashboard.config.yes') : t('trainingDashboard.config.no'), color: config.gradient_checkpointing ? 'text-emerald-400' : 'text-gray-600' },
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

const EVENT_ICONS: Record<string, React.ReactNode> = {
  start: <Rocket className="w-3.5 h-3.5" />,
  epoch: <BarChart3 className="w-3.5 h-3.5" />,
  checkpoint: <Save className="w-3.5 h-3.5" />,
  complete: <CheckCircle className="w-3.5 h-3.5" />,
  error: <XCircle className="w-3.5 h-3.5" />,
  stop: <Square className="w-3.5 h-3.5" />,
  info: <Info className="w-3.5 h-3.5" />,
};
const EVENT_COLORS: Record<string, string> = {
  start: 'text-violet-400', epoch: 'text-blue-400', checkpoint: 'text-amber-400',
  complete: 'text-emerald-400', error: 'text-red-400', stop: 'text-gray-400', info: 'text-gray-400',
};

// ── Error Recovery Panel ──────────────────────────────────────────────────

function ErrorRecoveryPanel({
  mode, errorMsg, config, events,
  onOpenKIAssistant, onSendCodeToKI, devScript,
}: {
  mode: 'standard' | 'dev';
  errorMsg: string;
  config?: Partial<TrainingConfig>;
  events?: SessionEvent[];
  onOpenKIAssistant?: () => void;
  onSendCodeToKI?: (script: string, error: string) => void;
  devScript?: string;
}) {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);
  const [sendState, setSendState] = useState<'idle' | 'sending' | 'sent' | 'failed'>('idle');
  const [errorExpanded, setErrorExpanded] = useState(false);
  const { category, title, hint } = analyzeError(errorMsg, t);
  // Mehr als drei Zeilen? Dann lohnt der Aufklapp-Button.
  const errorIsLong = errorMsg.split('\n').length > 3 || errorMsg.length > 240;

  // Fehler-Logs an das FrameTrain-Team senden
  const handleSendReport = async () => {
    if (sendState === 'sending' || sendState === 'sent') return;
    setSendState('sending');
    try {
      const ok = await sendAppErrorReport({
        error_type: `${mode === 'dev' ? 'devtrain' : 'training'}:${category}`,
        title: t('trainingDashboard.errorRecovery.reportHeader'),
        message: errorMsg,
        details: buildDiagReport(),
        logs: (events ?? []).slice(0, 50).map(ev => `${ev.time} [${ev.type}] ${ev.message}`).join('\n'),
        ...(mode === 'dev' && devScript ? { script_full: devScript } : {}),
        error_analysis: category,
        error_category: title,
        ...(config ? { config: config as Record<string, unknown> } : {}),
      });
      setSendState(ok ? 'sent' : 'failed');
    } catch {
      setSendState('failed');
    }
  };

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
      t('trainingDashboard.errorRecovery.reportHeader'),
      t('trainingDashboard.errorRecovery.reportMode').replace('{mode}', mode === 'dev' ? t('trainingDashboard.modeDev') : t('trainingDashboard.modeStandard')),
      t('trainingDashboard.errorRecovery.reportCategory').replace('{category}', title),
      t('trainingDashboard.errorRecovery.reportError').replace('{error}', errorMsg),
      '',
      t('trainingDashboard.errorRecovery.reportConfigSection'),
      ...(config ? Object.entries(config).map(([k, v]) => `${k}: ${v}`) : [t('trainingDashboard.errorRecovery.reportConfigDev')]),
      '',
      t('trainingDashboard.errorRecovery.reportEnd'),
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

  // Echter "An FrameTrain senden"-Button (Standard- und Dev-Modus)
  const sendReportButton = (
    <button
      onClick={handleSendReport}
      disabled={sendState === 'sending' || sendState === 'sent'}
      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/25 transition-all text-left disabled:cursor-default"
    >
      {sendState === 'sending'
        ? <Loader2 className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0" />
        : sendState === 'sent'
          ? <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          : <Send className="w-4 h-4 text-blue-400 flex-shrink-0" />}
      <div>
        <p className={`text-xs font-medium ${sendState === 'sent' ? 'text-emerald-300' : sendState === 'failed' ? 'text-red-300' : 'text-blue-300'}`}>
          {sendState === 'sent'
            ? t('trainingDashboard.errorRecovery.sendReportSent')
            : sendState === 'sending'
              ? t('trainingDashboard.errorRecovery.sendReportSending')
              : sendState === 'failed'
                ? t('trainingDashboard.errorRecovery.sendReportFailed')
                : t('trainingDashboard.errorRecovery.sendToTeamTitle')}
        </p>
        <p className="text-gray-500 text-[10px]">{t('trainingDashboard.errorRecovery.sendToTeamDesc')}</p>
      </div>
    </button>
  );

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
        {/* break-all: lange Dateipfade wurden sonst abgeschnitten statt umgebrochen,
            wodurch der Python-Traceback unsichtbar blieb. */}
        <pre
          className={`text-red-300/80 text-[10px] font-mono whitespace-pre-wrap break-all leading-relaxed ${
            errorExpanded ? 'max-h-72 overflow-y-auto' : 'line-clamp-3'
          }`}
        >
          {errorMsg}
        </pre>
        {errorIsLong && (
          <button
            onClick={() => setErrorExpanded(v => !v)}
            className="mt-2 text-[10px] text-red-300/70 hover:text-red-200 underline underline-offset-2"
          >
            {errorExpanded
              ? t('trainingDashboard.errorRecovery.showLess')
              : t('trainingDashboard.errorRecovery.showFull')}
          </button>
        )}
      </div>

      {/* Actions */}
      <div className="p-3 space-y-2">
        <p className="text-gray-500 text-[10px] uppercase tracking-wide font-medium mb-2">{t('trainingDashboard.errorRecovery.optionsTitle')}</p>

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
                  <p className="text-violet-300 text-xs font-medium">{t('trainingDashboard.errorRecovery.kiMetricsButton')}</p>
                  <p className="text-gray-500 text-[10px]">{t('trainingDashboard.errorRecovery.kiMetricsDesc')}</p>
                </div>
              </button>
            )}

            {/* Memory-specific: suggest LoRA */}
            {(category === 'memory' || category === 'cuda') && config && !config.use_lora && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <MemoryStick className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-amber-300 text-xs font-medium">{t('trainingDashboard.errorRecovery.ramTipTitle')}</p>
                  <p className="text-gray-400 text-[10px]">{t('trainingDashboard.errorRecovery.ramTipDesc')}</p>
                </div>
              </div>
            )}

            {/* Dataset warning */}
            {category === 'dataset' && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <Database className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-blue-300 text-xs font-medium">{t('trainingDashboard.errorRecovery.datasetTipTitle')}</p>
                  <p className="text-gray-400 text-[10px]">{t('trainingDashboard.errorRecovery.datasetTipDesc')}</p>
                </div>
              </div>
            )}

            {/* Report kopieren */}
            <button
              onClick={handleCopyReport}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-left"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" /> : <Copy className="w-4 h-4 text-gray-400 flex-shrink-0" />}
              <div>
                <p className={`text-xs font-medium ${copied ? 'text-emerald-300' : 'text-gray-300'}`}>
                  {copied ? t('trainingDashboard.errorRecovery.copyReportCopied') : t('trainingDashboard.errorRecovery.copyReportButton')}
                </p>
                <p className="text-gray-600 text-[10px]">{t('trainingDashboard.errorRecovery.copyReportDesc')}</p>
              </div>
            </button>

            {/* Error-Logs an das FrameTrain-Team senden */}
            {sendReportButton}
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
                  <p className="text-cyan-300 text-xs font-medium">{t('trainingDashboard.errorRecovery.codeFixButton')}</p>
                  <p className="text-gray-500 text-[10px]">{t('trainingDashboard.errorRecovery.codeFixDesc')}</p>
                </div>
              </button>
            )}

            {/* Memory hint for dev mode */}
            {(category === 'memory' || category === 'cuda') && (
              <div className="flex items-start gap-3 px-3 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <MemoryStick className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-amber-300 text-xs font-medium">{t('trainingDashboard.errorRecovery.memoryHintTitle')}</p>
                  <p className="text-gray-400 text-[10px]">{t('trainingDashboard.errorRecovery.memoryHintDesc')}</p>
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
                  {copied ? t('trainingDashboard.errorRecovery.copied') : t('trainingDashboard.errorRecovery.copyReportButton')}
                </p>
                <p className="text-gray-600 text-[10px]">{t('trainingDashboard.errorRecovery.copyReportDevDesc')}</p>
              </div>
            </button>

            {/* Error-Logs an das FrameTrain-Team senden */}
            {sendReportButton}
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
  const { t, language } = useLanguage();
  const { currentTheme } = useTheme();
  const [elapsed, setElapsed] = useState(0);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [showFullConfig, setShowFullConfig] = useState(false);
  const prevEpochRef = useRef(-1);
  const prevStatusRef = useRef('');
  const eventsRef = useRef<SessionEvent[]>([]);

  eventsRef.current = events;

  // Die Uhr laeuft nur, solange das Training laeuft. Vorher zaehlte sie auch
  // nach einem Abbruch weiter — ein sofort gescheitertes Dev-Script zeigte so
  // spaeter "Laufzeit 6h 43m".
  const isTerminal =
    job?.status === 'completed' || job?.status === 'failed' || job?.status === 'stopped';

  useEffect(() => {
    if (!isOpen || isTerminal) return;
    setElapsed(Date.now() - startedAt);
    const id = setInterval(() => setElapsed(Date.now() - startedAt), 1000);
    return () => clearInterval(id);
  }, [isOpen, startedAt, isTerminal]);


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
          newEvents = [{ time: new Date().toISOString(), type: 'start', message: t('trainingDashboard.eventLog.eventStart').replace('{model}', modelName) }, ...newEvents];
          changed = true;
        }
      } else if (job.status === 'completed') {
        const lastLoss = lossPoints[lossPoints.length - 1];
        newEvents = [{ time: new Date().toISOString(), type: 'complete', message: t('trainingDashboard.eventLog.eventComplete').replace('{loss}', lastLoss?.train_loss?.toFixed(4) ?? '—').concat(lastLoss?.val_loss != null ? t('trainingDashboard.eventLog.eventCompleteVal').replace('{val}', lastLoss.val_loss.toFixed(4)) : '') }, ...newEvents];
        changed = true;
      } else if (job.status === 'failed') {
        newEvents = [{ time: new Date().toISOString(), type: 'error', message: t('trainingDashboard.eventLog.eventError').replace('{error}', job.error ?? t('common.unknown')) }, ...newEvents];
        changed = true;
      } else if (job.status === 'stopped') {
        newEvents = [{ time: new Date().toISOString(), type: 'stop', message: t('trainingDashboard.eventLog.eventStopped') }, ...newEvents];
        changed = true;
      }
    }

    if (progress && progress.epoch !== prevEpochRef.current && progress.epoch > 0) {
      prevEpochRef.current = progress.epoch;
      const valStr = progress.val_loss != null ? ` · Val: ${progress.val_loss.toFixed(4)}` : '';
      newEvents = [{ time: new Date().toISOString(), type: 'epoch', message: t('trainingDashboard.eventLog.eventEpoch').replace('{epoch}', String(progress.epoch)).replace('{total}', String(progress.total_epochs)).replace('{trainLoss}', progress.train_loss?.toFixed(4) ?? '—').concat(valStr ? t('trainingDashboard.eventLog.eventEpochVal').replace('{val}', progress.val_loss?.toFixed(4) ?? '—') : '') }, ...newEvents];
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

  // Wurde nie live gemessen (Dashboard erst nach dem Ende geoeffnet), ist eine
  // Zahl geraten — dann lieber "—" zeigen als eine erfundene Laufzeit.
  const durationLabel = isTerminal && elapsed === 0 ? '—' : formatDuration(elapsed);

  // Eigene Dev-Train-Scripts melden meist nur step/total_steps und kein
  // progress_percent — der Balken blieb dann bei 0 %, obwohl "Step 30 / 60"
  // danebenstand. Fehlt der Prozentwert, wird er aus den Schritten abgeleitet.
  // Der Schrittzaehler blieb in Einzelfaellen hinter dem Loss-Verlauf zurueck
  // ("Step 30 / 60", waehrend der Chart schon Punkt 60 zeigte). Solange die
  // Ursache nicht gefunden ist, gilt der weiter fortgeschrittene der beiden
  // Werte — die Anzeige widerspricht sich damit nicht mehr selbst.
  const lastLossStep = lossPoints.length ? (lossPoints[lossPoints.length - 1].step ?? 0) : 0;
  const shownStep = Math.max(progress?.step ?? 0, lastLossStep);

  const percent = (() => {
    if (!progress) return 0;
    const given = progress.progress_percent ?? 0;
    if (given > 0) return Math.min(100, given);
    const step = shownStep;
    const total = progress.total_steps ?? 0;
    if (total > 0 && step > 0) return Math.min(100, (step / total) * 100);
    const ep = progress.epoch ?? 0;
    const eps = progress.total_epochs ?? 0;
    return eps > 0 && ep > 0 ? Math.min(100, (ep / eps) * 100) : 0;
  })();

  const eta = (() => {
    if (!progress || !isRunning || percent <= 1) return null;
    const elapsedSec = elapsed / 1000;
    const totalSec = elapsedSec / (percent / 100);
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
            {isRunning ? t('trainingDashboard.statusRunning') : isCompleted ? t('trainingDashboard.statusCompleted') : isFailed ? t('trainingDashboard.statusFailed') : t('trainingDashboard.statusStopped')}
          </p>
          {progress && (
            <p className="text-gray-500 text-[10px]">
              {t('trainingDashboard.minimized.epochInfo').replace('{epoch}', String(progress.epoch)).replace('{total}', String(progress.total_epochs)).replace('{loss}', progress.train_loss?.toFixed(4) ?? '—').replace('{duration}', durationLabel)}
            </p>
          )}
        </div>
        {progress && (
          <div className="w-24 h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${percent}%` }} />
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
                {isRunning ? t('trainingDashboard.statusRunning') : isCompleted ? t('trainingDashboard.statusCompleted') : isFailed ? t('trainingDashboard.statusFailed') : isStopped ? t('trainingDashboard.statusStopped') : t('trainingDashboard.titleDefault')}
              </h2>
              <p className="text-gray-500 text-xs">
                {modelName} · {datasetName} · {mode === 'dev' ? t('trainingDashboard.modeDev') : t('trainingDashboard.modeStandard')}
                {progress && ` · ${Math.round(percent)}%`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isRunning && (
              <button onClick={onStop} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-300 text-xs font-medium transition-all">
                <Square className="w-3.5 h-3.5" /> {t('trainingDashboard.header.stopButton')}
              </button>
            )}
            <button onClick={onMinimize} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all" title={t('trainingDashboard.header.minimizeTooltip')}>
              <Minimize2 className="w-4 h-4" />
            </button>
            {/* Close button – nur wenn nicht am Laufen */}
            {isDone && onClose && (
              <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all" title={t('trainingDashboard.header.closeTooltip')}>
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">

          {/* Metrics strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: t('trainingDashboard.metrics.trainLoss'), value: progress?.train_loss?.toFixed(4) ?? '—', sub: lossImprovement != null ? `${lossImprovement > 0 ? '↓' : '↑'} ${Math.abs(lossImprovement).toFixed(1)}% ${t('trainingDashboard.metrics.trainLossSub').replace('{dir}', '').replace('{pct}', '')}` : undefined, icon: <TrendingDown className="w-4 h-4" />, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
              { label: t('trainingDashboard.metrics.valLoss'),   value: progress?.val_loss?.toFixed(4) ?? '—', icon: <BarChart3 className="w-4 h-4" />, color: 'text-purple-400', bg: 'bg-purple-500/10 border-purple-500/20' },
              { label: t('trainingDashboard.metrics.learningRate'), value: progress?.learning_rate?.toExponential(2) ?? (config?.learning_rate?.toExponential(2) ?? '—'), icon: <Zap className="w-4 h-4" />, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20' },
              { label: t('trainingDashboard.metrics.duration'),  value: durationLabel, sub: eta ? t('trainingDashboard.metrics.eta').replace('{eta}', eta) : (isCompleted ? t('trainingDashboard.metrics.completed') : isStopped ? t('trainingDashboard.metrics.stopped') : undefined), icon: <Clock className="w-4 h-4" />, color: 'text-blue-400', bg: 'bg-blue-500/10 border-blue-500/20' },
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
                <span>{t('trainingDashboard.progress.epochStep').replace('{epoch}', String(progress.epoch)).replace('{totalEpochs}', String(progress.total_epochs)).replace('{step}', String(shownStep)).replace('{totalSteps}', String(progress.total_steps))}</span>
                <span className="font-mono">{Math.round(percent)}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-white/10 overflow-hidden">
                <div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${percent}%` }} />
              </div>
            </div>
          )}

          {/* Error Recovery Panel — nur bei Fehler (nicht bei manuellem Stopp) */}
          {isFailed && job?.error && (
            <ErrorRecoveryPanel
              mode={mode}
              errorMsg={job.error}
              config={config}
              events={events}
              onOpenKIAssistant={onOpenKIAssistant}
              devScript={devScript}
              onSendCodeToKI={onSendCodeToKI}
            />
          )}

          {/* Manuell gestoppt – einfacher Hinweis */}
          {isStopped && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 border border-white/10">
              <Square className="w-4 h-4 text-gray-400 flex-shrink-0" />
              <p className="text-gray-400 text-sm">{t('trainingDashboard.stopped.message')}</p>
            </div>
          )}

          {/* Abgeschlossen – Erfolgsmeldung + Analyse starten */}
          {isCompleted && (
            <div className="space-y-2">
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <p className="text-emerald-300 text-sm font-medium">{t('trainingDashboard.completed.message')}</p>
              </div>
              {completedVersionId && onNavigateToAnalysis && (
                <button
                  onClick={() => onNavigateToAnalysis(completedVersionId)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:opacity-90 text-white font-semibold text-sm transition-all shadow-lg"
                >
                  <BarChart3 className="w-4 h-4" /> {t('trainingDashboard.completed.analyzeButton')}
                </button>
              )}
            </div>
          )}

          {/* Chart + Config */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-medium text-gray-400 flex items-center gap-1.5"><TrendingDown className="w-3.5 h-3.5 text-emerald-400" /> {t('trainingDashboard.chart.title')}</p>
                <span className="text-[10px] text-gray-600">{t('trainingDashboard.chart.points').replace('{count}', String(lossPoints.length))}</span>
              </div>
              <BigLossChart points={lossPoints} />
              {lossPoints.length >= 2 && (
                <div className="flex items-center gap-4 text-[10px] text-gray-500 border-t border-white/8 pt-2">
                  <span>{t('trainingDashboard.chart.start')} <span className="text-gray-300 font-mono">{firstLoss?.toFixed(4)}</span></span>
                  <span>{t('trainingDashboard.chart.current')} <span className="text-gray-300 font-mono">{lastLoss?.toFixed(4)}</span></span>
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
                <span className="flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-blue-400" /> {t('trainingDashboard.config.title')}</span>
                {showFullConfig ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
              <ConfigSummary config={config} mode={mode} />
            </div>
          </div>

          {/* Event Log */}
          <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/10">
              <Clock className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-xs font-medium text-gray-400">{t('trainingDashboard.eventLog.title')}</span>
              <span className="ml-auto text-[10px] text-gray-600">{t('trainingDashboard.eventLog.entriesNote').replace('{count}', String(events.length))}</span>
            </div>
            <div className="max-h-40 overflow-y-auto p-3 space-y-1.5">
              {events.length === 0 ? (
                <p className="text-gray-600 text-xs text-center py-4 italic">{t('trainingDashboard.eventLog.waiting')}</p>
              ) : events.map((ev, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px]">
                  <span className="text-gray-600 tabular-nums flex-shrink-0 font-mono text-[10px]">
                    {new Date(ev.time).toLocaleTimeString(dateLocale(language), { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  <span className="flex-shrink-0 text-[13px] leading-[1.1]">{EVENT_ICONS[ev.type] ?? EVENT_ICONS.info}</span>
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
