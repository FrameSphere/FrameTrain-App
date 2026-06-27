// TrainingPanel.tsx – Vollständiges Training-Interface (v5 – LoRA/QLoRA + Error Recovery)

import { useState, useEffect, useRef, useCallback, useContext } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Settings2, Loader2, ChevronDown, ChevronRight,
  Gauge, TrendingDown, Zap, AlertCircle, CheckCircle, Sparkles,
  Trash2, X, HelpCircle, BarChart3, MemoryStick, SlidersHorizontal,
  BookOpen, Code2, RefreshCw, Folder, XCircle, Clock, Lightbulb,
  AlertTriangle, Layers, Save, Plus, ChevronRight as ChevronRightIcon,
  ClipboardList, History, Check,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { useTrainingContext } from '../contexts/TrainingContext';
import { useLanguage, type Language } from '../contexts/LanguageContext';
import { openAICoach } from '../ai/aiCoachEvents';
import { detectPlugin } from '../plugins/registry';
import { checkDatasetCompat } from '../plugins/datasetCompat';
import DatasetCompatBadge from './DatasetCompatBadge';
import DevTrainPanel from './DevTrainPanel';
// TrainingDashboard wird global in Dashboard.tsx gerendert

// ── Types ──────────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: string; name: string; source: string;
  source_path: string | null; local_path: string;
  model_type: string | null; size_bytes?: number;
}

interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }
interface VersionTreeItem { id: string; name: string; is_root: boolean; version_number: number; }

export interface DatasetInfo {
  id: string; name: string; model_id: string;
  status: 'unused' | 'split'; file_count: number;
  size_bytes: number; extensions?: string[]; storage_path?: string;
  // v2: Typ-System
  dataset_type?: import('../plugins/datasetCompatHelpers').DatasetType;
  pairing_status?: import('../plugins/datasetCompatHelpers').PairingStatus | null;
  warnings?: string[];
}

export interface TrainingConfig {
  // Basis
  epochs: number; batch_size: number; learning_rate: number;
  weight_decay: number; warmup_ratio: number; warmup_steps: number;
  max_steps: number; max_seq_length: number;
  gradient_accumulation_steps: number;
  fp16: boolean; bf16: boolean;
  // Optimizer
  optimizer: string; scheduler: string;
  adam_beta1: number; adam_beta2: number; adam_epsilon: number;
  // Regularisierung
  dropout: number; max_grad_norm: number; label_smoothing: number;
  // Evaluation & Saving
  eval_strategy: string; eval_steps: number;
  save_steps: number; save_total_limit: number; logging_steps: number;
  seed: number;
  // Datenlader
  num_workers: number; pin_memory: boolean;
  // Flags
  gradient_checkpointing: boolean; group_by_length: boolean;
  // LoRA / QLoRA
  use_lora: boolean; lora_r: number; lora_alpha: number;
  lora_dropout: number; lora_target_modules: string;
  load_in_4bit: boolean; load_in_8bit: boolean;

  // Plugin-Routing (Backend)
  task_type?: string;
  plugin_config?: Record<string, unknown>;
}

export interface TrainingProgress {
  epoch: number; total_epochs: number; step: number; total_steps: number;
  train_loss: number; val_loss: number | null; learning_rate: number; progress_percent: number;
}

export interface TrainingJob {
  id: string;
  model_id?: string;
  model_name?: string;
  dataset_id?: string;
  dataset_name?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  config?: TrainingConfig;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  output_path?: string | null;
  progress: TrainingProgress;
  error: string | null;
}

interface MetricsTemplate {
  id: string; name: string; description?: string;
  config: Partial<TrainingConfig>; source?: string; created_at?: string;
}

interface RequirementsCheck {
  python_installed: boolean; python_version: string; torch_installed: boolean;
  torch_version: string; transformers_installed: boolean; cuda_available: boolean;
  mps_available: boolean; ready: boolean;
}

export interface LossPoint { step: number; epoch: number; train_loss: number; val_loss?: number; }
interface TrainingPanelProps { 
  userData?: { userId: string; email: string; apiKey: string; password: string };
  onNavigateToAnalysis: (versionId: string) => void; 
}

// ── AI Helper ─────────────────────────────────────────────────────────────

import type { AISettings } from '../contexts/AISettingsContext';
import { callAI as callAIClient } from '../ai/aiClient';

export async function callAI(settings: AISettings, systemPrompt: string, userPrompt: string, history?: { role: 'user' | 'assistant'; content: string }[], responseLanguage?: Language): Promise<string> {
  const messages = [...(history ?? []), { role: 'user' as const, content: userPrompt }];
  return callAIClient(settings, { system: systemPrompt, messages, maxTokens: 2000, temperature: 0.7, responseLanguage });
}

// ── Defaults ───────────────────────────────────────────────────────────────

export const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 3, batch_size: 8, learning_rate: 2e-5, weight_decay: 0.01,
  warmup_ratio: 0.06, warmup_steps: 0, max_steps: -1,
  max_seq_length: 128, gradient_accumulation_steps: 1,
  fp16: false, bf16: false,
  optimizer: 'adamw', scheduler: 'linear',
  adam_beta1: 0.9, adam_beta2: 0.999, adam_epsilon: 1e-8,
  dropout: 0.1, max_grad_norm: 1.0, label_smoothing: 0.0,
  eval_strategy: 'epoch', eval_steps: 500, save_steps: 500,
  save_total_limit: 3, logging_steps: 10, seed: 42,
  num_workers: 4, pin_memory: true,
  gradient_checkpointing: false, group_by_length: false,
  use_lora: false, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05,
  lora_target_modules: 'q_proj,v_proj',
  load_in_4bit: false, load_in_8bit: false,
};

function getBuiltinTemplates(t: (key: string) => string): MetricsTemplate[] {
  return [
    { id: 'standard', name: t('trainingPanel.templates.builtinTemplates.standard.name'), description: t('trainingPanel.templates.builtinTemplates.standard.description'), config: { epochs: 3, batch_size: 8, learning_rate: 2e-5, warmup_ratio: 0.06, max_seq_length: 128 }, source: 'builtin' },
    { id: 'small', name: t('trainingPanel.templates.builtinTemplates.small.name'), description: t('trainingPanel.templates.builtinTemplates.small.description'), config: { epochs: 5, batch_size: 8, learning_rate: 3e-5, warmup_ratio: 0.1, max_seq_length: 64 }, source: 'builtin' },
    { id: 'large', name: t('trainingPanel.templates.builtinTemplates.large.name'), description: t('trainingPanel.templates.builtinTemplates.large.description'), config: { epochs: 2, batch_size: 32, learning_rate: 1e-5, warmup_ratio: 0.04, max_seq_length: 256 }, source: 'builtin' },
    { id: 'lowram', name: t('trainingPanel.templates.builtinTemplates.lowram.name'), description: t('trainingPanel.templates.builtinTemplates.lowram.description'), config: { epochs: 4, batch_size: 2, learning_rate: 2e-5, gradient_accumulation_steps: 8, max_seq_length: 64, fp16: true, gradient_checkpointing: true }, source: 'builtin' },
    { id: 'lora', name: t('trainingPanel.templates.builtinTemplates.lora.name'), description: t('trainingPanel.templates.builtinTemplates.lora.description'), config: { use_lora: true, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05, gradient_checkpointing: true, batch_size: 4 }, source: 'builtin' },
    { id: 'qlora', name: t('trainingPanel.templates.builtinTemplates.qlora.name'), description: t('trainingPanel.templates.builtinTemplates.qlora.description'), config: { use_lora: true, load_in_4bit: true, lora_r: 16, lora_alpha: 32, lora_dropout: 0.05, gradient_checkpointing: true, batch_size: 2, gradient_accumulation_steps: 8 }, source: 'builtin' },
  ];
}

// ── UI Atoms ──────────────────────────────────────────────────────────────

function SectionCard({ title, icon, expanded, onToggle, children, badge }: { title: string; icon: React.ReactNode; expanded: boolean; onToggle: () => void; children: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      <button onClick={onToggle} className="w-full flex items-center justify-between p-4 hover:bg-white/[0.03] transition-all">
        <div className="flex items-center gap-3">{icon}<span className="font-medium text-white text-sm">{title}</span>{badge}</div>
        {expanded ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRightIcon className="w-4 h-4 text-gray-400" />}
      </button>
      {expanded && <div className="px-4 pb-5 space-y-4 border-t border-white/10 pt-4">{children}</div>}
    </div>
  );
}

function Field({ label, tooltip, children }: { label: string; tooltip?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <label className="text-xs text-gray-400">{label}</label>
        {tooltip && (
          <div className="group relative">
            <HelpCircle className="w-3 h-3 text-gray-600 cursor-help" />
            <div className="absolute left-0 bottom-full mb-2 w-52 p-2.5 bg-slate-800 border border-white/10 rounded-xl text-xs text-gray-300 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-20 shadow-xl">{tooltip}</div>
          </div>
        )}
      </div>
      {children}
    </div>
  );
}

function NumInput({ value, onChange, min, max, step = 'any' }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number | 'any' }) {
  return <input type="number" value={value} min={min} max={max} step={step} onChange={e => onChange(parseFloat(e.target.value) || 0)} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500/50 transition-all" />;
}

function SelectInput({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500/50 transition-all appearance-none">
      {options.map(o => <option key={o.value} value={o.value} className="bg-slate-900">{o.label}</option>)}
    </select>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-xs text-gray-400">{label}</span>
      <button onClick={() => onChange(!checked)} className={`relative w-10 rounded-full transition-all ${checked ? 'bg-emerald-500' : 'bg-white/10'}`} style={{ height: '22px', minWidth: '40px' }}>
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-[22px]' : 'translate-x-0.5'}`} />
      </button>
    </div>
  );
}

// ── RAM Calculator ─────────────────────────────────────────────────────────

function RamCalculator({ config, modelSizeGb }: { config: TrainingConfig; modelSizeGb: number }) {
  const { t } = useLanguage();
  const isFp16 = config.fp16 || config.bf16;
  const isQuantized = config.load_in_4bit || config.load_in_8bit;
  const quantFactor = config.load_in_4bit ? 0.25 : config.load_in_8bit ? 0.5 : 1.0;
  const weightRam = modelSizeGb * quantFactor * (isFp16 ? 0.5 : 1.0);
  const gradRam = config.use_lora ? modelSizeGb * 0.05 : modelSizeGb;
  const optimizerRam = config.use_lora ? modelSizeGb * 0.1 : modelSizeGb * 2;
  const activationBase = modelSizeGb > 1.5 ? 1.0 : 0.4;
  const activationRam = (config.batch_size / 8) * activationBase * (isFp16 ? 0.5 : 1.0) * (config.gradient_checkpointing ? 0.4 : 1.0);
  const total = weightRam + gradRam + optimizerRam + activationRam + 0.4;
  const color = total > 20 ? 'text-red-400' : total > 12 ? 'text-amber-400' : total > 6 ? 'text-yellow-400' : 'text-emerald-400';

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 space-y-3">
      <div className="flex items-center gap-2">
        <MemoryStick className="w-4 h-4 text-blue-400" />
        <span className="text-sm font-medium text-white">{t('trainingPanel.ramCalculator.title')}</span>
        <span className="text-xs text-gray-500">{t('trainingPanel.ramCalculator.subtitle').replace('{sizeGb}', modelSizeGb.toFixed(2)).replace('{lora}', config.use_lora ? t('trainingPanel.ramCalculator.loraLabel') : '').replace('{quant}', isQuantized ? t('trainingPanel.ramCalculator.quantLabel').replace('{bits}', config.load_in_4bit ? '4' : '8') : '')}</span>
      </div>
      <div className="space-y-1.5 text-xs">
        {[
          [t('trainingPanel.ramCalculator.weights'), weightRam],
          [config.use_lora ? t('trainingPanel.ramCalculator.gradientsLora') : t('trainingPanel.ramCalculator.gradients'), gradRam],
          [config.use_lora ? t('trainingPanel.ramCalculator.optimizerLora') : t('trainingPanel.ramCalculator.optimizer'), optimizerRam],
          [t('trainingPanel.ramCalculator.activations').replace('{batch}', String(config.batch_size)).replace('{gradCkpt}', config.gradient_checkpointing ? t('trainingPanel.ramCalculator.gradCkptLabel') : ''), activationRam],
          [t('trainingPanel.ramCalculator.misc'), 0.4],
        ].map(([l, v]) => (
          <div key={l as string} className="flex justify-between"><span className="text-gray-400">{l as string}</span><span className="text-gray-300 tabular-nums">{(v as number).toFixed(2)} GB</span></div>
        ))}
        <div className="flex justify-between pt-2 border-t border-white/10 font-semibold"><span className="text-gray-300">{t('trainingPanel.ramCalculator.total')}</span><span className={`${color} tabular-nums`}>~{total.toFixed(1)} GB</span></div>
      </div>
      {!isFp16 && !config.use_lora && total > 8 && (
        <p className="text-amber-400 text-xs bg-amber-500/10 rounded-lg px-3 py-2 flex items-center gap-2">
          <Lightbulb className="w-3.5 h-3.5 flex-shrink-0" />
          {t('trainingPanel.ramCalculator.fp16Tip').replace('{save}', (gradRam * 0.5 + activationRam * 0.5).toFixed(1))}
        </p>
      )}
      {!config.use_lora && total > 12 && (
        <p className="text-violet-300 text-xs bg-violet-500/10 rounded-lg px-3 py-2 flex items-center gap-2">
          <Lightbulb className="w-3.5 h-3.5 flex-shrink-0" />
          {t('trainingPanel.ramCalculator.loraTip').replace('{save}', (total * 0.15).toFixed(1))}
        </p>
      )}
      {total > 20 && <div className="text-red-300 text-xs bg-red-500/10 rounded-lg px-3 py-2 space-y-1"><p className="font-medium">{t('trainingPanel.ramCalculator.highRamTitle')}</p><p>{t('trainingPanel.ramCalculator.highRamDesc')}</p></div>}
    </div>
  );
}

// ── Loss Chart ─────────────────────────────────────────────────────────────

export function LossChart({ points }: { points: LossPoint[] }) {
  const { t } = useLanguage();
  if (points.length < 2) return <div className="h-32 flex items-center justify-center text-gray-600 text-xs">{t('trainingPanel.lossChart.waitingForData')}</div>;
  const W = 500; const H = 120; const PAD = { l: 40, r: 12, t: 12, b: 28 };
  const iW = W - PAD.l - PAD.r; const iH = H - PAD.t - PAD.b;
  const trains = points.map(p => p.train_loss); const vals = points.map(p => p.val_loss).filter((v): v is number => v != null);
  const all = [...trains, ...vals]; const minV = Math.min(...all) * 0.95; const maxV = Math.max(...all) * 1.05;
  const toX = (i: number) => PAD.l + (i / (points.length - 1)) * iW;
  const toY = (v: number) => PAD.t + iH - ((v - minV) / (maxV - minV || 1)) * iH;
  const trainPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i)},${toY(p.train_loss)}`).join(' ');
  const valPath = points.filter(p => p.val_loss != null).map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(points.indexOf(p))},${toY(p.val_loss!)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 128 }}>
      {[0.25, 0.5, 0.75].map(f => <line key={f} x1={PAD.l} x2={W - PAD.r} y1={PAD.t + iH * f} y2={PAD.t + iH * f} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />)}
      {[0, 0.5, 1].map(f => <text key={f} x={PAD.l - 4} y={PAD.t + iH * f + 4} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="9">{(maxV - f * (maxV - minV)).toFixed(3)}</text>)}
      <path d={trainPath} fill="none" stroke="#10b981" strokeWidth="2" strokeLinejoin="round" />
      {vals.length > 0 && <path d={valPath} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="4,2" strokeLinejoin="round" />}
      <circle cx={PAD.l + 4} cy={H - 10} r="4" fill="#10b981" /><text x={PAD.l + 12} y={H - 6} fill="rgba(255,255,255,0.5)" fontSize="9">{t('trainingPanel.lossChart.legendTrain')}</text>
      {vals.length > 0 && <><line x1={PAD.l + 70} y1={H - 10} x2={PAD.l + 84} y2={H - 10} stroke="#a855f7" strokeWidth="2" strokeDasharray="3,2" /><text x={PAD.l + 88} y={H - 6} fill="rgba(255,255,255,0.5)" fontSize="9">{t('trainingPanel.lossChart.legendVal')}</text></>}
    </svg>
  );
}

// ── Templates Modal ────────────────────────────────────────────────────────

function TemplatesModal({ onApply, onClose, onSave, currentConfig }: { onApply: (cfg: Partial<TrainingConfig>) => void; onClose: () => void; onSave: (name: string, desc: string) => void; currentConfig: TrainingConfig; }) {
  const { t } = useLanguage();
  const [userTemplates, setUserTemplates] = useState<MetricsTemplate[]>([]);
  const [tab, setTab] = useState<'builtin' | 'user'>('builtin');
  const [saveName, setSaveName] = useState('');
  const [saveDesc, setSaveDesc] = useState('');
  const [showSaveForm, setShowSaveForm] = useState(false);
  const { success, error } = useNotification();

  useEffect(() => { invoke<MetricsTemplate[]>('get_metrics_templates').then(setUserTemplates).catch(() => {}); }, []);

  const handleDelete = async (id: string) => {
    try { await invoke('delete_metrics_template', { templateId: id }); setUserTemplates(t => t.filter(x => x.id !== id)); success(t('trainingPanel.templates.deleteSuccess'), ''); }
    catch (e) { error(t('trainingPanel.templates.deleteError'), String(e)); }
  };

  const handleSave = () => {
    if (!saveName.trim()) return;
    onSave(saveName.trim(), saveDesc.trim());
    setSaveName(''); setSaveDesc(''); setShowSaveForm(false);
    setTimeout(() => invoke<MetricsTemplate[]>('get_metrics_templates').then(setUserTemplates).catch(() => {}), 600);
  };

  void currentConfig;
  const all = tab === 'builtin' ? getBuiltinTemplates(t) : userTemplates;
  const builtinIcon = (id: string) => {
    const cls = 'w-4 h-4';
    switch (id) {
      case 'standard': return <BookOpen className={`${cls} text-blue-300`} />;
      case 'small': return <Zap className={`${cls} text-amber-300`} />;
      case 'large': return <BarChart3 className={`${cls} text-purple-300`} />;
      case 'lowram': return <MemoryStick className={`${cls} text-cyan-300`} />;
      case 'lora': return <Layers className={`${cls} text-violet-300`} />;
      case 'qlora': return <Zap className={`${cls} text-fuchsia-300`} />;
      default: return <BookOpen className={`${cls} text-gray-300`} />;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[85vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><BookOpen className="w-5 h-5 text-blue-400" /><h2 className="text-lg font-bold text-white">{t('trainingPanel.templates.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex border-b border-white/10 px-6 flex-shrink-0">
          {(['builtin', 'user'] as const).map(tabKey => (
            <button key={tabKey} onClick={() => setTab(tabKey)} className={`px-4 py-3 text-sm font-medium border-b-2 transition-all ${tab === tabKey ? 'text-blue-300 border-blue-400' : 'text-gray-400 hover:text-white border-transparent'}`}>
              <span className="inline-flex items-center gap-2">
                {tabKey === 'builtin' ? <Zap className="w-4 h-4" /> : <Folder className="w-4 h-4" />}
                {tabKey === 'builtin' ? t('trainingPanel.templates.tabBuiltin') : t('trainingPanel.templates.tabUser').replace('{count}', String(userTemplates.length))}
              </span>
            </button>
          ))}
        </div>
        <div className="p-5 overflow-y-auto flex-1 space-y-3">
          {all.length === 0 ? <p className="text-gray-500 text-sm text-center py-8">{t('trainingPanel.templates.noTemplates')}</p> : all.map(template => (
            <div key={template.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium text-sm flex items-center gap-2">
                    {tab === 'builtin' ? builtinIcon(template.id) : <BookOpen className="w-4 h-4 text-gray-500" />}
                    <span>{template.name}</span>
                  </p>
                  {template.description && <p className="text-gray-500 text-xs mt-0.5">{template.description}</p>}
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {template.config.learning_rate != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">LR: {template.config.learning_rate}</span>}
                    {template.config.batch_size     != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/20">Batch: {template.config.batch_size}</span>}
                    {template.config.epochs         != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-400 border border-purple-500/20">Epochs: {template.config.epochs}</span>}
                    {template.config.max_seq_length != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/20">Seq: {template.config.max_seq_length}</span>}
                    {template.config.fp16           && <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 border border-cyan-500/20">FP16</span>}
                    {template.config.use_lora       && <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-400 border border-violet-500/20">LoRA r={template.config.lora_r}</span>}
                    {template.config.load_in_4bit   && <span className="text-[10px] px-2 py-0.5 rounded-full bg-fuchsia-500/15 text-fuchsia-400 border border-fuchsia-500/20">{t('trainingPanel.templates.builtinTemplates.qlora.name')}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {tab === 'user' && <button onClick={() => handleDelete(template.id)} className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>}
                  <button onClick={() => { onApply(template.config); onClose(); }} className="px-3 py-1.5 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-xs font-medium transition-all">{t('trainingPanel.templates.loadButton')}</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 pb-5 flex-shrink-0 border-t border-white/10 pt-4">
          {showSaveForm ? (
            <div className="space-y-2">
              <input value={saveName} onChange={e => setSaveName(e.target.value)} placeholder={t('trainingPanel.templates.saveNamePlaceholder')} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/20" />
              <input value={saveDesc} onChange={e => setSaveDesc(e.target.value)} placeholder={t('trainingPanel.templates.saveDescPlaceholder')} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/20" />
              <div className="flex gap-2">
                <button onClick={handleSave} disabled={!saveName.trim()} className="flex-1 py-2 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-sm font-medium transition-all disabled:opacity-40"><Save className="w-3.5 h-3.5 inline mr-1.5" />{t('trainingPanel.templates.saveButton')}</button>
                <button onClick={() => setShowSaveForm(false)} className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all">{t('trainingPanel.templates.cancelButton')}</button>
              </div>
            </div>
          ) : (
            <button onClick={() => { setTab('user'); setShowSaveForm(true); }} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all">
              <Plus className="w-4 h-4" /> {t('trainingPanel.templates.saveCurrentButton')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── AI Metric Assistant ────────────────────────────────────────────────────

const KI_CONFIG_FIELDS = `
ALLE VERFÜGBAREN METRIKEN (du kannst ALLE davon in deinem JSON verwenden):

--- BASIS ---
- epochs: number (Anzahl Trainingsepochen, z.B. 3)
- batch_size: number (Batch-Größe, z.B. 8, 16, 32)
- learning_rate: number (Lernrate, z.B. 0.00002 = 2e-5)
- weight_decay: number (L2-Regularisierung, z.B. 0.01)
- warmup_ratio: number (Anteil Warmup-Schritte relativ, 0.0–0.2)
- warmup_steps: number (Absolute Warmup-Schritte, z.B. 100; 0 = warmup_ratio nutzen)
- max_steps: number (Maximale Trainings-Schritte; -1 = alle Epochen)
- max_seq_length: number (Maximale Token-Länge, 16–512)
- gradient_accumulation_steps: number (Effektive Batch-Vergrößerung)
- fp16: boolean (FP16 Mixed Precision)
- bf16: boolean (BF16 Mixed Precision, nur wenn fp16=false)

--- OPTIMIZER ---
- optimizer: "adamw"|"adam"|"sgd"|"adafactor"
- scheduler: "linear"|"cosine"|"constant"|"polynomial"
- adam_beta1: number (Adam β1, Standard 0.9)
- adam_beta2: number (Adam β2, Standard 0.999)
- adam_epsilon: number (Adam ε, Standard 1e-8)

--- REGULARISIERUNG ---
- dropout: number (Dropout-Rate, 0.0–0.5)
- max_grad_norm: number (Gradient Clipping, z.B. 1.0)
- label_smoothing: number (Label Smoothing, 0.0–0.2)

--- EVALUATION & SAVING ---
- eval_strategy: "epoch"|"steps"|"no"
- eval_steps: number (Eval alle N Schritte, wenn eval_strategy="steps")
- save_steps: number (Checkpoint alle N Schritte)
- save_total_limit: number (Max. gespeicherte Checkpoints)
- logging_steps: number (Log alle N Schritte)
- seed: number (Zufalls-Seed für Reproduzierbarkeit)

--- DATENLADER ---
- num_workers: number (DataLoader Worker-Threads, 0–8)
- pin_memory: boolean (Pinned Memory für GPU, schneller)

--- FLAGS ---
- gradient_checkpointing: boolean (Spart RAM, etwas langsamer)
- group_by_length: boolean (Ähnliche Längen zusammenfassen, effizienter)

--- LORA / QLORA ---
- use_lora: boolean (LoRA aktivieren – spart massiv RAM, ideal für große Modelle)
- lora_r: number (LoRA Rank, z.B. 8, 16, 32 – höher = mehr Parameter)
- lora_alpha: number (LoRA Alpha, z.B. 16 – meist 2× lora_r)
- lora_dropout: number (LoRA Dropout, 0.0–0.1)
- lora_target_modules: string (Komma-getrennte Module, z.B. "q_proj,v_proj")
- load_in_4bit: boolean (QLoRA: 4-bit Quantisierung – sehr wenig RAM, braucht bitsandbytes)
- load_in_8bit: boolean (8-bit Quantisierung – braucht bitsandbytes)

HINWEIS: Wenn das Modell viel RAM braucht → use_lora=true, lora_r=8, load_in_4bit=true empfehlen.`;

function AIMetricAssistant({ config, datasetName, datasetSize, modelName, onApply, onClose, onSaveAsTemplate, initialGoal }: {
  config: TrainingConfig; datasetName: string; datasetSize: number; modelName: string;
  onApply: (patch: Partial<TrainingConfig>) => void;
  onClose: () => void;
  onSaveAsTemplate: (cfg: Partial<TrainingConfig>) => void;
  initialGoal?: string;
}) {
  const { t } = useLanguage();
  const { settings: aiSettings } = useAISettings();
  const { language } = useLanguage();
  const [goalText, setGoalText] = useState(initialGoal ?? '');
  const [loading, setLoading] = useState(false);
  const [suggestion, setSuggestion] = useState<string | null>(null);
  const [parsed, setParsed] = useState<Partial<TrainingConfig> | null>(null);
  const [applied, setApplied] = useState(false);
  const [savedAsTemplate, setSavedAsTemplate] = useState(false);
  const [phase, setPhase] = useState<'input' | 'result'>(initialGoal ? 'input' : 'input');

  // Auto-trigger analysis if initialGoal provided (e.g. from error recovery)
  useEffect(() => {
    if (initialGoal && initialGoal.trim()) setGoalText(initialGoal);
  }, [initialGoal]);

  const ask = async () => {
    setLoading(true); setSuggestion(null); setParsed(null); setApplied(false); setSavedAsTemplate(false);
    setPhase('result');
    const prompt = `Du bist ein ML-Experte für HuggingFace Fine-Tuning.

AKTUELLE KONFIGURATION:
${Object.entries(config).map(([k, v]) => `- ${k}: ${v}`).join('\n')}

KONTEXT:
- Modell: ${modelName}
- Dataset: ${datasetName} (${datasetSize} Dateien)
${goalText ? `\nZIEL / PROBLEM DES USERS:\n${goalText}` : ''}

${KI_CONFIG_FIELDS}

AUFGABE:
1. Analysiere die aktuelle Konfiguration kurz (3-4 Sätze auf Deutsch)
2. Berücksichtige das Ziel / Problem des Users falls angegeben
3. Erstelle optimierte Hyperparameter

WICHTIG: Gib am Ende EIN valides JSON-Objekt mit ALLEN Metriken die du ändern möchtest.
Nur Felder die sich ändern sollen. Beispiel: {"epochs":4,"learning_rate":0.00002,"fp16":true,"use_lora":true,"lora_r":8}
Kein Markdown-Code-Block, nur das reine JSON-Objekt am Ende.`;

    try {
      const text = await callAI(aiSettings, 'Du bist ein präziser ML-Experte. Antworte auf Deutsch. Gib am Ende exakt ein valides JSON-Objekt aus.', prompt, undefined, language);
      setSuggestion(text);
      const matches = [...text.matchAll(/\{[^{}]*\}/g)];
      if (matches.length > 0) {
        try { setParsed(JSON.parse(matches[matches.length - 1][0])); } catch { /* ignore */ }
      }
    } catch (err) { setSuggestion(`Fehler: ${String(err)}`); } finally { setLoading(false); }
  };

  const textOnly = suggestion?.replace(/\{[^{}]*\}/g, '').trim() ?? '';

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-violet-400" /><h2 className="text-lg font-bold text-white">{t('trainingPanel.aiAssistant.title')}</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {phase === 'input' ? (
            <div className="p-6 space-y-5">
              <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-2">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">{t('trainingPanel.aiAssistant.configTitle')}</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-1">
                  {[
                    ['Epochs', config.epochs], ['Batch', config.batch_size], ['LR', config.learning_rate],
                    ['Seq Len', config.max_seq_length], ['Optimizer', config.optimizer], ['Scheduler', config.scheduler],
                    ['FP16', config.fp16 ? 'ja' : 'nein'], ['GradAcc', config.gradient_accumulation_steps],
                    ['LoRA', config.use_lora ? `r=${config.lora_r}` : 'nein'],
                  ].map(([k, v]) => (
                    <div key={k as string} className="flex items-center gap-2 text-xs">
                      <span className="text-gray-500">{k}:</span>
                      <span className="text-gray-200 font-mono">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-xl border border-dashed border-white/15 bg-white/[0.02] space-y-2">
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-gray-500" />
                  <p className="text-sm font-medium text-gray-400">{t('trainingPanel.aiAssistant.historyTitle')}</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-amber-500/15 text-amber-400 border border-amber-500/20">{t('trainingPanel.aiAssistant.historySoon')}</span>
                </div>
                <p className="text-xs text-gray-600">{t('trainingPanel.aiAssistant.historyHint')}</p>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-white">{t('trainingPanel.aiAssistant.goalLabel')}</label>
                <p className="text-xs text-gray-500">{t('trainingPanel.aiAssistant.goalHint')}</p>
                <textarea
                  value={goalText}
                  onChange={e => setGoalText(e.target.value)}
                  placeholder={t('trainingPanel.aiAssistant.goalPlaceholder')}
                  rows={4}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/50 resize-none transition-all"
                />
              </div>

              <button onClick={ask} className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:opacity-90 text-white font-semibold text-sm transition-all">
                <Sparkles className="w-4 h-4" /> {t('trainingPanel.aiAssistant.startButton')}
              </button>
            </div>
          ) : (
            <div className="p-6 space-y-5">
              {loading ? (
                <div className="flex flex-col items-center gap-3 py-12">
                  <Loader2 className="w-10 h-10 text-violet-400 animate-spin" />
                  <p className="text-gray-400 text-sm">{t('trainingPanel.aiAssistant.analysingText')}</p>
                  {goalText && <p className="text-gray-600 text-xs max-w-sm text-center">{t('trainingPanel.aiAssistant.analysingGoal').replace('{goal}', goalText.slice(0, 80))}</p>}
                </div>
              ) : (
                <>
                  {textOnly && (
                    <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/20 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">{textOnly}</div>
                  )}
                  {parsed && Object.keys(parsed).length > 0 && (
                    <div className="space-y-3">
                      <p className="text-white text-sm font-medium flex items-center gap-2">
                        <ClipboardList className="w-4 h-4 text-violet-400" />
                        {t('trainingPanel.aiAssistant.suggestionsTitle').replace('{count}', String(Object.keys(parsed).length))}
                      </p>
                      <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                        {Object.entries(parsed).map(([k, v]) => (
                          <div key={k} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/5 border border-white/10">
                            <span className="text-gray-400 font-mono text-xs">{k}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-gray-600 text-xs line-through">{String((config as unknown as Record<string, unknown>)[k] ?? '—')}</span>
                              <span className="text-emerald-400 font-semibold text-xs">→ {String(v)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="space-y-2">
                        {applied ? (
                          <div className="flex items-center gap-2 justify-center py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-sm">
                        <Check className="w-4 h-4" /> {t('trainingPanel.aiAssistant.applyDoneLabel')}
                          </div>
                        ) : (
                          <div className="flex gap-2">
                            <button onClick={() => { onApply(parsed); setApplied(true); }} className="flex-1 py-2.5 rounded-xl bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/40 text-violet-300 text-sm font-medium transition-all">
                              <span className="inline-flex items-center gap-2">
                                <Check className="w-4 h-4" />
                                {t('trainingPanel.aiAssistant.applyButton').replace('{count}', String(Object.keys(parsed).length))}
                              </span>
                            </button>
                            <button onClick={onClose} className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all">{t('trainingPanel.aiAssistant.discardButton')}</button>
                          </div>
                        )}
                        {applied && (
                          savedAsTemplate ? (
                            <div className="flex items-center gap-2 justify-center py-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs">
                              <Check className="w-3.5 h-3.5" /> {t('trainingPanel.aiAssistant.savedAsTemplateLabel')}
                            </div>
                          ) : (
                            <button onClick={() => { onSaveAsTemplate(parsed); setSavedAsTemplate(true); }}
                              className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"
                            >
                              <BookOpen className="w-3.5 h-3.5" /> {t('trainingPanel.aiAssistant.saveAsTemplateButton')}
                            </button>
                          )
                        )}

                        {applied && parsed && (
                          <button
                            onClick={() => {
                              const payload = JSON.stringify(parsed, null, 2);
                              openAICoach({
                                newChat: true,
                                prefill: `Ich habe im Training-Panel folgende KI-vorgeschlagene Metriken übernommen:\n\n\`\`\`json\n${payload}\n\`\`\`\n\nZiel: ${goalText || '(kein Ziel angegeben)'}\n\nBitte schlage mir als nächstes ein kleines Experiment-Setup vor (2–3 Varianten), inkl. worauf ich in der Analyse achten soll.`,
                                titleHint: 'Training-Metriken-Followup',
                              });
                            }}
                            className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 text-violet-200 text-xs font-medium transition-all"
                          >
                            <Sparkles className="w-3.5 h-3.5" /> {t('trainingPanel.aiAssistant.continueInCoachButton')}
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className="px-6 pb-5 flex-shrink-0 border-t border-white/10 pt-4 flex gap-2">
          {phase === 'result' && (
            <button onClick={() => setPhase('input')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all">{t('trainingPanel.aiAssistant.backButton')}</button>
          )}
          {phase === 'result' && !loading && (
            <button onClick={ask} disabled={loading} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all">
              <RefreshCw className="w-3.5 h-3.5" /> {t('trainingPanel.aiAssistant.reanalyzeButton')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function TrainingPanel({ userData, onNavigateToAnalysis }: TrainingPanelProps) {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  const { success, error, warning } = useNotification();
  const { setCurrentPageContent } = usePageContext();
  const {
    state: trainingState,
    setShowDashboard: setShowDashboardContext,
    setIsDashMinimized: setIsDashMinimizedContext,
    setCurrentJob: setCurrentJobContext,
    addLossPoint: addLossPointContext,
    setLossPoints: setLossPointsContext,
    setSessionId: setSessionIdContext,
    setDashStartedAt: setDashStartedAtContext,
    setTrainingInfo: setTrainingInfoContext,
    setTrainingConfig: setTrainingConfigContext,
    setCompletedVersionId: setCompletedVersionIdContext,
  } = useTrainingContext();

  const [mode, setMode] = useState<'train' | 'dev'>('train');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedVersionPath, setSelectedVersionPath] = useState<string>('');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [loadingData, setLoadingData] = useState(true);

  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const updateConfig = useCallback((patch: Partial<TrainingConfig>) => setConfig(c => ({ ...c, ...patch })), []);

  const [sections, setSections] = useState({ basic: true, optimizer: false, advanced: false, lora: false, ram: true });
  const toggleSection = (k: keyof typeof sections) => setSections(s => ({ ...s, [k]: !s[k] }));

  const [showAIAssistant, setShowAIAssistant] = useState(false);
  const [aiInitialGoal, setAiInitialGoal] = useState('');
  const [showTemplates, setShowTemplates] = useState(false);
  // showDashboard & isDashMinimized leben jetzt im TrainingContext (global in Dashboard.tsx)

  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [lossPoints, setLossPoints] = useState<LossPoint[]>([]);
  const [reqs, setReqs] = useState<RequirementsCheck | null>(null);
  const [modelSizeGb, setModelSizeGb] = useState(0.56);
  const [showHistory, setShowHistory] = useState(false);
  const [historyJobs, setHistoryJobs] = useState<TrainingJob[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyFilter, setHistoryFilter] = useState<'all' | 'completed' | 'failed' | 'stopped' | 'running'>('all');

  void checkDatasetCompat;

  useEffect(() => { initLoad(); invoke<RequirementsCheck>('check_training_requirements').then(setReqs).catch(() => {}); }, []);
  useEffect(() => { if (selectedModelId) { loadDatasetsForModel(selectedModelId); loadModelSize(selectedModelId); } }, [selectedModelId]);
  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); setSelectedVersionPath(''); return; }
    const m = modelsWithVersions.find(x => x.id === selectedModelId);
    if (!m?.versions.length) { setSelectedVersionId(null); setSelectedVersionPath(''); return; }
    setSelectedVersionId([...m.versions].sort((a, b) => b.version_number - a.version_number)[0].id);
  }, [selectedModelId, modelsWithVersions]);

  // Load version path when selectedVersionId changes
  useEffect(() => {
    if (!selectedVersionId) { setSelectedVersionPath(''); return; }
    
    invoke<string>('get_version_path_for_ui', { versionId: selectedVersionId })
      .then(path => setSelectedVersionPath(path))
      .catch(err => {
        console.error('Error loading version path:', err);
        setSelectedVersionPath('');
      });
  }, [selectedVersionId]);

  const initLoad = async () => {
    setLoadingData(true);
    try {
      const [listModels, listWithVersions] = await Promise.all([
        invoke<ModelInfo[]>('list_models'),
        invoke<ModelWithVersionTree[]>('list_models_with_version_tree'),
      ]);
      setModels(listModels);
      setModelsWithVersions(listWithVersions);
      if (listWithVersions.length > 0) setSelectedModelId(listWithVersions[0].id);
    } catch (e) { console.error('[Training] initLoad:', e); }
    finally { setLoadingData(false); }
  };

  const loadDatasetsForModel = async (modelId: string) => {
    try {
      const list = await invoke<DatasetInfo[]>('list_datasets_for_model', { modelId });
      setDatasets(list);
      const split = list.find(d => d.status === 'split');
      setSelectedDatasetId(split?.id ?? list[0]?.id ?? null);
    } catch { setDatasets([]); }
  };

  const loadModelSize = async (modelId: string) => {
    try {
      const m = models.find(x => x.id === modelId);
      if (m?.size_bytes) { setModelSizeGb(m.size_bytes / (1024 ** 3)); return; }
      const info = await invoke<{ size_bytes?: number; param_billion?: number }>('get_model_info', { modelId });
      if (info.size_bytes) setModelSizeGb(info.size_bytes / (1024 ** 3));
      else if (info.param_billion) setModelSizeGb(info.param_billion * 2);
    } catch { setModelSizeGb(0.56); }
  };

  // Wenn wir auf die Trainingsseite zurückkommen, kann im Context noch ein laufender Job stecken
  // (z.B. TrainingDashboard minimiert). In diesem Fall darf das lokale `currentJob` (initial null)
  // den Context nicht "weg-null-en".
  useEffect(() => {
    if (currentJob == null && trainingState.currentJob != null) return;
    setCurrentJobContext(currentJob);
  }, [currentJob, setCurrentJobContext, trainingState.currentJob]);

  useEffect(() => {
    if (currentJob == null && trainingState.currentJob != null) {
      setCurrentJob(trainingState.currentJob as unknown as TrainingJob);
    }
  }, [currentJob, trainingState.currentJob]);

  // Refs to keep latest callback functions without triggering effect re-runs
  const successRef = useRef(success);
  const addLossPointContextRef = useRef(addLossPointContext);
  const setCompletedVersionIdContextRef = useRef(setCompletedVersionIdContext);

  useEffect(() => {
    successRef.current = success;
    addLossPointContextRef.current = addLossPointContext;
    setCompletedVersionIdContextRef.current = setCompletedVersionIdContext;
  }, [success, addLossPointContext, setCompletedVersionIdContext]);

  // Register event listeners only once
  useEffect(() => {
    let u1: (() => void) | undefined, u2: (() => void) | undefined, u3: (() => void) | undefined;
    listen<{ data: TrainingProgress }>('training-progress', e => {
      const d = e.payload.data;
      setCurrentJob(j => (j ? { ...j, status: 'running', progress: d } : null));
      if (d.train_loss != null) {
        const newPoint = { step: d.step, epoch: d.epoch, train_loss: d.train_loss, val_loss: d.val_loss ?? undefined };
        setLossPoints(pts => [...pts, newPoint]);
        addLossPointContextRef.current(newPoint);
      }
    }).then(fn => { u1 = fn; });
    listen<{ new_version_id?: string }>('training-complete', e => {
      setCurrentJob(j => (j ? { ...j, status: 'completed' } : null));
      invoke('disable_prevent_sleep').catch(() => {});
      successRef.current(t('trainingPanel.notifications.trainingComplete'), t('trainingPanel.notifications.trainingCompleteDetail'));
      // Version-ID im Context speichern → TrainingDashboard zeigt "Analyse starten"-Button
      if (e.payload.new_version_id) {
        setCompletedVersionIdContextRef.current(e.payload.new_version_id);
      }
      // Kein automatisches onNavigateToAnalysis mehr – User entscheidet selbst über Dashboard-Button
    }).then(fn => { u2 = fn; });
    listen<{ data?: { error?: string } }>('training-error', e => {
      setCurrentJob(j => (j ? { ...j, status: 'failed', error: e.payload.data?.error ?? t('common.error') } : null));
      invoke('disable_prevent_sleep').catch(() => {});
    }).then(fn => { u3 = fn; });
    return () => { u1?.(); u2?.(); u3?.(); };
  }, []);

  useEffect(() => {
    const lines: string[] = [
      t('trainingPanel.pageContext.title'),
      '',
      t('trainingPanel.pageContext.purposeTitle'),
      t('trainingPanel.pageContext.purposeBody'),
      t('trainingPanel.pageContext.purposeFlow'),
      '',
      t('trainingPanel.pageContext.currentSelection'),
    ];

    if (!selectedModel) {
      lines.push(t('trainingPanel.pageContext.modelMissing'));
    } else {
      lines.push(t('trainingPanel.pageContext.modelSelected').replace('{name}', selectedModel.name).replace('{type}', selectedModel.model_type ?? '?'));
      if (selectedVersionTree) {
        lines.push(t('trainingPanel.pageContext.versionSelected').replace('{name}', selectedVersionTree.name).replace('{version}', String(selectedVersionTree.version_number)));
      } else {
        lines.push(t('trainingPanel.pageContext.versionMissing'));
      }
      if (detectionKey && detection?.supported) {
        lines.push(t('trainingPanel.pageContext.pluginSelected').replace('{name}', detection.plugin.name).replace('{task}', detection.plugin.taskType));
      } else {
        lines.push(t('trainingPanel.pageContext.pluginUnsupported'));
      }
    }

    if (!selectedDataset) {
      lines.push(t('trainingPanel.pageContext.datasetMissing'));
    } else {
      lines.push(t('trainingPanel.pageContext.datasetSelected').replace('{name}', selectedDataset.name));
      lines.push(t('trainingPanel.pageContext.datasetStatus').replace('{status}', selectedDataset.status === 'split' ? t('trainingPanel.pageContext.datasetSplit') : t('trainingPanel.pageContext.datasetUnsplit')));
    }

    lines.push('');
    lines.push(t('trainingPanel.pageContext.trainingModeTitle'));
    lines.push(t('trainingPanel.pageContext.trainingMode').replace('{mode}', mode === 'train' ? t('trainingPanel.header.modeTraining') : t('trainingPanel.header.modeDev')));

    if (currentJob) {
      lines.push('');
      lines.push(t('trainingPanel.pageContext.trainingStatusTitle'));
      lines.push(t('trainingPanel.pageContext.status').replace('{status}', currentJob.status));
      if (currentJob.status === 'running') {
        lines.push(t('trainingPanel.pageContext.epoch').replace('{epoch}', String(currentJob.progress.epoch)).replace('{total}', String(currentJob.progress.total_epochs)));
        lines.push(t('trainingPanel.pageContext.step').replace('{step}', String(currentJob.progress.step)).replace('{total}', String(currentJob.progress.total_steps)));
        lines.push(t('trainingPanel.pageContext.trainLoss').replace('{value}', currentJob.progress.train_loss.toFixed(4)));
        if (currentJob.progress.val_loss) lines.push(t('trainingPanel.pageContext.valLoss').replace('{value}', currentJob.progress.val_loss.toFixed(4)));
        lines.push(t('trainingPanel.pageContext.progress').replace('{value}', `${currentJob.progress.progress_percent}%`));
      } else if (currentJob.status === 'failed' && currentJob.error) {
        lines.push(t('trainingPanel.pageContext.error').replace('{error}', currentJob.error));
      }
    } else {
      lines.push('');
      lines.push(t('trainingPanel.pageContext.readyTitle'));
      lines.push(t('trainingPanel.pageContext.readyBody'));
    }

    lines.push('');
    lines.push(t('trainingPanel.pageContext.layoutTitle'));
    lines.push(t('trainingPanel.pageContext.layoutTop'));
    lines.push(t('trainingPanel.pageContext.layoutTopModel'));
    lines.push(t('trainingPanel.pageContext.layoutTopDataset'));
    lines.push(t('trainingPanel.pageContext.layoutTopMode'));
    lines.push('');
    lines.push(t('trainingPanel.pageContext.layoutMiddle'));
    lines.push(t('trainingPanel.pageContext.layoutMiddleBasic'));
    lines.push(t('trainingPanel.pageContext.layoutMiddleOptimizer'));
    lines.push(t('trainingPanel.pageContext.layoutMiddleAdvanced'));
    lines.push(t('trainingPanel.pageContext.layoutMiddleLora'));
    lines.push(t('trainingPanel.pageContext.layoutMiddleInfo'));
    lines.push('');
    lines.push(t('trainingPanel.pageContext.layoutBottom'));
    lines.push(t('trainingPanel.pageContext.layoutBottomStart'));
    lines.push(t('trainingPanel.pageContext.layoutBottomDashboard'));
    lines.push(t('trainingPanel.pageContext.layoutBottomHistory'));
    lines.push('');
    lines.push(t('trainingPanel.pageContext.availableActions'));
    if (!selectedModel || !selectedDataset) {
      lines.push(t('trainingPanel.pageContext.actionPickModel'));
      lines.push(t('trainingPanel.pageContext.actionPickDataset'));
    } else if (!isSupported) {
      lines.push(t('trainingPanel.pageContext.actionDevMode'));
      lines.push(t('trainingPanel.pageContext.actionDevScript'));
    } else if (selectedDataset.status !== 'split') {
      lines.push(t('trainingPanel.pageContext.actionOpenDatasets'));
      lines.push(t('trainingPanel.pageContext.actionOpenDataset').replace('{name}', selectedDataset.name));
      lines.push(t('trainingPanel.pageContext.actionSplitDataset'));
    } else if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      lines.push(t('trainingPanel.pageContext.actionAdjustConfig'));
      lines.push(t('trainingPanel.pageContext.actionStartTraining'));
      lines.push(t('trainingPanel.pageContext.actionDashboardOpens'));
    } else if (currentJob.status === 'running') {
      lines.push(t('trainingPanel.pageContext.actionMonitor'));
      lines.push(t('trainingPanel.pageContext.actionPauseStop'));
    }

    setCurrentPageContent(lines.join('\n'));
  }, [selectedModelId, selectedDatasetId, mode, currentJob, setCurrentPageContent]);

  const selectedModel   = models.find(m => m.id === selectedModelId);
  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);
  const selectedModelTree = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersionTree = selectedModelTree?.versions.find(v => v.id === selectedVersionId);
  const detectionKey    = selectedModel?.source_path ?? selectedModel?.name ?? '';
  const detection       = detectionKey ? detectPlugin(detectionKey, selectedModel?.model_type ? { model_type: selectedModel.model_type } : undefined) : null;
  const isSupported     = detection?.supported === true;
  const pluginId        = detection?.supported ? detection.plugin.id : null;

  const handleStartTraining = async () => {
    if (!selectedModelId || !selectedDatasetId) { warning(t('trainingPanel.notifications.missingSelection'), t('trainingPanel.notifications.missingSelectionDetail')); return; }
    if (!isSupported) { warning(t('trainingPanel.notifications.notSupported'), t('trainingPanel.notifications.notSupportedDetail')); return; }
    const isCanvasModel = selectedModelId.startsWith('canvas_');
    if (!isCanvasModel && selectedDataset?.status !== 'split') { warning(t('trainingPanel.notifications.noSplit'), t('trainingPanel.notifications.noSplitDetail')); return; }

    setLossPoints([]);
    setLossPointsContext([]);
    try {
      // Konvertiere lora_target_modules von String zu Array falls nötig
      const configForBackend = {
        ...config,
        lora_target_modules: typeof config.lora_target_modules === 'string'
          ? config.lora_target_modules.split(',').map(m => m.trim()).filter(m => m)
          : config.lora_target_modules,
        task_type: detection?.supported ? detection.plugin.taskType : 'seq_classification',
        plugin_config: detection?.supported ? (detection.plugin.defaultPluginConfig ?? {}) : {},
      };
      
      const job = await invoke<TrainingJob>('start_training', {
        modelId: selectedModelId, modelName: selectedModel?.name ?? '',
        datasetId: selectedDatasetId, datasetName: selectedDataset?.name ?? '', config: configForBackend,
      });
      setCurrentJob(job);
      setCurrentJobContext(job);
      invoke('enable_prevent_sleep').catch(() => {});
      const sessionId = `train_${Date.now()}`;
      const startedAt = Date.now();
      
      // Update global context für minimiertes Panel
      setShowDashboardContext(true);
      setIsDashMinimizedContext(false);
      setSessionIdContext(sessionId);
      setDashStartedAtContext(startedAt);
      setTrainingInfoContext('standard', selectedModel?.name ?? '', selectedDataset?.name ?? '');
      setTrainingConfigContext(config);
      
      success(t('trainingPanel.notifications.started'), t('trainingPanel.notifications.startedDetail'));
    } catch (err: unknown) { error(t('trainingPanel.notifications.startFailed'), String(err)); }
  };

  const handleStopTraining = async () => {
    try { await invoke('stop_training'); invoke('disable_prevent_sleep').catch(() => {}); success(t('trainingPanel.notifications.stopped'), ''); } catch (err: unknown) { error(t('trainingPanel.notifications.stopFailed'), String(err)); }
  };

  const handleSaveTemplate = async (name: string, desc: string) => {
    try { await invoke('save_metrics_template', { name, description: desc, config, source: 'user' }); success(t('trainingPanel.notifications.templateSaved'), name); }
    catch (err: unknown) { error(t('common.error'), String(err)); }
  };

  const handleSaveAIAsTemplate = async (cfg: Partial<TrainingConfig>) => {
    try { await invoke('save_metrics_template', { name: `KI-Vorschlag ${new Date().toLocaleDateString('de-DE')}`, description: 'Automatisch vom KI-Assistenten erstellt.', config: { ...DEFAULT_CONFIG, ...cfg }, source: 'ai' }); success(t('trainingPanel.notifications.aiTemplateSaved'), t('trainingPanel.notifications.aiTemplateSavedDetail')); }
    catch { /* ignore */ }
  };

  const handleOpenHistory = async () => {
    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const jobs = await invoke<TrainingJob[]>('get_training_history');
      setHistoryJobs(jobs);
    } catch { setHistoryJobs([]); }
    finally { setHistoryLoading(false); }
  };

  const openKIAssistantFromError = (errorMsg: string) => {
    setAiInitialGoal(`FEHLER beim Training: ${errorMsg}\n\nBitte analysiere die Konfiguration und schlage Fixes vor.`);
    setShowDashboardContext(false);
    setIsDashMinimizedContext(false);
    setShowAIAssistant(true);
  };

  const isRunning = currentJob?.status === 'running' || currentJob?.status === 'pending';
  const progress  = currentJob?.progress;

  if (loadingData) return <div className="flex items-center justify-center py-24"><Loader2 className="w-8 h-8 text-gray-500 animate-spin" /></div>;

  return (
    <div className="space-y-6">

      {/* Training History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-3xl max-h-[85vh] flex flex-col">
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <History className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-bold text-white">{t('trainingPanel.history.title')}</h2>
              </div>
              <button onClick={() => setShowHistory(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            {/* Filter */}
            <div className="flex gap-2 px-6 py-3 border-b border-white/10 flex-wrap">
              {(['all', 'running', 'completed', 'failed', 'stopped'] as const).map(f => {
                const labels: Record<string, string> = { all: t('trainingPanel.history.filterAll'), running: t('trainingPanel.history.filterRunning'), completed: t('trainingPanel.history.filterCompleted'), failed: t('trainingPanel.history.filterFailed'), stopped: t('trainingPanel.history.filterStopped') };
                const colors: Record<string, string> = { all: 'bg-white/10 text-white', running: 'bg-blue-500/20 text-blue-300 border-blue-500/30', completed: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30', failed: 'bg-red-500/20 text-red-300 border-red-500/30', stopped: 'bg-gray-500/20 text-gray-300 border-gray-500/30' };
                const icon: Record<string, React.ReactNode> = {
                  all: <ClipboardList className="w-3.5 h-3.5" />,
                  running: <Play className="w-3.5 h-3.5" />,
                  completed: <CheckCircle className="w-3.5 h-3.5" />,
                  failed: <XCircle className="w-3.5 h-3.5" />,
                  stopped: <Square className="w-3.5 h-3.5" />,
                };
                return (
                  <button key={f} onClick={() => setHistoryFilter(f)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
                      historyFilter === f ? colors[f] : 'bg-white/5 border-white/10 text-gray-500 hover:text-gray-300'
                    }`}>
                    <span className="inline-flex items-center gap-1.5">
                      {icon[f]}
                      <span>{labels[f]}</span>
                    </span>
                  </button>
                );
              })}
              <div className="ml-auto text-xs text-gray-500 flex items-center">
                {historyJobs.filter(j => historyFilter === 'all' || j.status === historyFilter).length} {t('trainingPanel.history.entries')}
              </div>
            </div>
            {/* List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {historyLoading ? (
                <div className="flex items-center justify-center py-16"><Loader2 className="w-6 h-6 text-purple-400 animate-spin" /></div>
              ) : (() => {
                const filtered = historyJobs.filter(j => historyFilter === 'all' || j.status === historyFilter);
                if (filtered.length === 0) return (
                  <div className="text-center py-16 text-gray-500">
                    <ClipboardList className="w-10 h-10 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">{t('trainingPanel.history.noEntries')}</p>
                  </div>
                );
                return filtered.map(job => {
                  const statusIcon: Record<string, React.ReactNode> = {
                    completed: <CheckCircle className="w-3.5 h-3.5" />,
                    failed: <XCircle className="w-3.5 h-3.5" />,
                    stopped: <Square className="w-3.5 h-3.5" />,
                    running: <Play className="w-3.5 h-3.5" />,
                    pending: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
                  };
                  const statusLabel: Record<string, string> = { completed: t('trainingPanel.history.statusCompleted'), failed: t('trainingPanel.history.statusFailed'), stopped: t('trainingPanel.history.statusStopped'), running: t('trainingPanel.history.statusRunning'), pending: t('trainingPanel.history.statusPending') };
                  const statusColor: Record<string, string> = { completed: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', failed: 'text-red-400 bg-red-500/10 border-red-500/20', stopped: 'text-gray-400 bg-gray-500/10 border-gray-500/20', running: 'text-blue-400 bg-blue-500/10 border-blue-500/20', pending: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20' };
                  const durMs = job.completed_at && job.started_at ? new Date(job.completed_at).getTime() - new Date(job.started_at).getTime() : null;
                  const durStr = durMs ? (durMs > 3600000 ? `${Math.floor(durMs/3600000)}h ${Math.floor((durMs%3600000)/60000)}m` : durMs > 60000 ? `${Math.floor(durMs/60000)}m ${Math.floor((durMs%60000)/1000)}s` : `${Math.floor(durMs/1000)}s`) : null;
                  return (
                    <div key={job.id} className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${statusColor[job.status] ?? statusColor.pending}`}>
                              {statusIcon[job.status] ?? statusIcon.pending} {statusLabel[job.status] ?? statusLabel.pending}
                            </span>
                            {durStr && <span className="text-xs text-gray-500 inline-flex items-center gap-1"><Clock className="w-3.5 h-3.5" /> {durStr}</span>}
                          </div>
                          <p className="text-white font-medium text-sm truncate">{job.model_name}</p>
                          <p className="text-gray-500 text-xs truncate">{t('trainingPanel.history.datasetLabel')} {job.dataset_name}</p>
                          {job.error && <p className="text-red-400 text-xs mt-1 truncate">{t('trainingPanel.history.errorLabel')} {job.error}</p>}
                        </div>
                        <div className="text-right flex-shrink-0">
                          <p className="text-xs text-gray-500">{new Date(job.created_at).toLocaleDateString('de-DE', { day:'2-digit', month:'2-digit', year:'2-digit' })}</p>
                          <p className="text-xs text-gray-600">{new Date(job.created_at).toLocaleTimeString('de-DE', { hour:'2-digit', minute:'2-digit' })}</p>
                          {job.progress && job.progress.progress_percent > 0 && (
                            <p className="text-xs text-gray-500 mt-1">{job.progress.progress_percent.toFixed(0)}% · {t('trainingPanel.history.epochLabel')} {job.progress.epoch}/{job.progress.total_epochs}</p>
                          )}
                        </div>
                      </div>
                      {job.progress && job.progress.train_loss > 0 && (
                        <div className="mt-2 pt-2 border-t border-white/10 flex gap-4 text-xs text-gray-500">
                          <span>{t('trainingPanel.history.lossLabel')} <span className="text-gray-300">{job.progress.train_loss.toFixed(4)}</span></span>
                          {job.progress.val_loss && <span>{t('trainingPanel.history.valLabel')} <span className="text-gray-300">{job.progress.val_loss.toFixed(4)}</span></span>}
                          <span>{t('trainingPanel.history.epochLabel')} {job.progress.epoch}/{job.progress.total_epochs}</span>
                        </div>
                      )}
                    </div>
                  );
                });
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div><h1 className="text-2xl font-bold text-white">{t('trainingPanel.title')}</h1><p className="text-gray-400 mt-1">{t('trainingPanel.subtitle')}</p></div>
        <div className="flex items-center gap-2">
          <button onClick={handleOpenHistory} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 text-sm transition-all">
            <History className="w-4 h-4" />
            {t('trainingPanel.history.title')}
          </button>
          <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
            {(['train', 'dev'] as const).map(m => (
              <button key={m} onClick={() => setMode(m)} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${mode === m ? (m === 'train' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-blue-500/20 text-blue-300 border border-blue-500/30') : 'text-gray-400 hover:text-white'}`}>
                {m === 'train' ? <><Play className="w-3.5 h-3.5" /> {t('trainingPanel.header.modeTraining')}</> : <><Code2 className="w-3.5 h-3.5" /> {t('trainingPanel.header.modeDev')}</>}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Model & Dataset */}
      {models.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-12 text-center space-y-3">
          <Layers className="w-10 h-10 text-gray-500 mx-auto" />
          <p className="text-white font-medium">{t('trainingPanel.noModels.title')}</p>
          <p className="text-gray-500 text-sm">{t('trainingPanel.noModels.description')}</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Model + Dataset Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* Model Block (Parent of Version) */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-white">{t('trainingPanel.modelBlock.modelLabel')}</label>
                <select value={selectedModelId ?? ''} onChange={e => { setSelectedModelId(e.target.value); setSelectedDatasetId(null); }} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none mt-1">
                  {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-900">{m.name}</option>)}
                </select>
              </div>
              
              {/* Version Selection */}
              <div>
                <label className="block text-sm font-medium text-white">{t('trainingPanel.modelBlock.versionLabel')}</label>
                <select value={selectedVersionId ?? ''} onChange={e => setSelectedVersionId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none mt-1">
                  {selectedModelTree?.versions?.length ? [...selectedModelTree.versions].sort((a, b) => b.version_number - a.version_number).map((v, idx) => (
                    <option key={v.id} value={v.id} className="bg-slate-900">
                      {v.name} {idx === 0 ? t('trainingPanel.modelBlock.versionLatest') : ''}
                    </option>
                  )) : <option value="">{t('trainingPanel.modelBlock.noVersions')}</option>}
                </select>
              </div>
              
              {/* Support Info */}
              {selectedModel && (isSupported
                ? <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 mt-1"><CheckCircle className="w-4 h-4 text-emerald-400" /><span className="text-emerald-300 text-xs font-medium">{t('trainingPanel.modelBlock.pluginSupported').replace('{name}', detection.plugin.name)}</span></div>
                : <div className="space-y-2 mt-1">
                    <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20"><AlertTriangle className="w-4 h-4 text-amber-400" /><span className="text-amber-300 text-xs">{t('trainingPanel.modelBlock.pluginUnsupported')}</span></div>
                    <button onClick={() => setMode('dev')} className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"><Code2 className="w-3.5 h-3.5" /> {t('trainingPanel.modelBlock.devModeButton')}</button>
                  </div>
              )}
            </div>

            {/* Dataset Selection */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
              <label className="block text-sm font-medium text-white">{t('trainingPanel.datasetBlock.label')}</label>
              {datasets.length === 0
                ? <p className="text-gray-500 text-sm">{t('trainingPanel.datasetBlock.noDataset')}</p>
                : <select value={selectedDatasetId ?? ''} onChange={e => setSelectedDatasetId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                    {datasets.map(d => (
                      <option key={d.id} value={d.id} className="bg-slate-900">
                        {d.name}{d.status === 'split' ? t('trainingPanel.datasetBlock.statusSplit') : t('trainingPanel.datasetBlock.statusUnsplit')}
                      </option>
                    ))}
                  </select>
              }
              {selectedDataset && pluginId && (
                <DatasetCompatBadge
                  modelPluginId={pluginId}
                  extensions={selectedDataset.extensions ?? []}
                  analysis={selectedDataset.dataset_type ? {
                    detected_type: selectedDataset.dataset_type,
                    confidence: 80,
                    pairing_status: selectedDataset.pairing_status ?? null,
                    warnings: selectedDataset.warnings ?? [],
                    file_count: selectedDataset.file_count,
                    dir_count: 0,
                    extensions: selectedDataset.extensions ?? [],
                    schema_hint: null,
                  } : null}
                />
              )}
              {selectedDataset?.status === 'unused' && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20"><AlertCircle className="w-3.5 h-3.5 text-amber-400" /><span className="text-amber-300 text-xs">{t('trainingPanel.datasetBlock.noSplitWarning')}</span></div>
              )}
              {selectedModelId?.startsWith('canvas_') && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-violet-500/10 border border-violet-500/20 mt-1">
                  <span className="text-violet-300 text-xs">{t('trainingPanel.datasetBlock.canvasBadge')}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Dev Train */}
      {mode === 'dev' && (
        <DevTrainPanel modelInfo={selectedModel ?? null} selectedVersionPath={selectedVersionPath} datasets={datasets} onNavigateToAnalysis={onNavigateToAnalysis} userData={userData} />
      )}

      {/* Standard Training */}
      {mode === 'train' && isSupported && (
        <>
          {reqs && !reqs.ready && (
            <div className="p-4 rounded-2xl border border-red-500/30 bg-red-500/10 space-y-2">
              <div className="flex items-center gap-2"><AlertCircle className="w-4 h-4 text-red-400" /><span className="text-red-300 font-medium text-sm">{t('trainingPanel.requirements.notReadyTitle')}</span></div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {[{label:t('trainingPanel.requirements.python'),ok:reqs.python_installed,ver:reqs.python_version},{label:t('trainingPanel.requirements.pytorch'),ok:reqs.torch_installed,ver:reqs.torch_version},{label:t('trainingPanel.requirements.transformers'),ok:reqs.transformers_installed},{label:t('trainingPanel.requirements.cudaMps'),ok:reqs.cuda_available||reqs.mps_available}].map(r => (
                  <div key={r.label} className={`flex items-center gap-1.5 ${r.ok ? 'text-emerald-400' : 'text-red-400'}`}>{r.ok ? <CheckCircle className="w-3 h-3" /> : <X className="w-3 h-3" />}{r.label} {r.ver ? `(${r.ver})` : ''}</div>
                ))}
              </div>
            </div>
          )}

          {/* Toolbar */}
          <div className="flex items-center gap-2 flex-wrap">
            <button onClick={() => setShowTemplates(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"><BookOpen className="w-3.5 h-3.5" /> {t('trainingPanel.toolbar.templatesButton')}</button>
            <button onClick={() => { setAiInitialGoal(''); setShowAIAssistant(true); }} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 text-violet-300 text-xs font-medium transition-all"><Sparkles className="w-3.5 h-3.5" /> {t('trainingPanel.toolbar.aiButton')}</button>
            <button onClick={() => { updateConfig(DEFAULT_CONFIG); setLossPoints([]); }} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all"><RefreshCw className="w-3.5 h-3.5" /> {t('trainingPanel.toolbar.resetButton')}</button>
          </div>

          {/* Config Sections */}
          <div className="space-y-3">
            <SectionCard title={t('trainingPanel.sections.basic')} icon={<Settings2 className="w-4 h-4 text-emerald-400" />} expanded={sections.basic} onToggle={() => toggleSection('basic')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label={t('trainingPanel.fields.epochs')} tooltip={t('trainingPanel.fields.epochsTooltip')}><NumInput value={config.epochs} onChange={v => updateConfig({ epochs: v })} min={1} max={100} step={1} /></Field>
                <Field label={t('trainingPanel.fields.batchSize')}><NumInput value={config.batch_size} onChange={v => updateConfig({ batch_size: v })} min={1} step={1} /></Field>
                <Field label={t('trainingPanel.fields.learningRate')} tooltip={t('trainingPanel.fields.learningRateTooltip')}><NumInput value={config.learning_rate} onChange={v => updateConfig({ learning_rate: v })} step={0.000001} /></Field>
                <Field label={t('trainingPanel.fields.maxSeqLength')} tooltip={t('trainingPanel.fields.maxSeqLengthTooltip')}><NumInput value={config.max_seq_length} onChange={v => updateConfig({ max_seq_length: v })} min={16} max={512} step={16} /></Field>
                <Field label={t('trainingPanel.fields.warmupRatio')}><NumInput value={config.warmup_ratio} onChange={v => updateConfig({ warmup_ratio: v })} step={0.01} min={0} max={0.3} /></Field>
                <Field label={t('trainingPanel.fields.gradientAccumulation')} tooltip={t('trainingPanel.fields.gradientAccumulationTooltip')}><NumInput value={config.gradient_accumulation_steps} onChange={v => updateConfig({ gradient_accumulation_steps: v })} min={1} step={1} /></Field>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-1">
                <Toggle checked={config.fp16} onChange={v => updateConfig({ fp16: v, bf16: v ? false : config.bf16 })} label={t('trainingPanel.fields.fp16')} />
                <Toggle checked={config.bf16} onChange={v => updateConfig({ bf16: v, fp16: v ? false : config.fp16 })} label={t('trainingPanel.fields.bf16')} />
              </div>
            </SectionCard>

            <SectionCard title={t('trainingPanel.sections.optimizer')} icon={<Gauge className="w-4 h-4 text-blue-400" />} expanded={sections.optimizer} onToggle={() => toggleSection('optimizer')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label={t('trainingPanel.fields.optimizer')}><SelectInput value={config.optimizer} onChange={v => updateConfig({ optimizer: v })} options={[{value:'adamw',label:'AdamW'},{value:'adam',label:'Adam'},{value:'sgd',label:'SGD'},{value:'adafactor',label:'Adafactor'}]} /></Field>
                <Field label={t('trainingPanel.fields.scheduler')}><SelectInput value={config.scheduler} onChange={v => updateConfig({ scheduler: v })} options={[{value:'linear',label:'Linear'},{value:'cosine',label:'Cosine'},{value:'constant',label:'Constant'},{value:'polynomial',label:'Polynomial'}]} /></Field>
                <Field label={t('trainingPanel.fields.weightDecay')}><NumInput value={config.weight_decay} onChange={v => updateConfig({ weight_decay: v })} step={0.001} min={0} /></Field>
                <Field label={t('trainingPanel.fields.maxGradNorm')}><NumInput value={config.max_grad_norm} onChange={v => updateConfig({ max_grad_norm: v })} step={0.1} min={0} /></Field>
                <Field label={t('trainingPanel.fields.adamBeta1')} tooltip={t('trainingPanel.fields.adamBeta1Tooltip')}><NumInput value={config.adam_beta1} onChange={v => updateConfig({ adam_beta1: v })} step={0.001} min={0} max={1} /></Field>
                <Field label={t('trainingPanel.fields.adamBeta2')} tooltip={t('trainingPanel.fields.adamBeta2Tooltip')}><NumInput value={config.adam_beta2} onChange={v => updateConfig({ adam_beta2: v })} step={0.0001} min={0} max={1} /></Field>
                <Field label={t('trainingPanel.fields.adamEpsilon')} tooltip={t('trainingPanel.fields.adamEpsilonTooltip')} ><NumInput value={config.adam_epsilon} onChange={v => updateConfig({ adam_epsilon: v })} step={1e-9} min={0} /></Field>
              </div>
            </SectionCard>

            <SectionCard title={t('trainingPanel.sections.advanced')} icon={<SlidersHorizontal className="w-4 h-4 text-purple-400" />} expanded={sections.advanced} onToggle={() => toggleSection('advanced')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label={t('trainingPanel.fields.dropout')}><NumInput value={config.dropout} onChange={v => updateConfig({ dropout: v })} step={0.01} min={0} max={0.5} /></Field>
                <Field label={t('trainingPanel.fields.labelSmoothing')}><NumInput value={config.label_smoothing} onChange={v => updateConfig({ label_smoothing: v })} step={0.01} min={0} max={0.3} /></Field>
                <Field label={t('trainingPanel.fields.warmupSteps')} tooltip={t('trainingPanel.fields.warmupStepsTooltip')}><NumInput value={config.warmup_steps} onChange={v => updateConfig({ warmup_steps: v })} min={0} step={10} /></Field>
                <Field label={t('trainingPanel.fields.maxSteps')} tooltip={t('trainingPanel.fields.maxStepsTooltip')}><NumInput value={config.max_steps} onChange={v => updateConfig({ max_steps: v })} min={-1} step={100} /></Field>
                <Field label={t('trainingPanel.fields.evalStrategy')}><SelectInput value={config.eval_strategy} onChange={v => updateConfig({ eval_strategy: v })} options={[{value:'epoch',label:t('trainingPanel.fields.evalStrategyEpoch')},{value:'steps',label:t('trainingPanel.fields.evalStrategySteps')},{value:'no',label:t('trainingPanel.fields.evalStrategyNone')}]} /></Field>
                <Field label={t('trainingPanel.fields.evalSteps')} tooltip={t('trainingPanel.fields.evalStepsTooltip')}><NumInput value={config.eval_steps} onChange={v => updateConfig({ eval_steps: v })} min={1} step={100} /></Field>
                <Field label={t('trainingPanel.fields.saveSteps')}><NumInput value={config.save_steps} onChange={v => updateConfig({ save_steps: v })} min={1} step={100} /></Field>
                <Field label={t('trainingPanel.fields.saveTotalLimit')} tooltip={t('trainingPanel.fields.saveTotalLimitTooltip')}><NumInput value={config.save_total_limit} onChange={v => updateConfig({ save_total_limit: v })} min={1} step={1} /></Field>
                <Field label={t('trainingPanel.fields.loggingSteps')}><NumInput value={config.logging_steps} onChange={v => updateConfig({ logging_steps: v })} min={1} step={5} /></Field>
                <Field label={t('trainingPanel.fields.seed')}><NumInput value={config.seed} onChange={v => updateConfig({ seed: v })} min={0} step={1} /></Field>
                <Field label={t('trainingPanel.fields.numWorkers')} tooltip={t('trainingPanel.fields.numWorkersTooltip')}><NumInput value={config.num_workers} onChange={v => updateConfig({ num_workers: v })} min={0} max={8} step={1} /></Field>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-2">
                <Toggle checked={config.gradient_checkpointing} onChange={v => updateConfig({ gradient_checkpointing: v })} label={t('trainingPanel.fields.gradientCheckpointing')} />
                <Toggle checked={config.group_by_length} onChange={v => updateConfig({ group_by_length: v })} label={t('trainingPanel.fields.groupByLength')} />
                <Toggle checked={config.pin_memory} onChange={v => updateConfig({ pin_memory: v })} label={t('trainingPanel.fields.pinMemory')} />
              </div>
            </SectionCard>

            <SectionCard
              title={t('trainingPanel.sections.lora')}
              icon={<span className="text-violet-400 text-sm font-bold w-4 h-4 flex items-center justify-center">L</span>}
              expanded={sections.lora}
              onToggle={() => toggleSection('lora')}
              badge={config.use_lora ? <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded-md bg-violet-500/20 text-violet-300 border border-violet-500/30">{t('trainingPanel.sections.loraActiveBadge')}</span> : undefined}
            >
              <div className="space-y-3">
                <div className="grid grid-cols-1 gap-3 pb-1">
                  <Toggle checked={config.use_lora} onChange={v => updateConfig({ use_lora: v, load_in_4bit: v ? config.load_in_4bit : false, load_in_8bit: v ? config.load_in_8bit : false })} label={t('trainingPanel.fields.loraActivate')} />
                  <Toggle checked={config.load_in_4bit} onChange={v => updateConfig({ load_in_4bit: v, load_in_8bit: v ? false : config.load_in_8bit, use_lora: v ? true : config.use_lora })} label={t('trainingPanel.fields.lora4bit')} />
                  <Toggle checked={config.load_in_8bit} onChange={v => updateConfig({ load_in_8bit: v, load_in_4bit: v ? false : config.load_in_4bit, use_lora: v ? true : config.use_lora })} label={t('trainingPanel.fields.lora8bit')} />
                </div>
                {config.use_lora ? (
                  <div className="grid grid-cols-2 gap-4 pt-1 border-t border-white/10">
                    <Field label={t('trainingPanel.fields.loraRank')} tooltip={t('trainingPanel.fields.loraRankTooltip')}><NumInput value={config.lora_r} onChange={v => updateConfig({ lora_r: v })} min={1} max={256} step={2} /></Field>
                    <Field label={t('trainingPanel.fields.loraAlpha')} tooltip={t('trainingPanel.fields.loraAlphaTooltip')}><NumInput value={config.lora_alpha} onChange={v => updateConfig({ lora_alpha: v })} min={1} step={1} /></Field>
                    <Field label={t('trainingPanel.fields.loraDropout')} tooltip={t('trainingPanel.fields.loraDropoutTooltip')}><NumInput value={config.lora_dropout} onChange={v => updateConfig({ lora_dropout: v })} step={0.01} min={0} max={0.5} /></Field>
                    <Field label={t('trainingPanel.fields.loraTargetModules')} tooltip={t('trainingPanel.fields.loraTargetModulesTooltip')}>
                      <input
                        type="text"
                        value={config.lora_target_modules}
                        onChange={e => updateConfig({ lora_target_modules: e.target.value })}
                        placeholder={t('trainingPanel.fields.loraTargetModulesPlaceholder')}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-violet-500/50 transition-all font-mono"
                      />
                    </Field>
                  </div>
                ) : (
                  <div className="px-3 py-2.5 rounded-xl bg-violet-500/[0.08] border border-violet-500/15">
                    <p className="text-violet-300 text-xs flex items-start gap-2">
                      <Lightbulb className="w-4 h-4 flex-shrink-0 mt-0.5" />
                      <span>{t('trainingPanel.fields.loraHint')}</span>
                    </p>
                  </div>
                )}
              </div>
            </SectionCard>

            <SectionCard title={t('trainingPanel.ramCalculator.title')} icon={<MemoryStick className="w-4 h-4 text-amber-400" />} expanded={sections.ram} onToggle={() => toggleSection('ram')}>
              <RamCalculator config={config} modelSizeGb={modelSizeGb} />
            </SectionCard>
          </div>

          {/* Progress (inline, für wenn Dashboard minimiert oder geschlossen) */}
          {currentJob && !trainingState.showDashboard && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {isRunning ? <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" /> : currentJob.status === 'completed' ? <CheckCircle className="w-4 h-4 text-emerald-400" /> : <AlertCircle className="w-4 h-4 text-red-400" />}
                  <span className="text-white font-medium text-sm">{isRunning ? t('trainingPanel.progress.title') : t('trainingPanel.progress.statusLabel').replace('{status}', currentJob.status)}</span>
                </div>
                <button onClick={() => { setShowDashboardContext(true); setIsDashMinimizedContext(false); }} className="text-xs text-gray-500 hover:text-white px-2 py-1 rounded-lg bg-white/5 transition-all">{t('trainingPanel.progress.openDashboardButton')}</button>
              </div>
              {progress && <div className="h-2 rounded-full bg-white/10 overflow-hidden"><div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${progress.progress_percent}%` }} /></div>}
              {lossPoints.length > 1 && <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3"><p className="text-xs text-gray-500 mb-2">{t('trainingPanel.progress.lossHistory')}</p><LossChart points={lossPoints} /></div>}
            </div>
          )}

          {/* Start / Stop */}
          <div className="flex gap-3">
            {isRunning ? (
              <button onClick={handleStopTraining} className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-sm transition-all">
                <Square className="w-4 h-4" /> {t('trainingPanel.actions.stopButton')}
              </button>
            ) : (
              <button onClick={handleStartTraining} disabled={!selectedModelId || !selectedDatasetId || (selectedDataset?.status !== 'split' && !selectedModelId?.startsWith('canvas_'))}
                className={`flex-1 flex items-center justify-center gap-2 py-3.5 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg`}>
                <Play className="w-4 h-4" /> {t('trainingPanel.actions.startButton')}
              </button>
            )}
          </div>
          {selectedDataset?.status !== 'split' && selectedDataset && !selectedModelId?.startsWith('canvas_') && (
            <p className="text-amber-400 text-xs text-center inline-flex items-center justify-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              {t('trainingPanel.actions.noSplitWarning')}
            </p>
          )}
        </>
      )}

      {/* Nicht unterstützt */}
      {mode === 'train' && !isSupported && selectedModel && (
        <div className="p-6 rounded-2xl border border-amber-500/30 bg-amber-500/5 space-y-3">
          <div className="flex items-start gap-3"><AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" /><div><p className="text-amber-300 font-semibold">{t('trainingPanel.unsupported.title')}</p><p className="text-gray-400 text-sm mt-1">{t('trainingPanel.unsupported.description')}</p></div></div>
          <button onClick={() => setMode('dev')} className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-sm font-medium transition-all"><Code2 className="w-4 h-4" /> {t('trainingPanel.unsupported.devModeButton')}</button>
        </div>
      )}

      {showTemplates && <TemplatesModal onApply={updateConfig} onClose={() => setShowTemplates(false)} onSave={handleSaveTemplate} currentConfig={config} />}
      {showAIAssistant && (
        <AIMetricAssistant
          config={config}
          datasetName={selectedDataset?.name ?? ''}
          datasetSize={selectedDataset?.file_count ?? 0}
          modelName={selectedModel?.name ?? ''}
          onApply={updateConfig}
          onClose={() => setShowAIAssistant(false)}
          onSaveAsTemplate={handleSaveAIAsTemplate}
          initialGoal={aiInitialGoal}
        />
      )}
      {/* TrainingDashboard wird jetzt global in Dashboard.tsx als Overlay gerendert */}
    </div>
  );
}
