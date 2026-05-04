// TrainingPanel.tsx – Vollständiges Training-Interface (v5 – LoRA/QLoRA + Error Recovery)

import { useState, useEffect, useRef, useCallback, useContext } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play, Square, Settings2, Loader2, ChevronDown, ChevronRight,
  Gauge, TrendingDown, Zap, AlertCircle, CheckCircle, Sparkles,
  Trash2, X, HelpCircle, BarChart3, MemoryStick, SlidersHorizontal,
  BookOpen, Code2, RefreshCw,
  AlertTriangle, Layers, Save, Plus, ChevronRight as ChevronRightIcon,
  ClipboardList, History, Check,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import { useAISettings } from '../contexts/AISettingsContext';
import { useTrainingContext } from '../contexts/TrainingContext';
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
interface TrainingPanelProps { onNavigateToAnalysis: (versionId: string) => void; }

// ── AI Helper ─────────────────────────────────────────────────────────────

import type { AISettings } from '../contexts/AISettingsContext';

export async function callAI(settings: AISettings, systemPrompt: string, userPrompt: string, history?: { role: 'user' | 'assistant'; content: string }[]): Promise<string> {
  if (!settings.enabled) throw new Error('KI-Assistent deaktiviert. Bitte in Einstellungen aktivieren.');
  const needsKey = settings.provider !== 'ollama';
  if (needsKey && !settings.apiKey) throw new Error(`API-Key für ${settings.provider} fehlt.`);
  const msgs = [...(history ?? []), { role: 'user' as const, content: userPrompt }];

  if (settings.provider === 'anthropic') {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': settings.apiKey, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model: settings.selectedModel || 'claude-haiku-4-5', max_tokens: 2000, system: systemPrompt, messages: msgs }),
    });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error((e as {error?: {message?: string}})?.error?.message ?? `HTTP ${res.status}`); }
    const data = await res.json();
    return data.content?.find((b: { type: string }) => b.type === 'text')?.text ?? '';
  }
  if (settings.provider === 'openai' || settings.provider === 'groq') {
    const url = settings.provider === 'groq' ? 'https://api.groq.com/openai/v1/chat/completions' : 'https://api.openai.com/v1/chat/completions';
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${settings.apiKey}` },
      body: JSON.stringify({ model: settings.selectedModel || (settings.provider === 'groq' ? 'llama-3.3-70b-versatile' : 'gpt-4o-mini'), max_tokens: 2000, messages: [{ role: 'system', content: systemPrompt }, ...msgs] }),
    });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error((e as {error?: {message?: string}})?.error?.message ?? `HTTP ${res.status}`); }
    return (await res.json()).choices?.[0]?.message?.content ?? '';
  }
  const ctx = [systemPrompt, ...(history ?? []).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`), `User: ${userPrompt}`, 'Assistant:'].join('\n\n');
  const res = await fetch('http://localhost:11434/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: settings.ollamaModel || 'llama3.2', prompt: ctx, stream: false }) });
  if (!res.ok) throw new Error('Ollama nicht erreichbar (localhost:11434)');
  return (await res.json()).response ?? '';
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

const BUILTIN_TEMPLATES: MetricsTemplate[] = [
  { id: 'standard', name: '📘 Standard', description: 'Balance für die meisten Aufgaben.', config: { epochs: 3, batch_size: 8, learning_rate: 2e-5, warmup_ratio: 0.06, max_seq_length: 128 }, source: 'builtin' },
  { id: 'small', name: '⚡ Kleines Dataset (<5k)', description: 'Mehr Epochen, höhere LR.', config: { epochs: 5, batch_size: 8, learning_rate: 3e-5, warmup_ratio: 0.1, max_seq_length: 64 }, source: 'builtin' },
  { id: 'large', name: '🎯 Großes Dataset (>50k)', description: 'Weniger Epochen, größere Batch.', config: { epochs: 2, batch_size: 32, learning_rate: 1e-5, warmup_ratio: 0.04, max_seq_length: 256 }, source: 'builtin' },
  { id: 'lowram', name: '💾 Low RAM', description: 'Grad Accumulation + FP16.', config: { epochs: 4, batch_size: 2, learning_rate: 2e-5, gradient_accumulation_steps: 8, max_seq_length: 64, fp16: true, gradient_checkpointing: true }, source: 'builtin' },
  { id: 'lora', name: '🔬 LoRA (RAM-sparend)', description: 'LoRA r=8, ideal für große Modelle.', config: { use_lora: true, lora_r: 8, lora_alpha: 16, lora_dropout: 0.05, gradient_checkpointing: true, batch_size: 4 }, source: 'builtin' },
  { id: 'qlora', name: '⚡ QLoRA 4-bit', description: 'QLoRA für maximale RAM-Ersparnis.', config: { use_lora: true, load_in_4bit: true, lora_r: 16, lora_alpha: 32, lora_dropout: 0.05, gradient_checkpointing: true, batch_size: 2, gradient_accumulation_steps: 8 }, source: 'builtin' },
];

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
        <span className="text-sm font-medium text-white">RAM-Schätzung</span>
        <span className="text-xs text-gray-500">({modelSizeGb.toFixed(2)} GB Modell{config.use_lora ? ' · LoRA' : ''}{isQuantized ? ` · ${config.load_in_4bit ? '4bit' : '8bit'}` : ''})</span>
      </div>
      <div className="space-y-1.5 text-xs">
        {[
          ['Modell-Gewichte', weightRam],
          [config.use_lora ? 'LoRA-Gradienten (~5%)' : 'Gradienten', gradRam],
          [config.use_lora ? 'LoRA-Optimizer (~10%)' : 'Optimizer (Adam)', optimizerRam],
          [`Aktivierungen (Batch ${config.batch_size}${config.gradient_checkpointing ? ' · GradCkpt' : ''})`, activationRam],
          ['Misc', 0.4],
        ].map(([l, v]) => (
          <div key={l as string} className="flex justify-between"><span className="text-gray-400">{l as string}</span><span className="text-gray-300 tabular-nums">{(v as number).toFixed(2)} GB</span></div>
        ))}
        <div className="flex justify-between pt-2 border-t border-white/10 font-semibold"><span className="text-gray-300">Gesamt</span><span className={`${color} tabular-nums`}>~{total.toFixed(1)} GB</span></div>
      </div>
      {!isFp16 && !config.use_lora && total > 8 && <p className="text-amber-400 text-xs bg-amber-500/10 rounded-lg px-3 py-2">💡 FP16 aktivieren spart ~{(gradRam * 0.5 + activationRam * 0.5).toFixed(1)} GB RAM</p>}
      {!config.use_lora && total > 12 && <p className="text-violet-300 text-xs bg-violet-500/10 rounded-lg px-3 py-2">💡 LoRA aktivieren reduziert den Bedarf auf ~{(total * 0.15).toFixed(1)} GB</p>}
      {total > 20 && <div className="text-red-300 text-xs bg-red-500/10 rounded-lg px-3 py-2 space-y-1"><p className="font-medium">RAM senken:</p><p>• LoRA r=8, 4-bit QLoRA, Batch 2–4, FP16 + Grad Checkpointing</p></div>}
    </div>
  );
}

// ── Loss Chart ─────────────────────────────────────────────────────────────

export function LossChart({ points }: { points: LossPoint[] }) {
  if (points.length < 2) return <div className="h-32 flex items-center justify-center text-gray-600 text-xs">Warte auf Trainings-Daten…</div>;
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
      <circle cx={PAD.l + 4} cy={H - 10} r="4" fill="#10b981" /><text x={PAD.l + 12} y={H - 6} fill="rgba(255,255,255,0.5)" fontSize="9">Train</text>
      {vals.length > 0 && <><line x1={PAD.l + 70} y1={H - 10} x2={PAD.l + 84} y2={H - 10} stroke="#a855f7" strokeWidth="2" strokeDasharray="3,2" /><text x={PAD.l + 88} y={H - 6} fill="rgba(255,255,255,0.5)" fontSize="9">Val</text></>}
    </svg>
  );
}

// ── Templates Modal ────────────────────────────────────────────────────────

function TemplatesModal({ onApply, onClose, onSave, currentConfig }: { onApply: (cfg: Partial<TrainingConfig>) => void; onClose: () => void; onSave: (name: string, desc: string) => void; currentConfig: TrainingConfig; }) {
  const [userTemplates, setUserTemplates] = useState<MetricsTemplate[]>([]);
  const [tab, setTab] = useState<'builtin' | 'user'>('builtin');
  const [saveName, setSaveName] = useState('');
  const [saveDesc, setSaveDesc] = useState('');
  const [showSaveForm, setShowSaveForm] = useState(false);
  const { success, error } = useNotification();

  useEffect(() => { invoke<MetricsTemplate[]>('get_metrics_templates').then(setUserTemplates).catch(() => {}); }, []);

  const handleDelete = async (id: string) => {
    try { await invoke('delete_metrics_template', { templateId: id }); setUserTemplates(t => t.filter(x => x.id !== id)); success('Gelöscht', ''); }
    catch (e) { error('Fehler', String(e)); }
  };

  const handleSave = () => {
    if (!saveName.trim()) return;
    onSave(saveName.trim(), saveDesc.trim());
    setSaveName(''); setSaveDesc(''); setShowSaveForm(false);
    setTimeout(() => invoke<MetricsTemplate[]>('get_metrics_templates').then(setUserTemplates).catch(() => {}), 600);
  };

  void currentConfig;
  const all = tab === 'builtin' ? BUILTIN_TEMPLATES : userTemplates;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[85vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-2"><BookOpen className="w-5 h-5 text-blue-400" /><h2 className="text-lg font-bold text-white">Metriken-Templates</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>
        <div className="flex border-b border-white/10 px-6 flex-shrink-0">
          {(['builtin', 'user'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} className={`px-4 py-3 text-sm font-medium border-b-2 transition-all ${tab === t ? 'text-blue-300 border-blue-400' : 'text-gray-400 hover:text-white border-transparent'}`}>
              {t === 'builtin' ? '⚡ Vordefiniert' : `📁 Meine (${userTemplates.length})`}
            </button>
          ))}
        </div>
        <div className="p-5 overflow-y-auto flex-1 space-y-3">
          {all.length === 0 ? <p className="text-gray-500 text-sm text-center py-8">Keine Templates vorhanden.</p> : all.map(t => (
            <div key={t.id} className="p-4 rounded-xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-all group">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium text-sm">{t.name}</p>
                  {t.description && <p className="text-gray-500 text-xs mt-0.5">{t.description}</p>}
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {t.config.learning_rate != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">LR: {t.config.learning_rate}</span>}
                    {t.config.batch_size     != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/20">Batch: {t.config.batch_size}</span>}
                    {t.config.epochs         != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/15 text-purple-400 border border-purple-500/20">Epochs: {t.config.epochs}</span>}
                    {t.config.max_seq_length != null && <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/20">Seq: {t.config.max_seq_length}</span>}
                    {t.config.fp16           && <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 border border-cyan-500/20">FP16</span>}
                    {t.config.use_lora       && <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-400 border border-violet-500/20">LoRA r={t.config.lora_r}</span>}
                    {t.config.load_in_4bit   && <span className="text-[10px] px-2 py-0.5 rounded-full bg-fuchsia-500/15 text-fuchsia-400 border border-fuchsia-500/20">QLoRA 4bit</span>}
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {tab === 'user' && <button onClick={() => handleDelete(t.id)} className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>}
                  <button onClick={() => { onApply(t.config); onClose(); }} className="px-3 py-1.5 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-xs font-medium transition-all">Laden</button>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 pb-5 flex-shrink-0 border-t border-white/10 pt-4">
          {showSaveForm ? (
            <div className="space-y-2">
              <input value={saveName} onChange={e => setSaveName(e.target.value)} placeholder="Template-Name…" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/20" />
              <input value={saveDesc} onChange={e => setSaveDesc(e.target.value)} placeholder="Beschreibung (optional)…" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/20" />
              <div className="flex gap-2">
                <button onClick={handleSave} disabled={!saveName.trim()} className="flex-1 py-2 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-sm font-medium transition-all disabled:opacity-40"><Save className="w-3.5 h-3.5 inline mr-1.5" />Speichern</button>
                <button onClick={() => setShowSaveForm(false)} className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all">Abbrechen</button>
              </div>
            </div>
          ) : (
            <button onClick={() => { setTab('user'); setShowSaveForm(true); }} className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm transition-all">
              <Plus className="w-4 h-4" /> Aktuelle Config als Template speichern
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
  const { settings: aiSettings } = useAISettings();
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
      const text = await callAI(aiSettings, 'Du bist ein präziser ML-Experte. Antworte auf Deutsch. Gib am Ende exakt ein valides JSON-Objekt aus.', prompt);
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
          <div className="flex items-center gap-2"><Sparkles className="w-5 h-5 text-violet-400" /><h2 className="text-lg font-bold text-white">KI-Metrik-Assistent</h2></div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all"><X className="w-5 h-5" /></button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {phase === 'input' ? (
            <div className="p-6 space-y-5">
              <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-2">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">Aktuelle Konfiguration</p>
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
                  <p className="text-sm font-medium text-gray-400">Vorherige Trainings-Analysen</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-amber-500/15 text-amber-400 border border-amber-500/20">Bald</span>
                </div>
                <p className="text-xs text-gray-600">Sobald Trainings-Analysen verfügbar sind, kann die KI auf vorherige Loss-Kurven zugreifen.</p>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-white">Ziel / Problem</label>
                <p className="text-xs text-gray-500">z. B. Training schlägt fehl, weniger Overfitting, geringerer RAM-Verbrauch, bessere Genauigkeit…</p>
                <textarea
                  value={goalText}
                  onChange={e => setGoalText(e.target.value)}
                  placeholder="z. B. Ich bekomme einen Out-of-Memory Fehler mit 16 GB RAM. Modell ist 3B Parameter."
                  rows={4}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-violet-500/50 resize-none transition-all"
                />
              </div>

              <button onClick={ask} className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:opacity-90 text-white font-semibold text-sm transition-all">
                <Sparkles className="w-4 h-4" /> KI-Analyse starten
              </button>
            </div>
          ) : (
            <div className="p-6 space-y-5">
              {loading ? (
                <div className="flex flex-col items-center gap-3 py-12">
                  <Loader2 className="w-10 h-10 text-violet-400 animate-spin" />
                  <p className="text-gray-400 text-sm">Analysiere Hyperparameter…</p>
                  {goalText && <p className="text-gray-600 text-xs max-w-sm text-center">Berücksichtige: {goalText.slice(0, 80)}…</p>}
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
                        Vorgeschlagene Änderungen ({Object.keys(parsed).length} Metriken)
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
                            <Check className="w-4 h-4" /> Übernommen!
                          </div>
                        ) : (
                          <div className="flex gap-2">
                            <button onClick={() => { onApply(parsed); setApplied(true); }} className="flex-1 py-2.5 rounded-xl bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/40 text-violet-300 text-sm font-medium transition-all">
                              ✅ Alle {Object.keys(parsed).length} Metriken übernehmen
                            </button>
                            <button onClick={onClose} className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-sm transition-all">Verwerfen</button>
                          </div>
                        )}
                        {applied && (
                          savedAsTemplate ? (
                            <div className="flex items-center gap-2 justify-center py-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs">
                              <Check className="w-3.5 h-3.5" /> Als Template gespeichert
                            </div>
                          ) : (
                            <button onClick={() => { onSaveAsTemplate(parsed); setSavedAsTemplate(true); }}
                              className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"
                            >
                              <BookOpen className="w-3.5 h-3.5" /> Als Metriken-Template speichern
                            </button>
                          )
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
            <button onClick={() => setPhase('input')} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all">← Zurück</button>
          )}
          {phase === 'result' && !loading && (
            <button onClick={ask} disabled={loading} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all">
              <RefreshCw className="w-3.5 h-3.5" /> Neu analysieren
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function TrainingPanel({ onNavigateToAnalysis }: TrainingPanelProps) {
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
      successRef.current('Training abgeschlossen! 🎉', 'Das Modell wurde gespeichert.');
      // Version-ID im Context speichern → TrainingDashboard zeigt "Analyse starten"-Button
      if (e.payload.new_version_id) {
        setCompletedVersionIdContextRef.current(e.payload.new_version_id);
      }
      // Kein automatisches onNavigateToAnalysis mehr – User entscheidet selbst über Dashboard-Button
    }).then(fn => { u2 = fn; });
    listen<{ data?: { error?: string } }>('training-error', e => {
      setCurrentJob(j => (j ? { ...j, status: 'failed', error: e.payload.data?.error ?? 'Fehler' } : null));
      invoke('disable_prevent_sleep').catch(() => {});
    }).then(fn => { u3 = fn; });
    return () => { u1?.(); u2?.(); u3?.(); };
  }, []);

  useEffect(() => {
    setCurrentPageContent(['=== FrameTrain Training ===', `Modell: ${selectedModel?.name ?? '—'}`, `Dataset: ${selectedDataset?.name ?? '—'}`, `Modus: ${mode}`].join('\n'));
  }, [selectedModelId, selectedDatasetId, mode]);

  const selectedModel   = models.find(m => m.id === selectedModelId);
  const selectedDataset = datasets.find(d => d.id === selectedDatasetId);
  const selectedModelTree = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersionTree = selectedModelTree?.versions.find(v => v.id === selectedVersionId);
  const detectionKey    = selectedModel?.source_path ?? selectedModel?.name ?? '';
  const detection       = detectionKey ? detectPlugin(detectionKey) : null;
  const isSupported     = detection?.supported === true;
  const pluginId        = detection?.supported ? detection.plugin.id : null;

  const handleStartTraining = async () => {
    if (!selectedModelId || !selectedDatasetId) { warning('Fehlende Auswahl', 'Bitte Modell und Dataset wählen.'); return; }
    if (!isSupported) { warning('Nicht unterstützt', 'Nutze den Dev Train Modus.'); return; }
    if (selectedDataset?.status !== 'split') { warning('Kein Split', 'Das Dataset muss erst im Dataset-Manager aufgeteilt werden.'); return; }

    setLossPoints([]);
    setLossPointsContext([]);
    try {
      // Konvertiere lora_target_modules von String zu Array falls nötig
      const configForBackend = {
        ...config,
        lora_target_modules: typeof config.lora_target_modules === 'string'
          ? config.lora_target_modules.split(',').map(m => m.trim()).filter(m => m)
          : config.lora_target_modules,
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
      
      success('Training gestartet!', 'Job läuft…');
    } catch (err: unknown) { error('Start fehlgeschlagen', String(err)); }
  };

  const handleStopTraining = async () => {
    try { await invoke('stop_training'); invoke('disable_prevent_sleep').catch(() => {}); success('Gestoppt', ''); } catch (err: unknown) { error('Fehler', String(err)); }
  };

  const handleSaveTemplate = async (name: string, desc: string) => {
    try { await invoke('save_metrics_template', { name, description: desc, config, source: 'user' }); success('Template gespeichert!', name); }
    catch (err: unknown) { error('Fehler', String(err)); }
  };

  const handleSaveAIAsTemplate = async (cfg: Partial<TrainingConfig>) => {
    try { await invoke('save_metrics_template', { name: `KI-Vorschlag ${new Date().toLocaleDateString('de-DE')}`, description: 'Automatisch vom KI-Assistenten erstellt.', config: { ...DEFAULT_CONFIG, ...cfg }, source: 'ai' }); success('Template gespeichert!', 'KI-Vorschlag als Template hinterlegt.'); }
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
                <h2 className="text-lg font-bold text-white">Trainings-Verlauf</h2>
              </div>
              <button onClick={() => setShowHistory(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            {/* Filter */}
            <div className="flex gap-2 px-6 py-3 border-b border-white/10 flex-wrap">
              {(['all', 'running', 'completed', 'failed', 'stopped'] as const).map(f => {
                const labels: Record<string, string> = { all: 'Alle', running: '▶ Läuft', completed: '✅ Erfolgreich', failed: '❌ Fehlgeschlagen', stopped: '⏹ Gestoppt' };
                const colors: Record<string, string> = { all: 'bg-white/10 text-white', running: 'bg-blue-500/20 text-blue-300 border-blue-500/30', completed: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30', failed: 'bg-red-500/20 text-red-300 border-red-500/30', stopped: 'bg-gray-500/20 text-gray-300 border-gray-500/30' };
                return (
                  <button key={f} onClick={() => setHistoryFilter(f)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
                      historyFilter === f ? colors[f] : 'bg-white/5 border-white/10 text-gray-500 hover:text-gray-300'
                    }`}>
                    {labels[f]}
                  </button>
                );
              })}
              <div className="ml-auto text-xs text-gray-500 flex items-center">
                {historyJobs.filter(j => historyFilter === 'all' || j.status === historyFilter).length} Einträge
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
                    <p className="text-sm">Keine Trainings in dieser Kategorie</p>
                  </div>
                );
                return filtered.map(job => {
                  const statusIcon: Record<string, string> = { completed: '✅', failed: '❌', stopped: '⏹', running: '▶', pending: '⏳' };
                  const statusColor: Record<string, string> = { completed: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', failed: 'text-red-400 bg-red-500/10 border-red-500/20', stopped: 'text-gray-400 bg-gray-500/10 border-gray-500/20', running: 'text-blue-400 bg-blue-500/10 border-blue-500/20', pending: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20' };
                  const durMs = job.completed_at && job.started_at ? new Date(job.completed_at).getTime() - new Date(job.started_at).getTime() : null;
                  const durStr = durMs ? (durMs > 3600000 ? `${Math.floor(durMs/3600000)}h ${Math.floor((durMs%3600000)/60000)}m` : durMs > 60000 ? `${Math.floor(durMs/60000)}m ${Math.floor((durMs%60000)/1000)}s` : `${Math.floor(durMs/1000)}s`) : null;
                  return (
                    <div key={job.id} className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${statusColor[job.status] ?? statusColor.pending}`}>
                              {statusIcon[job.status] ?? '⏳'} {job.status}
                            </span>
                            {durStr && <span className="text-xs text-gray-500">⏱ {durStr}</span>}
                          </div>
                          <p className="text-white font-medium text-sm truncate">{job.model_name}</p>
                          <p className="text-gray-500 text-xs truncate">Dataset: {job.dataset_name}</p>
                          {job.error && <p className="text-red-400 text-xs mt-1 truncate">Fehler: {job.error}</p>}
                        </div>
                        <div className="text-right flex-shrink-0">
                          <p className="text-xs text-gray-500">{new Date(job.created_at).toLocaleDateString('de-DE', { day:'2-digit', month:'2-digit', year:'2-digit' })}</p>
                          <p className="text-xs text-gray-600">{new Date(job.created_at).toLocaleTimeString('de-DE', { hour:'2-digit', minute:'2-digit' })}</p>
                          {job.progress && job.progress.progress_percent > 0 && (
                            <p className="text-xs text-gray-500 mt-1">{job.progress.progress_percent.toFixed(0)}% · Epoche {job.progress.epoch}/{job.progress.total_epochs}</p>
                          )}
                        </div>
                      </div>
                      {job.progress && job.progress.train_loss > 0 && (
                        <div className="mt-2 pt-2 border-t border-white/10 flex gap-4 text-xs text-gray-500">
                          <span>Loss: <span className="text-gray-300">{job.progress.train_loss.toFixed(4)}</span></span>
                          {job.progress.val_loss && <span>Val: <span className="text-gray-300">{job.progress.val_loss.toFixed(4)}</span></span>}
                          <span>Epoche {job.progress.epoch}/{job.progress.total_epochs}</span>
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
        <div><h1 className="text-2xl font-bold text-white">Training</h1><p className="text-gray-400 mt-1">Trainiere Sequenzklassifikations-Modelle</p></div>
        <div className="flex items-center gap-2">
          <button onClick={handleOpenHistory} className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 text-sm transition-all">
            <History className="w-4 h-4" />
            Verlauf
          </button>
          <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
            {(['train', 'dev'] as const).map(m => (
              <button key={m} onClick={() => setMode(m)} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${mode === m ? (m === 'train' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-blue-500/20 text-blue-300 border border-blue-500/30') : 'text-gray-400 hover:text-white'}`}>
                {m === 'train' ? <><Play className="w-3.5 h-3.5" /> Training</> : <><Code2 className="w-3.5 h-3.5" /> Dev Train</>}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Model & Dataset */}
      {models.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-12 text-center space-y-3">
          <Layers className="w-10 h-10 text-gray-500 mx-auto" />
          <p className="text-white font-medium">Kein Modell vorhanden</p>
          <p className="text-gray-500 text-sm">Füge zuerst ein Modell im Model-Manager hinzu.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Model + Dataset Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* Model Block (Parent of Version) */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-white">Modell</label>
                <select value={selectedModelId ?? ''} onChange={e => { setSelectedModelId(e.target.value); setSelectedDatasetId(null); }} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none mt-1">
                  {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-900">{m.name}</option>)}
                </select>
              </div>
              
              {/* Version Selection */}
              <div>
                <label className="block text-sm font-medium text-white">Version</label>
                <select value={selectedVersionId ?? ''} onChange={e => setSelectedVersionId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none mt-1">
                  {selectedModelTree?.versions?.length ? [...selectedModelTree.versions].sort((a, b) => b.version_number - a.version_number).map((v, idx) => (
                    <option key={v.id} value={v.id} className="bg-slate-900">
                      {v.name} {idx === 0 ? '(neueste)' : ''}
                    </option>
                  )) : <option value="">Keine Versionen</option>}
                </select>
              </div>
              
              {/* Support Info */}
              {selectedModel && (isSupported
                ? <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 mt-1"><CheckCircle className="w-4 h-4 text-emerald-400" /><span className="text-emerald-300 text-xs font-medium">{detection.plugin.name} – unterstützt</span></div>
                : <div className="space-y-2 mt-1">
                    <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20"><AlertTriangle className="w-4 h-4 text-amber-400" /><span className="text-amber-300 text-xs">Noch nicht unterstützt</span></div>
                    <button onClick={() => setMode('dev')} className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"><Code2 className="w-3.5 h-3.5" /> Dev Train Mode →</button>
                  </div>
              )}
            </div>

            {/* Dataset Selection */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
              <label className="block text-sm font-medium text-white">Dataset</label>
              {datasets.length === 0
                ? <p className="text-gray-500 text-sm">Kein Dataset für dieses Modell.</p>
                : <select value={selectedDatasetId ?? ''} onChange={e => setSelectedDatasetId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                    {datasets.map(d => <option key={d.id} value={d.id} className="bg-slate-900">{d.name} {d.status === 'split' ? '✅' : '⚠️'}</option>)}
                  </select>
              }
              {selectedDataset && pluginId && <DatasetCompatBadge modelPluginId={pluginId} extensions={selectedDataset.extensions ?? []} />}
              {selectedDataset?.status === 'unused' && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20"><AlertCircle className="w-3.5 h-3.5 text-amber-400" /><span className="text-amber-300 text-xs">Kein Split – erst im Dataset-Manager aufteilen.</span></div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Dev Train */}
      {mode === 'dev' && (
        <DevTrainPanel modelInfo={selectedModel ?? null} selectedVersionPath={selectedVersionPath} datasets={datasets} onNavigateToAnalysis={onNavigateToAnalysis} />
      )}

      {/* Standard Training */}
      {mode === 'train' && isSupported && (
        <>
          {reqs && !reqs.ready && (
            <div className="p-4 rounded-2xl border border-red-500/30 bg-red-500/10 space-y-2">
              <div className="flex items-center gap-2"><AlertCircle className="w-4 h-4 text-red-400" /><span className="text-red-300 font-medium text-sm">Python-Umgebung nicht bereit</span></div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {[{label:'Python',ok:reqs.python_installed,ver:reqs.python_version},{label:'PyTorch',ok:reqs.torch_installed,ver:reqs.torch_version},{label:'Transformers',ok:reqs.transformers_installed},{label:'CUDA/MPS',ok:reqs.cuda_available||reqs.mps_available}].map(r => (
                  <div key={r.label} className={`flex items-center gap-1.5 ${r.ok ? 'text-emerald-400' : 'text-red-400'}`}>{r.ok ? <CheckCircle className="w-3 h-3" /> : <X className="w-3 h-3" />}{r.label} {r.ver ? `(${r.ver})` : ''}</div>
                ))}
              </div>
            </div>
          )}

          {/* Toolbar */}
          <div className="flex items-center gap-2 flex-wrap">
            <button onClick={() => setShowTemplates(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 text-blue-300 text-xs font-medium transition-all"><BookOpen className="w-3.5 h-3.5" /> Templates</button>
            <button onClick={() => { setAiInitialGoal(''); setShowAIAssistant(true); }} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 text-violet-300 text-xs font-medium transition-all"><Sparkles className="w-3.5 h-3.5" /> KI-Assistent</button>
            <button onClick={() => { updateConfig(DEFAULT_CONFIG); setLossPoints([]); }} className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 text-xs transition-all"><RefreshCw className="w-3.5 h-3.5" /> Reset</button>
          </div>

          {/* Config Sections */}
          <div className="space-y-3">
            <SectionCard title="Basis-Parameter" icon={<Settings2 className="w-4 h-4 text-emerald-400" />} expanded={sections.basic} onToggle={() => toggleSection('basic')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Epochs" tooltip="Anzahl Trainingsepochen."><NumInput value={config.epochs} onChange={v => updateConfig({ epochs: v })} min={1} max={100} step={1} /></Field>
                <Field label="Batch Size"><NumInput value={config.batch_size} onChange={v => updateConfig({ batch_size: v })} min={1} step={1} /></Field>
                <Field label="Learning Rate" tooltip="1e-5 bis 5e-5 für Encoder."><NumInput value={config.learning_rate} onChange={v => updateConfig({ learning_rate: v })} step={0.000001} /></Field>
                <Field label="Max Seq Length" tooltip="Standard: 128, Max: 512"><NumInput value={config.max_seq_length} onChange={v => updateConfig({ max_seq_length: v })} min={16} max={512} step={16} /></Field>
                <Field label="Warmup Ratio"><NumInput value={config.warmup_ratio} onChange={v => updateConfig({ warmup_ratio: v })} step={0.01} min={0} max={0.3} /></Field>
                <Field label="Gradient Accumulation" tooltip="Effektive Batch-Vergrößerung."><NumInput value={config.gradient_accumulation_steps} onChange={v => updateConfig({ gradient_accumulation_steps: v })} min={1} step={1} /></Field>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-1">
                <Toggle checked={config.fp16} onChange={v => updateConfig({ fp16: v, bf16: v ? false : config.bf16 })} label="FP16 Mixed Precision" />
                <Toggle checked={config.bf16} onChange={v => updateConfig({ bf16: v, fp16: v ? false : config.fp16 })} label="BF16 Mixed Precision" />
              </div>
            </SectionCard>

            <SectionCard title="Optimizer & Scheduler" icon={<Gauge className="w-4 h-4 text-blue-400" />} expanded={sections.optimizer} onToggle={() => toggleSection('optimizer')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Optimizer"><SelectInput value={config.optimizer} onChange={v => updateConfig({ optimizer: v })} options={[{value:'adamw',label:'AdamW'},{value:'adam',label:'Adam'},{value:'sgd',label:'SGD'},{value:'adafactor',label:'Adafactor'}]} /></Field>
                <Field label="Scheduler"><SelectInput value={config.scheduler} onChange={v => updateConfig({ scheduler: v })} options={[{value:'linear',label:'Linear'},{value:'cosine',label:'Cosine'},{value:'constant',label:'Constant'},{value:'polynomial',label:'Polynomial'}]} /></Field>
                <Field label="Weight Decay"><NumInput value={config.weight_decay} onChange={v => updateConfig({ weight_decay: v })} step={0.001} min={0} /></Field>
                <Field label="Max Grad Norm"><NumInput value={config.max_grad_norm} onChange={v => updateConfig({ max_grad_norm: v })} step={0.1} min={0} /></Field>
                <Field label="Adam β1" tooltip="Erster Momentum-Koeffizient (Standard: 0.9)"><NumInput value={config.adam_beta1} onChange={v => updateConfig({ adam_beta1: v })} step={0.001} min={0} max={1} /></Field>
                <Field label="Adam β2" tooltip="Zweiter Momentum-Koeffizient (Standard: 0.999)"><NumInput value={config.adam_beta2} onChange={v => updateConfig({ adam_beta2: v })} step={0.0001} min={0} max={1} /></Field>
                <Field label="Adam ε" tooltip="Numerische Stabilität (Standard: 1e-8)" ><NumInput value={config.adam_epsilon} onChange={v => updateConfig({ adam_epsilon: v })} step={1e-9} min={0} /></Field>
              </div>
            </SectionCard>

            <SectionCard title="Erweitert & Evaluation" icon={<SlidersHorizontal className="w-4 h-4 text-purple-400" />} expanded={sections.advanced} onToggle={() => toggleSection('advanced')}>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Dropout"><NumInput value={config.dropout} onChange={v => updateConfig({ dropout: v })} step={0.01} min={0} max={0.5} /></Field>
                <Field label="Label Smoothing"><NumInput value={config.label_smoothing} onChange={v => updateConfig({ label_smoothing: v })} step={0.01} min={0} max={0.3} /></Field>
                <Field label="Warmup Steps" tooltip="Absolute Schritte (0 = warmup_ratio verwenden)"><NumInput value={config.warmup_steps} onChange={v => updateConfig({ warmup_steps: v })} min={0} step={10} /></Field>
                <Field label="Max Steps" tooltip="-1 = alle Epochen durchlaufen"><NumInput value={config.max_steps} onChange={v => updateConfig({ max_steps: v })} min={-1} step={100} /></Field>
                <Field label="Eval Strategy"><SelectInput value={config.eval_strategy} onChange={v => updateConfig({ eval_strategy: v })} options={[{value:'epoch',label:'Epoch'},{value:'steps',label:'Steps'},{value:'no',label:'Keine'}]} /></Field>
                <Field label="Eval Steps" tooltip="Nur wenn eval_strategy='steps'"><NumInput value={config.eval_steps} onChange={v => updateConfig({ eval_steps: v })} min={1} step={100} /></Field>
                <Field label="Save Steps"><NumInput value={config.save_steps} onChange={v => updateConfig({ save_steps: v })} min={1} step={100} /></Field>
                <Field label="Save Total Limit" tooltip="Max. Checkpoints."><NumInput value={config.save_total_limit} onChange={v => updateConfig({ save_total_limit: v })} min={1} step={1} /></Field>
                <Field label="Logging Steps"><NumInput value={config.logging_steps} onChange={v => updateConfig({ logging_steps: v })} min={1} step={5} /></Field>
                <Field label="Seed"><NumInput value={config.seed} onChange={v => updateConfig({ seed: v })} min={0} step={1} /></Field>
                <Field label="Num Workers" tooltip="DataLoader Worker-Threads (0–8)"><NumInput value={config.num_workers} onChange={v => updateConfig({ num_workers: v })} min={0} max={8} step={1} /></Field>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-2">
                <Toggle checked={config.gradient_checkpointing} onChange={v => updateConfig({ gradient_checkpointing: v })} label="Gradient Checkpointing" />
                <Toggle checked={config.group_by_length} onChange={v => updateConfig({ group_by_length: v })} label="Group by Length" />
                <Toggle checked={config.pin_memory} onChange={v => updateConfig({ pin_memory: v })} label="Pin Memory (GPU)" />
              </div>
            </SectionCard>

            <SectionCard
              title="LoRA / QLoRA"
              icon={<span className="text-violet-400 text-sm font-bold w-4 h-4 flex items-center justify-center">L</span>}
              expanded={sections.lora}
              onToggle={() => toggleSection('lora')}
              badge={config.use_lora ? <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded-md bg-violet-500/20 text-violet-300 border border-violet-500/30">Aktiv</span> : undefined}
            >
              <div className="space-y-3">
                <div className="grid grid-cols-1 gap-3 pb-1">
                  <Toggle checked={config.use_lora} onChange={v => updateConfig({ use_lora: v, load_in_4bit: v ? config.load_in_4bit : false, load_in_8bit: v ? config.load_in_8bit : false })} label="LoRA aktivieren" />
                  <Toggle checked={config.load_in_4bit} onChange={v => updateConfig({ load_in_4bit: v, load_in_8bit: v ? false : config.load_in_8bit, use_lora: v ? true : config.use_lora })} label="4-bit QLoRA (braucht bitsandbytes)" />
                  <Toggle checked={config.load_in_8bit} onChange={v => updateConfig({ load_in_8bit: v, load_in_4bit: v ? false : config.load_in_4bit, use_lora: v ? true : config.use_lora })} label="8-bit Quantisierung (braucht bitsandbytes)" />
                </div>
                {config.use_lora ? (
                  <div className="grid grid-cols-2 gap-4 pt-1 border-t border-white/10">
                    <Field label="LoRA Rank (r)" tooltip="Höher = mehr Parameter, mehr RAM. Typisch: 8, 16, 32"><NumInput value={config.lora_r} onChange={v => updateConfig({ lora_r: v })} min={1} max={256} step={2} /></Field>
                    <Field label="LoRA Alpha" tooltip="Meist 2× lora_r. Skaliert die LoRA-Updates"><NumInput value={config.lora_alpha} onChange={v => updateConfig({ lora_alpha: v })} min={1} step={1} /></Field>
                    <Field label="LoRA Dropout" tooltip="Dropout in LoRA-Layern (0.0–0.1)"><NumInput value={config.lora_dropout} onChange={v => updateConfig({ lora_dropout: v })} step={0.01} min={0} max={0.5} /></Field>
                    <Field label="Target Modules" tooltip="Komma-getrennt, z.B. q_proj,v_proj">
                      <input
                        type="text"
                        value={config.lora_target_modules}
                        onChange={e => updateConfig({ lora_target_modules: e.target.value })}
                        placeholder="q_proj,v_proj"
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-violet-500/50 transition-all font-mono"
                      />
                    </Field>
                  </div>
                ) : (
                  <div className="px-3 py-2.5 rounded-xl bg-violet-500/[0.08] border border-violet-500/15">
                    <p className="text-violet-300 text-xs">💡 <span className="font-medium">LoRA</span> aktivieren um massiv RAM zu sparen — ideal für Modelle &gt;1B Parameter. QLoRA (4-bit) reduziert den Bedarf um ~75%.</p>
                  </div>
                )}
              </div>
            </SectionCard>

            <SectionCard title="RAM-Rechner" icon={<MemoryStick className="w-4 h-4 text-amber-400" />} expanded={sections.ram} onToggle={() => toggleSection('ram')}>
              <RamCalculator config={config} modelSizeGb={modelSizeGb} />
            </SectionCard>
          </div>

          {/* Progress (inline, für wenn Dashboard minimiert oder geschlossen) */}
          {currentJob && !trainingState.showDashboard && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {isRunning ? <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" /> : currentJob.status === 'completed' ? <CheckCircle className="w-4 h-4 text-emerald-400" /> : <AlertCircle className="w-4 h-4 text-red-400" />}
                  <span className="text-white font-medium text-sm">{isRunning ? 'Training läuft…' : `Status: ${currentJob.status}`}</span>
                </div>
                <button onClick={() => { setShowDashboardContext(true); setIsDashMinimizedContext(false); }} className="text-xs text-gray-500 hover:text-white px-2 py-1 rounded-lg bg-white/5 transition-all">Dashboard öffnen</button>
              </div>
              {progress && <div className="h-2 rounded-full bg-white/10 overflow-hidden"><div className={`h-full rounded-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all`} style={{ width: `${progress.progress_percent}%` }} /></div>}
              {lossPoints.length > 1 && <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3"><p className="text-xs text-gray-500 mb-2">Loss-Verlauf</p><LossChart points={lossPoints} /></div>}
            </div>
          )}

          {/* Start / Stop */}
          <div className="flex gap-3">
            {isRunning ? (
              <button onClick={handleStopTraining} className="flex-1 flex items-center justify-center gap-2 py-3.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-sm transition-all">
                <Square className="w-4 h-4" /> Training stoppen
              </button>
            ) : (
              <button onClick={handleStartTraining} disabled={!selectedModelId || !selectedDatasetId || selectedDataset?.status !== 'split'}
                className={`flex-1 flex items-center justify-center gap-2 py-3.5 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg`}>
                <Play className="w-4 h-4" /> Training starten
              </button>
            )}
          </div>
          {selectedDataset?.status !== 'split' && selectedDataset && (
            <p className="text-amber-400 text-xs text-center">⚠️ Dataset muss erst im Dataset-Manager aufgeteilt werden.</p>
          )}
        </>
      )}

      {/* Nicht unterstützt */}
      {mode === 'train' && !isSupported && selectedModel && (
        <div className="p-6 rounded-2xl border border-amber-500/30 bg-amber-500/5 space-y-3">
          <div className="flex items-start gap-3"><AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" /><div><p className="text-amber-300 font-semibold">Modell wird noch nicht unterstützt</p><p className="text-gray-400 text-sm mt-1">Nutze den <span className="text-blue-300 font-medium">Dev Train Mode</span>.</p></div></div>
          <button onClick={() => setMode('dev')} className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-sm font-medium transition-all"><Code2 className="w-4 h-4" /> Dev Train Mode öffnen</button>
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
