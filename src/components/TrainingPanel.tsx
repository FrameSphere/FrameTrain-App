import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import {
  Play,
  Square,
  Settings2,
  RefreshCw,
  Loader2,
  ChevronDown,
  ChevronRight,
  Layers,
  Database,
  Cpu,
  Gauge,
  TrendingDown,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Info,
  Sparkles,
  History,
  Trash2,
  Download,
  X,
  HelpCircle,
  Star,
  ThumbsUp,
  ThumbsDown,
  BarChart3,
  FileUp,
  GitBranch,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelInfo {
  id: string;
  name: string;
  source: string;
}

interface ModelWithVersionTree {
  id: string;
  name: string;
  versions: VersionTreeItem[];
}

interface VersionTreeItem {
  id: string;
  name: string;
  is_root: boolean;
  version_number: number;
}

interface DatasetInfo {
  id: string;
  name: string;
  model_id: string;
  status: 'unused' | 'split';
  file_count: number;
  size_bytes: number;
}

interface TrainingConfig {
  model_path: string;
  dataset_path: string;
  output_path: string;
  checkpoint_dir: string;
  epochs: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  max_steps: number;
  learning_rate: number;
  weight_decay: number;
  warmup_steps: number;
  warmup_ratio: number;
  optimizer: string;
  adam_beta1: number;
  adam_beta2: number;
  adam_epsilon: number;
  sgd_momentum: number;
  scheduler: string;
  scheduler_step_size: number;
  scheduler_gamma: number;
  cosine_min_lr: number;
  dropout: number;
  max_grad_norm: number;
  label_smoothing: number;
  fp16: boolean;
  bf16: boolean;
  use_lora: boolean;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_target_modules: string[];
  load_in_8bit: boolean;
  load_in_4bit: boolean;
  max_seq_length: number;
  num_workers: number;
  pin_memory: boolean;
  eval_steps: number;
  eval_strategy: string;
  save_steps: number;
  save_strategy: string;
  save_total_limit: number;
  logging_steps: number;
  seed: number;
  dataloader_drop_last: boolean;
  group_by_length: boolean;
  training_type: string;
  task_type: string;
}

interface PresetConfig {
  id: string;
  name: string;
  description: string;
  config: TrainingConfig;
}

interface RatingInfo {
  score: number;
  label: string;
  color: string;
}

interface ParameterRating {
  score: number;
  rating: string;
  rating_info: RatingInfo;
  issues: string[];
  warnings: string[];
  tips: string[];
}

interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  train_loss: number;
  val_loss: number | null;
  learning_rate: number;
  progress_percent: number;
  metrics: Record<string, number>;
}

interface TrainingJob {
  id: string;
  model_id: string;
  model_name: string;
  dataset_id: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  config: TrainingConfig;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: TrainingProgress;
  output_path: string | null;
  error: string | null;
}

interface RequirementsCheck {
  python_installed: boolean;
  python_version: string;
  torch_installed: boolean;
  torch_version: string;
  cuda_available: boolean;
  mps_available: boolean;
  transformers_installed: boolean;
  transformers_version: string;
  peft_installed: boolean;
  peft_version: string;
  ready: boolean;
}

// ============ Default Config ============

const defaultConfig: TrainingConfig = {
  model_path: '',
  dataset_path: '',
  output_path: '',
  checkpoint_dir: '',
  epochs: 3,
  batch_size: 8,
  gradient_accumulation_steps: 1,
  max_steps: -1,
  learning_rate: 5e-5,
  weight_decay: 0.01,
  warmup_steps: 0,
  warmup_ratio: 0.0,
  optimizer: 'adamw',
  adam_beta1: 0.9,
  adam_beta2: 0.999,
  adam_epsilon: 1e-8,
  sgd_momentum: 0.9,
  scheduler: 'linear',
  scheduler_step_size: 1,
  scheduler_gamma: 0.1,
  cosine_min_lr: 0.0,
  dropout: 0.1,
  max_grad_norm: 1.0,
  label_smoothing: 0.0,
  fp16: false,
  bf16: false,
  use_lora: false,
  lora_r: 8,
  lora_alpha: 32,
  lora_dropout: 0.1,
  lora_target_modules: ['q_proj', 'v_proj'],
  load_in_8bit: false,
  load_in_4bit: false,
  max_seq_length: 512,
  num_workers: 4,
  pin_memory: true,
  eval_steps: 500,
  eval_strategy: 'steps',
  save_steps: 500,
  save_strategy: 'steps',
  save_total_limit: 3,
  logging_steps: 100,
  seed: 42,
  dataloader_drop_last: false,
  group_by_length: false,
  training_type: 'fine_tuning',
  task_type: 'causal_lm',
};

// ============ Helper Functions ============

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

function formatLearningRate(lr: number): string {
  if (lr >= 0.01) return lr.toFixed(3);
  return lr.toExponential(1);
}

// ============ Sub-Components ============

interface ConfigSectionProps {
  title: string;
  icon: React.ReactNode;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function ConfigSection({ title, icon, expanded, onToggle, children }: ConfigSectionProps) {
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-all"
      >
        <div className="flex items-center gap-3">
          {icon}
          <span className="font-medium text-white">{title}</span>
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>
      {expanded && <div className="px-4 pb-4 space-y-4">{children}</div>}
    </div>
  );
}

interface InputFieldProps {
  label: string;
  value: number | string;
  onChange: (value: number | string) => void;
  type?: 'number' | 'text' | 'select';
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  primaryColor: string;
}

function InputField({
  label,
  value,
  onChange,
  type = 'number',
  options,
  min,
  max,
  step,
  tooltip,
  primaryColor,
}: InputFieldProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-400">{label}</label>
        {tooltip && (
          <div className="group relative">
            <HelpCircle className="w-3.5 h-3.5 text-gray-500 cursor-help" />
            <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-slate-800 rounded-lg text-xs text-gray-300 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
              {tooltip}
            </div>
          </div>
        )}
      </div>
      {type === 'select' && options ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 transition-all"
          style={{ '--tw-ring-color': primaryColor } as React.CSSProperties}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} className="bg-slate-800">
              {opt.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) =>
            onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)
          }
          min={min}
          max={max}
          step={step}
          className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 transition-all"
          style={{ '--tw-ring-color': primaryColor } as React.CSSProperties}
        />
      )}
    </div>
  );
}

interface ToggleFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
  primaryColor: string;
}

function ToggleField({ label, checked, onChange, tooltip, primaryColor }: ToggleFieldProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-400">{label}</span>
        {tooltip && (
          <div className="group relative">
            <HelpCircle className="w-3.5 h-3.5 text-gray-500 cursor-help" />
            <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-slate-800 rounded-lg text-xs text-gray-300 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10">
              {tooltip}
            </div>
          </div>
        )}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`w-11 h-6 rounded-full transition-all ${
          checked ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'bg-white/10'
        }`}
        style={checked ? { background: `linear-gradient(to right, ${primaryColor}, ${primaryColor}dd)` } : {}}
      >
        <div
          className={`w-5 h-5 rounded-full bg-white shadow-lg transform transition-transform ${
            checked ? 'translate-x-5' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  );
}

// ============ Rating Modal ============

interface RatingModalProps {
  rating: ParameterRating;
  onClose: () => void;
  primaryColor: string;
  gradient: string;
}

function RatingModal({ rating, onClose, primaryColor, gradient }: RatingModalProps) {
  const colorMap: Record<string, string> = {
    green: 'text-green-400 bg-green-500/20 border-green-500/50',
    blue: 'text-blue-400 bg-blue-500/20 border-blue-500/50',
    yellow: 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50',
    orange: 'text-orange-400 bg-orange-500/20 border-orange-500/50',
    red: 'text-red-400 bg-red-500/20 border-red-500/50',
  };

  const ratingColors = colorMap[rating.rating_info.color] || colorMap.blue;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Gauge className="w-6 h-6 text-purple-400" />
            <h2 className="text-xl font-bold text-white">Parameter-Bewertung</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto max-h-[60vh]">
          {/* Score Display */}
          <div className="text-center">
            <div
              className={`inline-flex flex-col items-center p-6 rounded-2xl border ${ratingColors}`}
            >
              <div className="text-5xl font-bold">{rating.score}</div>
              <div className="text-lg font-medium mt-1">{rating.rating_info.label}</div>
            </div>
          </div>

          {/* Star Rating */}
          <div className="flex justify-center gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <Star
                key={star}
                className={`w-6 h-6 ${
                  star <= rating.rating_info.score
                    ? 'text-yellow-400 fill-yellow-400'
                    : 'text-gray-600'
                }`}
              />
            ))}
          </div>

          {/* Issues */}
          {rating.issues.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-red-400 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Probleme
              </h3>
              <div className="space-y-2">
                {rating.issues.map((issue, i) => (
                  <div
                    key={i}
                    className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-300"
                  >
                    {issue}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          {rating.warnings.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-amber-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                Warnungen
              </h3>
              <div className="space-y-2">
                {rating.warnings.map((warning, i) => (
                  <div
                    key={i}
                    className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-sm text-amber-300"
                  >
                    {warning}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tips */}
          {rating.tips.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-blue-400 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Tipps
              </h3>
              <div className="space-y-2">
                {rating.tips.map((tip, i) => (
                  <div
                    key={i}
                    className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg text-sm text-blue-300"
                  >
                    {tip}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-white/10">
          <button
            onClick={onClose}
            className={`w-full py-3 bg-gradient-to-r ${gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            Verstanden
          </button>
        </div>
      </div>
    </div>
  );
}

// ============ Requirements Modal ============

interface RequirementsModalProps {
  requirements: RequirementsCheck;
  onClose: () => void;
  onRefresh: () => void;
  gradient: string;
}

function RequirementsModal({ requirements, onClose, onRefresh, gradient }: RequirementsModalProps) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg">
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <h2 className="text-xl font-bold text-white">System-Anforderungen</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Python */}
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-3">
              {requirements.python_installed ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400" />
              )}
              <div>
                <div className="text-white font-medium">Python</div>
                <div className="text-xs text-gray-400">{requirements.python_version}</div>
              </div>
            </div>
          </div>

          {/* PyTorch */}
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-3">
              {requirements.torch_installed ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400" />
              )}
              <div>
                <div className="text-white font-medium">PyTorch</div>
                <div className="text-xs text-gray-400">{requirements.torch_version}</div>
              </div>
            </div>
          </div>

          {/* GPU */}
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-3">
              {requirements.cuda_available || requirements.mps_available ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              )}
              <div>
                <div className="text-white font-medium">GPU-Beschleunigung</div>
                <div className="text-xs text-gray-400">
                  {requirements.cuda_available
                    ? 'NVIDIA CUDA verfügbar'
                    : requirements.mps_available
                    ? 'Apple Silicon MPS verfügbar'
                    : 'Nur CPU (langsamer)'}
                </div>
              </div>
            </div>
          </div>

          {/* Transformers */}
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-3">
              {requirements.transformers_installed ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              )}
              <div>
                <div className="text-white font-medium">Transformers</div>
                <div className="text-xs text-gray-400">{requirements.transformers_version}</div>
              </div>
            </div>
          </div>

          {/* PEFT */}
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-3">
              {requirements.peft_installed ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              )}
              <div>
                <div className="text-white font-medium">PEFT (für LoRA)</div>
                <div className="text-xs text-gray-400">{requirements.peft_version}</div>
              </div>
            </div>
          </div>

          {/* Overall Status */}
          <div
            className={`p-4 rounded-lg border ${
              requirements.ready
                ? 'bg-green-500/10 border-green-500/30'
                : 'bg-red-500/10 border-red-500/30'
            }`}
          >
            <div className="flex items-center gap-3">
              {requirements.ready ? (
                <>
                  <CheckCircle className="w-6 h-6 text-green-400" />
                  <div>
                    <div className="text-green-400 font-medium">Bereit für Training!</div>
                    <div className="text-xs text-green-300/70">
                      Alle erforderlichen Komponenten sind installiert.
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <AlertCircle className="w-6 h-6 text-red-400" />
                  <div>
                    <div className="text-red-400 font-medium">Nicht bereit</div>
                    <div className="text-xs text-red-300/70">
                      Bitte installiere die fehlenden Komponenten.
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {!requirements.ready && (
            <div className="text-sm text-gray-400">
              <p className="mb-2">Installation:</p>
              <code className="block p-2 bg-black/30 rounded text-xs">
                pip install torch transformers peft bitsandbytes
              </code>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-white/10 flex gap-3">
          <button
            onClick={onRefresh}
            className="flex-1 flex items-center justify-center gap-2 py-3 bg-white/5 hover:bg-white/10 rounded-lg text-white transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            Erneut prüfen
          </button>
          <button
            onClick={onClose}
            className={`flex-1 py-3 bg-gradient-to-r ${gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            Schließen
          </button>
        </div>
      </div>
    </div>
  );
}

// ============ Training History Modal ============

interface HistoryModalProps {
  jobs: TrainingJob[];
  onClose: () => void;
  onDelete: (jobId: string) => void;
  gradient: string;
}

function HistoryModal({ jobs, onClose, onDelete, gradient }: HistoryModalProps) {
  const statusColors: Record<string, string> = {
    pending: 'bg-gray-500/20 text-gray-400',
    running: 'bg-blue-500/20 text-blue-400',
    completed: 'bg-green-500/20 text-green-400',
    failed: 'bg-red-500/20 text-red-400',
    stopped: 'bg-amber-500/20 text-amber-400',
  };

  const statusLabels: Record<string, string> = {
    pending: 'Wartend',
    running: 'Läuft',
    completed: 'Abgeschlossen',
    failed: 'Fehlgeschlagen',
    stopped: 'Gestoppt',
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <History className="w-6 h-6 text-purple-400" />
            <h2 className="text-xl font-bold text-white">Training-Verlauf</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {jobs.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Noch keine Trainings durchgeführt.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className="p-4 bg-white/5 rounded-xl border border-white/10 hover:bg-white/[0.07] transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-white">{job.model_name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${statusColors[job.status]}`}>
                          {statusLabels[job.status]}
                        </span>
                      </div>
                      <div className="text-sm text-gray-400 mt-1">
                        Dataset: {job.dataset_name}
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        {formatDate(job.created_at)}
                      </div>
                    </div>
                    <button
                      onClick={() => onDelete(job.id)}
                      className="p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {job.status === 'completed' && (
                    <div className="mt-3 pt-3 border-t border-white/10 grid grid-cols-3 gap-3 text-xs">
                      <div className="text-center">
                        <div className="text-gray-400">Train Loss</div>
                        <div className="text-white font-medium">
                          {job.progress.train_loss.toFixed(4)}
                        </div>
                      </div>
                     <div className="text-center">
                        <div className="text-gray-400">Val Loss</div>
                        <div className="text-white font-medium">
                          {job.progress.val_loss?.toFixed(4) || '-'}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400">Epochen</div>
                        <div className="text-white font-medium">
                          {job.progress.epoch}/{job.progress.total_epochs}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="p-6 border-t border-white/10">
          <button
            onClick={onClose}
            className={`w-full py-3 bg-gradient-to-r ${gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
          >
            Schließen
          </button>
        </div>
      </div>
    </div>
  );
}

// ============ Loss Chart Component ============

interface LossChartProps {
  history: { epoch: number; train_loss: number; val_loss: number | null }[];
  primaryColor: string;
}

function LossChart({ history, primaryColor }: LossChartProps) {
  if (history.length === 0) return null;

  const maxLoss = Math.max(
    ...history.map((h) => Math.max(h.train_loss, h.val_loss || 0))
  );
  const minLoss = Math.min(
    ...history.map((h) => Math.min(h.train_loss, h.val_loss || Infinity))
  );
  const range = maxLoss - minLoss || 1;

  const getY = (loss: number) => {
    return 100 - ((loss - minLoss) / range) * 80 - 10;
  };

  const trainPoints = history
    .map((h, i) => `${(i / (history.length - 1 || 1)) * 100},${getY(h.train_loss)}`)
    .join(' ');

  const valPoints = history
    .filter((h) => h.val_loss !== null)
    .map((h, i, arr) => `${(i / (arr.length - 1 || 1)) * 100},${getY(h.val_loss!)}`)
    .join(' ');

  return (
    <div className="bg-white/5 rounded-xl p-4 border border-white/10">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-300">Loss-Verlauf</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-blue-400" />
            <span className="text-gray-400">Train</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-purple-400" />
            <span className="text-gray-400">Val</span>
          </div>
        </div>
      </div>
      <svg viewBox="0 0 100 100" className="w-full h-32" preserveAspectRatio="none">
        {/* Grid lines */}
        <line x1="0" y1="10" x2="100" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
        <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
        <line x1="0" y1="90" x2="100" y2="90" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />

        {/* Train loss line */}
        <polyline
          points={trainPoints}
          fill="none"
          stroke="#60a5fa"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Val loss line */}
        {valPoints && (
          <polyline
            points={valPoints}
            fill="none"
            stroke="#a855f7"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>Log Step 1</span>
        <span>Log Step {history.length}</span>
      </div>
    </div>
  );
}

// ============ Main Component ============

export default function TrainingPanel() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();

  // Data State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [presets, setPresets] = useState<PresetConfig[]>([]);
  const [loading, setLoading] = useState(true);

  // Selection State
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [showVersions, setShowVersions] = useState(false);

  // Config State
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);

  // UI State
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    basic: true,
    optimizer: false,
    scheduler: false,
    lora: false,
    advanced: false,
  });

  // Rating State
  const [rating, setRating] = useState<ParameterRating | null>(null);
  const [showRatingModal, setShowRatingModal] = useState(false);
  const [ratingLoading, setRatingLoading] = useState(false);

  // Requirements State
  const [requirements, setRequirements] = useState<RequirementsCheck | null>(null);
  const [showRequirementsModal, setShowRequirementsModal] = useState(false);
  const [checkingRequirements, setCheckingRequirements] = useState(false);

  // Training State
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [lossHistory, setLossHistory] = useState<{ epoch: number; train_loss: number; val_loss: number | null }[]>([]);
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null);
  const [trainingElapsed, setTrainingElapsed] = useState<number>(0);

  // History State
  const [trainingHistory, setTrainingHistory] = useState<TrainingJob[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);

  // JSON Upload State
  const [uploadingConfig, setUploadingConfig] = useState(false);
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [validationIssues, setValidationIssues] = useState<{field: string, value: any, defaultValue: any, reason: string}[]>([]);

  // ============ Load Data ============
  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      loadDatasets();
      
      // Update version selection when model changes
      const modelWithVersions = modelsWithVersions.find(m => m.id === selectedModelId);

      if (modelWithVersions && modelWithVersions.versions.length > 0) {

        // ✨ Neue Logik: Neueste Version automatisch auswählen
        const sortedVersions = [...modelWithVersions.versions].sort(
          (a, b) => b.version_number - a.version_number
        );
        const newestVersion = sortedVersions[0];

        setSelectedVersionId(newestVersion?.id || null);
        setShowVersions(true);

      } else {
        setSelectedVersionId(null);
        setShowVersions(false);
      }

    } else {
      setDatasets([]);
      setSelectedDatasetId(null);
      setSelectedVersionId(null);
      setShowVersions(false);
    }
  }, [selectedModelId, modelsWithVersions]);

  const selectedModel = models.find((m) => m.id === selectedModelId);
  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);
  const isTraining = currentJob?.status === 'running' || currentJob?.status === 'pending';

  // Rate config when it changes
  useEffect(() => {
    const timer = setTimeout(() => {
      rateConfig();
    }, 500);
    return () => clearTimeout(timer);
  }, [config]);

  // Timer for training elapsed time
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    
    if (trainingStartTime && isTraining) {
      interval = setInterval(() => {
        setTrainingElapsed(Date.now() - trainingStartTime);
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [trainingStartTime, isTraining]);

  // Listen to training events
  useEffect(() => {
    const unlisteners: (() => void)[] = [];

    const setupListeners = async () => {
      unlisteners.push(
        await listen<any>('training-started', (event) => {
          setTrainingStatus('Training gestartet...');
          setTrainingStartTime(Date.now());
          setTrainingElapsed(0);
        })
      );

      unlisteners.push(
        await listen<any>('training-progress', (event) => {
          const data = event.payload.data;
          if (data) {
            setCurrentJob((prev) =>
              prev ? { ...prev, progress: data, status: 'running' } : prev
            );
            // Update loss history every logging step
            setLossHistory((prev) => {
              const newEntry = {
                epoch: data.step / data.total_steps * data.total_epochs,
                train_loss: data.train_loss,
                val_loss: data.val_loss,
              };
              // Check if entry already exists for this step
              const existingIndex = prev.findIndex((h) => Math.abs(h.epoch - newEntry.epoch) < 0.001);
              if (existingIndex >= 0) {
                const updated = [...prev];
                updated[existingIndex] = newEntry;
                return updated;
              }
              return [...prev, newEntry];
            });
          }
        })
      );

      unlisteners.push(
        await listen<any>('training-status', (event) => {
          const data = event.payload.data;
          if (data?.message) {
            setTrainingStatus(data.message);
          }
        })
      );

      unlisteners.push(
        await listen<any>('training-complete', (event) => {
          success('Training abgeschlossen!', 'Das Modell wurde erfolgreich trainiert.');
          setCurrentJob((prev) =>
            prev ? { ...prev, status: 'completed' } : prev
          );
          setTrainingStatus('Training abgeschlossen');
          setTrainingStartTime(null);
          loadTrainingHistory();
        })
      );

      unlisteners.push(
        await listen<any>('training-error', (event) => {
          const data = event.payload.data || event.payload;
          error('Training-Fehler', data?.error || 'Unbekannter Fehler');
          setCurrentJob(null);
          setTrainingStatus('');
          setTrainingStartTime(null);
          setTrainingElapsed(0);
          loadTrainingHistory();
        })
      );

      unlisteners.push(
        await listen<any>('training-finished', (event) => {
          if (!event.payload.success) {
            setCurrentJob(null);
            setTrainingStartTime(null);
            setTrainingElapsed(0);
          }
          setTrainingStatus('');
        })
      );
    };

    setupListeners();

    return () => {
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);

      // Load models
      const modelList = await invoke<ModelInfo[]>('list_models');
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModelId(modelList[0].id);
      }

      // Load models with versions for version selection
      const modelsWithVersionsList = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree');
      setModelsWithVersions(modelsWithVersionsList);
      if (modelsWithVersionsList.length > 0 && modelsWithVersionsList[0].versions.length > 0) {
        // Select root version by default
        const rootVersion = modelsWithVersionsList[0].versions.find(v => v.is_root);
        if (rootVersion) {
          setSelectedVersionId(rootVersion.id);
        } else if (modelsWithVersionsList[0].versions.length > 0) {
          setSelectedVersionId(modelsWithVersionsList[0].versions[0].id);
        }
      }

      // Load presets
      const presetList = await invoke<PresetConfig[]>('get_training_presets');
      setPresets(presetList);

      // Load training history
      await loadTrainingHistory();

      // Check current training
      const current = await invoke<TrainingJob | null>('get_current_training');
      if (current) {
        setCurrentJob(current);
      }

      // Check requirements
      await checkRequirements();
    } catch (err: any) {
      console.error('Error loading data:', err);
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    if (!selectedModelId) return;

    try {
      const datasetList = await invoke<DatasetInfo[]>('list_datasets_for_model', {
        modelId: selectedModelId,
      });
      // Only show split datasets
      const splitDatasets = datasetList.filter((d) => d.status === 'split');
      setDatasets(splitDatasets);
      if (splitDatasets.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(splitDatasets[0].id);
      }
    } catch (err: any) {
      console.error('Error loading datasets:', err);
    }
  };

  const loadTrainingHistory = async () => {
    try {
      const history = await invoke<TrainingJob[]>('get_training_history');
      setTrainingHistory(history);
    } catch (err: any) {
      console.error('Error loading history:', err);
    }
  };

  const checkRequirements = async () => {
    setCheckingRequirements(true);
    try {
      const reqs = await invoke<RequirementsCheck>('check_training_requirements');
      setRequirements(reqs);
    } catch (err: any) {
      console.error('Error checking requirements:', err);
    } finally {
      setCheckingRequirements(false);
    }
  };

  const rateConfig = async () => {
    setRatingLoading(true);
    try {
      const result = await invoke<ParameterRating>('rate_training_config', { config });
      setRating(result);
    } catch (err: any) {
      console.error('Error rating config:', err);
    } finally {
      setRatingLoading(false);
    }
  };

  // ============ Actions ============

  const applyPreset = (presetId: string) => {
    const preset = presets.find((p) => p.id === presetId);
    if (preset) {
      setConfig((prev) => ({ ...prev, ...preset.config }));
      setSelectedPresetId(presetId);
      info('Preset angewendet', `"${preset.name}" wurde geladen.`);
    }
  };

  const validateConfigValue = (key: string, value: any, defaultVal: any): {valid: boolean, correctedValue: any, reason?: string} => {
    // Number validation
    if (typeof defaultVal === 'number') {
      const parsed = typeof value === 'string' ? parseFloat(value) : value;
      if (isNaN(parsed) || typeof parsed !== 'number') {
        return {valid: false, correctedValue: defaultVal, reason: 'Ungültiger Zahlenwert'};
      }
      return {valid: true, correctedValue: parsed};
    }
    
    // Boolean validation
    if (typeof defaultVal === 'boolean') {
      if (typeof value !== 'boolean') {
        return {valid: false, correctedValue: defaultVal, reason: 'Muss true oder false sein'};
      }
      return {valid: true, correctedValue: value};
    }
    
    // String validation
    if (typeof defaultVal === 'string') {
      if (typeof value !== 'string') {
        return {valid: false, correctedValue: defaultVal, reason: 'Muss ein Text sein'};
      }
      return {valid: true, correctedValue: value};
    }
    
    // Array validation
    if (Array.isArray(defaultVal)) {
      if (!Array.isArray(value)) {
        return {valid: false, correctedValue: defaultVal, reason: 'Muss ein Array sein'};
      }
      return {valid: true, correctedValue: value};
    }
    
    return {valid: true, correctedValue: value};
  };

  const handleUploadConfig = async () => {
    try {
      setUploadingConfig(true);
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        filters: [{ name: 'JSON', extensions: ['json'] }],
        title: 'Trainingskonfiguration auswählen'
      });
      
      if (selected && typeof selected === 'string') {
        const { readTextFile } = await import('@tauri-apps/plugin-fs');
        const content = await readTextFile(selected);
        const uploadedConfig = JSON.parse(content);
        
        // Validate and correct values
        const issues: {field: string, value: any, defaultValue: any, reason: string}[] = [];
        const correctedConfig: any = {};
        
        Object.keys(uploadedConfig).forEach(key => {
          const defaultValue = (defaultConfig as any)[key];
          if (defaultValue !== undefined) {
            const validation = validateConfigValue(key, uploadedConfig[key], defaultValue);
            if (!validation.valid) {
              issues.push({
                field: key,
                value: uploadedConfig[key],
                defaultValue: validation.correctedValue,
                reason: validation.reason || 'Ungültiger Wert'
              });
            }
            correctedConfig[key] = validation.correctedValue;
          }
        });
        
        if (issues.length > 0) {
          // Show validation modal
          setValidationIssues(issues);
          // Store corrected config temporarily
          (window as any).pendingConfig = correctedConfig;
          setShowValidationModal(true);
        } else {
          // Apply directly if no issues
          setConfig((prev) => ({
            ...prev,
            ...correctedConfig,
            model_path: prev.model_path,
            dataset_path: prev.dataset_path,
            output_path: prev.output_path,
            checkpoint_dir: prev.checkpoint_dir
          }));
          
          setSelectedPresetId(null);
          success('Konfiguration geladen', 'Die Parameter wurden aus der JSON-Datei übernommen.');
        }
      }
    } catch (err: any) {
      error('Fehler beim Laden', String(err));
    } finally {
      setUploadingConfig(false);
    }
  };

  const handleConfirmValidation = () => {
    const correctedConfig = (window as any).pendingConfig;
    if (correctedConfig) {
      setConfig((prev) => ({
        ...prev,
        ...correctedConfig,
        model_path: prev.model_path,
        dataset_path: prev.dataset_path,
        output_path: prev.output_path,
        checkpoint_dir: prev.checkpoint_dir
      }));
      
      setSelectedPresetId(null);
      success('Konfiguration geladen', `${validationIssues.length} Wert(e) wurden korrigiert.`);
      delete (window as any).pendingConfig;
    }
    setShowValidationModal(false);
    setValidationIssues([]);
  };

  const handleDownloadTemplate = () => {
    try {
      // Create a clean template with current config
      const template = {
        epochs: config.epochs,
        batch_size: config.batch_size,
        gradient_accumulation_steps: config.gradient_accumulation_steps,
        max_steps: config.max_steps,
        learning_rate: config.learning_rate,
        weight_decay: config.weight_decay,
        warmup_steps: config.warmup_steps,
        warmup_ratio: config.warmup_ratio,
        optimizer: config.optimizer,
        adam_beta1: config.adam_beta1,
        adam_beta2: config.adam_beta2,
        adam_epsilon: config.adam_epsilon,
        sgd_momentum: config.sgd_momentum,
        scheduler: config.scheduler,
        scheduler_step_size: config.scheduler_step_size,
        scheduler_gamma: config.scheduler_gamma,
        cosine_min_lr: config.cosine_min_lr,
        dropout: config.dropout,
        max_grad_norm: config.max_grad_norm,
        label_smoothing: config.label_smoothing,
        fp16: config.fp16,
        bf16: config.bf16,
        use_lora: config.use_lora,
        lora_r: config.lora_r,
        lora_alpha: config.lora_alpha,
        lora_dropout: config.lora_dropout,
        lora_target_modules: config.lora_target_modules,
        load_in_8bit: config.load_in_8bit,
        load_in_4bit: config.load_in_4bit,
        max_seq_length: config.max_seq_length,
        num_workers: config.num_workers,
        pin_memory: config.pin_memory,
        eval_steps: config.eval_steps,
        eval_strategy: config.eval_strategy,
        save_steps: config.save_steps,
        save_strategy: config.save_strategy,
        save_total_limit: config.save_total_limit,
        logging_steps: config.logging_steps,
        seed: config.seed,
        dataloader_drop_last: config.dataloader_drop_last,
        group_by_length: config.group_by_length,
        training_type: config.training_type,
        task_type: config.task_type
      };
      
      const jsonContent = JSON.stringify(template, null, 2);
      
      // Create blob and download
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'training_config_template.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      success('Vorlage gespeichert', 'Die Datei wurde in deinem Downloads-Ordner gespeichert.');
    } catch (err: any) {
      error('Fehler beim Speichern', String(err));
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const updateConfig = (key: keyof TrainingConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
    setSelectedPresetId(null); // Clear preset when manually editing
  };

  const handleStartTraining = async () => {
    if (!selectedModelId || !selectedDatasetId) {
      warning('Auswahl fehlt', 'Bitte wähle ein Modell und ein Dataset aus.');
      return;
    }

    if (!requirements?.ready) {
      setShowRequirementsModal(true);
      return;
    }

    const selectedModel = models.find((m) => m.id === selectedModelId);
    const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

    if (!selectedModel || !selectedDataset) return;

    try {
      setLossHistory([]);
      const job = await invoke<TrainingJob>('start_training', {
        modelId: selectedModelId,
        modelName: selectedModel.name,
        datasetId: selectedDatasetId,
        datasetName: selectedDataset.name,
        config,
        versionId: selectedVersionId,
      });
      setCurrentJob(job);
      success('Training gestartet!', 'Das Training wurde erfolgreich gestartet.');
    } catch (err: any) {
      error('Start fehlgeschlagen', String(err));
    }
  };

  const handleStopTraining = async () => {
    try {
      await invoke('stop_training');
      setCurrentJob(null);
      setTrainingStartTime(null);
      setTrainingElapsed(0);
      warning('Training gestoppt', 'Das Training wurde abgebrochen.');
    } catch (err: any) {
      error('Stoppen fehlgeschlagen', String(err));
    }
  };

  const handleDeleteHistoryJob = async (jobId: string) => {
    try {
      await invoke('delete_training_job', { jobId });
      await loadTrainingHistory();
      success('Gelöscht', 'Der Eintrag wurde entfernt.');
    } catch (err: any) {
      error('Löschen fehlgeschlagen', String(err));
    }
  };

  // ============ Render ============

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin" />
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Training</h1>
          <p className="text-gray-400 mt-1">Trainiere deine Modelle</p>
        </div>
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white/5 mb-4">
            <Layers className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
          <p className="text-gray-400">
            Füge zuerst ein Modell hinzu, bevor du mit dem Training beginnen kannst.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Training</h1>
          <p className="text-gray-400 mt-1">Trainiere deine Modelle mit PyTorch</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Requirements Button */}
          <button
            onClick={() => setShowRequirementsModal(true)}
            className={`p-2 rounded-lg transition-all ${
              requirements?.ready
                ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
            }`}
            title="System-Anforderungen"
          >
            {checkingRequirements ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : requirements?.ready ? (
              <CheckCircle className="w-5 h-5" />
            ) : (
              <AlertCircle className="w-5 h-5" />
            )}
          </button>

          {/* History Button */}
          <button
            onClick={() => {
              loadTrainingHistory();
              setShowHistoryModal(true);
            }}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Training-Verlauf"
          >
            <History className="w-5 h-5" />
          </button>

          {/* Refresh Button */}
          <button
            onClick={loadInitialData}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Aktualisieren"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model & Dataset Selection */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Modell & Dataset
            </h2>
            <div className="space-y-4">
              {/* Model Selector */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Modell</label>
                <div className="relative">
                  <select
                    value={selectedModelId || ''}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    disabled={isTraining}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                    style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                  >
                    {models.map((model) => (
                      <option key={model.id} value={model.id} className="bg-slate-800">
                        {model.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                </div>
              </div>

              {/* Version Selector */}
              {showVersions && selectedModelId && (() => {
                const modelWithVersions = modelsWithVersions.find(m => m.id === selectedModelId);
                return modelWithVersions && modelWithVersions.versions.length > 0 ? (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2 flex items-center gap-2">
                      <GitBranch className="w-4 h-4" />
                      Modell-Version
                    </label>
                    <div className="relative">
                      <select
                        value={selectedVersionId || ''}
                        onChange={(e) => setSelectedVersionId(e.target.value)}
                        disabled={isTraining}
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                        style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                      >
                        {modelWithVersions.versions.map((version) => (
                          <option key={version.id} value={version.id} className="bg-slate-800">
                            {version.is_root ? '⭐ ' : ''}{version.name}{version.is_root ? ' (Original)' : ` (v${version.version_number})`}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Wähle eine Version des Modells zum Trainieren</p>
                  </div>
                ) : null;
              })()}

              {/* Dataset Selector */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Dataset (aufgeteilt)</label>
                <div className="relative">
                  <select
                    value={selectedDatasetId || ''}
                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                    disabled={isTraining || datasets.length === 0}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white appearance-none cursor-pointer focus:outline-none focus:ring-2 disabled:opacity-50 transition-all"
                    style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                  >
                    {datasets.length === 0 ? (
                      <option value="" className="bg-slate-800">
                        Kein aufgeteiltes Dataset
                      </option>
                    ) : (
                      datasets.map((dataset) => (
                        <option key={dataset.id} value={dataset.id} className="bg-slate-800">
                          {dataset.name} ({dataset.file_count} Dateien)
                        </option>
                      ))
                    )}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                </div>
                {datasets.length === 0 && selectedModelId && (
                  <p className="text-xs text-amber-400 mt-2">
                    Bitte teile zuerst ein Dataset auf der Datensätze-Seite auf.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Presets */}
          <div className="bg-white/5 rounded-xl border border-white/10 p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Voreinstellungen
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {presets.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => applyPreset(preset.id)}
                  disabled={isTraining}
                  className={`p-3 rounded-lg border text-left transition-all disabled:opacity-50 ${
                    selectedPresetId === preset.id
                      ? 'bg-purple-500/20 border-purple-500/50'
                      : 'bg-white/5 border-white/10 hover:bg-white/10'
                  }`}
                >
                  <div className="font-medium text-white text-sm">{preset.name}</div>
                  <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                    {preset.description}
                  </div>
                </button>
              ))}
              
              {/* Upload JSON Button */}
              <button
                onClick={handleUploadConfig}
                disabled={isTraining || uploadingConfig}
                className="p-3 rounded-lg border border-dashed border-white/20 bg-white/5 hover:bg-white/10 hover:border-white/30 text-left transition-all disabled:opacity-50 group"
              >
                <div className="flex items-center gap-2 text-white text-sm font-medium">
                  {uploadingConfig ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <FileUp className="w-4 h-4 group-hover:scale-110 transition-transform" />
                  )}
                  JSON hochladen
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {uploadingConfig ? 'Lade...' : 'Eigene Konfiguration'}
                </div>
              </button>
            </div>
          </div>

          {/* Configuration Sections */}
          <div className="space-y-4">
            {/* Basic Settings */}
            <ConfigSection
              title="Basis-Einstellungen"
              icon={<Settings2 className="w-5 h-5 text-blue-400" />}
              expanded={expandedSections.basic}
              onToggle={() => toggleSection('basic')}
            >
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <InputField
                  label="Epochen"
                  value={config.epochs}
                  onChange={(v) => updateConfig('epochs', v)}
                  min={1}
                  max={100}
                  tooltip="Anzahl der Durchläufe durch das gesamte Dataset"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Batch Size"
                  value={config.batch_size}
                  onChange={(v) => updateConfig('batch_size', v)}
                  min={1}
                  max={128}
                  tooltip="Anzahl der Samples pro Trainingsschritt"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Learning Rate"
                  value={config.learning_rate}
                  onChange={(v) => updateConfig('learning_rate', v)}
                  step={0.00001}
                  tooltip="Lernrate - wie stark die Gewichte angepasst werden"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Gradient Accumulation"
                  value={config.gradient_accumulation_steps}
                  onChange={(v) => updateConfig('gradient_accumulation_steps', v)}
                  min={1}
                  max={32}
                  tooltip="Schritte für Gradienten-Akkumulation (effektiv größere Batch Size)"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Max Seq Length"
                  value={config.max_seq_length}
                  onChange={(v) => updateConfig('max_seq_length', v)}
                  min={32}
                  max={4096}
                  tooltip="Maximale Sequenzlänge für Tokenisierung"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Seed"
                  value={config.seed}
                  onChange={(v) => updateConfig('seed', v)}
                  tooltip="Random Seed für Reproduzierbarkeit"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <ToggleField
                  label="FP16 (Mixed Precision)"
                  checked={config.fp16}
                  onChange={(v) => updateConfig('fp16', v)}
                  tooltip="Verwendet 16-bit Fließkommazahlen für schnelleres Training"
                  primaryColor={currentTheme.colors.primary}
                />
                <ToggleField
                  label="BF16 (Brain Float)"
                  checked={config.bf16}
                  onChange={(v) => updateConfig('bf16', v)}
                  tooltip="Brain Float 16 - besser für neuere GPUs"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
            </ConfigSection>

            {/* Optimizer Settings */}
            <ConfigSection
              title="Optimizer"
              icon={<Zap className="w-5 h-5 text-yellow-400" />}
              expanded={expandedSections.optimizer}
              onToggle={() => toggleSection('optimizer')}
            >
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <InputField
                  label="Optimizer"
                  value={config.optimizer}
                  onChange={(v) => updateConfig('optimizer', v)}
                  type="select"
                  options={[
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'sgd', label: 'SGD' },
                    { value: 'adagrad', label: 'Adagrad' },
                    { value: 'rmsprop', label: 'RMSprop' },
                  ]}
                  tooltip="Optimierungs-Algorithmus"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Weight Decay"
                  value={config.weight_decay}
                  onChange={(v) => updateConfig('weight_decay', v)}
                  step={0.001}
                  tooltip="L2-Regularisierung zur Vermeidung von Overfitting"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Max Grad Norm"
                  value={config.max_grad_norm}
                  onChange={(v) => updateConfig('max_grad_norm', v)}
                  step={0.1}
                  tooltip="Gradient Clipping - verhindert explodierende Gradienten"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
              {config.optimizer.includes('adam') && (
                <div className="grid grid-cols-3 gap-4 mt-4">
                  <InputField
                    label="Beta 1"
                    value={config.adam_beta1}
                    onChange={(v) => updateConfig('adam_beta1', v)}
                    step={0.01}
                    min={0}
                    max={1}
                    tooltip="Exponentieller Zerfall f\u00fcr ersten Moment"
                    primaryColor={currentTheme.colors.primary}
                  />
                  <InputField
                    label="Beta 2"
                    value={config.adam_beta2}
                    onChange={(v) => updateConfig('adam_beta2', v)}
                    step={0.001}
                    min={0}
                    max={1}
                    tooltip="Exponentieller Zerfall f\u00fcr zweiten Moment"
                    primaryColor={currentTheme.colors.primary}
                  />
                  <InputField
                    label="Epsilon"
                    value={config.adam_epsilon}
                    onChange={(v) => updateConfig('adam_epsilon', v)}
                    step={0.0000001}
                    tooltip="Numerische Stabilit\u00e4t"
                    primaryColor={currentTheme.colors.primary}
                  />
                </div>
              )}
              {config.optimizer === 'sgd' && (
                <div className="mt-4">
                  <InputField
                    label="Momentum"
                    value={config.sgd_momentum}
                    onChange={(v) => updateConfig('sgd_momentum', v)}
                    step={0.1}
                    min={0}
                    max={1}
                    tooltip="Momentum-Faktor f\u00fcr SGD"
                    primaryColor={currentTheme.colors.primary}
                  />
                </div>
              )}
            </ConfigSection>

            {/* Scheduler Settings */}
            <ConfigSection
              title="Learning Rate Scheduler"
              icon={<TrendingDown className="w-5 h-5 text-green-400" />}
              expanded={expandedSections.scheduler}
              onToggle={() => toggleSection('scheduler')}
            >
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <InputField
                  label="Scheduler"
                  value={config.scheduler}
                  onChange={(v) => updateConfig('scheduler', v)}
                  type="select"
                  options={[
                    { value: 'linear', label: 'Linear' },
                    { value: 'cosine', label: 'Cosine' },
                    { value: 'constant', label: 'Constant' },
                    { value: 'polynomial', label: 'Polynomial' },
                    { value: 'one_cycle', label: 'One Cycle' },
                    { value: 'step', label: 'Step' },
                    { value: 'exponential', label: 'Exponential' },
                  ]}
                  tooltip="Learning Rate Scheduler-Typ"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Warmup Ratio"
                  value={config.warmup_ratio}
                  onChange={(v) => updateConfig('warmup_ratio', v)}
                  step={0.01}
                  min={0}
                  max={0.5}
                  tooltip="Anteil der Schritte f\u00fcr Warmup (0-0.5)"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Warmup Steps"
                  value={config.warmup_steps}
                  onChange={(v) => updateConfig('warmup_steps', v)}
                  min={0}
                  tooltip="Absolute Anzahl Warmup-Schritte (\u00fcberschreibt Ratio)"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
              {config.scheduler === 'cosine' && (
                <div className="mt-4">
                  <InputField
                    label="Min LR"
                    value={config.cosine_min_lr}
                    onChange={(v) => updateConfig('cosine_min_lr', v)}
                    step={0.000001}
                    tooltip="Minimale Learning Rate am Ende"
                    primaryColor={currentTheme.colors.primary}
                  />
                </div>
              )}
              {(config.scheduler === 'step' || config.scheduler === 'exponential') && (
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <InputField
                    label="Step Size"
                    value={config.scheduler_step_size}
                    onChange={(v) => updateConfig('scheduler_step_size', v)}
                    min={1}
                    tooltip="Schritte zwischen LR-Anpassungen"
                    primaryColor={currentTheme.colors.primary}
                  />
                  <InputField
                    label="Gamma"
                    value={config.scheduler_gamma}
                    onChange={(v) => updateConfig('scheduler_gamma', v)}
                    step={0.01}
                    min={0}
                    max={1}
                    tooltip="Multiplikationsfaktor f\u00fcr LR-Reduktion"
                    primaryColor={currentTheme.colors.primary}
                  />
                </div>
              )}
            </ConfigSection>

            {/* LoRA Settings */}
            <ConfigSection
              title="LoRA / QLoRA"
              icon={<Cpu className="w-5 h-5 text-purple-400" />}
              expanded={expandedSections.lora}
              onToggle={() => toggleSection('lora')}
            >
              <ToggleField
                label="LoRA aktivieren"
                checked={config.use_lora}
                onChange={(v) => updateConfig('use_lora', v)}
                tooltip="Low-Rank Adaptation f\u00fcr effizientes Fine-Tuning"
                primaryColor={currentTheme.colors.primary}
              />
              {config.use_lora && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                    <InputField
                      label="LoRA Rank (r)"
                      value={config.lora_r}
                      onChange={(v) => updateConfig('lora_r', v)}
                      min={1}
                      max={256}
                      tooltip="Rang der LoRA-Matrizen - h\u00f6her = mehr Parameter"
                      primaryColor={currentTheme.colors.primary}
                    />
                    <InputField
                      label="LoRA Alpha"
                      value={config.lora_alpha}
                      onChange={(v) => updateConfig('lora_alpha', v)}
                      min={1}
                      max={512}
                      tooltip="Skalierungsfaktor - typischerweise 2*r"
                      primaryColor={currentTheme.colors.primary}
                    />
                    <InputField
                      label="LoRA Dropout"
                      value={config.lora_dropout}
                      onChange={(v) => updateConfig('lora_dropout', v)}
                      step={0.01}
                      min={0}
                      max={0.5}
                      tooltip="Dropout-Rate f\u00fcr LoRA-Layer"
                      primaryColor={currentTheme.colors.primary}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <ToggleField
                      label="4-bit Quantisierung"
                      checked={config.load_in_4bit}
                      onChange={(v) => updateConfig('load_in_4bit', v)}
                      tooltip="QLoRA - l\u00e4dt Modell in 4-bit f\u00fcr weniger VRAM"
                      primaryColor={currentTheme.colors.primary}
                    />
                    <ToggleField
                      label="8-bit Quantisierung"
                      checked={config.load_in_8bit}
                      onChange={(v) => updateConfig('load_in_8bit', v)}
                      tooltip="L\u00e4dt Modell in 8-bit"
                      primaryColor={currentTheme.colors.primary}
                    />
                  </div>
                </>
              )}
            </ConfigSection>

            {/* Advanced Settings */}
            <ConfigSection
              title="Erweiterte Einstellungen"
              icon={<Settings2 className="w-5 h-5 text-gray-400" />}
              expanded={expandedSections.advanced}
              onToggle={() => toggleSection('advanced')}
            >
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <InputField
                  label="Eval Steps"
                  value={config.eval_steps}
                  onChange={(v) => updateConfig('eval_steps', v)}
                  min={10}
                  tooltip="Schritte zwischen Evaluierungen"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Save Steps"
                  value={config.save_steps}
                  onChange={(v) => updateConfig('save_steps', v)}
                  min={10}
                  tooltip="Schritte zwischen Checkpoint-Speicherungen"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Logging Steps"
                  value={config.logging_steps}
                  onChange={(v) => updateConfig('logging_steps', v)}
                  min={1}
                  tooltip="Schritte zwischen Log-Ausgaben"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Save Total Limit"
                  value={config.save_total_limit}
                  onChange={(v) => updateConfig('save_total_limit', v)}
                  min={1}
                  max={10}
                  tooltip="Maximale Anzahl gespeicherter Checkpoints"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Num Workers"
                  value={config.num_workers}
                  onChange={(v) => updateConfig('num_workers', v)}
                  min={0}
                  max={16}
                  tooltip="Anzahl Datenlader-Threads"
                  primaryColor={currentTheme.colors.primary}
                />
                <InputField
                  label="Dropout"
                  value={config.dropout}
                  onChange={(v) => updateConfig('dropout', v)}
                  step={0.01}
                  min={0}
                  max={0.5}
                  tooltip="Dropout-Rate f\u00fcr Regularisierung"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <ToggleField
                  label="Pin Memory"
                  checked={config.pin_memory}
                  onChange={(v) => updateConfig('pin_memory', v)}
                  tooltip="Pinned Memory f\u00fcr schnelleren GPU-Transfer"
                  primaryColor={currentTheme.colors.primary}
                />
                <ToggleField
                  label="Group by Length"
                  checked={config.group_by_length}
                  onChange={(v) => updateConfig('group_by_length', v)}
                  tooltip="Gruppiert Samples nach L\u00e4nge f\u00fcr effizienteres Training"
                  primaryColor={currentTheme.colors.primary}
                />
              </div>
            </ConfigSection>
          </div>
        </div>

        {/* Right Column - Status & Controls */}
        <div className="space-y-6">
          {/* Parameter Rating */}
          {rating && (
            <div
              className={`bg-white/5 rounded-xl border p-5 cursor-pointer hover:bg-white/[0.07] transition-all ${
                rating.rating_info.color === 'green'
                  ? 'border-green-500/30'
                  : rating.rating_info.color === 'blue'
                  ? 'border-blue-500/30'
                  : rating.rating_info.color === 'yellow'
                  ? 'border-yellow-500/30'
                  : rating.rating_info.color === 'orange'
                  ? 'border-orange-500/30'
                  : 'border-red-500/30'
              }`}
              onClick={() => setShowRatingModal(true)}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                  <Gauge className="w-4 h-4" />
                  Parameter-Bewertung
                </h3>
                {ratingLoading && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-3xl font-bold text-white">{rating.score}</div>
                  <div className="text-sm text-gray-400 mt-1">{rating.rating_info.label}</div>
                </div>
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Star
                      key={star}
                      className={`w-4 h-4 ${
                        star <= rating.rating_info.score
                          ? 'text-yellow-400 fill-yellow-400'
                          : 'text-gray-600'
                      }`}
                    />
                  ))}
                </div>
              </div>
              {(rating.issues.length > 0 || rating.warnings.length > 0) && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  {rating.issues.length > 0 && (
                    <div className="flex items-center gap-2 text-xs text-red-400">
                      <AlertCircle className="w-3 h-3" />
                      {rating.issues.length} Problem{rating.issues.length !== 1 ? 'e' : ''}
                    </div>
                  )}
                  {rating.warnings.length > 0 && (
                    <div className="flex items-center gap-2 text-xs text-amber-400 mt-1">
                      <AlertTriangle className="w-3 h-3" />
                      {rating.warnings.length} Warnung{rating.warnings.length !== 1 ? 'en' : ''}
                    </div>
                  )}
                </div>
              )}
              <div className="mt-3 text-xs text-gray-500 flex items-center gap-1">
                <Info className="w-3 h-3" />
                Klicken f\u00fcr Details
              </div>
            </div>
          )}

          {/* Training Status */}
          {isTraining ? (
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-300">Training l\u00e4uft</h3>
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  Live
                </div>
              </div>

              {currentJob && (
                <>
                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-gray-400 mb-2">
                      <span>Fortschritt</span>
                      <span>{currentJob.progress.progress_percent.toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all duration-300`}
                        style={{ width: `${currentJob.progress.progress_percent}%` }}
                      />
                    </div>
                    
                    {/* Timer */}
                    <div className="flex justify-between text-xs text-gray-500 mt-2">
                      <span>
                        {Math.floor(trainingElapsed / 60000)}:{String(Math.floor((trainingElapsed % 60000) / 1000)).padStart(2, '0')} vergangen
                      </span>
                      {currentJob.progress.progress_percent > 0 && (
                        <span>
                          ~{Math.floor((trainingElapsed / currentJob.progress.progress_percent * (100 - currentJob.progress.progress_percent)) / 60000)}:{String(Math.floor(((trainingElapsed / currentJob.progress.progress_percent * (100 - currentJob.progress.progress_percent)) % 60000) / 1000)).padStart(2, '0')} verbleibend
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Epoch</span>
                      <span className="text-white font-medium">
                        {currentJob.progress.epoch} / {currentJob.progress.total_epochs}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Step</span>
                      <span className="text-white font-medium">
                        {currentJob.progress.step} / {currentJob.progress.total_steps}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Train Loss</span>
                      <span className="text-white font-medium font-mono">
                        {currentJob.progress.train_loss.toFixed(4)}
                      </span>
                    </div>
                    {currentJob.progress.val_loss !== null && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Val Loss</span>
                        <span className="text-white font-medium font-mono">
                          {currentJob.progress.val_loss.toFixed(4)}
                        </span>
                      </div>
                    )}
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Learning Rate</span>
                      <span className="text-white font-medium font-mono text-xs">
                        {formatLearningRate(currentJob.progress.learning_rate)}
                      </span>
                    </div>
                  </div>

                  {trainingStatus && (
                    <div className="mt-4 p-3 bg-white/5 rounded-lg text-xs text-gray-400">
                      {trainingStatus}
                    </div>
                  )}

                  {/* Stop Button */}
                  <button
                    onClick={handleStopTraining}
                    className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-3 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 rounded-lg text-red-400 font-medium transition-all"
                  >
                    <Square className="w-4 h-4" />
                    Training stoppen
                  </button>
                </>
              )}
            </div>
          ) : (
            /* Start Training Button */
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <h3 className="text-sm font-medium text-gray-300 mb-4">Training starten</h3>
              <div className="space-y-3 text-sm mb-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Modell</span>
                  <span className="text-white font-medium truncate ml-2 max-w-[150px]" title={selectedModel?.name}>
                    {selectedModel?.name || '-'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Dataset</span>
                  <span className="text-white font-medium truncate ml-2 max-w-[150px]" title={selectedDataset?.name}>
                    {selectedDataset?.name || '-'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Epochen</span>
                  <span className="text-white font-medium">{config.epochs}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Batch Size</span>
                  <span className="text-white font-medium">{config.batch_size}</span>
                </div>
              </div>
              <button
                onClick={handleStartTraining}
                disabled={!selectedModelId || !selectedDatasetId || !requirements?.ready}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <Play className="w-4 h-4" />
                Training starten
              </button>
              {!requirements?.ready && (
                <p className="text-xs text-amber-400 mt-3 text-center">
                  System-Anforderungen pr\u00fcfen
                </p>
              )}
            </div>
          )}

          {/* Loss Chart */}
          {lossHistory.length > 0 && (
            <LossChart history={lossHistory} primaryColor={currentTheme.colors.primary} />
          )}

          {/* Info Cards */}
          <div className="space-y-3">
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-blue-300 font-medium mb-1">Tipp</p>
                  <p className="text-xs text-blue-200/70">
                    Nutze Presets f\u00fcr schnellen Start. Die Parameter werden automatisch bewertet.
                  </p>
                </div>
              </div>
            </div>

            {/* Download Template Button */}
            <button
              onClick={handleDownloadTemplate}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 rounded-lg text-gray-300 hover:text-white transition-all group"
            >
              <Download className="w-4 h-4 group-hover:scale-110 transition-transform" />
              <span className="text-sm font-medium">Standard JSON-Format herunterladen</span>
            </button>
            <p className="text-xs text-gray-500 text-center -mt-1">
              Wird in deinen Downloads-Ordner gespeichert
            </p>
          </div>
        </div>
      </div>

      {/* Modals */}
      {showValidationModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-6 h-6 text-amber-400" />
                <h2 className="text-xl font-bold text-white">Konfiguration validiert</h2>
              </div>
              <button
                onClick={() => {
                  setShowValidationModal(false);
                  setValidationIssues([]);
                  delete (window as any).pendingConfig;
                }}
                className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto max-h-[50vh]">
              <p className="text-gray-300 mb-4">
                {validationIssues.length} Parameter konnte(n) nicht übernommen werden und wurde(n) durch Standardwerte ersetzt:
              </p>
              <div className="space-y-3">
                {validationIssues.map((issue, i) => (
                  <div key={i} className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="font-medium text-white mb-1">{issue.field}</div>
                        <div className="text-sm text-amber-300 mb-2">{issue.reason}</div>
                        <div className="flex items-center gap-3 text-sm">
                          <div>
                            <span className="text-gray-400">Ungültiger Wert:</span>
                            <code className="ml-2 px-2 py-0.5 bg-red-500/20 text-red-300 rounded">
                              {JSON.stringify(issue.value)}
                            </code>
                          </div>
                          <ChevronRight className="w-4 h-4 text-gray-500" />
                          <div>
                            <span className="text-gray-400">Ersetzt durch:</span>
                            <code className="ml-2 px-2 py-0.5 bg-green-500/20 text-green-300 rounded">
                              {JSON.stringify(issue.defaultValue)}
                            </code>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6 border-t border-white/10 flex gap-3">
              <button
                onClick={() => {
                  setShowValidationModal(false);
                  setValidationIssues([]);
                  delete (window as any).pendingConfig;
                }}
                className="flex-1 py-3 bg-white/5 hover:bg-white/10 rounded-lg text-white transition-all"
              >
                Abbrechen
              </button>
              <button
                onClick={handleConfirmValidation}
                className={`flex-1 py-3 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white font-medium hover:opacity-90 transition-all`}
              >
                Bestätigen und laden
              </button>
            </div>
          </div>
        </div>
      )}

      {showRatingModal && rating && (
        <RatingModal
          rating={rating}
          onClose={() => setShowRatingModal(false)}
          primaryColor={currentTheme.colors.primary}
          gradient={currentTheme.colors.gradient}
        />
      )}

      {showRequirementsModal && requirements && (
        <RequirementsModal
          requirements={requirements}
          onClose={() => setShowRequirementsModal(false)}
          onRefresh={checkRequirements}
          gradient={currentTheme.colors.gradient}
        />
      )}

      {showHistoryModal && (
        <HistoryModal
          jobs={trainingHistory}
          onClose={() => setShowHistoryModal(false)}
          onDelete={handleDeleteHistoryJob}
          gradient={currentTheme.colors.gradient}
        />
      )}
    </div>
  );
}
