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
  Moon,
  MemoryStick,
  SlidersHorizontal,
  ChevronUp,
  Brain,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

interface ModelRamInfo {
  param_billion: number;
  model_type: string;
  readable_size: string;
  hidden_size: number;
  num_hidden_layers: number;
}

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
  gradient_checkpointing: boolean;
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
  gradient_checkpointing: false,
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

// ============ Post-Training Modal ============

interface PostTrainingModalProps {
  versionId: string;
  modelName: string;
  metrics: { final_train_loss?: number; total_epochs?: number; training_duration_seconds?: number } | null;
  onClose: () => void;
  onGoToAnalysis: () => void;
  gradient: string;
  primaryColor: string;
}

function PostTrainingModal({ versionId, modelName, metrics, onClose, onGoToAnalysis, gradient, primaryColor }: PostTrainingModalProps) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-lg overflow-hidden">
        <div className="p-6 text-center">
          <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-1">Training abgeschlossen!</h2>
          <p className="text-gray-400 mb-5">{modelName}</p>
          <div className="grid grid-cols-3 gap-3 mb-6">
            {metrics?.final_train_loss !== undefined && (
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-0.5">Final Loss</div>
                <div className="text-white font-bold">{metrics.final_train_loss.toFixed(4)}</div>
              </div>
            )}
            {metrics?.total_epochs !== undefined && (
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-0.5">Epochen</div>
                <div className="text-white font-bold">{metrics.total_epochs}</div>
              </div>
            )}
            {metrics?.training_duration_seconds !== undefined && (
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-0.5">Dauer</div>
                <div className="text-white font-bold text-sm">{
                  metrics.training_duration_seconds > 3600
                    ? `${Math.floor(metrics.training_duration_seconds/3600)}h ${Math.floor((metrics.training_duration_seconds%3600)/60)}m`
                    : metrics.training_duration_seconds > 60
                    ? `${Math.floor(metrics.training_duration_seconds/60)}m`
                    : `${metrics.training_duration_seconds}s`
                }</div>
              </div>
            )}
          </div>
          <div className="space-y-3">
            <button
              onClick={onGoToAnalysis}
              className={`w-full py-3 bg-gradient-to-r ${gradient} rounded-xl text-white font-semibold hover:opacity-90 transition-all flex items-center justify-center gap-2`}
            >
              <Brain className="w-5 h-5" />
              Trainingsanalyse öffnen
            </button>
            <button
              onClick={onClose}
              className="w-full py-3 bg-white/5 hover:bg-white/10 rounded-xl text-gray-300 hover:text-white transition-all text-sm"
            >
              Schließen
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ KI-Assistent Modal ============

interface AIAssistantModalProps {
  config: TrainingConfig;
  modelInfo: ModelRamInfo | null;
  selectedModel: ModelInfo | undefined;
  selectedDataset: DatasetInfo | undefined;
  systemRamGb: number;
  requirements: RequirementsCheck | null;
  prefilledContext?: string | null;
  onApply: (patch: Partial<TrainingConfig>) => void;
  onClose: () => void;
  gradient: string;
  primaryColor: string;
}

type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';

const PROVIDER_META: Record<AIProvider, {
  label: string; emoji: string; needsKey: boolean;
  keyPlaceholder: string; keyHint: string; keyLink: string;
  models: string[];
}> = {
  anthropic: {
    label: 'Claude (Anthropic)',
    emoji: '🤖',
    needsKey: true,
    keyPlaceholder: 'sk-ant-api03-...',
    keyHint: 'Kostenlos testen: console.anthropic.com',
    keyLink: 'https://console.anthropic.com',
    models: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  },
  openai: {
    label: 'GPT-4o (OpenAI)',
    emoji: '🟢',
    needsKey: true,
    keyPlaceholder: 'sk-...',
    keyHint: 'platform.openai.com/api-keys',
    keyLink: 'https://platform.openai.com/api-keys',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
  },
  groq: {
    label: 'Groq (Kostenlos)',
    emoji: '⚡',
    needsKey: true,
    keyPlaceholder: 'gsk_...',
    keyHint: '✅ Kostenloser Account — console.groq.com',
    keyLink: 'https://console.groq.com',
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
  },
  ollama: {
    label: 'Ollama (Lokal, kein Key)',
    emoji: '🦙',
    needsKey: false,
    keyPlaceholder: '',
    keyHint: '✅ Kein Account nötig — ollama.com installieren',
    keyLink: 'https://ollama.com',
    models: ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5'],
  },
};

function AIAssistantModal({
  config, modelInfo, selectedModel, selectedDataset,
  systemRamGb, requirements, prefilledContext, onApply, onClose, gradient, primaryColor,
}: AIAssistantModalProps) {
  const [provider, setProvider] = useState<AIProvider>(
    () => (localStorage.getItem('ft_ai_provider') as AIProvider) || 'ollama'
  );
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('ft_ai_api_key') || '');
  const [ollamaModel, setOllamaModel] = useState(() => localStorage.getItem('ft_ollama_model') || 'llama3.2');
  const [selectedModel2, setSelectedModel2] = useState(() => {
    const saved = localStorage.getItem('ft_ai_model');
    return saved || PROVIDER_META.ollama.models[0];
  });
  const [ollamaStatus, setOllamaStatus] = useState<'unchecked' | 'ok' | 'error'>('unchecked');
  const [prompt, setPrompt] = useState(prefilledContext || '');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string>('');
  const [parsedConfig, setParsedConfig] = useState<Partial<TrainingConfig> | null>(null);
  const [error, setError] = useState('');

  const meta = PROVIDER_META[provider];

  const saveSettings = () => {
    localStorage.setItem('ft_ai_api_key', apiKey);
    localStorage.setItem('ft_ai_provider', provider);
    localStorage.setItem('ft_ollama_model', ollamaModel);
    localStorage.setItem('ft_ai_model', selectedModel2);
  };

  // Ollama-Verbindung prüfen
  const checkOllama = async () => {
    try {
      const res = await fetch('http://localhost:11434/api/tags', {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const data = await res.json();
        // Verfügbare Modelle extrahieren
        const available = (data.models || []).map((m: any) => m.name.split(':')[0]);
        if (available.length > 0 && !available.includes(ollamaModel)) {
          setOllamaModel(available[0]);
        }
        setOllamaStatus('ok');
      } else {
        setOllamaStatus('error');
      }
    } catch {
      setOllamaStatus('error');
    }
  };

  // Ollama beim Wählen des Providers prüfen
  const handleProviderChange = (p: AIProvider) => {
    setProvider(p);
    // Modell auf ersten der Liste setzen wenn neuer Provider
    setSelectedModel2(PROVIDER_META[p].models[0]);
    if (p === 'ollama') checkOllama();
  };

  const buildSystemPrompt = () => {
    const modelName = selectedModel?.name || 'Unbekannt';
    const modelType = modelInfo?.model_type || 'unknown';
    const paramB = modelInfo?.param_billion?.toFixed(2) || '?';
    const hiddenSize = modelInfo?.hidden_size || 768;
    const numLayers = modelInfo?.num_hidden_layers || 12;
    const datasetFiles = selectedDataset?.file_count || 0;
    const datasetMb = ((selectedDataset?.size_bytes || 0) / 1e6).toFixed(0);

    return `Du bist ein Experte für das Training von HuggingFace-Transformer-Modellen mit PyTorch.

KONTEXT DES USERS:
- Modell: ${modelName} (${modelType}, ${paramB}B Parameter, hidden_size=${hiddenSize}, num_layers=${numLayers})
- Dataset: ${datasetFiles} Dateien, ${datasetMb} MB
- System-RAM: ${systemRamGb} GB (Apple Silicon MPS oder CPU, kein dediziertes VRAM)
\n- Hardware/GPU: ${requirements?.cuda_available ? 'NVIDIA GPU mit CUDA — bitsandbytes verfügbar, load_in_4bit/8bit möglich' : requirements?.mps_available ? 'Apple Silicon MPS — bitsandbytes NICHT verfügbar! Kein load_in_4bit oder load_in_8bit empfehlen. Stattdessen use_lora=true für RAM-Effizienz nutzen. fp16=false und bf16=false setzen.' : 'Nur CPU — bitsandbytes NICHT verfügbar! Kein load_in_4bit oder load_in_8bit empfehlen.'}
- Aktuelle Konfiguration:
${JSON.stringify({
  epochs: config.epochs, batch_size: config.batch_size,
  gradient_accumulation_steps: config.gradient_accumulation_steps,
  learning_rate: config.learning_rate, optimizer: config.optimizer,
  scheduler: config.scheduler, use_lora: config.use_lora,
  lora_r: config.lora_r, lora_alpha: config.lora_alpha,
  fp16: config.fp16, bf16: config.bf16, load_in_4bit: config.load_in_4bit,
  max_seq_length: config.max_seq_length, gradient_checkpointing: config.gradient_checkpointing,
  num_workers: config.num_workers,
}, null, 2)}

DEINE AUFGABE:
1. Analysiere das Ziel des Users
2. Empfehle optimale Trainingsparameter
3. Erkläre kurz (3-5 Sätze) warum diese Parameter gewählt wurden
4. Gib am ENDE deiner Antwort exakt einen JSON-Block mit den empfohlenen Parametern aus

WICHTIG für Apple Silicon / CPU-Training:
- Kein CUDA verfügbar, nur MPS oder CPU
- fp16=false empfehlen (MPS-Instabilitäten bei fp16), stattdessen bf16=false auch (stabiler auf CPU)
- num_workers=0 oder 1 empfehlen (MPS + multi-processing = Deadlocks)
- Bei wenig RAM: use_lora=true + load_in_4bit=true + gradient_checkpointing=true + batch_size=1 + gradient_accumulation_steps=8

JSON-FORMAT (nur diese Felder, exakt dieser Block am Ende):
\`\`\`json
{
  "epochs": 3,
  "batch_size": 2,
  "gradient_accumulation_steps": 4,
  "learning_rate": 0.0002,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "warmup_ratio": 0.05,
  "weight_decay": 0.01,
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "load_in_4bit": false,
  "fp16": false,
  "bf16": false,
  "max_seq_length": 512,
  "gradient_checkpointing": true,
  "num_workers": 0,
  "max_grad_norm": 1.0
}
\`\`\`
`;
  };

  const handleAsk = async () => {
    if (meta.needsKey && !apiKey.trim()) { setError('Bitte API-Key eingeben.'); return; }
    if (!prompt.trim()) { setError('Bitte beschreibe dein Trainingsziel.'); return; }
    setLoading(true); setError(''); setResult(''); setParsedConfig(null);
    saveSettings();

    try {
      let responseText = '';

      if (provider === 'ollama') {
        // Ollama: Lokale REST-API, kein Key nötig
        const model = ollamaModel.trim() || 'llama3.2';
        const res = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            prompt: buildSystemPrompt() + '\n\nUser: ' + prompt + '\n\nAssistant:',
            stream: false,
            options: { temperature: 0.3, num_ctx: 4096 },
          }),
        });
        if (!res.ok) throw new Error(`Ollama nicht erreichbar (HTTP ${res.status}). Läuft Ollama? Starte es mit: ollama serve`);
        const data = await res.json();
        responseText = data.response || '';

      } else if (provider === 'groq') {
        // Groq: OpenAI-kompatibel, kostenloser Tier
        const model = selectedModel2 || 'llama-3.3-70b-versatile';
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({
            model,
            max_tokens: 1500,
            temperature: 0.3,
            messages: [
              { role: 'system', content: buildSystemPrompt() },
              { role: 'user', content: prompt },
            ],
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';

      } else if (provider === 'anthropic') {
        const model = selectedModel2 || 'claude-opus-4-5';
        const res = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          body: JSON.stringify({
            model,
            max_tokens: 1500,
            system: buildSystemPrompt(),
            messages: [{ role: 'user', content: prompt }],
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.content?.[0]?.text || '';

      } else {
        // OpenAI
        const model = selectedModel2 || 'gpt-4o';
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({
            model,
            max_tokens: 1500,
            temperature: 0.3,
            messages: [
              { role: 'system', content: buildSystemPrompt() },
              { role: 'user', content: prompt },
            ],
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';
      }

      setResult(responseText);
      // JSON-Block extrahieren
      const match = responseText.match(/```json\s*([\s\S]*?)```/);
      if (match) {
        try { setParsedConfig(JSON.parse(match[1].trim())); }
        catch (_) { setError('JSON konnte nicht geparst werden. Parameter manuell übertragen.'); }
      }
    } catch (e: any) {
      setError('Fehler: ' + String(e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[92vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-purple-500/20 flex items-center justify-center">
              <span className="text-lg">🤖</span>
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">KI-Assistent</h2>
              <p className="text-xs text-gray-400">Parameter automatisch optimieren lassen</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* Provider-Auswahl */}
          <div className="space-y-3">
            <div className="text-xs font-bold uppercase tracking-widest text-gray-500">KI-Anbieter wählen</div>
            <div className="grid grid-cols-2 gap-2">
              {(Object.entries(PROVIDER_META) as [AIProvider, typeof PROVIDER_META[AIProvider]][]).map(([key, m]) => (
                <button
                  key={key}
                  onClick={() => handleProviderChange(key)}
                  className={`flex items-center gap-2.5 p-3 rounded-xl border text-left transition-all ${
                    provider === key
                      ? 'bg-purple-500/20 border-purple-500/50 text-white'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <span className="text-lg flex-shrink-0">{m.emoji}</span>
                  <div className="min-w-0">
                    <div className="text-sm font-semibold truncate">{m.label}</div>
                    <div className="text-xs opacity-60 mt-0.5">{m.needsKey ? 'API-Key nötig' : '✅ Kein Key'}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Key / Modell-Konfiguration */}
          {meta.needsKey ? (
            <div className="bg-white/[0.04] rounded-xl border border-white/10 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold uppercase tracking-widest text-gray-500">API-Key</div>
                <a
                  href={meta.keyLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Key holen ↗
                </a>
              </div>
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder={meta.keyPlaceholder}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
              <p className="text-xs text-gray-400">{meta.keyHint}</p>
              {/* Modell-Auswahl */}
              <div>
                <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">Modell</div>
                <div className="flex gap-2 flex-wrap">
                  {meta.models.map(m => (
                    <button
                      key={m}
                      onClick={() => setSelectedModel2(m)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                        selectedModel2 === m
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-xs text-gray-500">Key wird nur lokal gespeichert, nie an FrameTrain übertragen.</p>
            </div>
          ) : (
            /* Ollama: kein Key, aber Modell-Name und Status */
            <div className="bg-white/[0.04] rounded-xl border border-white/10 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold uppercase tracking-widest text-gray-500">Ollama-Konfiguration</div>
                <button
                  onClick={checkOllama}
                  className="text-xs text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Verbindung prüfen
                </button>
              </div>

              {/* Status-Anzeige */}
              {ollamaStatus !== 'unchecked' && (
                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${
                  ollamaStatus === 'ok'
                    ? 'bg-green-500/10 border border-green-500/30 text-green-300'
                    : 'bg-red-500/10 border border-red-500/30 text-red-300'
                }`}>
                  {ollamaStatus === 'ok'
                    ? <><CheckCircle className="w-3.5 h-3.5" /> Ollama läuft und ist erreichbar</>
                    : <><AlertCircle className="w-3.5 h-3.5" /> Ollama nicht erreichbar. Starte es mit: <code className="ml-1 font-mono">ollama serve</code></>}
                </div>
              )}

              {/* Modell-Name */}
              <div>
                <div className="text-xs text-gray-400 mb-1.5">Modell-Name (muss lokal installiert sein)</div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={ollamaModel}
                    onChange={e => setOllamaModel(e.target.value)}
                    placeholder="llama3.2"
                    className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                </div>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {PROVIDER_META.ollama.models.map(m => (
                    <button
                      key={m}
                      onClick={() => setOllamaModel(m)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono border transition-all ${
                        ollamaModel === m
                          ? 'bg-green-500/20 border-green-500/50 text-green-300'
                          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-3 bg-white/[0.03] rounded-lg border border-white/10 text-xs text-gray-400 space-y-1">
                <div className="font-semibold text-gray-300">Ollama noch nicht installiert?</div>
                <div>1. <a href="https://ollama.com" target="_blank" className="text-purple-400 underline">ollama.com</a> — kostenlos herunterladen</div>
                <div>2. Im Terminal: <code className="bg-black/30 px-1 py-0.5 rounded font-mono">ollama pull llama3.2</code></div>
                <div>3. Ollama startet automatisch im Hintergrund</div>
              </div>
            </div>
          )}

          {/* Kontext-Info */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            {[
              { label: 'Modell', value: selectedModel?.name || '–' },
              { label: 'Dataset', value: selectedDataset ? `${selectedDataset.file_count} Dateien` : '–' },
              { label: 'System-RAM', value: `${systemRamGb} GB` },
            ].map(item => (
              <div key={item.label} className="bg-white/[0.04] rounded-lg p-3">
                <div className="text-gray-500 mb-1">{item.label}</div>
                <div className="text-white font-medium truncate" title={item.value}>{item.value}</div>
              </div>
            ))}
          </div>

          {/* Prompt */}
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase tracking-widest text-gray-500">Dein Trainingsziel</label>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              rows={4}
              placeholder="z.B. 'Ich möchte ein deutsches Sprachmodell für Sentiment-Analyse fine-tunen. Mein RAM ist begrenzt (16GB). Stabilität ist wichtiger als Geschwindigkeit.'"
              className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm resize-none focus:outline-none focus:ring-2 focus:ring-purple-500/50"
            />
            <div className="flex flex-wrap gap-2">
              {[
                'RAM ist begrenzt, max. Effizienz',
                'Schnelles Training, Qualität sekundär',
                'Höchste Modellqualität, Zeit egal',
                'Erstes Fine-Tuning, sicher & stabil',
              ].map(hint => (
                <button
                  key={hint}
                  onClick={() => setPrompt(p => p ? p + ' ' + hint : hint)}
                  className="text-xs px-2.5 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full text-gray-400 hover:text-white transition-all"
                >
                  + {hint}
                </button>
              ))}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-300">{error}</div>
          )}

          {/* KI-Antwort */}
          {result && (
            <div className="space-y-3">
              <div className="text-xs font-bold uppercase tracking-widest text-gray-500">KI-Empfehlung</div>
              <div className="p-4 bg-white/[0.03] border border-white/10 rounded-xl max-h-64 overflow-y-auto">
                <pre className="text-sm text-gray-300 whitespace-pre-wrap break-words leading-relaxed font-sans">{result}</pre>
              </div>
              {parsedConfig && (
                <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <div className="flex items-center gap-2 text-green-400 text-sm font-semibold mb-2">
                    <CheckCircle className="w-4 h-4" />
                    {Object.keys(parsedConfig).length} Parameter erkannt und übertragbar
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(parsedConfig).map(([k, v]) => (
                      <span key={k} className="text-xs px-2 py-0.5 bg-green-500/10 text-green-300 rounded font-mono">
                        {k}: {String(v)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-white/10 flex gap-3 flex-shrink-0">
          <button
            onClick={handleAsk}
            disabled={loading}
            className={`flex-1 flex items-center justify-center gap-2 py-3 bg-gradient-to-r ${gradient} rounded-xl text-white font-semibold hover:opacity-90 transition-all disabled:opacity-50`}
          >
            {loading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> KI denkt nach…</>
              : <>{meta.emoji} Parameter optimieren</>}
          </button>
          {parsedConfig && (
            <button
              onClick={() => { if (parsedConfig) { onApply(parsedConfig); onClose(); } }}
              className="flex-1 flex items-center justify-center gap-2 py-3 bg-green-500/20 hover:bg-green-500/30 border border-green-500/40 rounded-xl text-green-300 font-semibold transition-all"
            >
              <CheckCircle className="w-4 h-4" /> Parameter übernehmen
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ============ Training Error Modal ============

interface TrainingErrorModalProps {
  errorTitle: string;
  errorMessage: string;
  errorDetails: string;
  errorLogs: string[];
  configSnapshot: string;
  onClose: () => void;
  onOpenAIWithError?: (errorContext: string) => void;
  gradient: string;
  primaryColor: string;
}

function TrainingErrorModal({
  errorTitle,
  errorMessage,
  errorDetails,
  errorLogs,
  configSnapshot,
  onClose,
  onOpenAIWithError,
  gradient,
  primaryColor,
}: TrainingErrorModalProps) {
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [sendError, setSendError] = useState('');

  // Kategorisiere den Fehler für den User
  const isRamError = errorMessage.toLowerCase().includes('sigkill') ||
    errorMessage.toLowerCase().includes('arbeitsspeicher') ||
    errorMessage.toLowerCase().includes('ram') ||
    errorMessage.toLowerCase().includes('memory') ||
    errorMessage.toLowerCase().includes('oom') ||
    errorTitle.toLowerCase().includes('ram') ||
    // Exit-Code 1 nach Training-Start = häufig OOM oder MPS-Crash
    (errorMessage.toLowerCase().includes('exit-code') && errorLogs.some(l =>
      l.toLowerCase().includes('killed') ||
      l.toLowerCase().includes('error') ||
      l.toLowerCase().includes('oom')
    )) ||
    // Klassischer macOS-OOM: Prozess stirbt nach mehreren Schritten
    errorMessage.toLowerCase().includes('unerwartet beendet');

  const isLoraError = errorDetails.toLowerCase().includes('target modules') ||
    errorDetails.toLowerCase().includes('not found in the base model') ||
    errorDetails.toLowerCase().includes('lora');

  const isConfigError = errorDetails.toLowerCase().includes('konfigurationsfehler') ||
    errorDetails.toLowerCase().includes('configuration') ||
    errorTitle.toLowerCase().includes('konfiguration');

  // Nutzerfreundliche Erklärung je nach Fehlertyp
  const getExplanation = () => {
    if (isRamError) return {
      icon: '🧠',
      color: 'text-orange-400',
      bg: 'bg-orange-500/10',
      border: 'border-orange-500/30',
      title: 'Nicht genug Arbeitsspeicher (RAM)',
      text: 'Dein System hatte nicht genug RAM für dieses Training. Auf Apple Silicon (MPS) teilen sich CPU und GPU denselben Speicher — der Verbrauch übersteigt die theoretische Schätzung deutlich.',
      fixes: [
        'Batch Size auf 1 setzen + Gradient Accumulation auf 8–16 erhöhen',
        'LoRA aktivieren — trainiert nur 1–5% der Parameter',
        'Gradient Checkpointing aktivieren (im Erweitert-Abschnitt)',
        'num_workers auf 0 setzen (MPS + Worker = RAM-Spike + Deadlocks)',
        'Sequenzlänge halbieren (z.B. 512 → 256 → 128)',
        '4-bit QLoRA aktivieren für maximale RAM-Ersparnis',
        'Alle anderen Apps schließen (Chrome, Slack etc.)',
        '→ KI-Assistent nutzen: liefert optimale Parameter für deinen RAM',
      ],
    };
    if (isLoraError) return {
      icon: '⚙️',
      color: 'text-purple-400',
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      title: 'LoRA Konfigurationsfehler',
      text: 'Die LoRA Target Modules (z.B. q_proj, v_proj) wurden im Modell nicht gefunden. Nicht alle Modelle unterstützen dieselben Modul-Namen.',
      fixes: [
        'Versuche andere Module: k_proj, o_proj, gate_proj, up_proj, down_proj',
        'Für GPT-2 Style: c_attn, c_proj verwenden',
        'Für BERT Style: query, value verwenden',
        'LoRA deaktivieren und Standard-Fine-Tuning ausprobieren',
        'Überprüfe die model_type in der config.json des Modells',
      ],
    };
    if (isConfigError) return {
      icon: '📋',
      color: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      title: 'Konfigurationsfehler',
      text: 'Ein ungültiger Konfigurationsparameter hat das Training verhindert.',
      fixes: [
        'Überprüfe alle Parameter auf gültige Werte',
        'Nutze ein Preset als Ausgangspunkt',
        'Stelle sicher dass Batch Size ≥ 1 ist',
        'Learning Rate muss zwischen 1e-6 und 1e-2 liegen',
      ],
    };
    return {
      icon: '❌',
      color: 'text-red-400',
      bg: 'bg-red-500/10',
      border: 'border-red-500/30',
      title: 'Training-Fehler',
      text: 'Das Training ist mit einem unerwarteten Fehler abgebrochen.',
      fixes: [
        'FrameTrain neu starten',
        'Python-Pakete aktualisieren: pip install -U torch transformers peft',
        'Modell und Dataset erneut prüfen',
        'Mit einem kleineren Preset (Schnelltest) testen',
      ],
    };
  };

  const explanation = getExplanation();

  const handleSendReport = async () => {
    setSending(true);
    setSendError('');
    try {
      const errorType = isRamError ? 'training:oom' : isLoraError ? 'training:lora_config' : isConfigError ? 'training:config' : 'training:error';
      const response = await fetch('https://webcontrol-hq-api.karol-paschek.workers.dev/api/app-errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          site_id: 'frametrain',
          error_type: errorType,
          title: errorTitle || 'Training-Fehler',
          message: errorMessage,
          details: errorDetails,
          logs: errorLogs.join('\n'),
          config_snapshot: configSnapshot,
          platform: navigator.platform || 'unknown',
          app_version: 'desktop-app2',
        }),
      });
      if (response.ok) {
        setSent(true);
      } else {
        setSendError('Senden fehlgeschlagen. Bitte prüfe deine Internetverbindung.');
      }
    } catch (e: any) {
      setSendError('Netzwerkfehler: ' + String(e));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-red-500/20 flex items-center justify-center">
              <AlertCircle className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Training fehlgeschlagen</h2>
              <p className="text-xs text-gray-400 mt-0.5">{errorTitle}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* Categorized explanation */}
          <div className={`rounded-xl border p-4 ${explanation.bg} ${explanation.border}`}>
            <div className="flex items-start gap-3">
              <span className="text-2xl flex-shrink-0">{explanation.icon}</span>
              <div className="flex-1">
                <div className={`font-bold text-sm mb-1 ${explanation.color}`}>{explanation.title}</div>
                <p className="text-sm text-gray-300 leading-relaxed">{explanation.text}</p>
              </div>
            </div>
          </div>

          {/* Fehler-Nachricht */}
          {errorMessage && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500">
                <AlertTriangle className="w-3 h-3" /> Fehlermeldung
              </div>
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                <p className="text-sm text-red-300 leading-relaxed font-mono whitespace-pre-wrap break-words">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* Details */}
          {errorDetails && errorDetails !== errorMessage && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500">
                <Info className="w-3 h-3" /> Details
              </div>
              <div className="p-3 bg-white/5 border border-white/10 rounded-lg max-h-40 overflow-y-auto">
                <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words leading-relaxed">{errorDetails}</pre>
              </div>
            </div>
          )}

          {/* Logs */}
          {errorLogs.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500">
                <BarChart3 className="w-3 h-3" /> Logs (letzte {errorLogs.length} Zeilen)
              </div>
              <div className="p-3 bg-black/40 border border-white/10 rounded-lg max-h-48 overflow-y-auto">
                <pre className="text-xs text-gray-400 whitespace-pre-wrap break-words leading-relaxed font-mono">{errorLogs.join('\n')}</pre>
              </div>
            </div>
          )}

          {/* Was kann ich tun */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-gray-500">
              <CheckCircle className="w-3 h-3" /> Was kann ich tun?
            </div>
            <div className="space-y-2">
              {explanation.fixes.map((fix, i) => (
                <div key={i} className="flex items-start gap-2.5 p-2.5 bg-white/[0.04] rounded-lg">
                  <span className="text-blue-400 font-bold text-xs flex-shrink-0 mt-0.5">{i + 1}.</span>
                  <span className="text-sm text-gray-300">{fix}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Send report */}
          <div className="bg-white/[0.03] rounded-xl border border-white/10 p-4">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="text-sm font-semibold text-white mb-1">Fehlerbericht an FrameTrain senden?</div>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Sendet Fehlermeldung, Logs und Konfiguration anonym an das FrameTrain-Team zur Analyse und Verbesserung.
                </p>
              </div>
            </div>
            {sent ? (
              <div className="mt-3 flex items-center gap-2 text-green-400 text-sm">
                <CheckCircle className="w-4 h-4" />
                Bericht gesendet – danke!
              </div>
            ) : (
              <div className="mt-3 space-y-2">
                {sendError && (
                  <p className="text-xs text-red-400">{sendError}</p>
                )}
                <button
                  onClick={handleSendReport}
                  disabled={sending}
                  className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 hover:text-white transition-all disabled:opacity-50"
                >
                  {sending ? (
                    <><Loader2 className="w-4 h-4 animate-spin" /> Wird gesendet…</>
                  ) : (
                    <><Download className="w-4 h-4" /> Fehlerbericht senden</>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-white/10 flex gap-3 flex-col flex-shrink-0">
          {onOpenAIWithError && errorLogs && errorLogs.length > 0 && (
            <button
              onClick={() => {
                const ctx = `FEHLER: ${errorTitle}\n\nMELDUNG: ${errorMessage}\n\nDETAILS:\n${errorDetails}\n\nLOGS:\n${errorLogs.join('\n')}\n\nAktuelle Config:\n${configSnapshot}`;
                onClose();
                onOpenAIWithError(ctx);
              }}
              className={`w-full py-3 bg-gradient-to-r ${gradient} rounded-xl text-white font-semibold hover:opacity-90 transition-all flex items-center justify-center gap-2`}
            >
              <Sparkles className="w-4 h-4" />
              Mit KI-Assistent analysieren & Parameter korrigieren
            </button>
          )}
          <button
            onClick={onClose}
            className={`flex-1 py-3 bg-white/5 border border-white/10 rounded-xl text-white font-semibold hover:bg-white/10 transition-all`}
          >
            Schließen & Einstellungen anpassen
          </button>
        </div>
      </div>
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

// ============ RAM Calculator Component ============

const SYSTEM_RAM_OPTIONS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

interface RamBreakdown {
  modelGb: number;
  gradientsGb: number;
  optimizerGb: number;
  activationsGb: number;
  workersGb: number;
  overheadGb: number;
  totalGb: number;
}

// Plattform-Typ ermitteln (MPS = Apple Silicon, CUDA = NVIDIA, CPU)
function detectPlatform(): 'mps' | 'cuda' | 'cpu' {
  // Aus navigator.platform lässt sich kein sicherer Schluss ziehen;
  // wir prüfen ob es ein Mac ist (MPS wahrscheinlich)
  if (navigator.platform?.toLowerCase().includes('mac') ||
      navigator.userAgent?.toLowerCase().includes('mac')) return 'mps';
  return 'cpu'; // Fallback; CUDA würde der Rust-Backend melden
}

function calcRam(
  paramsBillion: number,
  cfg: TrainingConfig,
  datasetSizeBytes: number,
  hiddenSize: number = 768,
  numLayers: number = 12,
  isMps: boolean = false,
): RamBreakdown {
  const p = paramsBillion * 1e9;

  // Bytes pro Parameter im Modell je nach Quantisierung
  const bpp = cfg.load_in_4bit ? 0.5 : cfg.load_in_8bit ? 1 : cfg.fp16 || cfg.bf16 ? 2 : 4;
  const modelGb = (p * bpp) / 1e9;

  // Gradienten: immer fp32 (auch bei fp16-Training hält PyTorch fp32-Kopie)
  // Bei LoRA nur Gradienten für trainierbare Parameter
  const trainableRatio = cfg.use_lora ? Math.min((cfg.lora_r / 64) * 0.05, 0.05) : 1.0;
  const gradientsGb = (p * trainableRatio * 4) / 1e9;

  // Optimizer: AdamW = 2x fp32-Momentum-Tensoren pro trainierbarem Parameter
  const optimizerGb = (p * trainableRatio * 2 * 4) / 1e9;

  // Aktivierungen: physikalisch korrekte Formel für Transformer-Architekturen
  // Pro Layer und Token werden gespeichert:
  //   Attention: Q, K, V + scores + output = ~5 * H * B * S * 4 bytes
  //   FFN:       up-proj + activation + down-proj = ~3 * 4H * B * S * 4 bytes (mittelt sich auf ~4H)
  //   LayerNorm: 2 * H * B * S * 4 bytes
  //   Residuals: 2 * H * B * S * 4 bytes
  // Gesamt pro Layer: (5H + 16H + 4H) * B * S * 4 ≈ (H * 13 + 4*H) * B * S * 4
  //                 = (13 + 4) * H * B * S * 4 ≈ 17 * H * B * S * 4 bytes
  // Backward-Pass verdoppelt das näherungsweise (Temp-Tensoren für Gradienten-Berechnung)
  const H = hiddenSize;
  const L = numLayers;
  const B = cfg.batch_size;
  const S = cfg.max_seq_length;
  const storedLayers = cfg.gradient_checkpointing ? Math.ceil(Math.sqrt(L)) : L;
  const activationBytesPerLayer = 17 * H * B * S * 4; // forward
  const backwardMultiplier = 1.8; // ~1.8x wegen Backward-Temp-Tensoren
  const activationsGb = (activationBytesPerLayer * storedLayers * backwardMultiplier) / 1e9;

  // DataLoader-Worker-Prozesse: jeder Worker lädt Dataset-Shard in eigenen RAM
  // Auf Apple MPS: tokenizerbasierter Deadlock zwingt zu num_workers=0 oder wenig Workers
  const workerRamMb = 450; // ~450 MB pro Python-Worker-Prozess (Basis + Dataset-Shard)
  const workersGb = (cfg.num_workers * workerRamMb) / 1024;

  // Datensatz: HuggingFace lädt Arrow-Format memory-mapped, aber cached pages
  const datasetCacheGb = Math.min(Math.max(datasetSizeBytes * 0.05 / 1e9, 0.2), 2.0);

  // Framework-Overhead:
  //   Python-Interpreter: ~300 MB
  //   PyTorch-Core + Autograd-Graph: ~800 MB
  //   MPS/CUDA-Kontext + Allocator-Reserve: 1.2 GB (MPS hält immer Reserve)
  //   Tokenizer + HuggingFace-Bibliotheken: ~400 MB
  const frameworkGb = isMps ? 3.2 : 2.5;

  // MPS (Apple Unified Memory) zusätzlicher Overhead:
  // MPS und CPU teilen denselben physischen RAM, aber PyTorch auf MPS kopiert
  // Tensoren beim Transfer zusätzlich und hat ein weniger reifes Memory-Management.
  // Empirisch: MPS verbraucht ca. 1.4-1.6x mehr als theoretisch.
  const mpsFactor = isMps ? 1.35 : 1.0;

  const rawTotal = modelGb + gradientsGb + optimizerGb + activationsGb + workersGb + datasetCacheGb + frameworkGb;
  const totalGb = rawTotal * mpsFactor;

  return {
    modelGb: modelGb * mpsFactor,
    gradientsGb: gradientsGb * mpsFactor,
    optimizerGb: optimizerGb * mpsFactor,
    activationsGb: activationsGb * mpsFactor,
    workersGb: workersGb * mpsFactor,
    overheadGb: (datasetCacheGb + frameworkGb) * mpsFactor,
    totalGb,
  };
}

interface RamCalculatorProps {
  config: TrainingConfig;
  datasetSizeBytes: number;
  selectedModelId: string | null;
  primaryColor: string;
  requirements: RequirementsCheck | null;
}

function RamCalculator({ config, datasetSizeBytes, selectedModelId, primaryColor, requirements }: RamCalculatorProps) {
  // Echte Daten vom Rust-Backend
  const [systemRamGb, setSystemRamGb] = useState<number>(16);
  const [systemRamOverride, setSystemRamOverride] = useState<number | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelRamInfo | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [datasetFlash, setDatasetFlash] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  // System-RAM beim Mounten einmalig abrufen
  useEffect(() => {
    invoke<number>('get_system_ram_gb')
      .then(gb => setSystemRamGb(Math.round(gb)))
      .catch(() => setSystemRamGb(16));
  }, []);

  // Modell-Info neu laden wenn sich das ausgewählte Modell ändert
  useEffect(() => {
    if (!selectedModelId) {
      setModelInfo(null);
      return;
    }
    setLoadingModel(true);
    invoke<ModelRamInfo>('get_model_ram_info', { modelId: selectedModelId })
      .then(info => setModelInfo(info))
      .catch(() => setModelInfo(null))
      .finally(() => setLoadingModel(false));
  }, [selectedModelId]);

  // Kurzes Flash-Feedback wenn sich der Datensatz ändert
  const prevDatasetRef = useRef<number>(datasetSizeBytes);
  useEffect(() => {
    if (prevDatasetRef.current !== datasetSizeBytes) {
      prevDatasetRef.current = datasetSizeBytes;
      setDatasetFlash(true);
      const t = setTimeout(() => setDatasetFlash(false), 800);
      return () => clearTimeout(t);
    }
  }, [datasetSizeBytes]);

  const effectiveRam = systemRamOverride ?? systemRamGb;
  const paramBillion = modelInfo?.param_billion ?? 0.35;
  const isMps = requirements?.mps_available ?? detectPlatform() === 'mps';
  const ram = calcRam(
    paramBillion,
    config,
    datasetSizeBytes,
    modelInfo?.hidden_size ?? 768,
    modelInfo?.num_hidden_layers ?? 12,
    isMps,
  );
  const usedPct = Math.min((ram.totalGb / effectiveRam) * 100, 100);

  const status: 'ok' | 'tight' | 'critical' =
    usedPct <= 70 ? 'ok' : usedPct <= 90 ? 'tight' : 'critical';

  const statusColor = {
    ok:       { text: 'text-green-400', border: 'border-green-500/30', badge: 'bg-green-500/20 text-green-300', label: 'Ausreichend' },
    tight:    { text: 'text-amber-400', border: 'border-amber-500/30', badge: 'bg-amber-500/20 text-amber-300', label: 'Knapp' },
    critical: { text: 'text-red-400',   border: 'border-red-500/30',   badge: 'bg-red-500/20 text-red-300',   label: 'Zu wenig' },
  }[status];

  // Alias für config
  const cfg = config;

  // Kontextbewusste Optimierungstipps
  const tips: string[] = [];
  if (status !== 'ok') {
    if (!cfg.use_lora)                tips.push('LoRA aktivieren — Optimizer-RAM sinkt um bis zu 95%');
    if (!cfg.load_in_4bit && cfg.use_lora) tips.push('4-bit QLoRA — halbiert nochmals den Modell-RAM');
    if (!cfg.gradient_checkpointing)  tips.push('Gradient Checkpointing — spart ~60% Aktivierungs-RAM (im Erweitert-Abschnitt)');
    if (cfg.batch_size > 2)           tips.push(`Batch Size auf ${Math.max(1, cfg.batch_size >> 1)} halbieren`);
    if (cfg.max_seq_length > 256)     tips.push(`Sequenzlänge auf ${Math.max(128, cfg.max_seq_length >> 1)} halbieren`);
    if (cfg.gradient_accumulation_steps < 4) tips.push('Gradient Accumulation auf 4–8 erhöhen + Batch auf 1 senken');
  }

  const fmt = (gb: number) => gb < 1 ? `${(gb * 1024).toFixed(0)} MB` : `${gb.toFixed(1)} GB`;

  const segments = [
    { label: 'Modell',        gb: ram.modelGb,       color: 'bg-blue-500' },
    { label: 'Gradienten',   gb: ram.gradientsGb,   color: 'bg-sky-400' },
    { label: 'Optimizer',    gb: ram.optimizerGb,   color: 'bg-purple-500' },
    { label: 'Aktivierungen',gb: ram.activationsGb, color: 'bg-orange-500' },
    { label: 'Workers',      gb: ram.workersGb,     color: 'bg-cyan-500' },
    { label: 'Overhead',     gb: ram.overheadGb,    color: 'bg-slate-500' },
  ];

  return (
    <div className={`bg-white/5 rounded-xl border ${statusColor.border} overflow-hidden`}>
      {/* Header — immer sichtbar */}
      <button
        onClick={() => setIsExpanded(e => !e)}
        className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-all"
      >
        <div className="flex items-center gap-3">
          <MemoryStick className="w-5 h-5 text-blue-400" />
          <span className="font-medium text-white text-sm">RAM-Rechner</span>
          {loadingModel
            ? <Loader2 className="w-3.5 h-3.5 text-gray-400 animate-spin" />
            : <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor.badge}`}>
                {statusColor.label}
              </span>
          }
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold ${statusColor.text}`}>
            {fmt(ram.totalGb)} / {effectiveRam} GB
          </span>
          {isExpanded
            ? <ChevronUp className="w-4 h-4 text-gray-400" />
            : <ChevronDown className="w-4 h-4 text-gray-400" />}
        </div>
      </button>

      {/* RAM-Balken — immer sichtbar */}
      <div className="px-4 pb-3">
        <div className={`w-full h-3 bg-white/10 rounded-full overflow-hidden flex transition-all duration-300 ${
          datasetFlash ? 'ring-2 ring-cyan-400/60' : ''
        }`}>
          {segments.map(s => (
            <div
              key={s.label}
              className={`${s.color} h-full transition-all duration-500`}
              style={{ width: `${Math.min((s.gb / effectiveRam) * 100, 100)}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0 GB</span>
          <div className="flex items-center gap-2">
            {datasetFlash && (
              <span className="text-cyan-400 text-xs animate-pulse">Datensatz aktualisiert</span>
            )}
            <span className={statusColor.text}>{Math.round(usedPct)}% belegt</span>
          </div>
          <span>{effectiveRam} GB</span>
        </div>
      </div>

      {/* Aufgeklappter Detailbereich */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-white/10 pt-4">

          {/* Modell-Info + RAM-Override */}
          <div className="grid grid-cols-2 gap-3">
            {/* Modell-Info (automatisch) */}
            <div className="col-span-1">
              <div className="text-xs text-gray-400 mb-1">Ausgewähltes Modell</div>
              <div className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg">
                {loadingModel ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
                    <span className="text-xs text-gray-400">Lese...</span>
                  </div>
                ) : modelInfo ? (
                  <div>
                    <span className="text-white text-xs font-medium">{modelInfo.readable_size} Params</span>
                    <span className="text-gray-500 text-xs ml-2">({modelInfo.model_type})</span>
                  </div>
                ) : (
                  <span className="text-gray-500 text-xs">Kein Modell gewählt</span>
                )}
              </div>
            </div>

            {/* RAM-Override (falls Auto-Erkennung falsch) */}
            <div>
              <div className="text-xs text-gray-400 mb-1">
                System-RAM
                <span className="ml-1 text-gray-600">(Auto: {systemRamGb} GB)</span>
              </div>
              <select
                value={systemRamOverride ?? systemRamGb}
                onChange={e => {
                  const val = parseInt(e.target.value);
                  setSystemRamOverride(val === systemRamGb ? null : val);
                }}
                className="w-full px-2 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-xs focus:outline-none"
              >
                {SYSTEM_RAM_OPTIONS.map(gb => (
                  <option key={gb} value={gb} className="bg-slate-800">
                    {gb} GB{gb === systemRamGb ? ' ✓' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Aufschlüsselung */}
          <div className="space-y-1.5">
            {segments.map(s => (
              <div key={s.label} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-sm ${s.color}`} />
                  <span className="text-gray-400">{s.label}</span>
                </div>
                <span className="text-white font-mono">{fmt(s.gb)}</span>
              </div>
            ))}
            <div className="border-t border-white/10 pt-1.5 flex justify-between text-xs">
              <span className="text-gray-300 font-medium">Gesamt</span>
              <span className={`font-bold font-mono ${statusColor.text}`}>{fmt(ram.totalGb)}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Verbleibend</span>
              <span className={`font-mono ${
                effectiveRam - ram.totalGb > 2 ? 'text-green-400' : 'text-red-400'
              }`}>
                {fmt(Math.max(0, effectiveRam - ram.totalGb))}
              </span>
            </div>
          </div>

          {/* Hinweise */}
          <div className="text-xs text-gray-500 space-y-1">
            {isMps && (
              <div className="flex items-start gap-2 text-amber-400/80 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2 mb-1">
                <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
                <span><strong>Apple Silicon (MPS):</strong> CPU und GPU teilen denselben RAM. PyTorch reserviert zusätzliche Puffer (+35%). Dieser Schätzwert ist realistischer als reine Theorie.</span>
              </div>
            )}
            <div className="flex items-start gap-2">
              <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <span>{cfg.use_lora
                ? 'LoRA aktiv: Gradienten + Optimizer nur für trainierbare Parameter (~1–5%)'
                : 'Kein LoRA: Gradienten + AdamW-Momentum für alle Parameter'}
              </span>
            </div>
            <div className="flex items-start gap-2">
              <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <span>Aktivierungen basieren auf H={modelInfo?.hidden_size ?? 768}, L={modelInfo?.num_hidden_layers ?? 12}, B={cfg.batch_size}, S={cfg.max_seq_length}. Backward-Pass inclus.</span>
            </div>
            {cfg.gradient_checkpointing && (
              <div className="flex items-center gap-2 text-green-400/70">
                <CheckCircle className="w-3 h-3 flex-shrink-0" />
                <span>Gradient Checkpointing: nur {Math.ceil(Math.sqrt(modelInfo?.num_hidden_layers ?? 12))} Layers gecacht statt {modelInfo?.num_hidden_layers ?? 12}</span>
              </div>
            )}
            <div className="flex items-start gap-2 text-yellow-400/60">
              <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <span>Schätzwert ±20% — echter Verbrauch hängt von Modellarchitektur, Batch-Padding und OS-Caching ab.</span>
            </div>
          </div>

          {/* Optimierungstipps — nur bei Engpass */}
          {tips.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-medium text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                Empfehlungen
              </div>
              {tips.map((tip, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-amber-300/80 bg-amber-500/5 rounded-lg px-3 py-2">
                  <span className="text-amber-400 flex-shrink-0">→</span>
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          )}

          {status === 'ok' && (
            <div className="flex items-center gap-2 text-xs text-green-400/80 bg-green-500/5 rounded-lg px-3 py-2">
              <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
              Genug RAM für dieses Training.
            </div>
          )}
        </div>
      )}
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

export default function TrainingPanel({ onNavigateToAnalysis }: { onNavigateToAnalysis?: (versionId: string | null) => void }) {
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

  // Stderr-Log-Puffer für das Error-Modal
  const stderrLogsRef = useRef<string[]>([]);

  // Sleep Prevention State
  const [preventSleep, setPreventSleep] = useState(false);

  // History State
  const [trainingHistory, setTrainingHistory] = useState<TrainingJob[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);

  // Training Error Modal State
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [trainingErrorTitle, setTrainingErrorTitle] = useState('');
  const [trainingErrorMessage, setTrainingErrorMessage] = useState('');
  const [trainingErrorDetails, setTrainingErrorDetails] = useState('');
  const [trainingErrorLogs, setTrainingErrorLogs] = useState<string[]>([]);
  const [trainingErrorConfigSnapshot, setTrainingErrorConfigSnapshot] = useState('');

  // KI-Assistent State
  const [showAIAssistant, setShowAIAssistant] = useState(false);
  const [aiPrefilledContext, setAiPrefilledContext] = useState<string | null>(null);

  // Post-Training Modal State
  const [showPostTrainingModal, setShowPostTrainingModal] = useState(false);
  const [postTrainingVersionId, setPostTrainingVersionId] = useState<string | null>(null);
  const [postTrainingMetrics, setPostTrainingMetrics] = useState<any>(null);
  const [aiSystemRamGb, setAiSystemRamGb] = useState(16);
  const [mainModelInfo, setMainModelInfo] = useState<ModelRamInfo | null>(null);

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
      // Modell-Info für KI-Assistent laden
      invoke<ModelRamInfo>('get_model_ram_info', { modelId: selectedModelId })
        .then(info => setMainModelInfo(info))
        .catch(() => setMainModelInfo(null));
      
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
            // Logs für Error-Modal puffern (max. 80 Zeilen)
            stderrLogsRef.current = [...stderrLogsRef.current, data.message].slice(-80);
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
          // Sleep-Prevention deaktivieren
          invoke('disable_prevent_sleep').then(() => setPreventSleep(false)).catch(() => {});
          // Post-Training Modal triggern
          if (event.payload?.new_version_id) {
            const d = event.payload.data || event.payload;
            setPostTrainingVersionId(event.payload.new_version_id);
            setPostTrainingMetrics(d?.final_metrics || d || null);
            setShowPostTrainingModal(true);
          }
          loadTrainingHistory();
        })
      );

      unlisteners.push(
        await listen<any>('training-error', (event) => {
          const data = event.payload.data || event.payload;
          const errMsg   = data?.error   || data?.message || 'Unbekannter Fehler';
          const errDetail = data?.details || '';

          // Logs aus dem Puffer nehmen
          const logs = stderrLogsRef.current.slice(-40);
          stderrLogsRef.current = [];

          // Error-Modal öffnen
          setTrainingErrorTitle(errMsg.length > 80 ? errMsg.slice(0, 80) + '…' : errMsg);
          setTrainingErrorMessage(errMsg);
          setTrainingErrorDetails(errDetail);
          setTrainingErrorLogs(logs);
          setTrainingErrorConfigSnapshot(
            JSON.stringify(config, null, 2).slice(0, 4000)
          );
          setShowErrorModal(true);

          setCurrentJob(null);
          setTrainingStatus('');
          setTrainingStartTime(null);
          setTrainingElapsed(0);
          // Sleep-Prevention deaktivieren
          invoke('disable_prevent_sleep').then(() => setPreventSleep(false)).catch(() => {});
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

      // System-RAM auch für KI-Assistenten laden
      invoke<number>('get_system_ram_gb').then(gb => setAiSystemRamGb(Math.round(gb))).catch(() => {});
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

  const togglePreventSleep = async () => {
    try {
      if (preventSleep) {
        await invoke('disable_prevent_sleep');
        setPreventSleep(false);
      } else {
        await invoke('enable_prevent_sleep');
        setPreventSleep(true);
      }
    } catch (err: any) {
      error('Sleep-Prevention Fehler', String(err));
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
      // Aktiviere Sleep-Prevention automatisch beim Trainingsstart
      try {
        await invoke('enable_prevent_sleep');
        setPreventSleep(true);
      } catch (e) {
        console.warn('Sleep-Prevention konnte nicht aktiviert werden:', e);
      }
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
      // Sleep-Prevention deaktivieren
      try { await invoke('disable_prevent_sleep'); setPreventSleep(false); } catch (_) {}
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

          {/* Sleep Prevention Toggle */}
          <button
            onClick={togglePreventSleep}
            title={preventSleep ? 'Sleep-Prevention aktiv – klicken zum Deaktivieren' : 'Computer während Training wachhalten'}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              preventSleep
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white border border-white/10'
            }`}
          >
            <Moon className={`w-4 h-4 ${preventSleep ? 'fill-amber-400' : ''}`} />
            <span className="hidden sm:inline">{preventSleep ? 'Kein Sleep' : 'Sleep'}</span>
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
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Voreinstellungen
              </h2>
              <button
                onClick={() => setShowAIAssistant(true)}
                disabled={isTraining}
                className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/40 rounded-lg text-purple-300 text-xs font-semibold transition-all disabled:opacity-50"
              >
                <span>🤖</span> KI-Assistent
              </button>
            </div>
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
          {/* RAM Calculator */}
          <RamCalculator
            config={config}
            datasetSizeBytes={selectedDataset?.size_bytes ?? 0}
            selectedModelId={selectedModelId}
            primaryColor={currentTheme.colors.primary}
            requirements={requirements}
          />

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

      {showPostTrainingModal && (
        <PostTrainingModal
          versionId={postTrainingVersionId || ''}
          modelName={selectedModel?.name || ''}
          metrics={postTrainingMetrics}
          onClose={() => setShowPostTrainingModal(false)}
          onGoToAnalysis={() => {
            setShowPostTrainingModal(false);
            onNavigateToAnalysis?.(postTrainingVersionId);
          }}
          gradient={currentTheme.colors.gradient}
          primaryColor={currentTheme.colors.primary}
        />
      )}

      {showAIAssistant && (
        <AIAssistantModal
          config={config}
          modelInfo={mainModelInfo}
          selectedModel={selectedModel}
          selectedDataset={selectedDataset}
          systemRamGb={aiSystemRamGb}
          requirements={requirements}
          prefilledContext={aiPrefilledContext}
          onApply={(patch) => {
            setConfig(prev => ({ ...prev, ...patch }));
            setSelectedPresetId(null);
          }}
          onClose={() => setShowAIAssistant(false)}
          gradient={currentTheme.colors.gradient}
          primaryColor={currentTheme.colors.primary}
        />
      )}

      {showErrorModal && (
        <TrainingErrorModal
          errorTitle={trainingErrorTitle}
          errorMessage={trainingErrorMessage}
          errorDetails={trainingErrorDetails}
          errorLogs={trainingErrorLogs}
          configSnapshot={trainingErrorConfigSnapshot}
          onClose={() => setShowErrorModal(false)}
          onOpenAIWithError={(ctx) => {
            setShowErrorModal(false);
            setAiPrefilledContext(ctx);
            setShowAIAssistant(true);
          }}
          gradient={currentTheme.colors.gradient}
          primaryColor={currentTheme.colors.primary}
        />
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