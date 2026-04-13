import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  BarChart3, TrendingDown, Activity, Target, Clock, Layers,
  RefreshCw, Loader2, ChevronDown, AlertCircle, Info, Download, GitBranch,
  CheckCircle, XCircle, FileText, Award, Brain, Sparkles, MessageSquare,
  Send, Trash2, RotateCcw, ChevronUp, Bot, User, Cpu, Database,
  Save, BookOpen, Zap,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

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
  exported_at: string;
  training_summary: Record<string, any>;
  config: Record<string, any>;
  hardware: Record<string, any>;
  model_info: Record<string, any>;
  dataset_info: Record<string, any>;
  epoch_summaries: EpochSummary[];
  step_logs: LogEntry[];
  derived_stats: Record<string, any>;
}
interface AIAnalysisReport { version_id: string; report_text: string; provider: string; model: string; generated_at: string; }
interface ChatMessage { role: 'user' | 'assistant'; content: string; }
interface MetricsTemplate { id: string; name: string; description: string; config: Record<string, any>; created_at: string; source: string; }

// ============ AI Provider ============

type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';
const PROVIDER_META: Record<AIProvider, { label: string; needsKey: boolean; defaultModel: string; endpoint: string; authHeader: (k: string) => Record<string, string>; buildBody: (m: string, msgs: any[], sys: string) => object; extractText: (d: any) => string; }> = {
  anthropic: { label: 'Claude (Anthropic)', needsKey: true, defaultModel: 'claude-sonnet-4-5', endpoint: 'https://api.anthropic.com/v1/messages', authHeader: k => ({ 'x-api-key': k, 'anthropic-version': '2023-06-01' }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, system: sys, messages: msgs }), extractText: d => d.content?.[0]?.text || '' },
  openai: { label: 'GPT-4o (OpenAI)', needsKey: true, defaultModel: 'gpt-4o', endpoint: 'https://api.openai.com/v1/chat/completions', authHeader: k => ({ Authorization: `Bearer ${k}` }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.choices?.[0]?.message?.content || '' },
  groq: { label: 'Groq', needsKey: true, defaultModel: 'llama-3.3-70b-versatile', endpoint: 'https://api.groq.com/openai/v1/chat/completions', authHeader: k => ({ Authorization: `Bearer ${k}` }), buildBody: (m, msgs, sys) => ({ model: m, max_tokens: 6000, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.choices?.[0]?.message?.content || '' },
  ollama: { label: 'Ollama (Lokal)', needsKey: false, defaultModel: 'llama3.2', endpoint: 'http://localhost:11434/api/chat', authHeader: () => ({}), buildBody: (m, msgs, sys) => ({ model: m, stream: false, messages: [{ role: 'system', content: sys }, ...msgs] }), extractText: d => d.message?.content || '' },
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

// ============ Helpers ============
function formatBytes(b: number) { if (!b) return '0 B'; const k = 1024, s = ['B','KB','MB','GB']; const i = Math.floor(Math.log(b)/Math.log(k)); return parseFloat((b/Math.pow(k,i)).toFixed(2))+' '+s[i]; }
function formatDuration(s: number | null) { if (!s) return '-'; const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60); return h>0?`${h}h ${m}m`:m>0?`${m}m ${sec}s`:`${sec}s`; }
function formatDate(d: string) { return new Date(d).toLocaleDateString('de-DE',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'}); }

// ============ Chart Components ============

function SvgChart({ data, color, h = 48 }: { data: number[]; color: string; h?: number }) {
  if (!data.length) return null;
  const max = Math.max(...data), min = Math.min(...data), range = max - min || 1;
  const gx = (i: number) => (i / Math.max(data.length - 1, 1)) * 100;
  const gy = (v: number) => h - ((v - min) / range) * (h - 4) - 2;
  const points = data.map((v, i) => `${gx(i)},${gy(v)}`).join(' ');
  return (
    <svg viewBox={`0 0 100 ${h}`} className="w-full" preserveAspectRatio="none" style={{ height: h * 2 }}>
      <defs><linearGradient id={`g${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="0.3"/><stop offset="100%" stopColor={color} stopOpacity="0.02"/></linearGradient></defs>
      <polygon points={`0,${h} ${points} 100,${h}`} fill={`url(#g${color.replace('#','')})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function DualChart({ trainData, valData, label }: { trainData: number[]; valData: number[]; label: string }) {
  const all = [...trainData, ...valData];
  if (!all.length) return null;
  const max = Math.max(...all), min = Math.min(...all), range = max - min || 1;
  const H = 120;
  const gx = (i: number, total: number) => (i / Math.max(total - 1, 1)) * 100;
  const gy = (v: number) => H - ((v - min) / range) * (H - 8) - 4;
  const tp = trainData.map((v, i) => `${gx(i, trainData.length)},${gy(v)}`).join(' ');
  const vp = valData.map((v, i) => `${gx(i, valData.length)},${gy(v)}`).join(' ');
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">{label}</span>
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-blue-400"/><span>Train</span></div>
          {valData.length > 0 && <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-emerald-400"/><span>Val</span></div>}
        </div>
      </div>
      <svg viewBox={`0 0 100 ${H}`} className="w-full" style={{ height: H * 1.8 }} preserveAspectRatio="none">
        {[0.15, 0.4, 0.65, 0.9].map(p => <line key={p} x1="0" y1={H*p} x2="100" y2={H*p} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5"/>)}
        {tp && <polyline points={tp} fill="none" stroke="#60a5fa" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>}
        {vp && valData.length > 0 && <polyline points={vp} fill="none" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>}
      </svg>
      <div className="flex justify-between text-xs text-gray-600 mt-1"><span>Start</span><span>Ende</span></div>
      <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
        {trainData.length > 0 && <div className="bg-white/5 rounded p-2 text-center"><div className="text-gray-400">Train: {trainData[0]?.toFixed(4)} → {trainData[trainData.length-1]?.toFixed(4)}</div></div>}
        {valData.length > 0 && <div className="bg-white/5 rounded p-2 text-center"><div className="text-gray-400">Val: {valData[0]?.toFixed(4)} → {valData[valData.length-1]?.toFixed(4)}</div></div>}
      </div>
    </div>
  );
}

function EpochChart({ summaries }: { summaries: EpochSummary[] }) {
  if (!summaries.length) return null;
  const trainLosses = summaries.map(s => s.avg_train_loss ?? 0);
  const valLosses = summaries.filter(s => s.val_loss !== null).map(s => s.val_loss!);
  return <DualChart trainData={trainLosses} valData={valLosses} label="Durchschnittlicher Loss pro Epoche" />;
}

function GradNormChart({ logs }: { logs: LogEntry[] }) {
  const norms = logs.filter(l => l.grad_norm != null).map(l => l.grad_norm!);
  if (!norms.length) return null;
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">Gradient Norm</span>
        <span className="text-xs text-gray-400">Max: {Math.max(...norms).toFixed(3)} · Avg: {(norms.reduce((a,b)=>a+b,0)/norms.length).toFixed(3)}</span>
      </div>
      <SvgChart data={norms} color="#f59e0b" h={60} />
    </div>
  );
}

function LrChart({ logs }: { logs: LogEntry[] }) {
  const lrs = logs.filter(l => l.learning_rate > 0).map(l => l.learning_rate);
  if (!lrs.length) return null;
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">Learning Rate Schedule</span>
        <span className="text-xs text-gray-400">{lrs[0]?.toExponential(2)} → {lrs[lrs.length-1]?.toExponential(2)}</span>
      </div>
      <SvgChart data={lrs} color="#a78bfa" h={50} />
    </div>
  );
}

// ============ Report Renderer ============
function ReportText({ text }: { text: string }) {
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {text.split('\n').map((line, i) => {
        if (line.startsWith('## ')) return <h3 key={i} className="text-base font-bold text-white mt-4 mb-1">{line.slice(3)}</h3>;
        if (line.startsWith('# ')) return <h2 key={i} className="text-lg font-bold text-white mt-4 mb-2">{line.slice(2)}</h2>;
        if (line.startsWith('### ')) return <h4 key={i} className="text-sm font-semibold text-purple-300 mt-3 mb-1">{line.slice(4)}</h4>;
        if (line.startsWith('- ') || line.startsWith('* ')) return <div key={i} className="flex items-start gap-2"><span className="text-purple-400 mt-1 flex-shrink-0">•</span><span className="text-gray-300" dangerouslySetInnerHTML={{ __html: line.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>') }}/></div>;
        if (!line.trim()) return <div key={i} className="h-2"/>;
        return <p key={i} className="text-gray-300" dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>') }}/>;
      })}
    </div>
  );
}

// ============ Props ============
interface AnalysisPanelProps { initialVersionId?: string | null; }

// ============ Main ============
export default function AnalysisPanel({ initialVersionId }: AnalysisPanelProps) {
  const { currentTheme } = useTheme();
  const { success, error: notifyError, info } = useNotification();

  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [versionDetails, setVersionDetails] = useState<VersionDetails | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [fullData, setFullData] = useState<FullTrainingData | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // AI
  const [provider, setProvider] = useState<AIProvider>(() => (localStorage.getItem('ft_ai_provider') as AIProvider) || 'ollama');
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('ft_ai_api_key') || '');
  const [aiModel, setAiModel] = useState(() => localStorage.getItem('ft_ai_model') || PROVIDER_META.ollama.defaultModel);
  const [report, setReport] = useState<AIAnalysisReport | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [showProviderSettings, setShowProviderSettings] = useState(false);

  // Chat
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Templates
  const [templates, setTemplates] = useState<MetricsTemplate[]>([]);
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDesc, setTemplateDesc] = useState('');
  const [showTemplates, setShowTemplates] = useState(false);

  useEffect(() => { loadModels(); loadTemplates(); }, []);
  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); return; }
    const m = modelsWithVersions.find(x => x.id === selectedModelId);
    if (!m?.versions.length) { setSelectedVersionId(null); return; }
    setSelectedVersionId([...m.versions].sort((a,b) => b.version_number - a.version_number)[0].id);
  }, [selectedModelId, modelsWithVersions]);
  useEffect(() => {
    if (selectedVersionId) loadAnalysisData();
    else { setMetrics(null); setVersionDetails(null); setLogs([]); setFullData(null); setReport(null); setChatMessages([]); }
  }, [selectedVersionId]);
  useEffect(() => {
    if (initialVersionId && modelsWithVersions.length > 0) {
      for (const m of modelsWithVersions) {
        if (m.versions.some(v => v.id === initialVersionId)) { setSelectedModelId(m.id); setSelectedVersionId(initialVersionId); break; }
      }
    }
  }, [initialVersionId, modelsWithVersions]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  const loadModels = async () => {
    try { setLoading(true); const list = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree'); setModelsWithVersions(list); if (list.length > 0) setSelectedModelId(list[0].id); }
    catch (e: any) { notifyError('Fehler', String(e)); }
    finally { setLoading(false); }
  };
  const loadTemplates = async () => {
    try { setTemplates(await invoke<MetricsTemplate[]>('get_metrics_templates')); }
    catch { setTemplates([]); }
  };
  const loadAnalysisData = async () => {
    if (!selectedVersionId) return;
    setLoadingAnalysis(true);
    try {
      try { setMetrics(await invoke<TrainingMetrics>('get_training_metrics', { versionId: selectedVersionId })); } catch { setMetrics(null); }
      try { setVersionDetails(await invoke<VersionDetails>('get_version_details', { versionId: selectedVersionId })); } catch { setVersionDetails(null); }
      try { setLogs(await invoke<LogEntry[]>('get_training_logs', { versionId: selectedVersionId })); } catch { setLogs([]); }
      try { setFullData(await invoke<FullTrainingData | null>('get_training_full_data', { versionId: selectedVersionId })); } catch { setFullData(null); }
      try {
        const r = await invoke<AIAnalysisReport | null>('get_ai_analysis_report', { versionId: selectedVersionId });
        setReport(r);
        if (r) setChatMessages([{ role: 'assistant', content: r.report_text }]);
        else setChatMessages([]);
      } catch { setReport(null); }
    } finally { setLoadingAnalysis(false); }
  };

  // ── AI Analysis Context ──────────────────────────────────────────────────

  function buildFullContext(): string {
    const modelName = modelsWithVersions.find(m => m.id === selectedModelId)?.name || 'Unbekannt';
    const lines: string[] = [`=== TRAININGSANALYSE: ${modelName} (${versionDetails?.version_name || selectedVersionId}) ===`];

    if (fullData) {
      const s = fullData.training_summary;
      const cfg = fullData.config;
      const hw = fullData.hardware;
      const mi = fullData.model_info;
      const ds = fullData.dataset_info;
      const st = fullData.derived_stats || {};

      lines.push('\n--- TRAINING ZUSAMMENFASSUNG ---');
      lines.push(`Final Train Loss: ${s.final_train_loss ?? 'N/A'}`);
      lines.push(`Final Val Loss: ${s.final_val_loss ?? 'N/A (kein Validierungsdatensatz)'}`);
      lines.push(`Beste Epoche: ${s.best_epoch || 'N/A'} (Best Val Loss: ${s.best_val_loss ?? 'N/A'})`);
      lines.push(`Epochen: ${s.total_epochs} | Steps: ${s.total_steps}`);
      lines.push(`Trainingsdauer: ${formatDuration(s.training_duration_seconds)}`);

      lines.push('\n--- ABGELEITETE STATISTIKEN ---');
      if (st.loss_reduction_pct !== undefined) lines.push(`Loss-Reduktion gesamt: ${st.loss_reduction_pct}%`);
      if (st.initial_train_loss !== undefined) lines.push(`Initialer Loss: ${st.initial_train_loss} → Finaler Loss: ${st.final_train_loss}`);
      if (st.min_train_loss !== undefined) lines.push(`Min. Train Loss: ${st.min_train_loss} | Max. Train Loss: ${st.max_train_loss}`);
      if (st.overfitting_gap_pct !== undefined) lines.push(`Overfitting-Gap (Val-Train): ${st.overfitting_gap_pct}%`);
      if (st.avg_grad_norm !== undefined) lines.push(`Ø Gradient Norm: ${st.avg_grad_norm} | Max: ${st.max_grad_norm}`);
      if (st.initial_lr !== undefined) lines.push(`Learning Rate: ${st.initial_lr} → ${st.final_lr}`);
      lines.push(`Logs erfasst: ${st.total_log_entries || logs.length} Einträge`);

      lines.push('\n--- TRAININGSPARAMETER (CONFIG) ---');
      const cfgKeys = ['epochs','batch_size','gradient_accumulation_steps','learning_rate','optimizer','scheduler','warmup_ratio','weight_decay','use_lora','lora_r','lora_alpha','fp16','bf16','load_in_4bit','max_seq_length','gradient_checkpointing','max_grad_norm'];
      for (const k of cfgKeys) {
        if (cfg[k] !== undefined) lines.push(`${k}: ${cfg[k]}`);
      }

      lines.push('\n--- MODELL & DATASET ---');
      lines.push(`Architektur: ${mi.architecture || 'N/A'} | LoRA aktiv: ${mi.lora_active}`);
      if (mi.trainable_params) lines.push(`Trainierbare Parameter: ${mi.trainable_params?.toLocaleString()} (${mi.trainable_pct}% des Modells)`);
      if (mi.total_params) lines.push(`Gesamt-Parameter: ${mi.total_params?.toLocaleString()}`);
      lines.push(`Datensatz: ${ds.n_train} Trainingssamples | Validierung: ${ds.has_validation ? ds.n_val + ' Samples' : 'NEIN (kein Val-Set!)'}`);
      lines.push(`Max Sequenzlänge: ${ds.max_seq_length}`);

      lines.push('\n--- HARDWARE ---');
      lines.push(`Gerät: ${hw.device?.toUpperCase()} | RAM: ${hw.system_ram_gb} GB`);
      if (hw.gpu_name) lines.push(`GPU: ${hw.gpu_name}`);

      if (fullData.epoch_summaries?.length > 0) {
        lines.push('\n--- EPOCHEN-VERLAUF ---');
        for (const e of fullData.epoch_summaries) {
          lines.push(`Epoche ${e.epoch}: Ø Loss=${e.avg_train_loss?.toFixed(4)} | Min=${e.min_train_loss?.toFixed(4)} | Val=${e.val_loss?.toFixed(4) ?? 'N/A'} | Dauer=${formatDuration(e.duration_seconds)}`);
        }
      }

      if (fullData.step_logs?.length > 0) {
        const sl = fullData.step_logs;
        lines.push('\n--- STEP-VERLAUF (Auszug) ---');
        const sampleSteps = [sl[0], sl[Math.floor(sl.length*0.25)], sl[Math.floor(sl.length*0.5)], sl[Math.floor(sl.length*0.75)], sl[sl.length-1]].filter(Boolean);
        for (const s2 of sampleSteps) {
          lines.push(`Step ${s2.step} (Epoche ${s2.epoch}): loss=${s2.train_loss?.toFixed(4)} lr=${s2.learning_rate?.toExponential(2)}${s2.grad_norm != null ? ` grad_norm=${s2.grad_norm?.toFixed(3)}` : ''}`);
        }
      }
    } else if (metrics) {
      lines.push(`\nFinal Train Loss: ${metrics.final_train_loss} | Epochen: ${metrics.total_epochs} | Steps: ${metrics.total_steps}`);
      lines.push(`Dauer: ${formatDuration(metrics.training_duration_seconds)}`);
      if (logs.length > 0) {
        lines.push(`\nLog-Verlauf (${logs.length} Einträge):`);
        lines.push(`Erster Loss: ${logs[0].train_loss.toFixed(4)} → Letzter: ${logs[logs.length-1].train_loss.toFixed(4)}`);
      }
    }
    return lines.join('\n');
  }

  const ANALYSIS_SYSTEM_PROMPT = `Du bist ein erfahrener Machine Learning Ingenieur und Modell-Training Experte. 
Analysiere die Trainingsdaten präzise, konkret und strukturiert auf Deutsch.
Nutze alle vorhandenen Zahlen und Metriken für deine Analyse.

Deine Analyse MUSS folgende Abschnitte enthalten:

## 🎯 Gesamtbewertung
Bewerte das Training mit einer Note (1-10) und begründe sie mit konkreten Zahlen.

## ✅ Was lief gut
Beziehe dich auf konkrete Werte (z.B. "Loss sank von X auf Y, das entspricht Z% Reduktion").

## ⚠️ Erkannte Probleme & Schwächen
Analysiere: Overfitting, Underfitting, instabile Gradienten, schlechte Konvergenz, fehlende Val-Daten usw.
Nenne immer konkrete Zahlenwerte als Belege.

## 💡 Detaillierte Verbesserungsvorschläge
Für jedes Problem eine konkrete Lösung mit Begründung.

## 🔧 Empfohlene Parameter für das nächste Training
Gib einen KONKRETEN JSON-Block mit den empfohlenen Trainingsparametern an.
Format:
\`\`\`json
{
  "epochs": X,
  "batch_size": X,
  "learning_rate": X,
  ...
}
\`\`\`
Erkläre JEDEN Parameter und warum du ihn so gewählt hast.

## 📊 Prognose
Was erwartest du vom nächsten Training wenn die Vorschläge umgesetzt werden?`;

  const runAIAnalysis = async () => {
    if (!selectedVersionId) return;
    const meta = PROVIDER_META[provider];
    if (meta.needsKey && !apiKey.trim()) { notifyError('API-Key fehlt', `${meta.label}-Key benötigt.`); setShowProviderSettings(true); return; }
    setGeneratingReport(true);
    try {
      const ctx = buildFullContext();
      const text = await callAI(provider, apiKey, aiModel, [{ role: 'user', content: `Analysiere folgendes Training vollständig:\n\n${ctx}` }], ANALYSIS_SYSTEM_PROMPT);
      await invoke('save_ai_analysis_report', { versionId: selectedVersionId, reportText: text, provider, model: aiModel });
      const newReport: AIAnalysisReport = { version_id: selectedVersionId, report_text: text, provider, model: aiModel, generated_at: new Date().toISOString() };
      setReport(newReport);
      setChatMessages([{ role: 'assistant', content: text }]);
      setShowChat(true);
      success('Analyse gespeichert', 'KI-Analyse erfolgreich erstellt.');
    } catch (e: any) { notifyError('Analyse fehlgeschlagen', String(e)); }
    finally { setGeneratingReport(false); }
  };

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading || !report) return;
    const meta = PROVIDER_META[provider];
    if (meta.needsKey && !apiKey.trim()) { notifyError('API-Key fehlt', `${meta.label}-Key benötigt.`); return; }
    const userMsg: ChatMessage = { role: 'user', content: chatInput.trim() };
    const updated = [...chatMessages, userMsg];
    setChatMessages(updated); setChatInput(''); setChatLoading(true);
    try {
      const sys = `${ANALYSIS_SYSTEM_PROMPT}\n\nDu hast bereits diese Analyse erstellt:\n${report.report_text}\n\nVolle Trainingsdaten:\n${buildFullContext()}\n\nBeantworte Folgefragen präzise und hilfreich.`;
      const reply = await callAI(provider, apiKey, aiModel, updated, sys);
      setChatMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (e: any) { setChatMessages(prev => [...prev, { role: 'assistant', content: `❌ Fehler: ${String(e)}` }]); }
    finally { setChatLoading(false); }
  };

  const saveAsTemplate = async () => {
    if (!templateName.trim() || !fullData?.config) return;
    try {
      const tmpl = await invoke<MetricsTemplate>('save_metrics_template', {
        name: templateName, description: templateDesc,
        config: fullData.config, source: 'user',
      });
      setTemplates(prev => [...prev, tmpl]);
      setShowSaveTemplate(false); setTemplateName(''); setTemplateDesc('');
      success('Template gespeichert', templateName);
    } catch (e: any) { notifyError('Fehler', String(e)); }
  };

  const deleteTemplate = async (id: string) => {
    try { await invoke('delete_metrics_template', { templateId: id }); setTemplates(prev => prev.filter(t => t.id !== id)); }
    catch (e: any) { notifyError('Fehler', String(e)); }
  };

  const handleExport = async () => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify({ metrics, logs, fullData, report, exported_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = `training_report_${versionDetails?.version_name || selectedVersionId}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    success('Exportiert', '');
  };

  // ── Render ──────────────────────────────────────────────────────────────

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="w-8 h-8 text-gray-400 animate-spin"/></div>;
  if (!modelsWithVersions.length) return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Trainingsanalyse</h1>
      <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
        <Layers className="w-12 h-12 text-gray-400 mx-auto mb-4 opacity-50"/>
        <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
        <p className="text-gray-400">Trainiere zunächst ein Modell.</p>
      </div>
    </div>
  );

  const trainLosses = logs.map(l => l.train_loss);
  const valLosses = logs.filter(l => l.val_loss !== null).map(l => l.val_loss!);
  const epochSummaries = fullData?.epoch_summaries || [];

  return (
    <div className="space-y-6 pb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div><h1 className="text-3xl font-bold text-white">Trainingsanalyse</h1><p className="text-gray-400 mt-1">Metriken · Graphiken · KI-Auswertung · Templates</p></div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowTemplates(p => !p)} className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 text-sm border border-white/10 transition-all">
            <BookOpen className="w-4 h-4"/><span>Templates ({templates.length})</span>
          </button>
          {metrics && <button onClick={handleExport} className="flex items-center gap-1.5 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 text-sm border border-white/10 transition-all"><Download className="w-4 h-4"/>Export</button>}
          <button onClick={loadAnalysisData} disabled={!selectedVersionId || loadingAnalysis} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-50">
            <RefreshCw className={`w-5 h-5 ${loadingAnalysis ? 'animate-spin' : ''}`}/>
          </button>
        </div>
      </div>

      {/* Templates Panel */}
      {showTemplates && (
        <div className="bg-white/5 rounded-xl border border-white/10 p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><BookOpen className="w-4 h-4"/>Gespeicherte Parameter-Templates</h3>
          {templates.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">Noch keine Templates. Trainiere ein Modell und speichere die Parameter als Template.</p>
          ) : (
            <div className="space-y-2">
              {templates.map(t => (
                <div key={t.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                  <div>
                    <div className="text-white font-medium text-sm">{t.name}</div>
                    {t.description && <div className="text-gray-400 text-xs">{t.description}</div>}
                    <div className="text-gray-500 text-xs">{t.source === 'ai' ? '🤖 KI-Empfehlung' : '👤 Eigenes Template'} · {formatDate(t.created_at)}</div>
                  </div>
                  <button onClick={() => deleteTemplate(t.id)} className="p-1.5 text-gray-500 hover:text-red-400 transition-colors"><Trash2 className="w-3.5 h-3.5"/></button>
                </div>
              ))}
            </div>
          )}
          {fullData?.config && (
            <div className="mt-3 pt-3 border-t border-white/10">
              {!showSaveTemplate ? (
                <button onClick={() => setShowSaveTemplate(true)} className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm hover:opacity-90 transition-all`}>
                  <Save className="w-4 h-4"/>Aktuelle Parameter als Template speichern
                </button>
              ) : (
                <div className="space-y-2">
                  <input value={templateName} onChange={e => setTemplateName(e.target.value)} placeholder="Template-Name" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"/>
                  <input value={templateDesc} onChange={e => setTemplateDesc(e.target.value)} placeholder="Beschreibung (optional)" className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"/>
                  <div className="flex gap-2">
                    <button onClick={saveAsTemplate} disabled={!templateName.trim()} className={`flex-1 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white text-sm disabled:opacity-50`}>Speichern</button>
                    <button onClick={() => setShowSaveTemplate(false)} className="px-4 py-2 bg-white/5 rounded-lg text-gray-300 text-sm">Abbrechen</button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Model & Version */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5">Modell</label>
            <div className="relative"><select value={selectedModelId||''} onChange={e => setSelectedModelId(e.target.value)} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none">{modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-800">{m.name}</option>)}</select><ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"/></div>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1"><GitBranch className="w-3 h-3"/>Version</label>
            <div className="relative"><select value={selectedVersionId||''} onChange={e => setSelectedVersionId(e.target.value)} disabled={!modelsWithVersions.find(m=>m.id===selectedModelId)?.versions.length} className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none disabled:opacity-50">{modelsWithVersions.find(m=>m.id===selectedModelId)?.versions.map(v => <option key={v.id} value={v.id} className="bg-slate-800">{v.is_root?'⭐ ':''}{v.name}{v.is_root?' (Original)':`(v${v.version_number})`}</option>)||<option value="">–</option>}</select><ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"/></div>
          </div>
        </div>
      </div>

      {loadingAnalysis && <div className="flex items-center justify-center py-12"><Loader2 className="w-10 h-10 text-purple-400 animate-spin"/></div>}

      {!loadingAnalysis && selectedVersionId && !metrics && (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-3"/>
          <h3 className="text-lg font-semibold text-white mb-1">Keine Trainingsdaten</h3>
          <p className="text-gray-400 text-sm">Diese Version wurde noch nicht trainiert.</p>
        </div>
      )}

      {!loadingAnalysis && metrics && (
        <div className="space-y-5">
          {/* Metriken-Karten */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { l: 'Final Train Loss', v: metrics.final_train_loss.toFixed(4), s: `Val: ${metrics.final_val_loss?.toFixed(4)||'N/A'}`, i: <TrendingDown className="w-4 h-4 text-blue-400"/>, color: 'text-blue-400' },
              { l: 'Epochen / Steps', v: `${metrics.total_epochs}E`, s: `${metrics.total_steps.toLocaleString()} Steps`, i: <Activity className="w-4 h-4 text-purple-400"/>, color: 'text-purple-400' },
              { l: 'Dauer', v: formatDuration(metrics.training_duration_seconds), s: metrics.best_epoch?`Best: E${metrics.best_epoch}`:'', i: <Clock className="w-4 h-4 text-yellow-400"/>, color: 'text-yellow-400' },
              { l: 'Status', v: 'Fertig', s: formatDate(metrics.created_at), i: <CheckCircle className="w-4 h-4 text-green-400"/>, color: 'text-green-400' },
            ].map((c,i) => (
              <div key={i} className="bg-white/5 rounded-xl border border-white/10 p-4">
                <div className="flex items-center justify-between mb-2">{c.i}<span className="text-xs text-gray-400">{c.l}</span></div>
                <div className={`text-xl font-bold ${c.color}`}>{c.v}</div>
                <div className="text-xs text-gray-500 mt-0.5">{c.s}</div>
              </div>
            ))}
          </div>

          {/* Derived Stats wenn vorhanden */}
          {fullData?.derived_stats && (() => {
            const st = fullData.derived_stats;
            return (
              <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Zap className="w-4 h-4"/>Abgeleitete Statistiken</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  {st.loss_reduction_pct !== undefined && <div className="bg-white/5 rounded-lg p-2.5 text-center"><div className="text-gray-400 mb-0.5">Loss-Reduktion</div><div className="text-white font-bold text-base">{st.loss_reduction_pct}%</div></div>}
                  {st.overfitting_gap_pct !== undefined && <div className={`rounded-lg p-2.5 text-center ${Math.abs(st.overfitting_gap_pct) > 20 ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-white/5'}`}><div className="text-gray-400 mb-0.5">Overfitting-Gap</div><div className={`font-bold text-base ${Math.abs(st.overfitting_gap_pct) > 20 ? 'text-amber-400' : 'text-white'}`}>{st.overfitting_gap_pct}%</div></div>}
                  {st.avg_grad_norm !== undefined && <div className="bg-white/5 rounded-lg p-2.5 text-center"><div className="text-gray-400 mb-0.5">Ø Grad Norm</div><div className="text-white font-bold text-base">{st.avg_grad_norm}</div></div>}
                  {st.total_log_entries !== undefined && <div className="bg-white/5 rounded-lg p-2.5 text-center"><div className="text-gray-400 mb-0.5">Log-Einträge</div><div className="text-white font-bold text-base">{st.total_log_entries}</div></div>}
                </div>
              </div>
            );
          })()}

          {/* Charts */}
          {(trainLosses.length > 0 || epochSummaries.length > 0) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {trainLosses.length > 0 && <DualChart trainData={trainLosses} valData={valLosses} label="Loss-Verlauf (Steps)"/>}
              {epochSummaries.length > 0 && <EpochChart summaries={epochSummaries}/>}
              {logs.length > 0 && <LrChart logs={logs}/>}
              {logs.some(l => l.grad_norm != null) && <GradNormChart logs={logs}/>}
            </div>
          )}

          {/* Epoch Summaries Tabelle */}
          {epochSummaries.length > 0 && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Target className="w-4 h-4"/>Epoch-Zusammenfassung</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr className="text-left text-gray-400 border-b border-white/10">{['Epoche','Ø Train Loss','Min Loss','Val Loss','Dauer','Steps'].map(h=><th key={h} className="pb-2 pr-4">{h}</th>)}</tr></thead>
                  <tbody>
                    {epochSummaries.map((e, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-2 pr-4 text-white font-medium">{e.epoch}</td>
                        <td className="py-2 pr-4 text-blue-400">{e.avg_train_loss?.toFixed(4)||'-'}</td>
                        <td className="py-2 pr-4 text-emerald-400">{e.min_train_loss?.toFixed(4)||'-'}</td>
                        <td className="py-2 pr-4 text-purple-400">{e.val_loss?.toFixed(4)||'N/A'}</td>
                        <td className="py-2 pr-4 text-gray-300">{formatDuration(e.duration_seconds)}</td>
                        <td className="py-2 text-gray-400">{e.steps}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Hardware & Config Info */}
          {fullData && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Cpu className="w-4 h-4"/>Hardware</h3>
                <div className="space-y-1.5 text-xs">
                  {Object.entries(fullData.hardware).map(([k,v]) => v!==null && v!==undefined && <div key={k} className="flex justify-between"><span className="text-gray-400">{k}</span><span className="text-white font-medium">{String(v)}</span></div>)}
                </div>
              </div>
              <div className="bg-white/5 rounded-xl border border-white/10 p-4">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Database className="w-4 h-4"/>Dataset & Modell</h3>
                <div className="space-y-1.5 text-xs">
                  {Object.entries({...fullData.dataset_info, ...fullData.model_info}).map(([k,v]) => v!==null && v!==undefined && <div key={k} className="flex justify-between"><span className="text-gray-400">{k}</span><span className="text-white font-medium">{String(v)}</span></div>)}
                </div>
              </div>
            </div>
          )}

          {/* ══════ KI-ANALYSE ══════ */}
          <div className="bg-gradient-to-br from-purple-500/10 via-blue-500/5 to-transparent rounded-2xl border border-purple-500/20 p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center"><Brain className="w-5 h-5 text-purple-400"/></div>
                <div>
                  <h2 className="text-lg font-bold text-white">KI-Trainingsanalyse</h2>
                  <p className="text-xs text-gray-400">{report ? `${report.provider} · ${report.model} · ${formatDate(report.generated_at)}` : 'Noch keine Analyse'}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setShowProviderSettings(p => !p)} className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-300 border border-white/10 transition-all">
                  {PROVIDER_META[provider].label} {showProviderSettings ? <ChevronUp className="w-3 h-3"/> : <ChevronDown className="w-3 h-3"/>}
                </button>
                {report && <><button onClick={runAIAnalysis} disabled={generatingReport} className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-300 border border-white/10 transition-all disabled:opacity-50"><RotateCcw className="w-3 h-3"/>Neu</button><button onClick={async () => { await invoke('delete_ai_analysis_report', { versionId: selectedVersionId }); setReport(null); setChatMessages([]); setShowChat(false); }} className="p-1.5 bg-white/5 hover:bg-red-500/20 rounded-lg text-gray-400 hover:text-red-400 border border-white/10 transition-all"><Trash2 className="w-3.5 h-3.5"/></button></>}
              </div>
            </div>

            {showProviderSettings && (
              <div className="bg-black/20 rounded-xl p-4 border border-white/10 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div><label className="text-xs text-gray-400 mb-1 block">Anbieter</label>
                    <select value={provider} onChange={e => { const p=e.target.value as AIProvider; setProvider(p); setAiModel(PROVIDER_META[p].defaultModel); localStorage.setItem('ft_ai_provider',p); }} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none">
                      {(Object.entries(PROVIDER_META) as [AIProvider,any][]).map(([k,v]) => <option key={k} value={k} className="bg-slate-800">{v.label}</option>)}
                    </select></div>
                  <div><label className="text-xs text-gray-400 mb-1 block">Modell</label><input value={aiModel} onChange={e => { setAiModel(e.target.value); localStorage.setItem('ft_ai_model',e.target.value); }} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none"/></div>
                </div>
                {PROVIDER_META[provider].needsKey && <div><label className="text-xs text-gray-400 mb-1 block">API-Key</label><input type="password" value={apiKey} onChange={e => { setApiKey(e.target.value); localStorage.setItem('ft_ai_api_key',e.target.value); }} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none font-mono"/></div>}
              </div>
            )}

            {!report && !generatingReport && (
              <button onClick={runAIAnalysis} className={`w-full py-4 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white font-semibold text-base hover:opacity-90 transition-all flex items-center justify-center gap-3 shadow-lg`}>
                <Sparkles className="w-5 h-5"/>KI Analyse starten
              </button>
            )}
            {generatingReport && <div className="flex items-center justify-center gap-3 py-8 text-gray-300"><Loader2 className="w-6 h-6 animate-spin text-purple-400"/>Analysiere mit {PROVIDER_META[provider].label}…</div>}

            {report && !generatingReport && (
              <div className="space-y-4">
                <div className="bg-black/20 rounded-xl p-5 border border-white/10 max-h-[32rem] overflow-y-auto"><ReportText text={report.report_text}/></div>

                {/* Template aus KI-Empfehlung speichern */}
                {fullData?.config && (
                  <button onClick={() => { setTemplateName('KI-Empfehlung'); setTemplateDesc(`Basierend auf Analyse vom ${formatDate(report.generated_at)}`); setShowSaveTemplate(true); setShowTemplates(true); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                    className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-300 hover:text-white transition-all">
                    <Save className="w-4 h-4"/>Parameter als Template speichern
                  </button>
                )}

                <button onClick={() => setShowChat(p => !p)} className="w-full flex items-center justify-center gap-2 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-gray-300 hover:text-white transition-all text-sm font-medium">
                  <MessageSquare className="w-4 h-4"/>{showChat ? 'Chat ausblenden' : 'Mit KI über Analyse chatten'}{showChat ? <ChevronUp className="w-4 h-4"/> : <ChevronDown className="w-4 h-4"/>}
                </button>

                {showChat && (
                  <div className="bg-black/20 rounded-xl border border-white/10 overflow-hidden">
                    <div className="h-80 overflow-y-auto p-4 space-y-3">
                      {chatMessages.slice(1).map((msg, i) => (
                        <div key={i} className={`flex gap-3 ${msg.role==='user' ? 'flex-row-reverse' : ''}`}>
                          <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center ${msg.role==='user' ? 'bg-purple-500/30' : 'bg-blue-500/20'}`}>{msg.role==='user' ? <User className="w-3.5 h-3.5 text-purple-300"/> : <Bot className="w-3.5 h-3.5 text-blue-300"/>}</div>
                          <div className={`max-w-[80%] rounded-xl px-3 py-2 text-sm ${msg.role==='user' ? 'bg-purple-500/20 text-white' : 'bg-white/5 text-gray-300'}`}>{msg.content}</div>
                        </div>
                      ))}
                      {chatLoading && <div className="flex gap-3"><div className="w-7 h-7 rounded-full bg-blue-500/20 flex items-center justify-center"><Bot className="w-3.5 h-3.5 text-blue-300"/></div><div className="bg-white/5 rounded-xl px-3 py-2"><Loader2 className="w-4 h-4 animate-spin text-gray-400"/></div></div>}
                      {chatMessages.length <= 1 && !chatLoading && <div className="text-center text-gray-500 text-sm py-8">Stelle Fragen zur Analyse, zu Parameteroptimierungen oder zum nächsten Training.</div>}
                      <div ref={chatEndRef}/>
                    </div>
                    <div className="border-t border-white/10 p-3 flex gap-2">
                      <input value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();} }} placeholder="Frage stellen…" className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none"/>
                      <button onClick={sendChat} disabled={!chatInput.trim()||chatLoading} className={`p-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white hover:opacity-90 transition-all disabled:opacity-40`}><Send className="w-4 h-4"/></button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Log-Tabelle */}
          {logs.length > 0 && (
            <details className="bg-white/5 rounded-xl border border-white/10">
              <summary className="p-4 cursor-pointer hover:bg-white/5 transition-colors list-none">
                <div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-white flex items-center gap-2"><FileText className="w-4 h-4"/>Training Logs ({logs.length})</h3><ChevronDown className="w-4 h-4 text-gray-400"/></div>
              </summary>
              <div className="p-4 pt-0 max-h-64 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-900"><tr className="text-left text-gray-400 border-b border-white/10">{['E','Step','Train','Val','LR','GradNorm','t'].map(h=><th key={h} className="pb-2 pr-3">{h}</th>)}</tr></thead>
                  <tbody>{logs.map((l,i) => <tr key={i} className="border-b border-white/5 hover:bg-white/5"><td className="py-1.5 pr-3 text-white">{l.epoch}</td><td className="py-1.5 pr-3 text-gray-300">{l.step}</td><td className="py-1.5 pr-3 text-blue-400 font-medium">{l.train_loss.toFixed(4)}</td><td className="py-1.5 pr-3 text-emerald-400">{l.val_loss?.toFixed(4)||'-'}</td><td className="py-1.5 pr-3 text-amber-400 font-mono">{l.learning_rate.toExponential(2)}</td><td className="py-1.5 pr-3 text-purple-400">{l.grad_norm?.toFixed(3)||'-'}</td><td className="py-1.5 text-gray-500">{new Date(l.timestamp).toLocaleTimeString('de-DE')}</td></tr>)}</tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
