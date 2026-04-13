import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  BarChart3, TrendingDown, TrendingUp, Activity, Zap, Target, Clock, Layers,
  RefreshCw, Loader2, ChevronDown, AlertCircle, Info, Download, GitBranch,
  CheckCircle, XCircle, FileText, Award, Brain, Sparkles, MessageSquare,
  Send, Trash2, RotateCcw, ChevronUp, Bot, User,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ============ Types ============

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
interface TrainingMetrics {
  id: string;
  version_id: string;
  final_train_loss: number;
  final_val_loss: number | null;
  total_epochs: number;
  total_steps: number;
  best_epoch: number | null;
  training_duration_seconds: number | null;
  created_at: string;
}
interface LogEntry {
  epoch: number;
  step: number;
  train_loss: number;
  val_loss: number | null;
  learning_rate: number;
  timestamp: string;
}
interface VersionDetails {
  id: string;
  model_id: string;
  version_name: string;
  version_number: number;
  path: string;
  size_bytes: number;
  file_count: number;
  created_at: string;
  is_root: boolean;
  parent_version_id: string | null;
}
interface AIAnalysisReport {
  version_id: string;
  report_text: string;
  provider: string;
  model: string;
  generated_at: string;
}
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ============ AI Provider Config (same keys as TrainingPanel) ============

type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';

const PROVIDER_META: Record<AIProvider, {
  label: string; needsKey: boolean; defaultModel: string;
  endpoint: string; authHeader: (key: string) => Record<string, string>;
  buildBody: (model: string, messages: {role:string;content:string}[], system: string) => object;
  extractText: (data: any) => string;
}> = {
  anthropic: {
    label: 'Claude (Anthropic)', needsKey: true,
    defaultModel: 'claude-sonnet-4-5',
    endpoint: 'https://api.anthropic.com/v1/messages',
    authHeader: (key) => ({ 'x-api-key': key, 'anthropic-version': '2023-06-01' }),
    buildBody: (model, messages, system) => ({ model, max_tokens: 4096, system, messages }),
    extractText: (d) => d.content?.[0]?.text || '',
  },
  openai: {
    label: 'GPT-4o (OpenAI)', needsKey: true,
    defaultModel: 'gpt-4o',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    authHeader: (key) => ({ Authorization: `Bearer ${key}` }),
    buildBody: (model, messages, system) => ({
      model, max_tokens: 4096,
      messages: [{ role: 'system', content: system }, ...messages],
    }),
    extractText: (d) => d.choices?.[0]?.message?.content || '',
  },
  groq: {
    label: 'Groq', needsKey: true,
    defaultModel: 'llama-3.3-70b-versatile',
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    authHeader: (key) => ({ Authorization: `Bearer ${key}` }),
    buildBody: (model, messages, system) => ({
      model, max_tokens: 4096,
      messages: [{ role: 'system', content: system }, ...messages],
    }),
    extractText: (d) => d.choices?.[0]?.message?.content || '',
  },
  ollama: {
    label: 'Ollama (Lokal)', needsKey: false,
    defaultModel: 'llama3.2',
    endpoint: 'http://localhost:11434/api/chat',
    authHeader: () => ({}),
    buildBody: (model, messages, system) => ({
      model, stream: false,
      messages: [{ role: 'system', content: system }, ...messages],
    }),
    extractText: (d) => d.message?.content || '',
  },
};

async function callAI(
  provider: AIProvider, apiKey: string, model: string,
  messages: ChatMessage[], systemPrompt: string,
): Promise<string> {
  const meta = PROVIDER_META[provider];
  const resp = await fetch(meta.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...meta.authHeader(apiKey) },
    body: JSON.stringify(meta.buildBody(model, messages, systemPrompt)),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`API-Fehler (${resp.status}): ${err}`);
  }
  const data = await resp.json();
  const text = meta.extractText(data);
  if (!text) throw new Error('Leere Antwort vom KI-Modell');
  return text;
}

// ============ Helpers ============

function formatBytes(bytes: number) {
  if (!bytes) return '0 B';
  const k = 1024, sizes = ['B','KB','MB','GB','TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
function formatDuration(s: number | null) {
  if (!s) return '-';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function formatDate(d: string) {
  return new Date(d).toLocaleDateString('de-DE', { day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' });
}

// ============ Mini Charts ============

function LossChart({ logs, primaryColor }: { logs: LogEntry[]; primaryColor: string }) {
  if (!logs.length) return null;
  const trainData = logs.map((l, i) => ({ x: i, y: l.train_loss }));
  const valData = logs.filter(l => l.val_loss !== null).map((l, i) => ({ x: i, y: l.val_loss! }));
  const all = [...trainData, ...valData];
  const maxY = Math.max(...all.map(d => d.y));
  const minY = Math.min(...all.map(d => d.y));
  const rangeY = maxY - minY || 1;
  const getX = (i: number) => (i / Math.max(all.length - 1, 1)) * 100;
  const getY = (y: number) => 100 - (((y - minY) / rangeY) * 80 + 10);
  const tPoints = trainData.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');
  const vPoints = valData.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300">Loss-Verlauf</span>
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-blue-400" /><span>Train</span></div>
          {valData.length > 0 && <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-emerald-400" /><span>Val</span></div>}
        </div>
      </div>
      <svg viewBox="0 0 100 100" className="w-full h-40" preserveAspectRatio="none">
        {[10,32,55,77,90].map(y => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(255,255,255,0.07)" strokeWidth="0.5" />)}
        <polyline points={tPoints} fill="none" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {valData.length > 0 && <polyline points={vPoints} fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />}
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Schritt 1</span><span>Schritt {logs.length}</span>
      </div>
    </div>
  );
}

function LrChart({ logs }: { logs: LogEntry[] }) {
  if (!logs.length || !logs.some(l => l.learning_rate > 0)) return null;
  const data = logs.map((l, i) => ({ x: i, y: l.learning_rate }));
  const maxY = Math.max(...data.map(d => d.y));
  const minY = Math.min(...data.map(d => d.y));
  const rangeY = maxY - minY || 1;
  const getX = (i: number) => (i / Math.max(data.length - 1, 1)) * 100;
  const getY = (y: number) => 100 - (((y - minY) / rangeY) * 80 + 10);
  const points = data.map(d => `${getX(d.x)},${getY(d.y)}`).join(' ');
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-300">Learning Rate</span>
        <div className="flex items-center gap-1 text-xs text-gray-400"><div className="w-3 h-0.5 bg-amber-400" /><span>LR</span></div>
      </div>
      <svg viewBox="0 0 100 100" className="w-full h-32" preserveAspectRatio="none">
        {[10,55,90].map(y => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="rgba(255,255,255,0.07)" strokeWidth="0.5" />)}
        <polyline points={points} fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

// ============ Report Renderer ============

function ReportText({ text }: { text: string }) {
  // Simple markdown-like rendering: ##, **, *, bullets
  const lines = text.split('\n');
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (line.startsWith('## ')) return <h3 key={i} className="text-base font-bold text-white mt-4 mb-1">{line.slice(3)}</h3>;
        if (line.startsWith('# ')) return <h2 key={i} className="text-lg font-bold text-white mt-4 mb-2">{line.slice(2)}</h2>;
        if (line.startsWith('### ')) return <h4 key={i} className="text-sm font-semibold text-purple-300 mt-3 mb-1">{line.slice(4)}</h4>;
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <div key={i} className="flex items-start gap-2">
              <span className="text-purple-400 mt-1 flex-shrink-0">•</span>
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

// ============ Props ============

interface AnalysisPanelProps {
  initialVersionId?: string | null;
  onNavigateToTraining?: () => void;
}

// ============ Main Component ============

export default function AnalysisPanel({ initialVersionId, onNavigateToTraining }: AnalysisPanelProps) {
  const { currentTheme } = useTheme();
  const { success, error: notifyError, info } = useNotification();

  // Data
  const [modelsWithVersions, setModelsWithVersions] = useState<ModelWithVersionTree[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [versionDetails, setVersionDetails] = useState<VersionDetails | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // AI Provider (reads same localStorage as TrainingPanel AIAssistantModal)
  const [provider, setProvider] = useState<AIProvider>(
    () => (localStorage.getItem('ft_ai_provider') as AIProvider) || 'ollama'
  );
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('ft_ai_api_key') || '');
  const [aiModel, setAiModel] = useState(() => {
    const saved = localStorage.getItem('ft_ai_model');
    return saved || PROVIDER_META.ollama.defaultModel;
  });

  // AI Analysis State
  const [report, setReport] = useState<AIAnalysisReport | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [showProviderSettings, setShowProviderSettings] = useState(false);

  // Chat State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ============ Load ============

  useEffect(() => { loadModels(); }, []);

  useEffect(() => {
    if (!selectedModelId) { setSelectedVersionId(null); return; }
    const model = modelsWithVersions.find(m => m.id === selectedModelId);
    if (!model?.versions.length) { setSelectedVersionId(null); return; }
    const sorted = [...model.versions].sort((a, b) => b.version_number - a.version_number);
    setSelectedVersionId(sorted[0].id);
  }, [selectedModelId, modelsWithVersions]);

  useEffect(() => {
    if (selectedVersionId) { loadAnalysisData(); }
    else { setMetrics(null); setVersionDetails(null); setLogs([]); setReport(null); setChatMessages([]); }
  }, [selectedVersionId]);

  useEffect(() => {
    if (initialVersionId && modelsWithVersions.length > 0) {
      // Find which model this version belongs to
      for (const m of modelsWithVersions) {
        if (m.versions.some(v => v.id === initialVersionId)) {
          setSelectedModelId(m.id);
          setSelectedVersionId(initialVersionId);
          break;
        }
      }
    }
  }, [initialVersionId, modelsWithVersions]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const loadModels = async () => {
    try {
      setLoading(true);
      const list = await invoke<ModelWithVersionTree[]>('list_models_with_version_tree');
      setModelsWithVersions(list);
      if (list.length > 0) setSelectedModelId(list[0].id);
    } catch (e: any) {
      notifyError('Fehler beim Laden', String(e));
    } finally {
      setLoading(false);
    }
  };

  const loadAnalysisData = async () => {
    if (!selectedVersionId) return;
    setLoadingAnalysis(true);
    try {
      try { setMetrics(await invoke<TrainingMetrics>('get_training_metrics', { versionId: selectedVersionId })); }
      catch { setMetrics(null); }
      try { setVersionDetails(await invoke<VersionDetails>('get_version_details', { versionId: selectedVersionId })); }
      catch { setVersionDetails(null); }
      try { setLogs(await invoke<LogEntry[]>('get_training_logs', { versionId: selectedVersionId })); }
      catch { setLogs([]); }
      try {
        const r = await invoke<AIAnalysisReport | null>('get_ai_analysis_report', { versionId: selectedVersionId });
        setReport(r);
        if (r) setChatMessages([{ role: 'assistant', content: r.report_text }]);
        else setChatMessages([]);
      } catch { setReport(null); setChatMessages([]); }
    } finally {
      setLoadingAnalysis(false);
    }
  };

  // ============ AI Analysis ============

  function buildAnalysisContext() {
    const modelName = modelsWithVersions.find(m => m.id === selectedModelId)?.name || 'Unbekannt';
    const vname = versionDetails?.version_name || selectedVersionId || '';
    const logSummary = logs.length > 0
      ? `Erster Train-Loss: ${logs[0]?.train_loss.toFixed(4)}, Letzter: ${logs[logs.length-1]?.train_loss.toFixed(4)}, Steps: ${logs.length}`
      : 'Keine detaillierten Logs verfügbar';
    const valInfo = metrics?.final_val_loss
      ? `Validation Loss: ${metrics.final_val_loss.toFixed(4)}`
      : 'Kein Validierungsdatensatz';
    const overfitInfo = (() => {
      if (!logs.length || !metrics?.final_val_loss) return '';
      const gap = ((metrics.final_val_loss - metrics.final_train_loss) / metrics.final_train_loss) * 100;
      return ` | Overfitting-Gap: ${gap.toFixed(1)}%`;
    })();
    return `
Modell: ${modelName} | Version: ${vname}
Trainingsergebnisse:
- Final Train Loss: ${metrics?.final_train_loss.toFixed(6)}
- ${valInfo}${overfitInfo}
- Epochen: ${metrics?.total_epochs} | Steps: ${metrics?.total_steps}
- Dauer: ${formatDuration(metrics?.training_duration_seconds || null)}
- Beste Epoche: ${metrics?.best_epoch || 'N/A'}
- Logs: ${logSummary}
${logs.length > 0 ? `\nVerlauf (erste 5): ${logs.slice(0,5).map(l=>`E${l.epoch}/S${l.step}: loss=${l.train_loss.toFixed(4)}`).join(', ')}` : ''}
${logs.length > 5 ? `Verlauf (letzte 5): ${logs.slice(-5).map(l=>`E${l.epoch}/S${l.step}: loss=${l.train_loss.toFixed(4)}`).join(', ')}` : ''}
    `.trim();
  }

  const ANALYSIS_SYSTEM_PROMPT = `Du bist ein Experte für Machine Learning und Modell-Training. 
Analysiere die Trainingsdaten präzise und strukturiert auf Deutsch.
Deine Analyse soll folgende Abschnitte enthalten:
## 🎯 Gesamtbewertung
## ✅ Was lief gut
## ⚠️ Probleme & Schwächen  
## 💡 Verbesserungsvorschläge
## 🔧 Empfohlene Parameter für das nächste Training

Sei konkret, nenne echte Zahlenwerte aus den Daten, und gib klare Handlungsempfehlungen.`;

  const runAIAnalysis = async (regenerate = false) => {
    if (!selectedVersionId || !metrics) return;
    const meta = PROVIDER_META[provider];
    if (meta.needsKey && !apiKey.trim()) {
      notifyError('API-Key fehlt', `Bitte trage deinen ${meta.label}-Key in den Einstellungen ein.`);
      setShowProviderSettings(true);
      return;
    }
    setGeneratingReport(true);
    try {
      const ctx = buildAnalysisContext();
      const text = await callAI(
        provider, apiKey, aiModel,
        [{ role: 'user', content: `Analysiere folgendes Training:\n\n${ctx}` }],
        ANALYSIS_SYSTEM_PROMPT,
      );
      const newReport: AIAnalysisReport = {
        version_id: selectedVersionId,
        report_text: text,
        provider,
        model: aiModel,
        generated_at: new Date().toISOString(),
      };
      await invoke('save_ai_analysis_report', {
        versionId: selectedVersionId,
        reportText: text,
        provider,
        model: aiModel,
      });
      setReport(newReport);
      setChatMessages([{ role: 'assistant', content: text }]);
      setShowChat(true);
      success('Analyse gespeichert', 'KI-Analyse wurde erstellt und gespeichert.');
    } catch (e: any) {
      notifyError('Analyse fehlgeschlagen', String(e));
    } finally {
      setGeneratingReport(false);
    }
  };

  const deleteReport = async () => {
    if (!selectedVersionId) return;
    try {
      await invoke('delete_ai_analysis_report', { versionId: selectedVersionId });
      setReport(null);
      setChatMessages([]);
      setShowChat(false);
      info('Analyse gelöscht', '');
    } catch (e: any) {
      notifyError('Fehler', String(e));
    }
  };

  // ============ Chat ============

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading || !report) return;
    const meta = PROVIDER_META[provider];
    if (meta.needsKey && !apiKey.trim()) {
      notifyError('API-Key fehlt', `${meta.label}-Key benötigt.`);
      return;
    }
    const userMsg: ChatMessage = { role: 'user', content: chatInput.trim() };
    const updatedMessages = [...chatMessages, userMsg];
    setChatMessages(updatedMessages);
    setChatInput('');
    setChatLoading(true);
    try {
      const ctx = buildAnalysisContext();
      const systemPrompt = `${ANALYSIS_SYSTEM_PROMPT}\n\nDu hast bereits folgende Analyse erstellt:\n${report.report_text}\n\nTrainingsdaten zur Referenz:\n${ctx}\n\nBeantworte nun Folgefragen des Users konkret und hilfreich.`;
      const reply = await callAI(provider, apiKey, aiModel, updatedMessages, systemPrompt);
      setChatMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (e: any) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: `❌ Fehler: ${String(e)}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleExport = async () => {
    if (!metrics || !selectedVersionId) return;
    const blob = new Blob([JSON.stringify({ metrics, logs, report, version: versionDetails, exported_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url;
    a.download = `training_report_${versionDetails?.version_name || selectedVersionId}.json`;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
    success('Report exportiert', '');
  };

  // ============ Render ============

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="w-8 h-8 text-gray-400 animate-spin" /></div>;

  if (!modelsWithVersions.length) return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Analyse</h1>
      <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
        <Layers className="w-12 h-12 text-gray-400 mx-auto mb-4 opacity-50" />
        <h3 className="text-xl font-semibold text-white mb-2">Kein Modell vorhanden</h3>
        <p className="text-gray-400">Trainiere zunächst ein Modell, um die Analyse zu nutzen.</p>
      </div>
    </div>
  );

  const selectedModel = modelsWithVersions.find(m => m.id === selectedModelId);
  const selectedVersion = selectedModel?.versions.find(v => v.id === selectedVersionId);

  return (
    <div className="space-y-6 pb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Trainingsanalyse</h1>
          <p className="text-gray-400 mt-1">Metriken, Graphiken & KI-gestützte Auswertung</p>
        </div>
        <div className="flex items-center gap-2">
          {metrics && (
            <button onClick={handleExport} className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 hover:text-white transition-all text-sm border border-white/10">
              <Download className="w-4 h-4" />Export
            </button>
          )}
          <button onClick={loadAnalysisData} disabled={!selectedVersionId || loadingAnalysis}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-all disabled:opacity-50">
            <RefreshCw className={`w-5 h-5 ${loadingAnalysis ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Model & Version Selection */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-5">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5">Modell</label>
            <div className="relative">
              <select value={selectedModelId || ''} onChange={e => setSelectedModelId(e.target.value)}
                className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none">
                {modelsWithVersions.map(m => <option key={m.id} value={m.id} className="bg-slate-800">{m.name}</option>)}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1.5 flex items-center gap-1"><GitBranch className="w-3 h-3" />Version</label>
            <div className="relative">
              <select value={selectedVersionId || ''} onChange={e => setSelectedVersionId(e.target.value)}
                disabled={!selectedModel?.versions.length}
                className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm appearance-none focus:outline-none disabled:opacity-50">
                {selectedModel?.versions.map(v => (
                  <option key={v.id} value={v.id} className="bg-slate-800">
                    {v.is_root ? '⭐ ' : ''}{v.name}{v.is_root ? ' (Original)' : ` (v${v.version_number})`}
                  </option>
                )) || <option value="">Keine Versionen</option>}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {loadingAnalysis && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-10 h-10 text-purple-400 animate-spin" />
        </div>
      )}

      {!loadingAnalysis && selectedVersionId && !metrics && (
        <div className="bg-white/5 rounded-2xl border border-white/10 p-12 text-center">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-white mb-1">Keine Trainingsdaten</h3>
          <p className="text-gray-400 text-sm">Diese Version wurde noch nicht trainiert.</p>
        </div>
      )}

      {!loadingAnalysis && metrics && (
        <div className="space-y-6">
          {/* ── Metriken-Karten ── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Final Train Loss', value: metrics.final_train_loss.toFixed(4), sub: `Val: ${metrics.final_val_loss?.toFixed(4) || 'N/A'}`, icon: <TrendingDown className="w-5 h-5 text-blue-400" /> },
              { label: 'Epochen', value: String(metrics.total_epochs), sub: `${metrics.total_steps.toLocaleString()} Steps`, icon: <Activity className="w-5 h-5 text-purple-400" /> },
              { label: 'Dauer', value: formatDuration(metrics.training_duration_seconds), sub: metrics.best_epoch ? `Best: Epoch ${metrics.best_epoch}` : '', icon: <Clock className="w-5 h-5 text-yellow-400" /> },
              { label: 'Status', value: 'Abgeschlossen', sub: formatDate(metrics.created_at), icon: <CheckCircle className="w-5 h-5 text-green-400" /> },
            ].map((c, i) => (
              <div key={i} className="bg-white/5 rounded-xl border border-white/10 p-4">
                <div className="flex items-center justify-between mb-2">{c.icon}<span className="text-xs text-gray-400">{c.label}</span></div>
                <div className="text-xl font-bold text-white">{c.value}</div>
                <div className="text-xs text-gray-500 mt-0.5">{c.sub}</div>
              </div>
            ))}
          </div>

          {/* ── Graphiken ── */}
          {logs.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <LossChart logs={logs} primaryColor={currentTheme.colors.primary} />
              <LrChart logs={logs} />
            </div>
          )}

          {/* ── Performance ── */}
          {logs.length > 0 && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2"><Target className="w-4 h-4" />Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="bg-white/5 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-400 mb-1">Loss-Reduktion</div>
                  <div className="text-lg font-bold text-white">{(((logs[0].train_loss - logs[logs.length-1].train_loss) / logs[0].train_loss) * 100).toFixed(1)}%</div>
                </div>
                {metrics.final_val_loss && (
                  <div className="bg-white/5 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-400 mb-1">Overfitting-Gap</div>
                    <div className="text-lg font-bold text-white">{(((metrics.final_val_loss - metrics.final_train_loss) / metrics.final_train_loss)*100).toFixed(1)}%</div>
                  </div>
                )}
                <div className="bg-white/5 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-400 mb-1">Log-Einträge</div>
                  <div className="text-lg font-bold text-white">{logs.length}</div>
                </div>
                {metrics.training_duration_seconds && metrics.total_steps && (
                  <div className="bg-white/5 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-400 mb-1">Steps/Sek</div>
                    <div className="text-lg font-bold text-white">{(metrics.total_steps / metrics.training_duration_seconds).toFixed(2)}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Versionsinfo ── */}
          {versionDetails && (
            <div className="bg-white/5 rounded-xl border border-white/10 p-5">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2"><Info className="w-4 h-4" />Version</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                {[
                  ['Name', versionDetails.version_name],
                  ['Größe', formatBytes(versionDetails.size_bytes)],
                  ['Dateien', String(versionDetails.file_count)],
                  ['Erstellt', formatDate(versionDetails.created_at)],
                ].map(([k, v]) => (
                  <div key={k}><div className="text-xs text-gray-400 mb-0.5">{k}</div><div className="text-white font-medium">{v}</div></div>
                ))}
              </div>
              <div className="mt-3 p-2 bg-white/5 rounded-lg text-xs text-gray-400 font-mono truncate" title={versionDetails.path}>{versionDetails.path}</div>
            </div>
          )}

          {/* ══════════════════ KI-ANALYSE SEKTION ══════════════════ */}
          <div className="bg-gradient-to-br from-purple-500/10 via-blue-500/5 to-transparent rounded-2xl border border-purple-500/20 p-6 space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">KI-Trainingsanalyse</h2>
                  <p className="text-xs text-gray-400">{report ? `Erstellt von ${report.provider} · ${report.model} · ${formatDate(report.generated_at)}` : 'Noch keine Analyse vorhanden'}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setShowProviderSettings(p => !p)}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-300 border border-white/10 transition-all">
                  {PROVIDER_META[provider].label} {showProviderSettings ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
                {report && (
                  <>
                    <button onClick={() => runAIAnalysis(true)} disabled={generatingReport}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-300 border border-white/10 transition-all disabled:opacity-50">
                      <RotateCcw className="w-3 h-3" />Neu
                    </button>
                    <button onClick={deleteReport}
                      className="p-1.5 bg-white/5 hover:bg-red-500/20 rounded-lg text-gray-400 hover:text-red-400 border border-white/10 transition-all">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Provider Settings */}
            {showProviderSettings && (
              <div className="bg-black/20 rounded-xl p-4 border border-white/10 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-400 mb-1 block">Anbieter</label>
                    <select value={provider} onChange={e => {
                      const p = e.target.value as AIProvider;
                      setProvider(p);
                      setAiModel(PROVIDER_META[p].defaultModel);
                      localStorage.setItem('ft_ai_provider', p);
                    }} className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none">
                      {(Object.entries(PROVIDER_META) as [AIProvider, any][]).map(([k, v]) => (
                        <option key={k} value={k} className="bg-slate-800">{v.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 mb-1 block">Modell</label>
                    <input value={aiModel} onChange={e => { setAiModel(e.target.value); localStorage.setItem('ft_ai_model', e.target.value); }}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none" />
                  </div>
                </div>
                {PROVIDER_META[provider].needsKey && (
                  <div>
                    <label className="text-xs text-gray-400 mb-1 block">API-Key</label>
                    <input type="password" value={apiKey} onChange={e => { setApiKey(e.target.value); localStorage.setItem('ft_ai_api_key', e.target.value); }}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none font-mono" />
                  </div>
                )}
              </div>
            )}

            {/* Start Button (no report yet) */}
            {!report && !generatingReport && (
              <button onClick={() => runAIAnalysis(false)}
                className={`w-full py-4 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white font-semibold text-base hover:opacity-90 transition-all flex items-center justify-center gap-3 shadow-lg`}>
                <Sparkles className="w-5 h-5" />
                KI Analyse starten
              </button>
            )}

            {/* Generating */}
            {generatingReport && (
              <div className="flex items-center justify-center gap-3 py-8 text-gray-300">
                <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                <span>Analysiere Trainingsdaten mit {PROVIDER_META[provider].label}…</span>
              </div>
            )}

            {/* Report */}
            {report && !generatingReport && (
              <div className="space-y-4">
                <div className="bg-black/20 rounded-xl p-5 border border-white/10 max-h-96 overflow-y-auto">
                  <ReportText text={report.report_text} />
                </div>

                {/* Chat Toggle */}
                <button onClick={() => setShowChat(p => !p)}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-gray-300 hover:text-white transition-all text-sm font-medium">
                  <MessageSquare className="w-4 h-4" />
                  {showChat ? 'Chat ausblenden' : 'Mit KI über Analyse chatten'}
                  {showChat ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {/* Chat */}
                {showChat && (
                  <div className="bg-black/20 rounded-xl border border-white/10 overflow-hidden">
                    {/* Messages */}
                    <div className="h-72 overflow-y-auto p-4 space-y-3">
                      {chatMessages.slice(1).map((msg, i) => (
                        <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                          <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center ${msg.role === 'user' ? 'bg-purple-500/30' : 'bg-blue-500/20'}`}>
                            {msg.role === 'user' ? <User className="w-3.5 h-3.5 text-purple-300" /> : <Bot className="w-3.5 h-3.5 text-blue-300" />}
                          </div>
                          <div className={`max-w-[80%] rounded-xl px-3 py-2 text-sm ${msg.role === 'user' ? 'bg-purple-500/20 text-white' : 'bg-white/5 text-gray-300'}`}>
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
                      {chatMessages.length <= 1 && !chatLoading && (
                        <div className="text-center text-gray-500 text-sm py-8">
                          Stelle Fragen zur Analyse, zu Parameteroptimierungen oder zum nächsten Training.
                        </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>
                    {/* Input */}
                    <div className="border-t border-white/10 p-3 flex gap-2">
                      <input value={chatInput} onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                        placeholder="Frage zur Analyse stellen…"
                        className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none" />
                      <button onClick={sendChat} disabled={!chatInput.trim() || chatLoading}
                        className={`p-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-lg text-white hover:opacity-90 transition-all disabled:opacity-40`}>
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Log-Tabelle ── */}
          {logs.length > 0 && (
            <details className="bg-white/5 rounded-xl border border-white/10">
              <summary className="p-5 cursor-pointer hover:bg-white/5 transition-colors list-none">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2"><FileText className="w-4 h-4" />Training Logs ({logs.length})</h3>
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </div>
              </summary>
              <div className="p-5 pt-0 max-h-72 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-900">
                    <tr className="text-left text-gray-400 border-b border-white/10">
                      {['Epoch','Step','Train Loss','Val Loss','LR','Zeit'].map(h => <th key={h} className="pb-2 pr-3">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((l, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-1.5 pr-3 text-white">{l.epoch}</td>
                        <td className="py-1.5 pr-3 text-gray-300">{l.step}</td>
                        <td className="py-1.5 pr-3 text-blue-400 font-medium">{l.train_loss.toFixed(4)}</td>
                        <td className="py-1.5 pr-3 text-emerald-400">{l.val_loss?.toFixed(4) || '-'}</td>
                        <td className="py-1.5 pr-3 text-amber-400 font-mono">{l.learning_rate.toExponential(2)}</td>
                        <td className="py-1.5 text-gray-500">{new Date(l.timestamp).toLocaleTimeString('de-DE')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
