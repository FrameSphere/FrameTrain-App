// HomePanel.tsx – Startseite der App.
//
// Landepunkt nach dem Login: beantwortet die Frage "Was ist passiert, während
// ich weg war?" und führt von dort per Deeplink in die eigentlichen Arbeits-
// bereiche. Bewusst KEIN zweites Menü — jede Kachel ist ein Sprung mit Kontext.
//
// Alle Daten kommen aus bereits vorhandenen Backend-Commands; die Seite fügt
// dem Backend nichts hinzu, sie aggregiert nur.

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Home,
  Layers,
  Upload,
  Play,
  FlaskConical,
  GitBranch,
  RefreshCw,
  CheckCircle,
  XCircle,
  MinusCircle,
  Loader2,
  Clock,
  ArrowRight,
  Activity,
  Sparkles,
  AlertTriangle,
  Info,
  TrendingDown,
  TrendingUp,
  Trophy,
  HardDrive,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { usePageContext } from '../contexts/PageContext';
import { navigateTo, type AppView } from '../ui/navigationEvents';
import { dateLocale } from '../utils/dateLocale';
import { formatBytes } from '../utils/formatBytes';
import HomeBriefing from './HomeBriefing';
import {
  buildInsights, lossTrend, trendImprovementPct, sparklinePoints, topResults,
  buildBriefingFacts, factsHash, type InsightInput,
} from './homeInsights';

// ============ Types (Spiegel der Rust-Structs) ============

interface TrainingJob {
  id: string;
  model_id?: string;
  model_name: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  progress?: {
    epoch: number;
    total_epochs: number;
    step: number;
    total_steps: number;
    train_loss: number;
    val_loss?: number | null;
    progress_percent: number;
  };
  error?: string | null;
}

interface TestJob {
  id: string;
  model_id?: string;
  model_name: string;
  version_name: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  completed_at?: string | null;
  results?: {
    total_samples: number;
    accuracy?: number | null;
    average_loss?: number | null;
  } | null;
}

interface ActiveTraining {
  training_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  progress_percentage: number;
  train_loss: number;
  elapsed_time_seconds: number;
}

interface ModelWithVersions {
  id: string;
  name: string;
  version_count: number;
  total_size: number;
}

interface DatasetInfo {
  id: string;
  name: string;
}

interface HomePanelProps {
  userEmail: string;
  userId: string;
}

// ============ Helpers ============

const LAST_SEEN_PREFIX = 'ft_home_last_seen_';

/** Sicherer Date-Parse — das Backend liefert teils ISO, teils null. */
function parseDate(value?: string | null): Date | null {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Ende-Zeitpunkt eines Jobs (completed_at, sonst created_at als Näherung). */
function jobEndedAt(job: { completed_at?: string | null; created_at: string }): Date | null {
  return parseDate(job.completed_at) ?? parseDate(job.created_at);
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '–';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export default function HomePanel({ userEmail, userId }: HomePanelProps) {
  const { currentTheme } = useTheme();
  const { t, language } = useLanguage();
  const { setCurrentPageContent } = usePageContext();

  const [loading, setLoading] = useState(true);
  const [trainings, setTrainings] = useState<TrainingJob[]>([]);
  const [tests, setTests] = useState<TestJob[]>([]);
  const [active, setActive] = useState<ActiveTraining[]>([]);
  const [models, setModels] = useState<ModelWithVersions[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [sleepPrevented, setSleepPrevented] = useState(true);

  // Zeitpunkt des letzten Besuchs — einmalig beim Mount gelesen und danach
  // eingefroren, damit die "seit deinem letzten Besuch"-Liste nicht verschwindet
  // während man sie liest. Zurückgeschrieben wird erst beim Verlassen der Seite.
  const lastSeenRef = useRef<Date | null>(null);
  if (lastSeenRef.current === null) {
    const stored = localStorage.getItem(LAST_SEEN_PREFIX + userId);
    lastSeenRef.current = parseDate(stored);
  }
  const lastSeen = lastSeenRef.current;

  useEffect(() => {
    const key = LAST_SEEN_PREFIX + userId;
    const stamp = () => localStorage.setItem(key, new Date().toISOString());
    window.addEventListener('beforeunload', stamp);
    return () => {
      window.removeEventListener('beforeunload', stamp);
      stamp();
    };
  }, [userId]);

  const load = useCallback(async () => {
    setLoading(true);
    const [th, sh, at, ml, ds, sp] = await Promise.allSettled([
      invoke<TrainingJob[]>('get_training_history'),
      invoke<TestJob[]>('get_test_history'),
      invoke<ActiveTraining[]>('list_active_trainings'),
      invoke<ModelWithVersions[]>('list_models_with_versions'),
      invoke<DatasetInfo[]>('list_all_datasets'),
      invoke<boolean>('get_prevent_sleep_status'),
    ]);
    // Einzelne fehlschlagende Quellen dürfen die ganze Startseite nicht leeren.
    setTrainings(th.status === 'fulfilled' ? th.value : []);
    setTests(sh.status === 'fulfilled' ? sh.value : []);
    setActive(at.status === 'fulfilled' ? at.value : []);
    setModels(ml.status === 'fulfilled' ? ml.value : []);
    setDatasets(ds.status === 'fulfilled' ? ds.value : []);
    // Im Zweifel KEINE Warnung: ein nicht lesbarer Status ist kein Befund.
    setSleepPrevented(sp.status === 'fulfilled' ? sp.value : true);
    setLoading(false);
  }, []);

  useEffect(() => { void load(); }, [load]);

  // ── Abgeleitete Werte ────────────────────────────────────────────────────
  const sortedTrainings = [...trainings].sort((a, b) => {
    const da = jobEndedAt(a)?.getTime() ?? 0;
    const db = jobEndedAt(b)?.getTime() ?? 0;
    return db - da;
  });
  const sortedTests = [...tests].sort((a, b) => {
    const da = jobEndedAt(a)?.getTime() ?? 0;
    const db = jobEndedAt(b)?.getTime() ?? 0;
    return db - da;
  });

  const sinceLastVisit = lastSeen
    ? {
        completed: sortedTrainings.filter(j => j.status === 'completed' && (jobEndedAt(j)?.getTime() ?? 0) > lastSeen.getTime()),
        failed:    sortedTrainings.filter(j => (j.status === 'failed' || j.status === 'stopped') && (jobEndedAt(j)?.getTime() ?? 0) > lastSeen.getTime()),
        tested:    sortedTests.filter(j => j.status === 'completed' && (jobEndedAt(j)?.getTime() ?? 0) > lastSeen.getTime()),
      }
    : null;
  const hasNews = !!sinceLastVisit &&
    (sinceLastVisit.completed.length + sinceLastVisit.failed.length + sinceLastVisit.tested.length) > 0;

  const versionCount = models.reduce((sum, m) => sum + (m.version_count || 0), 0);

  const totalSize = models.reduce((sum, m) => sum + (m.total_size || 0), 0);

  // Frischer Account: statt vier Nullen lieber sagen, wo es losgeht.
  const isFirstRun = models.length === 0 && datasets.length === 0 && trainings.length === 0 && tests.length === 0;

  const insightInput: InsightInput = {
    trainings, tests, models, datasets,
    hasActiveTraining: active.length > 0,
    sleepPrevented,
  };
  const insights = buildInsights(insightInput);
  const trend = lossTrend(trainings);
  const trendPct = trendImprovementPct(trend);
  const ranking = topResults(tests);
  const briefingFacts = buildBriefingFacts(insightInput);

  // ── Formatierung ─────────────────────────────────────────────────────────
  const locale = dateLocale(language);

  const formatDateTime = (value?: string | null): string => {
    const d = parseDate(value);
    if (!d) return '–';
    return d.toLocaleString(locale, { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  };

  /** Grobe Relativzeit ("vor 9 Std."), bewusst ohne zusätzliche Abhängigkeit. */
  const formatRelative = (d: Date): string => {
    const diffMin = Math.round((Date.now() - d.getTime()) / 60000);
    if (diffMin < 1) return t('home.relative.now');
    if (diffMin < 60) return t('home.relative.minutes', { n: String(diffMin) });
    const diffH = Math.round(diffMin / 60);
    if (diffH < 24) return t('home.relative.hours', { n: String(diffH) });
    const diffD = Math.round(diffH / 24);
    return t('home.relative.days', { n: String(diffD) });
  };

  const greeting = (): string => {
    const h = new Date().getHours();
    if (h < 11) return t('home.greeting.morning');
    if (h < 18) return t('home.greeting.day');
    return t('home.greeting.evening');
  };

  // Anrede aus der Mailadresse ableiten ("karol@…" → "Karol").
  const rawName = userEmail.includes('@') ? userEmail.split('@')[0] : userEmail;
  const displayName = rawName ? rawName.charAt(0).toUpperCase() + rawName.slice(1) : rawName;

  // ── Seiten-Kontext für den AI-Coach ──────────────────────────────────────
  useEffect(() => {
    const lines = [
      'SEITE: Start',
      `Modelle: ${models.length}, Versionen: ${versionCount}, Datasets: ${datasets.length}, Trainings gesamt: ${trainings.length}`,
      `Laufende Trainings: ${active.length}`,
      `Letztes Training: ${sortedTrainings[0] ? `${sortedTrainings[0].model_name} (${sortedTrainings[0].status})` : 'keins'}`,
      `Letzter Test: ${sortedTests[0] ? `${sortedTests[0].model_name} (${sortedTests[0].status})` : 'keiner'}`,
      `Bestes Testergebnis: ${ranking[0] ? `${ranking[0].model_name} ${ranking[0].version_name} — ${(ranking[0].accuracy * 100).toFixed(1)}%` : 'noch keins'}`,
      `Loss-Trend: ${trendPct !== null ? `${trendPct >= 0 ? '-' : '+'}${Math.abs(trendPct).toFixed(1)}% ueber ${trend.length} Laeufe` : 'zu wenig Daten'}`,
      `Offene Hinweise: ${insights.length > 0 ? insights.map(i => i.kind).join(', ') : 'keine'}`,
    ];
    setCurrentPageContent(lines.join('\n'), 'home');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, datasets, trainings, tests, active, setCurrentPageContent]);

  // ── Bausteine ────────────────────────────────────────────────────────────
  /** Gezaehlte Hinweise brauchen eine Singularform ("1 Modelle" liest sich falsch). */
  const insightKey = (insight: { kind: string; params: Record<string, string> }) =>
    insight.params.count === '1'
      ? `home.insights.${insight.kind}_one`
      : `home.insights.${insight.kind}`;

  const statusVisual = (status: string) => {
    switch (status) {
      case 'completed': return { Icon: CheckCircle, cls: 'text-emerald-400' };
      case 'failed':    return { Icon: XCircle,     cls: 'text-red-400' };
      case 'stopped':   return { Icon: MinusCircle, cls: 'text-amber-400' };
      case 'running':
      case 'pending':   return { Icon: Loader2,     cls: 'text-blue-400 animate-spin' };
      default:          return { Icon: MinusCircle, cls: 'text-gray-500' };
    }
  };

  const StatTile = ({ icon: Icon, value, label, view, note }: {
    icon: typeof Layers; value: number; label: string; view: AppView; note?: string;
  }) => (
    <button
      onClick={() => navigateTo(view)}
      className="rounded-2xl border border-white/10 bg-white/5 p-5 text-left hover:bg-white/[0.09] hover:border-white/20 transition-all group"
    >
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient}`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        <ArrowRight className="w-4 h-4 text-gray-600 group-hover:text-gray-300 group-hover:translate-x-0.5 transition-all" />
      </div>
      <div className="text-2xl font-bold text-white tabular-nums">{value}</div>
      <div className="text-xs text-gray-400 mt-0.5">{label}</div>
      {note && (
        <div className="text-[11px] text-gray-500 mt-1 flex items-center gap-1">
          <HardDrive className="w-3 h-3" />{note}
        </div>
      )}
    </button>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 text-gray-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">

      {/* ── Begrüßung ──────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-2xl bg-gradient-to-r ${currentTheme.colors.gradient} flex-shrink-0`}>
            <Home className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">
              {greeting()}, {displayName}
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              {new Date().toLocaleDateString(locale, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}
              {lastSeen && <span className="text-gray-600"> · {t('home.lastVisit', { when: formatRelative(lastSeen) })}</span>}
            </p>
          </div>
        </div>
        <button
          onClick={() => void load()}
          title={t('common.refresh')}
          className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white border border-white/10 transition-all flex-shrink-0"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* ── Läuft gerade ───────────────────────────────────────────────── */}
      {active.length > 0 && (
        <div className="rounded-2xl border border-blue-500/30 bg-blue-500/10 p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4 text-blue-300" />
            <h3 className="text-sm font-semibold text-blue-200">{t('home.running.title')}</h3>
          </div>
          <div className="space-y-3">
            {active.map(job => (
              <button
                key={job.training_id}
                onClick={() => navigateTo('training')}
                className="w-full text-left group"
              >
                <div className="flex items-center justify-between text-sm mb-1.5">
                  <span className="text-white font-medium">
                    {t('home.running.step', { step: String(job.current_step), total: String(job.total_steps) })}
                    <span className="text-gray-400 font-normal">
                      {' · '}{t('home.running.epoch', { epoch: String(job.current_epoch), total: String(job.total_epochs) })}
                      {' · '}{t('home.running.loss', { loss: job.train_loss.toFixed(4) })}
                    </span>
                  </span>
                  <span className="text-gray-400 tabular-nums flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatDuration(job.elapsed_time_seconds)}
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-400 to-cyan-400 transition-all"
                    style={{ width: `${Math.min(100, Math.max(0, job.progress_percentage))}%` }}
                  />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Seit deinem letzten Besuch ─────────────────────────────────── */}
      {hasNews && sinceLastVisit && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-gray-300" />
            <h3 className="text-sm font-semibold text-white">{t('home.since.title')}</h3>
          </div>
          <div className="space-y-2">
            {sinceLastVisit.completed.map(job => (
              <button
                key={job.id}
                onClick={() => navigateTo('analysis')}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/15 transition-all text-left group"
              >
                <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span className="text-sm text-white truncate flex-1">
                  {t('home.since.trainingDone', { model: job.model_name })}
                  {typeof job.progress?.train_loss === 'number' && (
                    <span className="text-gray-400"> · {t('home.since.finalLoss', { loss: job.progress.train_loss.toFixed(4) })}</span>
                  )}
                </span>
                <span className="text-xs text-gray-500 tabular-nums flex-shrink-0">{formatDateTime(job.completed_at)}</span>
                <ArrowRight className="w-3.5 h-3.5 text-gray-600 group-hover:text-gray-300 flex-shrink-0" />
              </button>
            ))}
            {sinceLastVisit.failed.map(job => (
              <button
                key={job.id}
                onClick={() => navigateTo('training')}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-red-500/5 hover:bg-red-500/10 border border-red-500/20 hover:border-red-500/40 transition-all text-left group"
              >
                <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                <span className="text-sm text-white truncate flex-1">
                  {job.status === 'stopped'
                    ? t('home.since.trainingStopped', { model: job.model_name })
                    : t('home.since.trainingFailed', { model: job.model_name })}
                  {job.error && <span className="text-gray-400"> · {job.error}</span>}
                </span>
                <span className="text-xs text-gray-500 tabular-nums flex-shrink-0">{formatDateTime(job.completed_at)}</span>
                <ArrowRight className="w-3.5 h-3.5 text-gray-600 group-hover:text-gray-300 flex-shrink-0" />
              </button>
            ))}
            {sinceLastVisit.tested.map(job => (
              <button
                key={job.id}
                onClick={() => navigateTo('tests')}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/15 transition-all text-left group"
              >
                <FlaskConical className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                <span className="text-sm text-white truncate flex-1">
                  {t('home.since.testDone', { model: job.model_name, version: job.version_name })}
                  {typeof job.results?.accuracy === 'number' && (
                    <span className="text-gray-400"> · {t('home.since.accuracy', { value: (job.results.accuracy * 100).toFixed(1) })}</span>
                  )}
                </span>
                <span className="text-xs text-gray-500 tabular-nums flex-shrink-0">{formatDateTime(job.completed_at)}</span>
                <ArrowRight className="w-3.5 h-3.5 text-gray-600 group-hover:text-gray-300 flex-shrink-0" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Erststart ──────────────────────────────────────────────────── */}
      {isFirstRun && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
          <h3 className="text-lg font-semibold text-white mb-2">{t('home.firstRun.title')}</h3>
          <p className="text-sm text-gray-400 max-w-2xl leading-relaxed">{t('home.firstRun.text')}</p>
          <button
            onClick={() => navigateTo('models')}
            className={`mt-4 inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Layers className="w-4 h-4" />
            {t('home.firstRun.cta')}
          </button>
        </div>
      )}

      {/* ── Braucht Aufmerksamkeit ─────────────────────────────────────── */}
      {insights.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-4 h-4 text-amber-300" />
            <h3 className="text-sm font-semibold text-white">{t('home.insights.title')}</h3>
          </div>
          <div className="space-y-2">
            {insights.map(insight => {
              const warn = insight.severity === 'warn';
              const Icon = warn ? AlertTriangle : Info;
              return (
                <button
                  key={insight.id}
                  onClick={() => navigateTo(insight.target)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-all text-left group ${
                    warn
                      ? 'bg-amber-500/5 border-amber-500/20 hover:bg-amber-500/10 hover:border-amber-500/40'
                      : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/15'
                  }`}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 ${warn ? 'text-amber-400' : 'text-gray-400'}`} />
                  {/* Nicht abschneiden: ein halber Warnsatz ist wertlos. */}
                  <span className="text-sm text-white flex-1 min-w-0">
                    {t(insightKey(insight), insight.params)}
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-gray-600 group-hover:text-gray-300 flex-shrink-0" />
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Kennzahlen ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile icon={Layers}    value={models.length}    label={t('home.stats.models')}    view="models" />
        <StatTile icon={GitBranch} value={versionCount}     label={t('home.stats.versions')}  view="versions"
                   note={totalSize > 0 ? formatBytes(totalSize) : undefined} />
        <StatTile icon={Upload}    value={datasets.length}  label={t('home.stats.datasets')}  view="dataset" />
        <StatTile icon={Play}      value={trainings.length} label={t('home.stats.trainings')} view="training" />
      </div>

      {/* ── Letzte Trainings / Letzte Tests ────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Play className="w-4 h-4 text-gray-400" />
              {t('home.recentTrainings.title')}
            </h3>
            <button
              onClick={() => navigateTo('analysis')}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              {t('home.viewAll')}
            </button>
          </div>
          {sortedTrainings.length === 0 ? (
            <p className="text-sm text-gray-500 py-6 text-center">{t('home.recentTrainings.empty')}</p>
          ) : (
            <div className="space-y-1.5">
              {sortedTrainings.slice(0, 5).map(job => {
                const { Icon, cls } = statusVisual(job.status);
                const started = parseDate(job.started_at);
                const ended = parseDate(job.completed_at);
                const dur = started && ended ? (ended.getTime() - started.getTime()) / 1000 : null;
                return (
                  <button
                    key={job.id}
                    onClick={() => navigateTo(job.status === 'completed' ? 'analysis' : 'training')}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all text-left group"
                  >
                    <Icon className={`w-4 h-4 flex-shrink-0 ${cls}`} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-white truncate">{job.model_name}</div>
                      <div className="text-xs text-gray-500 truncate">
                        {job.dataset_name}
                        {dur !== null && <> · {formatDuration(dur)}</>}
                      </div>
                    </div>
                    {typeof job.progress?.train_loss === 'number' && job.status === 'completed' && (
                      <span className="text-xs text-gray-400 tabular-nums flex-shrink-0">
                        {job.progress.train_loss.toFixed(4)}
                      </span>
                    )}
                    <span className="text-xs text-gray-500 tabular-nums flex-shrink-0 w-20 text-right">
                      {formatDateTime(job.completed_at ?? job.created_at)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-gray-400" />
              {t('home.recentTests.title')}
            </h3>
            <button
              onClick={() => navigateTo('tests')}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              {t('home.viewAll')}
            </button>
          </div>
          {sortedTests.length === 0 ? (
            <p className="text-sm text-gray-500 py-6 text-center">{t('home.recentTests.empty')}</p>
          ) : (
            <div className="space-y-1.5">
              {sortedTests.slice(0, 5).map(job => {
                const { Icon, cls } = statusVisual(job.status);
                return (
                  <button
                    key={job.id}
                    onClick={() => navigateTo('tests')}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all text-left group"
                  >
                    <Icon className={`w-4 h-4 flex-shrink-0 ${cls}`} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-white truncate">{job.model_name}</div>
                      <div className="text-xs text-gray-500 truncate">{job.version_name} · {job.dataset_name}</div>
                    </div>
                    {typeof job.results?.accuracy === 'number' && (
                      <span className="text-xs text-emerald-300 tabular-nums flex-shrink-0">
                        {(job.results.accuracy * 100).toFixed(1)}%
                      </span>
                    )}
                    <span className="text-xs text-gray-500 tabular-nums flex-shrink-0 w-20 text-right">
                      {formatDateTime(job.completed_at ?? job.created_at)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Loss-Trend + Bestenliste ──────────────────────────────────── */}
      {(trend.length > 1 || ranking.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

          {trend.length > 1 && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  {trendPct !== null && trendPct >= 0
                    ? <TrendingDown className="w-4 h-4 text-emerald-400" />
                    : <TrendingUp className="w-4 h-4 text-amber-400" />}
                  {t('home.trend.title')}
                </h3>
                {trendPct !== null && (
                  <span className={`text-sm font-medium tabular-nums ${trendPct >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {trendPct >= 0 ? '−' : '+'}{Math.abs(trendPct).toFixed(1)}%
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500 mb-4">{t('home.trend.subtitle', { count: String(trend.length) })}</p>

              {/* Sparkline: kleiner Loss liegt oben, aelteste Laeufe links. */}
              <svg viewBox="0 0 300 60" preserveAspectRatio="none" className="w-full h-16" role="img"
                   aria-label={t('home.trend.title')}>
                <polyline
                  points={sparklinePoints(trend, 300, 60, 4)}
                  fill="none"
                  stroke="url(#ft-trend)"
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                />
                <defs>
                  <linearGradient id="ft-trend" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#60a5fa" />
                    <stop offset="100%" stopColor="#34d399" />
                  </linearGradient>
                </defs>
              </svg>

              <div className="flex items-center justify-between text-xs mt-2 tabular-nums">
                <span className="text-gray-400">{trend[0].loss.toFixed(4)}</span>
                <span className="text-gray-500 truncate px-2">{trend[trend.length - 1].label}</span>
                <span className="text-gray-200 font-medium">{trend[trend.length - 1].loss.toFixed(4)}</span>
              </div>
            </div>
          )}

          {ranking.length > 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-4">
                <Trophy className="w-4 h-4 text-amber-300" />
                {t('home.ranking.title')}
              </h3>
              <div className="space-y-1.5">
                {ranking.map((row, i) => (
                  <button
                    key={row.id}
                    onClick={() => navigateTo('tests')}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all text-left"
                  >
                    <span className={`w-5 text-center text-xs font-bold tabular-nums flex-shrink-0 ${
                      i === 0 ? 'text-amber-300' : 'text-gray-600'
                    }`}>{i + 1}</span>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-white truncate">{row.model_name}</div>
                      <div className="text-xs text-gray-500 truncate">{row.version_name} · {row.dataset_name}</div>
                    </div>
                    <span className="text-sm text-emerald-300 tabular-nums flex-shrink-0">
                      {(row.accuracy * 100).toFixed(1)}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── KI-Briefing ────────────────────────────────────────────────── */}
      {!isFirstRun && (
        <HomeBriefing facts={briefingFacts} factsKey={factsHash(briefingFacts)} userId={userId} />
      )}
    </div>
  );
}
