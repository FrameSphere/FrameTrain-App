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
import { open as openUrl } from '@tauri-apps/plugin-shell';
import {
  Home,
  Layers,
  Upload,
  Play,
  PlayCircle,
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
import { VIDEOS, youtubeWatchUrl } from '../utils/videos';
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
      // Dieselben Saetze, die der Nutzer auf der Seite liest. Vorher gingen die
      // internen Kennungen raus ("trainingFailed"), und der Coach zitierte sie
      // woertlich in seiner Antwort.
      insights.length > 0
        ? `Offene Hinweise:\n${insights.map(i => `  - ${t(insightKey(i), i.params)}`).join('\n')}`
        : 'Offene Hinweise: keine',
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

  // Status als Punkt + Wort: Farbe allein reicht nicht (Farbsehschwaeche)
  const statusDot = (status: string) => {
    switch (status) {
      case 'completed': return 'var(--ft-ok)';
      case 'failed':    return 'var(--ft-bad)';
      case 'stopped':   return 'var(--ft-warn)';
      case 'running':
      case 'pending':   return 'var(--ft-out)';
      default:          return 'var(--ft-faint)';
    }
  };
  const statusLabel = (status: string) => t(`home.status.${status}`, status);

  const plural = (key: string, n: number) =>
    t(n === 1 ? `home.summary.${key}_one` : `home.summary.${key}`, { n: n.toLocaleString(locale) });

  // Laufende Jobs: Modellname aus der Historie, falls dort schon eingetragen
  const runningName = trainings.find(j => j.status === 'running')?.model_name;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 text-gray-500 animate-spin" />
      </div>
    );
  }

  const stats: { value: number; label: string; view: AppView; note?: string }[] = [
    { value: models.length,    label: t('home.stats.models'),    view: 'models' },
    { value: versionCount,     label: t('home.stats.versions'),  view: 'versions', note: totalSize > 0 ? formatBytes(totalSize) : undefined },
    { value: datasets.length,  label: t('home.stats.datasets'),  view: 'dataset' },
    { value: trainings.length, label: t('home.stats.trainings'), view: 'training' },
  ];

  return (
    <div className="space-y-6">

      {/* ── Begrüßung ──────────────────────────────────────────────────── */}
      <header className="flex items-end gap-3 flex-wrap">
        <div className="flex-1 min-w-[260px]">
          <h1 className="ft-h1">{greeting()}, {displayName}</h1>
          <p className="text-[14px] text-gray-400 mt-1.5">
            {isFirstRun
              ? new Date().toLocaleDateString(locale, { weekday: 'long', day: 'numeric', month: 'long' })
              : <>
                  {plural('models', models.length)}, {plural('datasets', datasets.length)}, {plural('runs', trainings.length)}.{' '}
                  {active.length > 0 ? plural('running', active.length) : t('home.summary.idle')}
                </>}
            {lastSeen && <span className="text-gray-500"> · {t('home.lastVisit', { when: formatRelative(lastSeen) })}</span>}
          </p>
        </div>
        <button onClick={() => void load()} title={t('common.refresh')} aria-label={t('common.refresh')} className="ft-icon-btn">
          <RefreshCw className="w-4 h-4" />
        </button>
        <button onClick={() => navigateTo('training')} className="ft-btn ft-btn-primary">
          <Play className="w-3.5 h-3.5" />
          {t('home.newTraining')}
        </button>
      </header>

      {/* ── Läuft gerade (Ergebnis-Farbe) ──────────────────────────────── */}
      {active.length > 0 && (
        <section className="ft-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <span className="w-2 h-2 rounded-full" style={{ background: 'var(--ft-out)' }} aria-hidden="true" />
            <h3 className="ft-label" style={{ color: 'var(--ft-out-text)' }}>{t('home.running.title')}</h3>
            {runningName && <span className="ft-h2 ml-1">{runningName}</span>}
          </div>
          <div className="space-y-4">
            {active.map(job => (
              <button key={job.training_id} onClick={() => navigateTo('training')} className="w-full text-left group">
                <div className="flex items-baseline justify-between gap-4 mb-2">
                  <span className="text-[14px] text-white font-medium">
                    {t('home.running.step', { step: job.current_step.toLocaleString(locale), total: job.total_steps.toLocaleString(locale) })}
                  </span>
                  <span className="font-mono text-[12px] text-gray-400 tabular-nums flex items-center gap-3">
                    <span>{t('home.running.epoch', { epoch: String(job.current_epoch), total: String(job.total_epochs) })}</span>
                    <span style={{ color: 'var(--ft-out-text)' }}>{t('home.running.loss', { loss: job.train_loss.toFixed(4) })}</span>
                    <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{formatDuration(job.elapsed_time_seconds)}</span>
                  </span>
                </div>
                <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--ft-line-strong)' }}>
                  <div
                    className="h-full rounded-full transition-[width] duration-500"
                    style={{ width: `${Math.min(100, Math.max(0, job.progress_percentage))}%`, background: 'var(--ft-out)' }}
                  />
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* ── Erststart ──────────────────────────────────────────────────── */}
      {isFirstRun && (
        <section className="ft-card p-6">
          <h3 className="text-[17px] font-semibold text-white mb-2 tracking-[-0.01em]">{t('home.firstRun.title')}</h3>
          <p className="text-sm text-gray-400 max-w-2xl leading-relaxed">{t('home.firstRun.text')}</p>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button onClick={() => navigateTo('models')} className="ft-btn ft-btn-primary">
              <Layers className="w-4 h-4" />
              {t('home.firstRun.cta')}
            </button>
            {/* Trainings-Video im Standardbrowser (Tauri-Shell), kein eingebetteter Player */}
            <button onClick={() => { void openUrl(youtubeWatchUrl(VIDEOS.train)).catch(() => {}); }} className="ft-btn ft-btn-ghost">
              <PlayCircle className="w-4 h-4" />
              {t('home.firstRun.videoCta')}
            </button>
          </div>
        </section>
      )}

      {/* ── Kennzahlen: eine ruhige Zeile statt Kachelraster ───────────── */}
      {!isFirstRun && (
        <div className="ft-card grid grid-cols-2 lg:grid-cols-4 overflow-hidden">
          {stats.map((st, i) => (
            <button
              key={st.view}
              onClick={() => navigateTo(st.view)}
              className={`text-left px-5 py-4 hover:bg-white/[0.04] transition-colors group ${i > 0 ? 'lg:border-l' : ''} ${i % 2 === 1 ? 'border-l lg:border-l' : ''} ${i > 1 ? 'border-t lg:border-t-0' : ''}`}
              style={{ borderColor: 'var(--ft-line)' }}
            >
              <div className="ft-label flex items-center justify-between">
                {st.label}
                <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="flex items-baseline gap-2 mt-1.5">
                <span className="text-[26px] leading-none font-semibold tracking-[-0.04em] text-white tabular-nums">{st.value.toLocaleString(locale)}</span>
                {st.note && <span className="font-mono text-[11.5px] text-gray-500 flex items-center gap-1"><HardDrive className="w-3 h-3" />{st.note}</span>}
              </div>
            </button>
          ))}
        </div>
      )}

      {/* ── Zwei Spalten: Läufe links, Hinweise rechts ──────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-x-8 gap-y-6">

        <div className="space-y-6 min-w-0">
          {/* Letzte Trainings */}
          <section>
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="ft-h2">{t('home.recentTrainings.title')}</h3>
              <button onClick={() => navigateTo('analysis')} className="ft-link">{t('home.viewAll')}</button>
            </div>
            {sortedTrainings.length === 0 ? (
              <p className="text-sm text-gray-500 py-6 border-t" style={{ borderColor: 'var(--ft-line)' }}>{t('home.recentTrainings.empty')}</p>
            ) : (
              <div role="table" className="text-[13.5px]">
                <div role="row" className="ft-home-row ft-home-head">
                  <span role="columnheader">{t('home.cols.model')}</span>
                  <span role="columnheader" className="text-right">{t('home.cols.loss')}</span>
                  <span role="columnheader">Status</span>
                  <span role="columnheader" className="text-right">{t('home.cols.when')}</span>
                </div>
                {sortedTrainings.slice(0, 6).map(job => {
                  const started = parseDate(job.started_at);
                  const ended = parseDate(job.completed_at);
                  const dur = started && ended ? (ended.getTime() - started.getTime()) / 1000 : null;
                  return (
                    <button
                      role="row"
                      key={job.id}
                      onClick={() => navigateTo(job.status === 'completed' ? 'analysis' : 'training')}
                      className="ft-home-row ft-home-link"
                      title={job.error ?? undefined}
                    >
                      <span role="cell" className="min-w-0">
                        <span className="block truncate text-white">{job.model_name}</span>
                        <span className="block truncate text-[12px] text-gray-500">
                          {job.dataset_name}{dur !== null && <> · <span className="font-mono">{formatDuration(dur)}</span></>}
                        </span>
                      </span>
                      <span role="cell" className="font-mono text-[12.5px] text-right tabular-nums text-white">
                        {typeof job.progress?.train_loss === 'number' && job.status === 'completed' ? job.progress.train_loss.toFixed(4) : '–'}
                      </span>
                      <span role="cell" className="flex items-center gap-2 text-gray-400 text-[12.5px]">
                        <span className="w-1.5 h-1.5 rounded-full flex-none" style={{ background: statusDot(job.status) }} aria-hidden="true" />
                        {statusLabel(job.status)}
                      </span>
                      <span role="cell" className="font-mono text-[12px] text-right tabular-nums text-gray-500">
                        {formatDateTime(job.completed_at ?? job.created_at)}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </section>

          {/* Letzte Tests */}
          <section>
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="ft-h2">{t('home.recentTests.title')}</h3>
              <button onClick={() => navigateTo('tests')} className="ft-link">{t('home.viewAll')}</button>
            </div>
            {sortedTests.length === 0 ? (
              <p className="text-sm text-gray-500 py-6 border-t" style={{ borderColor: 'var(--ft-line)' }}>{t('home.recentTests.empty')}</p>
            ) : (
              <div role="table" className="text-[13.5px]">
                <div role="row" className="ft-home-row ft-home-head">
                  <span role="columnheader">{t('home.cols.model')}</span>
                  <span role="columnheader" className="text-right">{t('home.cols.accuracy')}</span>
                  <span role="columnheader">Status</span>
                  <span role="columnheader" className="text-right">{t('home.cols.when')}</span>
                </div>
                {sortedTests.slice(0, 5).map(job => (
                  <button role="row" key={job.id} onClick={() => navigateTo('tests')} className="ft-home-row ft-home-link">
                    <span role="cell" className="min-w-0">
                      <span className="block truncate text-white">{job.model_name}</span>
                      <span className="block truncate text-[12px] text-gray-500">{job.version_name} · {job.dataset_name}</span>
                    </span>
                    <span role="cell" className="font-mono text-[12.5px] text-right tabular-nums text-white">
                      {typeof job.results?.accuracy === 'number' ? `${(job.results.accuracy * 100).toFixed(1)}%` : '–'}
                    </span>
                    <span role="cell" className="flex items-center gap-2 text-gray-400 text-[12.5px]">
                      <span className="w-1.5 h-1.5 rounded-full flex-none" style={{ background: statusDot(job.status) }} aria-hidden="true" />
                      {statusLabel(job.status)}
                    </span>
                    <span role="cell" className="font-mono text-[12px] text-right tabular-nums text-gray-500">
                      {formatDateTime(job.completed_at ?? job.created_at)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>
        </div>

        <div className="space-y-6 min-w-0 lg:border-l lg:pl-8" style={{ borderColor: 'var(--ft-line)' }}>
          {/* Seit dem letzten Besuch */}
          {hasNews && sinceLastVisit && (
            <section>
              <h3 className="ft-h2 mb-1">{t('home.since.title')}</h3>
              <div>
                {sinceLastVisit.completed.map(job => (
                  <button key={job.id} onClick={() => navigateTo('analysis')} className="ft-dot-row">
                    <span className="ft-dot" style={{ background: 'var(--ft-ok)' }} aria-hidden="true" />
                    <span className="flex-1 min-w-0">
                      {t('home.since.trainingDone', { model: job.model_name })}
                      {typeof job.progress?.train_loss === 'number' && (
                        <span className="block text-[12.5px] text-gray-400">{t('home.since.finalLoss', { loss: job.progress.train_loss.toFixed(4) })}</span>
                      )}
                    </span>
                    <span className="font-mono text-[11.5px] text-gray-500 tabular-nums">{formatDateTime(job.completed_at)}</span>
                  </button>
                ))}
                {sinceLastVisit.failed.map(job => (
                  <button key={job.id} onClick={() => navigateTo('training')} className="ft-dot-row">
                    <span className="ft-dot" style={{ background: job.status === 'stopped' ? 'var(--ft-warn)' : 'var(--ft-bad)' }} aria-hidden="true" />
                    <span className="flex-1 min-w-0">
                      {job.status === 'stopped'
                        ? t('home.since.trainingStopped', { model: job.model_name })
                        : t('home.since.trainingFailed', { model: job.model_name })}
                      {job.error && <span className="block text-[12.5px] text-gray-400 break-words">{job.error}</span>}
                    </span>
                    <span className="font-mono text-[11.5px] text-gray-500 tabular-nums">{formatDateTime(job.completed_at)}</span>
                  </button>
                ))}
                {sinceLastVisit.tested.map(job => (
                  <button key={job.id} onClick={() => navigateTo('tests')} className="ft-dot-row">
                    <span className="ft-dot" style={{ background: 'var(--ft-accent-text)' }} aria-hidden="true" />
                    <span className="flex-1 min-w-0">
                      {t('home.since.testDone', { model: job.model_name, version: job.version_name })}
                      {typeof job.results?.accuracy === 'number' && (
                        <span className="block text-[12.5px] text-gray-400">{t('home.since.accuracy', { value: (job.results.accuracy * 100).toFixed(1) })}</span>
                      )}
                    </span>
                    <span className="font-mono text-[11.5px] text-gray-500 tabular-nums">{formatDateTime(job.completed_at)}</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Braucht Aufmerksamkeit */}
          {insights.length > 0 && (
            <section>
              <h3 className="ft-h2 mb-1">{t('home.worth')}</h3>
              <div>
                {insights.map(insight => {
                  const warn = insight.severity === 'warn';
                  return (
                    <button key={insight.id} onClick={() => navigateTo(insight.target)} className="ft-dot-row">
                      <span className="ft-dot" style={{ background: warn ? 'var(--ft-warn)' : 'var(--ft-faint)' }} aria-hidden="true" />
                      {/* Nicht abschneiden: ein halber Warnsatz ist wertlos. */}
                      <span className="flex-1 min-w-0">
                        {warn && <span className="sr-only">{t('home.insights.title')}: </span>}
                        {t(insightKey(insight), insight.params)}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-gray-500 flex-none mt-1" />
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {/* Loss-Trend: Ergebnis-Kurve in Orange */}
          {trend.length > 1 && (
            <section>
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="ft-h2 flex items-center gap-2">
                  {trendPct !== null && trendPct >= 0
                    ? <TrendingDown className="w-4 h-4 text-emerald-400" />
                    : <TrendingUp className="w-4 h-4 text-amber-400" />}
                  {t('home.trend.title')}
                </h3>
                {trendPct !== null && (
                  <span className={`font-mono text-[12.5px] tabular-nums ${trendPct >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {trendPct >= 0 ? '−' : '+'}{Math.abs(trendPct).toFixed(1)}%
                  </span>
                )}
              </div>
              <p className="text-[12.5px] text-gray-500 mt-0.5 mb-3">{t('home.trend.subtitleModel', { count: String(trend.length), model: trend[trend.length - 1]?.label ?? '' })}</p>
              {/* Sparkline: kleiner Loss liegt oben, aelteste Laeufe links. */}
              <svg viewBox="0 0 300 60" preserveAspectRatio="none" className="w-full h-14" role="img" aria-label={t('home.trend.title')}>
                <polyline
                  points={sparklinePoints(trend, 300, 60, 4)}
                  fill="none"
                  stroke="var(--ft-out)"
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
              <div className="flex items-center justify-between font-mono text-[11.5px] mt-1.5 tabular-nums">
                <span className="text-gray-500">{trend[0].loss.toFixed(4)}</span>
                <span style={{ color: 'var(--ft-out-text)' }}>{trend[trend.length - 1].loss.toFixed(4)}</span>
              </div>
            </section>
          )}

          {/* Bestenliste */}
          {ranking.length > 0 && (
            <section>
              <h3 className="ft-h2 mb-1">{t('home.ranking.title')}</h3>
              <div>
                {ranking.map((row, i) => (
                  <button key={row.id} onClick={() => navigateTo('tests')} className="ft-dot-row items-center">
                    <span className={`w-4 font-mono text-[12px] tabular-nums flex-none ${i === 0 ? 'text-white' : 'text-gray-500'}`}>{i + 1}</span>
                    <span className="flex-1 min-w-0">
                      <span className="block truncate">{row.model_name}</span>
                      <span className="block truncate text-[12px] text-gray-500">{row.version_name} · {row.dataset_name}</span>
                    </span>
                    <span className="font-mono text-[12.5px] tabular-nums text-white">{(row.accuracy * 100).toFixed(1)}%</span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      {/* ── KI-Briefing ────────────────────────────────────────────────── */}
      {!isFirstRun && (
        <HomeBriefing facts={briefingFacts} factsKey={factsHash(briefingFacts)} userId={userId} />
      )}
    </div>
  );
}
