// HomePanel.tsx – Startseite der App.
//
// Landepunkt nach dem Login: Laeuft etwas? Was wartet auf mich? Wo steht jedes
// Modell? Jede Zeile ist ein Sprung mit Kontext in den passenden Arbeitsbereich.
//
// Alle Daten kommen aus bereits vorhandenen Backend-Commands; die Seite fügt
// dem Backend nichts hinzu, sie aggregiert nur (homePipeline.ts).

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import { Layers, Loader2, Play, PlayCircle, RefreshCw } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { usePageContext } from '../contexts/PageContext';
import { useTrainingContext } from '../contexts/TrainingContext';
import { navigateTo, setModelTargets } from '../ui/navigationEvents';
import { dateLocale } from '../utils/dateLocale';
import { VIDEOS, youtubeWatchUrl } from '../utils/videos';
import { detectPluginForModel, timeSeriesFamily } from '../plugins/registry';
import HomeBriefing from './HomeBriefing';
import { FirstRunSteps, PipelineHead, PipelineList } from './HomeStellwerk';
import { BenchmarkView } from './BenchmarkView';
import { buildBenchmarks } from './homeBenchmarks';
import { buildInsights, buildBriefingFacts, factsHash, type InsightInput } from './homeInsights';
import { buildCompare, buildPipeline, homeHeadline, pipelineSections } from './homePipeline';
import { buildHomeCoachContext } from './homeCoachContext';
import type { HostInfo } from './hosting/hostingModel';

// ============ Types (Spiegel der Rust-Structs) ============

interface TrainingJob {
  id: string;
  model_id?: string;
  model_name: string;
  dataset_id?: string;
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
  version_id?: string;
  version_name: string;
  dataset_id?: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  results?: {
    total_samples: number;
    accuracy?: number | null;
    average_loss?: number | null;
    metrics?: Record<string, unknown> | null;
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

interface ModelInfo {
  id: string;
  name: string;
  source_path?: string | null;
  model_type?: string | null;
  plugin_override?: string | null;
}

interface DatasetInfo {
  id: string;
  name: string;
  model_id?: string;
  training_count?: number;
  split_info?: { test_count: number } | null;
}

interface HomePanelProps {
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

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '–';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Zeilen, die ohne Scrollen passen; im Vergleich sind es so viele wie moeglich. */
function visibleRows(): number {
  const height = typeof window !== 'undefined' ? window.innerHeight : 900;
  return Math.max(8, Math.floor((height - 240) / 32));
}

export default function HomePanel({ userId }: HomePanelProps) {
  const { t, language } = useLanguage();
  const { setCurrentPageContent } = usePageContext();
  const { state: trainingState, setShowDashboard, setIsDashMinimized } = useTrainingContext();

  const [loading, setLoading] = useState(true);
  const [trainings, setTrainings] = useState<TrainingJob[]>([]);
  const [tests, setTests] = useState<TestJob[]>([]);
  const [active, setActive] = useState<ActiveTraining[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [hosts, setHosts] = useState<HostInfo[]>([]);
  const [sleepPrevented, setSleepPrevented] = useState(true);
  const [view, setView] = useState<'pipeline' | 'compare'>('pipeline');
  const [showBriefing, setShowBriefing] = useState(false);

  // Zeitpunkt des letzten Besuchs — einmalig beim Mount gelesen und danach
  // eingefroren, damit "Neu" nicht verschwindet, während man liest.
  // Zurückgeschrieben wird erst beim Verlassen der Seite.
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

  // Nur das erste Laden zeigt den Spinner. Beim Aktualisieren bleibt die Liste stehen,
  // damit Zeilen, die die Gruppe wechseln, sichtbar an ihren neuen Platz gleiten.
  const [refreshing, setRefreshing] = useState(false);
  const load = useCallback(async () => {
    setRefreshing(true);
    const [th, sh, at, ml, ds, hs, sp] = await Promise.allSettled([
      invoke<TrainingJob[]>('get_training_history'),
      invoke<TestJob[]>('get_test_history'),
      invoke<ActiveTraining[]>('list_active_trainings'),
      invoke<ModelInfo[]>('list_models'),
      invoke<DatasetInfo[]>('list_all_datasets'),
      invoke<HostInfo[]>('hosting_list'),
      invoke<boolean>('get_prevent_sleep_status'),
    ]);
    // Einzelne fehlschlagende Quellen dürfen die ganze Startseite nicht leeren.
    setTrainings(th.status === 'fulfilled' ? th.value : []);
    setTests(sh.status === 'fulfilled' ? sh.value : []);
    setActive(at.status === 'fulfilled' ? at.value : []);
    setModels(ml.status === 'fulfilled' ? ml.value : []);
    setDatasets(ds.status === 'fulfilled' ? ds.value : []);
    setHosts(hs.status === 'fulfilled' ? hs.value ?? [] : []);
    // Im Zweifel KEINE Warnung: ein nicht lesbarer Status ist kein Befund.
    setSleepPrevented(sp.status === 'fulfilled' ? sp.value : true);
    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => { void load(); }, [load]);

  const locale = dateLocale(language);
  const pipelineInput = useMemo(() => ({ models, datasets, trainings, tests, hosts }), [models, datasets, trainings, tests, hosts]);
  const allRows = useMemo(() => buildPipeline(pipelineInput), [pipelineInput]);

  const job = active[0];
  const isRunning = active.length > 0;
  const runningName = trainings.find(j => j.status === 'running')?.model_name
    ?? (isRunning ? trainingState.modelName || null : null);
  const rows = isRunning && runningName ? allRows.filter(r => r.name !== runningName) : allRows;
  const headline = homeHeadline(allRows, runningName, isRunning);
  const isFirstRun = models.length === 0 && datasets.length === 0 && trainings.length === 0 && tests.length === 0;
  const hasData = !isFirstRun;

  const attentionCount = rows.filter(r => r.group === 'attention').length;
  const needsYou = t(attentionCount === 1 ? 'home.pipeline.title.attention_one' : 'home.pipeline.title.attention', { n: String(attentionCount) });
  const title = headline.kind === 'training' ? t('home.pipeline.title.training', { model: headline.model })
    : headline.kind === 'attention' ? t(headline.count === 1 ? 'home.pipeline.title.attention_one' : 'home.pipeline.title.attention', { n: String(headline.count) })
      : headline.kind === 'clear' ? t('home.pipeline.title.clear')
        : headline.kind === 'noData' ? t('home.pipeline.title.noData')
          : t('home.pipeline.title.empty');
  // Zweiter Satz in gedaempfter Farbe wie im Website-Hero ("Your data. Your GPU. Your model."):
  // was nach den offenen Punkten als Naechstes geht.
  const subline = (() => {
    if (headline.kind !== 'attention' && headline.kind !== 'clear' && headline.kind !== 'training') return null;
    const counts = Object.fromEntries(pipelineSections(rows).sections.map(s => [s.key, s.rows.length]));
    for (const key of ['ready', 'test', 'hosted'] as const) {
      const n = counts[key] ?? 0;
      if (n > 0) return t(`home.pipeline.title.then.${key}${n === 1 ? '_one' : ''}`, { n: String(n) });
    }
    return null;
  })();

  // Aufgabe je Modell: dieselbe Erkennung wie im Training, nicht der Modellname.
  // Gruppiert wird nach Aufgabe, nicht nach Plugin: BERT und XLM-RoBERTa sind beide
  // Textklassifikation und gehoeren in denselben Vergleich.
  const taskKey = useCallback((modelId: string) => {
    const model = models.find(m => m.id === modelId);
    if (!model) return 'other';
    const detected = detectPluginForModel(model);
    if (detected.supported) return detected.plugin.taskType;
    return timeSeriesFamily(model.source_path || model.name) ? 'time_series' : 'other';
  }, [models]);
  const groupLabel = (key: string) => t(`home.pipeline.task.${key}`, t('home.pipeline.task.other'));
  // Der Hinweis gilt nur ohne eigenen Testteil: mit Split testet die Engine auf ungesehenen Daten.
  const compare = useMemo(() => {
    const result = buildCompare(allRows, taskKey);
    const hasTestSplit = (name: string | null) => (datasets.find(d => d.name === name)?.split_info?.test_count ?? 0) > 0;
    return {
      ...result,
      groups: result.groups.map(g => ({
        ...g,
        rows: g.rows.map(r => ({ ...r, sameData: r.sameData && !hasTestSplit(r.row.test.dataset) })),
      })),
    };
  }, [allRows, taskKey, datasets]);
  const benchmarks = useMemo(() => buildBenchmarks(
    allRows, taskKey,
    name => (datasets.find(d => d.name === name)?.split_info?.test_count ?? 0) > 0,
  ), [allRows, taskKey, datasets]);

  const insightInput: InsightInput = {
    trainings, tests, models, datasets,
    hasActiveTraining: isRunning,
    sleepPrevented,
  };
  const insights = buildInsights(insightInput);
  const briefingFacts = buildBriefingFacts(insightInput);
  const sleepRisk = insights.some(i => i.kind === 'sleepRisk');

  const format = useMemo(() => ({
    date: (ms: number | null | undefined) => {
      if (!ms) return '–';
      return new Date(ms).toLocaleString(locale, { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
    },
    ago: (ms: number | null | undefined) => {
      if (!ms) return '–';
      const sec = (ms - Date.now()) / 1000;
      const units: [Intl.RelativeTimeFormatUnit, number][] = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]];
      const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
      for (const [unit, size] of units) {
        if (Math.abs(sec) >= size) return rtf.format(Math.round(sec / size), unit);
      }
      return rtf.format(0, 'minute');
    },
    duration: (seconds: number | null | undefined) => (seconds === null || seconds === undefined ? '–' : formatDuration(seconds)),
    number: (value: number, digits: number) => value.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }),
  }), [locale]);

  // ── Seiten-Kontext für den AI-Coach ──────────────────────────────────────
  // Dieselben Gruppen, Zahlen und Woerter wie auf dem Bildschirm (homeCoachContext.ts).
  const coachContext = buildHomeCoachContext({
    language, t, ago: format.ago, number: format.number, title, view, rows,
    running: job ? { model: runningName ?? '', percent: Math.round(job.progress_percentage) } : null,
    sleepRisk, compare, groupLabel,
  });
  useEffect(() => {
    setCurrentPageContent(coachContext, 'home');
  }, [coachContext, setCurrentPageContent]);

  // Der Coach springt per Modellname mit demselben Fokus wie die Knoepfe der Seite.
  useEffect(() => {
    setModelTargets(new Map(allRows.map(r => [r.name.trim().toLowerCase(), { modelId: r.id, datasetId: r.data.id, versionId: r.test.versionId }])));
  }, [allRows]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 text-gray-500 animate-spin" />
      </div>
    );
  }

  const openTraining = () => {
    if (trainingState.currentJob) { setShowDashboard(true); setIsDashMinimized(false); }
    else navigateTo('training');
  };

  const liveLoss = trainingState.lossPoints.map(p => p.train_loss).filter(v => Number.isFinite(v));

  return (
    <div className="ft-sw">
      <header className="ft-sw-top">
        <h1>{title}{subline && <>.<span className="ft-h1-sub"> {subline}</span></>}</h1>
        {hasData ? (
          <>
            <div className="ft-seg" role="group" aria-label={t('home.pipeline.view')}>
              <button className={`ft-seg-item ${view === 'pipeline' ? 'is-active' : ''}`} aria-pressed={view === 'pipeline'} onClick={() => setView('pipeline')}>{t('home.pipeline.pipeline')}</button>
              <button className={`ft-seg-item ${view === 'compare' ? 'is-active' : ''}`} aria-pressed={view === 'compare'} onClick={() => setView('compare')}>{t('home.pipeline.compare.title')}</button>
            </div>
            <button onClick={() => void load()} title={t('common.refresh')} aria-label={t('common.refresh')} className="ft-icon-btn">
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={() => setShowBriefing(v => !v)} aria-expanded={showBriefing} className="ft-btn ft-btn-ghost">{t('home.pipeline.briefing')}</button>
            <button onClick={() => navigateTo('training')} className="ft-btn ft-btn-primary">
              <Play className="w-3.5 h-3.5" />
              {t('home.newTraining')}
            </button>
          </>
        ) : (
          <>
            <button onClick={() => { void openUrl(youtubeWatchUrl(VIDEOS.train)).catch(() => {}); }} className="ft-btn ft-btn-ghost">
              <PlayCircle className="w-4 h-4" />
              {t('home.firstRun.videoCta')}
            </button>
            <button onClick={() => navigateTo('models')} className="ft-btn ft-btn-primary">
              <Layers className="w-4 h-4" />
              {t('home.firstRun.cta')}
            </button>
          </>
        )}
      </header>

      {showBriefing && hasData && (
        <div className="mb-4">
          <HomeBriefing facts={briefingFacts} factsKey={factsHash(briefingFacts)} userId={userId} />
        </div>
      )}

      {!hasData && <FirstRunSteps />}

      {hasData && view === 'pipeline' && (
        <div className="ft-sw-list">
          <PipelineHead />
          {job && (
            <>
              <div className="ft-run">
                <div className="ft-run-who">
                  <b title={runningName ?? undefined}>{runningName ?? t('home.running.title')}</b>
                  <span>{t('home.pipeline.running', { pct: String(Math.round(job.progress_percentage)) })}</span>
                </div>
                <button className="ft-track" onClick={openTraining}>
                  <i style={{ width: `${Math.min(100, Math.max(0, job.progress_percentage))}%` }} aria-hidden="true" />
                  <div className="ft-kv"><span>{t('home.pipeline.run.epoch')}</span><b>{job.current_epoch} / {job.total_epochs}</b></div>
                  <div className="ft-kv"><span>{t('home.pipeline.run.step')}</span><b>{job.current_step.toLocaleString(locale)} / {job.total_steps.toLocaleString(locale)}</b></div>
                  <div className="ft-kv is-out"><span>{t('home.pipeline.run.loss')}</span><b>{format.number(job.train_loss, 4)}</b></div>
                  <div className="ft-kv"><span>{t('home.pipeline.run.elapsed')}</span><b>{formatDuration(job.elapsed_time_seconds)}</b></div>
                  {liveLoss.length > 1 && <LiveCurve values={liveLoss} />}
                </button>
              </div>
              {sleepRisk && (
                <div className="ft-strip">
                  <span aria-hidden="true">▲</span>
                  {t('home.pipeline.sleepRisk')}
                  <button className="ft-go" onClick={() => { void invoke('enable_prevent_sleep').then(() => setSleepPrevented(true)).catch(() => {}); }}>{t('home.pipeline.keepAwake')}</button>
                </div>
              )}
            </>
          )}
          <PipelineList
            rows={rows}
            f={format}
            lastSeen={lastSeen?.getTime() ?? null}
            limit={visibleRows()}
          />
        </div>
      )}

      {hasData && view === 'compare' && (
        <div className="ft-sw-list">
          <BenchmarkView tasks={benchmarks.tasks} withoutResult={benchmarks.withoutResult} taskLabel={groupLabel} f={format} />
        </div>
      )}
    </div>
  );
}

function LiveCurve({ values }: { values: number[] }) {
  const recent = values.slice(-80);
  const max = Math.max(...recent), min = Math.min(...recent), span = max - min || 1;
  const pts = recent.map((v, i) => [(i / (recent.length - 1)) * 170, 2 + ((max - v) / span) * 36] as const);
  const [lx, ly] = pts[pts.length - 1];
  const [px, py] = pts[pts.length - 2];
  // Der neueste Abschnitt zeichnet sich nach; der Rest steht schon.
  return (
    <svg viewBox="0 0 170 40" className="ft-track-curve" aria-hidden="true">
      <polyline points={pts.slice(0, -1).map(p => p.join(',')).join(' ')} />
      <line key={values.length} x1={px} y1={py} x2={lx} y2={ly} pathLength={1} className="ft-curve-new" />
      <circle key={`d${values.length}`} cx={lx} cy={ly} r={2.4} className="ft-curve-dot" />
    </svg>
  );
}
