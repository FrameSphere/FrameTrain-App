// homePipeline.ts – Stellwerk der Startseite.
//
// Reine Logik ohne React: leitet aus den vorhandenen Verlaeufen ab, wo jedes
// Modell steht (Daten, Training, Test, Hosting) und was als Naechstes ansteht.
// Die Aktion steht immer in der Spalte ihres Schritts: ein gescheiterter Test
// wird in der Test-Spalte wiederholt, nicht unter "Hosting".

import { OVERFIT_RATIO, REGRESSION_RATIO, WEAK_ACCURACY, byNewest, endedAt } from './homeInsights';

export interface PipelineModel { id: string; name: string }
export interface PipelineDataset { id: string; name: string; model_id?: string }

export interface PipelineTraining {
  id: string;
  model_id?: string;
  model_name: string;
  dataset_id?: string;
  dataset_name?: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  progress?: { train_loss?: number | null; val_loss?: number | null };
  config?: { epochs?: number | null; learning_rate?: number | null; batch_size?: number | null } | null;
}

export interface PipelineTest {
  id: string;
  model_id?: string;
  model_name: string;
  version_id?: string;
  version_name?: string;
  dataset_id?: string;
  dataset_name?: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  results?: {
    accuracy?: number | null;
    total_time?: number | null;
    metrics?: Record<string, unknown> | null;
  } | null;
}

export interface PipelineHost { model_id: string; status: string }

export type StepState = 'none' | 'done' | 'failed' | 'stopped' | 'running';
export type GroupId = 'attention' | 'next' | 'hosted';
export type NextKind = 'addDataset' | 'train' | 'retryTrain' | 'startTest' | 'retryTest' | 'host';
export type NoteKind = 'trainingFailed' | 'trainingStopped' | 'testFailed' | 'testStopped' | 'regression' | 'overfitting' | 'weakResult';

export interface RunPoint {
  id: string;
  at: number;
  loss: number;
  durationSec: number | null;
  dataset: string;
  /** Epochen, Lernrate und Batch-Groesse; ohne Angabe ist der Lauf nicht vergleichbar */
  setup: string | null;
}

export interface TestMetric {
  label: string;
  value: number;
  /** 0..1, nur dann darf ein Balken mit fester Skala gezeigt werden */
  fraction: number | null;
}

export interface ModelRow {
  id: string;
  name: string;
  group: GroupId;
  data: { name: string | null; id: string | null };
  training: {
    state: StepState;
    loss: number | null;
    valLoss: number | null;
    error: string | null;
    /** Erfolgreiche Laeufe mit Loss, aelteste zuerst */
    runs: RunPoint[];
  };
  test: {
    state: StepState;
    metric: TestMetric | null;
    versionId: string | null;
    versionName: string | null;
    dataset: string | null;
    error: string | null;
    at: number | null;
    attempts: PipelineTest[];
  };
  hosting: { state: 'none' | 'ready' | 'loading' | 'idle' | 'failed' };
  next: NextKind | null;
  note: { kind: NoteKind; params: Record<string, string> } | null;
  /** Dauer und Zeitpunkt gehoeren zum selben, juengsten Ereignis */
  last: { kind: 'training' | 'test'; at: number; durationSec: number | null } | null;
}

export interface PipelineInput {
  models: PipelineModel[];
  datasets: PipelineDataset[];
  trainings: PipelineTraining[];
  tests: PipelineTest[];
  hosts: PipelineHost[];
}

const stepState = (status?: string): StepState => {
  switch (status) {
    case 'completed': return 'done';
    case 'failed': return 'failed';
    case 'stopped': return 'stopped';
    case 'running':
    case 'pending': return 'running';
    default: return 'none';
  }
};

const isValidLoss = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0;

function durationOf(job: { started_at?: string | null; completed_at?: string | null }): number | null {
  if (!job.started_at || !job.completed_at) return null;
  const sec = (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000;
  return Number.isFinite(sec) && sec >= 0 ? sec : null;
}

/** Kennzahl eines Tests. Accuracy zuerst; sonst die erste Zahl aus `metrics`, ohne feste Skala. */
export function testMetric(test: PipelineTest | undefined): TestMetric | null {
  const r = test?.results;
  if (!r) return null;
  if (typeof r.accuracy === 'number') return { label: 'accuracy', value: r.accuracy, fraction: r.accuracy };
  for (const [label, value] of Object.entries(r.metrics ?? {})) {
    if (typeof value === 'number' && Number.isFinite(value)) return { label, value, fraction: null };
  }
  return null;
}

function setupOf(job: PipelineTraining): string | null {
  const c = job.config;
  return c?.epochs != null && c.learning_rate != null && c.batch_size != null
    ? `${c.epochs}|${c.learning_rate}|${c.batch_size}`
    : null;
}

/**
 * Bester frueherer Lauf, wenn der letzte deutlich schlechter ist. Loss-Werte sind
 * nur bei gleichen Daten und gleichen Einstellungen vergleichbar.
 */
function regressionFrom(runs: RunPoint[]): number | null {
  const latest = runs[runs.length - 1];
  if (!latest?.dataset || !latest.setup) return null;
  const peers = runs.slice(0, -1).filter(r => r.dataset === latest.dataset && r.setup === latest.setup);
  if (peers.length === 0) return null;
  const best = Math.min(...peers.map(r => r.loss));
  return latest.loss >= best * REGRESSION_RATIO ? best : null;
}

const belongsTo = (job: { model_id?: string; model_name: string }, m: PipelineModel) =>
  job.model_id ? job.model_id === m.id : job.model_name === m.name;

function buildRow(m: PipelineModel, input: PipelineInput): ModelRow {
  const trainings = byNewest(input.trainings.filter(j => belongsTo(j, m)));
  const tests = byNewest(input.tests.filter(j => belongsTo(j, m)));
  const lastTrain = trainings[0];
  const lastDone = trainings.find(j => j.status === 'completed' && isValidLoss(j.progress?.train_loss));
  const lastTest = tests[0];
  const lastPassed = tests.find(j => j.status === 'completed');

  const ownDataset = input.datasets.find(d => d.model_id === m.id);
  const data = lastTrain?.dataset_name
    ? { name: lastTrain.dataset_name, id: lastTrain.dataset_id ?? null }
    : lastTest?.dataset_name
      ? { name: lastTest.dataset_name, id: lastTest.dataset_id ?? null }
      : { name: ownDataset?.name ?? null, id: ownDataset?.id ?? null };

  const runs: RunPoint[] = trainings
    .filter(j => j.status === 'completed' && isValidLoss(j.progress?.train_loss))
    .map(j => ({
      id: j.id,
      at: endedAt(j),
      loss: j.progress!.train_loss as number,
      durationSec: durationOf(j),
      dataset: j.dataset_name ?? '',
      setup: setupOf(j),
    }))
    .reverse();

  const trainState = stepState(lastTrain?.status);
  const trainingLoss = isValidLoss(lastDone?.progress?.train_loss) ? lastDone!.progress!.train_loss as number : null;
  const valLoss = isValidLoss(lastDone?.progress?.val_loss) ? lastDone!.progress!.val_loss as number : null;
  const testState = stepState(lastTest?.status);
  const metric = testMetric(lastPassed);

  const hostStates = input.hosts.filter(h => h.model_id === m.id).map(h => h.status);
  const hosting: ModelRow['hosting'] = {
    state: hostStates.includes('ready') ? 'ready'
      : hostStates.includes('loading') ? 'loading'
        : hostStates.includes('error') ? 'failed'
          : hostStates.length > 0 ? 'idle'
            : 'none',
  };

  // Ein Test am aktuellen Stand zaehlt nur, wenn nach ihm nicht neu trainiert wurde.
  const retrainedSinceTest = !!lastTest && !!lastDone && endedAt(lastDone) > endedAt(lastTest);
  const testOpen = testState === 'none' || retrainedSinceTest;

  let next: NextKind | null = null;
  if (trainState === 'running' || testState === 'running') next = null;
  else if (trainState === 'failed' || trainState === 'stopped') next = 'retryTrain';
  else if (!retrainedSinceTest && (testState === 'failed' || testState === 'stopped')) next = 'retryTest';
  else if (!data.name) next = 'addDataset';
  else if (testOpen) next = 'startTest';
  else if (testState === 'done' && hosting.state === 'none') next = 'host';

  const regressedFrom = trainState === 'done' ? regressionFrom(runs) : null;

  let note: ModelRow['note'] = null;
  if (trainState === 'failed') note = { kind: 'trainingFailed', params: { error: lastTrain?.error ?? '' } };
  else if (trainState === 'stopped') note = { kind: 'trainingStopped', params: {} };
  else if (!retrainedSinceTest && testState === 'failed') note = { kind: 'testFailed', params: { error: lastTest?.error ?? '' } };
  else if (!retrainedSinceTest && testState === 'stopped') note = { kind: 'testStopped', params: {} };
  else if (regressedFrom !== null) {
    note = { kind: 'regression', params: { best: regressedFrom.toFixed(4), last: runs[runs.length - 1].loss.toFixed(4) } };
  } else if (trainingLoss !== null && valLoss !== null && valLoss > trainingLoss * OVERFIT_RATIO) {
    note = { kind: 'overfitting', params: { val: valLoss.toFixed(4), train: trainingLoss.toFixed(4) } };
  } else if (metric?.fraction != null && metric.fraction < WEAK_ACCURACY) {
    note = { kind: 'weakResult', params: { value: (metric.fraction * 100).toFixed(1) } };
  }

  const group: GroupId = note ? 'attention'
    : hosting.state !== 'none' ? 'hosted'
      : 'next';

  const trainAt = lastTrain ? endedAt(lastTrain) : 0;
  const testAt = lastTest ? endedAt(lastTest) : 0;
  const last: ModelRow['last'] = !lastTrain && !lastTest ? null
    : testAt >= trainAt
      ? { kind: 'test', at: testAt, durationSec: lastTest ? durationOf(lastTest) : null }
      : { kind: 'training', at: trainAt, durationSec: durationOf(lastTrain!) };

  return {
    id: m.id,
    name: m.name,
    group,
    data,
    training: { state: trainState, loss: trainingLoss, valLoss, error: lastTrain?.error ?? null, runs },
    test: {
      state: testState,
      metric,
      versionId: lastPassed?.version_id ?? lastTest?.version_id ?? null,
      versionName: lastPassed?.version_name ?? lastTest?.version_name ?? null,
      dataset: lastPassed?.dataset_name ?? lastTest?.dataset_name ?? null,
      error: lastTest?.error ?? null,
      at: lastPassed ? endedAt(lastPassed) : lastTest ? endedAt(lastTest) : null,
      attempts: tests,
    },
    hosting,
    next,
    note,
    last,
  };
}

const NOTE_RANK: Record<NoteKind, number> = {
  trainingFailed: 0, testFailed: 1, regression: 2, overfitting: 3, trainingStopped: 4, testStopped: 5, weakResult: 6,
};

/** Probleme zuerst, sonst das zuletzt Bearbeitete; nie benutzte Modelle am Ende. */
export function buildPipeline(input: PipelineInput): ModelRow[] {
  return input.models.map(m => buildRow(m, input)).sort((a, b) => {
    const order: GroupId[] = ['attention', 'next', 'hosted'];
    const g = order.indexOf(a.group) - order.indexOf(b.group);
    if (g !== 0) return g;
    if (a.group === 'attention' && a.note && b.note) {
      const n = NOTE_RANK[a.note.kind] - NOTE_RANK[b.note.kind];
      if (n !== 0) return n;
    }
    return (b.last?.at ?? 0) - (a.last?.at ?? 0) || a.name.localeCompare(b.name);
  });
}

export type SectionKey = 'attention' | 'ready' | 'test' | 'next' | 'hosted';

/**
 * Gruppen der Startseite in Bildschirm-Reihenfolge, benannt nach dem naechsten
 * Schritt. Modelle ohne jeden Lauf stehen getrennt (zugeklappt), ausser es gibt
 * nur solche. Seite und Coach-Kontext lesen beide hier, damit Zahlen und Namen
 * uebereinstimmen.
 */
export function pipelineSections(rows: ModelRow[]): { sections: { key: SectionKey; rows: ModelRow[] }[]; unused: ModelRow[] } {
  const allUnused = rows.filter(r => r.group === 'next' && !r.last);
  const unused = allUnused.length === rows.length ? [] : allUnused;
  const others = rows.filter(r => r.group !== 'attention' && !unused.includes(r));
  return {
    sections: [
      { key: 'attention', rows: rows.filter(r => r.group === 'attention') },
      { key: 'ready', rows: others.filter(r => r.group === 'next' && r.next === 'host') },
      { key: 'test', rows: others.filter(r => r.group === 'next' && r.next === 'startTest') },
      { key: 'next', rows: others.filter(r => r.group === 'next' && r.next !== 'host' && r.next !== 'startTest') },
      { key: 'hosted', rows: others.filter(r => r.group === 'hosted') },
    ],
    unused,
  };
}

// ── Benchmarks ───────────────────────────────────────────────────────────

export interface CompareRow {
  row: ModelRow;
  best: boolean;
  /** Hinweis, dass das Ergebnis wegen gleicher Trainingsdaten zu hoch sein kann */
  sameData: boolean;
}

export interface CompareGroup {
  key: string;
  rows: CompareRow[];
  /** Alle Modelle mit Ergebnis auf demselben Test-Dataset */
  sameDataset: boolean;
}

/**
 * Gruppiert nach Aufgabe: Accuracy eines YOLO und eines Textmodells sind nicht
 * vergleichbar. "Best" gibt es nur bei mindestens zwei Ergebnissen auf demselben
 * Test-Dataset, sonst waere das Etikett eine Behauptung ohne Grundlage.
 */
export function buildCompare(rows: ModelRow[], taskKey: (modelId: string) => string): {
  groups: CompareGroup[];
  withoutResult: ModelRow[];
} {
  const groups = new Map<string, ModelRow[]>();
  const withoutResult: ModelRow[] = [];
  for (const row of rows) {
    const attempted = row.test.state !== 'none';
    if (!row.test.metric && !attempted) { withoutResult.push(row); continue; }
    const key = taskKey(row.id);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }

  const result: CompareGroup[] = [...groups.entries()].map(([key, members]) => {
    const scored = members.filter(r => r.test.metric?.fraction != null);
    const datasets = new Set(scored.map(r => r.test.dataset));
    const sameDataset = scored.length >= 2 && datasets.size === 1;
    const top = sameDataset ? Math.max(...scored.map(r => r.test.metric!.fraction as number)) : null;
    const sorted = [...members].sort((a, b) =>
      (b.test.metric?.fraction ?? -1) - (a.test.metric?.fraction ?? -1) || a.name.localeCompare(b.name));
    return {
      key,
      sameDataset,
      rows: sorted.map(row => ({
        row,
        best: top !== null && row.test.metric?.fraction === top,
        sameData: !!row.test.metric && !!row.test.dataset && row.test.dataset === row.data.name
          && row.training.state === 'done',
      })),
    };
  });
  return {
    groups: result.sort((a, b) => b.rows.length - a.rows.length || a.key.localeCompare(b.key)),
    withoutResult,
  };
}

export type HomeHeadline =
  | { kind: 'empty' }
  | { kind: 'training'; model: string }
  | { kind: 'attention'; count: number }
  | { kind: 'noData' }
  | { kind: 'clear' };

export function homeHeadline(rows: ModelRow[], runningModel: string | null, isRunning: boolean): HomeHeadline {
  if (rows.length === 0 && !isRunning) return { kind: 'empty' };
  if (isRunning && runningModel) return { kind: 'training', model: runningModel };
  const count = rows.filter(r => r.group === 'attention').length;
  if (count > 0) return { kind: 'attention', count };
  return rows.every(r => r.next === 'addDataset' && !r.last) ? { kind: 'noData' } : { kind: 'clear' };
}
