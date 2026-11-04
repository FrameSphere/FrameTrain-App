// homeBenchmarks.ts – Benchmark-Uebersicht der Startseite (Ansicht "Vergleich").
//
// Jedes Test-Dataset ist ein Benchmark. Pro Aufgabe stehen die Ergebnisse je
// Test-Dataset nebeneinander; gerankt wird nur, wo mindestens zwei Modelle auf
// demselben Test-Dataset einen Wert mit fester Skala haben. Zahlen ohne
// Aussagekraft (Test auf den Trainingsdaten, zu wenige Beispiele) bekommen
// einen Hinweis statt eines Platzes auf dem Podest.

import { testMetric, type ModelRow, type PipelineTest, type TestMetric } from './homePipeline';
import { endedAt } from './homeInsights';

/** Unter dieser Anzahl Testbeispiele sagt ein Ergebnis kaum etwas aus */
export const FEW_SAMPLES = 30;
/** So viele Modellfarben gibt es (tokens.css, --ft-series-N) */
export const SERIES_COUNT = 8;

export interface BenchResult {
  /** Modell und Version: jede getestete Version ist ein eigener Eintrag */
  key: string;
  modelId: string;
  model: string;
  versionId: string | null;
  version: string | null;
  metric: TestMetric;
  samples: number | null;
  msPerSample: number | null;
  /** Zeit pro Beispiel beim vorherigen Test derselben Version auf demselben Dataset */
  msBefore: number | null;
  at: number;
  sameData: boolean;
  fewSamples: boolean;
  rank: number | null;
}

/** Eine Version im Diagramm: Farbe vom Modell, Versionen heller bis kraeftig (neueste kraeftig). */
export interface BenchSeries {
  key: string;
  modelId: string;
  model: string;
  version: string | null;
  color: number;
  /** 1 = neueste Version des Modells, aeltere blasser */
  shade: number;
  /** Hat das Modell mehrere getestete Versionen? Dann wird die Version mitgenannt. */
  multi: boolean;
}

/** "v2 ist 2,1 Punkte besser als v1" – je Modell neueste gegen vorherige Version */
export interface VersionStep {
  model: string;
  newer: string;
  older: string;
  /** Prozentpunkte, positiv = besser */
  points: number;
}

export interface BenchDataset {
  name: string;
  results: BenchResult[];
  /** Mindestens zwei Ergebnisse mit fester Skala: Plaetze und Diagramm sind sinnvoll */
  ranked: boolean;
  steps: VersionStep[];
}

export interface BenchTask {
  key: string;
  /** Anzahl verschiedener Modelle mit Ergebnis */
  modelCount: number;
  series: BenchSeries[];
  datasets: BenchDataset[];
  /** Versucht, aber ohne Ergebnis (fehlgeschlagen, gestoppt, laeuft) */
  pending: ModelRow[];
}

const msOf = (test: PipelineTest): number | null => {
  const r = test.results as (PipelineTest['results'] & { average_inference_time?: number | null }) | null | undefined;
  const sec = r?.average_inference_time;
  return typeof sec === 'number' && Number.isFinite(sec) && sec > 0 ? sec * 1000 : null;
};

const samplesOf = (test: PipelineTest): number | null => {
  const n = (test.results as { total_samples?: unknown } | null | undefined)?.total_samples;
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
};

const versionKey = (job: PipelineTest) => job.version_id ?? job.version_name ?? 'latest';

export function buildBenchmarks(
  rows: ModelRow[],
  taskKey: (modelId: string) => string,
  hasTestSplit: (dataset: string) => boolean,
): { tasks: BenchTask[]; withoutResult: ModelRow[] } {
  type Acc = { results: Map<string, BenchResult[]>; pending: ModelRow[]; models: ModelRow[]; firstTested: Map<string, number> };
  const tasks = new Map<string, Acc>();
  const withoutResult: ModelRow[] = [];
  const taskOf = (key: string) => {
    const task = tasks.get(key) ?? { results: new Map(), pending: [], models: [], firstTested: new Map() };
    tasks.set(key, task);
    return task;
  };

  for (const row of rows) {
    const done = row.test.attempts.filter(j => j.status === 'completed' && testMetric(j));
    if (done.length === 0) {
      if (row.test.state === 'none') withoutResult.push(row);
      else taskOf(taskKey(row.id)).pending.push(row);
      continue;
    }
    const task = taskOf(taskKey(row.id));
    task.models.push(row);
    // attempts sind neueste zuerst: der erste Treffer je Dataset und Version ist der aktuelle Stand.
    const groups = new Map<string, PipelineTest[]>();
    for (const job of done) {
      const id = `${job.dataset_name ?? '–'}\u0000${versionKey(job)}`;
      groups.set(id, [...(groups.get(id) ?? []), job]);
      const sk = `${row.id}:${versionKey(job)}`;
      task.firstTested.set(sk, Math.min(task.firstTested.get(sk) ?? Infinity, endedAt(job)));
    }
    for (const jobs of groups.values()) {
      const latest = jobs[0];
      const dataset = latest.dataset_name ?? '–';
      const samples = samplesOf(latest);
      const ms = msOf(latest);
      const before = jobs[1] ? msOf(jobs[1]) : null;
      const result: BenchResult = {
        key: `${row.id}:${versionKey(latest)}`,
        modelId: row.id,
        model: row.name,
        versionId: latest.version_id ?? null,
        version: latest.version_name ?? null,
        metric: testMetric(latest)!,
        samples,
        msPerSample: ms,
        msBefore: before !== null && ms !== null && Math.abs(before - ms) / before > 0.15 ? before : null,
        at: endedAt(latest),
        sameData: row.training.state === 'done' && row.data.name === dataset && !hasTestSplit(dataset),
        fewSamples: samples !== null && samples < FEW_SAMPLES,
        rank: null,
      };
      task.results.set(dataset, [...(task.results.get(dataset) ?? []), result]);
    }
  }

  const out: BenchTask[] = [...tasks.entries()].map(([key, task]) => {
    const models = [...task.models].sort((a, b) => a.name.localeCompare(b.name));
    const all = [...task.results.values()].flat();
    const series: BenchSeries[] = models.flatMap((m, i) => {
      // Versionen in der Reihenfolge, in der sie zuerst getestet wurden; die neueste ist die kraeftigste.
      const keys = [...new Set(all.filter(r => r.modelId === m.id).map(r => r.key))]
        .sort((a, b) => (task.firstTested.get(a) ?? 0) - (task.firstTested.get(b) ?? 0));
      return keys.map((k, idx) => {
        const sample = all.find(r => r.key === k)!;
        const fromNewest = keys.length - 1 - idx;
        return {
          key: k, modelId: m.id, model: m.name, version: sample.version,
          color: (i % SERIES_COUNT) + 1,
          shade: fromNewest === 0 ? 1 : fromNewest === 1 ? 0.7 : 0.45,
          multi: keys.length > 1,
        };
      });
    });
    const order = new Map(series.map((s, i) => [s.key, i]));
    const datasets: BenchDataset[] = [...task.results.entries()].map(([name, results]) => {
      const scored = results.filter(r => r.metric.fraction !== null);
      const ranked = scored.length >= 2 && scored.length === results.length;
      const sorted = [...results].sort((a, b) => (b.metric.fraction ?? -1) - (a.metric.fraction ?? -1) || a.model.localeCompare(b.model));
      if (ranked) {
        let place = 0;
        let last: number | null = null;
        sorted.forEach((r, i) => {
          if (r.metric.fraction !== last) { place = i + 1; last = r.metric.fraction; }
          r.rank = place;
        });
      }
      const steps: VersionStep[] = [];
      for (const m of models) {
        const versions = scored.filter(r => r.modelId === m.id).sort((a, b) => (order.get(a.key) ?? 0) - (order.get(b.key) ?? 0));
        if (versions.length < 2) continue;
        const [older, newer] = versions.slice(-2);
        steps.push({
          model: m.name, newer: newer.version ?? '', older: older.version ?? '',
          points: (newer.metric.fraction! - older.metric.fraction!) * 100,
        });
      }
      return { name, results: sorted, ranked, steps };
    }).sort((a, b) => b.results.length - a.results.length || a.name.localeCompare(b.name));
    return { key, modelCount: models.length, series, datasets, pending: task.pending };
  }).sort((a, b) => b.modelCount - a.modelCount || a.key.localeCompare(b.key));

  return { tasks: out, withoutResult };
}
