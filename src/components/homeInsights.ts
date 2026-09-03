// Ableitungen für die Startseite.
//
// Bewusst als reine Funktionen ohne React/Tauri: das ist der Teil, der still
// falsch sein kann (falsches Modell gewarnt, Trend verdreht) und deshalb Tests
// braucht. Die Komponente uebersetzt und rendert nur noch.

import type { AppView } from '../ui/navigationEvents';

// ── Eingaben (Ausschnitte der Backend-Structs) ──────────────────────────────

export interface TrainingLike {
  id: string;
  model_id?: string;
  model_name: string;
  dataset_name?: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  progress?: { train_loss?: number; val_loss?: number | null } | null;
}

export interface TestLike {
  id: string;
  model_id?: string;
  model_name: string;
  version_name: string;
  dataset_name?: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  results?: { accuracy?: number | null } | null;
}

export interface ModelLike { id: string; name: string; version_count?: number; total_size?: number }
export interface DatasetLike { id: string; name: string; training_count?: number }

export interface InsightInput {
  trainings: TrainingLike[];
  tests: TestLike[];
  models: ModelLike[];
  datasets: DatasetLike[];
  /** Läuft gerade ein Training? */
  hasActiveTraining: boolean;
  /** Ist der Ruhezustand des Rechners unterdrückt? */
  sleepPrevented: boolean;
}

// ── Hinweise ───────────────────────────────────────────────────────────────

export type InsightKind =
  | 'sleepRisk'
  | 'trainingFailed'
  | 'overfitting'
  | 'weakAccuracy'
  | 'untestedModels'
  | 'unusedDatasets';

export interface Insight {
  id: string;
  kind: InsightKind;
  severity: 'warn' | 'info';
  /** Platzhalter für den i18n-Text (`home.insights.<kind>`). */
  params: Record<string, string>;
  /** Wohin der Klick führt. */
  target: AppView;
}

/** Zeitpunkt, an dem ein Job zu Ende ging (completed_at, sonst created_at). */
export function endedAt(job: { completed_at?: string | null; created_at: string }): number {
  const raw = job.completed_at ?? job.created_at;
  const t = new Date(raw).getTime();
  return Number.isNaN(t) ? 0 : t;
}

/** Neueste zuerst. */
export function byNewest<T extends { completed_at?: string | null; created_at: string }>(jobs: T[]): T[] {
  return [...jobs].sort((a, b) => endedAt(b) - endedAt(a));
}

/** Ab diesem Verhaeltnis val_loss/train_loss sprechen wir von Overfitting-Verdacht. */
export const OVERFIT_RATIO = 1.5;
/** Unterhalb dieser Accuracy weisen wir auf ein schwaches Ergebnis hin. */
export const WEAK_ACCURACY = 0.6;
/** Mehr Hinweise auf einmal ueberfordern eher, als dass sie helfen. */
export const MAX_INSIGHTS = 5;

/**
 * Leitet die Hinweise fuer "Braucht Aufmerksamkeit" ab.
 *
 * Reihenfolge: Warnungen vor Infos, innerhalb dessen die Reihenfolge der
 * Regeln — die dringendste (schlafender Rechner waehrend eines Nachtlaufs)
 * steht oben.
 */
export function buildInsights(input: InsightInput): Insight[] {
  const { trainings, tests, models, datasets, hasActiveTraining, sleepPrevented } = input;
  const out: Insight[] = [];

  // 1) Ein Nachtlauf nuetzt nichts, wenn der Rechner dabei einschlaeft.
  if (hasActiveTraining && !sleepPrevented) {
    out.push({ id: 'sleepRisk', kind: 'sleepRisk', severity: 'warn', params: {}, target: 'settings' });
  }

  // 2) Letzter Lauf je Modell fehlgeschlagen und seitdem kein erfolgreicher.
  const latestPerModel = new Map<string, TrainingLike>();
  for (const job of byNewest(trainings)) {
    const key = job.model_id || job.model_name;
    if (!latestPerModel.has(key)) latestPerModel.set(key, job);
  }
  for (const job of latestPerModel.values()) {
    if (job.status === 'failed' || job.status === 'stopped') {
      out.push({
        id: `failed:${job.id}`,
        kind: 'trainingFailed',
        severity: 'warn',
        params: { model: job.model_name },
        target: 'training',
      });
    }
  }

  // 3) Overfitting-Verdacht im juengsten abgeschlossenen Lauf.
  const lastDone = byNewest(trainings).find(j => j.status === 'completed');
  const trainLoss = lastDone?.progress?.train_loss;
  const valLoss = lastDone?.progress?.val_loss;
  if (
    lastDone && typeof trainLoss === 'number' && trainLoss > 0 &&
    typeof valLoss === 'number' && valLoss > trainLoss * OVERFIT_RATIO
  ) {
    out.push({
      id: `overfit:${lastDone.id}`,
      kind: 'overfitting',
      severity: 'info',
      params: { model: lastDone.model_name, train: trainLoss.toFixed(4), val: valLoss.toFixed(4) },
      target: 'analysis',
    });
  }

  // 4) Schwaches Testergebnis im juengsten abgeschlossenen Test.
  const lastTest = byNewest(tests).find(j => j.status === 'completed');
  const acc = lastTest?.results?.accuracy;
  if (lastTest && typeof acc === 'number' && acc < WEAK_ACCURACY) {
    out.push({
      id: `weak:${lastTest.id}`,
      kind: 'weakAccuracy',
      severity: 'info',
      params: { model: lastTest.model_name, value: (acc * 100).toFixed(1) },
      target: 'tests',
    });
  }

  // 5) Modelle, fuer die es noch keinen abgeschlossenen Test gibt.
  const testedModels = new Set(
    tests.filter(t => t.status === 'completed').map(t => t.model_id || t.model_name),
  );
  const untested = models.filter(m => !testedModels.has(m.id) && !testedModels.has(m.name));
  if (models.length > 0 && untested.length > 0) {
    out.push({
      id: 'untested',
      kind: 'untestedModels',
      severity: 'info',
      params: { count: String(untested.length), first: untested[0].name },
      target: 'tests',
    });
  }

  // 6) Datasets, die noch nie in einem Training benutzt wurden.
  const unused = datasets.filter(d => (d.training_count ?? 0) === 0);
  if (datasets.length > 0 && unused.length > 0) {
    out.push({
      id: 'unused',
      kind: 'unusedDatasets',
      severity: 'info',
      params: { count: String(unused.length), first: unused[0].name },
      target: 'dataset',
    });
  }

  const rank = (i: Insight) => (i.severity === 'warn' ? 0 : 1);
  return out.sort((a, b) => rank(a) - rank(b)).slice(0, MAX_INSIGHTS);
}

// ── Loss-Trend ─────────────────────────────────────────────────────────────

export interface TrendPoint {
  id: string;
  label: string;
  loss: number;
  at: number;
}

/**
 * Final-Loss der letzten abgeschlossenen Trainings, aelteste zuerst — die
 * Leserichtung eines Verlaufs. Laeufe ohne verwertbaren Loss (0 oder fehlend)
 * fallen raus, sonst kippt die Kurve auf die Grundlinie.
 */
export function lossTrend(trainings: TrainingLike[], limit = 10): TrendPoint[] {
  return byNewest(trainings)
    .filter(j => j.status === 'completed' && typeof j.progress?.train_loss === 'number' && j.progress.train_loss > 0)
    .slice(0, limit)
    .map(j => ({ id: j.id, label: j.model_name, loss: j.progress!.train_loss as number, at: endedAt(j) }))
    .reverse();
}

/** Verbesserung des Trends in Prozent (positiv = Loss gefallen), sonst null. */
export function trendImprovementPct(points: TrendPoint[]): number | null {
  if (points.length < 2) return null;
  const first = points[0].loss;
  const last = points[points.length - 1].loss;
  if (first <= 0 || first === last) return null;
  return ((first - last) / first) * 100;
}

/**
 * Punkte einer Sparkline in einem `width` x `height` grossen Feld.
 *
 * Achsenrichtung wie bei jeder Loss-Kurve: hoher Loss oben, niedriger unten —
 * ein fallender Loss ergibt also eine fallende Linie. (SVG zaehlt y von oben,
 * deshalb die Umkehrung.) Ein einzelner Punkt oder eine flache Reihe landet
 * auf der Mittellinie statt am Rand.
 */
export function sparklinePoints(points: TrendPoint[], width: number, height: number, pad = 2): string {
  if (points.length === 0) return '';
  const losses = points.map(p => p.loss);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = max - min;
  const usableW = width - pad * 2;
  const usableH = height - pad * 2;
  return points
    .map((p, i) => {
      const x = points.length === 1 ? width / 2 : pad + (i / (points.length - 1)) * usableW;
      const y = span === 0 ? height / 2 : pad + ((max - p.loss) / span) * usableH;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

// ── Bestenliste ────────────────────────────────────────────────────────────

export interface RankedResult {
  id: string;
  model_name: string;
  version_name: string;
  dataset_name: string;
  accuracy: number;
}

/**
 * Beste Testergebnisse, absteigend nach Accuracy. Pro Modell-Version zaehlt nur
 * der beste Lauf — sonst belegt ein oft getestetes Modell die ganze Liste.
 */
export function topResults(tests: TestLike[], limit = 3): RankedResult[] {
  const best = new Map<string, RankedResult>();
  for (const t of tests) {
    const acc = t.results?.accuracy;
    if (t.status !== 'completed' || typeof acc !== 'number') continue;
    const key = `${t.model_id || t.model_name}|${t.version_name}`;
    const row: RankedResult = {
      id: t.id,
      model_name: t.model_name,
      version_name: t.version_name,
      dataset_name: t.dataset_name ?? '',
      accuracy: acc,
    };
    const prev = best.get(key);
    if (!prev || acc > prev.accuracy) best.set(key, row);
  }
  return [...best.values()].sort((a, b) => b.accuracy - a.accuracy).slice(0, limit);
}

// ── Faktenblatt für das KI-Briefing ────────────────────────────────────────

/**
 * Verdichtet den Stand zu einem kurzen, stabilen Faktenblock. Stabil ist
 * wichtig: derselbe Stand ergibt denselben Text und damit denselben Cache-
 * Schluessel — so kostet ein erneuter Blick auf die Startseite keine Tokens.
 */
export function buildBriefingFacts(input: InsightInput): string {
  const { trainings, tests, models, datasets, hasActiveTraining } = input;
  const lines: string[] = [];

  lines.push(`Modelle: ${models.length}`);
  lines.push(`Datasets: ${datasets.length}${datasets.filter(d => (d.training_count ?? 0) === 0).length ? ` (davon ${datasets.filter(d => (d.training_count ?? 0) === 0).length} nie benutzt)` : ''}`);
  lines.push(`Trainings insgesamt: ${trainings.length}`);
  lines.push(`Laeuft gerade ein Training: ${hasActiveTraining ? 'ja' : 'nein'}`);

  const recent = byNewest(trainings).slice(0, 8);
  if (recent.length) {
    lines.push('', 'Letzte Trainings (neueste zuerst):');
    for (const j of recent) {
      const loss = typeof j.progress?.train_loss === 'number' ? ` train_loss=${j.progress.train_loss.toFixed(4)}` : '';
      const val = typeof j.progress?.val_loss === 'number' ? ` val_loss=${j.progress.val_loss.toFixed(4)}` : '';
      lines.push(`- ${j.model_name} auf ${j.dataset_name ?? '?'}: ${j.status}${loss}${val}`);
    }
  }

  const recentTests = byNewest(tests).slice(0, 5);
  if (recentTests.length) {
    lines.push('', 'Letzte Tests (neueste zuerst):');
    for (const t of recentTests) {
      const acc = typeof t.results?.accuracy === 'number' ? ` accuracy=${(t.results.accuracy * 100).toFixed(1)}%` : '';
      lines.push(`- ${t.model_name} ${t.version_name} auf ${t.dataset_name ?? '?'}: ${t.status}${acc}`);
    }
  }

  const insights = buildInsights(input);
  if (insights.length) {
    lines.push('', 'Automatisch erkannte Auffaelligkeiten:');
    for (const i of insights) {
      lines.push(`- ${i.kind} ${JSON.stringify(i.params)}`);
    }
  }

  return lines.join('\n');
}

/** Kurzer, stabiler Hash über die Fakten — Cache-Schluessel fuers Briefing. */
export function factsHash(facts: string): string {
  let h = 5381;
  for (let i = 0; i < facts.length; i++) h = ((h << 5) + h + facts.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

/**
 * Trennt die Handlungsempfehlung vom Rest des Briefings.
 *
 * Der Prompt verlangt als letzte Zeile "**Naechster Schritt:** …". Genau die
 * wird hier abgetrennt, damit sie in der Karte hervorgehoben stehen kann statt
 * als vierter Absatz unterzugehen. Haelt sich ein Modell nicht daran, bleibt
 * `nextStep` leer und der Text wird unveraendert am Stueck gerendert.
 */
export function splitBriefing(text: string): { body: string; nextStep: string } {
  const lines = text.trimEnd().split('\n');
  let last = lines.length - 1;
  while (last >= 0 && lines[last].trim() === '') last--;
  if (last < 0) return { body: text, nextStep: '' };

  const candidate = lines[last].trim();
  // Muss als fetter Label beginnen ("**… :**") — sonst ist es ein normaler Satz.
  if (!/^\*\*[^*]+:\*\*/.test(candidate)) return { body: text, nextStep: '' };
  // Eine Zeile allein ist kein Briefing: dann lieber alles am Stueck zeigen.
  if (lines.slice(0, last).every(l => l.trim() === '')) return { body: text, nextStep: '' };

  return { body: lines.slice(0, last).join('\n').trimEnd(), nextStep: candidate };
}
