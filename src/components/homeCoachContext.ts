// homeCoachContext.ts – Live-Zustand der Startseite fuer den AI-Coach.
//
// Der Coach soll dieselbe Seite sehen wie der Nutzer: dieselben Gruppen in
// derselben Reihenfolge, dieselben Zahlen und dieselben Woerter in der Sprache
// der Oberflaeche. Vorher gingen interne Kennungen raus ("startTest", "failed"),
// der Coach zitierte sie und zaehlte Modelle anders als die Seite.

import { pipelineSections, type CompareGroup, type ModelRow, type NextKind } from './homePipeline';

type T = (key: string, params?: Record<string, string>) => string;

export interface HomeCoachInput {
  language: 'de' | 'en';
  t: T;
  /** Relative Zeit wie in der Liste */
  ago: (ms: number | null | undefined) => string;
  number: (value: number, digits: number) => string;
  title: string;
  view: 'pipeline' | 'compare';
  rows: ModelRow[];
  running: { model: string; percent: number } | null;
  sleepRisk: boolean;
  compare: { groups: CompareGroup[]; withoutResult: ModelRow[] };
  groupLabel: (key: string) => string;
}

const WORDS = {
  de: { page: 'Start', heading: 'Überschrift', view: 'Ansicht', running: 'Läuft gerade', groups: 'Gruppen von oben nach unten', collapsed: 'zugeklappt', button: 'Knopf', in: 'in Spalte', hint: 'Hinweis', runs: 'Läufe', lastDone: 'letzter fertiger Lauf', compare: 'Vergleich (Ansicht "Compare")', best: 'Bestes', sleep: 'Leiste unter dem laufenden Training' },
  en: { page: 'Home', heading: 'Heading', view: 'View', running: 'Running now', groups: 'Groups from top to bottom', collapsed: 'collapsed', button: 'Button', in: 'in column', hint: 'Hint', runs: 'runs', lastDone: 'last finished run', compare: 'Comparison ("Compare" view)', best: 'Best', sleep: 'Bar below the running training' },
} as const;

const NEXT_COLUMN: Record<NextKind, string> = {
  addDataset: 'data', train: 'training', retryTrain: 'training',
  startTest: 'test', retryTest: 'test', host: 'hosting',
};

function noteText(row: ModelRow, t: T): string | null {
  const note = row.note;
  if (!note) return null;
  if ((note.kind === 'trainingFailed' || note.kind === 'testFailed') && note.params.error) return note.params.error;
  return t(`home.pipeline.note.${note.kind}`, note.params);
}

function describeRow(row: ModelRow, input: HomeCoachInput): string {
  const { t, number, ago, language } = input;
  const w = WORDS[language];
  const col = (k: string) => t(`home.pipeline.cols.${k}`);
  const state = (s: string) => t(`home.pipeline.state.${s}`);

  const training = row.training.state === 'running' ? state('running')
    : row.training.state === 'failed' || row.training.state === 'stopped'
      ? `${state(row.training.state)}${row.training.loss !== null ? ` (${w.lastDone}: loss ${number(row.training.loss, 4)})` : ''}`
      : row.training.loss !== null
        ? `loss ${number(row.training.loss, 4)}${row.training.runs.length > 1 ? `, ${row.training.runs.length} ${w.runs}` : ''}`
        : row.next === 'addDataset' ? '–' : t('home.pipeline.original');
  const metric = row.test.metric;
  const test = row.test.state === 'running' ? state('running')
    : row.next === 'retryTest' ? state(row.test.state)
      : metric
        ? `${metric.fraction !== null ? `${number(metric.fraction * 100, 1)}%` : `${metric.label} ${number(metric.value, 3)}`}${row.test.dataset ? ` (${row.test.dataset})` : ''}`
        : row.test.state === 'done' ? state('done') : '–';
  const hosting = row.hosting.state === 'none' ? '–'
    : row.hosting.state === 'failed' ? state('failed')
      : t(`home.pipeline.hosting.${row.hosting.state === 'ready' ? 'ready' : row.hosting.state === 'loading' ? 'loading' : 'idle'}`);

  const parts = [
    `${col('data')}: ${row.data.name ?? '–'}`,
    `${col('training')}: ${training}`,
    `${col('test')}: ${test}`,
    `${col('hosting')}: ${hosting}`,
  ];
  if (row.next) parts.push(`${w.button} "${t(`home.pipeline.next.${row.next}`)}" ${w.in} ${col(NEXT_COLUMN[row.next])}`);
  parts.push(`${col('last')}: ${row.last ? ago(row.last.at) : '–'}`);
  const note = noteText(row, t);
  return `  - ${row.name} | ${parts.join(' | ')}${note ? `\n    ${w.hint}: ${note}` : ''}`;
}

export function buildHomeCoachContext(input: HomeCoachInput): string {
  const { t, language } = input;
  const w = WORDS[language];
  const lines: string[] = [
    `=== FrameTrain ${w.page} ===`,
    `${w.heading}: "${input.title}"`,
    `${w.view}: ${input.view === 'pipeline' ? t('home.pipeline.pipeline') : t('home.pipeline.compare.title')}`,
  ];
  if (input.running) lines.push(`${w.running}: ${input.running.model}, ${input.running.percent}%`);
  if (input.sleepRisk) lines.push(`${w.sleep}: ${t('home.pipeline.sleepRisk')}`);

  const { sections, unused } = pipelineSections(input.rows);
  lines.push('', `--- ${w.groups} ---`);
  for (const s of sections) {
    if (s.rows.length === 0) continue;
    lines.push(`## ${t(`home.pipeline.group.${s.key}`)} (${s.rows.length})`);
    for (const row of s.rows) lines.push(describeRow(row, input));
  }
  if (unused.length > 0) {
    lines.push(`## ${t('home.pipeline.group.unused')} (${unused.length}, ${w.collapsed}): ${unused.map(r => r.name).join(', ')}`);
  }

  const scored = input.compare.groups.filter(g => g.rows.some(r => r.row.test.metric));
  if (scored.length > 0) {
    lines.push('', `--- ${w.compare} ---`);
    for (const g of scored) {
      const sub = g.rows.filter(r => r.row.test.metric).length > 1
        ? ` – ${t(g.sameDataset ? 'home.pipeline.compare.sameDataset' : 'home.pipeline.compare.mixedDatasets')}`
        : '';
      lines.push(`## ${input.groupLabel(g.key)}${sub}`);
      for (const { row, best, sameData } of g.rows) {
        const m = row.test.metric;
        if (!m) continue;
        const value = m.fraction !== null ? `${input.number(m.fraction * 100, 1)}%` : `${m.label} ${input.number(m.value, 3)}`;
        lines.push(`  - ${row.name}: ${value} (${row.test.dataset ?? '–'})${best ? ` – ${w.best}` : ''}${sameData ? `\n    ${w.hint}: ${t('home.pipeline.compare.sameData')}` : ''}`);
      }
    }
  }
  return lines.join('\n');
}
