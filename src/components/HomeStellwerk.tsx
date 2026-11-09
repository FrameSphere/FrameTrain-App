// HomeStellwerk.tsx – Bausteine der Startseite.
//
// Jedes Modell ist eine Zeile durch Daten, Training, Test und Hosting. Der Stand
// liest sich an den Spalten, der Knopf steht in der Spalte des naechsten Schritts.
// Reihenfolge und Zustaende kommen aus homePipeline.ts.

import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, XCircle, StopCircle, Loader2 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { navigateToModel, type AppView } from '../ui/navigationEvents';
import { pipelineSections, type ModelRow, type NextKind, type RunPoint, type StepState, type TestMetric } from './homePipeline';

type Format = {
  date: (ms: number | null | undefined) => string;
  /** Relative Zeit fuer die Liste ("vor 5 Tagen"); das Datum steht im Tooltip */
  ago: (ms: number | null | undefined) => string;
  duration: (seconds: number | null | undefined) => string;
  number: (value: number, digits: number) => string;
};

const NEXT_TARGET: Record<NextKind, AppView> = {
  addDataset: 'dataset', train: 'training', retryTrain: 'training',
  startTest: 'tests', retryTest: 'tests', host: 'hosting',
};

function ActionButton({ row, kind }: { row: ModelRow; kind: NextKind }) {
  const { t } = useLanguage();
  return (
    <button
      className="ft-go"
      onClick={() => navigateToModel(NEXT_TARGET[kind], { modelId: row.id, datasetId: row.data.id, versionId: row.test.versionId })}
    >
      {t(`home.pipeline.next.${kind}`)}
    </button>
  );
}

type Tone = 'bad' | 'warn' | 'live';

/** Status: Zeichen und Wort, nie klickbar. */
function Status({ tone, label }: { tone: Tone; label: string }) {
  const glyph = tone === 'bad' ? <XCircle className="w-3.5 h-3.5" />
    : tone === 'warn' ? <StopCircle className="w-3.5 h-3.5" />
      : <Loader2 className="w-3.5 h-3.5 animate-spin" />;
  return (
    <span className={`ft-st is-${tone}`} title={label}>
      <span aria-hidden="true" className="inline-flex">{glyph}</span>
      <span className="ft-st-word">{label}</span>
    </span>
  );
}

function FailedOrStopped({ state }: { state: StepState }) {
  const { t } = useLanguage();
  return state === 'failed'
    ? <Status tone="bad" label={t('home.pipeline.state.failed')} />
    : <Status tone="warn" label={t('home.pipeline.state.stopped')} />;
}

function Running() {
  const { t } = useLanguage();
  return <Status tone="live" label={t('home.pipeline.state.running')} />;
}

function metricText(metric: TestMetric, f: Format): string {
  return metric.fraction !== null ? `${f.number(metric.fraction * 100, 1)}%` : f.number(metric.value, 3);
}

const Cell = ({ children }: { children?: ReactNode }) => <div className="ft-cell">{children}</div>;

function DataCell({ row }: { row: ModelRow }) {
  if (row.next === 'addDataset') return <Cell><ActionButton row={row} kind="addDataset" /></Cell>;
  if (!row.data.name) return <Cell><span className="ft-quiet">–</span></Cell>;
  return <Cell><span className="ft-ds" title={row.data.name}>{row.data.name}</span></Cell>;
}

function TrainingCell({ row, f }: { row: ModelRow; f: Format }) {
  const { t } = useLanguage();
  const { state, loss } = row.training;
  if (state === 'running') return <Cell><Running /></Cell>;
  if (state === 'failed' || state === 'stopped') return <Cell><FailedOrStopped state={state} /><ActionButton row={row} kind="retryTrain" /></Cell>;
  if (loss !== null) {
    const warn = row.note?.kind === 'overfitting' || row.note?.kind === 'regression';
    return <Cell><span className={`ft-v ${warn ? 'is-warn' : ''}`}>{f.number(loss, 4)}</span></Cell>;
  }
  if (row.next === 'addDataset') return <Cell />;
  return <Cell><span className="ft-quiet">{t('home.pipeline.original')}</span></Cell>;
}

function TestCell({ row, f, quiet }: { row: ModelRow; f: Format; quiet?: boolean }) {
  const { t } = useLanguage();
  const { state, metric } = row.test;
  if (state === 'running') return <Cell><Running /></Cell>;
  if (row.next === 'retryTest') return <Cell><FailedOrStopped state={state} /><ActionButton row={row} kind="retryTest" /></Cell>;
  if (metric) {
    const weak = row.note?.kind === 'weakResult';
    return (
      <Cell>
        <span className={`ft-v ${weak ? 'is-warn' : ''}`} title={metric.fraction === null ? metric.label : undefined}>{metricText(metric, f)}</span>
        {row.next === 'startTest' && <ActionButton row={row} kind="startTest" />}
      </Cell>
    );
  }
  if (row.next === 'startTest') {
    // In einer Gruppe, in der jede Zeile denselben Knopf haette, steht er erst beim Zeigen auf die Zeile.
    if (quiet) {
      return (
        <Cell>
          <span className="ft-quiet ft-rest-label">{t('home.pipeline.state.notTested')}</span>
          <span className="ft-hover-go"><ActionButton row={row} kind="startTest" /></span>
        </Cell>
      );
    }
    return <Cell><ActionButton row={row} kind="startTest" /></Cell>;
  }
  if (state === 'done') return <Cell><span className="ft-quiet">{t('home.pipeline.state.done')}</span></Cell>;
  return <Cell />;
}

function HostingCell({ row }: { row: ModelRow }) {
  const { t } = useLanguage();
  const state = row.hosting.state;
  if (state === 'failed') return <Cell><Status tone="bad" label={t('home.pipeline.state.failed')} /></Cell>;
  if (state === 'loading') return <Cell><Status tone="live" label={t('home.pipeline.hosting.loading')} /></Cell>;
  if (state === 'ready' || state === 'idle') {
    return (
      <Cell>
        <span className={`ft-dot ${state === 'ready' ? 'is-on' : ''}`} aria-hidden="true" />
        <span className="ft-quiet">{t(`home.pipeline.hosting.${state}`)}</span>
      </Cell>
    );
  }
  if (row.next === 'host') return <Cell><ActionButton row={row} kind="host" /></Cell>;
  return <Cell />;
}

function noteText(row: ModelRow, t: (key: string, p?: Record<string, string>) => string): string | null {
  const note = row.note;
  if (!note) return null;
  if ((note.kind === 'trainingFailed' || note.kind === 'testFailed') && note.params.error) return note.params.error;
  return t(`home.pipeline.note.${note.kind}`, note.params);
}

function RunChart({ runs, f }: { runs: RunPoint[]; f: Format }) {
  const W = 560, H = 150, left = 42, right = 16, top = 12, bottom = 22;
  const losses = runs.map(r => r.loss);
  const max = Math.max(...losses), min = Math.min(...losses), span = max - min || 1;
  const x = (i: number) => left + (runs.length === 1 ? 0 : (i / (runs.length - 1)) * (W - left - right));
  const y = (v: number) => top + ((max - v) / span) * (H - top - bottom);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[150px] block" role="img" aria-label={`${f.number(runs[0].loss, 4)} – ${f.number(runs[runs.length - 1].loss, 4)}`}>
      {[max, min].map(v => (
        <g key={v}>
          <line x1={left} x2={W - right} y1={y(v)} y2={y(v)} stroke="var(--ft-line)" />
          <text x={0} y={y(v) + 4} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="var(--ft-faint)">{v.toFixed(2)}</text>
        </g>
      ))}
      <polyline fill="none" stroke="var(--ft-out)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" points={runs.map((r, i) => `${x(i)},${y(r.loss)}`).join(' ')} />
      {runs.map((r, i) => <circle key={r.id} cx={x(i)} cy={y(r.loss)} r="3.5" fill="var(--ft-out)" />)}
    </svg>
  );
}

function RowPanel({ row, f }: { row: ModelRow; f: Format }) {
  const { t } = useLanguage();
  const runs = row.training.runs;
  const tests = row.test.attempts.slice(0, 3);
  const analysisLink = (
    <button className="ft-ref" onClick={() => navigateToModel('analysis', { modelId: row.id })}>{t('home.pipeline.panel.openAnalysis')}</button>
  );
  const runList = (
    <>
      {runs.length > 0 && (
        <div className="ft-pl is-head">
          <span>{t('home.pipeline.panel.finished')}</span>
          <span>{t('home.pipeline.panel.loss')}</span>
          <span>{t('home.pipeline.cols.duration')}</span>
        </div>
      )}
      {[...runs].reverse().slice(0, 4).map(r => (
        <div key={r.id} className="ft-pl">
          <span>{f.date(r.at)}</span>
          <span className="font-mono tabular-nums">{f.number(r.loss, 4)}</span>
          <span className="font-mono tabular-nums">{f.duration(r.durationSec)}</span>
        </div>
      ))}
      {runs.length === 0 && <p>{t('home.pipeline.panel.noRuns')}</p>}
    </>
  );
  const testList = tests.length === 0
    ? <p>{t('home.pipeline.panel.noTests')}</p>
    : tests.map(job => {
      const at = new Date(job.completed_at ?? job.created_at).getTime();
      return (
        <div key={job.id} className="ft-pl">
          <span title={job.dataset_name}>{job.version_name} · {job.dataset_name}</span>
          <span className="font-mono tabular-nums">
            {typeof job.results?.accuracy === 'number' ? `${f.number(job.results.accuracy * 100, 1)}%` : t(`home.pipeline.state.${job.status === 'completed' ? 'done' : job.status === 'failed' ? 'failed' : job.status === 'stopped' ? 'stopped' : 'running'}`)}
          </span>
          <span>{f.date(at)}</span>
        </div>
      );
    });

  // Erst ab zwei Laeufen gibt es einen Verlauf; vorher bekommen Laeufe und Tests die volle Breite.
  if (runs.length < 2) {
    return (
      <section className="ft-peek is-flat" aria-label={row.name}>
        <div className="min-w-0">
          <div className="ft-peek-h"><span>{t('home.pipeline.panel.runs')}</span>{runs.length > 0 && analysisLink}</div>
          {runList}
        </div>
        <div className="min-w-0">
          <div className="ft-peek-h"><span>{t('home.pipeline.panel.tests')}</span></div>
          {testList}
        </div>
      </section>
    );
  }
  return (
    <section className="ft-peek" aria-label={row.name}>
      <div className="min-w-0">
        <div className="ft-peek-h"><span>{t('home.pipeline.panel.lossPerRun')}</span>{analysisLink}</div>
        <RunChart runs={runs} f={f} />
      </div>
      <div className="min-w-0">
        <div className="ft-peek-h"><span>{t('home.pipeline.panel.runs')}</span></div>
        {runList}
        <div className="ft-peek-h pt-5"><span>{t('home.pipeline.panel.tests')}</span></div>
        {testList}
      </div>
    </section>
  );
}

function PipelineRow({ row, f, isNew, quietAction, section }: { row: ModelRow; f: Format; isNew: boolean; quietAction?: boolean; section: string }) {
  const { t } = useLanguage();
  const [open, setOpen] = useState(false);
  const note = noteText(row, t);
  const noteTarget: AppView | null = row.note?.kind === 'regression' ? 'analysis'
    : row.note?.kind.startsWith('training') ? 'training'
      : row.note?.kind.startsWith('test') ? 'tests'
        : null;
  const Chevron = open ? ChevronDown : ChevronRight;
  // Aufklappen: die Details wachsen in der Hoehe, die Zeilen darunter ruecken mit.
  const peekRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = peekRef.current;
    if (!open || !el || prefersStill()) return;
    el.animate?.(
      [{ height: '0px', opacity: 0 }, { height: `${el.scrollHeight}px`, opacity: 1 }],
      { duration: 260, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' },
    );
  }, [open]);
  return (
    <div className={open ? 'ft-unit is-open' : 'ft-unit'} data-row={row.id} data-section={section}>
      <div className="ft-r ft-row">
        <button className="ft-name" onClick={() => setOpen(v => !v)} aria-expanded={open} title={row.name}>
          <Chevron className="w-3.5 h-3.5 flex-none" aria-hidden="true" />
          <span className="truncate">{row.name}</span>
          {isNew && <span className="ft-new">{t('home.pipeline.new')}</span>}
        </button>
        <DataCell row={row} />
        <TrainingCell row={row} f={f} />
        <TestCell row={row} f={f} quiet={quietAction} />
        <HostingCell row={row} />
        <span className="ft-t" title={row.last ? f.date(row.last.at) : undefined}>{row.last ? f.ago(row.last.at) : '–'}</span>
        {note && (
          <p className="ft-note" title={note}>
            {note}
            {noteTarget && (
              <button className="ft-ref" onClick={() => navigateToModel(noteTarget, { modelId: row.id })}>
                {t(`home.pipeline.open.${noteTarget}`)}
              </button>
            )}
          </p>
        )}
      </div>
      {open && <div ref={peekRef} className="ft-peek-wrap"><RowPanel row={row} f={f} /></div>}
    </div>
  );
}

export function PipelineHead() {
  const { t } = useLanguage();
  return (
    <div className="ft-r ft-heads">
      <span>{t('home.pipeline.cols.model')}</span>
      <span>{t('home.pipeline.cols.data')}</span>
      <span>{t('home.pipeline.cols.training')}</span>
      <span>{t('home.pipeline.cols.test')}</span>
      <span>{t('home.pipeline.cols.hosting')}</span>
      <span className="ft-num">{t('home.pipeline.cols.last')}</span>
    </div>
  );
}

function GroupHead({ label, count, sub }: { label: string; count: number; sub?: string }) {
  return <h2 className="ft-gh">{label}<b>{count}</b>{sub && <small>{sub}</small>}</h2>;
}

function GroupFold({ label, count, onOpen }: { label: string; count: number; onOpen: () => void }) {
  return (
    <button className="ft-gh-fold" aria-expanded="false" onClick={onOpen}>
      <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />
      {label}<b>{count}</b>
    </button>
  );
}

const prefersStill = () =>
  typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Wechselt ein Modell die Gruppe (Training fertig, Test gescheitert), gleitet
 * seine Zeile von der alten an die neue Stelle und leuchtet kurz auf, statt zu
 * springen. Aufklappen verschiebt andere Zeilen auch, bewegt aber nichts.
 */
function useRowMoves() {
  const ref = useRef<HTMLDivElement>(null);
  const before = useRef(new Map<string, { top: number; section: string }>());
  useLayoutEffect(() => {
    const list = ref.current;
    if (!list) return;
    const still = prefersStill();
    const now = new Map<string, { top: number; section: string }>();
    list.querySelectorAll<HTMLElement>('[data-row]').forEach(node => {
      const id = node.dataset.row!;
      const section = node.dataset.section ?? '';
      const top = node.getBoundingClientRect().top;
      now.set(id, { top, section });
      const prev = before.current.get(id);
      if (!prev || prev.section === section || still) return;
      node.classList.remove('is-moved');
      void node.offsetWidth;
      node.classList.add('is-moved');
      const dy = prev.top - top;
      if (Math.abs(dy) > 1) {
        node.animate?.([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: 360, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' });
      }
    });
    before.current = now;
  });
  return ref;
}

export function PipelineList({ rows, f, lastSeen, limit }: {
  rows: ModelRow[];
  f: Format;
  lastSeen: number | null;
  limit: number;
}) {
  const { t } = useLanguage();
  const [showAll, setShowAll] = useState(false);
  // Modelle ohne jeden Lauf wiederholen nur ihre erste Aktion: eine zugeklappte Gruppe reicht.
  const { sections: grouped, unused } = pipelineSections(rows);
  const sections: { key: string; rows: ModelRow[] }[] = [...grouped];
  const attention = grouped[0].rows;
  const rest = sections.slice(1).flatMap(s => s.rows);
  const shownRest = showAll ? rest : rest.slice(0, Math.max(4, limit - attention.length));
  const hiddenStarted = showAll ? 0 : rest.length - shownRest.length;
  const isNew = (r: ModelRow) => lastSeen !== null && (r.last?.at ?? 0) > lastSeen;
  if (showAll && unused.length > 0) sections.push({ key: 'unused', rows: unused });
  const listRef = useRowMoves();

  return (
    <div className="ft-pipe" ref={listRef}>
      {sections.map(s => {
        const shown = s.rows.filter(r => s.key === 'attention' || showAll || shownRest.includes(r));
        if (shown.length === 0) return null;
        return (
          <div key={s.key}>
            <GroupHead label={t(`home.pipeline.group.${s.key}`)} count={s.rows.length} />
            {shown.map(r => <PipelineRow key={r.id} row={r} f={f} isNew={isNew(r)} quietAction={s.key === 'test' && s.rows.length > 1} section={s.key} />)}
          </div>
        );
      })}
      {hiddenStarted > 0 && (
        <GroupFold
          label={t(hiddenStarted === 1 ? 'home.pipeline.more_one' : 'home.pipeline.more', { n: String(hiddenStarted) })}
          count={hiddenStarted}
          onOpen={() => setShowAll(true)}
        />
      )}
      {!showAll && unused.length > 0 && (
        <GroupFold label={t('home.pipeline.group.unused')} count={unused.length} onOpen={() => setShowAll(true)} />
      )}
    </div>
  );
}

export function FirstRunSteps() {
  const { t } = useLanguage();
  const steps = ['data', 'train', 'test', 'host'] as const;
  return (
    <>
      <p className="ft-intro">{t('home.pipeline.first.intro')}</p>
      <div className="ft-steps">
        <span className="ft-name ft-name-plain">{t('home.pipeline.first.model')}</span>
        {steps.map(step => (
          <div key={step} className="ft-step">
            <b>{t(`home.pipeline.first.${step}`)}</b>
            <span>{t(`home.pipeline.first.${step}Hint`)}</span>
          </div>
        ))}
      </div>
    </>
  );
}
