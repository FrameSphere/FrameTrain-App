// HomeStellwerk.tsx – Bausteine der Startseite.
//
// Jedes Modell ist ein Streifen aus Daten, Training, Test und Hosting. Der
// Streifen waechst nach rechts, die naechste Aktion steht in der Spalte ihres
// Schritts. Reihenfolge und Zustaende kommen aus homePipeline.ts.

import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, Square, X, AlertTriangle } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { navigateToModel, type AppView } from '../ui/navigationEvents';
import { sparklinePoints } from './homeInsights';
import type { CompareGroup, ModelRow, NextKind, RunPoint, TestMetric } from './homePipeline';

type Format = {
  date: (ms: number | null | undefined) => string;
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

function Spark({ runs }: { runs: RunPoint[] }) {
  const points = sparklinePoints(runs.map(r => ({ id: r.id, label: '', at: r.at, loss: r.loss })), 54, 16, 1);
  return (
    <svg viewBox="0 0 54 16" className="ft-spark" aria-hidden="true">
      <polyline points={points} />
    </svg>
  );
}

function StateWord({ glyph, label }: { glyph: ReactNode; label: string }) {
  return (
    <span className="ft-st" title={label}>
      <span className="ft-st-g" aria-hidden="true">{glyph}</span>
      <span className="ft-st-word">{label}</span>
    </span>
  );
}

function FailedWord() {
  const { t } = useLanguage();
  return <StateWord glyph={<X className="w-3.5 h-3.5" />} label={t('home.pipeline.state.failed')} />;
}

function StoppedWord() {
  const { t } = useLanguage();
  return <StateWord glyph={<Square className="w-2.5 h-2.5" fill="currentColor" strokeWidth={0} />} label={t('home.pipeline.state.stopped')} />;
}

function metricText(metric: TestMetric, f: Format): string {
  return metric.fraction !== null ? `${f.number(metric.fraction * 100, 1)}%` : f.number(metric.value, 3);
}

function DataCell({ row }: { row: ModelRow }) {
  if (row.next === 'addDataset') return <div className="ft-blk ft-front"><ActionButton row={row} kind="addDataset" /></div>;
  return <div className="ft-blk ft-done ft-first" title={row.data.name ?? undefined}><span className="ft-v">{row.data.name ?? '–'}</span></div>;
}

function TrainingCell({ row, f }: { row: ModelRow; f: Format }) {
  const { t } = useLanguage();
  const { state, loss, runs } = row.training;
  if (state === 'running') return <div className="ft-blk ft-live"><span className="ft-v">{t('home.pipeline.state.running')}</span></div>;
  if (state === 'failed' || state === 'stopped') {
    return (
      <div className={`ft-blk ${state === 'failed' ? 'ft-bad' : 'ft-warn'}`}>
        {state === 'failed' ? <FailedWord /> : <StoppedWord />}
        <ActionButton row={row} kind="retryTrain" />
      </div>
    );
  }
  if (loss !== null) {
    const overfit = row.note?.kind === 'overfitting';
    return (
      <div className={`ft-blk ${overfit ? 'ft-warn' : 'ft-done'}`}>
        {overfit && <span aria-hidden="true">▲</span>}
        <span className="ft-v font-mono tabular-nums">{f.number(loss, 4)}</span>
        {runs.length > 1 && <Spark runs={runs} />}
      </div>
    );
  }
  if (row.next === 'addDataset') return <div className="ft-blk ft-rest" />;
  return <div className="ft-blk ft-skip"><span className="ft-v">{t('home.pipeline.original')}</span></div>;
}

function TestCell({ row, f }: { row: ModelRow; f: Format }) {
  const { t } = useLanguage();
  const { state, metric } = row.test;
  if (state === 'running') return <div className="ft-blk ft-live"><span className="ft-v">{t('home.pipeline.state.running')}</span></div>;
  if (row.next === 'retryTest') {
    return (
      <div className={`ft-blk ${state === 'failed' ? 'ft-bad' : 'ft-warn'}`}>
        {state === 'failed' ? <FailedWord /> : <StoppedWord />}
        <ActionButton row={row} kind="retryTest" />
      </div>
    );
  }
  if (metric) {
    const weak = row.note?.kind === 'weakResult';
    return (
      <div className={`ft-blk ${weak ? 'ft-warn' : 'ft-done'}`}>
        <span className="ft-v font-mono tabular-nums" title={metric.fraction === null ? metric.label : undefined}>{metricText(metric, f)}</span>
        {row.next === 'startTest' && <ActionButton row={row} kind="startTest" />}
      </div>
    );
  }
  if (row.next === 'startTest') return <div className="ft-blk ft-front"><ActionButton row={row} kind="startTest" /></div>;
  if (state === 'done') return <div className="ft-blk ft-done"><span className="ft-v">{t('home.pipeline.state.done')}</span></div>;
  return <div className="ft-blk ft-rest" />;
}

function HostingCell({ row }: { row: ModelRow }) {
  const { t } = useLanguage();
  const state = row.hosting.state;
  if (state === 'failed') return <div className="ft-blk ft-bad ft-last"><FailedWord /></div>;
  if (state === 'loading') return <div className="ft-blk ft-live ft-last"><span className="ft-v">{t('home.pipeline.hosting.loading')}</span></div>;
  if (state === 'ready') return <div className="ft-blk ft-done ft-last"><span className="ft-v">{t('home.pipeline.hosting.ready')}</span></div>;
  if (state === 'idle') return <div className="ft-blk ft-done ft-last"><span className="ft-v">{t('home.pipeline.hosting.idle')}</span></div>;
  if (row.next === 'host') return <div className="ft-blk ft-front"><ActionButton row={row} kind="host" /></div>;
  return <div className="ft-blk ft-rest" />;
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
  return (
    <section className="ft-peek" aria-label={row.name}>
      <div className="min-w-0">
        <div className="ft-peek-h">
          <b>{t('home.pipeline.panel.lossPerRun')}</b>
          <button className="ft-link" onClick={() => navigateToModel('analysis', { modelId: row.id })}>{t('home.pipeline.panel.openAnalysis')}</button>
        </div>
        {runs.length > 1
          ? <RunChart runs={runs} f={f} />
          : <p>{t(runs.length === 1 ? 'home.pipeline.panel.oneRun' : 'home.pipeline.panel.noRuns')}</p>}
      </div>
      <div className="min-w-0">
        <div className="ft-peek-h"><b>{t('home.pipeline.panel.runs')}</b></div>
        {[...runs].reverse().slice(0, 4).map(r => (
          <div key={r.id} className="ft-pl">
            <span className="font-mono">{f.date(r.at)}</span>
            <span className="font-mono tabular-nums">{f.number(r.loss, 4)}</span>
            <span className="font-mono tabular-nums">{f.duration(r.durationSec)}</span>
          </div>
        ))}
        <div className="ft-peek-h pt-3.5"><b>{t('home.pipeline.panel.tests')}</b></div>
        {tests.length === 0
          ? <p>{t('home.pipeline.panel.noTests')}</p>
          : tests.map(job => (
            <div key={job.id} className="ft-pl">
              <span className="truncate" title={job.dataset_name}>{job.version_name} · {job.dataset_name}</span>
              <span className="font-mono tabular-nums">
                {typeof job.results?.accuracy === 'number' ? `${f.number(job.results.accuracy * 100, 1)}%` : t(`home.pipeline.state.${job.status === 'completed' ? 'done' : job.status === 'failed' ? 'failed' : job.status === 'stopped' ? 'stopped' : 'running'}`)}
              </span>
              <span className="font-mono">{f.date(new Date(job.completed_at ?? job.created_at).getTime())}</span>
            </div>
          ))}
      </div>
    </section>
  );
}

function PipelineRow({ row, f, isNew }: { row: ModelRow; f: Format; isNew: boolean }) {
  const { t } = useLanguage();
  const [open, setOpen] = useState(false);
  const note = noteText(row, t);
  const noteTarget: AppView | null = row.note?.kind.startsWith('training') ? 'training' : row.note?.kind.startsWith('test') ? 'tests' : null;
  const Chevron = open ? ChevronDown : ChevronRight;
  return (
    <>
      <div className="ft-r ft-row">
        <button className="ft-name" onClick={() => setOpen(v => !v)} aria-expanded={open} title={row.name}>
          <Chevron className="w-3.5 h-3.5 flex-none" aria-hidden="true" />
          <span className="truncate">{row.name}</span>
          {isNew && <span className="ft-new">{t('home.pipeline.new')}</span>}
        </button>
        <DataCell row={row} />
        <TrainingCell row={row} f={f} />
        <TestCell row={row} f={f} />
        <HostingCell row={row} />
        <span className="ft-t ft-dur font-mono tabular-nums">{f.duration(row.last?.durationSec)}</span>
        <span className="ft-t font-mono tabular-nums">{f.date(row.last?.at)}</span>
      </div>
      {note && (
        <div className="ft-r ft-note">
          <p title={note}>
            {note}
            {noteTarget && <button className="ft-link ml-3" onClick={() => navigateToModel(noteTarget, { modelId: row.id })}>{t('home.pipeline.openDetails')}</button>}
          </p>
        </div>
      )}
      {open && <RowPanel row={row} f={f} />}
    </>
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
      <span className="ft-num ft-dur">{t('home.pipeline.cols.duration')}</span>
      <span className="ft-num">{t('home.pipeline.cols.last')}</span>
    </div>
  );
}

export function PipelineList({ rows, f, lastSeen, limit, attentionHead }: {
  rows: ModelRow[];
  f: Format;
  lastSeen: number | null;
  limit: number;
  /** Laeuft ein Training, steht "braucht dich" nicht mehr in der Ueberschrift */
  attentionHead: string | null;
}) {
  const { t } = useLanguage();
  const [showAll, setShowAll] = useState(false);
  const attention = rows.filter(r => r.group === 'attention');
  // Nie benutzte Modelle wiederholen nur "Dataset hinzufuegen": eine Zeile reicht.
  const allUnused = rows.filter(r => r.group === 'next' && r.next === 'addDataset' && !r.last);
  const onlyUnused = allUnused.length === rows.length;
  const unused = onlyUnused ? [] : allUnused;
  const rest = rows.filter(r => r.group !== 'attention' && !unused.includes(r));
  const shownRest = showAll ? rest : rest.slice(0, Math.max(4, limit - attention.length));
  const hiddenCount = showAll ? 0 : rest.length - shownRest.length + unused.length;
  const next = [...shownRest.filter(r => r.group === 'next'), ...(showAll ? unused : [])];
  const hosted = shownRest.filter(r => r.group === 'hosted');
  const isNew = (r: ModelRow) => lastSeen !== null && (r.last?.at ?? 0) > lastSeen;

  return (
    <>
      {attentionHead && attention.length > 0 && <span className="ft-gh">{attentionHead}</span>}
      {attention.map(r => <PipelineRow key={r.id} row={r} f={f} isNew={isNew(r)} />)}
      {next.length > 0 && <span className="ft-gh">{t('home.pipeline.group.next')}</span>}
      {next.map(r => <PipelineRow key={r.id} row={r} f={f} isNew={isNew(r)} />)}
      {hosted.length > 0 && <span className="ft-gh">{t('home.pipeline.group.hosted')}</span>}
      {hosted.map(r => <PipelineRow key={r.id} row={r} f={f} isNew={isNew(r)} />)}
      {hiddenCount > 0 && (
        <div className="ft-more">
          <span>{t(hiddenCount === 1 ? 'home.pipeline.more_one' : 'home.pipeline.more', { n: String(hiddenCount) })}</span>
          <button className="ft-go ft-go-quiet" onClick={() => setShowAll(true)}>{t('home.pipeline.showAll')}</button>
        </div>
      )}
    </>
  );
}

// ── Benchmarks ───────────────────────────────────────────────────────────

function CompareAction({ row }: { row: ModelRow }) {
  const { t } = useLanguage();
  const failed = row.test.state === 'failed';
  return (
    <div className="ft-open">
      {(failed || row.test.state === 'stopped') && (
        <span className={`ft-blk ${failed ? 'ft-bad' : 'ft-warn'}`}>{failed ? <FailedWord /> : <StoppedWord />}</span>
      )}
      {row.next && <ActionButton row={row} kind={row.next} />}
      {!row.next && row.test.state === 'running' && <span className="ft-blk ft-live">{t('home.pipeline.state.running')}</span>}
      <span className="ft-dash" />
    </div>
  );
}

function CompareRowView({ row, best, sameData, f }: { row: ModelRow; best: boolean; sameData: boolean; f: Format }) {
  const { t } = useLanguage();
  const metric = row.test.metric;
  return (
    <>
      <div className="ft-cmp ft-crow">
        <span className="ft-name ft-name-plain" title={row.name}><span className="truncate">{row.name}</span></span>
        {metric ? (
          <div className={`ft-acc ${best ? 'is-best' : ''}`}>
            {metric.fraction !== null && <i style={{ width: `${Math.min(100, metric.fraction * 100)}%` }} aria-hidden="true" />}
            <span className="font-mono tabular-nums">{metricText(metric, f)}</span>
            {metric.fraction === null && <span className="ft-acc-label">{metric.label}</span>}
            {best && <em>{t('home.pipeline.compare.best')}</em>}
          </div>
        ) : <CompareAction row={row} />}
        <span className="ft-ds" title={row.test.dataset ?? undefined}>{row.test.dataset ?? '–'}</span>
        <span className="ft-t font-mono tabular-nums">{f.date(row.test.at)}</span>
      </div>
      {sameData && (
        <div className="ft-cmp ft-note">
          <p><AlertTriangle className="w-3 h-3 inline mr-1.5 -mt-0.5" aria-hidden="true" />{t('home.pipeline.compare.sameData')}</p>
        </div>
      )}
    </>
  );
}

export function CompareView({ groups, withoutResult, groupLabel, f }: {
  groups: CompareGroup[];
  withoutResult: ModelRow[];
  groupLabel: (key: string) => string;
  f: Format;
}) {
  const { t } = useLanguage();
  const [showAll, setShowAll] = useState(false);
  return (
    <>
      <div className="ft-cmp ft-heads">
        <span>{t('home.pipeline.cols.model')}</span>
        <span>{t('home.pipeline.compare.scale')}</span>
        <span>{t('home.pipeline.compare.dataset')}</span>
        <span className="ft-num">{t('home.pipeline.compare.lastTest')}</span>
      </div>
      {groups.length === 0 && <p className="ft-intro mt-4">{t('home.pipeline.compare.empty')}</p>}
      {groups.map(g => (
        <div key={g.key}>
          <span className="ft-gh">
            {groupLabel(g.key)}
            {g.rows.filter(r => r.row.test.metric).length > 1 && (
              <small>{t(g.sameDataset ? 'home.pipeline.compare.sameDataset' : 'home.pipeline.compare.mixedDatasets')}</small>
            )}
          </span>
          {g.rows.map(({ row, best, sameData }) => <CompareRowView key={row.id} row={row} best={best} sameData={sameData} f={f} />)}
        </div>
      ))}
      {withoutResult.length > 0 && !showAll && (
        <div className="ft-more">
          <span>{t(withoutResult.length === 1 ? 'home.pipeline.compare.more_one' : 'home.pipeline.compare.more', { n: String(withoutResult.length) })}</span>
          <button className="ft-go ft-go-quiet" onClick={() => setShowAll(true)}>{t('home.pipeline.showAll')}</button>
        </div>
      )}
      {showAll && withoutResult.length > 0 && <span className="ft-gh">{t('home.pipeline.compare.noResult')}</span>}
      {showAll && withoutResult.map(row => <CompareRowView key={row.id} row={row} best={false} sameData={false} f={f} />)}
    </>
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
