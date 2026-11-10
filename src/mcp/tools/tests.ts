// Domaene tests: Auswertung trainierter Versionen.

import { call, defineTool, fail, ok, optNum, optObj, optStr, reqStr, S } from '../util';
import type { McpTool } from '../types';
import { findDataset, findVersion } from './lookup';

interface TestJob {
  id: string; status?: unknown; progress?: unknown; results?: TestResults | null; error?: string | null; version_id?: string;
  created_at?: string; completed_at?: string | null;
}

interface Prediction { is_correct: boolean; [k: string]: unknown }
interface TestResults { total_samples: number; accuracy?: number | null; predictions?: Prediction[]; [k: string]: unknown }

/** Kennzahlen eines fertigen Laufs ohne die einzelnen Vorhersagen. */
function resultSummary(r: TestResults | null | undefined) {
  if (!r) return null;
  const sec = typeof r.average_inference_time === 'number' ? r.average_inference_time : null;
  return {
    total_samples: r.total_samples,
    accuracy: r.accuracy ?? null,
    average_loss: (r.average_loss as number | null | undefined) ?? null,
    ms_per_sample: sec !== null && sec > 0 ? Math.round(sec * 10000) / 10 : null,
    metrics: r.metrics ?? null,
  };
}

const endedAt = (j: TestJob) => Date.parse(j.completed_at ?? j.created_at ?? '') || 0;

/** Verlauf neueste zuerst. Die App fuehrt ihn in Startreihenfolge (aelteste zuerst). */
export function newestFirst(jobs: TestJob[]): TestJob[] {
  return [...jobs].sort((a, b) => endedAt(b) - endedAt(a));
}

function resultsBrief(r: TestResults) {
  const { predictions, ...rest } = r;
  return { ...rest, prediction_count: predictions?.length ?? 0, wrong_examples: (predictions ?? []).filter((p) => !p.is_correct).slice(0, 10) };
}

export const testsTools: McpTool[] = [
  defineTool({
    name: 'tests_start',
    title: 'Start test',
    description: 'Evaluates a trained model version on a dataset (accuracy, loss, per-sample predictions). Runs in the background; returns the test job immediately. tests_status reports progress, tests_results the outcome.',
    properties: {
      version_id: S.str('Trained version id.'),
      dataset_id: S.str('Dataset id (usually its test split is used).'),
      batch_size: S.int('Default: plugin default.', { minimum: 1 }),
      max_samples: S.int('Only evaluate this many samples.', { minimum: 1 }),
      task_type: S.str('Override the task type (e.g. "seq_classification", "causal_lm", "detect"). Default: derived from the trained version.'),
      plugin_config: S.obj('Plugin-specific test parameters.'),
    },
    required: ['version_id', 'dataset_id'],
    approval: 'always',
    annotations: { readOnlyHint: false },
    async run(args) {
      const { version, modelName } = await findVersion(reqStr(args, 'version_id'));
      const ds = await findDataset(reqStr(args, 'dataset_id'));
      const job = await call<TestJob>('start_test', {
        modelId: version.model_id, modelName, versionId: version.id, versionName: version.version_name,
        datasetId: ds.id, datasetName: ds.name, batchSize: optNum(args, 'batch_size') ?? null,
        maxSamples: optNum(args, 'max_samples') ?? null, taskType: optStr(args, 'task_type') ?? null,
        pluginConfig: optObj(args, 'plugin_config') ?? null,
      });
      return ok(`Test started (job ${job.id}). Progress: tests_status.`, { job });
    },
  }),
  defineTool({
    name: 'tests_status',
    title: 'Test status',
    description: 'Returns the running or most recent test job with progress and, when finished, a summary of the results.',
    async run() {
      const active = await call<TestJob | null>('get_active_test_job').catch(() => null);
      if (active) return ok(`Test ${active.id}: ${String(active.status ?? 'unknown')}.`, { job: active, running: true });
      // Nach dem Ende ist kein Job mehr "aktuell": dann der zuletzt beendete aus dem Verlauf,
      // damit der Assistent sein Ergebnis sieht statt { job: null }.
      const current = await call<TestJob | null>('get_current_test').catch(() => null);
      const last = current ?? newestFirst(await call<TestJob[]>('get_test_history').catch(() => []))[0];
      if (!last) return ok('No test has run yet.', { job: null, running: false });
      const { results, ...job } = last;
      return ok(`No test is running. Latest test ${last.id}: ${String(last.status ?? 'unknown')}.`, {
        job, running: false, result: resultSummary(results),
      });
    },
  }),
  defineTool({
    name: 'tests_stop',
    title: 'Stop test',
    description: 'Stops the running evaluation job started with tests_start.',
    approval: 'once',
    annotations: { destructiveHint: true },
    async run() {
      await call('stop_test');
      return ok('Stop requested.', { stopped: true });
    },
  }),
  defineTool({
    name: 'tests_results',
    title: 'Test results',
    description: 'Returns the stored test results of a version (newest first): totals, accuracy, metrics and up to 10 wrong predictions per run.',
    properties: { version_id: S.str('Trained version id.'), limit: S.int('Runs to return. Default 3.', { minimum: 1, maximum: 20 }) },
    required: ['version_id'],
    async run(args) {
      const results = await call<TestResults[]>('get_test_results_for_version', { versionId: reqStr(args, 'version_id') });
      const items = results.slice(0, optNum(args, 'limit', 3)).map(resultsBrief);
      return ok(`${results.length} stored test run(s).`, items);
    },
  }),
  defineTool({
    name: 'tests_history',
    title: 'Test history',
    description: 'Lists past test jobs (newest first) with model, version, dataset, status and the result summary (accuracy, samples, ms per sample).',
    properties: { limit: S.int('Default 20.', { minimum: 1, maximum: 100 }) },
    async run(args) {
      const history = newestFirst(await call<TestJob[]>('get_test_history'));
      const items = history.slice(0, optNum(args, 'limit', 20)).map(({ results, ...j }) => ({ ...j, result: resultSummary(results) }));
      return ok(`${history.length} test job(s), newest first.`, items);
    },
  }),
  defineTool({
    name: 'tests_single_input',
    title: 'Single test',
    description: 'Runs a trained version on one input without a dataset and returns the prediction (label, confidence, top predictions or generated text). input_type: "text" or "file" (absolute path to an image/audio file). Loads the model each time; for many inputs use lab_server_start + lab_infer.',
    properties: {
      version_id: S.str('Trained version id.'),
      input: S.str('Text or file path.'),
      input_type: S.enumOf('Kind of input.', ['text', 'file']),
      task_type: S.str('Override the task type. Default: derived from the trained version.'),
    },
    required: ['version_id', 'input', 'input_type'],
    async run(args) {
      // wait: true – vorher kam nur die Test-ID zurueck und das Ergebnis ging als UI-Event verloren.
      const out = await call<string>('test_single_input', {
        versionId: reqStr(args, 'version_id'), singleInput: reqStr(args, 'input'), singleInputType: reqStr(args, 'input_type'),
        taskType: optStr(args, 'task_type') ?? null, pluginConfig: null, wait: true,
      });
      let parsed: unknown = out;
      try { parsed = JSON.parse(out); } catch { /* Klartext */ }
      const p = parsed as { predicted_output?: unknown; confidence?: unknown } | string;
      const summary = typeof p === 'string' ? p
        : `Prediction: ${String(p.predicted_output ?? '?')}${typeof p.confidence === 'number' ? ` (${Math.round(p.confidence * 1000) / 10} %)` : ''}.`;
      return ok(summary, { result: parsed });
    },
  }),
  defineTool({
    name: 'tests_hard_examples_export',
    title: 'Export hard examples',
    description: 'Exports the wrong predictions of the latest stored test run of a version as a file (jsonl or csv) for retraining. Returns the file path.',
    properties: { version_id: S.str('Trained version id.'), format: S.enumOf('Default jsonl.', ['jsonl', 'csv']) },
    required: ['version_id'],
    approval: 'once',
    async run(args) {
      const results = await call<TestResults[]>('get_test_results_for_version', { versionId: reqStr(args, 'version_id') });
      const wrong = (results[0]?.predictions ?? []).filter((p) => !p.is_correct);
      if (!wrong.length) return fail('The latest test run has no wrong predictions (or no stored predictions).');
      const path = await call<string>('export_hard_examples', { predictions: wrong, format: optStr(args, 'format') ?? 'jsonl' });
      return ok(`Exported ${wrong.length} hard example(s) to ${path}.`, { path, count: wrong.length });
    },
  }),
];
