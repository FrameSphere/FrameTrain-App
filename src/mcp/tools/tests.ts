// Domaene tests: Auswertung trainierter Versionen.

import { call, defineTool, fail, ok, optNum, optObj, optStr, reqStr, S } from '../util';
import type { McpTool } from '../types';
import { findDataset, findVersion } from './lookup';

interface TestJob { id: string; status?: unknown; progress?: unknown; results?: unknown; error?: string | null; version_id?: string }

interface Prediction { is_correct: boolean; [k: string]: unknown }
interface TestResults { total_samples: number; accuracy?: number | null; predictions?: Prediction[]; [k: string]: unknown }

function resultsBrief(r: TestResults) {
  const { predictions, ...rest } = r;
  return { ...rest, prediction_count: predictions?.length ?? 0, wrong_examples: (predictions ?? []).filter((p) => !p.is_correct).slice(0, 10) };
}

export const testsTools: McpTool[] = [
  defineTool({
    name: 'tests_start',
    title: 'Start test',
    description: 'Evaluates a trained model version on a dataset (accuracy, loss, per-sample predictions). Runs in the background; returns the test job immediately — poll tests_status, then read tests_results.',
    properties: {
      version_id: S.str('Trained version id.'),
      dataset_id: S.str('Dataset id (usually its test split is used).'),
      batch_size: S.int('Default: plugin default.', { minimum: 1 }),
      max_samples: S.int('Only evaluate this many samples.', { minimum: 1 }),
      task_type: S.str('Override the plugin task type.'),
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
      return ok(`Test started (job ${job.id}). Poll tests_status.`, { job });
    },
  }),
  defineTool({
    name: 'tests_status',
    title: 'Test status',
    description: 'Returns the running or most recent test job with progress and, when finished, a summary of the results.',
    async run() {
      const active = await call<TestJob | null>('get_active_test_job').catch(() => null);
      const job = active ?? await call<TestJob | null>('get_current_test').catch(() => null);
      if (!job) return ok('No test is running.', { job: null });
      return ok(`Test ${job.id}: ${String(job.status ?? 'unknown')}.`, { job, running: !!active });
    },
  }),
  defineTool({
    name: 'tests_stop',
    title: 'Stop test',
    description: 'Stops the running evaluation job started with tests_start.',
    approval: 'once',
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
    description: 'Lists past test jobs (newest first) with model, version, dataset and status.',
    properties: { limit: S.int('Default 20.', { minimum: 1, maximum: 100 }) },
    async run(args) {
      const history = await call<TestJob[]>('get_test_history');
      const items = history.slice(0, optNum(args, 'limit', 20)).map((j) => ({ ...j, results: undefined }));
      return ok(`${history.length} test job(s).`, items);
    },
  }),
  defineTool({
    name: 'tests_single_input',
    title: 'Single test',
    description: 'Runs a trained version on one input without a dataset. input_type: "text" or "file" (absolute path to an image/audio file).',
    properties: {
      version_id: S.str('Trained version id.'),
      input: S.str('Text or file path.'),
      input_type: S.enumOf('Kind of input.', ['text', 'file']),
      task_type: S.str('Override the plugin task type.'),
    },
    required: ['version_id', 'input', 'input_type'],
    async run(args) {
      const out = await call<string>('test_single_input', {
        versionId: reqStr(args, 'version_id'), singleInput: reqStr(args, 'input'), singleInputType: reqStr(args, 'input_type'),
        taskType: optStr(args, 'task_type') ?? null, pluginConfig: null,
      });
      let parsed: unknown = out;
      try { parsed = JSON.parse(out); } catch { /* Klartext */ }
      return ok(typeof parsed === 'string' ? parsed : 'Prediction done.', { result: parsed });
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
