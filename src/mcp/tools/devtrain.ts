// Domaene devtrain: eigenes Python-Trainingsskript gegen Modell + Datensaetze.

import { buildDatasetRefs, refsToEnv } from '../../utils/devDatasetRefs';
import {
  getAttachedDevTrainPanel, listSavedDevScripts, readDevScript, saveDevScriptToLibrary, writeDevScript,
} from '../../utils/devScriptStore';
import { getRun, latestDevTestId, summarizeRun } from '../runLog';
import { call, clip, defineTool, errorText, fail, ok, optNum, optStr, reqStr, S, ToolFailure } from '../util';
import type { McpCallContext, McpTool, ToolArgs } from '../types';
import { announceChange } from './common';
import { findModel, type DatasetInfo, type ModelInfo } from './lookup';

/** Was ein Dev-Train-Skript wissen muss — fuer Beschreibung und devtrain_context. */
export const DEVTRAIN_PROTOCOL = [
  'Dev Train runs your Python script with the FrameTrain Python environment.',
  'Environment variables: MODEL_PATH (model or version folder), DATASET_PATH (selected dataset folder),',
  'DATASET_YAML (only YOLO datasets), DATASET_PATH_<n>/DATASET_YAML_<n> (all datasets of the model in import order, only if there are several), OUTPUT_PATH (save the trained model here, e.g. model.save_pretrained(OUTPUT_PATH)).',
  'Report to FrameTrain by printing one JSON object per line on stdout, flushed:',
  '{"type": "progress", "data": {"epoch": 1, "total_epochs": 3, "step": 10, "total_steps": 100, "train_loss": 0.5, "val_loss": 0.6, "learning_rate": 2e-5}}',
  '{"type": "status", "data": {"message": "..."}}, {"type": "complete", "data": {...final metrics...}}, {"type": "error", "data": {"error": "..."}}.',
  'After a successful run FrameTrain registers OUTPUT_PATH as a new model version.',
].join(' ');

function userOf(ctx: McpCallContext): string {
  if (!ctx.userId) throw new ToolFailure('No user is signed in to FrameTrain.');
  return ctx.userId;
}

async function modelPathFor(model: ModelInfo, versionId?: string): Promise<string> {
  if (versionId) return call<string>('get_version_path_for_ui', { versionId });
  return model.local_path || model.source_path || model.name;
}

/** Datensaetze des Modells in fester Reihenfolge, wie das Panel sie nummeriert. */
async function modelDatasets(modelId: string): Promise<DatasetInfo[]> {
  const list = await call<DatasetInfo[]>('list_datasets_for_model', { modelId });
  return [...list].sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
}

async function resolveRun(args: ToolArgs, ctx: McpCallContext) {
  const userId = userOf(ctx);
  const panelSel = getAttachedDevTrainPanel(userId)?.getSelection();
  const modelId = optStr(args, 'model_id') ?? panelSel?.modelId;
  if (!modelId) throw new ToolFailure('Pass model_id (no model is selected in the Dev Train panel).');
  const model = await findModel(modelId);
  const datasets = await modelDatasets(model.id);
  const wanted = optStr(args, 'dataset_id') ?? (panelSel?.modelId === model.id ? panelSel.datasetIds[0] : undefined);
  const selected = wanted ? datasets.find((d) => d.id === wanted) : datasets[0];
  if (wanted && !selected) throw new ToolFailure(`Dataset ${wanted} does not belong to model ${model.name}.`);
  const versionId = optStr(args, 'version_id') ?? (panelSel?.modelId === model.id ? panelSel.versionId ?? undefined : undefined);
  const modelPath = await modelPathFor(model, versionId);
  const refs = refsToEnv(modelPath, buildDatasetRefs(datasets, selected?.id));
  let script = optStr(args, 'script');
  if (script) writeDevScript(userId, script, `${ctx.client.name}: Skript gesetzt`);
  else script = readDevScript(userId)?.script;
  if (!script || !script.trim()) throw new ToolFailure('No script: pass "script" or set one with devtrain_script_set.');
  return { userId, model, datasets, selected, versionId, refs, script };
}

export const devtrainTools: McpTool[] = [
  defineTool({
    name: 'devtrain_context',
    title: 'Dev-Train-Kontext',
    description: `Describes what a Dev Train script will see for a model (and dataset): the environment variables with concrete paths, a description of the model files, dataset layout and installed Python packages, and the stdout protocol. Read this before writing a script. Protocol: ${DEVTRAIN_PROTOCOL}`,
    properties: {
      model_id: S.str('Model id. Default: model selected in the Dev Train panel.'),
      dataset_id: S.str('Dataset id of that model. Default: selected/first dataset.'),
      version_id: S.str('Start from this trained version instead of the base model.'),
    },
    async run(args, ctx) {
      const userId = userOf(ctx);
      const panelSel = getAttachedDevTrainPanel(userId)?.getSelection();
      const modelId = optStr(args, 'model_id') ?? panelSel?.modelId;
      if (!modelId) return fail('Pass model_id (no model is selected in the Dev Train panel).');
      const model = await findModel(modelId);
      const datasets = await modelDatasets(model.id);
      const selected = optStr(args, 'dataset_id') ? datasets.find((d) => d.id === args.dataset_id) : datasets[0];
      const modelPath = await modelPathFor(model, optStr(args, 'version_id'));
      const env = refsToEnv(modelPath, buildDatasetRefs(datasets, selected?.id));
      let facts: string | null = null;
      try { facts = await call<string>('describe_dev_context', { modelPath, datasetPath: selected?.storage_path ?? null }); } catch (e) { facts = `(not available: ${errorText(e)})`; }
      return ok(clip(`${DEVTRAIN_PROTOCOL}\n\n${facts ?? ''}`, 12000), {
        model: { id: model.id, name: model.name, path: modelPath },
        datasets: datasets.map((d) => ({ id: d.id, name: d.name, path: d.storage_path, selected: d.id === selected?.id })),
        env, protocol: DEVTRAIN_PROTOCOL, facts,
      });
    },
  }),
  defineTool({
    name: 'devtrain_script_get',
    title: 'Skript lesen',
    description: 'Returns the current Dev Train script (from the open editor or the pending draft), or a saved script from the library by script_id.',
    properties: { script_id: S.str('Id from devtrain_scripts_list. Default: the current editor script.') },
    async run(args, ctx) {
      const userId = userOf(ctx);
      const id = optStr(args, 'script_id');
      if (id) {
        const s = listSavedDevScripts(userId).find((x) => x.id === id);
        if (!s) return fail(`Script ${id} not found.`);
        return ok(clip(s.script, 30000), { script_id: s.id, name: s.name, script: s.script, saved_at: s.savedAt });
      }
      const cur = readDevScript(userId);
      if (!cur) return ok('The Dev Train editor is empty.', { script: '', source: 'none' });
      return ok(clip(cur.script, 30000), { script: cur.script, source: cur.source });
    },
  }),
  defineTool({
    name: 'devtrain_script_set',
    title: 'Skript schreiben',
    description: 'Replaces the Dev Train script. The open editor updates immediately; if the panel is closed the script is kept as a draft and loaded when the panel opens. save_name additionally stores it in the script library.',
    properties: {
      script: S.str('Full Python script.'),
      save_name: S.str('Also save to the script library under this name.'),
    },
    required: ['script'],
    approval: 'once',
    async run(args, ctx) {
      const userId = userOf(ctx);
      const script = reqStr(args, 'script');
      const target = writeDevScript(userId, script, `${ctx.client.name}: Skript gesetzt`);
      const name = optStr(args, 'save_name');
      const saved = name ? saveDevScriptToLibrary(userId, name, script) : null;
      announceChange('devtrain');
      return ok(`Script written to the ${target === 'panel' ? 'open editor' : 'draft'} (${script.split('\n').length} lines)${saved ? `, saved as ${saved.name}` : ''}.`, { written_to: target, saved_id: saved?.id ?? null });
    },
  }),
  defineTool({
    name: 'devtrain_scripts_list',
    title: 'Skript-Bibliothek',
    description: 'Lists saved Dev Train scripts (id, name, saved date, length).',
    async run(_args, ctx) {
      const list = listSavedDevScripts(userOf(ctx)).map((s) => ({ id: s.id, name: s.name, saved_at: s.savedAt, lines: s.script.split('\n').length }));
      return ok(`${list.length} saved script(s).`, list);
    },
  }),
  defineTool({
    name: 'devtrain_start',
    title: 'Dev Train starten',
    description: `Runs a Dev Train script (the given one, else the current editor script) for a model and dataset. Long-running; the user confirms each start. Returns the job id immediately; follow with devtrain_status. ${DEVTRAIN_PROTOCOL}`,
    properties: {
      model_id: S.str('Model id. Default: model selected in the panel.'),
      dataset_id: S.str('Dataset of that model for DATASET_PATH. Default: selected/first.'),
      version_id: S.str('Continue from this trained version (MODEL_PATH points to it).'),
      script: S.str('Script to run (also written to the editor).'),
    },
    approval: 'always',
    annotations: { readOnlyHint: false },
    async run(args, ctx) {
      const r = await resolveRun(args, ctx);
      const job = await call<{ id: string }>('start_dev_training', {
        script: r.script, modelId: r.model.id, modelName: r.model.name,
        datasetId: r.selected?.id ?? '', datasetName: r.selected?.name ?? '',
        refs: r.refs, versionId: r.versionId ?? null,
      });
      try { await call('enable_prevent_sleep'); } catch { /* optional */ }
      return ok(`Dev Train started (job ${job.id}). Poll devtrain_status.`, { job_id: job.id, env: r.refs });
    },
  }),
  defineTool({
    name: 'devtrain_status',
    title: 'Dev-Train-Status',
    description: 'Status, progress events and the last stdout/stderr lines of a Dev Train run (default: the current one). Use the log lines to debug a failing script, then fix it with devtrain_script_set and start again.',
    properties: {
      job_id: S.str('Job id from devtrain_start. Default: current training.'),
      log_lines: S.int('Recent lines (0-200). Default 60.', { minimum: 0, maximum: 200 }),
    },
    async run(args) {
      let id = optStr(args, 'job_id');
      if (!id) id = (await call<{ id: string } | null>('get_current_training').catch(() => null))?.id;
      if (!id) return ok('No Dev Train run is active.', { job: null });
      const run = summarizeRun(getRun(id), optNum(args, 'log_lines', 60));
      return ok(`Dev Train ${id}: ${run?.status ?? 'unknown (no events received yet)'}.`, { job_id: id, live: run });
    },
  }),
  defineTool({
    name: 'devtrain_stop',
    title: 'Dev Train stoppen',
    description: 'Stops the running Dev Train script; output written so far stays in devtrain_status.',
    approval: 'always',
    async run() {
      await call('stop_dev_training');
      try { await call('disable_prevent_sleep'); } catch { /* optional */ }
      return ok('Stop requested.', { stopped: true });
    },
  }),
  defineTool({
    name: 'devtrain_test_start',
    title: 'Dev Test starten',
    description: 'Runs a Dev Test script (evaluation/inference script, same environment variables as Dev Train) against a model or version and dataset. Returns immediately; poll devtrain_test_status for output.',
    properties: {
      script: S.str('Python test script.'),
      model_id: S.str('Model id. Default: panel selection.'),
      dataset_id: S.str('Dataset of that model.'),
      version_id: S.str('Test this trained version (MODEL_PATH).'),
    },
    required: ['script'],
    approval: 'always',
    annotations: { readOnlyHint: false },
    async run(args, ctx) {
      const userId = userOf(ctx);
      const panelSel = getAttachedDevTrainPanel(userId)?.getSelection();
      const modelId = optStr(args, 'model_id') ?? panelSel?.modelId;
      if (!modelId) return fail('Pass model_id.');
      const model = await findModel(modelId);
      const datasets = await modelDatasets(model.id);
      const selected = optStr(args, 'dataset_id') ? datasets.find((d) => d.id === args.dataset_id) : datasets[0];
      const refs = refsToEnv(await modelPathFor(model, optStr(args, 'version_id')), buildDatasetRefs(datasets, selected?.id));
      const jobId = await call<string>('start_dev_test', {
        script: reqStr(args, 'script'), modelId: model.id, modelName: model.name,
        datasetId: selected?.id ?? '', datasetName: selected?.name ?? '', refs,
      });
      return ok(`Dev Test started (${jobId}). Poll devtrain_test_status.`, { job_id: jobId });
    },
  }),
  defineTool({
    name: 'devtrain_test_status',
    title: 'Dev-Test-Status',
    description: 'Output lines and completion state of the latest (or given) Dev Test run.',
    properties: { job_id: S.str('Dev Test job id. Default: latest.'), log_lines: S.int('Default 80.', { minimum: 0, maximum: 200 }) },
    async run(args) {
      const id = optStr(args, 'job_id') ?? latestDevTestId();
      if (!id) return ok('No Dev Test output received yet.', { job: null });
      const run = summarizeRun(getRun(id), optNum(args, 'log_lines', 80));
      return ok(`Dev Test ${id}: ${run?.status ?? 'unknown'}.`, { job_id: id, live: run });
    },
  }),
  defineTool({
    name: 'devtrain_test_stop',
    title: 'Dev Test stoppen',
    description: 'Stops the running Dev Test script started with devtrain_test_start.',
    approval: 'once',
    async run() {
      await call('stop_dev_test');
      return ok('Stop requested.', { stopped: true });
    },
  }),
];
