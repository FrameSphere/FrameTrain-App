// Domaene models: Modellbibliothek (lokal + Hugging Face).

import { call, defineTool, ok, optNum, reqStr, S } from '../util';
import type { McpTool } from '../types';
import { runAsJob } from './common';
import { findModel, listModels, listVersions, modelBrief, type ModelInfo } from './lookup';

export const modelsTools: McpTool[] = [
  defineTool({
    name: 'models_list',
    title: 'List models',
    description: 'Lists all models in the local FrameTrain library (imported or downloaded base models and Synapse canvas models) with id, name, type, plugin and size. Model ids are needed by most training, dataset and test tools.',
    async run() {
      const models = await listModels();
      const items = models.map(modelBrief);
      return ok(`${items.length} model(s): ${items.map((m) => `${m.name} (${m.id})`).join(', ') || 'none'}.`, items);
    },
  }),
  defineTool({
    name: 'models_info',
    title: 'Model details',
    description: 'Returns details of one library model (paths, type, size) plus its trained versions (version ids, names, training metrics). Version ids are used by lab, tests, hosting and analysis tools.',
    properties: { model_id: S.str('Model id from models_list.') },
    required: ['model_id'],
    async run(args) {
      const modelId = reqStr(args, 'model_id');
      const info = await call<ModelInfo>('get_model_info', { modelId });
      let versions: unknown[] = [];
      try { versions = await listVersions(modelId); } catch { /* Modell ohne Versionen */ }
      return ok(`${info.name}: ${versions.length} version(s).`, { model: info, versions });
    },
  }),
  defineTool({
    name: 'models_search_hf',
    title: 'Search Hugging Face',
    description: 'Searches the Hugging Face Hub for models by free-text query. Returns repo ids, downloads, likes and pipeline tags. Use models_download_hf with a repo_id to add one to the library.',
    properties: {
      query: S.str('Search text, e.g. "distilbert sentiment" or "yolov8".'),
      limit: S.int('Maximum results (1-50). Default 20.', { minimum: 1, maximum: 50 }),
    },
    required: ['query'],
    annotations: { openWorldHint: true },
    async run(args) {
      const results = await call<{ id?: string; modelId?: string }[]>('search_huggingface_models', {
        query: reqStr(args, 'query'), limit: optNum(args, 'limit', 20),
      });
      return ok(`${results.length} result(s).`, results);
    },
  }),
  defineTool({
    name: 'models_hf_files',
    title: 'Hugging Face files',
    description: 'Lists the files of a Hugging Face model repo with sizes, to judge download size and format before a download.',
    properties: { repo_id: S.str('Repo id, e.g. "distilbert-base-uncased".') },
    required: ['repo_id'],
    annotations: { openWorldHint: true },
    async run(args) {
      const files = await call<{ size?: number }[]>('get_huggingface_model_files', { repoId: reqStr(args, 'repo_id') });
      const total = files.reduce((s, f) => s + (f.size ?? 0), 0);
      return ok(`${files.length} file(s), ${(total / 1e9).toFixed(2)} GB total.`, { files, total_bytes: total });
    },
  }),
  defineTool({
    name: 'models_download_hf',
    title: 'Download model',
    description: 'Downloads a model from Hugging Face into the local library. Runs in the background and returns a job_id immediately; system_job_status reports progress and the new model id.',
    properties: {
      repo_id: S.str('Hugging Face repo id.'),
      model_name: S.str('Display name in the library. Default: the repo id.'),
    },
    required: ['repo_id'],
    approval: 'once',
    annotations: { openWorldHint: true },
    async run(args) {
      const repoId = reqStr(args, 'repo_id');
      const modelName = typeof args.model_name === 'string' && args.model_name.trim() ? args.model_name : repoId;
      return runAsJob('download', `Download ${repoId}`, () => call<ModelInfo>('download_huggingface_model', { repoId, modelName }), {
        waitMs: 0,
        describe: (m) => `Downloaded ${m.name} (${m.id}).`,
      });
    },
  }),
  defineTool({
    name: 'models_import_local',
    title: 'Import local model',
    description: 'Imports a model folder from the local disk into the library (copied). The user confirms the source path in FrameTrain.',
    properties: {
      source_path: S.str('Absolute path of the model folder on this computer.'),
      model_name: S.str('Display name in the library.'),
    },
    required: ['source_path', 'model_name'],
    approval: 'always',
    async run(args) {
      return runAsJob('import', 'Model import', () => call<ModelInfo>('import_local_model', {
        sourcePath: reqStr(args, 'source_path'), modelName: reqStr(args, 'model_name'),
      }), { describe: (m) => `Imported ${m.name} (${m.id}).` });
    },
  }),
  defineTool({
    name: 'models_rename',
    title: 'Rename model',
    description: 'Renames a library model (display name only; id, files and versions stay unchanged).',
    properties: { model_id: S.str('Model id.'), new_name: S.str('New display name.') },
    required: ['model_id', 'new_name'],
    approval: 'once',
    annotations: { idempotentHint: true },
    async run(args) {
      await call('rename_model', { modelId: reqStr(args, 'model_id'), newName: reqStr(args, 'new_name') });
      return ok('Model renamed.', { model_id: args.model_id, name: args.new_name });
    },
  }),
  defineTool({
    name: 'models_delete',
    title: 'Delete model',
    description: 'Permanently deletes a library model including its files. The user must confirm every call.',
    properties: { model_id: S.str('Model id.') },
    required: ['model_id'],
    approval: 'always',
    annotations: { destructiveHint: true },
    async run(args) {
      const m = await findModel(reqStr(args, 'model_id'));
      await call('delete_model', { modelId: m.id });
      return ok(`Deleted model ${m.name}.`, { model_id: m.id, deleted: true });
    },
  }),
  defineTool({
    name: 'models_version_delete',
    title: 'Delete version',
    description: 'Permanently deletes one trained version of a model. The user must confirm every call.',
    properties: { version_id: S.str('Version id from models_info.') },
    required: ['version_id'],
    approval: 'always',
    annotations: { destructiveHint: true },
    async run(args) {
      const versionId = reqStr(args, 'version_id');
      await call('delete_model_version', { versionId });
      return ok('Version deleted.', { version_id: versionId, deleted: true });
    },
  }),
];
