// Domaene datasets: Datensaetze der Modelle.

import { call, clip, defineTool, ok, optNum, optStr, reqNum, reqStr, S, ToolFailure } from '../util';
import type { McpTool } from '../types';
import { announceChange, runAsJob } from './common';
import { datasetBrief, findDataset, findModel, listDatasets, type DatasetInfo } from './lookup';

interface DatasetFile { name: string; path: string; size: number; is_dir?: boolean; split?: string }

async function datasetFiles(datasetId: string): Promise<DatasetFile[]> {
  return call<DatasetFile[]>('get_dataset_files', { datasetId });
}

export const datasetsTools: McpTool[] = [
  defineTool({
    name: 'datasets_list',
    title: 'List datasets',
    description: 'Lists datasets in FrameTrain with id, name, owning model, type, file count and split info. Optionally only the datasets of one model (datasets belong to a model in FrameTrain).',
    properties: { model_id: S.str('Only datasets of this model.') },
    async run(args) {
      const modelId = optStr(args, 'model_id');
      const list = modelId
        ? await call<DatasetInfo[]>('list_datasets_for_model', { modelId })
        : await listDatasets();
      const items = list.map(datasetBrief);
      return ok(`${items.length} dataset(s): ${items.map((d) => `${d.name} (${d.id})`).join(', ') || 'none'}.`, items);
    },
  }),
  defineTool({
    name: 'datasets_info',
    title: 'Dataset details',
    description: 'Returns full details of a dataset: storage path, type, splits, warnings and (for YOLO datasets) the data.yaml contents.',
    properties: { dataset_id: S.str('Dataset id from datasets_list.') },
    required: ['dataset_id'],
    async run(args) {
      const d = await findDataset(reqStr(args, 'dataset_id'));
      let yaml: unknown = null;
      try { yaml = await call('get_dataset_yaml', { datasetId: d.id }); } catch { /* kein YOLO-Datensatz */ }
      return ok(`${d.name}: ${d.file_count} file(s), status ${d.status}.`, { dataset: d, yaml });
    },
  }),
  defineTool({
    name: 'datasets_files',
    title: 'Dataset files',
    description: 'Lists the files of a dataset (name, path, size, split). Paginated with offset/limit because image datasets can have many thousand files.',
    properties: {
      dataset_id: S.str('Dataset id.'),
      split: S.str('Only files of this split, e.g. "train", "val", "test".'),
      offset: S.int('Skip this many files. Default 0.', { minimum: 0 }),
      limit: S.int('Maximum files (1-500). Default 100.', { minimum: 1, maximum: 500 }),
    },
    required: ['dataset_id'],
    async run(args) {
      let files = await datasetFiles(reqStr(args, 'dataset_id'));
      const split = optStr(args, 'split');
      if (split) files = files.filter((f) => f.split === split);
      const offset = optNum(args, 'offset', 0)!;
      const limit = optNum(args, 'limit', 100)!;
      const page = files.slice(offset, offset + limit);
      const splits: Record<string, number> = {};
      files.forEach((f) => { const k = f.split ?? 'none'; splits[k] = (splits[k] ?? 0) + 1; });
      return ok(`${files.length} file(s), showing ${page.length} from ${offset}.`, { total: files.length, offset, files: page, per_split: splits });
    },
  }),
  defineTool({
    name: 'datasets_preview',
    title: 'Preview dataset',
    description: 'Shows the content of one dataset file: first rows of parquet/CSV/JSONL/text files, or a description for media files. Without file_path the first readable file is used. Only files belonging to the dataset can be read.',
    properties: {
      dataset_id: S.str('Dataset id.'),
      file_path: S.str('Path of a file from datasets_files. Optional.'),
      max_rows: S.int('Rows for parquet previews (1-100). Default 20.', { minimum: 1, maximum: 100 }),
    },
    required: ['dataset_id'],
    async run(args) {
      const files = (await datasetFiles(reqStr(args, 'dataset_id'))).filter((f) => !f.is_dir);
      const wanted = optStr(args, 'file_path');
      const file = wanted
        ? files.find((f) => f.path === wanted || f.name === wanted)
        : files.find((f) => /\.(parquet|csv|tsv|jsonl?|txt)$/i.test(f.name)) ?? files[0];
      if (!file) throw new ToolFailure(wanted ? `File "${wanted}" is not part of this dataset.` : 'The dataset has no files.');
      if (/\.parquet$/i.test(file.name)) {
        const preview = await call<unknown>('preview_parquet_file', { filePath: file.path, maxRows: optNum(args, 'max_rows', 20) });
        return ok(`Parquet preview of ${file.name}.`, { file, preview });
      }
      const text = await call<string>('read_dataset_file', { filePath: file.path });
      return ok(clip(text, 6000), { file, content: clip(text, 20000) });
    },
  }),
  defineTool({
    name: 'datasets_search_hf',
    title: 'Search Hugging Face datasets',
    description: 'Searches Hugging Face Hub datasets by query, optionally filtered by task (e.g. "text-classification") and language (e.g. "de").',
    properties: {
      query: S.str('Search text.'),
      limit: S.int('Maximum results (1-50). Default 20.', { minimum: 1, maximum: 50 }),
      task: S.str('Task filter, e.g. "text-classification", "image-classification".'),
      language: S.str('Language code filter, e.g. "en", "de".'),
    },
    required: ['query'],
    annotations: { openWorldHint: true },
    async run(args) {
      const results = await call<unknown[]>('search_huggingface_datasets', {
        query: reqStr(args, 'query'), limit: optNum(args, 'limit', 20),
        filterTask: optStr(args, 'task') ?? null, filterLanguage: optStr(args, 'language') ?? null,
      });
      return ok(`${results.length} result(s).`, results);
    },
  }),
  defineTool({
    name: 'datasets_download_hf',
    title: 'Download dataset',
    description: 'Downloads a Hugging Face dataset and attaches it to a library model. Returns a job_id immediately; poll system_job_status (the result contains the new dataset id).',
    properties: {
      repo_id: S.str('Hugging Face dataset repo id, e.g. "imdb".'),
      model_id: S.str('Model the dataset belongs to.'),
      dataset_name: S.str('Display name. Default: the repo id.'),
    },
    required: ['repo_id', 'model_id'],
    approval: 'once',
    annotations: { openWorldHint: true },
    async run(args) {
      const repoId = reqStr(args, 'repo_id');
      const modelId = (await findModel(reqStr(args, 'model_id'))).id;
      const datasetName = optStr(args, 'dataset_name') ?? repoId;
      return runAsJob('download', `Download dataset ${repoId}`, async () => {
        const d = await call<DatasetInfo>('download_huggingface_dataset', { repoId, datasetName, modelId });
        announceChange('datasets', { datasetId: d.id });
        return d;
      }, { waitMs: 0, describe: (d) => `Downloaded ${d.name} (${d.id}).` });
    },
  }),
  defineTool({
    name: 'datasets_import_local',
    title: 'Import local dataset',
    description: 'Imports a file or folder from the local disk as a dataset of a model. structured=true for already split folders (train/val/test, YOLO data.yaml). The user confirms the source path in FrameTrain.',
    properties: {
      source_path: S.str('Absolute path of the dataset file or folder.'),
      model_id: S.str('Model the dataset belongs to.'),
      dataset_name: S.str('Display name.'),
      structured: S.bool('Keep an existing split structure. Default false.'),
    },
    required: ['source_path', 'model_id', 'dataset_name'],
    approval: 'always',
    async run(args) {
      const cmd = args.structured === true ? 'import_structured_dataset' : 'import_local_dataset';
      const payload = { sourcePath: reqStr(args, 'source_path'), datasetName: reqStr(args, 'dataset_name'), modelId: reqStr(args, 'model_id') };
      return runAsJob('import', 'Dataset import', async () => {
        const d = await call<DatasetInfo>(cmd, payload);
        announceChange('datasets', { datasetId: d.id });
        return d;
      }, { describe: (d) => `Imported ${d.name} (${d.id}), ${d.file_count} file(s).` });
    },
  }),
  defineTool({
    name: 'datasets_split',
    title: 'Split dataset',
    description: 'Splits a dataset into train/val/test by ratio (ratios must sum to 1). Moves files on disk.',
    properties: {
      dataset_id: S.str('Dataset id.'),
      train_ratio: S.num('e.g. 0.8', { minimum: 0, maximum: 1 }),
      val_ratio: S.num('e.g. 0.1', { minimum: 0, maximum: 1 }),
      test_ratio: S.num('e.g. 0.1', { minimum: 0, maximum: 1 }),
    },
    required: ['dataset_id', 'train_ratio', 'val_ratio', 'test_ratio'],
    approval: 'once',
    async run(args) {
      const d = await findDataset(reqStr(args, 'dataset_id'));
      const train = reqNum(args, 'train_ratio'), val = reqNum(args, 'val_ratio'), test = reqNum(args, 'test_ratio');
      if (Math.abs(train + val + test - 1) > 0.001) throw new ToolFailure('train_ratio + val_ratio + test_ratio must equal 1.');
      const res = await call<DatasetInfo>('split_dataset', { datasetId: d.id, modelId: d.model_id, trainRatio: train, valRatio: val, testRatio: test });
      announceChange('datasets', { datasetId: d.id });
      return ok(`Split ${d.name} into ${train}/${val}/${test}.`, { dataset: res });
    },
  }),
  defineTool({
    name: 'datasets_delete',
    title: 'Delete dataset',
    description: 'Permanently deletes a dataset and its files. The user must confirm every call.',
    properties: { dataset_id: S.str('Dataset id.') },
    required: ['dataset_id'],
    approval: 'always',
    annotations: { destructiveHint: true },
    async run(args) {
      const d = await findDataset(reqStr(args, 'dataset_id'));
      await call('delete_dataset', { datasetId: d.id, modelId: d.model_id });
      announceChange('datasets', { datasetId: d.id });
      return ok(`Deleted dataset ${d.name}.`, { dataset_id: d.id, deleted: true });
    },
  }),
];
