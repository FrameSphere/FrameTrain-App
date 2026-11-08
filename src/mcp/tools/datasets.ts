// Domaene datasets: Datensaetze der Modelle.

import { call, clip, defineTool, ok, optNum, optStr, reqNum, reqStr, S, strArr, ToolArgError, ToolFailure, withImages } from '../util';
import type { McpTool } from '../types';
import { announceChange, runAsJob } from './common';
import { datasetBrief, findDataset, findModel, listDatasets, type DatasetInfo, type ModelInfo } from './lookup';

interface DatasetFile { name: string; path: string; size: number; is_dir?: boolean; split?: string }

async function datasetFiles(datasetId: string): Promise<DatasetFile[]> {
  return call<DatasetFile[]>('get_dataset_files', { datasetId });
}


interface TableSummary {
  rows: number; columns: string[]; task: string; detected: Record<string, unknown>;
  labels: Record<string, number>; label_count: number; examples: Record<string, unknown>[];
}
interface FetchReport {
  format: string; images: number; yolo_labels: number; classes: string[];
  splits: Record<string, number>; problems: string[];
  audio?: number; video?: number; transcripts?: number; table?: TableSummary | null;
}
interface FetchResult { dataset: DatasetInfo; report: FetchReport; conversion?: { images: number; boxes: number; classes: string[] } | null }

/** Objekterkennung (YOLO, DETR, ...): braucht Bilder plus Boxen. */
export function isDetectionModel(m: Pick<ModelInfo, 'model_type' | 'plugin_override' | 'name'>): boolean {
  const s = `${m.model_type ?? ''} ${m.plugin_override ?? ''} ${m.name}`.toLowerCase();
  return /yolo|detr|rtdetr|detection|ssd|retinanet|faster.?rcnn/.test(s);
}

/** Bild-Modelle allgemein: ohne Bilder ist ein Dataset fuer sie nutzlos. */
export function isVisionModel(m: Pick<ModelInfo, 'model_type' | 'plugin_override' | 'name'>): boolean {
  const s = `${m.model_type ?? ''} ${m.plugin_override ?? ''}`.toLowerCase();
  return isDetectionModel(m) || /resnet|vit|convnext|efficientnet|mobilenet|swin|deit|beit|image|vision|canvas|videomae|clip/.test(s);
}

/** Kurztext zum Pruefbericht, inklusive Warnung, wenn das Modell damit nicht trainieren kann. */
export function describeReport(r: FetchReport, model?: Pick<ModelInfo, 'model_type' | 'plugin_override' | 'name'>): string {
  const parts = [`format ${r.format}`];
  if (r.images) parts.push(`${r.images} image(s)`);
  if (r.audio) parts.push(`${r.audio} audio file(s)${r.transcripts ? `, ${r.transcripts} with transcript` : ''}`);
  if (r.video) parts.push(`${r.video} video(s)`);
  if (r.table) {
    const t = r.table;
    parts.push(`${t.rows} row(s), task ${t.task}, columns ${t.columns.slice(0, 8).join(', ')}`);
    const det = Object.entries(t.detected ?? {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('+') : String(v)}`).join(', ');
    if (det) parts.push(`detected ${det}`);
    const labels = Object.entries(t.labels ?? {});
    if (labels.length) parts.push(`labels ${labels.slice(0, 10).map(([k, v]) => `${k} ${v}`).join(', ')}${t.label_count > 10 ? ', ...' : ''}`);
  }
  if (r.yolo_labels) parts.push(`${r.yolo_labels} YOLO label file(s)`);
  if (r.classes.length) parts.push(`classes: ${r.classes.slice(0, 12).join(', ')}${r.classes.length > 12 ? ', ...' : ''}`);
  const splits = Object.entries(r.splits).map(([k, v]) => `${k} ${v}`).join(', ');
  if (splits) parts.push(`splits: ${splits}`);
  let text = parts.join('; ') + '.';
  const problems = [...r.problems];
  if (model && isVisionModel(model) && r.images === 0) {
    problems.unshift(`Not usable for "${model.name}": it needs images. Delete it with datasets_delete and pick another source.`);
  } else if (model && isDetectionModel(model) && r.format !== 'yolo' && r.format !== 'voc' && r.format !== 'coco') {
    problems.unshift(`"${model.name}" needs bounding-box labels (YOLO, Pascal VOC or COCO); this dataset has none.`);
  }
  if (problems.length) text += ` Problems: ${problems.join(' ')}`;
  return text;
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
    name: 'datasets_hf_files',
    title: 'Hugging Face dataset files',
    description: 'Shows the folders, file types and sizes of a Hugging Face dataset repo before downloading: total size, the largest folders with their image counts, and whether the data is stored as parquet. Use the folder paths as include for datasets_download_hf.',
    properties: { repo_id: S.str('Hugging Face dataset repo id, e.g. "rishitdagli/cppe-5".') },
    required: ['repo_id'],
    annotations: { openWorldHint: true },
    async run(args) {
      const info = await call<{ total_mb: number; files: number; folders: { path: string; mb: number; images: number }[]; has_parquet: boolean; image_files: number }>(
        'hf_dataset_files', { repoId: reqStr(args, 'repo_id') });
      const top = info.folders.slice(0, 6).map((f) => `${f.path} ${f.mb} MB${f.images ? ` (${f.images} images)` : ''}`).join(', ');
      return ok(`${info.files} file(s), ${info.total_mb} MB total${info.has_parquet ? ', parquet data' : ''}${info.image_files ? `, ${info.image_files} image files` : ''}. Folders: ${top}.`, info);
    },
  }),
  defineTool({
    name: 'datasets_download_hf',
    title: 'Download dataset',
    description: 'Downloads a Hugging Face dataset and attaches it to a library model. include limits the download to folders, files or glob patterns (e.g. ["sample/"]); max_mb refuses larger selections. For object detection models (YOLO etc.) images and boxes stored in parquet are converted into YOLO folders with data.yaml; max_rows keeps only the first N rows/images and downloads only as many parquet files as needed, so it works on very large repos like COCO. The result includes a content check (images, labels, classes, problems). Returns a job_id; system_job_status reports progress and the new dataset id.',
    properties: {
      repo_id: S.str('Hugging Face dataset repo id, e.g. "imdb".'),
      model_id: S.str('Model the dataset belongs to.'),
      dataset_name: S.str('Display name. Default: the repo id.'),
      include: S.strArr('Folders ("sample/"), files or glob patterns to download. Default: all data files.'),
      max_mb: S.num('Refuse the download if the selection is larger. Default 2000, max 20000.', { minimum: 1, maximum: 20000 }),
      max_rows: S.int('Detection datasets from parquet: keep at most this many images, e.g. 300 for a first run.', { minimum: 1 }),
      bbox_format: S.enumOf('Box format in the source data. Default "coco" (x, y, width, height in pixels; Hugging Face standard).', ['coco', 'xyxy', 'yolo']),
    },
    required: ['repo_id', 'model_id'],
    approval: 'once',
    annotations: { openWorldHint: true },
    async run(args) {
      const repoId = reqStr(args, 'repo_id');
      const model = await findModel(reqStr(args, 'model_id'));
      const datasetName = optStr(args, 'dataset_name') ?? repoId;
      const include = strArr(args, 'include');
      const maxMb = optNum(args, 'max_mb');
      const maxRows = optNum(args, 'max_rows');
      const detection = isDetectionModel(model);
      const targeted = detection || include.length > 0 || maxMb !== undefined || maxRows !== undefined;
      if (targeted) {
        return runAsJob('download', `Download dataset ${repoId}`, async () => {
          const r = await call<FetchResult>('fetch_dataset', {
            source: { kind: 'hf', repo_id: repoId, include, max_mb: maxMb, max_rows: maxRows, bbox_format: optStr(args, 'bbox_format'), yolo: detection },
            datasetName, modelId: model.id,
          });
          announceChange('datasets', { datasetId: r.dataset.id });
          return r;
        }, {
          waitMs: 0,
          describe: (r) => `Downloaded ${r.dataset.name} (${r.dataset.id})${r.conversion ? `, converted ${r.conversion.images} images with ${r.conversion.boxes} boxes to YOLO` : ''}. Check: ${describeReport(r.report, model)}`,
        });
      }
      return runAsJob('download', `Download dataset ${repoId}`, async () => {
        const d = await call<DatasetInfo>('download_huggingface_dataset', { repoId, datasetName, modelId: model.id });
        announceChange('datasets', { datasetId: d.id });
        const report = await call<FetchReport>('inspect_dataset', { datasetId: d.id }).catch(() => null);
        return { dataset: d, report };
      }, {
        waitMs: 0,
        describe: (r) => `Downloaded ${r.dataset.name} (${r.dataset.id}).${r.report ? ` Check: ${describeReport(r.report, model)}` : ''}`,
      });
    },
  }),
  defineTool({
    name: 'datasets_import_url',
    title: 'Import dataset from URL',
    description: 'Downloads data from one or more https URLs (ZIP/TAR archives or single files such as CSV, JSONL, parquet, images, audio) and attaches it to a library model, e.g. Zenodo, GitHub releases, Roboflow or university servers. Large ZIPs are read partially when the server allows it: max_files takes only that many media files, include filters archive entries by folder or pattern. With several URLs (e.g. an image archive and a label archive), later archives only contribute files whose names match the chosen media. COCO JSON is converted to YOLO, separate image and label trees are paired. The user confirms the URLs in FrameTrain. The result includes a content check.',
    properties: {
      urls: S.strArr('https URLs, e.g. [images.zip, labels.zip]. The first archive with media decides which files are taken.'),
      url: S.str('Single https URL (alternative to urls).'),
      model_id: S.str('Model the dataset belongs to.'),
      dataset_name: S.str('Display name.'),
      include: S.strArr('Only archive entries in these folders or matching these patterns, e.g. ["camera1/"] or ["*.jpg", "*.txt"].'),
      max_files: S.int('Take at most this many media files (images, audio, video) from the first archive, e.g. 100 for a first run.', { minimum: 1 }),
      class_names: S.strArr('Class names for YOLO label ids when the source has no class list (in id order).'),
      max_mb: S.num('Refuse larger downloads. Default 2000, max 20000.', { minimum: 1, maximum: 20000 }),
    },
    required: ['model_id', 'dataset_name'],
    approval: 'always',
    annotations: { openWorldHint: true },
    async run(args) {
      const urls = strArr(args, 'urls');
      const single = optStr(args, 'url');
      if (single) urls.push(single);
      if (urls.length === 0) throw new ToolArgError('Give url or urls.');
      if (urls.length > 10) throw new ToolArgError('At most 10 URLs per import.');
      for (const u of urls) if (!/^https:\/\//i.test(u)) throw new ToolFailure('Only https:// URLs are allowed.');
      const model = await findModel(reqStr(args, 'model_id'));
      const datasetName = reqStr(args, 'dataset_name');
      return runAsJob('import', `Import ${datasetName} from URL`, async () => {
        const r = await call<FetchResult>('fetch_dataset', {
          source: {
            kind: 'url', urls, include: strArr(args, 'include'), max_files: optNum(args, 'max_files'),
            class_names: strArr(args, 'class_names'), max_mb: optNum(args, 'max_mb'), yolo: isDetectionModel(model),
          },
          datasetName, modelId: model.id,
        });
        announceChange('datasets', { datasetId: r.dataset.id });
        return r;
      }, { waitMs: 0, describe: (r) => `Imported ${r.dataset.name} (${r.dataset.id})${r.conversion ? ` (${String((r.conversion as { kind?: string }).kind ?? 'converted')})` : ''}. Check: ${describeReport(r.report, model)}` });
    },
  }),
  defineTool({
    name: 'datasets_view',
    title: 'View dataset images',
    description: 'Returns images or video frames of a dataset as pictures, with its labels drawn in (numbered boxes with class names), so you can judge image quality and label correctness yourself. Pick files with file_paths (from datasets_files) or let it take count files from the start or at random.',
    properties: {
      dataset_id: S.str('Dataset id from datasets_list.'),
      file_paths: S.strArr('Files to show (paths from datasets_files). Max 6.'),
      count: S.int('Without file_paths: number of images to show (1-6). Default 3.', { minimum: 1, maximum: 6 }),
      random: S.bool('Without file_paths: pick random images instead of the first ones.'),
      max_size: S.int('Longest image side in pixels (256-1280). Default 768.', { minimum: 256, maximum: 1280 }),
    },
    required: ['dataset_id'],
    async run(args) {
      const d = await findDataset(reqStr(args, 'dataset_id'));
      let paths = strArr(args, 'file_paths').slice(0, 6);
      if (paths.length === 0) {
        const files = (await datasetFiles(d.id)).filter((f) => !f.is_dir && /\.(jpe?g|png|bmp|webp|tiff?|mp4|mov|avi|mkv|webm)$/i.test(f.name));
        if (files.length === 0) throw new ToolFailure(`${d.name} has no images or videos. Use datasets_preview for text and tables.`);
        const n = Math.min(optNum(args, 'count', 3)!, 6);
        const pool = args.random === true ? [...files].sort(() => Math.random() - 0.5) : files;
        paths = pool.slice(0, n).map((f) => f.path);
      }
      const images: { data: string; mimeType: string }[] = [];
      const lines: string[] = [];
      for (const [i, path] of paths.entries()) {
        try {
          const r = await call<{ data: string; mime: string; boxes: number; classes: string[] }>('preview_media', { path, maxSize: optNum(args, 'max_size', 768) });
          images.push({ data: r.data, mimeType: r.mime });
          lines.push(`${i + 1}. ${path.split(/[\\/]/).pop()}: ${r.boxes} box(es)`);
        } catch (e) {
          lines.push(`${i + 1}. ${path}: ${(e as Error).message ?? String(e)}`);
        }
      }
      return withImages(ok(`${d.name}: ${images.length} image(s).\n${lines.join('\n')}`), images);
    },
  }),
  defineTool({
    name: 'datasets_check',
    title: 'Check dataset',
    description: 'Checks what a dataset contains: format (yolo, voc, coco, images, parquet, table), image and label counts, classes, splits, and problems such as missing images or labels. With model_id it also says whether that model can train on it.',
    properties: {
      dataset_id: S.str('Dataset id from datasets_list.'),
      model_id: S.str('Optional: model to check the dataset against.'),
    },
    required: ['dataset_id'],
    async run(args) {
      const d = await findDataset(reqStr(args, 'dataset_id'));
      const modelId = optStr(args, 'model_id') ?? d.model_id;
      const model = modelId ? await findModel(modelId).catch(() => undefined) : undefined;
      const report = await call<FetchReport>('inspect_dataset', { datasetId: d.id });
      return ok(`${d.name}: ${describeReport(report, model)}`, report);
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
