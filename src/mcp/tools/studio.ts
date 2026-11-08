// Domaene studio: Dataset Builder (Projekte, Samples, Annotation, Export).

import {
  call, defineTool, ok, optBool, optNum, optObj, optStr, reqStr, S, strArr, ToolArgError, ToolFailure, withImages,
} from '../util';
import type { JsonSchema, McpTool, ToolArgs } from '../types';
import { announceChange, runAsJob } from './common';
import { findModel } from './lookup';

/**
 * Projektarten wie im Anlegen-Dialog (StudioPanel.tsx, PROJEKTARTEN).
 * Kopie statt Import, damit die Werkzeuge nicht die ganze Seite laden.
 */
export const STUDIO_KINDS = {
  image_boxes:    { modality: 'image', task: 'bbox',           targetFormat: 'yolo_bbox',        classes: true,  hint: 'Object detection: draw boxes per class (YOLO export).' },
  image_classify: { modality: 'image', task: 'classify',       targetFormat: 'folder_class',     classes: true,  hint: 'Image classification: one class per image.' },
  video_classify: { modality: 'video', task: 'classify',       targetFormat: 'folder_class',     classes: true,  hint: 'Video clip classification.' },
  text_classify:  { modality: 'text',  task: 'classification', targetFormat: 'flat_file',        classes: true,  hint: 'Text classification: one label per text.' },
  text_pairs:     { modality: 'text',  task: 'pairs',          targetFormat: 'flat_file',        classes: false, hint: 'Text pairs (input -> target), e.g. for seq2seq or LLM fine-tuning.' },
  audio_classify: { modality: 'audio', task: 'classification', targetFormat: 'folder_class',     classes: true,  hint: 'Audio classification.' },
  audio_transcript: { modality: 'audio', task: 'transcript',   targetFormat: 'audio_transcript', classes: false, hint: 'Speech recognition: audio with transcript.' },
} as const;
type StudioKind = keyof typeof STUDIO_KINDS;


interface ProjectBrief { id: string; name: string; modality: string; task: string; classes: string[] }

/** Projektart aus modality + task (Gegenstueck zu STUDIO_KINDS). */
export function kindOf(p: Pick<ProjectBrief, 'modality' | 'task'>): StudioKind | null {
  const hit = Object.entries(STUDIO_KINDS).find(([, v]) => v.modality === p.modality && v.task === p.task);
  return hit ? (hit[0] as StudioKind) : null;
}

async function findProject(pid: string): Promise<ProjectBrief> {
  const list = await call<ProjectBrief[]>('studio_list_projects');
  const p = list.find((x) => x.id === pid);
  if (!p) throw new ToolFailure(`Studio project "${pid}" not found. Use studio_projects_list.`);
  return p;
}

interface StagedResult {
  studio_source: 'folder' | 'text' | 'audio' | 'videos'; path: string; work_dir: string;
  label_classes?: string[]; text_column?: string; label_column?: string; paths?: string[];
  samples: number; per_class?: Record<string, number>; dropped_classes?: Record<string, number>;
  source_columns?: Record<string, unknown>; notes?: string[];
}

/** Dataset -> Studio: vorbereiten (Klassen, Stichprobe), mit dem passenden Import einlesen, aufraeumen. */
async function importDataset(pid: string, args: ToolArgs): Promise<ImportReport & { staged: StagedResult }> {
  const project = await findProject(pid);
  const kind = kindOf(project);
  if (!kind) throw new ToolFailure(`Unsupported project type ${project.modality}/${project.task}.`);
  const classMap = optObj(args, 'class_map');
  if (classMap) {
    for (const [k, v] of Object.entries(classMap)) {
      if (v !== null && typeof v !== 'string') throw new ToolArgError(`class_map["${k}"] must be a class name or null.`);
    }
  }
  const staged = await call<StagedResult>('stage_dataset', {
    datasetId: reqStr(args, 'dataset_id'),
    request: {
      project_kind: kind, class_map: classMap ?? {}, keep_unmapped: args.keep_unmapped !== false,
      max_samples: optNum(args, 'max_samples'), per_class: optBool(args, 'per_class'), keep_empty: optBool(args, 'keep_empty'),
      text_column: optStr(args, 'text_column'), label_column: optStr(args, 'label_column'),
    },
  });
  try {
    let report: ImportReport;
    switch (staged.studio_source) {
      case 'folder':
        report = await call('studio_import_folder', { projectId: pid, sourcePath: staged.path, labelClasses: staged.label_classes ?? null, ignoreLabels: false });
        break;
      case 'text':
        report = await call('studio_import_text', { projectId: pid, sourcePath: staged.path, textColumn: staged.text_column ?? null, labelColumn: staged.label_column ?? null, ignoreLabels: false });
        break;
      case 'audio':
        report = await call('studio_import_audio', { projectId: pid, sourcePath: staged.path, ignoreLabels: false });
        break;
      case 'videos':
        report = await call('studio_import_videos', { projectId: pid, paths: staged.paths ?? [], clipSeconds: null, ignoreLabels: false });
        break;
    }
    return { ...report, staged };
  } finally {
    await call('discard_staged', { workDir: staged.work_dir }).catch(() => undefined);
  }
}

function clipText(t: string, n: number): string { return t.length > n ? `${t.slice(0, n)} ...` : t; }

const SAMPLE_STATUS = ['new', 'suggested', 'confirmed', 'skipped'] as const;

interface ImportReport { added: number; duplicates: number; unreadable: number; with_labels?: number; classes_added?: string[] }

function importText(r: ImportReport): string {
  return `Added ${r.added} sample(s), ${r.duplicates} duplicate(s) skipped, ${r.unreadable} unreadable.`;
}

const projectId: JsonSchema = S.str('Studio project id from studio_projects_list.');

function changed(projectIdValue: string) {
  announceChange('studio', { projectId: projectIdValue });
}

/** base64 (auch data:-URL) zu Byte-Array, wie Tauri es fuer Vec<u8> erwartet. */
export function base64ToBytes(input: string): number[] {
  const b64 = input.includes(',') && input.startsWith('data:') ? input.slice(input.indexOf(',') + 1) : input;
  let bin: string;
  try { bin = atob(b64.replace(/\s+/g, '')); } catch { throw new ToolArgError('image_base64 is not valid base64.'); }
  const out = new Array<number>(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function boxesArg(args: ToolArgs): { cls: number; x: number; y: number; w: number; h: number }[] {
  const raw = args.boxes;
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) throw new ToolArgError('boxes must be a list.');
  return raw.map((b, i) => {
    const o = b as Record<string, unknown>;
    const nums = ['cls', 'x', 'y', 'w', 'h'].map((k) => o?.[k]);
    if (nums.some((n) => typeof n !== 'number')) throw new ToolArgError(`boxes[${i}] needs numeric cls, x, y, w, h.`);
    return { cls: o.cls as number, x: o.x as number, y: o.y as number, w: o.w as number, h: o.h as number };
  });
}

export const studioTools: McpTool[] = [
  defineTool({
    name: 'studio_projects_list',
    title: 'Studio projects',
    description: 'Lists Dataset Builder (Studio) projects with id, name, modality, task, classes and sample counts. A Studio project collects and labels raw samples and is exported into a training dataset with studio_export.',
    async run() {
      const list = await call<unknown[]>('studio_list_projects');
      return ok(`${list.length} project(s).`, list);
    },
  }),
  defineTool({
    name: 'studio_project_create',
    title: 'Create studio project',
    description: `Creates a Dataset Builder project. kind: ${Object.entries(STUDIO_KINDS).map(([k, v]) => `${k} = ${v.hint}`).join(' ')} Classes are required for kinds with classes and can be extended later with studio_project_update.`,
    properties: {
      name: S.str('Project name.'),
      kind: S.enumOf('Project kind.', Object.keys(STUDIO_KINDS)),
      classes: S.strArr('Class names (for classification and box projects).'),
    },
    required: ['name', 'kind'],
    approval: 'once',
    async run(args) {
      const kind = reqStr(args, 'kind') as StudioKind;
      const art = STUDIO_KINDS[kind];
      if (!art) throw new ToolArgError(`Unknown kind "${kind}".`);
      const classes = art.classes ? strArr(args, 'classes') : [];
      const project = await call<{ id: string; name: string }>('studio_create_project', {
        name: reqStr(args, 'name'), modality: art.modality, task: art.task, targetFormat: art.targetFormat, classes,
      });
      changed(project.id);
      return ok(`Created project ${project.name} (${project.id}).`, { project });
    },
  }),
  defineTool({
    name: 'studio_project_update',
    title: 'Update studio project',
    description: 'Renames a Studio project and/or replaces its class list. Keep the existing order when adding classes: box annotations reference classes by index.',
    properties: { project_id: projectId, name: S.str('New name.'), classes: S.strArr('Full new class list.') },
    required: ['project_id'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const project = await call('studio_update_project', {
        projectId: pid, name: optStr(args, 'name') ?? null, classes: args.classes === undefined ? null : strArr(args, 'classes'),
      });
      changed(pid);
      return ok('Project updated.', { project });
    },
  }),
  defineTool({
    name: 'studio_project_delete',
    title: 'Delete studio project',
    description: 'Permanently deletes a Studio project with all samples and annotations. The user must confirm every call.',
    properties: { project_id: projectId },
    required: ['project_id'],
    approval: 'always',
    annotations: { destructiveHint: true },
    async run(args) {
      const pid = reqStr(args, 'project_id');
      await call('studio_delete_project', { projectId: pid });
      changed(pid);
      return ok('Project deleted.', { project_id: pid, deleted: true });
    },
  }),
  defineTool({
    name: 'studio_samples_list',
    title: 'List samples',
    description: 'Lists samples of a Studio project (paginated), optionally filtered by status (new, suggested, confirmed, skipped). Each sample has its id, content/text or media path, label/target/boxes and status.',
    properties: {
      project_id: projectId,
      status: S.enumOf('Only samples with this status.', SAMPLE_STATUS),
      offset: S.int('Default 0.', { minimum: 0 }),
      limit: S.int('1-200, default 50.', { minimum: 1, maximum: 200 }),
    },
    required: ['project_id'],
    async run(args) {
      const page = await call<{ total: number; items: unknown[] }>('studio_list_samples', {
        projectId: reqStr(args, 'project_id'), status: optStr(args, 'status') ?? null,
        offset: optNum(args, 'offset', 0), limit: optNum(args, 'limit', 50),
      });
      return ok(`${page.items.length} of ${page.total} sample(s).`, page);
    },
  }),
  defineTool({
    name: 'studio_view_samples',
    title: 'View samples',
    description: 'Shows samples of a Studio project as pictures so you can check them yourself: images (and video clips as a middle frame) with their boxes drawn in, numbered with class names. Text and audio samples are returned as text with their label or transcript. Use it to verify labels, find missing or wrong boxes, then fix them with studio_set_annotation; boxes are listed in the same order as the numbers in the picture.',
    properties: {
      project_id: projectId,
      sample_ids: S.strArr('Samples to show (from studio_samples_list). Max 6.'),
      status: S.enumOf('Without sample_ids: show samples with this status.', SAMPLE_STATUS),
      count: S.int('Without sample_ids: how many (1-6). Default 4.', { minimum: 1, maximum: 6 }),
      max_size: S.int('Longest image side in pixels (256-1280). Default 768.', { minimum: 256, maximum: 1280 }),
    },
    required: ['project_id'],
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const project = await findProject(pid);
      const wanted = strArr(args, 'sample_ids').slice(0, 6);
      type Sample = { id: string; status: string; content?: string | null; abs_path?: string; mime?: string;
        ann?: { label?: string | null; target?: string | null; boxes?: { cls: number; x: number; y: number; w: number; h: number }[] } };
      const picked: Sample[] = [];
      for (let offset = 0; offset < 20000 && picked.length < (wanted.length || 6); offset += 200) {
        const page = await call<{ total: number; items: Sample[] }>('studio_list_samples', {
          projectId: pid, status: wanted.length ? null : optStr(args, 'status') ?? null, offset, limit: 200,
        });
        for (const it of page.items) {
          if (wanted.length ? wanted.includes(it.id) : picked.length < optNum(args, 'count', 4)!) picked.push(it);
        }
        if (offset + 200 >= page.total) break;
      }
      if (picked.length === 0) throw new ToolFailure('No matching samples.');
      const images: { data: string; mimeType: string }[] = [];
      const lines: string[] = [];
      for (const [i, sm] of picked.entries()) {
        const boxes = sm.ann?.boxes ?? [];
        const head = `${i + 1}. ${sm.id} (${sm.status})`;
        if (sm.abs_path && /^(image|video)\//.test(sm.mime ?? '') ) {
          try {
            const r = await call<{ data: string; mime: string }>('preview_media', {
              path: sm.abs_path, boxes, classes: project.classes, maxSize: optNum(args, 'max_size', 768),
            });
            images.push({ data: r.data, mimeType: r.mime });
            const list = boxes.map((b, j) => `${j + 1}:${project.classes[b.cls] ?? b.cls}`).join(', ');
            lines.push(`${head}: image ${images.length}${sm.ann?.label ? `, label ${sm.ann.label}` : ''}${list ? `, boxes ${list}` : ', no boxes'}`);
            continue;
          } catch (e) {
            lines.push(`${head}: preview failed (${(e as Error).message ?? e})`);
            continue;
          }
        }
        const label = sm.ann?.label ?? sm.ann?.target ?? '(none)';
        lines.push(`${head}: ${clipText(sm.content ?? sm.abs_path ?? '', 400)} -> ${label}`);
      }
      return withImages(ok(`${project.name} (${project.classes.join(', ') || 'no classes'}):\n${lines.join('\n')}`), images);
    },
  }),
  defineTool({
    name: 'studio_stats',
    title: 'Project statistics',
    description: 'Returns counts per status, boxes per class, empty confirmed samples and model doubts for a Studio project. Shows whether a project is ready for export.',
    properties: { project_id: projectId },
    required: ['project_id'],
    async run(args) {
      const stats = await call<Record<string, unknown>>('studio_stats', { projectId: reqStr(args, 'project_id') });
      return ok(`${stats.total} sample(s): ${stats.confirmed} confirmed, ${stats.suggested} suggested, ${stats.new} new.`, stats);
    },
  }),
  defineTool({
    name: 'studio_add_texts',
    title: 'Add texts',
    description: 'Adds text samples to a text Studio project. For text_classify projects set label (a class name); for text_pairs projects set target (the expected output). Samples with label/target arrive as suggestions; confirm them with studio_set_annotation or in the app.',
    properties: {
      project_id: projectId,
      items: S.arr('Texts to add (max 1000 per call).', S.obj('One text sample.', {
        text: S.str('The text (input).'),
        label: S.str('Class name (classification projects).'),
        target: S.str('Target text (pair projects).'),
      }, { required: ['text'], additionalProperties: false }), { minItems: 1, maxItems: 1000 }),
      origin: S.str('Where the texts come from (shown as provenance), e.g. "generated by Claude".'),
    },
    required: ['project_id', 'items'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const items = (args.items as { text: string; label?: string; target?: string }[])
        .map((i) => ({ text: i.text, label: i.label ?? null, target: i.target ?? null }));
      const report = await call<ImportReport>('studio_add_texts', { projectId: pid, items, origin: optStr(args, 'origin') ?? `mcp` });
      changed(pid);
      return ok(importText(report), report);
    },
  }),
  defineTool({
    name: 'studio_edit_text',
    title: 'Edit text',
    description: 'Replaces the text content of one text sample in a Studio project (annotation stays).',
    properties: { project_id: projectId, sample_id: S.str('Sample id.'), content: S.str('New text.') },
    required: ['project_id', 'sample_id', 'content'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      await call('studio_edit_text', { projectId: pid, sampleId: reqStr(args, 'sample_id'), content: reqStr(args, 'content') });
      changed(pid);
      return ok('Text updated.', { sample_id: args.sample_id });
    },
  }),
  defineTool({
    name: 'studio_add_image',
    title: 'Add image',
    description: 'Adds one image (base64-encoded PNG/JPEG/WebP, max ~15 MB) to an image Studio project. studio_fetch_web collects many images from the web at once.',
    properties: {
      project_id: projectId,
      image_base64: S.str('Image bytes as base64 or data: URL.'),
      origin: S.str('Provenance, e.g. a source URL.'),
    },
    required: ['project_id', 'image_base64'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const bytes = base64ToBytes(reqStr(args, 'image_base64'));
      if (bytes.length > 15 * 1024 * 1024) throw new ToolArgError('Image is larger than 15 MB.');
      const report = await call<ImportReport>('studio_add_image', { projectId: pid, bytes, origin: optStr(args, 'origin') ?? 'mcp' });
      changed(pid);
      return ok(importText(report), report);
    },
  }),
  defineTool({
    name: 'studio_delete_samples',
    title: 'Delete samples',
    description: 'Permanently removes samples from a Studio project. The user must confirm every call.',
    properties: { project_id: projectId, sample_ids: S.strArr('Sample ids to delete.', { minItems: 1 }) },
    required: ['project_id', 'sample_ids'],
    approval: 'always',
    annotations: { destructiveHint: true },
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const n = await call<number>('studio_delete_samples', { projectId: pid, sampleIds: strArr(args, 'sample_ids', true) });
      changed(pid);
      return ok(`Deleted ${n} sample(s).`, { deleted: n });
    },
  }),
  defineTool({
    name: 'studio_inspect_source',
    title: 'Inspect source',
    description: 'Inspects a local folder (images/audio with class subfolders or YOLO labels) or a text file (CSV/JSONL/TXT: detected columns, label column) before importing it. The user confirms the path.',
    properties: {
      source_path: S.str('Absolute path of the folder or file.'),
      as_text: S.bool('Inspect as text file (CSV/JSONL/TXT). Default false = folder.'),
      text_column: S.str('Text column for text files.'),
      label_column: S.str('Label column for text files.'),
    },
    required: ['source_path'],
    approval: 'always',
    async run(args) {
      const sourcePath = reqStr(args, 'source_path');
      const res = optBool(args, 'as_text')
        ? await call('studio_inspect_text', { sourcePath, textColumn: optStr(args, 'text_column') ?? null, labelColumn: optStr(args, 'label_column') ?? null })
        : await call('studio_inspect_folder', { sourcePath });
      return ok('Source inspected.', { inspection: res });
    },
  }),
  defineTool({
    name: 'studio_import',
    title: 'Import into project',
    description: 'Imports data into a Studio project. source "dataset" takes a FrameTrain dataset (from datasets_list, e.g. after datasets_download_hf or datasets_import_url) of any kind: images with boxes, class folders of images/audio/video, audio with transcripts, text tables for classification or pairs. class_map renames, merges or drops classes so several sources end up with one class list (e.g. {"car": "vehicle", "truck": "vehicle", "dog": null}); keep_unmapped=false drops every class not in class_map; max_samples (optionally per_class) draws a sample. Other sources read local files: "folder" (images, class subfolders or YOLO labels), "text" (CSV/JSONL/TXT file), "audio" (folder), "videos" (list of video files as clips), "video_frames" (frames of one video). The user confirms the import. Large imports return a job_id.',
    properties: {
      project_id: projectId,
      source: S.enumOf('Import type.', ['dataset', 'folder', 'text', 'audio', 'videos', 'video_frames']),
      dataset_id: S.str('Dataset id (source "dataset").'),
      class_map: S.obj('Source class name -> project class name, or null to drop (source "dataset"). Case-insensitive.'),
      keep_unmapped: S.bool('Keep classes that are not in class_map under their own name. Default true.'),
      max_samples: S.int('Take at most this many samples (source "dataset").', { minimum: 1 }),
      per_class: S.bool('With max_samples: balance the sample across classes (rare classes first).'),
      keep_empty: S.bool('Boxes: keep images whose boxes were all dropped by class_map (as background). Default false.'),
      source_path: S.str('Folder or file path (all sources except "videos").'),
      paths: S.strArr('Video file paths (source "videos").'),
      ignore_labels: S.bool('Do not take over existing labels. Default false.'),
      label_classes: S.strArr('Class names for YOLO label ids (source "folder").'),
      text_column: S.str('Text column (source "text").'),
      label_column: S.str('Label column (source "text").'),
      clip_seconds: S.num('Clip length in seconds (source "videos").', { minimum: 0.5 }),
      every_n: S.int('Take every n-th frame (source "video_frames"). Default 30.', { minimum: 1 }),
      max_frames: S.int('Maximum frames (source "video_frames"). Default 200.', { minimum: 1, maximum: 5000 }),
    },
    required: ['project_id', 'source'],
    approval: 'always',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const ignoreLabels = optBool(args, 'ignore_labels');
      const source = reqStr(args, 'source');
      let work: () => Promise<ImportReport>;
      switch (source) {
        case 'dataset':
          work = () => importDataset(pid, args);
          break;
        case 'folder':
          work = () => call('studio_import_folder', { projectId: pid, sourcePath: reqStr(args, 'source_path'), labelClasses: args.label_classes === undefined ? null : strArr(args, 'label_classes'), ignoreLabels });
          break;
        case 'text':
          work = () => call('studio_import_text', { projectId: pid, sourcePath: reqStr(args, 'source_path'), textColumn: optStr(args, 'text_column') ?? null, labelColumn: optStr(args, 'label_column') ?? null, ignoreLabels });
          break;
        case 'audio':
          work = () => call('studio_import_audio', { projectId: pid, sourcePath: reqStr(args, 'source_path'), ignoreLabels });
          break;
        case 'videos':
          work = () => call('studio_import_videos', { projectId: pid, paths: strArr(args, 'paths', true), clipSeconds: optNum(args, 'clip_seconds') ?? null, ignoreLabels });
          break;
        case 'video_frames':
          work = () => call('studio_import_video', { projectId: pid, videoPath: reqStr(args, 'source_path'), everyN: optNum(args, 'every_n', 30), maxFrames: optNum(args, 'max_frames', 200) });
          break;
        default:
          throw new ToolArgError(`Unknown source "${source}".`);
      }
      return runAsJob('studio_import', `Studio import (${source})`, async () => {
        const r = await work();
        changed(pid);
        return r;
      }, {
        describe: (r) => {
          const st = (r as ImportReport & { staged?: StagedResult }).staged;
          if (!st) return importText(r);
          const per = st.per_class ? ` Per class: ${Object.entries(st.per_class).map(([k, v]) => `${k} ${v}`).join(', ')}.` : '';
          const dropped = st.dropped_classes ? ` Dropped classes: ${Object.keys(st.dropped_classes).slice(0, 10).join(', ')}.` : '';
          const fmt = st.source_columns?.format ? ` Read as ${String(st.source_columns.format)} (${(st.source_columns.columns as string[] | undefined)?.join(' + ') ?? ''}).` : '';
          const notes = st.notes?.length ? ` ${st.notes.join(' ')}` : '';
          return `${importText(r)}${per}${dropped}${fmt}${notes}`;
        },
      });
    },
  }),
  defineTool({
    name: 'studio_fetch_web',
    title: 'Fetch from the web',
    description: 'Downloads images or text from web pages into a Studio project (respecting robots.txt). mode "urls" fetches the given pages/files, "crawl" follows links up to max_depth within the allowlist, "sitemap" reads sitemaps. Fetched samples arrive as suggestions (with label if given). Returns a job_id when it takes longer than ~20 s.',
    properties: {
      project_id: projectId,
      urls: S.strArr('Start URLs (http/https).', { minItems: 1 }),
      mode: S.enumOf('Fetch mode. Default "urls".', ['urls', 'crawl', 'sitemap']),
      max_depth: S.int('Link depth for crawl. Default 1.', { minimum: 0, maximum: 5 }),
      max_pages: S.int('Default 50.', { minimum: 1, maximum: 1000 }),
      max_files: S.int('Default 500.', { minimum: 1, maximum: 5000 }),
      allowlist: S.strArr('Allowed domains (include subdomains). Default: domains of the start URLs.'),
      label: S.str('Class for everything fetched in this run.'),
      license: S.str('License note stored as provenance.'),
      paragraphs: S.bool('Split text pages into paragraphs. Default true.'),
      min_chars: S.int('Minimum characters per text sample. Default 40.', { minimum: 1 }),
      min_side: S.int('Skip images smaller than this (px). Default 64.', { minimum: 1 }),
    },
    required: ['project_id', 'urls'],
    approval: 'once',
    annotations: { openWorldHint: true },
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const urls = strArr(args, 'urls', true);
      if (urls.some((u) => !/^https?:\/\//i.test(u))) throw new ToolArgError('Only http(s) URLs are allowed.');
      const options = {
        mode: optStr(args, 'mode') ?? 'urls',
        max_depth: optNum(args, 'max_depth', 1),
        max_pages: optNum(args, 'max_pages', 50),
        max_files: optNum(args, 'max_files', 500),
        allowlist: args.allowlist === undefined ? urls.map((u) => { try { return new URL(u).hostname; } catch { return ''; } }).filter(Boolean) : strArr(args, 'allowlist'),
        max_mb: 25,
        min_side: optNum(args, 'min_side', 64),
        paragraphs: optBool(args, 'paragraphs', true),
        min_chars: optNum(args, 'min_chars', 40),
        license: optStr(args, 'license') ?? null,
        label: optStr(args, 'label') ?? null,
      };
      return runAsJob('studio_fetch', 'Web fetch', async () => {
        const r = await call<Record<string, unknown>>('studio_fetch_web', { projectId: pid, urls, options });
        changed(pid);
        return r;
      }, { describe: (r) => `Web fetch done: ${r.fetched ?? 0} item(s) fetched.` });
    },
  }),
  defineTool({
    name: 'studio_set_annotation',
    title: 'Set annotation',
    description: 'Sets the annotation of one sample: status (new, suggested, confirmed, skipped) plus label (class name for classification), target (pair/transcript projects) or boxes (detection: cls = class index, x/y = box center, w/h = size, all normalized 0..1). Only confirmed samples are exported by default.',
    properties: {
      project_id: projectId,
      sample_id: S.str('Sample id.'),
      status: S.enumOf('New status.', SAMPLE_STATUS),
      label: S.str('Class name.'),
      target: S.str('Target text.'),
      boxes: S.arr('Bounding boxes (detection projects).', S.obj('Box', {
        cls: S.int('Class index.', { minimum: 0 }), x: S.num('Center x 0..1'), y: S.num('Center y 0..1'), w: S.num('Width 0..1'), h: S.num('Height 0..1'),
      }, { required: ['cls', 'x', 'y', 'w', 'h'], additionalProperties: false })),
    },
    required: ['project_id', 'sample_id', 'status'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      await call('studio_set_annotation', {
        projectId: pid, sampleId: reqStr(args, 'sample_id'), boxes: boxesArg(args), status: reqStr(args, 'status'),
        label: optStr(args, 'label') ?? null, target: optStr(args, 'target') ?? null,
      });
      changed(pid);
      return ok('Annotation saved.', { sample_id: args.sample_id, status: args.status });
    },
  }),
  defineTool({
    name: 'studio_suggest',
    title: 'Model suggestions',
    description: 'Lets a trained model version pre-label open samples of a Studio project (boxes, classes or target texts). Confirmed samples are never changed. Can take a while; returns a job_id after ~20 s.',
    properties: {
      project_id: projectId,
      version_id: S.str('Trained model version id whose task matches the project.'),
      min_confidence: S.num('Ignore predictions below this confidence (0..1). Default 0.4.', { minimum: 0, maximum: 1 }),
      add_unknown_classes: S.bool('Add classes the model knows but the project does not. Default false.'),
      limit: S.int('Only this many samples.', { minimum: 1 }),
    },
    required: ['project_id', 'version_id'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      return runAsJob('studio_suggest', 'Model suggestions', async () => {
        const r = await call<Record<string, unknown>>('studio_suggest', {
          projectId: pid, versionId: reqStr(args, 'version_id'), minConfidence: optNum(args, 'min_confidence', 0.4),
          addUnknownClasses: optBool(args, 'add_unknown_classes'), limit: optNum(args, 'limit') ?? null,
        });
        changed(pid);
        return r;
      }, { describe: (r) => `Processed ${r.processed ?? '?'} sample(s).` });
    },
  }),
  defineTool({
    name: 'studio_review',
    title: 'Model review',
    description: 'Lets a trained model check confirmed labels and marks samples where the model disagrees as doubts (see studio_stats.doubts). Returns checked/agree/doubts counts.',
    properties: {
      project_id: projectId,
      version_id: S.str('Trained model version id.'),
      min_confidence: S.num('Only flag disagreements above this confidence. Default 0.6.', { minimum: 0, maximum: 1 }),
    },
    required: ['project_id', 'version_id'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      return runAsJob('studio_review', 'Model review', async () => {
        const r = await call<Record<string, number>>('studio_review', { projectId: pid, versionId: reqStr(args, 'version_id'), minConfidence: optNum(args, 'min_confidence', 0.6) });
        changed(pid);
        return r;
      }, { describe: (r) => `Checked ${r.checked}, ${r.doubts} doubt(s), ${r.agree} agree.` });
    },
  }),
  defineTool({
    name: 'studio_near_duplicates',
    title: 'Near duplicates',
    description: 'Finds groups of near-duplicate samples (similar texts or images; exact duplicates for audio/video). Images without a perceptual hash are reported under "missing" (hashes are computed when the project is opened in the app).',
    properties: { project_id: projectId },
    required: ['project_id'],
    async run(args) {
      const r = await call<{ kind: string; groups: string[][]; checked: number; missing: unknown[] }>('studio_near_duplicates', { projectId: reqStr(args, 'project_id') });
      return ok(`${r.groups.length} group(s) among ${r.checked} checked sample(s).`, { kind: r.kind, groups: r.groups, checked: r.checked, missing_hashes: r.missing.length });
    },
  }),
  defineTool({
    name: 'studio_quality',
    title: 'Quality report',
    description: 'One-call quality overview of a Studio project: status counts, class balance, empty confirmed samples, model doubts and near-duplicate groups, plus plain-language warnings.',
    properties: { project_id: projectId },
    required: ['project_id'],
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const stats = await call<Record<string, number | number[]>>('studio_stats', { projectId: pid });
      let dups: { groups: string[][] } | null = null;
      try { dups = await call('studio_near_duplicates', { projectId: pid }); } catch { /* optional */ }
      const warnings: string[] = [];
      const perClass = (stats.per_class as number[] | undefined) ?? [];
      if ((stats.confirmed as number) < 20) warnings.push('Fewer than 20 confirmed samples — too little for training.');
      if (perClass.length > 1) {
        const max = Math.max(...perClass), min = Math.min(...perClass);
        if (min === 0) warnings.push('At least one class has no annotations.');
        else if (max / min > 5) warnings.push(`Class imbalance: largest class has ${max}, smallest ${min}.`);
      }
      if ((stats.doubts as number) > 0) warnings.push(`${stats.doubts} sample(s) flagged as doubtful by a model review.`);
      if (dups && dups.groups.length) warnings.push(`${dups.groups.length} near-duplicate group(s).`);
      return ok(warnings.length ? warnings.join(' ') : 'No quality issues found.', { stats, near_duplicate_groups: dups?.groups.length ?? null, warnings });
    },
  }),
  defineTool({
    name: 'studio_export',
    title: 'Export as dataset',
    description: 'Exports a Studio project as a training dataset attached to a model (with train/val/test split, provenance and data card). Returns the new dataset id (via job_id if it takes longer than ~20 s).',
    properties: {
      project_id: projectId,
      model_id: S.str('Model the new dataset belongs to.'),
      dataset_name: S.str('Name of the new dataset.'),
      include_suggested: S.bool('Also export unconfirmed suggestions. Default false.'),
      train_ratio: S.num('Default 0.8.', { minimum: 0, maximum: 1 }),
      val_ratio: S.num('Default 0.1 (rest is test).', { minimum: 0, maximum: 1 }),
    },
    required: ['project_id', 'model_id', 'dataset_name'],
    approval: 'once',
    async run(args) {
      const pid = reqStr(args, 'project_id');
      const model = await findModel(reqStr(args, 'model_id'));
      const train = optNum(args, 'train_ratio', 0.8)!, val = optNum(args, 'val_ratio', 0.1)!;
      if (train + val > 1) throw new ToolFailure('train_ratio + val_ratio must not exceed 1.');
      return runAsJob('studio_export', 'Studio export', async () => {
        const r = await call<{ dataset: { id: string; name: string }; report: unknown; path: string }>('studio_export', {
          projectId: pid, modelId: model.id, datasetName: reqStr(args, 'dataset_name'),
          includeSuggested: optBool(args, 'include_suggested'), trainRatio: train, valRatio: val,
        });
        announceChange('datasets', { datasetId: r.dataset.id });
        return r;
      }, { describe: (r) => `Exported as dataset ${r.dataset.name} (${r.dataset.id}).` });
    },
  }),
  defineTool({
    name: 'studio_model_classes',
    title: 'Model classes',
    description: 'Returns the class names a trained model version predicts — useful to set up a Studio project that matches a model for studio_suggest.',
    properties: { version_id: S.str('Trained model version id.') },
    required: ['version_id'],
    async run(args) {
      const classes = await call<string[]>('studio_model_classes', { versionId: reqStr(args, 'version_id') });
      return ok(`${classes.length} class(es): ${classes.join(', ')}`, { classes });
    },
  }),
];
