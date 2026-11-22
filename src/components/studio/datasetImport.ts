// Ein FrameTrain-Dataset in ein Werkstatt-Projekt holen.
//
// Derselbe Weg fuer die Oberflaeche und fuer KI-Clients (mcp/tools/studio.ts):
// das Dataset wird vorbereitet (Klassen zuordnen, Stichprobe), mit dem passenden
// Import eingelesen, danach steht als Herkunft das Dataset da und nicht der
// Zwischenordner, den es dann nicht mehr gibt.

export type StudioKind =
  | 'image_boxes' | 'image_classify' | 'video_classify' | 'text_classify' | 'text_pairs'
  | 'audio_classify' | 'audio_transcript';

/** Projektart aus modality + task. */
export function studioKindOf(p: { modality: string; task: string }): StudioKind | null {
  const art: Record<string, StudioKind> = {
    'image/bbox': 'image_boxes', 'image/classify': 'image_classify', 'video/classify': 'video_classify',
    'text/classification': 'text_classify', 'text/pairs': 'text_pairs',
    'audio/classification': 'audio_classify', 'audio/transcript': 'audio_transcript',
  };
  return art[`${p.modality}/${p.task}`] ?? null;
}

export interface StagedResult {
  studio_source: 'folder' | 'text' | 'audio' | 'videos'; path: string; work_dir: string;
  label_classes?: string[]; text_column?: string; label_column?: string; paths?: string[];
  samples: number; per_class?: Record<string, number>; dropped_classes?: Record<string, number>;
  source_columns?: Record<string, unknown>; notes?: string[];
}

export interface DatasetImportReport {
  added: number; duplicates: number; unreadable: number; with_labels?: number; classes_added?: string[];
}

export interface DatasetImportRequest {
  datasetId: string;
  datasetName?: string;
  classMap?: Record<string, string | null>;
  keepUnmapped?: boolean;
  maxSamples?: number;
  perClass?: boolean;
  keepEmpty?: boolean;
  textColumn?: string;
  labelColumn?: string;
}

export type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

export async function importDatasetIntoProject(
  call: Invoke, project: { id: string; modality: string; task: string }, req: DatasetImportRequest,
): Promise<DatasetImportReport & { staged: StagedResult }> {
  const kind = studioKindOf(project);
  if (!kind) throw new Error(`Unsupported project type ${project.modality}/${project.task}.`);
  const projectId = project.id;
  const staged = await call<StagedResult>('stage_dataset', {
    datasetId: req.datasetId,
    request: {
      project_kind: kind, class_map: req.classMap ?? {}, keep_unmapped: req.keepUnmapped !== false,
      max_samples: req.maxSamples, per_class: req.perClass === true, keep_empty: req.keepEmpty === true,
      text_column: req.textColumn, label_column: req.labelColumn,
    },
  });
  try {
    let report: DatasetImportReport;
    switch (staged.studio_source) {
      case 'folder':
        report = await call('studio_import_folder', { projectId, sourcePath: staged.path, labelClasses: staged.label_classes ?? null, ignoreLabels: false });
        break;
      case 'text':
        report = await call('studio_import_text', { projectId, sourcePath: staged.path, textColumn: staged.text_column ?? null, labelColumn: staged.label_column ?? null, ignoreLabels: false });
        break;
      case 'audio':
        report = await call('studio_import_audio', { projectId, sourcePath: staged.path, ignoreLabels: false });
        break;
      case 'videos':
        // Der Ordner statt der einzelnen Dateien: nur so traegt der Unterordner die Klasse.
        report = await call('studio_import_videos', { projectId, paths: [staged.path], clipSeconds: null, ignoreLabels: false });
        break;
    }
    await call('studio_rewrite_origin', {
      projectId, prefix: staged.work_dir, datasetName: req.datasetName ?? req.datasetId,
    }).catch(() => undefined);
    return { ...report, staged };
  } finally {
    await call('discard_staged', { workDir: staged.work_dir }).catch(() => undefined);
  }
}
