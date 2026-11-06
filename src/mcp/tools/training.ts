// Domaene training: Feintuning von Bibliotheksmodellen.

import { estimateTrainingRam, ramVerdict, type ModelArch } from '../../ai/resourceEstimate';
import { getRun, summarizeRun } from '../runLog';
import { call, defineTool, ok, optNum, optObj, optStr, reqStr, S } from '../util';
import type { JsonSchema, McpTool } from '../types';
import { findDataset, findModel } from './lookup';

interface TrainingJob {
  id: string;
  model_id: string;
  model_name: string;
  dataset_id: string;
  dataset_name: string;
  status: unknown;
  progress?: Record<string, unknown>;
  output_path?: string | null;
  error?: string | null;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  config?: Record<string, unknown>;
}

/** Die wichtigsten Felder von TrainingConfig (Rust); weitere werden durchgereicht. */
export const TRAINING_CONFIG_SCHEMA: JsonSchema = {
  type: 'object',
  description: 'Training hyperparameters (snake_case, like training_presets). Unlisted fields of the FrameTrain TrainingConfig are passed through; omitted fields use defaults.',
  properties: {
    epochs: S.int('Number of epochs.', { minimum: 1 }),
    batch_size: S.int('Batch size.', { minimum: 1 }),
    learning_rate: S.num('Learning rate, e.g. 2e-5.'),
    weight_decay: S.num('Weight decay.'),
    gradient_accumulation_steps: S.int('Gradient accumulation steps.', { minimum: 1 }),
    warmup_ratio: S.num('Warmup ratio 0..1.'),
    optimizer: S.str('e.g. "adamw", "sgd".'),
    scheduler: S.str('e.g. "linear", "cosine".'),
    max_seq_length: S.int('Max tokens per sample (text models).'),
    fp16: S.bool('Half precision (CUDA).'),
    bf16: S.bool('bfloat16 precision.'),
    use_lora: S.bool('Train LoRA adapters instead of all weights.'),
    lora_r: S.int('LoRA rank.'),
    lora_alpha: S.int('LoRA alpha.'),
    load_in_4bit: S.bool('QLoRA 4-bit loading (CUDA only).'),
    gradient_checkpointing: S.bool('Trade compute for memory.'),
    task_type: S.str('Plugin task type, e.g. "seq_classification", "causal_lm", "image_classification", "detection".'),
    plugin_config: S.obj('Plugin-specific parameters (image_size, num_classes, ...).'),
  },
  additionalProperties: true,
};

function jobBrief(j: TrainingJob | null | undefined) {
  if (!j) return null;
  return {
    id: j.id, status: j.status, model_id: j.model_id, model_name: j.model_name,
    dataset_id: j.dataset_id, dataset_name: j.dataset_name, progress: j.progress ?? null,
    output_path: j.output_path ?? null, error: j.error ?? null,
    started_at: j.started_at ?? null, completed_at: j.completed_at ?? null,
  };
}

export const trainingTools: McpTool[] = [
  defineTool({
    name: 'training_presets',
    title: 'Training presets',
    description: 'Returns the built-in training presets (id, name, description, full config). Good starting points for training_start.config.',
    async run() {
      const presets = await call<{ id: string; name: string }[]>('get_training_presets');
      return ok(`${presets.length} preset(s): ${presets.map((p) => p.id).join(', ')}.`, presets);
    },
  }),
  defineTool({
    name: 'training_check_requirements',
    title: 'Check requirements',
    description: 'Checks whether the Python ML stack is ready for training (Python, PyTorch, CUDA/MPS, transformers, peft) and returns versions.',
    async run() {
      const r = await call<Record<string, unknown>>('check_training_requirements');
      return ok(r.ready ? 'Training stack is ready.' : 'Training stack is NOT ready — see details.', r);
    },
  }),
  defineTool({
    name: 'training_estimate_resources',
    title: 'Estimate memory',
    description: 'Estimates peak RAM/VRAM for training a model with a config (weights, gradients, optimizer, activations) and compares it with system RAM. Verdict: fits, tight, exceeds or unknown. Helps choose batch size, LoRA or precision.',
    properties: { model_id: S.str('Model id.'), config: TRAINING_CONFIG_SCHEMA },
    required: ['model_id'],
    async run(args) {
      const model = await findModel(reqStr(args, 'model_id'));
      const cfg = optObj(args, 'config') ?? {};
      const [info, systemRam] = await Promise.all([
        call<Record<string, number | string | boolean>>('get_model_ram_info', { modelId: model.id }).catch(() => null),
        call<number>('get_system_ram_gb').catch(() => null),
      ]);
      const arch: ModelArch | null = info && info.has_config ? {
        hiddenSize: Number(info.hidden_size), numLayers: Number(info.num_hidden_layers),
        numHeads: Number(info.num_attention_heads), vocabSize: Number(info.vocab_size),
        paramBillion: Number(info.param_billion) || undefined,
      } : null;
      const est = estimateTrainingRam({
        batch_size: Number(cfg.batch_size ?? 8),
        max_seq_length: Number(cfg.max_seq_length ?? 512),
        gradient_checkpointing: cfg.gradient_checkpointing === true,
        fp16: cfg.fp16 === true, bf16: cfg.bf16 === true,
        use_lora: cfg.use_lora === true, load_in_4bit: cfg.load_in_4bit === true, load_in_8bit: cfg.load_in_8bit === true,
        optimizer: String(cfg.optimizer ?? 'adamw'),
      }, model.size_bytes / 1024 ** 3, { arch });
      const verdict = ramVerdict(est.total, systemRam);
      return ok(`Estimated peak ${est.total.toFixed(1)} GB, system RAM ${systemRam ? systemRam.toFixed(1) : '?'} GB: ${verdict}.`, {
        estimate_gb: est, system_ram_gb: systemRam, verdict, model_arch: info,
      });
    },
  }),
  defineTool({
    name: 'training_rate_config',
    title: 'Rate configuration',
    description: 'Rates a training config with FrameTrain\'s rule set (score plus warnings and hints for learning rate, batch size, epochs, etc.).',
    properties: { config: TRAINING_CONFIG_SCHEMA },
    required: ['config'],
    async run(args) {
      const rating = await call<Record<string, unknown>>('rate_training_config', { config: optObj(args, 'config') });
      return ok('Config rated.', rating);
    },
  }),
  defineTool({
    name: 'training_start',
    title: 'Start training',
    description: 'Starts fine-tuning a library model (or continuing from one of its versions) on a dataset. Uses the GPU/CPU for a long time; the user confirms every start. Returns the training job id immediately; follow it with training_status. Only one training runs at a time.',
    properties: {
      model_id: S.str('Model id.'),
      dataset_id: S.str('Dataset id (should belong to the model).'),
      config: TRAINING_CONFIG_SCHEMA,
      version_id: S.str('Continue training from this version instead of the base model.'),
    },
    required: ['model_id', 'dataset_id'],
    approval: 'always',
    annotations: { readOnlyHint: false },
    async run(args) {
      const model = await findModel(reqStr(args, 'model_id'));
      const ds = await findDataset(reqStr(args, 'dataset_id'));
      const job = await call<TrainingJob>('start_training', {
        modelId: model.id, modelName: model.name, datasetId: ds.id, datasetName: ds.name,
        config: optObj(args, 'config') ?? {}, versionId: optStr(args, 'version_id') ?? null,
      });
      try { await call('enable_prevent_sleep'); } catch { /* optional */ }
      return ok(`Training started (job ${job.id}): ${model.name} on ${ds.name}. Progress: training_status.`, { job: jobBrief(job) });
    },
  }),
  defineTool({
    name: 'training_status',
    title: 'Training status',
    description: 'Returns the current or a specific training job: status, epoch/step progress, latest losses and recent log lines. Without job_id it reports the currently running training (also Synapse and Dev Train runs).',
    properties: {
      job_id: S.str('Training job id. Default: current training.'),
      log_lines: S.int('Number of recent log lines (0-200). Default 30.', { minimum: 0, maximum: 200 }),
    },
    async run(args) {
      const wanted = optStr(args, 'job_id');
      const lines = optNum(args, 'log_lines', 30)!;
      const current = await call<TrainingJob | null>('get_current_training').catch(() => null);
      const active = await call<Record<string, unknown>[]>('list_active_trainings').catch(() => []);
      let job: TrainingJob | null = current && (!wanted || current.id === wanted) ? current : null;
      if (!job && wanted) {
        const history = await call<TrainingJob[]>('get_training_history').catch(() => []);
        job = history.find((h) => h.id === wanted) ?? null;
      }
      const id = wanted ?? job?.id;
      const run = id ? summarizeRun(getRun(id), lines) : null;
      if (!job && !run) return ok(wanted ? `No training with id ${wanted} found.` : 'No training is running.', { job: null, active });
      const status = run?.status ?? String(job?.status ?? 'unknown');
      return ok(`Training ${id}: ${status}.`, { job: jobBrief(job), live: run, active });
    },
  }),
  defineTool({
    name: 'training_stop',
    title: 'Stop training',
    description: 'Stops the currently running (regular) training. The user confirms. Dev Train runs are stopped with devtrain_stop.',
    approval: 'always',
    async run() {
      await call('stop_training');
      try { await call('disable_prevent_sleep'); } catch { /* optional */ }
      return ok('Stop requested.', { stopped: true });
    },
  }),
  defineTool({
    name: 'training_history',
    title: 'Training history',
    description: 'Lists past training jobs (newest first) with status, model, dataset, final progress and errors.',
    properties: { limit: S.int('Maximum entries (1-100). Default 20.', { minimum: 1, maximum: 100 }) },
    async run(args) {
      const history = await call<TrainingJob[]>('get_training_history');
      const sorted = [...history].sort((a, b) => String(b.created_at ?? '').localeCompare(String(a.created_at ?? '')));
      const items = sorted.slice(0, optNum(args, 'limit', 20)).map(jobBrief);
      return ok(`${items.length} of ${history.length} job(s).`, items);
    },
  }),
];
