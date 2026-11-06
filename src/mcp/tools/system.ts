// Domaene system: Zustand der App, Hardware, Hintergrund-Auftraege.

import { getJob, listJobs, waitForJob } from '../jobs';
import { call, defineTool, errorText, fail, ok, optNum, reqStr, S } from '../util';
import type { McpTool } from '../types';
import { MCP_DOMAINS } from '../types';

async function appVersion(): Promise<string | null> {
  try {
    const { getVersion } = await import('@tauri-apps/api/app');
    return await getVersion();
  } catch {
    return null;
  }
}

async function safe<T>(fn: () => Promise<T>): Promise<T | { error: string }> {
  try { return await fn(); } catch (e) { return { error: errorText(e) }; }
}

/** Python-Pruefungen starten einen Interpreter (bis ~13 s beim ersten Mal).
 *  Ergebnis 10 min merken; system_info wartet hoechstens INFO_WAIT_MS darauf. */
const PROBE_TTL_MS = 10 * 60 * 1000;
export const INFO_WAIT_MS = 4000;
const probes = new Map<string, { at: number; value: Promise<unknown> }>();

export function cachedProbe(command: string, now = Date.now()): Promise<unknown> {
  const hit = probes.get(command);
  if (hit && now - hit.at < PROBE_TTL_MS) return hit.value;
  const value = safe(() => call<unknown>(command));
  probes.set(command, { at: now, value });
  return value;
}

export function resetProbeCache(): void { probes.clear(); }

const PENDING = Symbol('pending');
function withinWait<T>(p: Promise<T>, ms: number): Promise<T | typeof PENDING> {
  return Promise.race([p, new Promise<typeof PENDING>((r) => setTimeout(() => r(PENDING), ms))]);
}

export const systemTools: McpTool[] = [
  defineTool({
    name: 'system_ping',
    title: 'Ping',
    description: 'Checks that the FrameTrain desktop app is running and reachable. Returns the app version and whether a user is signed in. Useful for checking the connection.',
    requiresUser: false,
    async run(_args, ctx) {
      const version = await appVersion();
      const data = { app: 'FrameTrain', version, signed_in: !!ctx.userId, client: ctx.client.name };
      return ok(`FrameTrain ${version ?? ''} is running${ctx.userId ? '' : ' (no user signed in)'}.`, data);
    },
  }),
  defineTool({
    name: 'system_info',
    title: 'System info',
    description: 'Returns app version, system RAM, Python interpreter and the ML stack status (PyTorch, CUDA/MPS GPU availability, transformers, peft). Shows what the machine can train.',
    requiresUser: false,
    async run(_args, ctx) {
      const [version, ram, pythonRaw, requirementsRaw] = await Promise.all([
        appVersion(),
        safe(() => call<number>('get_system_ram_gb')),
        withinWait(cachedProbe('get_python_info'), INFO_WAIT_MS),
        withinWait(cachedProbe('check_training_requirements'), INFO_WAIT_MS),
      ]);
      const stillChecking = pythonRaw === PENDING || requirementsRaw === PENDING;
      const python = pythonRaw === PENDING ? { status: 'checking' } : pythonRaw;
      const requirements = requirementsRaw === PENDING ? { status: 'checking' } : requirementsRaw;
      const data = { version, signed_in: !!ctx.userId, system_ram_gb: ram, python, requirements };
      const req = requirements as Record<string, unknown>;
      const gpu = stillChecking
        ? 'still being checked'
        : req && !('error' in req)
          ? (req.cuda_available ? 'CUDA GPU' : req.mps_available ? 'Apple MPS GPU' : 'CPU only')
          : 'unknown';
      const hint = stillChecking ? ' The Python/GPU check is still running; call system_info again in ~10 s for the full ML stack status.' : '';
      return ok(`FrameTrain ${version ?? '?'}, RAM ${typeof ram === 'number' ? ram.toFixed(1) + ' GB' : '?'}, compute: ${gpu}.${hint}`, data);
    },
  }),
  defineTool({
    name: 'system_capabilities',
    title: 'Capabilities',
    description: 'Describes what FrameTrain can do through these tools (one entry per domain) and lists the installed task plugins (model families/tasks the app can train). Gives an overview of the available workflows.',
    requiresUser: false,
    async run() {
      const plugins = await safe(() => call<{ id: string; name: string; category: string; description: string }[]>('get_available_plugins'));
      const domains: Record<string, string> = {
        models: 'Local model library: list, inspect, import, download from Hugging Face, rename, delete.',
        datasets: 'Datasets attached to models: list, inspect files, preview, search/download from Hugging Face, import, split, delete.',
        studio: 'Dataset Builder: create labeling projects (image boxes/classes, text classes/pairs, audio, video), add or import samples, fetch from the web, annotate, get model suggestions, check quality, export as a dataset.',
        training: 'Fine-tune library models on datasets with presets or custom hyperparameters; check requirements, estimate RAM, start/stop, status, history.',
        synapse: 'Synapse Builder: design own neural networks as a node graph (read/write/edit graph, validate shapes, generate PyTorch code, save as model, train).',
        devtrain: 'Dev Train: run a custom Python training script against a model and datasets; read/write the script, start/stop, logs, dev tests.',
        lab: 'Laboratory: load a trained version into a model server and run single inferences.',
        tests: 'Evaluate a trained version on a dataset, single-input tests, results, hard-example export.',
        analysis: 'Training metrics, logs and comparisons of trained versions.',
        hosting: 'Local model hosting: load/unload versions and run inference against hosted models.',
        system: 'App status, hardware, background jobs started by other tools.',
      };
      const plist = Array.isArray(plugins) ? plugins.map((p) => ({ id: p.id, name: p.name, category: p.category, description: p.description })) : plugins;
      return ok(`Domains: ${MCP_DOMAINS.join(', ')}.`, { domains, plugins: plist });
    },
  }),
  defineTool({
    name: 'system_job_status',
    title: 'Job status',
    description: 'Returns the state of a background job (download, export, web fetch, ...) that another tool started and returned as job_id. Optionally waits up to wait_seconds (max 45) for it to finish. Training runs have their own training_status / devtrain_status tools.',
    properties: {
      job_id: S.str('The job_id returned by the tool that started the job.'),
      wait_seconds: S.num('Wait up to this many seconds for completion (0-45). Default 0.', { minimum: 0, maximum: 45 }),
    },
    required: ['job_id'],
    requiresUser: false,
    async run(args) {
      const id = reqStr(args, 'job_id');
      const wait = Math.min(45, Math.max(0, optNum(args, 'wait_seconds', 0)!));
      const job = wait > 0 ? await waitForJob(id, wait * 1000) : getJob(id);
      if (!job) return fail(`Unknown job_id "${id}". Jobs are kept in memory until the app restarts.`);
      const text = job.state === 'running' ? `Job ${id} (${job.label}) is still running.`
        : job.state === 'done' ? `Job ${id} (${job.label}) finished.`
          : `Job ${id} (${job.label}) failed: ${job.error}`;
      return ok(text, { ...job });
    },
  }),
  defineTool({
    name: 'system_jobs_list',
    title: 'Jobs',
    description: 'Lists background jobs started by tools in this app session (newest first) with their state.',
    requiresUser: false,
    async run() {
      const jobs = listJobs().map((j) => ({ id: j.id, kind: j.kind, label: j.label, state: j.state, startedAt: j.startedAt, finishedAt: j.finishedAt ?? null, error: j.error ?? null }));
      return ok(`${jobs.length} job(s).`, jobs);
    },
  }),
];
