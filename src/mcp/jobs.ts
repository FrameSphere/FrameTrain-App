// Hintergrund-Auftraege der Werkzeuge.
//
// Vertrag Abschnitt 1: kein Werkzeug-Aufruf blockiert laenger als 60 s.
// Lange Tauri-Commands (Downloads, Export, Web-Abruf) laufen deshalb als
// Promise weiter; das Werkzeug gibt eine job_id zurueck, `system_job_status`
// liefert spaeter Stand und Ergebnis.

import { errorText } from './util';

export type JobState = 'running' | 'done' | 'error' | 'cancelled';

export interface McpJob {
  id: string;
  kind: string;
  label: string;
  state: JobState;
  startedAt: string;
  finishedAt?: string;
  result?: unknown;
  error?: string;
  /** Letzte Fortschrittsmeldung (Downloads), damit die KI sieht, dass es weitergeht. */
  progress?: string;
  /** true, wenn sich der Auftrag mit system_job_cancel stoppen laesst. */
  cancellable?: boolean;
}

/** Wird an die Arbeit uebergeben: eigene id und Abbruch-Funktion anmelden. */
export interface JobControl {
  id: string;
  onCancel: (fn: () => Promise<unknown>) => void;
}

export type CancelOutcome = 'cancelled' | 'unknown' | 'finished' | 'not_cancellable';

const cancelers = new Map<string, () => Promise<unknown>>();
const cancelRequested = new Set<string>();

const MAX_JOBS = 100;
const jobs = new Map<string, McpJob>();
const waiters = new Map<string, Promise<void>>();
let seq = 0;

function newId(kind: string): string {
  seq += 1;
  return `${kind}_${Date.now().toString(36)}_${seq}`;
}

function prune() {
  if (jobs.size <= MAX_JOBS) return;
  const finished = [...jobs.values()].filter((j) => j.state !== 'running');
  for (const j of finished.slice(0, jobs.size - MAX_JOBS)) {
    jobs.delete(j.id);
    waiters.delete(j.id);
  }
}

const PROGRESS_KINDS = new Set(['download', 'import']);
let progressListening = false;

/** Haengt Fortschrittsmeldungen von fetch_dataset an laufende Download-Auftraege. */
function trackFetchProgress() {
  if (progressListening) return;
  progressListening = true;
  import('@tauri-apps/api/event')
    .then(({ listen }) => listen<Record<string, unknown>>('dataset-fetch-progress', (e) => {
      const p = e.payload ?? {};
      const msg = typeof p.message === 'string' ? p.message : typeof p.type === 'string' ? p.type : '';
      if (!msg) return;
      for (const j of jobs.values()) {
        if (j.state === 'running' && PROGRESS_KINDS.has(j.kind)) j.progress = msg.slice(0, 300);
      }
    }))
    .catch(() => { progressListening = false; });
}

/** Startet einen Auftrag; das Promise laeuft unabhaengig vom Werkzeug-Aufruf. */
export function startJob(kind: string, label: string, work: (ctl: JobControl) => Promise<unknown>): McpJob {
  const job: McpJob = { id: newId(kind), kind, label, state: 'running', startedAt: new Date().toISOString() };
  jobs.set(job.id, job);
  const ctl: JobControl = {
    id: job.id,
    onCancel: (fn) => { cancelers.set(job.id, fn); job.cancellable = true; },
  };
  if (PROGRESS_KINDS.has(kind)) trackFetchProgress();
  const done = (async () => {
    try {
      job.result = await work(ctl);
      job.state = 'done';
    } catch (e) {
      if (cancelRequested.has(job.id)) {
        job.state = 'cancelled';
        job.error = 'Cancelled.';
      } else {
        job.error = errorText(e);
        job.state = 'error';
      }
    } finally {
      job.finishedAt = new Date().toISOString();
      job.cancellable = false;
      cancelers.delete(job.id);
      cancelRequested.delete(job.id);
    }
  })();
  waiters.set(job.id, done);
  prune();
  return job;
}

/** Wartet hoechstens `ms` auf das Ende. Rueckgabe: der (evtl. noch laufende) Auftrag. */
export async function waitForJob(id: string, ms: number): Promise<McpJob | undefined> {
  const job = jobs.get(id);
  const waiter = waiters.get(id);
  if (!job || !waiter || job.state !== 'running') return job;
  let timer: ReturnType<typeof setTimeout> | undefined;
  await Promise.race([waiter, new Promise<void>((r) => { timer = setTimeout(r, ms); })]);
  if (timer) clearTimeout(timer);
  return job;
}

/** Stoppt einen laufenden Auftrag, wenn er eine Abbruch-Funktion angemeldet hat. */
export async function cancelJob(id: string): Promise<CancelOutcome> {
  const job = jobs.get(id);
  if (!job) return 'unknown';
  if (job.state !== 'running') return 'finished';
  const fn = cancelers.get(id);
  if (!fn) return 'not_cancellable';
  cancelRequested.add(id);
  await fn();
  await waitForJob(id, 10_000);
  return 'cancelled';
}

export function getJob(id: string): McpJob | undefined {
  return jobs.get(id);
}

export function listJobs(): McpJob[] {
  return [...jobs.values()].reverse();
}

/** Nur fuer Tests. */
export function resetJobs() {
  jobs.clear();
  waiters.clear();
}
