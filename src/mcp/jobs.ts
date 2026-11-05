// Hintergrund-Auftraege der Werkzeuge.
//
// Vertrag Abschnitt 1: kein Werkzeug-Aufruf blockiert laenger als 60 s.
// Lange Tauri-Commands (Downloads, Export, Web-Abruf) laufen deshalb als
// Promise weiter; das Werkzeug gibt eine job_id zurueck, `system_job_status`
// liefert spaeter Stand und Ergebnis.

import { errorText } from './util';

export type JobState = 'running' | 'done' | 'error';

export interface McpJob {
  id: string;
  kind: string;
  label: string;
  state: JobState;
  startedAt: string;
  finishedAt?: string;
  result?: unknown;
  error?: string;
}

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

/** Startet einen Auftrag; das Promise laeuft unabhaengig vom Werkzeug-Aufruf. */
export function startJob(kind: string, label: string, work: () => Promise<unknown>): McpJob {
  const job: McpJob = { id: newId(kind), kind, label, state: 'running', startedAt: new Date().toISOString() };
  jobs.set(job.id, job);
  const done = (async () => {
    try {
      job.result = await work();
      job.state = 'done';
    } catch (e) {
      job.error = errorText(e);
      job.state = 'error';
    } finally {
      job.finishedAt = new Date().toISOString();
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
