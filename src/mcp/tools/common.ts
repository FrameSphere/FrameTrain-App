// Werkzeug-Helfer, die mehrere Domaenen brauchen.

import { startJob, waitForJob, type JobControl } from '../jobs';
import { fail, ok } from '../util';
import type { CallToolResult } from '../types';

/** Wartezeit, nach der ein Werkzeug statt des Ergebnisses eine job_id liefert. */
export const INLINE_WAIT_MS = 20_000;

/**
 * Startet `work` als Hintergrund-Auftrag. Ist er nach `waitMs` fertig, kommt
 * das Ergebnis direkt zurueck; sonst die job_id fuer `system_job_status`.
 * waitMs = 0: sofort die job_id (Downloads).
 */
export async function runAsJob<T>(
  kind: string,
  label: string,
  work: (ctl: JobControl) => Promise<T>,
  opts: { waitMs?: number; describe: (result: T) => string; shape?: (result: T) => unknown },
): Promise<CallToolResult> {
  const job = startJob(kind, label, work);
  const waitMs = opts.waitMs ?? INLINE_WAIT_MS;
  const now = waitMs > 0 ? await waitForJob(job.id, waitMs) : job;
  if (now?.state === 'done') {
    const result = now.result as T;
    return ok(opts.describe(result), { job_id: job.id, state: 'done', result: opts.shape ? opts.shape(result) : result });
  }
  if (now?.state === 'error') {
    return fail(`${label} failed: ${now.error}`, { job_id: job.id, state: 'error', error: now.error });
  }
  return ok(
    `${label} started in the background (job_id ${job.id}). This is not an error. Call system_job_status with this job_id and wait_seconds 45, repeatedly, until state is done or error; downloads can take several minutes.${job.cancellable ? " If it is too large or too slow, stop it with system_job_cancel." : ""}`,
    { job_id: job.id, state: 'running', cancellable: job.cancellable === true },
  );
}

/** Teilt der offenen Oberflaeche mit, dass ein Werkzeug Daten geaendert hat. */
export function announceChange(domain: string, detail: Record<string, unknown> = {}) {
  try {
    window.dispatchEvent(new CustomEvent('ft-mcp-changed', { detail: { domain, ...detail } }));
  } catch { /* kein window (Tests ohne DOM) */ }
}
