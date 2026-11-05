// Mitschnitt der Trainings- und Test-Events fuer die *_status-Werkzeuge.
//
// Die Rust-Seite meldet Fortschritt nur per Event an die Webview. Ein
// KI-Client fragt aber per Werkzeug nach — deshalb haelt die Bridge die
// letzten Zeilen und Messwerte je Job im Speicher.

import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export interface RunRecord {
  jobId: string;
  kind: 'training' | 'devtest';
  status: string;
  lastProgress?: Record<string, unknown>;
  metrics: Record<string, unknown>[];
  lines: string[];
  error?: unknown;
  complete?: unknown;
  updatedAt: string;
}

const MAX_LINES = 200;
const MAX_METRICS = 500;
const runs = new Map<string, RunRecord>();
/** Ersatzschluessel, falls ein Dev-Test-Event ohne job_id kommt. */
export const DEV_TEST_KEY = 'dev-test';
let lastDevTestId: string | null = null;

/** Job-ID des zuletzt gemeldeten Dev-Tests. */
export function latestDevTestId(): string | null {
  return lastDevTestId;
}

function rec(jobId: string, kind: RunRecord['kind'] = 'training'): RunRecord {
  let r = runs.get(jobId);
  if (!r) {
    r = { jobId, kind, status: 'running', metrics: [], lines: [], updatedAt: new Date().toISOString() };
    runs.set(jobId, r);
  }
  r.updatedAt = new Date().toISOString();
  return r;
}

function pushLine(r: RunRecord, line: string) {
  for (const l of line.split('\n')) {
    if (!l.trim()) continue;
    r.lines.push(l);
  }
  if (r.lines.length > MAX_LINES) r.lines.splice(0, r.lines.length - MAX_LINES);
}

type Payload = Record<string, unknown> & { job_id?: string; data?: Record<string, unknown> };

/** Verarbeitet ein Event — exportiert fuer Tests. */
export function recordEvent(event: string, payload: Payload) {
  const jobId = typeof payload?.job_id === 'string' ? payload.job_id : undefined;
  switch (event) {
    case 'training-progress': {
      if (!jobId) return;
      const r = rec(jobId);
      if (payload.data) {
        r.lastProgress = payload.data;
        r.metrics.push(payload.data);
        if (r.metrics.length > MAX_METRICS) r.metrics.splice(0, r.metrics.length - MAX_METRICS);
      }
      return;
    }
    case 'training-status': {
      if (!jobId) return;
      const r = rec(jobId);
      const msg = payload.data?.message ?? payload.data?.status;
      if (typeof msg === 'string') pushLine(r, msg);
      return;
    }
    case 'dev-training-output': {
      if (!jobId) return;
      const r = rec(jobId);
      const line = (payload as Record<string, unknown>).line;
      if (typeof line === 'string') pushLine(r, line);
      return;
    }
    case 'training-error': {
      if (!jobId) return;
      const r = rec(jobId);
      r.status = 'error';
      r.error = payload.data ?? payload;
      return;
    }
    case 'training-complete': {
      if (!jobId) return;
      const r = rec(jobId);
      r.status = 'completed';
      r.complete = payload;
      return;
    }
    case 'training-finished': {
      if (!jobId) return;
      const r = rec(jobId);
      if (r.status === 'running') r.status = payload.success === false ? 'failed' : 'completed';
      return;
    }
    case 'dev-test-output': {
      const key = jobId ?? DEV_TEST_KEY;
      lastDevTestId = key;
      const r = rec(key, 'devtest');
      const line = (payload as Record<string, unknown>).line ?? (payload as Record<string, unknown>).output;
      if (typeof line === 'string') pushLine(r, line);
      return;
    }
    case 'dev-test-complete': {
      const key = jobId ?? DEV_TEST_KEY;
      lastDevTestId = key;
      const r = rec(key, 'devtest');
      const code = (payload as Record<string, unknown>).exit_code;
      r.status = typeof code === 'number' && code !== 0 ? 'failed' : 'completed';
      r.complete = payload;
      if (payload.data?.error) r.error = payload.data.error;
      return;
    }
  }
}

export function getRun(jobId: string): RunRecord | undefined {
  return runs.get(jobId);
}

/** Nur fuer Tests. */
export function resetRuns() {
  runs.clear();
  lastDevTestId = null;
}

/** Kompakte Sicht fuer Werkzeug-Ergebnisse. */
export function summarizeRun(r: RunRecord | undefined, lines = 40) {
  if (!r) return null;
  return {
    status: r.status,
    last_progress: r.lastProgress ?? null,
    metric_points: r.metrics.length,
    recent_lines: r.lines.slice(-lines),
    error: r.error ?? null,
    updated_at: r.updatedAt,
  };
}

const EVENTS = [
  'training-progress', 'training-status', 'training-error', 'training-complete',
  'training-finished', 'dev-training-output', 'dev-test-output', 'dev-test-complete',
];

/** Haengt sich an alle Lauf-Events. Rueckgabe: Abmelden. */
export async function startRunLog(): Promise<() => void> {
  const offs: UnlistenFn[] = [];
  for (const ev of EVENTS) {
    try {
      offs.push(await listen<Payload>(ev, (e) => recordEvent(ev, e.payload)));
    } catch { /* ausserhalb von Tauri (Tests, Vorschau) */ }
  }
  return () => offs.forEach((off) => off());
}
