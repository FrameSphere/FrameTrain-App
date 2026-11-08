import { describe, expect, it } from 'vitest';
import { cancelJob, getJob, startJob, waitForJob } from '../jobs';

describe('Auftraege abbrechen', () => {
  it('ruft die angemeldete Abbruch-Funktion und meldet cancelled', async () => {
    let stop: (e: Error) => void = () => {};
    const job = startJob('download', 'Download x', (ctl) => new Promise((_, reject) => {
      stop = reject;
      ctl.onCancel(async () => { stop(new Error('Download cancelled.')); });
    }));
    expect(job.cancellable).toBe(true);
    expect(await cancelJob(job.id)).toBe('cancelled');
    expect(getJob(job.id)?.state).toBe('cancelled');
    expect(getJob(job.id)?.cancellable).toBe(false);
  });

  it('meldet not_cancellable ohne Abbruch-Funktion und finished nach dem Ende', async () => {
    let done: () => void = () => {};
    const job = startJob('export', 'Export', () => new Promise<void>((r) => { done = r; }));
    expect(await cancelJob(job.id)).toBe('not_cancellable');
    done();
    await waitForJob(job.id, 1000);
    expect(await cancelJob(job.id)).toBe('finished');
    expect(await cancelJob('gibt-es-nicht')).toBe('unknown');
  });

  it('ein normaler Fehler bleibt error', async () => {
    const job = startJob('download', 'D', async () => { throw new Error('kaputt'); });
    await waitForJob(job.id, 1000);
    expect(getJob(job.id)?.state).toBe('error');
  });
});
