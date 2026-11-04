// Domaene analysis: Metriken, Logs, Vergleiche trainierter Versionen.

import { call, defineTool, ok, optNum, reqStr, S, strArr } from '../util';
import type { McpTool } from '../types';

type Json = Record<string, unknown>;

/** Loss-Verlauf aus den gespeicherten Metriken in Kennzahlen umrechnen. */
export function summarizeMetrics(metrics: Json | null | undefined, full?: Json | null) {
  if (!metrics && !full) return null;
  const m = metrics ?? {};
  const pick = (k: string) => (typeof m[k] === 'number' ? (m[k] as number) : null);
  const src = full ?? m;
  const history = Array.isArray(src.loss_history) ? (src.loss_history as Json[])
    : Array.isArray(src.history) ? (src.history as Json[])
      : Array.isArray(src.metrics) ? (src.metrics as Json[]) : [];
  const train = history.map((h) => h.train_loss).filter((v): v is number => typeof v === 'number');
  const val = history.map((h) => h.val_loss).filter((v): v is number => typeof v === 'number');
  const finalVal = pick('final_val_loss') ?? (val.length ? val[val.length - 1] : null);
  return {
    final_train_loss: pick('final_train_loss') ?? (train.length ? train[train.length - 1] : null),
    final_val_loss: finalVal,
    // Ohne Verlauf ist der letzte Val-Loss die beste Schaetzung.
    best_val_loss: val.length ? Math.min(...val) : pick('best_val_loss') ?? finalVal,
    epochs: pick('total_epochs') ?? pick('epochs'),
    best_epoch: pick('best_epoch'),
    duration_seconds: pick('training_duration_seconds') ?? pick('duration_seconds'),
    points: history.length,
    overfitting_hint: val.length >= 3 && val[val.length - 1] > Math.min(...val) * 1.1,
  };
}

export const analysisTools: McpTool[] = [
  defineTool({
    name: 'analysis_metrics',
    title: 'Trainingsmetriken',
    description: 'Returns the stored training metrics of a version (final/best losses, epochs, steps, duration) plus a short summary with an overfitting hint. include_full adds the full stored training record (loss history, config) which can be large.',
    properties: { version_id: S.str('Trained version id.'), include_full: S.bool('Include the full training record. Default false.') },
    required: ['version_id'],
    async run(args) {
      const versionId = reqStr(args, 'version_id');
      const metrics = await call<Json | null>('get_training_metrics', { versionId });
      const full = await call<Json | null>('get_training_full_data', { versionId }).catch(() => null);
      const summary = summarizeMetrics(metrics, full);
      if (!summary) return ok('No metrics stored for this version.', { summary: null });
      return ok(`Final train loss ${summary.final_train_loss ?? '?'}, best val loss ${summary.best_val_loss ?? '?'}.`, { summary, metrics, ...(args.include_full === true ? { full } : {}) });
    },
  }),
  defineTool({
    name: 'analysis_logs',
    title: 'Trainingslogs',
    description: 'Returns the stored training log entries of a version (newest last), limited to the last N entries.',
    properties: { version_id: S.str('Trained version id.'), limit: S.int('Default 100.', { minimum: 1, maximum: 1000 }) },
    required: ['version_id'],
    async run(args) {
      const logs = await call<unknown[]>('get_training_logs', { versionId: reqStr(args, 'version_id') });
      const limit = optNum(args, 'limit', 100)!;
      return ok(`${logs.length} log entr(ies), returning the last ${Math.min(limit, logs.length)}.`, { total: logs.length, entries: logs.slice(-limit) });
    },
  }),
  defineTool({
    name: 'analysis_version_details',
    title: 'Versionsdetails',
    description: 'Returns details of a trained version: model, parent version, dataset, config, metrics and stored AI analysis report if any.',
    properties: { version_id: S.str('Trained version id.') },
    required: ['version_id'],
    async run(args) {
      const versionId = reqStr(args, 'version_id');
      const details = await call<Json>('get_version_details', { versionId });
      let report: unknown = null;
      try { report = await call('get_ai_analysis_report', { versionId }); } catch { /* keiner */ }
      return ok('Version details.', { details, ai_report: report });
    },
  }),
  defineTool({
    name: 'analysis_runs_compare',
    title: 'Laeufe vergleichen',
    description: 'Compares 2-10 trained versions side by side: final/best losses, epochs, duration and key hyperparameters. Use it to pick the best run or to explain what changed.',
    properties: { version_ids: S.strArr('Version ids to compare.', { minItems: 2, maxItems: 10 }) },
    required: ['version_ids'],
    async run(args) {
      const ids = strArr(args, 'version_ids', true);
      const rows = await Promise.all(ids.map(async (id) => {
        try {
          const m = await call<Json | null>('get_training_metrics', { versionId: id });
          const full = await call<Json | null>('get_training_full_data', { versionId: id }).catch(() => null);
          const cfg = ((full?.config ?? m?.config ?? {}) as Json);
          return {
            version_id: id, ...summarizeMetrics(m, full),
            learning_rate: cfg.learning_rate ?? null, batch_size: cfg.batch_size ?? null, epochs_config: cfg.epochs ?? null,
            use_lora: cfg.use_lora ?? null, optimizer: cfg.optimizer ?? null,
          };
        } catch (e) {
          return { version_id: id, error: String(e instanceof Error ? e.message : e) };
        }
      }));
      const scored = rows.filter((r): r is typeof r & { best_val_loss: number } => typeof (r as Json).best_val_loss === 'number');
      const best = scored.sort((a, b) => a.best_val_loss - b.best_val_loss)[0];
      return ok(best ? `Lowest validation loss: ${best.version_id} (${best.best_val_loss}).` : 'Compared runs (no validation loss available).', { runs: rows, best_version_id: best?.version_id ?? null });
    },
  }),
];
