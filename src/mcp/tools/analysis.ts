// Domaene analysis: Metriken, Logs, Vergleiche trainierter Versionen.

import { call, defineTool, ok, optNum, reqStr, S, strArr } from '../util';
import type { McpTool } from '../types';

type Json = Record<string, unknown>;

/**
 * Wiederholungen desselben Schritts entfernen. Die Engine meldet am Ende zwei
 * Abschlusszeilen mit Start-Lernrate und Durchschnitts-Loss; sie erschienen als
 * Spitze im Verlauf (Benchmark 09.11.). Ein Val-Loss darin wird uebernommen.
 */
export function dedupeSteps(logs: Json[]): Json[] {
  const out: Json[] = [];
  for (const l of logs) {
    const prev = out[out.length - 1];
    if (prev && typeof prev.step === 'number' && prev.step === l.step) {
      if (prev.val_loss == null && l.val_loss != null) out[out.length - 1] = { ...prev, val_loss: l.val_loss };
      continue;
    }
    out.push(l);
  }
  return out;
}

/** Loss-Verlauf aus den gespeicherten Metriken in Kennzahlen umrechnen. */
export function summarizeMetrics(metrics: Json | null | undefined, full?: Json | null) {
  if (!metrics && !full) return null;
  const m = metrics ?? {};
  const pick = (k: string) => (typeof m[k] === 'number' ? (m[k] as number) : null);
  const src = full ?? m;
  // Das gespeicherte Trainingsprotokoll heisst step_logs; vorher wurde nur nach
  // loss_history gesucht, der Verlauf war leer (points: 0) und der "beste"
  // Val-Loss war einfach der letzte.
  const raw = [src.step_logs, src.loss_history, src.history, src.metrics].find(Array.isArray) as Json[] | undefined;
  const history = dedupeSteps(raw ?? []);
  const train = history.map((h) => h.train_loss).filter((v): v is number => typeof v === 'number');
  const val = history.map((h) => h.val_loss).filter((v): v is number => typeof v === 'number');
  const finalVal = pick('final_val_loss') ?? (val.length ? val[val.length - 1] : null);
  const bestVal = val.length ? Math.min(...val) : pick('best_val_loss') ?? finalVal;
  return {
    final_train_loss: pick('final_train_loss') ?? (train.length ? train[train.length - 1] : null),
    final_val_loss: finalVal,
    best_val_loss: bestVal,
    val_losses: val,
    epochs: pick('total_epochs') ?? pick('epochs'),
    best_epoch: pick('best_epoch'),
    duration_seconds: pick('training_duration_seconds') ?? pick('duration_seconds'),
    points: history.length,
    // Steigt der Val-Loss am Ende deutlich ueber sein Minimum, lernt das Modell auswendig.
    overfitting_hint: val.length >= 2 && bestVal !== null && val[val.length - 1] > bestVal * 1.1,
  };
}

export const analysisTools: McpTool[] = [
  defineTool({
    name: 'analysis_metrics',
    title: 'Training metrics',
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
    title: 'Training logs',
    description: 'Returns the stored training log entries of a version (newest last), limited to the last N entries.',
    properties: { version_id: S.str('Trained version id.'), limit: S.int('Default 100.', { minimum: 1, maximum: 1000 }) },
    required: ['version_id'],
    async run(args) {
      const logs = dedupeSteps(await call<Json[]>('get_training_logs', { versionId: reqStr(args, 'version_id') }));
      const limit = optNum(args, 'limit', 100)!;
      return ok(`${logs.length} log entr(ies), returning the last ${Math.min(limit, logs.length)}.`, { total: logs.length, entries: logs.slice(-limit) });
    },
  }),
  defineTool({
    name: 'analysis_version_details',
    title: 'Version details',
    description: 'Returns details of a trained version: model, parent version, dataset, config, metrics and stored AI analysis report if any.',
    properties: { version_id: S.str('Trained version id.') },
    required: ['version_id'],
    async run(args) {
      const versionId = reqStr(args, 'version_id');
      const details = await call<Json>('get_version_details', { versionId });
      let report: unknown = null;
      try { report = await call('get_ai_analysis_report', { versionId }); } catch { /* keiner */ }
      // Dataset, Einstellungen und Kennzahlen standen in der Beschreibung, fehlten aber in der Antwort.
      const metrics = await call<Json | null>('get_training_metrics', { versionId }).catch(() => null);
      const full = await call<Json | null>('get_training_full_data', { versionId }).catch(() => null);
      const cfg = (full?.config ?? null) as Json | null;
      const dataset = cfg?.dataset_path ? { path: cfg.dataset_path, id: String(cfg.dataset_path).split(/[\\/]/).pop() } : null;
      const config = cfg ? Object.fromEntries(Object.entries(cfg).filter(([k]) => !k.startsWith('canvas_') && !k.endsWith('_path') && k !== 'checkpoint_dir')) : null;
      return ok('Version details.', {
        details, dataset, config, metrics: summarizeMetrics(metrics, full),
        test_metrics: full?.classification_metrics ?? full?.detection_metrics ?? null, ai_report: report,
      });
    },
  }),
  defineTool({
    name: 'analysis_runs_compare',
    title: 'Compare runs',
    description: 'Compares 2-10 trained versions side by side: final/best losses, epochs, duration and key hyperparameters. Shows which run performed best and what changed between them.',
    properties: { version_ids: S.strArr('Version ids to compare.', { minItems: 2, maxItems: 10 }) },
    required: ['version_ids'],
    async run(args) {
      const ids = strArr(args, 'version_ids', true);
      const rows = await Promise.all(ids.map(async (id) => {
        try {
          const m = await call<Json | null>('get_training_metrics', { versionId: id });
          const full = await call<Json | null>('get_training_full_data', { versionId: id }).catch(() => null);
          const cfg = ((full?.config ?? m?.config ?? {}) as Json);
          const summary = summarizeMetrics(m, full);
          return {
            version_id: id, ...summary, val_losses: undefined,
            dataset: typeof cfg.dataset_path === 'string' ? cfg.dataset_path.split(/[\\/]/).pop() : null,
            learning_rate: cfg.learning_rate ?? null, batch_size: cfg.batch_size ?? null, epochs_config: cfg.epochs ?? null,
            use_lora: cfg.use_lora ?? null, optimizer: cfg.optimizer ?? null,
          };
        } catch (e) {
          return { version_id: id, error: String(e instanceof Error ? e.message : e) };
        }
      }));
      const scored = rows.filter((r): r is typeof r & { best_val_loss: number } => typeof (r as Json).best_val_loss === 'number');
      // Val-Losses sind nur auf denselben Validierungsdaten vergleichbar. Vorher
      // wurde quer ueber Modelle und Datasets ein "Bester" gekuert.
      const datasets = new Set(scored.map((r) => (r as Json).dataset ?? null));
      if (datasets.size > 1) {
        return ok('Runs use different datasets, so their validation losses are not comparable; compare test results on a shared test dataset instead.', { runs: rows, best_version_id: null });
      }
      const best = scored.sort((a, b) => a.best_val_loss - b.best_val_loss)[0];
      return ok(best ? `Lowest validation loss: ${best.version_id} (${best.best_val_loss}).` : 'Compared runs (no validation loss available).', { runs: rows, best_version_id: best?.version_id ?? null });
    },
  }),
];
