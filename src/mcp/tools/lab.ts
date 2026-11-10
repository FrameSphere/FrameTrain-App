// Domaene lab: Modell-Server im Labor, Einzel-Inferenz.

import { call, defineTool, ok, optNum, optStr, reqStr, S, ToolArgError } from '../util';
import type { McpTool } from '../types';
import { runAsJob } from './common';

export const labTools: McpTool[] = [
  defineTool({
    name: 'lab_server_start',
    title: 'Start lab server',
    description: 'Loads a trained model version into the Laboratory model server (one at a time) so lab_infer can run predictions. Loading can take a while; returns a job_id if it takes longer than ~20 s.',
    properties: { version_id: S.str('Trained model version id.') },
    required: ['version_id'],
    approval: 'once',
    annotations: { readOnlyHint: false },
    async run(args) {
      const versionId = reqStr(args, 'version_id');
      return runAsJob('lab_server', 'Lab server start', async () => {
        await call('lab_start_model_server', { versionId });
        // Der Start kehrt zurueck, waehrend das Modell noch laedt. Vorher meldete der
        // Auftrag "done" mit status "loading", und lab_infer scheiterte direkt danach.
        const deadline = Date.now() + 180_000;
        let status = await call<Record<string, unknown>>('lab_get_server_status');
        while (status.status === 'loading' && Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 1000));
          status = await call<Record<string, unknown>>('lab_get_server_status');
        }
        if (status.status !== 'ready') throw new Error(`Model server did not become ready (status: ${String(status.status ?? 'unknown')}${status.error ? `, ${String(status.error)}` : ''}).`);
        return status;
      }, { describe: () => `Model server ready for version ${versionId}.` });
    },
  }),
  defineTool({
    name: 'lab_server_status',
    title: 'Lab server status',
    description: 'Returns whether a Laboratory model server is running, which version it serves and its task/modality.',
    async run() {
      const status = await call<Record<string, unknown>>('lab_get_server_status');
      // Das Feld heisst status ("ready", "loading", ...); status.running gab es nie,
      // die Meldung lautete deshalb immer "No model server running".
      const st = String(status.status ?? '');
      const text = st === 'ready' ? `Model server is ready (version ${String(status.version_id ?? '?')}).`
        : st === 'loading' ? 'Model server is still loading.'
          : st ? `Model server status: ${st}.` : 'No model server running.';
      return ok(text, status);
    },
  }),
  defineTool({
    name: 'lab_server_stop',
    title: 'Stop lab server',
    description: 'Stops the Laboratory model server and frees its memory.',
    approval: 'once',
    async run() {
      await call('lab_stop_model_server');
      return ok('Model server stopped.', { stopped: true });
    },
  }),
  defineTool({
    name: 'lab_infer',
    title: 'Single inference',
    description: 'Runs one prediction on the loaded Laboratory model: text input for text models, or file_path (image/audio/video inside FrameTrain data folders) for media models; question for vision-language models; start/end seconds for video clips. Requires a model server started with lab_server_start.',
    properties: {
      text: S.str('Text input (text models, LLM prompt).'),
      file_path: S.str('Absolute path of an image/audio/video file, e.g. from datasets_files.'),
      question: S.str('Question about the image (VLM).'),
      start: S.num('Clip start in seconds (video).'),
      end: S.num('Clip end in seconds (video).'),
    },
    approval: 'none',
    async run(args) {
      const text = optStr(args, 'text') ?? '';
      const filePath = optStr(args, 'file_path') ?? null;
      if (!text && !filePath) throw new ToolArgError('Pass text or file_path.');
      const res = await call<Record<string, unknown>>('lab_infer_sample', {
        text, filePath, question: optStr(args, 'question') ?? null, start: optNum(args, 'start') ?? null, end: optNum(args, 'end') ?? null,
      });
      return ok(typeof res.text === 'string' ? res.text : 'Prediction done.', res);
    },
  }),
  defineTool({
    name: 'lab_sessions',
    title: 'Lab sessions',
    description: 'Lists saved Laboratory sessions (manual evaluation sessions).',
    async run() {
      const sessions = await call<unknown[]>('lab_get_sessions');
      return ok(`${sessions.length} session(s).`, sessions);
    },
  }),
  defineTool({
    name: 'lab_stats',
    title: 'Lab statistics',
    description: 'Returns aggregate statistics of saved Laboratory sessions (evaluated samples, corrections). Single lab_infer calls outside a session are not counted.',
    async run() {
      const stats = await call<Record<string, unknown>>('lab_get_stats');
      return ok('Laboratory statistics.', stats);
    },
  }),
];
