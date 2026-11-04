// Domaene hosting: Modelle lokal bereitstellen und abfragen.
// Bewusst ohne Einstellungen/Token: der API-Token des Hostings bleibt in der App.

import { call, defineTool, ok, optBool, optStr, reqStr, S, ToolArgError } from '../util';
import type { McpTool } from '../types';
import { findVersion } from './lookup';

export const hostingTools: McpTool[] = [
  defineTool({
    name: 'hosting_list',
    title: 'Gehostete Modelle',
    description: 'Lists models registered for local hosting with status (loaded, loading, stopped, error), API name, modality, classes and whether busy.',
    async run() {
      const list = await call<{ id: string; name: string; status: unknown }[]>('hosting_list');
      return ok(`${list.length} hosted model(s).`, list);
    },
  }),
  defineTool({
    name: 'hosting_load',
    title: 'Modell bereitstellen',
    description: 'Loads a trained version into local hosting so it can answer hosting_infer requests (and the local hosting API). Uses RAM/VRAM while loaded.',
    properties: {
      version_id: S.str('Trained version id.'),
      name: S.str('Display name. Default: model + version name.'),
      autoload: S.bool('Load automatically when FrameTrain starts. Default false.'),
      make_default: S.bool('Make it the default hosted model. Default false.'),
    },
    required: ['version_id'],
    approval: 'once',
    annotations: { readOnlyHint: false },
    async run(args) {
      const { version, modelName } = await findVersion(reqStr(args, 'version_id'));
      const name = optStr(args, 'name') ?? `${modelName} ${version.version_name}`;
      await call('hosting_start', {
        versionId: version.id, modelId: version.model_id, name,
        autoload: optBool(args, 'autoload'), makeDefault: optBool(args, 'make_default'),
      });
      return ok(`Loading ${name}. Check hosting_list for status "loaded".`, { version_id: version.id, name });
    },
  }),
  defineTool({
    name: 'hosting_unload',
    title: 'Modell entladen',
    description: 'Unloads a hosted model (frees memory). It stays registered and can be loaded again.',
    properties: { id: S.str('Hosted model id from hosting_list.') },
    required: ['id'],
    approval: 'once',
    async run(args) {
      await call('hosting_stop', { id: reqStr(args, 'id') });
      return ok('Model unloaded.', { id: args.id });
    },
  }),
  defineTool({
    name: 'hosting_infer',
    title: 'Gehostetes Modell abfragen',
    description: 'Sends one request to a loaded hosted model: text (or a chat history in messages for LLMs), or file_path for image/audio/video models, optional question for vision-language models.',
    properties: {
      id: S.str('Hosted model id.'),
      text: S.str('Text input / prompt.'),
      messages: S.arr('Chat history for LLMs.', S.obj('Message', { role: S.enumOf('Role.', ['system', 'user', 'assistant']), content: S.str('Content.') }, { required: ['role', 'content'] })),
      file_path: S.str('Absolute path of an input file.'),
      question: S.str('Question about the image (VLM).'),
    },
    required: ['id'],
    async run(args) {
      const text = optStr(args, 'text') ?? '';
      const filePath = optStr(args, 'file_path') ?? null;
      const messages = Array.isArray(args.messages) ? args.messages : null;
      if (!text && !filePath && !messages) throw new ToolArgError('Pass text, messages or file_path.');
      const res = await call<Record<string, unknown>>('hosting_infer', {
        id: reqStr(args, 'id'), requestId: `mcp_${Date.now().toString(36)}`,
        input: { text, file_path: filePath, question: optStr(args, 'question') ?? null, start: null, end: null, messages, stream: false },
      });
      const out = typeof res.text === 'string' ? res.text : typeof res.output === 'string' ? res.output : 'Inference done.';
      return ok(out, res);
    },
  }),
];
