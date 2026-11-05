// Je Domaene mindestens ein Werkzeug mit gemocktem invoke: richtige
// Commands, richtige (camelCase-)Argumente, sinnvolles Ergebnis.

import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke }));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));
vi.mock('@tauri-apps/api/app', () => ({ getVersion: async () => '9.9.9' }));

import { registry } from '../registry';
import { getJob, resetJobs, waitForJob } from '../jobs';
import { recordEvent, resetRuns } from '../runLog';
import type { CallToolResult, McpCallContext } from '../types';
import { attachSynapseBuilder, autosaveKey, type SynapseSnapshot } from '../../components/synapse/synapseGraphStore';
import { attachDevTrainPanel, draftKey } from '../../utils/devScriptStore';

const ctx: McpCallContext = { callId: 't', client: { name: 'Test' }, source: 'local', userId: 'u1' };

async function run(name: string, args: Record<string, unknown> = {}): Promise<CallToolResult> {
  const tool = registry.get(name);
  if (!tool) throw new Error(`no tool ${name}`);
  return tool.run(args, ctx);
}

type Handler = (cmd: string, args: Record<string, unknown>) => unknown;
function backend(handlers: Record<string, unknown | Handler>) {
  mockInvoke.mockImplementation(async (cmd: string, args: Record<string, unknown> = {}) => {
    if (!(cmd in handlers)) throw new Error(`unexpected command ${cmd}`);
    const h = handlers[cmd];
    return typeof h === 'function' ? (h as Handler)(cmd, args) : h;
  });
}
const argsOf = (cmd: string) => mockInvoke.mock.calls.filter((c) => c[0] === cmd).map((c) => c[1]);

const MODEL = { id: 'm1', name: 'distilbert', source: 'hf', local_path: '/data/models/m1', size_bytes: 268_000_000, file_count: 5, created_at: '2026-01-01' };
const DATASET = { id: 'd1', name: 'imdb', model_id: 'm1', source: 'hf', storage_path: '/data/ds/d1', size_bytes: 1000, file_count: 3, created_at: '2026-01-02', status: 'ready' };

beforeEach(() => {
  mockInvoke.mockReset();
  mockListen.mockReset();
  mockListen.mockResolvedValue(() => {});
  localStorage.clear();
  resetJobs();
  resetRuns();
});

describe('system', () => {
  it('system_info sammelt Version, RAM und ML-Stack', async () => {
    backend({ get_system_ram_gb: 32, get_python_info: { version: '3.12' }, check_training_requirements: { ready: true, mps_available: true } });
    const r = await run('system_info');
    expect(r.content[0].text).toContain('9.9.9');
    expect(r.content[0].text).toContain('Apple MPS GPU');
    expect(r.structuredContent).toMatchObject({ system_ram_gb: 32, signed_in: true });
  });

  it('system_job_status meldet unbekannte Jobs als Fehler', async () => {
    expect((await run('system_job_status', { job_id: 'x' })).isError).toBe(true);
  });
});

describe('models', () => {
  it('models_list kuerzt auf die wichtigsten Felder', async () => {
    backend({ list_models: [MODEL] });
    const r = await run('models_list');
    expect(r.structuredContent).toMatchObject({ count: 1, items: [{ id: 'm1', name: 'distilbert' }] });
  });

  it('models_download_hf gibt sofort eine job_id, das Ergebnis kommt ueber system_job_status', async () => {
    let finish!: (v: unknown) => void;
    backend({ download_huggingface_model: () => new Promise((r) => { finish = r; }) });
    const r = await run('models_download_hf', { repo_id: 'org/model' });
    const jobId = (r.structuredContent as { job_id: string }).job_id;
    expect(r.structuredContent).toMatchObject({ state: 'running' });
    expect(argsOf('download_huggingface_model')[0]).toEqual({ repoId: 'org/model', modelName: 'org/model' });
    finish({ ...MODEL, id: 'm2', name: 'org/model' });
    await waitForJob(jobId, 100);
    expect(getJob(jobId)?.state).toBe('done');
    const status = await run('system_job_status', { job_id: jobId });
    expect(status.content[0].text).toContain('finished');
  });
});

describe('datasets', () => {
  it('datasets_preview liest nur Dateien des Datensatzes', async () => {
    backend({
      get_dataset_files: [{ name: 'train.csv', path: '/data/ds/d1/train/train.csv', size: 10, split: 'train' }],
      read_dataset_file: 'text,label\nhallo,1',
    });
    const r = await run('datasets_preview', { dataset_id: 'd1' });
    expect(r.content[0].text).toContain('hallo,1');
    expect(argsOf('read_dataset_file')[0]).toEqual({ filePath: '/data/ds/d1/train/train.csv' });
    await expect(run('datasets_preview', { dataset_id: 'd1', file_path: '/etc/passwd' })).rejects.toThrow(/not part of this dataset/);
  });

  it('datasets_split prueft die Summe und nimmt die Modell-ID aus dem Datensatz', async () => {
    backend({ list_all_datasets: [DATASET], split_dataset: DATASET });
    await expect(run('datasets_split', { dataset_id: 'd1', train_ratio: 0.8, val_ratio: 0.3, test_ratio: 0.1 })).rejects.toThrow(/equal 1/);
    await run('datasets_split', { dataset_id: 'd1', train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1 });
    expect(argsOf('split_dataset')[0]).toEqual({ datasetId: 'd1', modelId: 'm1', trainRatio: 0.8, valRatio: 0.1, testRatio: 0.1 });
  });
});

describe('studio', () => {
  it('studio_project_create uebersetzt die Projektart', async () => {
    backend({ studio_create_project: { id: 'sp_1', name: 'Spam' } });
    await run('studio_project_create', { name: 'Spam', kind: 'text_classify', classes: ['spam', 'ham'] });
    expect(argsOf('studio_create_project')[0]).toEqual({ name: 'Spam', modality: 'text', task: 'classification', targetFormat: 'flat_file', classes: ['spam', 'ham'] });
  });

  it('studio_add_texts reicht label/target durch', async () => {
    backend({ studio_add_texts: { added: 2, duplicates: 0, unreadable: 0 } });
    const r = await run('studio_add_texts', { project_id: 'sp_1', items: [{ text: 'a', label: 'spam' }, { text: 'b', target: 'B' }] });
    expect(r.content[0].text).toContain('Added 2');
    expect(argsOf('studio_add_texts')[0]).toEqual({
      projectId: 'sp_1', origin: 'mcp',
      items: [{ text: 'a', label: 'spam', target: null }, { text: 'b', label: null, target: 'B' }],
    });
  });

  it('studio_set_annotation prueft Boxen', async () => {
    backend({ studio_set_annotation: null });
    await run('studio_set_annotation', { project_id: 'p', sample_id: 's', status: 'confirmed', boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }] });
    expect(argsOf('studio_set_annotation')[0]).toMatchObject({ projectId: 'p', sampleId: 's', status: 'confirmed', boxes: [{ cls: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 }] });
    await expect(run('studio_set_annotation', { project_id: 'p', sample_id: 's', status: 'confirmed', boxes: [{ cls: 0 }] })).rejects.toThrow(/numeric/);
  });

  it('studio_fetch_web erlaubt nur http(s) und setzt die Allowlist aus den URLs', async () => {
    backend({ studio_fetch_web: { fetched: 4 } });
    await expect(run('studio_fetch_web', { project_id: 'p', urls: ['file:///etc'] })).rejects.toThrow(/http/);
    const r = await run('studio_fetch_web', { project_id: 'p', urls: ['https://example.org/cats'], label: 'cat' });
    expect(r.content[0].text).toContain('4 item');
    expect((argsOf('studio_fetch_web')[0] as { options: Record<string, unknown> }).options).toMatchObject({ mode: 'urls', allowlist: ['example.org'], label: 'cat' });
  });

  it('studio_export legt einen Datensatz am Modell an', async () => {
    backend({ list_models: [MODEL], studio_export: { dataset: { id: 'd9', name: 'Spam v1' }, report: {}, path: '/x' } });
    const r = await run('studio_export', { project_id: 'p', model_id: 'm1', dataset_name: 'Spam v1' });
    expect(r.content[0].text).toContain('d9');
    expect(argsOf('studio_export')[0]).toEqual({ projectId: 'p', modelId: 'm1', datasetName: 'Spam v1', includeSuggested: false, trainRatio: 0.8, valRatio: 0.1 });
  });
});

describe('training', () => {
  it('training_start loest Namen auf und reicht die Konfiguration durch', async () => {
    backend({
      list_models: [MODEL], list_all_datasets: [DATASET], enable_prevent_sleep: null,
      start_training: { id: 'job1', model_id: 'm1', model_name: 'distilbert', dataset_id: 'd1', dataset_name: 'imdb', status: 'Running' },
    });
    const r = await run('training_start', { model_id: 'm1', dataset_id: 'd1', config: { epochs: 2, learning_rate: 2e-5 } });
    expect(r.content[0].text).toContain('job1');
    expect(argsOf('start_training')[0]).toEqual({ modelId: 'm1', modelName: 'distilbert', datasetId: 'd1', datasetName: 'imdb', config: { epochs: 2, learning_rate: 2e-5 }, versionId: null });
  });

  it('training_status verbindet Job und Event-Mitschnitt', async () => {
    backend({ get_current_training: { id: 'job1', status: 'Running' }, list_active_trainings: [] });
    recordEvent('training-progress', { job_id: 'job1', data: { epoch: 1, train_loss: 0.7 } });
    recordEvent('training-status', { job_id: 'job1', data: { message: 'Epoche 1 fertig' } });
    const r = await run('training_status');
    expect(r.structuredContent).toMatchObject({ live: { status: 'running', last_progress: { epoch: 1, train_loss: 0.7 }, recent_lines: ['Epoche 1 fertig'] } });
  });

  it('training_estimate_resources rechnet mit der Architektur', async () => {
    backend({
      list_models: [MODEL], get_system_ram_gb: 16,
      get_model_ram_info: { has_config: true, hidden_size: 768, num_hidden_layers: 6, num_attention_heads: 12, vocab_size: 30522, param_billion: 0.066 },
    });
    const r = await run('training_estimate_resources', { model_id: 'm1', config: { batch_size: 16, max_seq_length: 128 } });
    const s = r.structuredContent as { verdict: string; estimate_gb: { total: number } };
    expect(['fits', 'tight', 'exceeds']).toContain(s.verdict);
    expect(s.estimate_gb.total).toBeGreaterThan(0);
  });
});

describe('synapse', () => {
  const blueprintGraph = async () => {
    const r = await run('synapse_blueprints', { blueprint_id: 'mlp' });
    return (r.structuredContent as { graph: { nodes: unknown[]; edges: unknown[] } }).graph;
  };

  it('Katalog filtert nach Kategorie und nennt Ports/Parameter', async () => {
    const r = await run('synapse_node_catalog', { category: 'layer', types: ['dense'] });
    const items = (r.structuredContent as { items: { type: string; params: { key: string }[]; inputs: unknown[] }[] }).items;
    expect(items).toHaveLength(1);
    expect(items[0].type).toBe('dense');
    expect(items[0].params.length).toBeGreaterThan(0);
  });

  it('graph_set schreibt bei geschlossener Seite in den Autosave, graph_get liest ihn', async () => {
    const graph = await blueprintGraph();
    const set = await run('synapse_graph_set', graph);
    expect(set.isError).toBeFalsy();
    expect(set.structuredContent).toMatchObject({ written_to: 'autosave' });
    const saved = JSON.parse(localStorage.getItem(autosaveKey('u1'))!);
    expect(saved.nodes).toHaveLength(graph.nodes.length);
    expect(saved.dirty).toBe(true);
    const get = await run('synapse_graph_get');
    const g = get.structuredContent as { graph: { nodes: { id: string; type: string }[] }; builder_open: boolean };
    expect(g.builder_open).toBe(false);
    expect(g.graph.nodes.map((n) => n.type)).toEqual((graph.nodes as { type: string }[]).map((n) => n.type));
  });

  it('graph_set lehnt unbekannte Knotentypen ab', async () => {
    const r = await run('synapse_graph_set', { nodes: [{ id: 'a', type: 'quantum_layer' }], edges: [] });
    expect(r.isError).toBe(true);
    expect(r.content[0].text).toContain('quantum_layer');
  });

  it('graph_edit nutzt den Agent-Executor und aktualisiert die offene Leinwand', async () => {
    let state: SynapseSnapshot = { nodes: [], edges: [], modelId: null, modelName: null, dirty: false };
    const apply = vi.fn((next: Partial<SynapseSnapshot>) => { state = { ...state, ...next }; });
    const detach = attachSynapseBuilder({ userId: 'u1', get: () => state, apply });
    const r = await run('synapse_graph_edit', {
      ops: [
        { op: 'add_node', nodeType: 'dense', id: 'd1' },
        { op: 'add_node', nodeType: 'relu', id: 'r1' },
        { op: 'add_edge', source: 'd1', target: 'r1' },
        { op: 'set_param', nodeId: 'd1', key: 'outputSize', value: 32 },
      ],
    });
    detach();
    expect(r.isError).toBeFalsy();
    expect(r.structuredContent).toMatchObject({ written_to: 'builder' });
    expect(apply).toHaveBeenCalledTimes(1);
    expect(state.nodes.map((n) => n.id)).toEqual(['d1', 'r1']);
    expect(state.edges).toHaveLength(1);
    expect((state.nodes[0].data as { params: { outputSize: number } }).params.outputSize).toBe(32);
    expect(state.dirty).toBe(true);
  });

  it('graph_edit stoppt beim ersten Fehler', async () => {
    const r = await run('synapse_graph_edit', { ops: [{ op: 'add_node', nodeType: 'dense', id: 'x' }, { op: 'remove_node', nodeId: 'gibtsnicht' }, { op: 'add_node', nodeType: 'relu', id: 'y' }] });
    expect(r.isError).toBe(true);
    const res = (r.structuredContent as { results: unknown[] }).results;
    expect(res).toHaveLength(2);
  });

  it('validate und generate_code arbeiten auf dem gespeicherten Graph', async () => {
    await run('synapse_graph_set', await blueprintGraph());
    backend({ synapse_check: { ok: true, params: 1234 } });
    const v = await run('synapse_validate');
    expect(v.structuredContent).toMatchObject({ runtime: { ok: true } });
    expect(argsOf('synapse_check')[0]).toHaveProperty('graph.execution_order');
    const code = await run('synapse_generate_code');
    expect((code.structuredContent as { code: string }).code).toMatch(/nn\.Module/);
  });

  it('synapse_train verweigert ungueltige Graphen', async () => {
    await run('synapse_graph_set', { nodes: [{ id: 'a', type: 'dense' }, { id: 'b', type: 'conv2d' }], edges: [{ source: 'a', target: 'b' }] });
    const r = await run('synapse_train', {});
    expect(mockInvoke.mock.calls.some((c) => c[0] === 'start_training')).toBe(false);
    expect(r.isError).toBe(true);
    expect(r.content[0].text).toMatch(/invalid/);
  });
});

describe('devtrain', () => {
  it('script_set schreibt bei geschlossenem Panel einen Entwurf, script_get liest ihn', async () => {
    const r = await run('devtrain_script_set', { script: 'print(1)', save_name: 'Test' });
    expect(r.structuredContent).toMatchObject({ written_to: 'draft' });
    expect(JSON.parse(localStorage.getItem(draftKey('u1'))!).script).toBe('print(1)');
    const g = await run('devtrain_script_get');
    expect(g.structuredContent).toMatchObject({ script: 'print(1)', source: 'draft' });
    const lib = await run('devtrain_scripts_list');
    expect((lib.structuredContent as { items: { name: string }[] }).items[0].name).toBe('Test');
  });

  it('devtrain_start baut die Umgebungsvariablen wie das Panel', async () => {
    let script = 'print("train")';
    const detach = attachDevTrainPanel({
      userId: 'u1', getScript: () => script, setScript: (s) => { script = s; },
      getSelection: () => ({ modelId: 'm1', datasetIds: ['d1'], versionId: null }),
    });
    backend({
      list_models: [MODEL],
      list_datasets_for_model: [DATASET, { ...DATASET, id: 'd2', name: 'yelp', storage_path: '/data/ds/d2', created_at: '2026-01-03' }],
      start_dev_training: { id: 'dev1' }, enable_prevent_sleep: null,
    });
    const r = await run('devtrain_start', {});
    detach();
    expect(r.content[0].text).toContain('dev1');
    expect(argsOf('start_dev_training')[0]).toMatchObject({
      script: 'print("train")', modelId: 'm1', modelName: 'distilbert', datasetId: 'd1', datasetName: 'imdb', versionId: null,
      refs: { MODEL_PATH: '/data/models/m1', DATASET_PATH: '/data/ds/d1', DATASET_PATH_1: '/data/ds/d1', DATASET_PATH_2: '/data/ds/d2' },
    });
  });

  it('devtrain_status zeigt die Ausgabezeilen', async () => {
    backend({ get_current_training: { id: 'dev1' } });
    recordEvent('dev-training-output', { job_id: 'dev1', line: 'Traceback: kaputt' } as never);
    const r = await run('devtrain_status');
    expect(r.structuredContent).toMatchObject({ live: { recent_lines: ['Traceback: kaputt'] } });
  });
});

describe('lab', () => {
  it('lab_infer braucht Text oder Datei und reicht beides durch', async () => {
    backend({ lab_infer_sample: { text: 'positive', confidence: 0.9 } });
    await expect(run('lab_infer', {})).rejects.toThrow(/text or file_path/);
    const r = await run('lab_infer', { text: 'great movie' });
    expect(r.content[0].text).toBe('positive');
    expect(argsOf('lab_infer_sample')[0]).toEqual({ text: 'great movie', filePath: null, question: null, start: null, end: null });
  });
});

describe('tests', () => {
  it('tests_start sucht die Version ueber alle Modelle', async () => {
    backend({
      list_models_with_versions: [{ id: 'm1', name: 'distilbert' }],
      list_model_versions: [{ id: 'v1', model_id: 'm1', version_name: 'v1 sentiment' }],
      list_all_datasets: [DATASET],
      start_test: { id: 't1' },
    });
    await run('tests_start', { version_id: 'v1', dataset_id: 'd1', max_samples: 50 });
    expect(argsOf('start_test')[0]).toEqual({
      modelId: 'm1', modelName: 'distilbert', versionId: 'v1', versionName: 'v1 sentiment', datasetId: 'd1', datasetName: 'imdb',
      batchSize: null, maxSamples: 50, taskType: null, pluginConfig: null,
    });
  });

  it('tests_hard_examples_export nimmt nur falsche Vorhersagen', async () => {
    backend({
      get_test_results_for_version: [{ total_samples: 3, predictions: [{ is_correct: true }, { is_correct: false, sample_id: 2 }, { is_correct: false, sample_id: 3 }] }],
      export_hard_examples: '/out/hard.jsonl',
    });
    const r = await run('tests_hard_examples_export', { version_id: 'v1' });
    expect(r.structuredContent).toEqual({ path: '/out/hard.jsonl', count: 2 });
    expect((argsOf('export_hard_examples')[0] as { predictions: unknown[] }).predictions).toHaveLength(2);
  });
});

describe('analysis', () => {
  it('analysis_runs_compare findet den Lauf mit dem kleinsten Val-Loss', async () => {
    backend({
      get_training_metrics: (_c: string, a: Record<string, unknown>) => ({ final_train_loss: 0.3, final_val_loss: a.versionId === 'v2' ? 0.2 : 0.5, total_epochs: 3 }),
      get_training_full_data: null,
    });
    const r = await run('analysis_runs_compare', { version_ids: ['v1', 'v2'] });
    expect(r.structuredContent).toMatchObject({ best_version_id: 'v2' });
  });
});

describe('hosting', () => {
  it('hosting_infer baut die Anfrage wie die Hosting-Seite', async () => {
    backend({ hosting_infer: { text: 'Hallo!' } });
    const r = await run('hosting_infer', { id: 'h1', text: 'Hi' });
    expect(r.content[0].text).toBe('Hallo!');
    const a = argsOf('hosting_infer')[0] as { id: string; requestId: string; input: Record<string, unknown> };
    expect(a.id).toBe('h1');
    expect(a.requestId).toMatch(/^mcp_/);
    expect(a.input).toMatchObject({ text: 'Hi', file_path: null, stream: false });
  });
});
