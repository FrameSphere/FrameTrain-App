// Domaene synapse: eigener Modellbau als Knoten-Graph.
//
// Lesen/Schreiben geht ueber synapseGraphStore: bei offener Seite direkt in
// die Leinwand, sonst in den Autosave, den der Builder beim Oeffnen laedt.
// Graph-Edits nutzen denselben Executor wie der eingebaute Synapse-Agent.

import type { Edge, Node } from '@xyflow/react';
import { NODE_DEFINITIONS, type NodeCategory } from '../../components/synapse/nodeTypes';
import blueprints from '../../components/synapse/blueprints.json';
import { createToolExecutor } from '../../components/synapse/ai/synapseAgentTools';
import { formatValidationError, validateFullGraph } from '../../components/synapse/graph-shape-validation';
import { buildCanvasGraphIR, LOADER_NODE_TYPES } from '../../components/synapse/graphIR';
import { extractCanvasModelClass, generateTrainingScript } from '../../components/synapse/codeGenerator';
import { generateModelConfigFromGraph } from '../../components/synapse/graphToModel';
import { exportCanvasNetworkToModelLibrary, updateCanvasNetworkModel } from '../../components/synapse/canvasModelBridge';
import { autoLayoutNodes, hasNodeOverlap } from '../../components/synapse/synapseLayout';
import {
  fromCompactGraph, getAttachedBuilder, readSynapseGraph, toCompactGraph, writeSynapseGraph,
  type CompactEdge, type CompactNode,
} from '../../components/synapse/synapseGraphStore';
import { getRun, summarizeRun } from '../runLog';
import { call, clip, defineTool, errorText, fail, ok, optBool, optNum, optStr, reqStr, S, ToolArgError, ToolFailure } from '../util';
import type { JsonSchema, McpCallContext, McpTool, ToolArgs } from '../types';
import { announceChange } from './common';
import { findDataset } from './lookup';

const DEFAULT_CFG = { epochs: 10, batchSize: 32, learningRate: 0.001, gpu: 'cpu', precision: 'fp32' as const, gradAccum: 1 };

function userOf(ctx: McpCallContext): string {
  if (!ctx.userId) throw new ToolFailure('No user is signed in to FrameTrain.');
  return ctx.userId;
}

function origin(ctx: McpCallContext, what: string): string {
  return `${ctx.client.name}: ${what}`;
}

function validationReport(nodes: Node[], edges: Edge[]) {
  const res = validateFullGraph(nodes, edges, NODE_DEFINITIONS);
  const errors = res.errors.filter((e) => e.severity === 'error');
  const warnings = res.errors.filter((e) => e.severity !== 'error');
  return {
    valid: res.valid,
    errors: errors.map((e) => ({ node: e.sourceNodeId, target: e.targetNodeId ?? null, message: e.message, suggestion: e.suggestion ?? null, text: formatValidationError(e) })),
    warnings: warnings.map((e) => ({ node: e.sourceNodeId, message: e.message, suggestion: e.suggestion ?? null })),
  };
}

/** Ordnet an, wenn Knoten keine Position hatten oder sich ueberlappen. */
function layoutIfNeeded(nodes: Node[], edges: Edge[], hadPositions: boolean): Node[] {
  if (hadPositions && !hasNodeOverlap(nodes)) return nodes;
  try { return autoLayoutNodes(nodes, edges); } catch { return nodes; }
}

const COMPACT_NODE: JsonSchema = S.obj('Node: id, type (from synapse_node_catalog), optional params/label/position.', {
  id: S.str('Unique node id, e.g. "dense1".'),
  type: S.str('Node type.'),
  label: S.str('Display label.'),
  params: S.obj('Parameter values by key (missing keys use defaults).'),
  position: S.obj('Canvas position.', { x: S.num('x'), y: S.num('y') }, { required: ['x', 'y'] }),
}, { required: ['id', 'type'] });

const COMPACT_EDGE: JsonSchema = S.obj('Edge from source node output port to target node input port.', {
  id: S.str('Optional edge id.'),
  source: S.str('Source node id.'),
  target: S.str('Target node id.'),
  sourceHandle: S.str('Output port id of the source (default: first output).'),
  targetHandle: S.str('Input port id of the target (default: first input).'),
}, { required: ['source', 'target'] });

const EDIT_OP: JsonSchema = S.obj('One edit operation. op is one of add_node(nodeType, id?, position?), remove_node(nodeId), set_param(nodeId, key, value), set_label(nodeId, label), move_node(nodeId, position), add_edge(source, target, sourceHandle?, targetHandle?), remove_edge(edgeId | source+target).', {
  op: S.enumOf('Operation.', ['add_node', 'remove_node', 'set_param', 'set_label', 'move_node', 'add_edge', 'remove_edge']),
  nodeType: S.str('add_node: node type.'),
  id: S.str('add_node: id for the new node.'),
  nodeId: S.str('Target node id.'),
  key: S.str('set_param: parameter key.'),
  value: { description: 'set_param: new value (number, string or boolean).' } as JsonSchema,
  label: S.str('set_label: new label.'),
  position: S.obj('Position {x, y}.', { x: S.num('x'), y: S.num('y') }),
  source: S.str('Edge source node id.'),
  target: S.str('Edge target node id.'),
  sourceHandle: S.str('Output port id.'),
  targetHandle: S.str('Input port id.'),
  edgeId: S.str('remove_edge: edge id.'),
}, { required: ['op'] });

async function saveGraph(snap: { nodes: Node[]; edges: Edge[]; modelId: string | null; modelName: string | null }, newName?: string) {
  const { nodes, edges } = snap;
  const graphConfig = generateModelConfigFromGraph(nodes, edges, NODE_DEFINITIONS) ?? {
    id: `canvas_${Date.now()}`, name: 'Canvas Network', layers: [], inputShape: '', outputSize: 0,
    graphMetadata: { nodeCount: nodes.length, edgeCount: edges.length, generatedAt: Date.now() },
  };
  const pythonCode = extractCanvasModelClass(generateTrainingScript(nodes, edges, { ...DEFAULT_CFG, epochs: 1 }));
  const graphIR = buildCanvasGraphIR(nodes, edges, DEFAULT_CFG);
  const design = { nodes, edges, graphConfig, pythonCode, schemaVersion: 1 };
  if (snap.modelId && !newName) {
    await updateCanvasNetworkModel(snap.modelId, graphConfig, pythonCode, graphIR);
    await call('save_canvas_model_design', { modelId: snap.modelId, designJson: JSON.stringify(design) });
    return { modelId: snap.modelId, name: snap.modelName ?? snap.modelId, created: false };
  }
  const name = (newName ?? '').trim() || `Synapse ${new Date().toLocaleDateString('de-DE')}`;
  const res = await exportCanvasNetworkToModelLibrary(graphConfig, pythonCode, name, graphIR);
  await call('save_canvas_model_design', { modelId: res.modelId, designJson: JSON.stringify(design) });
  return { modelId: res.modelId, name, created: true };
}

export const synapseTools: McpTool[] = [
  defineTool({
    name: 'synapse_node_catalog',
    title: 'Knoten-Katalog',
    description: 'Lists the node types available in the Synapse Builder with category, description, input/output port ids and parameters (key, type, default, options, range). Filter by category (data, layer, activation, training, math, logic) or by types to keep the answer small. Read this before building or editing a graph.',
    properties: {
      category: S.enumOf('Only this category.', ['data', 'layer', 'activation', 'training', 'math', 'logic']),
      types: S.strArr('Only these node types.'),
      include_params: S.bool('Include parameter definitions. Default true.'),
    },
    async run(args) {
      const cat = optStr(args, 'category') as NodeCategory | undefined;
      const types = Array.isArray(args.types) ? (args.types as string[]) : null;
      const withParams = optBool(args, 'include_params', true);
      const defs = NODE_DEFINITIONS.filter((d) => (!cat || d.category === cat) && (!types || types.includes(d.type)));
      const items = defs.map((d) => ({
        type: d.type, label: d.label, category: d.category, description: d.description,
        inputs: d.inputs.map((p) => ({ id: p.id, kind: p.portType })),
        outputs: d.outputs.map((p) => ({ id: p.id, kind: p.portType })),
        ...(withParams ? { params: d.params.map((p) => ({ key: p.key, type: p.type, default: p.default, ...(p.options ? { options: p.options } : {}), ...(p.min !== undefined ? { min: p.min } : {}), ...(p.max !== undefined ? { max: p.max } : {}), ...(p.description ? { description: p.description } : {}) })) } : {}),
      }));
      return ok(`${items.length} node type(s).`, items);
    },
  }),
  defineTool({
    name: 'synapse_blueprints',
    title: 'Vorlagen',
    description: 'Lists ready-made Synapse graph templates (MLP, CNN, ResNet, ViT, U-Net, LSTM, Transformer, Mini-GPT, audio, video, VAE, GAN, diffusion, detector). Pass blueprint_id to get the full graph, which you can adapt and write with synapse_graph_set.',
    properties: { blueprint_id: S.str('Return the full graph of this template.') },
    async run(args) {
      const list = blueprints as { id: string; goal: string; label: string; label_en?: string; description: string; description_en?: string; nodes: CompactNode[]; edges: (string | undefined)[][] }[];
      const id = optStr(args, 'blueprint_id');
      if (!id) {
        return ok(`${list.length} template(s).`, list.map((b) => ({ id: b.id, goal: b.goal, label: b.label_en ?? b.label, description: b.description_en ?? b.description, nodes: b.nodes.length })));
      }
      const bp = list.find((b) => b.id === id);
      if (!bp) return fail(`Unknown blueprint "${id}".`);
      const edges: CompactEdge[] = bp.edges.map(([source, target, sourceHandle, targetHandle]) => ({
        source: source!, target: target!, ...(sourceHandle ? { sourceHandle } : {}), ...(targetHandle ? { targetHandle } : {}),
      }));
      return ok(`Template ${bp.label_en ?? bp.label}.`, { id: bp.id, label: bp.label_en ?? bp.label, graph: { nodes: bp.nodes, edges } });
    },
  }),
  defineTool({
    name: 'synapse_graph_get',
    title: 'Graph lesen',
    description: 'Returns the current Synapse Builder graph (nodes with type/params/position, edges with ports), the saved model it belongs to, whether there are unsaved changes, and whether the builder page is open. Works when the page is closed (last autosave).',
    properties: { include_validation: S.bool('Also run the shape validation. Default true.') },
    async run(args, ctx) {
      const snap = readSynapseGraph(userOf(ctx));
      const graph = toCompactGraph(snap.nodes, snap.edges);
      const validation = optBool(args, 'include_validation', true) && snap.nodes.length ? validationReport(snap.nodes, snap.edges) : null;
      return ok(`${graph.nodes.length} node(s), ${graph.edges.length} edge(s)${snap.modelName ? `, model ${snap.modelName}` : ''}${snap.dirty ? ' (unsaved changes)' : ''}.`, {
        graph, model_id: snap.modelId, model_name: snap.modelName, unsaved_changes: snap.dirty,
        builder_open: snap.source === 'builder', validation,
      });
    },
  }),
  defineTool({
    name: 'synapse_graph_set',
    title: 'Graph ersetzen',
    description: 'Replaces the whole Synapse graph with the given nodes and edges (same format as synapse_graph_get / synapse_blueprints). Missing positions are auto-laid-out. The open builder canvas updates immediately; otherwise the graph appears next time the page opens. Returns the shape validation of the new graph. keep_model=false detaches the graph from the currently loaded saved model.',
    properties: {
      nodes: S.arr('Nodes.', COMPACT_NODE),
      edges: S.arr('Edges.', COMPACT_EDGE),
      keep_model: S.bool('Keep the link to the currently loaded saved model (so synapse_save overwrites it). Default true.'),
    },
    required: ['nodes', 'edges'],
    approval: 'once',
    async run(args, ctx) {
      const userId = userOf(ctx);
      const nodesIn = args.nodes as CompactNode[];
      const edgesIn = args.edges as CompactEdge[];
      const built = fromCompactGraph({ nodes: nodesIn, edges: edgesIn });
      if (built.errors.length) return fail(`Graph rejected: ${built.errors.join(' ')}`, { errors: built.errors });
      const nodes = layoutIfNeeded(built.nodes, built.edges, nodesIn.every((n) => !!n.position));
      const keep = optBool(args, 'keep_model', true);
      const target = writeSynapseGraph(userId, {
        nodes, edges: built.edges, dirty: true,
        ...(keep ? {} : { modelId: null, modelName: null }),
      }, origin(ctx, `Graph ersetzt (${nodes.length} Nodes)`));
      announceChange('synapse');
      const validation = validationReport(nodes, built.edges);
      return ok(`Graph written to the ${target === 'builder' ? 'open builder' : 'autosave'}: ${nodes.length} node(s), ${built.edges.length} edge(s), ${validation.valid ? 'valid' : `${validation.errors.length} error(s)`}.`, {
        written_to: target, warnings: built.warnings, validation, graph: toCompactGraph(nodes, built.edges),
      });
    },
  }),
  defineTool({
    name: 'synapse_graph_edit',
    title: 'Graph bearbeiten',
    description: 'Applies a list of edit operations to the current Synapse graph in order (add_node, remove_node, set_param, set_label, move_node, add_edge, remove_edge) — the same operations the built-in Synapse agent uses. Invalid ports fall back to the first port. Stops at the first failing operation unless continue_on_error is true. Returns per-op results and the validation of the edited graph.',
    properties: {
      ops: S.arr('Operations to apply.', EDIT_OP, { minItems: 1, maxItems: 200 }),
      continue_on_error: S.bool('Apply remaining ops after a failure. Default false.'),
      auto_layout: S.bool('Re-arrange nodes afterwards if they overlap. Default true.'),
    },
    required: ['ops'],
    approval: 'once',
    async run(args, ctx) {
      const userId = userOf(ctx);
      const snap = readSynapseGraph(userId);
      const exec = createToolExecutor({
        nodes: snap.nodes, edges: snap.edges, setNodes: () => {}, setEdges: () => {},
        nodeDefinitions: NODE_DEFINITIONS, visualDelayMs: 0,
      });
      const ops = args.ops as ToolArgs[];
      const results: { op: string; success: boolean; data?: unknown; error?: string }[] = [];
      const goOn = optBool(args, 'continue_on_error');
      for (const raw of ops) {
        const { op, ...rest } = raw as { op: string } & ToolArgs;
        const r = await exec.execute(op, rest);
        results.push({ op, success: r.success, data: r.data, error: r.error });
        if (!r.success && !goOn) break;
      }
      const failed = results.filter((r) => !r.success);
      const applied = results.filter((r) => r.success).length;
      if (applied === 0) return fail(`No operation applied: ${failed[0]?.error ?? 'unknown error'}`, { results });
      const state = exec.getState();
      const nodes = optBool(args, 'auto_layout', true) ? layoutIfNeeded(state.nodes, state.edges, true) : state.nodes;
      const target = writeSynapseGraph(userId, { nodes, edges: state.edges, dirty: true }, origin(ctx, `${applied} Aenderung(en) am Graph`));
      announceChange('synapse');
      const validation = validationReport(nodes, state.edges);
      const text = `${applied}/${ops.length} op(s) applied${failed.length ? `, ${failed.length} failed (${failed[0].error})` : ''}. Graph ${validation.valid ? 'valid' : `has ${validation.errors.length} error(s)`}.`;
      return { ...ok(text, { written_to: target, results, validation, graph: toCompactGraph(nodes, state.edges) }), ...(failed.length && !goOn ? { isError: true } : {}) };
    },
  }),
  defineTool({
    name: 'synapse_validate',
    title: 'Graph pruefen',
    description: 'Validates the current Synapse graph: connectivity, port types and tensor shapes (static check), and with runtime=true also builds the network in the Python shape service (real tensor shapes, parameter count, memory). Returns errors with the affected node ids and fix suggestions.',
    properties: { runtime: S.bool('Also run the Python shape service. Default true.') },
    async run(args, ctx) {
      const snap = readSynapseGraph(userOf(ctx));
      if (!snap.nodes.length) return fail('The graph is empty.');
      const report = validationReport(snap.nodes, snap.edges);
      let runtime: unknown = null;
      if (optBool(args, 'runtime', true)) {
        try {
          runtime = await call('synapse_check', { graph: buildCanvasGraphIR(snap.nodes, snap.edges, { ...DEFAULT_CFG, epochs: 1, batchSize: 1 }) });
        } catch (e) { runtime = { error: errorText(e) }; }
      }
      const text = report.valid
        ? 'Static validation passed.'
        : `Static validation found ${report.errors.length} error(s): ${report.errors.slice(0, 5).map((e) => e.text).join(' | ')}`;
      return ok(text, { ...report, runtime });
    },
  }),
  defineTool({
    name: 'synapse_generate_code',
    title: 'PyTorch-Code erzeugen',
    description: 'Generates PyTorch code from the current Synapse graph: either just the nn.Module class (part "model") or the full standalone training script (part "script").',
    properties: {
      part: S.enumOf('Which code. Default "model".', ['model', 'script']),
      epochs: S.int('Epochs written into the script. Default 10.', { minimum: 1 }),
      batch_size: S.int('Batch size in the script. Default 32.', { minimum: 1 }),
      learning_rate: S.num('Learning rate in the script. Default 0.001.'),
    },
    async run(args, ctx) {
      const snap = readSynapseGraph(userOf(ctx));
      if (!snap.nodes.length) return fail('The graph is empty.');
      const script = generateTrainingScript(snap.nodes, snap.edges, {
        ...DEFAULT_CFG, epochs: optNum(args, 'epochs', 10)!, batchSize: optNum(args, 'batch_size', 32)!, learningRate: optNum(args, 'learning_rate', 0.001)!,
      });
      const code = optStr(args, 'part') === 'script' ? script : extractCanvasModelClass(script);
      return ok(clip(code, 30000), { language: 'python', code });
    },
  }),
  defineTool({
    name: 'synapse_models_list',
    title: 'Gespeicherte Synapse-Modelle',
    description: 'Lists saved Synapse (canvas) models with id, name, task, number of classes and whether trained weights exist.',
    async run(_args, ctx) {
      const list = await call<unknown[]>('list_canvas_models_with_pt', { userId: userOf(ctx) });
      return ok(`${list.length} saved Synapse model(s).`, list);
    },
  }),
  defineTool({
    name: 'synapse_open_model',
    title: 'Synapse-Modell laden',
    description: 'Loads a saved Synapse model design into the builder (replacing the current graph). Unsaved changes of the current graph are lost unless saved first with synapse_save.',
    properties: { model_id: S.str('Canvas model id from synapse_models_list.') },
    required: ['model_id'],
    approval: 'once',
    async run(args, ctx) {
      const userId = userOf(ctx);
      const modelId = reqStr(args, 'model_id');
      const raw = await call<string | null>('load_canvas_model_design', { modelId });
      if (!raw) return fail(`No saved design for model ${modelId}.`);
      const design = JSON.parse(raw) as { nodes?: Node[]; edges?: Edge[] };
      const models = await call<{ model_id: string; name: string }[]>('list_canvas_models_with_pt', { userId }).catch(() => []);
      const name = models.find((m) => m.model_id === modelId)?.name ?? modelId;
      const nodes = design.nodes ?? [], edges = design.edges ?? [];
      const target = writeSynapseGraph(userId, { nodes, edges, modelId, modelName: name, dirty: false }, origin(ctx, `Modell ${name} geladen`));
      announceChange('synapse');
      return ok(`Loaded ${name} (${nodes.length} nodes).`, { written_to: target, model_id: modelId, model_name: name, graph: toCompactGraph(nodes, edges) });
    },
  }),
  defineTool({
    name: 'synapse_save',
    title: 'Als Modell speichern',
    description: 'Saves the current Synapse graph as a model in the library (graph, generated PyTorch code and IR). If the graph belongs to a saved model it is overwritten, unless name is given — then a new model is created. The model id can be used for synapse_train and the inference tab.',
    properties: { name: S.str('Create a new model with this name (instead of updating the loaded one).') },
    approval: 'once',
    async run(args, ctx) {
      const userId = userOf(ctx);
      const snap = readSynapseGraph(userId);
      if (!snap.nodes.length) return fail('The graph is empty.');
      const saved = await saveGraph(snap, optStr(args, 'name'));
      writeSynapseGraph(userId, { modelId: saved.modelId, modelName: saved.name, dirty: false }, origin(ctx, `Modell ${saved.name} gespeichert`));
      announceChange('models', { modelId: saved.modelId });
      return ok(`${saved.created ? 'Created' : 'Updated'} model ${saved.name} (${saved.modelId}).`, { model_id: saved.modelId, name: saved.name, created: saved.created });
    },
  }),
  defineTool({
    name: 'synapse_train',
    title: 'Synapse-Graph trainieren',
    description: 'Trains the current Synapse graph (runtime IR) on a dataset. The graph must validate; graphs with a data loader node need a dataset_id of the matching kind. If the builder page is open, training runs there with live charts. Uses GPU/CPU for a long time — the user confirms each start. Follow progress with training_status (or synapse_train_status). Save the graph first (synapse_save) so the trained weights are attached to a model.',
    properties: {
      dataset_id: S.str('Dataset id (required when the graph has a loader node).'),
      epochs: S.int('Default 10.', { minimum: 1 }),
      batch_size: S.int('Default 32.', { minimum: 1 }),
      learning_rate: S.num('Default 0.001.'),
      device: S.str('"cpu", "cuda:0", "mps". Default: best available (CUDA, then MPS, then CPU).'),
      precision: S.enumOf('Default fp32.', ['fp32', 'fp16', 'bf16']),
      grad_accum: S.int('Gradient accumulation. Default 1.', { minimum: 1 }),
    },
    approval: 'always',
    annotations: { readOnlyHint: false },
    async run(args, ctx) {
      const userId = userOf(ctx);
      const snap = readSynapseGraph(userId);
      if (!snap.nodes.length) return fail('The graph is empty.');
      const report = validationReport(snap.nodes, snap.edges);
      if (!report.valid) return fail(`Graph is invalid, training not started: ${report.errors.map((e) => e.text).join(' | ')}`, report);
      const datasetId = optStr(args, 'dataset_id');
      const ds = datasetId ? await findDataset(datasetId) : null;
      const hasLoader = snap.nodes.some((n) => LOADER_NODE_TYPES.includes(String((n.data as Record<string, unknown>)?.nodeType ?? '')));
      if (hasLoader && !ds) throw new ToolArgError('The graph has a data loader node — pass dataset_id.');
      const cfg = {
        epochs: optNum(args, 'epochs', 10)!, batchSize: optNum(args, 'batch_size', 32)!, learningRate: optNum(args, 'learning_rate', 0.001)!,
        gpu: optStr(args, 'device') ?? await bestDevice(), precision: (optStr(args, 'precision') ?? 'fp32') as 'fp32' | 'fp16' | 'bf16', gradAccum: optNum(args, 'grad_accum', 1)!,
      };
      const builder = getAttachedBuilder(userId);
      if (builder?.startTraining) {
        await builder.startTraining({ ...cfg, selectedDatasetId: ds?.id, selectedDatasetName: ds?.name });
        const current = await call<{ id: string } | null>('get_current_training').catch(() => null);
        return ok(`Training started in the open Synapse Builder${current ? ` (job ${current.id})` : ''}. Poll training_status.`, { job_id: current?.id ?? null, runs_in: 'builder' });
      }
      const ir = buildCanvasGraphIR(snap.nodes, snap.edges, cfg);
      ir.training.gpuExplicit = true;
      const datasetPath = ds ? await call<string>('get_dataset_path', { datasetId: ds.id }) : '';
      const job = await call<{ id: string; output_path?: string | null }>('start_training', {
        modelId: snap.modelId ?? '', modelName: snap.modelName || 'Synapse Canvas Model',
        datasetId: ds?.id ?? '', datasetName: ds?.name ?? '', versionId: null,
        config: {
          task_type: 'canvas', canvas_graph: ir, epochs: cfg.epochs, batch_size: cfg.batchSize, learning_rate: cfg.learningRate,
          fp16: cfg.precision === 'fp16', bf16: cfg.precision === 'bf16', dataset_path: datasetPath,
          optimizer: ir.training.optimizer, scheduler: ir.training.scheduler,
        },
      });
      registerVersionOnComplete(job.id, snap.modelName || 'Synapse Canvas Model', job.output_path ?? null);
      try { await call('enable_prevent_sleep'); } catch { /* optional */ }
      return ok(`Training started (job ${job.id}). Poll training_status.${snap.modelId ? '' : ' Note: the graph is not saved as a model, so the weights will not be attached to one.'}`, { job_id: job.id, runs_in: 'background' });
    },
  }),
  defineTool({
    name: 'synapse_train_status',
    title: 'Synapse-Trainingsstatus',
    description: 'Status, epoch losses and recent log lines of a Synapse training job (same data as training_status).',
    properties: { job_id: S.str('Training job id.') },
    async run(args) {
      let id = optStr(args, 'job_id');
      if (!id) id = (await call<{ id: string } | null>('get_current_training').catch(() => null))?.id;
      if (!id) return ok('No Synapse training is running.', { job: null });
      const run = summarizeRun(getRun(id));
      return ok(`Job ${id}: ${run?.status ?? 'unknown'}.`, { job_id: id, live: run });
    },
  }),
];

/** Wie die Trainingskonsole: CUDA vor MPS vor CPU. */
async function bestDevice(): Promise<string> {
  try {
    const r = await call<{ cuda_available?: boolean; mps_available?: boolean }>('check_training_requirements');
    return r.cuda_available ? 'cuda:0' : r.mps_available ? 'mps' : 'cpu';
  } catch {
    return 'cpu';
  }
}

/**
 * Ohne offenen Builder registriert niemand die neue Version, wenn Rust keine
 * `new_version_id` liefert. Wie im Builder: einmal auf das Ende warten.
 */
function registerVersionOnComplete(jobId: string, modelName: string, outputPath: string | null) {
  void (async () => {
    try {
      const { listen } = await import('@tauri-apps/api/event');
      const offs: (() => void)[] = [];
      const done = () => { offs.forEach((o) => o()); call('disable_prevent_sleep').catch(() => {}); };
      offs.push(await listen<{ job_id?: string; new_version_id?: string; data?: { output_path?: string } }>('training-complete', (e) => {
        if (e.payload?.job_id && e.payload.job_id !== jobId) return;
        if (!e.payload?.new_version_id) {
          const out = e.payload?.data?.output_path ?? (outputPath ? `${outputPath.replace(/\/$/, '')}/final_model` : null);
          if (out) call('register_synapse_training_version', { modelName, outputDir: out }).catch(() => {});
        }
        done();
      }));
      offs.push(await listen<{ job_id?: string }>('training-error', (e) => {
        if (e.payload?.job_id && e.payload.job_id !== jobId) return;
        done();
      }));
    } catch { /* ausserhalb von Tauri */ }
  })();
}
