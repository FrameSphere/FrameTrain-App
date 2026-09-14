// Task-spezifischer Kontext fuer die KI-Trainingsanalyse.
//
// Regression aus dem App-Durchgang vom 14.09.2026: Bei einem Canvas-Modell
// (CSV Loader -> Dense -> Output) lobte die KI "das Warm-up-Verhaeltnis von
// 0,06" und empfahl "Dropout auf 0,2 erhoehen" — beides Formularwerte, die das
// Canvas-Plugin gar nicht auswertet. Das Netz hatte weder Warmup noch Dropout.
// Die KI sah nur die Formular-Config, nicht den Graph.

import { hiddenTrainingFieldsForTaskType } from '../plugins/registry';
import { expandHiddenFields } from './coachToolEvents';

/** Felder, die bei Canvas-Modellen wirklich greifen (kommen aus dem Synapse Builder). */
export const CANVAS_TRAINING_FIELDS = new Set([
  'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer', 'scheduler',
  'max_grad_norm', 'gradient_accumulation_steps', 'label_smoothing',
]);

export function isCanvasTask(taskType?: string | null): boolean {
  return String(taskType ?? '').trim() === 'canvas';
}

/** Config-Felder, die fuer diesen Task-Typ keine Wirkung haben. */
export function unavailableAnalysisFields(taskType: string | null | undefined, allFields: Iterable<string>): Set<string> {
  const out = expandHiddenFields(hiddenTrainingFieldsForTaskType(taskType ?? undefined));
  if (isCanvasTask(taskType)) {
    for (const f of allFields) if (!CANVAS_TRAINING_FIELDS.has(f)) out.add(f);
  }
  return out;
}

interface IRNode { id: string; type: string; params?: Record<string, unknown> }
interface CanvasIR { nodes?: IRNode[]; execution_order?: string[] }

const PARAM_KEYS: Record<string, string[]> = {
  csv_loader: ['targetCol'], parquet_loader: ['targetCol'], image_loader: ['imageSize', 'channels'],
  dense: ['inputSize', 'outputSize'], conv2d: ['inChannels', 'outChannels', 'kernelSize'],
  dropout: ['p'], lstm: ['inputSize', 'hiddenSize'], embedding: ['vocabSize', 'embedDim'],
  output_node: ['numClasses', 'taskType'], loss: ['type', 'labelSmoothing'],
  optimizer: ['type', 'lr', 'weightDecay'], scheduler: ['type', 'warmupSteps'],
};

/** Kompakte Beschreibung des Canvas-Graphs in Ausfuehrungsreihenfolge. */
export function canvasGraphSummary(graph: unknown): string | null {
  const ir = graph as CanvasIR | null;
  const nodes = ir?.nodes ?? [];
  if (nodes.length === 0) return null;
  const byId = new Map(nodes.map(n => [n.id, n]));
  const order = (ir?.execution_order ?? []).map(id => byId.get(id)).filter((n): n is IRNode => !!n);
  const seen = new Set(order.map(n => n.id));
  const all = [...order, ...nodes.filter(n => !seen.has(n.id))];
  const describe = (n: IRNode) => {
    const keys = PARAM_KEYS[n.type] ?? [];
    const params = keys
      .filter(k => n.params?.[k] !== undefined && n.params?.[k] !== '')
      .map(k => `${k}=${String(n.params?.[k])}`);
    return params.length ? `${n.type}(${params.join(', ')})` : n.type;
  };
  const types = new Set(all.map(n => n.type));
  const absent = ['dropout', 'batchnorm', 'layernorm', 'scheduler'].filter(t => !types.has(t));
  const lines = [`Graph (${all.length} Nodes): ${all.map(describe).join(' -> ')}`];
  if (absent.length) lines.push(`Nicht im Graph vorhanden: ${absent.join(', ')}`);
  return lines.join('\n');
}

/** Zusatz fuer den System-Prompt bei Canvas-Modellen. */
export function canvasPromptBlock(language: string): string {
  return language === 'de'
    ? `Dies ist ein Canvas-Netz aus dem Synapse Builder. Regularisierung (Dropout, Normalisierung), Architektur und Warmup gibt es nur als Nodes im Graph — bewerte ausschliesslich, was im Graph steht, und erfinde keine Werte fuer Nodes, die fehlen.
Architektur-Aenderungen (z. B. "Dropout-Node nach Dense einfuegen", "zweiten Dense-Layer mit ReLU ergaenzen") gehoeren in die Verbesserungsvorschlaege als Text. Der JSON-Block enthaelt nur Felder, die im Synapse Builder als Trainingswerte gesetzt werden.`
    : `This is a Canvas network from the Synapse Builder. Regularization (dropout, normalization), architecture and warmup exist only as nodes in the graph — judge only what the graph contains and do not invent values for nodes that are absent.
Architecture changes (e.g. "add a Dropout node after Dense") belong in the improvement suggestions as text. The JSON block contains only fields that are set as training values in the Synapse Builder.`;
}
