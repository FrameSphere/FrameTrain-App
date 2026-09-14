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
interface CanvasIR {
  nodes?: IRNode[];
  execution_order?: string[];
  training?: { scheduler?: string; warmupSteps?: number; minLr?: number };
}

// Wie der Nutzer fehlende Nodes im Synapse Builder einbaut. Regression aus dem
// App-Durchgang mit 1.2.72: Der Chat empfahl "vor dem Scheduler einen
// Warmup-Node einfuegen" — den gibt es nicht. Warmup ist das Feld
// "Warmup Steps" im Node "LR Scheduler"; ohne diesen Node laeuft cosine ohne Warmup.
const ABSENT_HINTS: Record<string, string> = {
  dropout: "Node 'Dropout' (Parameter p)",
  batchnorm: "Node 'BatchNorm'",
  layernorm: "Node 'LayerNorm'",
  scheduler: "Node 'LR Scheduler' (Parameter Schedule, Warmup Steps, Min LR) — ohne ihn: cosine ohne Warmup",
};

const PARAM_KEYS: Record<string, string[]> = {
  csv_loader: ['targetCol'], parquet_loader: ['targetCol'], image_loader: ['imageSize', 'channels'],
  dense: ['inputSize', 'outputSize'], conv2d: ['inChannels', 'outChannels', 'kernelSize'],
  dropout: ['p'], lstm: ['inputSize', 'hiddenSize'], embedding: ['vocabSize', 'embedDim'],
  output_node: ['numClasses', 'taskType'], loss: ['type', 'labelSmoothing'],
  optimizer: ['type', 'lr', 'weightDecay'], scheduler: ['type', 'warmupSteps'],
};

/** Kompakte Beschreibung des Canvas-Graphs in Ausfuehrungsreihenfolge. */
export function canvasGraphSummary(graph: unknown, stepsPerEpoch?: number | null): string | null {
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
  if (absent.length) lines.push(`Nicht im Graph vorhanden: ${absent.map(t => `${t} = ${ABSENT_HINTS[t]}`).join('; ')}`);
  const tr = ir?.training;
  if (tr) {
    const warm = Number(tr.warmupSteps ?? 0);
    lines.push(`Scheduler im Training: ${tr.scheduler ?? 'cosine'}, warmupSteps=${warm}${warm > 0 ? '' : ' (kein Warmup)'}. Einen eigenen Warmup-Node gibt es nicht.`);
  }
  if (stepsPerEpoch && stepsPerEpoch > 0) {
    const spe = Math.round(stepsPerEpoch);
    lines.push(`Optimizer-Steps pro Epoche: ${spe}. Warmup Steps werden auf ganze Epochen aufgerundet (1-${spe} Steps = 1 Epoche Warmup, one_cycle ignoriert Warmup Steps).`);
  }
  return lines.join('\n');
}

/** Zusatz fuer den System-Prompt bei Canvas-Modellen. */
export function canvasPromptBlock(language: string): string {
  return language === 'de'
    ? `Dies ist ein Canvas-Netz aus dem Synapse Builder. Regularisierung (Dropout, Normalisierung) und Architektur gibt es nur als Nodes im Graph; Warmup ist das Feld "Warmup Steps" im Node "LR Scheduler" (es gibt keinen Warmup-Node). Bewerte ausschliesslich, was im Graph steht, und erfinde keine Werte fuer Nodes, die fehlen.
Architektur-Aenderungen (z. B. "Dropout-Node nach Dense einfuegen", "zweiten Dense-Layer mit ReLU ergaenzen") gehoeren in die Verbesserungsvorschlaege als Text. Der JSON-Block enthaelt nur Felder, die im Synapse Builder als Trainingswerte gesetzt werden.`
    : `This is a Canvas network from the Synapse Builder. Regularization (dropout, normalization) and architecture exist only as nodes in the graph; warmup is the "Warmup Steps" field of the "LR Scheduler" node (there is no warmup node). Judge only what the graph contains and do not invent values for nodes that are absent.
Architecture changes (e.g. "add a Dropout node after Dense") belong in the improvement suggestions as text. The JSON block contains only fields that are set as training values in the Synapse Builder.`;
}

/**
 * Entfernt aus ```json-Bloecken einer Chat-Antwort alle Felder, die nicht
 * setzbar sind oder fuer diesen Task-Typ nicht wirken.
 *
 * Regression aus dem App-Durchgang vom 14.09.2026: Auf "Soll ich Dropout
 * erhoehen oder Warmup einstellen?" antwortete der Chat eines Canvas-Modells
 * mit {"dropout_rate": 0.4, "warmup_steps": 4} — ein Feld, das es nirgends
 * gibt, und eines, das Canvas nur als Node kennt. Bleibt nichts uebrig,
 * verschwindet der Block ganz.
 */
export function sanitizeChatJson(reply: string, settable: Iterable<string>, unavailable: ReadonlySet<string>): string {
  const allowed = new Set([...settable].filter(k => !unavailable.has(k)));
  return reply.replace(/```json\s*([\s\S]*?)```\n?/gi, (block, body: string) => {
    let parsed: unknown;
    try { parsed = JSON.parse(body.trim()); } catch { return block; }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return block;
    const kept = Object.fromEntries(Object.entries(parsed as Record<string, unknown>).filter(([k]) => allowed.has(k)));
    if (Object.keys(kept).length === 0) return '';
    return '```json\n' + JSON.stringify(kept, null, 2) + '\n```\n';
  }).replace(/\n{3,}/g, '\n\n').trim();
}

/** Zusatz fuer den Chat-Prompt: welche Felder ein JSON-Block enthalten darf. */
export function chatFieldRule(language: string, fieldList: string, canvas: boolean): string {
  const de = language === 'de';
  const base = de
    ? `Ein JSON-Block darf ausschliesslich diese Felder enthalten (exakt diese Namen):\n${fieldList}`
    : `A JSON block may contain only these fields (exactly these names):\n${fieldList}`;
  if (!canvas) return base;
  return base + (de
    ? `\nFragt der User nach etwas, das bei Canvas nur als Node existiert (Dropout, Normalisierung, Warmup, Layer): sag klar, ob der Node im Graph vorhanden ist. Fehlt er, erklaere, welchen Node er wo im Synapse Builder einfuegen soll und mit welchem Parameter — als Text, niemals als JSON-Feld. Nenne nur Nodes und Parameter, die im Kontext stehen. Rechne Warmup in Optimizer-Steps mit den Steps pro Epoche aus dem Kontext.`
    : `\nIf the user asks about something that exists only as a node in Canvas (dropout, normalization, warmup, layers): say clearly whether that node is in the graph. If it is absent, explain which node to add where in the Synapse Builder and with which parameter — as text, never as a JSON field. Name only nodes and parameters given in the context. Compute warmup in optimizer steps using the steps per epoch from the context.`);
}
