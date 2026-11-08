// Zugriffsschicht auf den Synapse-Graphen ausserhalb der React-Komponente.
//
// Der Graph lebt im React-State des SynapseBuilder. Werkzeuge (MCP) muessen
// ihn lesen und schreiben koennen — auch wenn die Seite gerade nicht offen
// ist. Deshalb:
//   - Ist der Builder montiert, meldet er sich hier an (attachSynapseBuilder);
//     Lesen/Schreiben geht direkt an seinen State, die Leinwand aktualisiert
//     sich sofort.
//   - Sonst dient der Autosave im localStorage (synapse_autosave_<user>) als
//     Speicher. Der Builder stellt ihn beim naechsten Oeffnen ohnehin wieder
//     her — Aenderungen eines Werkzeugs sind dann sichtbar.

import type { Edge, Node } from "@xyflow/react";
import { NODE_DEFINITIONS, type NodeDef } from "./nodeTypes";
import { getSynapseNodeType } from "./graph-shape-validation";

export interface SynapseSnapshot {
  nodes: Node[];
  edges: Edge[];
  /** Gespeichertes Canvas-Modell, zu dem der Graph gehoert. */
  modelId: string | null;
  modelName: string | null;
  /** Ungespeicherte Aenderungen gegenueber dem Modell. */
  dirty: boolean;
}

export interface SynapseTrainRequest {
  epochs: number;
  batchSize: number;
  learningRate: number;
  gpu: string;
  precision: "fp32" | "fp16" | "bf16";
  gradAccum: number;
  selectedDatasetId?: string;
  selectedDatasetName?: string;
}

export interface SynapseBuilderHandle {
  userId: string;
  get(): SynapseSnapshot;
  /** Ersetzt Graph und/oder Modellbezug in der offenen Leinwand. */
  apply(next: Partial<SynapseSnapshot>, origin?: string): void;
  /** Startet ein Training mit der Oberflaeche (Konsole, Live-Kurven, Versionen). */
  startTraining?(config: SynapseTrainRequest): Promise<void>;
}

let builder: SynapseBuilderHandle | null = null;

export function attachSynapseBuilder(handle: SynapseBuilderHandle): () => void {
  builder = handle;
  return () => { if (builder === handle) builder = null; };
}

export function getAttachedBuilder(userId: string): SynapseBuilderHandle | null {
  return builder && builder.userId === userId ? builder : null;
}

export function autosaveKey(userId: string): string {
  return `synapse_autosave_${userId}`;
}

function readAutosave(userId: string): SynapseSnapshot {
  try {
    const raw = localStorage.getItem(autosaveKey(userId));
    const saved = raw ? JSON.parse(raw) : null;
    if (saved && Array.isArray(saved.nodes)) {
      return {
        nodes: saved.nodes,
        edges: Array.isArray(saved.edges) ? saved.edges : [],
        modelId: saved.activeCanvasModelId ?? null,
        modelName: saved.activeModelName ?? null,
        dirty: saved.dirty ?? false,
      };
    }
  } catch { /* defekter Autosave = leer */ }
  return { nodes: [], edges: [], modelId: null, modelName: null, dirty: false };
}

function writeAutosave(userId: string, snap: SynapseSnapshot) {
  if (snap.nodes.length === 0 && snap.edges.length === 0 && !snap.modelId) {
    localStorage.removeItem(autosaveKey(userId));
    return;
  }
  localStorage.setItem(autosaveKey(userId), JSON.stringify({
    nodes: snap.nodes,
    edges: snap.edges,
    activeCanvasModelId: snap.modelId,
    activeModelName: snap.modelName,
    dirty: snap.dirty,
    savedAt: Date.now(),
  }));
}

/** Aktueller Graph — aus der offenen Leinwand oder dem Autosave. */
export function readSynapseGraph(userId: string): SynapseSnapshot & { source: "builder" | "autosave" } {
  const b = getAttachedBuilder(userId);
  if (b) return { ...b.get(), source: "builder" };
  return { ...readAutosave(userId), source: "autosave" };
}

/** Schreibt Graph und/oder Modellbezug. Rueckgabe: wohin geschrieben wurde. */
export function writeSynapseGraph(
  userId: string, next: Partial<SynapseSnapshot>, origin?: string,
): "builder" | "autosave" {
  const b = getAttachedBuilder(userId);
  if (b) {
    b.apply(next, origin);
    return "builder";
  }
  writeAutosave(userId, { ...readAutosave(userId), ...next });
  return "autosave";
}

// ── Knoten bauen / kompakt darstellen ──────────────────────────────────────

export function findNodeDef(type: string, defs: NodeDef[] = NODE_DEFINITIONS): NodeDef | undefined {
  return defs.find((d) => d.type === type);
}

/** Ein Leinwand-Knoten, wie ihn auch der Agent und die Bibliothek anlegen. */
export function buildFlowNode(
  def: NodeDef, id: string, params: Record<string, unknown> = {},
  position: { x: number; y: number } = { x: 0, y: 0 }, label?: string,
): Node {
  const merged: Record<string, unknown> = {};
  (def.params ?? []).forEach((p) => { merged[p.key] = p.default; });
  Object.assign(merged, params);
  return {
    id,
    type: "synapseNode",
    position,
    data: {
      _def: def,
      nodeType: def.type,
      label: label ?? def.label,
      category: def.category,
      icon: def.icon,
      color: def.color,
      inputs: def.inputs ?? [],
      outputs: def.outputs ?? [],
      paramDefs: def.params ?? [],
      params: merged,
    },
  };
}

export interface CompactNode {
  id: string;
  type: string;
  label?: string;
  params?: Record<string, unknown>;
  position?: { x: number; y: number };
}

export interface CompactEdge {
  id?: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

/** Graph ohne Darstellungsdetails — so sehen ihn Werkzeuge und KI. */
export function toCompactGraph(nodes: Node[], edges: Edge[]): { nodes: CompactNode[]; edges: CompactEdge[] } {
  return {
    nodes: nodes.map((n) => {
      const d = (n.data ?? {}) as Record<string, unknown>;
      return {
        id: n.id,
        type: getSynapseNodeType(n),
        label: typeof d.label === "string" ? d.label : undefined,
        params: (d.params as Record<string, unknown>) ?? {},
        position: n.position ? { x: Math.round(n.position.x), y: Math.round(n.position.y) } : undefined,
      };
    }),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? undefined,
      targetHandle: e.targetHandle ?? undefined,
    })),
  };
}

/**
 * Kompakter Graph -> Leinwand. Unbekannte Typen und Kanten zu fehlenden Knoten
 * werden als Fehler gemeldet, nicht stillschweigend verworfen.
 * Fehlende Positionen: einfache Reihen-Anordnung links nach rechts.
 */
export function fromCompactGraph(
  graph: { nodes: CompactNode[]; edges: CompactEdge[] },
  defs: NodeDef[] = NODE_DEFINITIONS,
): { nodes: Node[]; edges: Edge[]; errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];
  const nodes: Node[] = [];
  const seen = new Set<string>();
  graph.nodes.forEach((cn, i) => {
    const def = findNodeDef(cn.type, defs);
    if (!def) { errors.push(`Unknown node type "${cn.type}" (node ${cn.id}). Use synapse_node_catalog.`); return; }
    if (!cn.id || seen.has(cn.id)) { errors.push(`Missing or duplicate node id "${cn.id}".`); return; }
    seen.add(cn.id);
    const pos = cn.position ?? { x: 160 + (i % 4) * 300, y: 120 + Math.floor(i / 4) * 220 };
    nodes.push(buildFlowNode(def, cn.id, cn.params ?? {}, pos, cn.label));
  });
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges: Edge[] = [];
  graph.edges.forEach((ce, i) => {
    const src = byId.get(ce.source), tgt = byId.get(ce.target);
    if (!src || !tgt) { errors.push(`Edge ${ce.source} -> ${ce.target}: node missing.`); return; }
    const outs = ((src.data as Record<string, unknown>).outputs as { id: string }[]) ?? [];
    const ins = ((tgt.data as Record<string, unknown>).inputs as { id: string }[]) ?? [];
    // Code-Nodes bekommen ihre Ports erst aus dem Code (Builder), Namen bleiben daher stehen.
    const code = (n: Node) => getSynapseNodeType(n) === "code";
    const sh = ce.sourceHandle && (code(src) || outs.some((p) => p.id === ce.sourceHandle)) ? ce.sourceHandle : outs[0]?.id;
    const th = ce.targetHandle && (code(tgt) || ins.some((p) => p.id === ce.targetHandle)) ? ce.targetHandle : ins[0]?.id;
    if (ce.sourceHandle && sh !== ce.sourceHandle) warnings.push(`Edge ${ce.source} -> ${ce.target}: unknown sourceHandle "${ce.sourceHandle}", using "${sh}".`);
    if (ce.targetHandle && th !== ce.targetHandle) warnings.push(`Edge ${ce.source} -> ${ce.target}: unknown targetHandle "${ce.targetHandle}", using "${th}".`);
    edges.push({
      id: ce.id ?? `e-${ce.source}-${ce.target}-${i}`,
      source: ce.source,
      target: ce.target,
      sourceHandle: sh,
      targetHandle: th,
      animated: true,
      style: { stroke: "#a78bfa", strokeWidth: 1.5 },
    });
  });
  return { nodes, edges, errors, warnings };
}
