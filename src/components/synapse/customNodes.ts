/**
 * Eigene Nodes: Code-Nodes und Bloecke aus der Bibliothek.
 *
 * Ein platzierter eigener Node ist ein normaler "code"- oder "block"-Node mit
 * Name, Icon, Farbe und einer Kopie der Definition. Das Modell braucht die
 * Bibliothek also nie; libraryId/libraryVersion zeigen nur, ob es ein Update gibt.
 */

import type { Edge, Node } from "@xyflow/react";
import {
  Activity, Box, Brain, Combine, Cpu, Ear, Eye, Filter, GitMerge, Image, Layers, Network,
  Repeat, Shuffle, Sigma, Sparkles, Split, Type, Waves, Zap, type LucideIcon,
} from "lucide-react";
import type { NodeDef, ParamDef, PortDef } from "./nodeTypes";

// ── Aussehen ────────────────────────────────────────────────────────────────

export const CUSTOM_ICONS: Record<string, LucideIcon> = {
  layers: Layers, merge: GitMerge, waves: Waves, brain: Brain, box: Box, sparkles: Sparkles,
  zap: Zap, filter: Filter, shuffle: Shuffle, sigma: Sigma, activity: Activity, cpu: Cpu,
  network: Network, combine: Combine, split: Split, repeat: Repeat, eye: Eye, ear: Ear,
  type: Type, image: Image,
};

export const CUSTOM_COLORS = [
  "#a78bfa", "#2dd4bf", "#fb7185", "#f59e0b", "#60a5fa", "#4ade80", "#f472b6", "#94a3b8",
];

export const DEFAULT_CODE = `class Node(nn.Module):
    def __init__(self, shape, hidden: int = 64):
        super().__init__()
        self.inner = nn.Linear(shape[-1], hidden)
        self.back = nn.Linear(hidden, shape[-1])

    def forward(self, x):
        return x + self.back(F.gelu(self.inner(x)))
`;

// ── Typen ───────────────────────────────────────────────────────────────────

/** Was der Formen-Dienst aus dem Code liest (synapse_describe_code). */
export interface CodeDescription {
  ok: boolean;
  error?: string;
  inputs?: { id: string; label: string; optional: boolean }[];
  outputs?: { id: string }[];
  params?: { key: string; default: unknown; kind: "int" | "float" | "boolean" | "select" | "string"; options: string[] }[];
  probe?: Record<string, number[]>;
  probeError?: string;
  parameters?: number;
}

export interface CustomNodeDefinition {
  icon: string;
  color: string;
  description: string;
  // Code-Node
  code?: string;
  inputs?: PortDef[];
  outputs?: PortDef[];
  params?: ParamDef[];
  values?: Record<string, unknown>;
  // Block
  graph?: unknown;
  repeat?: number;
}

export interface CustomNodeRecord {
  id: string;
  name: string;
  kind: "code" | "block";
  version: number;
  definition: CustomNodeDefinition;
  updated_at: number;
}

export interface CustomNodeInput {
  id?: string | null;
  name: string;
  kind: "code" | "block";
  definition: CustomNodeDefinition;
}

// ── Code -> Ports und Einstellungen ─────────────────────────────────────────

export const CODE_PARAM: ParamDef = { key: "code", label: "Python", type: "code", default: DEFAULT_CODE };

export function portsFromDescription(desc: CodeDescription): { inputs: PortDef[]; outputs: PortDef[] } {
  return {
    inputs: (desc.inputs ?? []).map((p) => ({
      id: p.id, label: p.optional ? `${p.label}?` : p.label, portType: "tensor" as const,
    })),
    outputs: (desc.outputs ?? []).map((p) => ({ id: p.id, label: p.id, portType: "tensor" as const })),
  };
}

export function paramsFromDescription(desc: CodeDescription): ParamDef[] {
  return (desc.params ?? []).map((p): ParamDef => {
    switch (p.kind) {
      case "int": return { key: p.key, label: p.key, type: "number", step: 1, default: p.default };
      case "float": return { key: p.key, label: p.key, type: "number", step: 0.01, default: p.default };
      case "boolean": return { key: p.key, label: p.key, type: "boolean", default: Boolean(p.default) };
      case "select": return { key: p.key, label: p.key, type: "select", options: p.options, default: String(p.default) };
      default: return { key: p.key, label: p.key, type: "string", default: p.default ?? "" };
    }
  });
}

const typeOf = (n: Node): string =>
  ((n.data as any)?._def?.type ?? (n.data as any)?.nodeType ?? "") as string;

/** Neue Beschreibung auf einen Code-Node legen: Werte bleiben, wo der Name bleibt. */
export function applyDescription(node: Node, desc: CodeDescription): Node {
  const data = node.data as Record<string, any>;
  if (!desc.ok) return { ...node, data: { ...data, codeError: desc.error ?? "?", _describedCode: data.params?.code } };
  const { inputs, outputs } = portsFromDescription(desc);
  const own = paramsFromDescription(desc);
  const params: Record<string, unknown> = { code: data.params?.code };
  for (const key of ["label", "icon", "color", "description", "libraryId", "libraryVersion"]) {
    if (data.params?.[key] !== undefined) params[key] = data.params[key];
  }
  for (const p of own) params[p.key] = data.params?.[p.key] ?? p.default;
  return {
    ...node,
    data: { ...data, inputs, outputs, paramDefs: [...own, CODE_PARAM], params, codeError: undefined,
      _describedCode: data.params?.code },
  };
}

/** Kanten an Ports, die es nicht mehr gibt, fallen weg. */
export function pruneEdges(edges: Edge[], nodes: Node[]): Edge[] {
  const ports = new Map(nodes.map((n) => {
    const d = n.data as Record<string, any>;
    return [n.id, {
      ins: new Set<string>((d.inputs ?? []).map((p: PortDef) => p.id)),
      outs: new Set<string>((d.outputs ?? []).map((p: PortDef) => p.id)),
    }];
  }));
  return edges.filter((e) => {
    const s = ports.get(e.source), t = ports.get(e.target);
    if (!s || !t) return false;
    if (e.sourceHandle && s.outs.size && !s.outs.has(e.sourceHandle)) return false;
    if (e.targetHandle && t.ins.size && !t.ins.has(e.targetHandle)) return false;
    return true;
  });
}

// ── Bibliothek <-> Leinwand ─────────────────────────────────────────────────

/** Ein Bibliotheks-Node als Leinwand-Node. `base` ist die code- bzw. block-Definition. */
export function instanceFromRecord(rec: CustomNodeRecord, base: NodeDef, id: string,
  position: { x: number; y: number }): Node {
  const def = rec.definition;
  const look = { label: rec.name, icon: def.icon, color: def.color, description: def.description ?? "" };
  let inputs = base.inputs, outputs = base.outputs, paramDefs = base.params;
  const params: Record<string, unknown> = {};
  if (rec.kind === "code") {
    inputs = def.inputs?.length ? def.inputs : base.inputs;
    outputs = def.outputs?.length ? def.outputs : base.outputs;
    paramDefs = [...(def.params ?? []), CODE_PARAM];
    for (const p of def.params ?? []) params[p.key] = def.values?.[p.key] ?? p.default;
    params.code = def.code ?? DEFAULT_CODE;
  } else {
    for (const p of base.params) params[p.key] = p.default;
    params.graph = def.graph;
    params.repeat = def.repeat ?? 1;
  }
  return {
    id,
    type: "synapseNode",
    position,
    data: {
      _def: base,
      nodeType: base.type,
      label: rec.name,
      category: base.category,
      icon: base.icon,
      color: def.color,
      inputs,
      outputs,
      paramDefs,
      params: { ...params, ...look, libraryId: rec.id, libraryVersion: rec.version },
      ...(rec.kind === "code" && def.inputs?.length ? { _describedCode: params.code } : {}),
    },
  };
}

/** Neue Bibliotheks-Version uebernehmen; gleichnamige Einstellungen behalten ihren Wert. */
export function updateFromRecord(node: Node, rec: CustomNodeRecord, base: NodeDef): Node {
  const fresh = instanceFromRecord(rec, base, node.id, node.position);
  const old = (node.data as any).params ?? {};
  const params = { ...(fresh.data as any).params };
  for (const p of (fresh.data as any).paramDefs as ParamDef[]) {
    if (p.key !== "code" && old[p.key] !== undefined) params[p.key] = old[p.key];
  }
  return { ...node, data: { ...fresh.data, params } };
}

export function isOutdated(node: Node, records: CustomNodeRecord[]): boolean {
  const p = (node.data as any)?.params ?? {};
  if (!p.libraryId) return false;
  const rec = records.find((r) => r.id === p.libraryId);
  return Boolean(rec && rec.version > Number(p.libraryVersion ?? 0));
}

/** Ein Code-Node oder Block von der Leinwand als Entwurf fuer die Bibliothek. */
export function draftFromNode(node: Node): CustomNodeInput | null {
  const kind = typeOf(node);
  if (kind !== "code" && kind !== "block") return null;
  const d = node.data as Record<string, any>;
  const p = d.params ?? {};
  const look = {
    icon: (p.icon as string) || (kind === "code" ? "sparkles" : "box"),
    color: (p.color as string) || CUSTOM_COLORS[0],
    description: (p.description as string) ?? "",
  };
  if (kind === "block") {
    return { id: p.libraryId ?? null, name: p.label ?? d.label ?? "Block", kind,
      definition: { ...look, graph: p.graph, repeat: Number(p.repeat ?? 1) } };
  }
  const own = ((d.paramDefs ?? []) as ParamDef[]).filter((x) => x.key !== "code");
  const values: Record<string, unknown> = {};
  for (const x of own) values[x.key] = p[x.key] ?? x.default;
  return {
    id: p.libraryId ?? null, name: p.label ?? (d.label === "Code" ? "" : d.label) ?? "", kind,
    definition: { ...look, code: p.code ?? DEFAULT_CODE, inputs: d.inputs, outputs: d.outputs, params: own, values },
  };
}

export function matchesSearch(rec: CustomNodeRecord, query: string): boolean {
  const q = query.trim().toLowerCase();
  return !q || rec.name.toLowerCase().includes(q) || (rec.definition.description ?? "").toLowerCase().includes(q);
}

/** Farbwerte fuer Rahmen und Leuchten, passend zu CATEGORY_META. */
export function colorMeta(color: string, label: string) {
  return { label, color, bg: `${color}1a`, border: `${color}59`, glow: `${color}40` };
}

// ── Tauri ───────────────────────────────────────────────────────────────────

async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

export const customNodeApi = {
  list: () => call<CustomNodeRecord[]>("list_custom_nodes"),
  save: (node: CustomNodeInput) => call<CustomNodeRecord>("save_custom_node", { node }),
  remove: (id: string) => call<void>("delete_custom_node", { id }),
  exportFile: (id: string, path: string) => call<void>("export_custom_node", { id, path }),
  readFile: (path: string) => call<CustomNodeInput>("read_custom_node_file", { path }),
  describe: (code: string, values: Record<string, unknown>, shape: number[]) =>
    call<CodeDescription>("synapse_describe_code", { code, values, shape }),
};
