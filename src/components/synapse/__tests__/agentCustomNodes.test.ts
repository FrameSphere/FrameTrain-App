import { describe, expect, it } from "vitest";
import { NODE_DEFINITIONS } from "../nodeTypes";
import { createToolExecutor } from "../ai/synapseAgentTools";
import { buildSynapseGraphContext } from "../ai/synapseGraphContext";
import type { CodeDescription, CustomNodeInput, CustomNodeRecord } from "../customNodes";

const FUSION_DESC: CodeDescription = {
  ok: true,
  inputs: [{ id: "in", label: "x", optional: false }, { id: "skip", label: "skip", optional: true }],
  outputs: [{ id: "out" }, { id: "gate" }],
  params: [{ key: "hidden", default: 32, kind: "int", options: [] }, { key: "act", default: "gelu", kind: "select", options: ["gelu", "relu"] }],
};

const LIB: CustomNodeRecord = {
  id: "cn_abc", name: "Gated Fusion", kind: "code", version: 2, updated_at: 0,
  definition: {
    icon: "merge", color: "#2dd4bf", description: "Zwei Zweige mit Gate", code: "class Node: ...",
    inputs: [{ id: "in", label: "x", portType: "tensor" }, { id: "skip", label: "skip?", portType: "tensor" }],
    outputs: [{ id: "out", label: "out", portType: "tensor" }, { id: "gate", label: "gate", portType: "tensor" }],
    params: [{ key: "hidden", label: "hidden", type: "number", step: 1, default: 32 },
      { key: "act", label: "act", type: "select", options: ["gelu", "relu"], default: "gelu" }],
    values: { hidden: 32, act: "gelu" },
  },
};

function executor(extra: Partial<Parameters<typeof createToolExecutor>[0]> = {}) {
  let state = { nodes: [] as any[], edges: [] as any[] };
  const ex = createToolExecutor({
    nodes: [], edges: [], nodeDefinitions: NODE_DEFINITIONS, visualDelayMs: 0,
    setNodes: (n) => { state = { ...state, nodes: n }; }, setEdges: (e) => { state = { ...state, edges: e }; },
    customNodes: [LIB], ...extra,
  });
  return ex;
}

describe("Synapse-KI mit eigenen Nodes", () => {
  it("legt einen Bibliotheks-Node per custom:<id> oder Namen an", async () => {
    const ex = executor();
    const r = await ex.execute("add_node", { nodeType: "custom:cn_abc", id: "f1" });
    expect(r.success).toBe(true);
    expect(r.data).toMatchObject({ nodeId: "f1", inputs: ["in", "skip"], outputs: ["out", "gate"] });
    expect((await ex.execute("add_node", { nodeType: "custom:gated fusion", id: "f2" })).success).toBe(true);
    const bad = await ex.execute("add_node", { nodeType: "custom:nope" });
    expect(bad.error).toContain("custom:cn_abc (Gated Fusion)");
    const n = ex.getState().nodes[0];
    expect(n.data.params).toMatchObject({ libraryId: "cn_abc", libraryVersion: 2, hidden: 32 });
  });

  it("verdrahtet benannte Ports und lehnt falsche ab", async () => {
    const ex = executor();
    await ex.execute("add_node", { nodeType: "dense", id: "d1" });
    await ex.execute("add_node", { nodeType: "custom:cn_abc", id: "f1" });
    await ex.execute("add_node", { nodeType: "dense", id: "d2" });
    expect((await ex.execute("add_edge", { source: "d1", target: "f1", targetHandle: "skip" })).success).toBe(true);
    expect((await ex.execute("add_edge", { source: "f1", target: "d2", sourceHandle: "gate" })).success).toBe(true);
    const wrong = await ex.execute("add_edge", { source: "d1", target: "f1", targetHandle: "residual" });
    expect(wrong.success).toBe(false);
    expect(wrong.error).toContain("Inputs: in, skip");
    const edges = ex.getState().edges;
    expect(edges.map((e: any) => [e.targetHandle, e.sourceHandle])).toEqual([["skip", "out"], ["in", "gate"]]);
  });

  it("neuer Code gibt dem Code-Node sofort seine Ports", async () => {
    const seen: string[] = [];
    const ex = executor({ describeCode: async (code) => { seen.push(code); return FUSION_DESC; } });
    await ex.execute("add_node", { nodeType: "code", id: "c1" });
    const r = await ex.execute("set_param", { nodeId: "c1", key: "code", value: "class Node: ..." });
    expect(seen).toEqual(["class Node: ..."]);
    expect(r.data).toMatchObject({ inputs: ["in", "skip"], outputs: ["out", "gate"], settings: ["hidden", "act"] });
    await ex.execute("add_node", { nodeType: "dense", id: "d1" });
    expect((await ex.execute("add_edge", { source: "d1", target: "c1", targetHandle: "skip" })).success).toBe(true);
    const fail = executor({ describeCode: async () => ({ ok: false, error: "Zeile 3: invalid syntax" }) });
    await fail.execute("add_node", { nodeType: "code", id: "c1" });
    expect((await fail.execute("set_param", { nodeId: "c1", key: "code", value: "x" })).error).toBe("Code error: Zeile 3: invalid syntax");
  });

  it("speichert einen Code-Node in die Bibliothek und nutzt ihn danach", async () => {
    const saved: CustomNodeInput[] = [];
    const ex = executor({
      customNodes: [],
      describeCode: async () => FUSION_DESC,
      saveCustomNode: async (input) => { saved.push(input); return { ...LIB, id: "cn_new", name: input.name, version: 1, definition: input.definition }; },
    });
    await ex.execute("add_node", { nodeType: "code", id: "c1" });
    await ex.execute("set_param", { nodeId: "c1", key: "code", value: "class Node: ..." });
    const r = await ex.execute("save_custom_node", { nodeId: "c1", name: "Fusion", description: "zwei Eingänge" });
    expect(r.data).toEqual({ nodeType: "custom:cn_new", name: "Fusion", version: 1 });
    expect(saved[0].definition).toMatchObject({ description: "zwei Eingänge", code: "class Node: ..." });
    expect(ex.getState().nodes[0].data.params.libraryId).toBe("cn_new");
    expect((await ex.execute("add_node", { nodeType: "custom:cn_new", id: "c2" })).success).toBe(true);
    expect((await executor().execute("save_custom_node", { nodeId: "x" })).success).toBe(false);
  });

  it("Kontext zeigt Ports, Einstellungen und die Bibliothek", async () => {
    const ex = executor();
    await ex.execute("add_node", { nodeType: "custom:cn_abc", id: "f1" });
    const ctx = buildSynapseGraphContext(ex.getState().nodes, [], null, NODE_DEFINITIONS, [LIB]);
    expect(ctx).toContain("f1[Gated Fusion custom:cn_abc v2](in=x,skip?→out,gate){hidden=32,act=gelu(gelu|relu)}");
    expect(ctx).toContain('CustomNodes(1): custom:cn_abc "Gated Fusion" (in=x,skip?→out,gate){hidden=32,act=gelu(gelu|relu)} — Zwei Zweige mit Gate');
    expect(buildSynapseGraphContext([], [], null, NODE_DEFINITIONS)).not.toContain("CustomNodes");
  });
});
