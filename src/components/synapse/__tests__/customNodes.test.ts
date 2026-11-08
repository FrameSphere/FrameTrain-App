import { describe, expect, it } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import { NODE_DEFINITIONS } from "../nodeTypes";
import {
  applyDescription, draftFromNode, instanceFromRecord, isOutdated, pruneEdges, updateFromRecord,
  type CodeDescription, type CustomNodeRecord,
} from "../customNodes";

const codeDef = NODE_DEFINITIONS.find((d) => d.type === "code")!;
const blockDef = NODE_DEFINITIONS.find((d) => d.type === "block")!;

const FUSION: CodeDescription = {
  ok: true,
  inputs: [{ id: "in", label: "x", optional: false }, { id: "skip", label: "skip", optional: true }],
  outputs: [{ id: "out" }, { id: "gate" }],
  params: [
    { key: "hidden", default: 8, kind: "int", options: [] },
    { key: "act", default: "gelu", kind: "select", options: ["gelu", "relu"] },
    { key: "bias", default: true, kind: "boolean", options: [] },
  ],
};

function record(version = 1, code = "class Node: ..."): CustomNodeRecord {
  return {
    id: "cn_1", name: "Fusion", kind: "code", version, updated_at: 0,
    definition: {
      icon: "merge", color: "#2dd4bf", description: "Zwei Zweige", code,
      inputs: [{ id: "in", label: "x", portType: "tensor" }, { id: "skip", label: "skip?", portType: "tensor" }],
      outputs: [{ id: "out", label: "out", portType: "tensor" }],
      params: [{ key: "hidden", label: "hidden", type: "number", step: 1, default: 8 }],
      values: { hidden: 16 },
    },
  };
}

function codeNode(params: Record<string, unknown> = {}): Node {
  return {
    id: "c", type: "synapseNode", position: { x: 0, y: 0 },
    data: { _def: codeDef, label: "Code", category: "layer", inputs: codeDef.inputs, outputs: codeDef.outputs,
      paramDefs: codeDef.params, params: { code: "x", ...params } },
  };
}

describe("eigene Nodes", () => {
  it("Code bestimmt Ports und Einstellungen, Werte bleiben", () => {
    const n = applyDescription(codeNode({ hidden: 32, gone: 1 }), FUSION);
    const d = n.data as any;
    expect(d.inputs.map((p: any) => [p.id, p.label])).toEqual([["in", "x"], ["skip", "skip?"]]);
    expect(d.outputs.map((p: any) => p.id)).toEqual(["out", "gate"]);
    expect(d.paramDefs.map((p: any) => [p.key, p.type])).toEqual([
      ["hidden", "number"], ["act", "select"], ["bias", "boolean"], ["code", "code"],
    ]);
    expect(d.params).toEqual({ code: "x", hidden: 32, act: "gelu", bias: true });
  });

  it("Fehler im Code steht am Node, Ports bleiben", () => {
    const n = applyDescription(codeNode(), { ok: false, error: "Zeile 2: invalid syntax" });
    expect((n.data as any).codeError).toBe("Zeile 2: invalid syntax");
    expect((n.data as any).inputs).toEqual(codeDef.inputs);
  });

  it("Kanten an verschwundenen Ports fallen weg", () => {
    const c = applyDescription(codeNode(), FUSION);
    const other: Node = { ...codeNode(), id: "o", data: { ...(codeNode().data as any) } };
    const edges: Edge[] = [
      { id: "1", source: "o", target: "c", targetHandle: "skip" },
      { id: "2", source: "o", target: "c", targetHandle: "weg" },
      { id: "3", source: "c", target: "o", sourceHandle: "gate", targetHandle: "in" },
      { id: "4", source: "c", target: "x" },
    ];
    expect(pruneEdges(edges, [c, other]).map((e) => e.id)).toEqual(["1", "3"]);
  });

  it("Bibliotheks-Node wird ein Code-Node mit Kopie der Definition", () => {
    const n = instanceFromRecord(record(), codeDef, "code-1", { x: 5, y: 6 });
    const d = n.data as any;
    expect(d._def.type).toBe("code");
    expect(d.label).toBe("Fusion");
    expect(d.inputs).toHaveLength(2);
    expect(d.params).toMatchObject({ code: "class Node: ...", hidden: 16, color: "#2dd4bf", icon: "merge",
      libraryId: "cn_1", libraryVersion: 1 });
  });

  it("Block aus der Bibliothek bringt Graph und Wiederholung mit", () => {
    const rec: CustomNodeRecord = { ...record(), kind: "block",
      definition: { icon: "box", color: "#fb7185", description: "", graph: { nodes: [{ id: "a" }], edges: [] }, repeat: 6 } };
    const d = instanceFromRecord(rec, blockDef, "b", { x: 0, y: 0 }).data as any;
    expect(d.params).toMatchObject({ repeat: 6, graph: { nodes: [{ id: "a" }] }, role: "" });
    expect(d.outputs).toEqual(blockDef.outputs);
  });

  it("Update nur bei neuerer Version, eigene Einstellungen bleiben", () => {
    const placed = instanceFromRecord(record(1), codeDef, "c", { x: 0, y: 0 });
    (placed.data as any).params.hidden = 99;
    expect(isOutdated(placed, [record(1)])).toBe(false);
    expect(isOutdated(placed, [record(2)])).toBe(true);
    expect(isOutdated(codeNode(), [record(2)])).toBe(false);
    const next = updateFromRecord(placed, record(2, "neu"), codeDef);
    expect((next.data as any).params).toMatchObject({ code: "neu", hidden: 99, libraryVersion: 2 });
    expect(next.position).toEqual(placed.position);
  });

  it("Entwurf aus Code-Node und Block", () => {
    const fromCode = draftFromNode(applyDescription(codeNode({ hidden: 4 }), FUSION))!;
    expect(fromCode.kind).toBe("code");
    expect(fromCode.id).toBeNull();
    expect(fromCode.definition.values).toEqual({ hidden: 4, act: "gelu", bias: true });
    expect(fromCode.definition.outputs?.map((p) => p.id)).toEqual(["out", "gate"]);
    const placed = instanceFromRecord(record(3), codeDef, "c", { x: 0, y: 0 });
    expect(draftFromNode(placed)!.id).toBe("cn_1");
    expect(draftFromNode(placed)!.name).toBe("Fusion");
    const block: Node = { id: "b", position: { x: 0, y: 0 },
      data: { _def: blockDef, label: "Block", params: { graph: { nodes: [] }, repeat: 3 } } };
    expect(draftFromNode(block)!.definition).toMatchObject({ repeat: 3, icon: "box" });
    const dense: Node = { id: "d", position: { x: 0, y: 0 }, data: { _def: NODE_DEFINITIONS[0], params: {} } };
    expect(draftFromNode(dense)).toBeNull();
  });
});
