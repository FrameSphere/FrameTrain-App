import { describe, expect, it } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import { decorateEdges, decorateNodes, formatParams, formatShape, structureKey, type LiveCheckResult } from "../liveCheck";
import { groupToBlock } from "../blockGroup";
import { emptyHistory, record, redo, undo } from "../graphHistory";

const node = (id: string, params = {}, position = { x: 0, y: 0 }) =>
  ({ id, position, data: { nodeType: "dense", params } }) as unknown as Node;
const edge = (source: string, target: string) => ({ id: `${source}-${target}`, source, target }) as Edge;

const result: LiveCheckResult = {
  ok: false, params: 12, input_shape: [1, 4],
  nodes: { a: { outputs: { out: [1, 8] }, params: 40 } },
  errors: [{ node: "b", message: "passt nicht" }],
  edge_sources: { "a:in": ["__input__", "out"] },
};

describe("Live-Prüfung", () => {
  it("prüft nur neu, wenn sich die Struktur ändert", () => {
    expect(structureKey([node("a")], [])).toBe(structureKey([node("a", {}, { x: 9, y: 9 })], []));
    expect(structureKey([node("a")], [])).not.toBe(structureKey([node("a", { outputSize: 2 })], []));
  });

  it("schreibt Formen an die Kanten", () => {
    const [ab, input] = decorateEdges([edge("a", "b"), edge("loader", "a")], result);
    expect(ab.label).toBe("[B,8]");
    expect(input.label).toBe("[B,4]");
  });

  it("legt Fehler und Parameter an die Nodes", () => {
    const [a, b] = decorateNodes([node("a"), node("b")], result);
    expect((a.data as any).liveParams).toBe(40);
    expect((b.data as any).liveError).toBe("passt nicht");
  });

  it("ändert ohne Ergebnis nichts", () => {
    const edges = [edge("a", "b")];
    expect(decorateEdges(edges, null)).toBe(edges);
  });

  it("formatiert Formen und Parameter", () => {
    expect(formatShape([1, 197, 192])).toBe("[B,197,192]");
    expect(formatParams(5_333_760)).toBe("5.3 Mio.");
    expect(formatParams(512)).toBe("512");
  });
});

describe("Undo/Redo", () => {
  it("geht zurück und wieder vor", () => {
    const h = record(emptyHistory<number>(), 1);
    const back = undo(h, 2)!;
    expect(back.value).toBe(1);
    expect(redo(back.history, 1)!.value).toBe(2);
  });

  it("verwirft Redo nach neuer Änderung", () => {
    const back = undo(record(emptyHistory<number>(), 1), 2)!;
    expect(record(back.history, 1).future).toEqual([]);
  });

  it("meldet leere Stapel", () => {
    expect(undo(emptyHistory<number>(), 1)).toBeNull();
    expect(redo(emptyHistory<number>(), 1)).toBeNull();
  });
});

describe("Block zusammenfassen", () => {
  const pick = (id: string) => ({ ...node(id), selected: true }) as Node;
  const template = { id: "block-1", position: { x: 0, y: 0 }, data: { params: { repeat: 1 } } } as unknown as Node;

  it("fasst die Auswahl zusammen und verdrahtet die Ränder neu", () => {
    const r = groupToBlock([node("a"), pick("b"), pick("c"), node("d")],
      [edge("a", "b"), edge("b", "c"), edge("c", "d")], template);
    if (typeof r === "string") throw new Error(r);
    expect(r.nodes.map((n) => n.id)).toEqual(["a", "d", "block-1"]);
    expect(r.edges.map((e) => [e.source, e.target])).toEqual([["a", "block-1"], ["block-1", "d"]]);
    expect((r.block.data as any).params.graph.nodes).toHaveLength(2);
  });

  it("lehnt Auswahl mit mehreren Eingängen ab", () => {
    expect(groupToBlock([node("a"), node("x"), pick("b"), pick("c")],
      [edge("a", "b"), edge("x", "c"), edge("b", "c")], template)).toBe("oneEdge");
  });

  it("braucht mindestens zwei Nodes", () => {
    expect(groupToBlock([pick("a")], [], template)).toBe("needTwo");
  });
});
