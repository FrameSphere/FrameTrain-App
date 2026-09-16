import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { autoLayoutNodes, hasNodeOverlap } from '../synapseLayout';

type G = { nodes: Node[]; edges: Edge[] };

function builder(): G & { add: (h?: number) => string; link: (a: string, b: string) => void; chain: (from: string | null, count: number) => string } {
  const g: G = { nodes: [], edges: [] };
  const add = (h = 90) => {
    const id = `n${g.nodes.length}`;
    // Alle Knoten starten gestapelt auf demselben Punkt — der schlimmste Fall
    g.nodes.push({ id, position: { x: 0, y: 0 }, data: {}, measured: { width: 195, height: h } });
    return id;
  };
  const link = (a: string, b: string) => { g.edges.push({ id: `${a}-${b}`, source: a, target: b }); };
  const chain = (from: string | null, count: number) => {
    let prev = from;
    for (let i = 0; i < count; i++) {
      const id = add(i % 3 === 0 ? 130 : 90);
      if (prev) link(prev, id);
      prev = id;
    }
    return prev!;
  };
  return Object.assign(g, { add, link, chain });
}

/** Nachbau eines großen Canvas-Modells (~56 Knoten, Verzweigungen, Merges). */
function bigGraph(): G {
  const g = builder();
  const input = g.add();
  const split = g.chain(input, 2);
  let prev = g.chain(split, 5);
  const labels = g.chain(split, 1);
  const ends: string[] = [];
  for (let s = 0; s < 3; s++) {
    const a = g.chain(prev, 3);
    const b = g.chain(prev, 1);
    const add = g.add();
    g.link(a, add); g.link(b, add);
    prev = g.chain(add, 3);
    ends.push(g.chain(prev, 1));
  }
  const cat = g.add();
  ends.forEach((e) => g.link(e, cat));
  const fl = g.chain(cat, 2);
  const cat2 = g.add();
  for (let h = 0; h < 3; h++) g.link(g.chain(fl, 3), cat2);
  const out = g.chain(cat2, 1);
  const loss = g.add();
  g.link(out, loss); g.link(labels, loss);
  g.chain(loss, 3);
  g.chain(out, 2);
  return { nodes: g.nodes, edges: g.edges };
}

function rect(nd: Node) {
  return { x: nd.position.x, y: nd.position.y, w: nd.measured!.width!, h: nd.measured!.height! };
}

function overlapCount(nodes: Node[]) {
  let c = 0;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = rect(nodes[i]), b = rect(nodes[j]);
      if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) c++;
    }
  }
  return c;
}

function aspect(nodes: Node[]) {
  const rs = nodes.map(rect);
  const w = Math.max(...rs.map((r) => r.x + r.w)) - Math.min(...rs.map((r) => r.x));
  const h = Math.max(...rs.map((r) => r.y + r.h)) - Math.min(...rs.map((r) => r.y));
  return w / h;
}

describe('autoLayoutNodes', () => {
  it('lässt leere und einzelne Graphen unverändert', () => {
    expect(autoLayoutNodes([], [])).toEqual([]);
    const one: Node[] = [{ id: 'a', position: { x: 5, y: 7 }, data: {} }];
    expect(autoLayoutNodes(one, [])).toBe(one);
  });

  it('erzeugt keine Überlappungen, auch bei unterschiedlich hohen Knoten', () => {
    const g = bigGraph();
    expect(g.nodes.length).toBeGreaterThanOrEqual(50);
    const out = autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 } });
    expect(overlapCount(out)).toBe(0);
  });

  it('ist deterministisch', () => {
    const g = bigGraph();
    const a = autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 } });
    const b = autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 } });
    expect(a.map((n) => n.position)).toEqual(b.map((n) => n.position));
  });

  it('fließt in einer Bahn von links nach rechts', () => {
    const g = bigGraph();
    const out = autoLayoutNodes(g.nodes, g.edges, { lanes: 1 });
    const pos = new Map(out.map((n) => [n.id, n.position]));
    for (const e of g.edges) expect(pos.get(e.target)!.x).toBeGreaterThan(pos.get(e.source)!.x);
    expect(pos.get('n0')!.x).toBe(Math.min(...out.map((n) => n.position.x)));
  });

  it('bricht lange Netze automatisch in Bahnen um', () => {
    const g = bigGraph();
    let plan: { lanes: number; ranks: number } | null = null;
    autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 900, height: 820 }, onPlan: (p) => { plan = p; } });
    expect(plan!.lanes).toBeGreaterThan(1);
    expect(plan!.ranks / plan!.lanes).toBeGreaterThanOrEqual(12);
    const single = autoLayoutNodes(g.nodes, g.edges, { lanes: 1 });
    const auto = autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 } });
    expect(aspect(single)).toBeGreaterThan(8);
    expect(aspect(auto)).toBeLessThan(4);
  });

  it('lässt kleine Netze einzeilig', () => {
    const g = builder();
    g.chain(null, 5);
    let plan: { lanes: number } | null = null;
    autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 }, onPlan: (p) => { plan = p; } });
    expect(plan!.lanes).toBe(1);
    const out = autoLayoutNodes(g.nodes, g.edges, { viewport: { width: 1200, height: 800 } });
    const xs = out.map((n) => n.position.x);
    for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeGreaterThan(xs[i - 1]);
  });

  it('kommt mit Zyklen und losen Knoten klar', () => {
    const g = builder();
    const a = g.add(), b = g.add(), c = g.add();
    g.add(); g.add();
    g.link(a, b); g.link(b, c); g.link(c, a);
    const out = autoLayoutNodes(g.nodes, g.edges);
    expect(out).toHaveLength(5);
    expect(overlapCount(out)).toBe(0);
    for (const n of out) expect(Number.isFinite(n.position.x) && Number.isFinite(n.position.y)).toBe(true);
  });

  it('erkennt aufeinanderliegende Knoten (für das Anordnen nach einem KI-Lauf)', () => {
    const pile: Node[] = [
      { id: 'a', position: { x: 100, y: 100 }, data: {}, measured: { width: 195, height: 90 } },
      { id: 'b', position: { x: 110, y: 105 }, data: {}, measured: { width: 195, height: 90 } },
    ];
    expect(hasNodeOverlap(pile)).toBe(true);
    const tidy = autoLayoutNodes(pile, [{ id: 'e', source: 'a', target: 'b' }]);
    expect(hasNodeOverlap(tidy)).toBe(false);
    expect(hasNodeOverlap(tidy, 12)).toBe(false);
  });

  it('sieht das Raster der KI-Platzierung als frei an', () => {
    // synapseAgentTools legt Knoten ohne Position auf 300 x 220 ab
    const grid: Node[] = Array.from({ length: 12 }, (_, i) => ({
      id: 'g' + i,
      position: { x: 160 + (i % 4) * 300, y: 120 + Math.floor(i / 4) * 220 },
      data: {},
      measured: { width: 195, height: 130 },
    }));
    expect(hasNodeOverlap(grid, 12)).toBe(false);
  });
});
