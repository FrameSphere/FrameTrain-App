import type { Node, Edge } from '@xyflow/react';

/**
 * Automatisches, übersichtliches Anordnen der Canvas-Knoten.
 *
 * Neuronale Graphen sind fast immer gerichtete azyklische Graphen (DAG) mit
 * Fluss von links (Daten) nach rechts (Loss/Optimizer). Wir legen die Knoten
 * deshalb in Ebenen (Layer) an — bestimmt über den längsten Pfad von den
 * Wurzeln (Sugiyama-artig): jede Kante schiebt das Ziel mindestens eine Spalte
 * nach rechts. Innerhalb einer Spalte werden die Knoten vertikal verteilt.
 *
 * Ergebnis: klare Spalten mit genug Abstand, sodass Kanten sichtbar sind —
 * statt eines überlappenden Haufens.
 */

const COL_GAP = 340; // horizontaler Abstand zwischen Ebenen (px)
const ROW_GAP = 130; // vertikaler Abstand innerhalb einer Ebene (px)
const ORIGIN_X = 80;
const ORIGIN_Y = 80;

export function autoLayoutNodes<N extends Node>(nodes: N[], edges: Edge[]): N[] {
  if (nodes.length === 0) return nodes;

  const ids = new Set(nodes.map((n) => n.id));
  const outAdj = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  for (const n of nodes) {
    outAdj.set(n.id, []);
    indeg.set(n.id, 0);
  }
  for (const e of edges) {
    if (!ids.has(e.source) || !ids.has(e.target) || e.source === e.target) continue;
    outAdj.get(e.source)!.push(e.target);
    indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
  }

  // Längster-Pfad-Layering via Kahn: layer(v) = max(layer(u)+1) über alle u→v
  const layer = new Map<string, number>();
  const workIndeg = new Map(indeg);
  const queue: string[] = [];
  for (const n of nodes) {
    if ((workIndeg.get(n.id) ?? 0) === 0) {
      layer.set(n.id, 0);
      queue.push(n.id);
    }
  }
  let qi = 0;
  while (qi < queue.length) {
    const u = queue[qi++];
    const lu = layer.get(u) ?? 0;
    for (const v of outAdj.get(u) ?? []) {
      layer.set(v, Math.max(layer.get(v) ?? 0, lu + 1));
      const d = (workIndeg.get(v) ?? 0) - 1;
      workIndeg.set(v, d);
      if (d === 0) queue.push(v);
    }
  }
  // Reste aus etwaigen Zyklen (in NN-Graphen selten): Spalte 0
  for (const n of nodes) if (!layer.has(n.id)) layer.set(n.id, 0);

  // Nach Spalten gruppieren
  const cols = new Map<number, string[]>();
  for (const n of nodes) {
    const l = layer.get(n.id) ?? 0;
    if (!cols.has(l)) cols.set(l, []);
    cols.get(l)!.push(n.id);
  }

  // Stabile Reihenfolge innerhalb einer Spalte: nach bisheriger y-Position,
  // damit sich das Layout beim wiederholten Anordnen kaum "umsortiert".
  const yById = new Map(nodes.map((n) => [n.id, n.position?.y ?? 0] as const));
  const newPos = new Map<string, { x: number; y: number }>();
  for (const l of [...cols.keys()].sort((a, b) => a - b)) {
    const col = cols.get(l)!.slice().sort((a, b) => (yById.get(a) ?? 0) - (yById.get(b) ?? 0));
    const total = (col.length - 1) * ROW_GAP;
    col.forEach((id, i) => {
      newPos.set(id, { x: ORIGIN_X + l * COL_GAP, y: ORIGIN_Y + i * ROW_GAP - total / 2 });
    });
  }

  // Alle Spalten so nach unten schieben, dass der oberste Knoten bei ORIGIN_Y liegt
  let minY = Infinity;
  for (const p of newPos.values()) if (p.y < minY) minY = p.y;
  const shift = Number.isFinite(minY) ? ORIGIN_Y - minY : 0;

  return nodes.map((n) => {
    const p = newPos.get(n.id);
    return p ? { ...n, position: { x: p.x, y: p.y + shift } } : n;
  });
}
