import type { Node, Edge } from '@xyflow/react';

/**
 * Kompaktes, lesbares Anordnen der Canvas-Knoten — auch (und gerade) für lange,
 * fast lineare Netze.
 *
 * Erkenntnis: neuronale Netze sind oft eine lange Kette (56 Knoten / 55 Kanten).
 * Jedes rein gerichtete Layout (eine Spalte pro Schritt) wird dann zwangsläufig
 * eine endlose Reihe. Und ein Force-Layout biegt die Kette zu einem Ring.
 *
 * Lösung: SPALTEN-SERPENTINE (links → rechts). Wir bringen die Knoten in ihre
 * logische Reihenfolge (topologische Sortierung: Input zuerst, Output zuletzt)
 * und legen sie SPALTENWEISE im Zickzack ab:
 *   Spalte 0: oben → unten
 *   Spalte 1: unten → oben
 *   Spalte 2: oben → unten …
 * Die Spalten wandern von links nach rechts — das liest sich wie „Eingabe-Spalte
 * → mittlere Schichten → Ausgabe-Spalte" (Fluss links→rechts, wie ein klassisches
 * Netz-Diagramm), bleibt aber die gefaltete Pipeline. Aufeinanderfolgende Knoten
 * sind IMMER direkt benachbart (kurze, sichtbare Kanten); Verzweigungen (Loss,
 * Optimizer, Heads) sitzen direkt hinter ihrem Elternknoten.
 */

const COL_GAP = 260; // horizontaler Abstand zwischen den Spalten/Schichten (px)
const ROW_GAP = 130; // vertikaler Abstand innerhalb einer Spalte (px)
const ORIGIN = 80;

export function autoLayoutNodes<N extends Node>(nodes: N[], edges: Edge[]): N[] {
  const n = nodes.length;
  if (n <= 1) return nodes;

  const index = new Map(nodes.map((nd, i) => [nd.id, i] as const));
  const outA: number[][] = Array.from({ length: n }, () => []);
  const indeg = new Array<number>(n).fill(0);
  for (const e of edges) {
    const s = index.get(e.source);
    const t = index.get(e.target);
    if (s === undefined || t === undefined || s === t) continue;
    outA[s].push(t);
    indeg[t]++;
  }

  // Topologische Reihenfolge (Kahn). Startknoten (Input/Quellen) nach ihrer
  // bisherigen Position stabil sortieren → deterministisch.
  const byPos = (a: number, b: number) => {
    const na = nodes[a].position, nb = nodes[b].position;
    return (na?.x ?? 0) - (nb?.x ?? 0) || (na?.y ?? 0) - (nb?.y ?? 0) || a - b;
  };
  const workIndeg = indeg.slice();
  const ready: number[] = [];
  for (let i = 0; i < n; i++) if (workIndeg[i] === 0) ready.push(i);
  ready.sort(byPos);
  const order: number[] = [];
  const seen = new Array<boolean>(n).fill(false);
  while (ready.length) {
    const u = ready.shift()!;
    if (seen[u]) continue;
    seen[u] = true;
    order.push(u);
    const freed: number[] = [];
    for (const v of outA[u]) if (!seen[v] && --workIndeg[v] === 0) freed.push(v);
    freed.sort(byPos);
    ready.push(...freed);
  }
  // Zyklen-/Restknoten hinten anhängen
  for (let i = 0; i < n; i++) if (!seen[i]) order.push(i);

  // Höhe je Spalte so wählen, dass mehrere Schicht-Spalten links→rechts entstehen
  const rowsPerCol = Math.min(10, Math.max(3, Math.round(Math.sqrt(n))));

  const px = new Array<number>(n);
  const py = new Array<number>(n);
  order.forEach((id, idx) => {
    const col = Math.floor(idx / rowsPerCol);
    let row = idx % rowsPerCol;
    if (col % 2 === 1) row = rowsPerCol - 1 - row; // Zickzack: ungerade Spalten umkehren
    px[id] = ORIGIN + col * COL_GAP;
    py[id] = ORIGIN + row * ROW_GAP;
  });

  return nodes.map((nd, i) => ({
    ...nd,
    position: { x: px[i], y: py[i] },
  }));
}
