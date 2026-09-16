import type { Node, Edge } from '@xyflow/react';

/**
 * Fluss-Layout für den Synapse-Canvas („Anordnen").
 *
 * Grundprinzip: der Graph fließt von links (Daten) nach rechts (Loss/Optimizer),
 * parallele Zweige liegen übereinander, und lange Netze brechen in mehrere
 * Bahnen um — wie Textzeilen, jede Bahn wieder links → rechts.
 *
 *   1. Ränge       Spalte = längster Weg vom Input (parallele Zweige teilen sich
 *                  eine Spalte).
 *   2. Bahnen      Ist das Band für die Canvas-Fläche zu lang, wird es an Stellen
 *                  mit möglichst wenigen Kanten über den Schnitt umgebrochen. Die
 *                  Bahnenzahl ergibt sich automatisch aus dem Seitenverhältnis.
 *   3. Reihenfolge Barycenter-Sweeps: jeder Knoten rutscht auf die mittlere Höhe
 *                  seiner Nachbarn → weniger Kreuzungen.
 *   4. Entspannen  Kurze Kräfte-Simulation (Abstoßung, Kanten-Federn, Anker an
 *                  der Spalte) für einen organischen Fluss.
 *   5. Aufräumen   Mit den echten (gemessenen) Knotengrößen auseinanderschieben,
 *                  bis nichts mehr überlappt.
 *
 * Alles ist deterministisch: zweimal Anordnen ergibt dasselbe Bild.
 */

const DEFAULT_W = 195;
const DEFAULT_H = 90;
const COL_PAD = 70; // horizontale Luft zwischen Spalten (für Kanten)
const ROW_GAP = 34; // vertikaler Abstand gestapelter Knoten
const LANE_GAP = 150; // Abstand zwischen Bahnen
const MIN_GAP_X = 45; // Mindestabstand beim Aufräumen
const MIN_GAP_Y = 28;
const ORIGIN = 80;
const MAX_LANES = 8;
const MIN_RANKS_PER_LANE = 12; // kürzere Bahnen lesen sich wieder wie Zickzack
const LANE_MIN_GAIN = 1.15; // Umbruch nur, wenn der Graph ≥ 15 % größer ins Bild passt
const FIT_PADDING = 0.7; // nutzbarer Anteil der Canvas-Fläche (fitView-Padding)
const FIT_MAX_ZOOM = 1.1;

export interface AutoLayoutOptions {
  /** Sichtbare Canvas-Fläche in px — steuert die automatische Bahnenzahl. */
  viewport?: { width: number; height: number };
  /** Bahnenzahl erzwingen (Tests); sonst automatisch. */
  lanes?: number;
  /** Meldet, wie angeordnet wurde (Bahnen, Spalten) — für Tests und Diagnose. */
  onPlan?: (plan: { lanes: number; ranks: number }) => void;
}

function sizeOf(nd: Node): { w: number; h: number } {
  const w = nd.measured?.width ?? nd.width ?? DEFAULT_W;
  const h = nd.measured?.height ?? nd.height ?? DEFAULT_H;
  return { w: w > 0 ? w : DEFAULT_W, h: h > 0 ? h : DEFAULT_H };
}

/**
 * Liegen Knoten aufeinander (bzw. enger als der Mindestabstand)? Wird genutzt,
 * um nach einem KI-Lauf zu entscheiden, ob der Canvas neu angeordnet wird.
 */
export function hasNodeOverlap(nodes: Node[], gap = 0): boolean {
  const boxes = nodes.map((nd) => ({ ...sizeOf(nd), x: nd.position?.x ?? 0, y: nd.position?.y ?? 0 }));
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (Math.abs(a.x - b.x) < (a.w + b.w) / 2 + gap && Math.abs(a.y - b.y) < (a.h + b.h) / 2 + gap) return true;
    }
  }
  return false;
}

export function autoLayoutNodes<N extends Node>(
  nodes: N[],
  edges: Edge[],
  options: AutoLayoutOptions = {},
): N[] {
  const n = nodes.length;
  if (n <= 1) return nodes;

  const sizes = nodes.map(sizeOf);
  const cellW = Math.min(420, Math.max(...sizes.map((s) => s.w)));
  const COL = cellW + COL_PAD;

  const index = new Map(nodes.map((nd, i) => [nd.id, i] as const));
  const outA: number[][] = Array.from({ length: n }, () => []);
  const inA: number[][] = Array.from({ length: n }, () => []);
  const edgeList: [number, number][] = [];
  for (const e of edges) {
    const s = index.get(e.source);
    const t = index.get(e.target);
    if (s === undefined || t === undefined || s === t) continue;
    outA[s].push(t);
    inA[t].push(s);
    edgeList.push([s, t]);
  }

  // ── 1. Ränge ──────────────────────────────────────────────────────────────
  // Topologische Reihenfolge (Kahn), Startknoten stabil nach bisheriger Position.
  const byPos = (a: number, b: number) => {
    const pa = nodes[a].position, pb = nodes[b].position;
    return (pa?.x ?? 0) - (pb?.x ?? 0) || (pa?.y ?? 0) - (pb?.y ?? 0) || a - b;
  };
  const workIndeg = inA.map((l) => l.length);
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
  for (let i = 0; i < n; i++) if (!seen[i]) order.push(i); // Zyklen-Reste

  const orderPos = new Array<number>(n);
  order.forEach((v, i) => { orderPos[v] = i; });
  // Längster Weg; Kanten gegen die Reihenfolge (nur bei Zyklen) zählen nicht.
  const rank = new Array<number>(n).fill(0);
  for (const u of order) {
    for (const v of outA[u]) {
      if (orderPos[v] > orderPos[u]) rank[v] = Math.max(rank[v], rank[u] + 1);
    }
  }
  const R = Math.max(...rank);

  const baseLayers: number[][] = Array.from({ length: R + 1 }, () => []);
  for (const v of order) baseLayers[rank[v]].push(v);
  for (const l of baseLayers) l.sort((a, b) => (nodes[a].position?.y ?? 0) - (nodes[b].position?.y ?? 0) || orderPos[a] - orderPos[b]);

  // ── 2. Bahnen ─────────────────────────────────────────────────────────────
  const spanAt = (c: number) => edgeList.reduce((acc, [s, t]) => acc + (rank[s] < c && rank[t] >= c ? 1 : 0), 0);

  const cutsFor = (laneCount: number): number[] => {
    const starts = [0];
    const per = (R + 1) / laneCount;
    const window = Math.max(1, Math.round(per / 4));
    for (let l = 1; l < laneCount; l++) {
      const target = Math.round(l * per);
      let best = -1, bestScore = Infinity;
      for (let c = target - window; c <= target + window; c++) {
        if (c < starts[starts.length - 1] + 2 || c > R - 1) continue;
        const score = spanAt(c) * 10 + Math.abs(c - target);
        if (score < bestScore) { bestScore = score; best = c; }
      }
      if (best < 0) break;
      starts.push(best);
    }
    return starts;
  };

  // Schritte 3–5 für eine gegebene Bahnaufteilung (starts = erste Spalte je Bahn)
  const arrange = (starts: number[]) => {
    const lane = rank.map((r) => {
      let l = 0;
      for (let i = 0; i < starts.length; i++) if (r >= starts[i]) l = i;
      return l;
    });
    const localRank = rank.map((r, i) => r - starts[lane[i]]);
    const sameLane = (a: number, b: number) => lane[a] === lane[b];

    // ── 3. Reihenfolge ────────────────────────────────────────────────────────
    // Koordinaten = Knotenmitten, jede Bahn in eigenem lokalem System um y = 0.
    const cx = localRank.map((r) => r * COL + cellW / 2);
    const cy = new Array<number>(n).fill(0);
    const stack = (l: number[]) => {
      const total = l.reduce((acc, v) => acc + sizes[v].h, 0) + Math.max(0, l.length - 1) * ROW_GAP;
      let y = -total / 2;
      for (const v of l) {
        cy[v] = y + sizes[v].h / 2;
        y += sizes[v].h + ROW_GAP;
      }
    };
    const layers = baseLayers.map((l) => l.slice());
    layers.forEach(stack);

    for (let sweep = 0; sweep < 8; sweep++) {
      const down = sweep % 2 === 0;
      for (let k = 0; k < R; k++) {
        const r = down ? k + 1 : R - 1 - k;
        const l = layers[r];
        if (l.length < 2) continue;
        const nb = down ? inA : outA;
        const key = new Map<number, number>();
        for (const v of l) {
          const ns = nb[v].filter((u) => sameLane(u, v));
          key.set(v, ns.length ? ns.reduce((acc, u) => acc + cy[u], 0) / ns.length : cy[v]);
        }
        l.sort((a, b) => key.get(a)! - key.get(b)! || orderPos[a] - orderPos[b]);
        stack(l);
      }
    }

    // ── 4. Entspannen ─────────────────────────────────────────────────────────
    const rowRef = ROW_GAP * 3;
    const iterations = n > 200 ? 120 : 260;
    const fx = new Array<number>(n);
    const fy = new Array<number>(n);
    for (let it = 0; it < iterations; it++) {
      const t = 1 - it / iterations;
      fx.fill(0);
      fy.fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (!sameLane(i, j)) continue;
          const dx = cx[j] - cx[i], dy = cy[j] - cy[i];
          const ax = COL * 1.25 - Math.abs(dx);
          const ay = (sizes[i].h + sizes[j].h) / 2 + rowRef - Math.abs(dy);
          if (ax <= 0 || ay <= 0) continue;
          const sy = dy !== 0 ? Math.sign(dy) : ((j - i) % 2 ? 1 : -1);
          const f = ay * 0.06;
          fy[i] -= f * sy; fy[j] += f * sy;
          const sx = dx >= 0 ? 1 : -1;
          fx[i] -= ax * 0.012 * sx; fx[j] += ax * 0.012 * sx;
        }
      }
      for (const [a, b] of edgeList) {
        if (!sameLane(a, b)) continue;
        const dy = cy[b] - cy[a];
        fy[a] += dy * 0.05; fy[b] -= dy * 0.05;
        // x-Feder nur zwischen Nachbarspalten — lange Skip-Kanten würden Knoten
        // sonst hinter ihre Vorgänger ziehen
        if (localRank[b] - localRank[a] !== 1) continue;
        const dx = cx[b] - cx[a] - COL;
        fx[a] += dx * 0.02; fx[b] -= dx * 0.02;
      }
      for (let i = 0; i < n; i++) {
        fx[i] += (localRank[i] * COL + cellW / 2 - cx[i]) * 0.06;
        fy[i] -= cy[i] * 0.003;
        cx[i] += Math.max(-40, Math.min(40, fx[i] * t));
        cy[i] += Math.max(-40, Math.min(40, fy[i] * t));
      }
    }

    // Fluss erzwingen: jeder Knoten liegt rechts von seinen Vorgängern (gleiche Bahn)
    const pushForward = () => {
      for (const v of order) {
        for (const u of inA[v]) {
          if (!sameLane(u, v) || orderPos[u] > orderPos[v]) continue;
          const minX = cx[u] + (sizes[u].w + sizes[v].w) / 2 + MIN_GAP_X;
          if (cx[v] < minX) cx[v] = minX;
        }
      }
    };
    pushForward();

    // Bahnen untereinander stapeln
    let offset = 0;
    for (let l = 0; l < starts.length; l++) {
      let top = Infinity, bottom = -Infinity;
      for (let i = 0; i < n; i++) {
        if (lane[i] !== l) continue;
        top = Math.min(top, cy[i] - sizes[i].h / 2);
        bottom = Math.max(bottom, cy[i] + sizes[i].h / 2);
      }
      if (!Number.isFinite(top)) continue;
      for (let i = 0; i < n; i++) if (lane[i] === l) cy[i] += offset - top;
      offset += bottom - top + LANE_GAP;
    }

    // ── 5. Aufräumen ──────────────────────────────────────────────────────────
    // Überlappende Paare entlang der kleineren Überlappung auseinanderschieben;
    // danach den Fluss erneut erzwingen, bis beides gleichzeitig gilt.
    const separate = (): boolean => {
      let moved = false;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = cx[j] - cx[i], dy = cy[j] - cy[i];
          const ox = (sizes[i].w + sizes[j].w) / 2 + MIN_GAP_X - Math.abs(dx);
          const oy = (sizes[i].h + sizes[j].h) / 2 + MIN_GAP_Y - Math.abs(dy);
          if (ox <= 0 || oy <= 0) continue;
          moved = true;
          if (oy <= ox) {
            const s = dy !== 0 ? Math.sign(dy) : 1;
            cy[i] -= (s * oy) / 2; cy[j] += (s * oy) / 2;
          } else {
            const s = dx >= 0 ? 1 : -1;
            cx[i] -= (s * ox) / 2; cx[j] += (s * ox) / 2;
          }
        }
      }
      return moved;
    };

    // Nur senkrecht entzerren — x (und damit der Fluss) bleibt unangetastet.
    const separateY = (): boolean => {
      let moved = false;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dy = cy[j] - cy[i];
          const oy = (sizes[i].h + sizes[j].h) / 2 + MIN_GAP_Y - Math.abs(dy);
          if (oy <= 0) continue;
          if ((sizes[i].w + sizes[j].w) / 2 + MIN_GAP_X - Math.abs(cx[j] - cx[i]) <= 0) continue;
          moved = true;
          const s = dy !== 0 ? Math.sign(dy) : 1;
          cy[i] -= (s * oy) / 2; cy[j] += (s * oy) / 2;
        }
      }
      return moved;
    };
    for (let round = 0; round < 6; round++) {
      let clean = true;
      for (let pass = 0; pass < 200 && separate(); pass++) clean = false;
      if (clean) break;
      pushForward();
    }
    // Zum Schluss den Fluss erzwingen und nur noch senkrecht entzerren: so ist
    // garantiert jede Kante vorwärts gerichtet UND nichts überlappt.
    pushForward();
    for (let pass = 0; pass < 400 && separateY(); pass++) { /* bis überlappungsfrei */ }

    return { cx, cy };
  };

  let result: { cx: number[]; cy: number[] };
  let lanes = 1;
  if (options.lanes && options.lanes > 0) {
    const starts = cutsFor(Math.min(options.lanes, Math.max(1, Math.floor((R + 1) / 2))));
    lanes = starts.length;
    result = arrange(starts);
  } else {
    // Jede Bahnenzahl wirklich durchrechnen und die wählen, bei der der Graph
    // am größten in die Canvas passt — aber jede Bahn bleibt eine erkennbare
    // Reihe (MIN_RANKS_PER_LANE), und umgebrochen wird nur bei spürbarem Gewinn.
    const vw = options.viewport?.width && options.viewport.width > 0 ? options.viewport.width : 1600;
    const vh = options.viewport?.height && options.viewport.height > 0 ? options.viewport.height : 1000;
    const fitOf = ({ cx, cy }: { cx: number[]; cy: number[] }) => {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) {
        x0 = Math.min(x0, cx[i] - sizes[i].w / 2); x1 = Math.max(x1, cx[i] + sizes[i].w / 2);
        y0 = Math.min(y0, cy[i] - sizes[i].h / 2); y1 = Math.max(y1, cy[i] + sizes[i].h / 2);
      }
      return Math.min(FIT_MAX_ZOOM, (vw * FIT_PADDING) / (x1 - x0), (vh * FIT_PADDING) / (y1 - y0));
    };
    result = arrange([0]);
    let bestFit = fitOf(result);
    let prevFit = bestFit;
    const maxLanes = Math.min(MAX_LANES, Math.floor((R + 1) / MIN_RANKS_PER_LANE));
    for (let l = 2; l <= maxLanes; l++) {
      const starts = cutsFor(l);
      if (starts.length !== l) break;
      const cand = arrange(starts);
      const fit = fitOf(cand);
      if (fit > bestFit * LANE_MIN_GAIN) { bestFit = fit; result = cand; lanes = l; }
      if (fit < prevFit) break; // ab hier wird es nur noch zu hoch
      prevFit = fit;
    }
  }
  options.onPlan?.({ lanes, ranks: R + 1 });
  const { cx, cy } = result;

  let minX = Infinity, minY = Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, cx[i] - sizes[i].w / 2);
    minY = Math.min(minY, cy[i] - sizes[i].h / 2);
  }

  return nodes.map((nd, i) => ({
    ...nd,
    position: {
      x: Math.round(cx[i] - sizes[i].w / 2 - minX + ORIGIN),
      y: Math.round(cy[i] - sizes[i].h / 2 - minY + ORIGIN),
    },
  }));
}
