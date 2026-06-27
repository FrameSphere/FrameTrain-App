/**
 * Shape error → user guide + agent prompt (DIAGNOSTIC MODE).
 */

import type { Node, Edge } from "@xyflow/react";
import {
  type ValidationError,
  ValidationErrorType,
  getSynapseNodeType,
  formatValidationError,
} from "../graph-shape-validation";

export type ShapeIssueSource =
  | "connection"
  | "graph"
  | "training"
  | "runtime";

export interface AffectedNodeInfo {
  id: string;
  label: string;
  type: string;
  role: "source" | "target" | "both";
}

export function collectAffectedNodeIds(errors: ValidationError[]): Set<string> {
  const ids = new Set<string>();
  for (const e of errors) {
    if (e.severity !== "error") continue;
    if (e.sourceNodeId) ids.add(e.sourceNodeId);
    if (e.targetNodeId) ids.add(e.targetNodeId);
  }
  return ids;
}

export function getAffectedNodes(
  errors: ValidationError[],
  nodes: Node[]
): AffectedNodeInfo[] {
  const roles = new Map<string, "source" | "target" | "both">();
  for (const e of errors) {
    if (e.severity !== "error") continue;
    if (e.sourceNodeId) {
      const prev = roles.get(e.sourceNodeId);
      roles.set(
        e.sourceNodeId,
        e.targetNodeId && prev === "target" ? "both" : prev ?? "source"
      );
    }
    if (e.targetNodeId) {
      const prev = roles.get(e.targetNodeId);
      roles.set(
        e.targetNodeId,
        e.sourceNodeId && prev === "source" ? "both" : prev ?? "target"
      );
    }
  }

  return [...roles.entries()].map(([id, role]) => {
    const n = nodes.find((x) => x.id === id);
    const d = (n?.data ?? {}) as Record<string, unknown>;
    const def = d._def as { label?: string; type?: string } | undefined;
    return {
      id,
      label: String(d.label ?? def?.label ?? id),
      type: getSynapseNodeType(n!),
      role,
    };
  });
}

export function buildShapeUserGuide(
  errors: ValidationError[],
  nodes: Node[],
  source: ShapeIssueSource
): string {
  const errs = errors.filter((e) => e.severity === "error");
  const affected = getAffectedNodes(errs, nodes);
  const lines: string[] = [];

  const sourceLabel: Record<ShapeIssueSource, string> = {
    connection: "Verbindung blockiert",
    graph: "Graph vor Training ungültig",
    training: "Training — Shape-Fehler",
    runtime: "Laufzeit — Tensor-Shape",
  };

  lines.push(`**${sourceLabel[source]}**`);
  if (affected.length > 0) {
    lines.push("");
    lines.push("**Betroffene Nodes** (im Canvas rot/orange markiert):");
    affected.forEach((a) => {
      const roleDe =
        a.role === "source"
          ? "liefert falsche Ausgabe"
          : a.role === "target"
            ? "erwartet anderen Input"
            : "Quelle & Ziel";
      lines.push(`• \`${a.id}\` [${a.type}] „${a.label}" — ${roleDe}`);
    });
  }

  lines.push("");
  lines.push("**Probleme:**");
  errs.forEach((e, i) => {
    lines.push(`${i + 1}. ${e.message}`);
    if (e.suggestion) lines.push(`   → ${e.suggestion}`);
    if (e.details?.paramKey) {
      lines.push(
        `   Parameter: \`${e.details.paramKey}\` — aktuell ${e.details.currentValue ?? "?"}, sollte ${e.details.expectedValue ?? "?"} sein`
      );
    }
  });

  lines.push("");
  lines.push(
    "**Manuell:** Property Panel rechts öffnen, markierte Node wählen, Parameter anpassen (z. B. `inputSize`, `inChannels`, `normalizedShape`)."
  );
  lines.push(
    "**Mit AI:** Unten steht eine Fix-Vorlage — Enter oder „Shape-Fix senden\"."
  );

  return lines.join("\n");
}

export function buildShapeAgentPrompt(
  errors: ValidationError[],
  nodes: Node[],
  edges: Edge[],
  source: ShapeIssueSource,
  extraContext?: string
): string {
  const errs = errors.filter((e) => e.severity === "error");
  const userGuide = buildShapeUserGuide(errs, nodes, source);
  const formatted = errs.map(formatValidationError).join("\n\n");

  return `[DIAGNOSTIC MODE — Shape-Fehler beheben]

${userGuide}

---
Technische Validierung:
${formatted}

${extraContext ? `\nZusatz (Runtime/Log):\n${extraContext}\n` : ""}

**Aufgabe:** Behebe alle Shape-/Dimensions-Fehler nur mit Tools (set_param, ggf. add_node reshape/flatten, remove_edge). 
Nutze exakte Node-IDs. Prüfe DenseFlow, ConvFlow, LayerNormFlow im Graph-Kontext.
Kurze Zusammenfassung am Ende welche Parameter du geändert hast.`;
}

/** Map validation + highlight roles onto React Flow nodes. */
export function applyShapeHighlightsToNodes(
  nodes: Node[],
  errors: ValidationError[]
): Node[] {
  const affected = getAffectedNodes(
    errors.filter((e) => e.severity === "error"),
    nodes
  );
  const roleMap = new Map(affected.map((a) => [a.id, a.role]));

  return nodes.map((n) => {
    const role = roleMap.get(n.id);
    const data = { ...(n.data as Record<string, unknown>) };
    if (role) {
      data._shapeErrorRole = role;
    } else {
      delete data._shapeErrorRole;
    }
    return { ...n, data };
  });
}

export function applyShapeHighlightsToEdges(
  edges: Edge[],
  errors: ValidationError[]
): Edge[] {
  const badPairs = new Set<string>();
  for (const e of errors) {
    if (e.severity !== "error" || !e.targetNodeId) continue;
    if (
      e.type === ValidationErrorType.SHAPE_MISMATCH ||
      e.type === ValidationErrorType.DIMENSION_ERROR
    ) {
      badPairs.add(`${e.sourceNodeId}→${e.targetNodeId}`);
    }
  }

  return edges.map((edge) => {
    const key = `${edge.source}→${edge.target}`;
    if (!badPairs.has(key)) {
      const style = edge.style as Record<string, unknown> | undefined;
      if (!style?._shapeError) return edge;
      const { _shapeError: _, ...rest } = style;
      return { ...edge, style: { ...rest, stroke: "#a78bfa80", strokeWidth: 1.5 } };
    }
    return {
      ...edge,
      animated: true,
      style: {
        ...(edge.style as object),
        stroke: "#f87171",
        strokeWidth: 2.5,
        _shapeError: true,
      },
    };
  });
}

export function clearShapeHighlights(nodes: Node[], edges: Edge[]): {
  nodes: Node[];
  edges: Edge[];
} {
  return {
    nodes: nodes.map((n) => {
      const data = { ...(n.data as Record<string, unknown>) };
      delete data._shapeErrorRole;
      return { ...n, data };
    }),
    edges: applyShapeHighlightsToEdges(edges, []),
  };
}

export function validationErrorsFromRuntimeDiag(
  diag: Record<string, unknown>,
  nodes: Node[]
): ValidationError[] {
  const actual = Number(diag.actual_output_features);
  const expected = Number(diag.expected_input_features);
  const raw = String(diag.raw_error ?? "");

  const denseNode = nodes.find((n) => {
    const d = n.data as Record<string, unknown>;
    const params = (d.params ?? {}) as Record<string, unknown>;
    const t = getSynapseNodeType(n);
    return t === "dense" && Number(params.inputSize) === expected;
  });

  return [
    {
      type: ValidationErrorType.DIMENSION_ERROR,
      severity: "error",
      sourceNodeId: denseNode?.id ?? "unknown",
      targetNodeId: denseNode?.id,
      message: raw || `Tensor-Shape: Ausgabe ${actual} Features → erwartet ${expected} Features am Ziel`,
      suggestion: denseNode
        ? `set_param(${denseNode.id}, "inputSize", ${actual})`
        : "Dense inputSize an vorherige Layer-Ausgabe anpassen",
      details: {
        paramKey: "inputSize",
        expectedValue: actual,
        currentValue: expected,
        actual_output_features: actual,
        expected_input_features: expected,
      },
    },
  ];
}
