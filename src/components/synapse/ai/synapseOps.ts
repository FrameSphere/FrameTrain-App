export type SynapseOp =
  | { op: 'add_node'; nodeType: string; id?: string; position?: { x: number; y: number } }
  | { op: 'remove_node'; nodeId: string }
  | { op: 'set_param'; nodeId: string; key: string; value: unknown }
  | { op: 'set_label'; nodeId: string; label: string }
  | { op: 'move_node'; nodeId: string; position: { x: number; y: number } }
  | { op: 'add_edge'; source: string; sourceHandle?: string; target: string; targetHandle?: string }
  | { op: 'remove_edge'; edgeId?: string; source?: string; target?: string; sourceHandle?: string; targetHandle?: string };

const FENCE = 'synapse_ops';

export function parseSynapseOps(text: string): { ops: SynapseOp[]; raw: string | null; error?: string } {
  const match = text.match(new RegExp('```' + FENCE + '\\s*([\\s\\S]*?)\\s*```', 'm'));
  if (!match) return { ops: [], raw: null, error: `Kein \`\`\`${FENCE}\`\`\` Block gefunden.` };
  const raw = match[1].trim();
  try {
    const parsed = JSON.parse(raw);
    const ops = Array.isArray(parsed) ? parsed : (parsed?.ops && Array.isArray(parsed.ops) ? parsed.ops : null);
    if (!ops) return { ops: [], raw, error: 'JSON muss ein Array sein (oder { "ops": [...] }).' };
    return { ops: ops as SynapseOp[], raw };
  } catch (e: any) {
    return { ops: [], raw, error: `JSON parse error: ${String(e?.message ?? e)}` };
  }
}

export function formatOpsPreview(ops: SynapseOp[]): string[] {
  return ops.map((op) => {
    switch (op.op) {
      case 'add_node': return `add_node ${op.nodeType}${op.id ? ` (id=${op.id})` : ''}`;
      case 'remove_node': return `remove_node ${op.nodeId}`;
      case 'set_param': return `set_param ${op.nodeId}.${op.key} = ${JSON.stringify(op.value)}`;
      case 'set_label': return `set_label ${op.nodeId} = ${JSON.stringify(op.label)}`;
      case 'move_node': return `move_node ${op.nodeId} -> (${op.position.x}, ${op.position.y})`;
      case 'add_edge': return `add_edge ${op.source}${op.sourceHandle ? `:${op.sourceHandle}` : ''} -> ${op.target}${op.targetHandle ? `:${op.targetHandle}` : ''}`;
      case 'remove_edge': return `remove_edge ${op.edgeId ?? `${op.source ?? '?'} -> ${op.target ?? '?'}`}`;
      default: return `op ${(op as any).op ?? '?'}`;
    }
  });
}

