// -----------------------------------------------------------------------------
// Synapse Agent Tools — Executor with internal mutable state
//
// KEY DESIGN: The executor keeps its OWN _nodes/_edges copy.
// Sequential tool calls within a batch see each other's changes immediately,
// without waiting for React re-renders.
// -----------------------------------------------------------------------------

export type AgentStep = {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  resultData?: unknown;
  error?: string;
  status: 'pending' | 'running' | 'success' | 'error';
};

export type AgentToolExecutor = (
  tool: string,
  args: Record<string, unknown>
) => Promise<{ success: boolean; data?: unknown; error?: string }>;

export type GraphMutationEvent = {
  type: 'node' | 'edge' | 'param' | 'remove';
  nodeId?: string;
  edgeId?: string;
};

export type ToolExecutorContext = {
  nodes: any[];
  edges: any[];
  setNodes: (nodes: any[]) => void;
  setEdges: (edges: any[]) => void;
  nodeDefinitions: any[];
  /** Fired after each visual sync so the canvas can animate (design layer only). */
  onGraphMutation?: (event: GraphMutationEvent) => void;
};

export type ToolExecutorHandle = {
  execute: AgentToolExecutor;
  /** Get current nodes/edges (reflects all mutations so far) */
  getState: () => { nodes: any[]; edges: any[] };
};

// --- Tool list for system prompt (plan phase only needs action tools) ---------

export const PLAN_TOOLS = `
add_node(nodeType*, id, position): Add node. nodeType must be exact string from the types list.
add_edge(source*, target*, sourceHandle, targetHandle): Connect nodes by ID. Use exact port IDs from the port reference.
remove_node(nodeId*): Remove a node and all its edges.
remove_edge(edgeId | source+target): Remove a connection.
set_param(nodeId*, key*, value*): Set a node parameter by exact key name.
set_label(nodeId*, label*): Rename a node.
move_node(nodeId*, position*): Move node to {x,y} position.
done(summary*): Signal task complete. MUST be last item.
`.trim();

// --- Position parser ----------------------------------------------------------
// Handles: {x:100,y:200}, "(100,200)", "100,200", [100,200]

export function parsePosition(raw: unknown): { x: number; y: number } | null {
  if (!raw) return null;
  if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) {
    const r = raw as Record<string, unknown>;
    const x = Number(r.x ?? r.X ?? 0);
    const y = Number(r.y ?? r.Y ?? 0);
    if (!isNaN(x) && !isNaN(y)) return { x, y };
  }
  if (Array.isArray(raw) && raw.length >= 2) {
    const x = Number(raw[0]); const y = Number(raw[1]);
    if (!isNaN(x) && !isNaN(y)) return { x, y };
  }
  if (typeof raw === 'string') {
    const nums = raw.match(/-?\d+(\.\d+)?/g);
    if (nums && nums.length >= 2) {
      return { x: parseFloat(nums[0]), y: parseFloat(nums[1]) };
    }
  }
  return null;
}

// --- Executor ----------------------------------------------------------------

let _edgeCounter = 0;
const VISUAL_MUTATION_DELAY_MS = 320;

const waitForVisualMutation = () =>
  new Promise((resolve) => setTimeout(resolve, VISUAL_MUTATION_DELAY_MS));

export function createToolExecutor(ctx: ToolExecutorContext): ToolExecutorHandle {
  // Mutable internal copies — updates are visible within the same batch
  let _nodes: any[] = [...ctx.nodes];
  let _edges: any[] = [...ctx.edges];

  // Sync to React (called after each tool execution)
  const sync = () => {
    ctx.setNodes([..._nodes]);
    ctx.setEdges([..._edges]);
  };

  const execute: AgentToolExecutor = async (tool, args) => {
    try {
      switch (tool) {

        case 'add_node': {
          const nodeType = args.nodeType as string;
          const customId = args.id as string | undefined;

          // Position: support {position:{x,y}}, top-level {x,y}, or {args:{x,y}}
          const rawPos = args.position
            ?? (args.x != null && args.y != null ? { x: args.x, y: args.y } : null)
            ?? null;

          const def = ctx.nodeDefinitions.find((d: any) => d.type === nodeType);
          if (!def) return { success: false, error: `Unknown nodeType: ${nodeType}. Valid: ${ctx.nodeDefinitions.map((d:any)=>d.type).join(',')}` };

          const nodeId = customId ?? `${def.type}-${Date.now()}-${Math.random().toString(36).slice(2,5)}`;

          // Node already exists with same ID — idempotent, treat as success
          if (_nodes.find((n: any) => n.id === nodeId)) {
            return { success: true, data: { nodeId, skipped: true } };
          }

          const pos = parsePosition(rawPos) ?? {
            x: 150 + (_nodes.length % 5) * 220,
            y: 120 + Math.floor(_nodes.length / 5) * 160,
          };

          const params: Record<string, unknown> = {};
          (def.params ?? []).forEach((p: any) => (params[p.key] = p.default));

          _nodes = [..._nodes, {
            id: nodeId,
            type: 'synapseNode',
            position: pos,
            data: {
              _def: def,
              nodeType: def.type,
              label: def.label,
              category: def.category,
              icon: def.icon,
              color: def.color,
              inputs: def.inputs ?? [],
              outputs: def.outputs ?? [],
              paramDefs: def.params ?? [],
              params,
              _sparkle: Date.now(),
            },
          }];
          sync();
          ctx.onGraphMutation?.({ type: 'node', nodeId });
          await waitForVisualMutation();
          return { success: true, data: { nodeId } };
        }

        case 'remove_node': {
          const nodeId = args.nodeId as string;
          if (!_nodes.find((n: any) => n.id === nodeId)) return { success: false, error: `Node not found: ${nodeId}` };
          _nodes = _nodes.filter((n: any) => n.id !== nodeId);
          _edges = _edges.filter((e: any) => e.source !== nodeId && e.target !== nodeId);
          sync();
          await waitForVisualMutation();
          return { success: true, data: {} };
        }

        case 'set_param': {
          const { nodeId, key, value } = args as { nodeId: string; key: string; value: unknown };
          const node = _nodes.find((n: any) => n.id === nodeId);
          if (!node) return { success: false, error: `Node not found: ${nodeId}` };
          _nodes = _nodes.map((n: any) =>
            n.id === nodeId ? { ...n, data: { ...n.data, params: { ...n.data.params, [key as string]: value } } } : n
          );
          sync();
          await waitForVisualMutation();
          return { success: true, data: {} };
        }

        case 'set_label': {
          const { nodeId, label } = args as { nodeId: string; label: string };
          if (!_nodes.find((n: any) => n.id === nodeId)) return { success: false, error: `Node not found: ${nodeId}` };
          _nodes = _nodes.map((n: any) => n.id === nodeId ? { ...n, data: { ...n.data, label } } : n);
          sync();
          await waitForVisualMutation();
          return { success: true, data: {} };
        }

        case 'add_edge': {
          const source = args.source as string;
          const target = args.target as string;
          const rawSH = args.sourceHandle;
          const rawTH = args.targetHandle;

          // Normalize handle IDs:
          // - Strip node-id prefix: "dense-2.out" → "out", "loss-1.pred" → "pred"
          // - If no dot but value looks like a node-id (contains "-" or matches a node ID),
          //   drop it entirely so auto-resolve kicks in
          const knownNodeIds = new Set(_nodes.map((n: any) => n.id));
          const stripPrefix = (raw: unknown): string | undefined => {
            if (raw == null || raw === '') return undefined;
            const s = String(raw);
            const dotIdx = s.lastIndexOf('.');
            const result = dotIdx >= 0 ? s.slice(dotIdx + 1) : s;
            // If it looks like a node ID (matches an existing node), drop it
            if (knownNodeIds.has(result)) return undefined;
            return result;
          };

          let sourceHandle = stripPrefix(rawSH);
          let targetHandle = stripPrefix(rawTH);

          const srcNode = _nodes.find((n: any) => n.id === source);
          const tgtNode = _nodes.find((n: any) => n.id === target);
          if (!srcNode) return { success: false, error: `Source not found: ${source}` };
          if (!tgtNode) return { success: false, error: `Target not found: ${target}` };

          // Validate sourceHandle — fall back to first output if invalid/missing
          const srcOutputs: any[] = srcNode.data?.outputs ?? srcNode.data?._def?.outputs ?? [];
          if (srcOutputs.length > 0) {
            const validSrc = sourceHandle && srcOutputs.find((p: any) => p.id === sourceHandle);
            if (!validSrc) sourceHandle = srcOutputs[0].id;
          } else {
            sourceHandle = undefined; // no outputs defined
          }

          // Validate targetHandle — fall back to first input if invalid/missing
          const tgtInputs: any[] = tgtNode.data?.inputs ?? tgtNode.data?._def?.inputs ?? [];
          if (tgtInputs.length > 0) {
            const validTgt = targetHandle && tgtInputs.find((p: any) => p.id === targetHandle);
            if (!validTgt) targetHandle = tgtInputs[0].id;
          } else {
            targetHandle = undefined; // no inputs defined
          }

          // Reject self-loops — a node cannot connect to itself
          if (source === target) {
            return { success: true, data: { skipped: true, reason: 'self-loop not allowed' } };
          }

          // Reject edges TO nodes with no input ports (optimizer, etc.) — unfixable
          if (tgtInputs.length === 0) {
            // Silent success — not an error worth retrying, just skip
            return { success: true, data: { skipped: true, reason: `${target} has no input ports` } };
          }

          // Reject edges FROM nodes with no output ports (output_node, etc.)
          if (srcOutputs.length === 0) {
            return { success: true, data: { skipped: true, reason: `${source} has no output ports` } };
          }

          const edgeId = `e-${source}-${target}-${Date.now()}-${++_edgeCounter}`;

          // Skip duplicate edge (same source+target+handles)
          const duplicate = _edges.find((e: any) =>
            e.source === source && e.target === target &&
            (e.sourceHandle ?? '') === (sourceHandle ?? '') &&
            (e.targetHandle ?? '') === (targetHandle ?? '')
          );
          if (duplicate) return { success: true, data: { edgeId: duplicate.id, skipped: true } };
          _edges = [..._edges, {
            id: edgeId, source, target,
            sourceHandle: sourceHandle ?? undefined,
            targetHandle: targetHandle ?? undefined,
            animated: true,
            style: { stroke: '#a78bfa', strokeWidth: 1.5 },
          }];
          sync();
          ctx.onGraphMutation?.({ type: 'edge', edgeId, nodeId: target });
          await waitForVisualMutation();
          return { success: true, data: { edgeId } };
        }

        case 'remove_edge': {
          const edgeId = args.edgeId as string | undefined;
          const source = args.source as string | undefined;
          const target = args.target as string | undefined;
          let removed = false;
          if (edgeId) {
            if (_edges.find((e: any) => e.id === edgeId)) { _edges = _edges.filter((e: any) => e.id !== edgeId); removed = true; }
          } else if (source && target) {
            const before = _edges.length;
            _edges = _edges.filter((e: any) => !(e.source === source && e.target === target));
            removed = _edges.length < before;
          }
          if (!removed) return { success: false, error: 'Edge not found' };
          sync();
          await waitForVisualMutation();
          return { success: true, data: {} };
        }

        case 'move_node': {
          const nodeId = args.nodeId as string;
          const pos = parsePosition(args.position);
          if (!pos) return { success: false, error: 'Invalid position' };
          if (!_nodes.find((n: any) => n.id === nodeId)) return { success: false, error: `Node not found: ${nodeId}` };
          _nodes = _nodes.map((n: any) => n.id === nodeId ? { ...n, position: pos } : n);
          sync();
          await waitForVisualMutation();
          return { success: true, data: {} };
        }

        default:
          return { success: false, error: `Unknown tool: ${tool}` };
      }
    } catch (e: any) {
      return { success: false, error: String(e?.message ?? e) };
    }
  };

  return {
    execute,
    getState: () => ({ nodes: _nodes, edges: _edges }),
  };
}
