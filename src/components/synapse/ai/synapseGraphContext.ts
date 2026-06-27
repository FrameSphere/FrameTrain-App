import type { Node, Edge } from '@xyflow/react';
import type { NodeDefinition } from '../nodeTypes';

function nodeType(node: Node): string {
  const data: any = node.data || {};
  return data?._def?.type ?? data?.nodeType ?? '?';
}

function nodeParams(node: Node): Record<string, unknown> {
  const data: any = node.data || {};
  return data?.params ?? {};
}

function buildDenseShapeContext(nodes: Node[], edges: Edge[]): string {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const lines: string[] = [];

  const denseNodes = nodes.filter((node) => nodeType(node) === 'dense');
  if (denseNodes.length > 0) {
    lines.push(
      `DenseParams: ${denseNodes.map((node) => {
        const params = nodeParams(node);
        return `${node.id}{inputSize=${params.inputSize ?? '?'},outputSize=${params.outputSize ?? '?'}}`;
      }).join(', ')}`
    );
  }

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return;

    const sourceType = nodeType(source);
    const targetType = nodeType(target);
    if (sourceType !== 'dense' || targetType !== 'dense') return;

    const sourceParams = nodeParams(source);
    const targetParams = nodeParams(target);
    const sourceOut = Number(sourceParams.outputSize);
    const targetIn = Number(targetParams.inputSize);
    const ok = Number.isFinite(sourceOut) && Number.isFinite(targetIn) && sourceOut === targetIn;

    lines.push(
      `DenseFlow: ${edge.source}.outputSize=${sourceParams.outputSize ?? '?'} → ${edge.target}.inputSize=${targetParams.inputSize ?? '?'} ${ok ? 'OK' : 'MISMATCH: set_param ' + edge.target + '.inputSize=' + (sourceParams.outputSize ?? '?')}`
    );
  });

  return lines.join('\n');
}

function outputFeatureSize(node: Node): unknown {
  const type = nodeType(node);
  const params = nodeParams(node);
  if (type === 'dense') return params.outputSize;
  if (type === 'conv2d') return params.outChannels;
  if (type === 'layernorm') return params.normalizedShape;
  if (type === 'embedding') return params.embeddingDim;
  if (type === 'lstm') return params.hiddenSize;
  return undefined;
}

function buildConvShapeContext(nodes: Node[], edges: Edge[]): string {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const convNodes = nodes.filter((node) => nodeType(node) === 'conv2d');
  const lines: string[] = [];

  if (convNodes.length > 0) {
    lines.push(
      `ConvParams: ${convNodes.map((node) => {
        const params = nodeParams(node);
        return `${node.id}{inChannels=${params.inChannels ?? '?'},outChannels=${params.outChannels ?? '?'}}`;
      }).join(', ')}`
    );
  }

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target || nodeType(target) !== 'conv2d') return;

    const sourceOut = outputFeatureSize(source);
    const targetParams = nodeParams(target);
    const sourceChannels = Number(sourceOut);
    const targetInChannels = Number(targetParams.inChannels);
    const comparable = Number.isFinite(sourceChannels) && Number.isFinite(targetInChannels);
    const ok = comparable && sourceChannels === targetInChannels;

    lines.push(
      `ConvFlow: ${edge.source}.channels=${sourceOut ?? '?'} → ${edge.target}.inChannels=${targetParams.inChannels ?? '?'} ${!comparable ? 'UNKNOWN' : ok ? 'OK' : 'MISMATCH: set_param ' + edge.target + '.inChannels=' + sourceOut}`
    );
  });

  return lines.join('\n');
}

function buildConvDenseShapeContext(nodes: Node[], edges: Edge[]): string {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const lines: string[] = [];

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return;

    const sourceType = nodeType(source);
    const targetType = nodeType(target);
    if (sourceType !== 'conv2d' || targetType !== 'dense') return;

    const sourceParams = nodeParams(source);
    const targetParams = nodeParams(target);
    const sourceChannels = Number(sourceParams.outChannels);
    const targetInputSize = Number(targetParams.inputSize);
    const comparable = Number.isFinite(sourceChannels) && Number.isFinite(targetInputSize);
    const ok = comparable && sourceChannels === targetInputSize;

    lines.push(
      `ConvDenseFlow: ${edge.source}.outChannels=${sourceParams.outChannels ?? '?'} → ${edge.target}.inputSize=${targetParams.inputSize ?? '?'} ${!comparable ? 'UNKNOWN' : ok ? 'OK' : 'MISMATCH: set_param ' + edge.target + '.inputSize=' + sourceParams.outChannels} (Dense auto-pools 4D CNN tensors to [batch, channels])`
    );
  });

  return lines.join('\n');
}

function buildNormShapeContext(nodes: Node[], edges: Edge[]): string {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const layerNormNodes = nodes.filter((node) => nodeType(node) === 'layernorm');
  const lines: string[] = [];

  if (layerNormNodes.length > 0) {
    lines.push(
      `LayerNormParams: ${layerNormNodes.map((node) => {
        const params = nodeParams(node);
        return `${node.id}{normalizedShape=${params.normalizedShape ?? '?'}}`;
      }).join(', ')}`
    );
  }

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target || nodeType(target) !== 'layernorm') return;

    const sourceOut = outputFeatureSize(source);
    const targetParams = nodeParams(target);
    const targetNorm = Number(targetParams.normalizedShape);
    const sourceOutNum = Number(sourceOut);
    const comparable = Number.isFinite(sourceOutNum) && Number.isFinite(targetNorm);
    const ok = comparable && sourceOutNum === targetNorm;

    lines.push(
      `LayerNormFlow: ${edge.source}.features=${sourceOut ?? '?'} → ${edge.target}.normalizedShape=${targetParams.normalizedShape ?? '?'} ${!comparable ? 'UNKNOWN' : ok ? 'OK' : 'MISMATCH: set_param ' + edge.target + '.normalizedShape=' + sourceOut}`
    );
  });

  return lines.join('\n');
}

function buildAttentionShapeContext(nodes: Node[], edges: Edge[]): string {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const attentionNodes = nodes.filter((node) => nodeType(node) === 'attention' || nodeType(node) === 'transformer_block');
  const lines: string[] = [];

  if (attentionNodes.length > 0) {
    lines.push(
      `AttentionParams: ${attentionNodes.map((node) => {
        const params = nodeParams(node);
        return `${node.id}{embedDim=${params.embedDim ?? '?'},numHeads=${params.numHeads ?? '?'}}`;
      }).join(', ')}`
    );
  }

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return;

    const targetType = nodeType(target);
    if (targetType !== 'attention' && targetType !== 'transformer_block') return;

    const sourceOut = outputFeatureSize(source);
    const targetParams = nodeParams(target);
    const targetEmbed = Number(targetParams.embedDim);
    const sourceOutNum = Number(sourceOut);
    const comparable = Number.isFinite(sourceOutNum) && Number.isFinite(targetEmbed);
    const ok = comparable && sourceOutNum === targetEmbed;

    lines.push(
      `AttentionFlow: ${edge.source}.features=${sourceOut ?? '?'} → ${edge.target}.embedDim=${targetParams.embedDim ?? '?'} ${!comparable ? 'UNKNOWN' : ok ? 'OK' : 'MISMATCH: set_param ' + edge.target + '.embedDim=' + sourceOut}`
    );
  });

  return lines.join('\n');
}

export function buildSynapseGraphContext(
  nodes: Node[],
  edges: Edge[],
  selectedNodeId: string | null,
  nodeDefs: NodeDefinition[],
): string {
  // ── Node list — show non-default params inline ────────────────────────────
  const nodeList = nodes.length === 0
    ? 'none'
    : nodes.map((n) => {
        const d: any = n.data || {};
        const def    = d?._def ?? null;
        const type   = def?.type ?? d?.nodeType ?? '?';
        const label  = d?.label ?? def?.label ?? type;
        const params = d?.params ?? {};

        // Collect params that differ from their default value
        const changedParams: string[] = [];
        if (def?.params) {
          for (const pDef of def.params) {
            const cur = params[pDef.key];
            if (cur !== undefined && cur !== pDef.default) {
              // Keep values short
              const val = typeof cur === 'string' && cur.length > 20
                ? cur.slice(0, 18) + '…'
                : cur;
              changedParams.push(`${pDef.key}=${val}`);
            }
          }
        }

        const labelPart = label !== def?.label ? `[${label}]` : `[${type}]`;
        const paramPart = changedParams.length > 0 ? `{${changedParams.join(',')}}` : '';
        return `${n.id}${labelPart}${paramPart}`;
      }).join(', ');

  // ── Edge list — deduplicated ──────────────────────────────────────────────
  const seenEdges = new Set<string>();
  const uniqueEdges = edges.filter((e) => {
    const sh  = (e as any).sourceHandle ?? '';
    const th  = (e as any).targetHandle ?? '';
    const key = `${e.source}.${sh}→${e.target}.${th}`;
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });

  const edgeList = uniqueEdges.length === 0
    ? 'none'
    : uniqueEdges.map((e) => {
        const sh = (e as any).sourceHandle;
        const th = (e as any).targetHandle;
        // Show as "source(handle)→target(handle)" to avoid model copying "source.handle" as a handle ID
        if (sh || th) {
          const srcPart = sh ? `${e.source}(${sh})` : e.source;
          const tgtPart = th ? `${e.target}(${th})` : e.target;
          return `${srcPart}→${tgtPart}`;
        }
        return `${e.source}→${e.target}`;
      }).join(', ');

  // ── Selected node ─────────────────────────────────────────────────────────
  let selectedStr = 'none';
  if (selectedNodeId) {
    const sel = nodes.find((n) => n.id === selectedNodeId);
    if (sel) {
      const d: any = sel.data || {};
      const params = d?.params ?? {};
      selectedStr = Object.keys(params).length > 0
        ? `${sel.id} params:${JSON.stringify(params)}`
        : sel.id;
    }
  }

  const typeNames = nodeDefs.map((d) => d.type).join(',');

  const edgeNote = uniqueEdges.length < edges.length
    ? ` (${edges.length - uniqueEdges.length} dupes removed)`
    : '';

  const denseShapeContext = buildDenseShapeContext(nodes, uniqueEdges);
  const convShapeContext = buildConvShapeContext(nodes, uniqueEdges);
  const convDenseShapeContext = buildConvDenseShapeContext(nodes, uniqueEdges);
  const normShapeContext = buildNormShapeContext(nodes, uniqueEdges);
  const attentionShapeContext = buildAttentionShapeContext(nodes, uniqueEdges);

  return [
    `Nodes(${nodes.length}): ${nodeList}`,
    `Edges(${uniqueEdges.length}${edgeNote}): ${edgeList}`,
    denseShapeContext,
    convShapeContext,
    convDenseShapeContext,
    normShapeContext,
    attentionShapeContext,
    `Selected: ${selectedStr}`,
    `Types(${nodeDefs.length}): ${typeNames}`,
  ].filter(Boolean).join('\n');
}
