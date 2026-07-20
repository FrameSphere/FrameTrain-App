// -----------------------------------------------------------------------------
// Synapse AI Agent — Hybrid C (Plan → Execute → Fix)
// -----------------------------------------------------------------------------

import { callAI } from '../../../ai/aiClient';
import type { AISettings } from '../../../contexts/AISettingsContext';
import { TOKEN_BUDGET_CONFIG } from '../../../contexts/AISettingsContext';
import type { ChatMessage } from '../../../ai/aiClient';
import { PLAN_TOOLS } from './synapseAgentTools';
import type { AgentStep, ToolExecutorHandle } from './synapseAgentTools';
import { debugLogRequest } from './synapseDebugLogger';

export type { AgentStep } from './synapseAgentTools';

// --- Types -------------------------------------------------------------------

export type AgentResumeState = {
  messages: ChatMessage[];
  actionLog: string[];
  completedSteps: AgentStep[];
  stepIndex: number;
  graphContextStr: string;
};

export type AgentRunResult = {
  steps: AgentStep[];
  summary: string;
  error?: string;
  canResume?: boolean;
  resumeState?: AgentResumeState;
};

export type AgentRunOptions = {
  userMessage: string;
  chatHistory: ChatMessage[];
  aiSettings: AISettings;
  responseLanguage?: string;
  graphContextStr: string;
  executorHandle: ToolExecutorHandle;
  getGraphContext: () => string;
  onStepsUpdate: (steps: AgentStep[]) => void;
  signal?: AbortSignal;
  resumeState?: AgentResumeState;
  /** Rollierende Zusammenfassung älterer Chat-Nachrichten (komprimierter Langzeit-Kontext) */
  chatSummary?: string;
  /** Lokale Graph-Validierung nach der Ausführung — löst bei Fehlern EINE Review-Runde aus */
  getValidationReport?: () => { valid: boolean; report: string };
};

function isEnglish(responseLanguage?: string): boolean {
  return (responseLanguage ?? '').toLowerCase().startsWith('en');
}

function textFor(responseLanguage: string | undefined, de: string, en: string): string {
  return isEnglish(responseLanguage) ? en : de;
}

type ToolCall = { tool: string; args: Record<string, unknown> };

// --- Rate-limit detection & retry delay ------------------------------------

function isRateLimitError(msg: string): boolean {
  const l = msg.toLowerCase();
  return l.includes('rate limit') || l.includes('429') || l.includes('tpm') ||
         l.includes('tokens per minute') || l.includes('too many requests') ||
         l.includes('tokens per day');
}

/** Extract "try again in X.XXs" delay from Groq rate-limit messages. Returns ms or 0. */
function extractRetryDelayMs(errMsg: string): number {
  const match = errMsg.match(/try again in (\d+\.?\d*)\s*s/i);
  if (!match) return 0;
  const seconds = parseFloat(match[1]);
  return isNaN(seconds) ? 0 : Math.ceil(seconds * 1000) + 600; // +600ms safety buffer
}

// --- Strip tool-call artifacts from display text -----------------------------

export function stripToolCallTags(text: string): string {
  return text
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
    .replace(/```(?:json)?\s*[\s\S]*?```/g, '')
    .trim();
}

// --- JSON Plan Parser --------------------------------------------------------
// Robust against:
//   1. Markdown fences ```json ... ```
//   2. Truncated output (incomplete closing brackets)
//   3. Bare edge objects without "tool" wrapper (fix-phase model regression)
//   4. Bare node objects without "tool" wrapper
//   5. "done." text after the JSON array (fix-phase sometimes does this)

function parsePlan(text: string): ToolCall[] | null {
  // Strip markdown fences before parsing
  const stripped = text
    .replace(/```(?:json)?\s*/g, '')
    .replace(/```\s*/g, '')
    .trim();

  const calls: ToolCall[] = [];
  let i = 0;

  while (i < stripped.length) {
    const start = stripped.indexOf('{', i);
    if (start === -1) break;

    // Find matching closing brace
    let depth = 0, j = start;
    while (j < stripped.length) {
      if (stripped[j] === '{') depth++;
      else if (stripped[j] === '}') { depth--; if (depth === 0) break; }
      j++;
    }
    if (depth > 0) break; // truncated — use what we have so far

    const candidate = stripped.slice(start, j + 1);

    try {
      const obj = JSON.parse(candidate);

      if (typeof obj?.tool === 'string') {
        // Standard: {"tool":"add_edge","args":{...}}
        calls.push({ tool: obj.tool, args: obj.args ?? {} });

      } else if (typeof obj?.source === 'string' && typeof obj?.target === 'string') {
        // Bare edge object — model forgot tool wrapper in fix-phase
        // e.g. {"source":"output-1","target":"loss-1","sourceHandle":"logits","targetHandle":"pred"}
        calls.push({
          tool: 'add_edge',
          args: {
            source: obj.source,
            target: obj.target,
            ...(obj.sourceHandle != null ? { sourceHandle: obj.sourceHandle } : {}),
            ...(obj.targetHandle != null ? { targetHandle: obj.targetHandle } : {}),
          },
        });

      } else if (typeof obj?.nodeType === 'string') {
        // Bare node object — model forgot tool wrapper
        calls.push({
          tool: 'add_node',
          args: {
            nodeType: obj.nodeType,
            ...(obj.id       ? { id: obj.id }             : {}),
            ...(obj.position ? { position: obj.position } : {}),
          },
        });
      }
    } catch { /* malformed — skip */ }

    i = j + 1;
  }

  // Model sometimes writes "done." as plain text after the array
  // If we have action calls but no done, check for trailing "done" keyword
  if (calls.length > 0 && !calls.find(c => c.tool === 'done')) {
    const tail = stripped.slice(stripped.lastIndexOf('}')).toLowerCase();
    if (tail.includes('done')) {
      calls.push({ tool: 'done', args: { summary: 'Corrections applied.' } });
    }
  }

  return calls.length > 0 ? calls : null;
}

// --- Token budget — aus AISettings.tokenBudget ----------------------------
//
// Groq free tier: 10k TPM
// Budget 'balanced': Plan ~3000 + Fix ~1000 = 4000 → comfortable
// Budget 'max':      Plan ~8000 + Fix ~2000 = 10000 → tight on Groq free
//
function getPlanMaxTokens(aiSettings: AISettings): number {
  const budget = TOKEN_BUDGET_CONFIG[aiSettings.tokenBudget ?? 'balanced'];
  // Synapse Plan bekommt ~60% des synapseMaxTokens, Fix ~25%
  return Math.floor(budget.synapseMaxTokens * 0.6);
}
function getFixMaxTokens(aiSettings: AISettings): number {
  const budget = TOKEN_BUDGET_CONFIG[aiSettings.tokenBudget ?? 'balanced'];
  return Math.floor(budget.synapseMaxTokens * 0.25);
}

const CHAT_CONTEXT_MESSAGES = 4;
const CHAT_CONTEXT_CHARS    = 300;

// --- Phase 1: Plan System Prompt ---------------------------------------------

function buildPlanSystem(graphContextStr: string, chatContext: string): string {
  return `You are a Synapse canvas AI for building neural network graphs.
Analyze the request and decide:
- CHANGES requested (add/connect/configure/remove nodes): return a JSON array of tool calls ending with done.
- EVALUATION/AUDIT (keywords: prüfen, check, analyze, überprüfen, probleme, issues, audit):
  * ACTIVELY find problems using tools (check for illegal connections, redundant nodes, missing edges)
  * Use remove_edge, remove_node, set_param to fix issues
  * Return tool calls to fix discovered problems, ending with done
  * If no problems found, return [{"tool":"done","args":{"summary":"✓ Graph OK — all rules respected"}}]
- DIAGNOSTIC MODE (keywords: shape error, shapes cannot be multiplied, tensor error, shape test failed):
  * Extract shape info from error (e.g. "65536x32 and 128x256")
  * FIRST: Use DenseParams, DenseFlow, ConvParams, ConvFlow, LayerNormParams, LayerNormFlow, AttentionParams, and AttentionFlow lines from Canvas context to identify exact node IDs
  * SECOND: For every dense A → dense B connection, A.outputSize MUST equal B.inputSize
  * THIRD: For every conv2d A → conv2d B connection, A.outChannels MUST equal B.inChannels
  * FOURTH: For every node A → layernorm B connection, B.normalizedShape MUST equal A's feature size
  * FIFTH: For every node A → attention/transformer B connection, B.embedDim MUST equal A's feature size
  * SIXTH: Prefer set_param on the downstream node parameter to match the upstream output features
  * SEVENTH: Report exact fixes applied with before/after values
  * Example summary: "✓ dense-2 inputSize: 128→32 korrigiert, weil dense-1.outputSize=32 ist"
  * Example summary: "✓ conv2d-2 inChannels: 3→64 korrigiert, weil conv2d-1.outChannels=64 ist"
  * Example summary: "✓ layernorm-1 normalizedShape: 512→256 korrigiert, weil vorheriger Layer 256 Features liefert"
  * Example summary: "✓ attention-1 embedDim: 512→256 korrigiert, weil vorheriger Layer 256 Features liefert"
- QUESTION/INFO (keywords: what, wie, warum, erklär): return ONLY [{"tool":"done","args":{"summary":"your answer max 3 sentences"}}]

OUTPUT: ONLY a valid JSON array. No prose, no markdown fences.

## CRITICAL RULES FOR PROBLEM-HUNTING (prüfen mode):
Scan graph for violations (highest priority):
1. ✗ ILLEGAL: Edges TO optimizer nodes — REMOVE THEM
2. ✗ ILLEGAL: Edges FROM output_node — REMOVE THEM  
3. ✗ ILLEGAL: loss(out) → optimizer(in) — REMOVE, use scheduler instead
4. ⚠️ REDUNDANT: Multiple output_node/loss/optimizer nodes — keep first, remove extras
5. ⚠️ DISCONNECTED: Data nodes not flowing into network → remove or reconnect
6. ⚠️ WRONG PORT: Dense layer has output connected to wrong input → set_param to fix size

For each violation found:
- Report in done summary: "Gefunden: N Probleme → K behoben"
- Use remove_edge/remove_node/set_param tools to fix
- Apply fixes FIRST, only report what couldn't be auto-fixed

CRITICAL: Only use nodeTypes from the Types list in Canvas section. Do NOT invent types.
## If a needed type doesn't exist, use the closest available one and explain in the done summary.

## CANVAS IS GROUND TRUTH — read it carefully before acting:
# - The Nodes list is accurate. Do NOT add_node for IDs already in the Nodes list.
# - The Edges list is accurate. Do NOT add_edge for paths already listed.

## Tools:
${PLAN_TOOLS}

## Position — object format only, never strings:
{"x": 100, "y": 200}

## Layout guide (spread nodes out — don't stack everything at y=200):
- Data nodes:        x=100-300,  y=100-500  (spread vertically)
- Network layers:    x=400-800,  y=100-500  (left-to-right flow)
- Training nodes:    x=900-1100, y=100-300
- Utility/Math:      x=400-800,  y=550-750

## Exact port IDs (inputs→outputs) for all node types:
# Data:        input(→out)  csv_loader(→out)  image_loader(→out)
#              tokenizer(text→tokens)  dataset_split(in→train/val/test)  augmentation(in→out)
# Layers:      dense(in→out)  conv2d(in→out)  embedding(tokens→out)
#              lstm(in→out,hidden)  layernorm(in→out)  batchnorm(in→out)  dropout(in→out)
#              attention(query,key,value→out)  transformer_block(in→out)
# Activation:  relu/gelu/sigmoid/softmax/tanh/leaky_relu/silu (in→out)
# Training:    optimizer(→optimizer) [NO inputs — standalone node]
#              loss(pred,targets→loss)
#              scheduler(optimizer→scheduler)
#              output_node(logits→) [NO outputs — terminal node]
# Math:        add_node/multiply_node/matmul/merge(a,b→out)
#              normalize/reshape/transpose(in→out)
# Logic:       split_node(in→a,b)  pool(in→out)

## Correct training subgraph:
# dense-N(out) → output_node(logits)   [TERMINAL]
# dense-N(out) → loss(pred)

## SHAPE ERROR ANALYSIS — When training fails with "shapes cannot be multiplied":
# Error Example: "mat1 and mat2 shapes cannot be multiplied (65536x32 and 128x256)"
#   → mat1 features = 32 (second number)
#   → next Linear expects inputSize = 128 (first number of mat2)
#   → FIX: The upstream output features must equal downstream inputSize.
#   → For dense A → dense B, prefer: set_param(B, "inputSize", A.outputSize)
#
# Dense Layer Shape Rules:
# - Each Dense node must have {outputSize: N} where N is an integer
# - If dense-1(out) → dense-2(in): dense-2 must have inputSize === dense-1.outputSize
# - After Conv2d/LSTM: use Reshape/Flatten to convert (C,H,W) → (C*H*W) before Dense
# - Dimension flow: [batch, in_features] → Dense → [batch, outputSize]
#
# Fix Strategy:
# 1. Read error message: extract "mat1 X mat2 shapes" from error
# 2. For each DenseFlow line, validate: source.outputSize matches target.inputSize
# 3. If mismatch found: Use set_param(targetDense, "inputSize", source.outputSize)
# 4. For LayerNorm errors like "normalized_shape=[512] ... input ... [32, 256]":
#    Use LayerNormFlow and set_param(layerNormNode, "normalizedShape", upstreamFeatureSize)
# 5. Common mistake: Conv2d(out) → Dense(in=64) but Dense never defined outputSize!

## RANK/LAYOUT MISMATCH — error says "outputs BD → ... expects BTC" (2D vs 3D):
# set_param (embedDim/normalizedShape/inputSize) can NEVER fix this — it is a
# tensor-RANK problem, not a size problem. Do NOT retry parameter values.
# Choose exactly ONE fix:
#  A) RECOMMENDED after dense layers: remove_node the attention/transformer node
#     (and its adjacent layernorm if it only feeds that attention), then add_edge
#     to reconnect the chain (previous node → next node). Attention over a single
#     feature vector adds nothing.
#  B) Keep attention: add_node reshape between source and attention,
#     set_param(reshapeId, "shape", "1, <features>")  → turns [B, F] into [B, 1, F],
#     add_edge source→reshape + reshape→attention, remove_edge source→attention.
# augmentation(out) → loss(targets)
# optimizer [standalone — no incoming edges]
# optimizer(optimizer) → scheduler(optimizer)  [optional]

## Canvas:
${graphContextStr}
${chatContext ? '\n' + chatContext : ''}

## Example — action:
[
  {"tool":"add_node","args":{"nodeType":"layernorm","id":"ln-1","position":{"x":700,"y":200}}},
  {"tool":"add_edge","args":{"source":"attention-1","target":"ln-1"}},
  {"tool":"set_param","args":{"nodeId":"dense-1","key":"outputSize","value":512}},
  {"tool":"done","args":{"summary":"LayerNorm nach Attention eingefügt, Dense auf 512 gesetzt."}}
]

## Example — evaluation:
[
  {"tool":"done","args":{"summary":"Das Netzwerk ist solide für ein kleines LLM. Fehlend: Loss-Node und Optimizer für Training. Empfehlung: loss-Node mit pred/targets verbinden, Adam-Optimizer hinzufügen."}}
]`;
}

// --- Phase 3: Fix System Prompt ----------------------------------------------

function buildFixSystem(
  graphContextStr: string,
  userMessage: string,
  failures: Array<{ tool: string; args: unknown; error: string }>,
): string {
  const failList = failures
    .map((f) => `- ${f.tool}(${JSON.stringify(f.args)}) => ${f.error}`)
    .join('\n');

  return `Synapse fixer. Correct ONLY the ${failures.length} failed step(s) listed below.
Return a MINIMAL JSON array — do NOT rebuild existing edges or add unrequested nodes.

EVERY item MUST use: {"tool":"TOOLNAME","args":{...}}
Do NOT return bare objects. End with {"tool":"done","args":{"summary":"..."}}

## SPECIAL: Shape Errors (mat1 and mat2 shapes cannot be multiplied)
If error contains shape mismatch like "(65536x32 and 128x256)":
1. IDENTIFY: Which DenseFlow line is MISMATCH.
2. FIX: Use set_param(downstreamDenseId, "inputSize", upstreamDense.outputSize) 
3. ALSO CHECK: Is there a Conv2d/LSTM before Dense without Reshape? Add reshape if needed.
4. VALIDATE: Each Dense node MUST have outputSize set to a number.
5. REPORT: "dense-2 inputSize: 128→32 korrigiert (war inkompatibel mit dense-1 outputSize=32)"

If error contains LayerNorm mismatch like "normalized_shape=[512] ... input of size[32, 256]":
1. IDENTIFY: Which LayerNormFlow line is MISMATCH.
2. FIX: Use set_param(layerNormNodeId, "normalizedShape", upstreamFeatureSize).
3. REPORT: "layernorm-1 normalizedShape: 512→256 korrigiert."

If error contains Conv2D channel mismatch like "expected input[...] to have 3 channels, but got 64 channels":
1. IDENTIFY: Which ConvFlow line is MISMATCH.
2. FIX: Use set_param(conv2dNodeId, "inChannels", upstreamOutChannels).
3. REPORT: "conv2d-2 inChannels: 3→64 korrigiert."

If error contains Attention embed mismatch like "was expecting embedding dimension of 512, but got 256":
1. IDENTIFY: Which AttentionFlow line is MISMATCH.
2. FIX: Use set_param(attentionNodeId, "embedDim", upstreamFeatureSize).
3. REPORT: "attention-1 embedDim: 512→256 korrigiert."

If error contains RANK mismatch like "outputs BD → attention (...) expects BTC" (2D vs 3D):
1. set_param can NEVER fix this — do NOT change embedDim/normalizedShape again.
2. EITHER remove_node the attention/transformer (+ its dedicated layernorm) and
   add_edge to reconnect the chain (recommended after dense layers),
3. OR add_node reshape + set_param(reshapeId, "shape", "1, <features>") and rewire:
   remove_edge source→attention, add_edge source→reshape, add_edge reshape→attention.

## HARD RULES (same as main agent):
# optimizer has NO input ports — NEVER connect anything TO optimizer.
# output_node has NO output ports — it is terminal, no edges can leave it.
# loss → optimizer is ALWAYS INVALID. Use: optimizer(optimizer) → scheduler(optimizer)
# Never connect a node to itself (no self-loops).

## Failed step(s) to fix:
${failList}

## Canvas now:
${graphContextStr}

## Original task: ${userMessage}

Return JSON array only. No markdown, no prose.`;
}

// --- Execute a batch of ToolCalls --------------------------------------------

async function executeBatch(
  calls: ToolCall[],
  handle: ToolExecutorHandle,
  currentSteps: AgentStep[],
  startIndex: number,
  onStepsUpdate: (steps: AgentStep[]) => void,
): Promise<{ steps: AgentStep[]; failures: Array<{ tool: string; args: unknown; error: string }>; actualChanges: number }> {
  let steps = [...currentSteps];
  const failures: Array<{ tool: string; args: unknown; error: string }> = [];
  let actualChanges = 0;
  const emit = () => onStepsUpdate([...steps]);

  for (let i = 0; i < calls.length; i++) {
    const call = calls[i];
    if (call.tool === 'done') break;

    const stepId = `step_${startIndex + i}_${Date.now()}`;
    steps = [...steps, { id: stepId, tool: call.tool, args: call.args, status: 'running' }];
    emit();

    let result: { success: boolean; data?: unknown; error?: string };
    try {
      result = await handle.execute(call.tool, call.args);
    } catch (e: any) {
      result = { success: false, error: String(e?.message ?? e) };
    }

    // Count as an actual change only when successful AND not a silent skip
    if (result.success && !(result.data as any)?.skipped) {
      actualChanges++;
    }

    steps = steps.map((s) =>
      s.id === stepId
        ? { ...s, status: result.success ? 'success' : 'error', resultData: result.data, error: result.error }
        : s
    );
    emit();

    if (!result.success) {
      failures.push({ tool: call.tool, args: call.args, error: result.error ?? 'unknown' });
    }
  }

  return { steps, failures, actualChanges };
}

// --- Main Agent Run ----------------------------------------------------------

export async function runSynapseAgent(opts: AgentRunOptions): Promise<AgentRunResult> {
  const { userMessage, chatHistory, aiSettings, responseLanguage, graphContextStr, executorHandle, getGraphContext, onStepsUpdate, signal } = opts;

  let currentSteps: AgentStep[] = [];
  onStepsUpdate([]);

  // Kontext = komprimierte Zusammenfassung älterer Nachrichten (falls vorhanden)
  // + die letzten Nachrichten wörtlich (gekürzt).
  const priorMessages = chatHistory.slice(-(CHAT_CONTEXT_MESSAGES + 1), -1);
  const summaryBlock = opts.chatSummary
    ? `## Conversation summary (compressed older context — treat as established facts):\n${opts.chatSummary}`
    : '';
  const recentBlock = priorMessages.length > 0
    ? `## Recent conversation:\n${priorMessages.map(m => `${m.role}: ${m.content.slice(0, CHAT_CONTEXT_CHARS)}`).join('\n')}`
    : '';
  const chatContext = [summaryBlock, recentBlock].filter(Boolean).join('\n\n');

  // ============================================================
  // PHASE 1 — PLAN
  // ============================================================
  if (signal?.aborted) return { steps: [], summary: textFor(responseLanguage, 'Abgebrochen', 'Aborted') };

  // Inject a compact canvas summary into the USER message so that even
  // small/weak models (8b) see the current state — not just in the system prompt.
  // Extracts the first two lines of graphContextStr: "Nodes(N): ..." and "Edges(M): ..."
  const canvasSummary = graphContextStr.match(/Nodes\(\d+\)[^\n]*/)?.[0]?.match(/Nodes\(\d+\)/)?.[0] ?? 'Nodes(?)';
  const edgeSummary   = graphContextStr.match(/Edges\(\d+\)/)?.[0] ?? 'Edges(?)';
  const enrichedUserMessage = `[Canvas: ${canvasSummary}, ${edgeSummary}]\n\n${userMessage}`;

  const planSystem   = buildPlanSystem(graphContextStr, chatContext);
  const planMessages: ChatMessage[] = [{ role: 'user', content: enrichedUserMessage }];

  const dbg1 = await debugLogRequest(0, planSystem, planMessages);
  let planReply: string;
  try {
    planReply = await callAI(aiSettings, {
      system: planSystem,
      messages: planMessages,
      maxTokens: getPlanMaxTokens(aiSettings),
      temperature: 0.1,
      responseLanguage,
    });
  } catch (e: any) {
    const errMsg = String(e?.message ?? e);
    await dbg1.onError(errMsg);
    if (isRateLimitError(errMsg)) {
      const delayMs = extractRetryDelayMs(errMsg);
      if (delayMs > 0 && delayMs <= 30_000) {
        await new Promise(res => setTimeout(res, delayMs));
        try {
          planReply = await callAI(aiSettings, {
            system: planSystem,
            messages: planMessages,
            maxTokens: getPlanMaxTokens(aiSettings),
            temperature: 0.1,
            responseLanguage,
          });
        } catch (e2: any) {
          const errMsg2 = String(e2?.message ?? e2);
          await dbg1.onError(`Auto-retry failed: ${errMsg2}`);
          return {
            steps: [],
            summary: '',
            error: textFor(responseLanguage, `Rate limit — bitte in ${Math.ceil(delayMs / 1000)}s erneut versuchen: ${errMsg2}`, `Rate limit — please try again in ${Math.ceil(delayMs / 1000)}s: ${errMsg2}`),
            canResume: false,
          };
        }
      } else {
        return { steps: [], summary: '', error: textFor(responseLanguage, `AI Fehler: ${errMsg}`, `AI error: ${errMsg}`), canResume: false };
      }
    } else {
      return { steps: [], summary: '', error: textFor(responseLanguage, `AI Fehler: ${errMsg}`, `AI error: ${errMsg}`) };
    }
  }

  const plan = parsePlan(planReply);
  await dbg1.onReply(planReply, plan ? { tool: 'plan', args: { steps: plan.length } } : null);

  if (!plan || plan.length === 0) {
    return { steps: [], summary: stripToolCallTags(planReply) };
  }

  const doneStep    = plan.find((c) => c.tool === 'done');
  // Reorder: all add_node calls first, then the rest in original order.
  // This prevents "Target not found" errors when the model emits an add_edge
  // before the add_node for the target node (common with weaker models).
  const rawActionSteps = plan.filter((c) => c.tool !== 'done');
  const actionSteps = [
    ...rawActionSteps.filter((c) => c.tool === 'add_node'),
    ...rawActionSteps.filter((c) => c.tool !== 'add_node'),
  ];

  // ============================================================
  // PHASE 2 — EXECUTE
  // ============================================================
  if (signal?.aborted) return { steps: currentSteps, summary: textFor(responseLanguage, 'Abgebrochen', 'Aborted') };

  const { steps: stepsAfterExecute, failures, actualChanges: phase2Changes } = await executeBatch(
    actionSteps, executorHandle, currentSteps, 0, onStepsUpdate
  );
  currentSteps = stepsAfterExecute;
  let totalActualChanges = phase2Changes;

  // ============================================================
  // PHASE 3 — FIX (only on real failures)
  // ============================================================
  let fixSummary: string | null = null;
  if (failures.length > 0) {
    if (signal?.aborted) return { steps: currentSteps, summary: textFor(responseLanguage, 'Abgebrochen', 'Aborted') };

    const freshContext = getGraphContext();
    const fixSystem    = buildFixSystem(freshContext, userMessage, failures);
    const fixMessages: ChatMessage[] = [{ role: 'user', content: 'Fix the failed steps.' }];

    const dbg3 = await debugLogRequest(1, fixSystem, fixMessages);
    try {
      const fixReply = await callAI(aiSettings, {
        system: fixSystem,
        messages: fixMessages,
        maxTokens: getFixMaxTokens(aiSettings),
        temperature: 0.05,
        responseLanguage,
      });

      const fixPlan = parsePlan(fixReply);
      await dbg3.onReply(fixReply, fixPlan ? { tool: 'fix', args: { steps: fixPlan.length } } : null);

      if (fixPlan && fixPlan.length > 0) {
        const { steps: stepsAfterFix, actualChanges: phase3Changes } = await executeBatch(
          fixPlan.filter((c) => c.tool !== 'done'),
          executorHandle, currentSteps, actionSteps.length, onStepsUpdate
        );
        currentSteps = stepsAfterFix;
        totalActualChanges += phase3Changes;

        const fixDone = fixPlan.find((c) => c.tool === 'done');
        if (fixDone?.args?.summary) fixSummary = String(fixDone.args.summary);
      }
    } catch (e: any) {
      const errMsg = String(e?.message ?? e);
      await dbg3.onError(errMsg);
      if (isRateLimitError(errMsg)) {
        return {
          steps: currentSteps,
          summary: '',
          error: textFor(responseLanguage, `AI Fehler (Fix): ${errMsg}`, `AI error (Fix): ${errMsg}`),
          canResume: true,
          resumeState: {
            messages: fixMessages,
            actionLog: failures.map(f => `FAIL:${f.tool}`),
            completedSteps: currentSteps,
            stepIndex: 1,
            graphContextStr: freshContext,
          },
        };
      }
    }
  }

  // ============================================================
  // PHASE 4 — REVIEW (max. eine Extra-Runde)
  // Nur wenn Aktionen ausgeführt wurden und die lokale Validierung danach
  // noch echte Shape-/Dimensions-Fehler meldet: dem Modell den Befund
  // zurückspiegeln und einmal nachbessern lassen. Best-effort — Fehler
  // in dieser Phase sind nie fatal.
  // ============================================================
  let reviewSummary: string | null = null;
  if (
    actionSteps.length > 0 &&
    totalActualChanges > 0 &&
    opts.getValidationReport &&
    !signal?.aborted
  ) {
    const check = opts.getValidationReport();
    if (!check.valid && check.report) {
      const reviewSystem = buildPlanSystem(getGraphContext(), chatContext);
      const reviewMessages: ChatMessage[] = [{
        role: 'user',
        content:
          `REVIEW ROUND — after your previous tool calls, graph validation still reports:\n` +
          `${check.report}\n\n` +
          `Fix these remaining issues with a MINIMAL JSON tool-call array ` +
          `(prefer set_param; add_node/add_edge/remove_edge only if required). ` +
          `End with {"tool":"done","args":{"summary":"..."}}. ` +
          `If the reported state is intentional, return ONLY the done call explaining why.`,
      }];

      const dbg4 = await debugLogRequest(2, reviewSystem, reviewMessages);
      try {
        const reviewReply = await callAI(aiSettings, {
          system: reviewSystem,
          messages: reviewMessages,
          maxTokens: getFixMaxTokens(aiSettings),
          temperature: 0.05,
          responseLanguage,
        });

        const reviewPlan = parsePlan(reviewReply);
        await dbg4.onReply(reviewReply, reviewPlan ? { tool: 'review', args: { steps: reviewPlan.length } } : null);

        if (reviewPlan && reviewPlan.length > 0) {
          const { steps: stepsAfterReview, actualChanges: phase4Changes } = await executeBatch(
            reviewPlan.filter((c) => c.tool !== 'done'),
            executorHandle, currentSteps, currentSteps.length, onStepsUpdate
          );
          currentSteps = stepsAfterReview;
          totalActualChanges += phase4Changes;

          const reviewDone = reviewPlan.find((c) => c.tool === 'done');
          if (reviewDone?.args?.summary) reviewSummary = String(reviewDone.args.summary);
        }
      } catch (e: any) {
        await dbg4.onError(String(e?.message ?? e));
      }
    }
  }

  // When the model tried to build things that already exist, give a clear message
  // instead of "✓ 30 Schritte ausgeführt" which makes the user think something changed.
  if (actionSteps.length > 0 && totalActualChanges === 0) {
    const doneSummary = doneStep?.args?.summary ? String(doneStep.args.summary) : '';
    return {
      steps: currentSteps,
      summary: textFor(
        responseLanguage,
        `Canvas bereits vollständig — alle angeforderten Elemente sind bereits vorhanden.${doneSummary ? ' ' + doneSummary : ''}`,
        `Canvas already complete — all requested elements are already present.${doneSummary ? ' ' + doneSummary : ''}`,
      ),
    };
  }

  let summary =
    fixSummary
    ?? (doneStep?.args?.summary
      ? String(doneStep.args.summary)
      : `${totalActualChanges} von ${actionSteps.length} Schritten effektiv`);
  if (reviewSummary) summary = `${summary}\n${reviewSummary}`;

  return { steps: currentSteps, summary };
}
