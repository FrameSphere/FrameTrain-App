/**
 * codeGenerator.ts — Synapse Builder → canvas_model.py
 * Converts the node graph into a complete, runnable PyTorch training script.
 */

import { Node, Edge } from "@xyflow/react";
import { TrainingConfig } from "./TrainingConsole";
import { computeExecutionOrder } from "./graph-to-dynamic-forward";

// ─── Safe Python identifiers ──────────────────────────────────────────────────
const pyAttr = (id: string) =>
  "ly_" + id.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 20);

const pyVar = (id: string) =>
  "h_" + id.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 16);

// ─── Node data accessors ──────────────────────────────────────────────────────
const getNodeType = (n: Node): string =>
  (n.data as any)?._def?.type ?? (n.data as any)?.nodeType ?? (n.data as any)?.category ?? "unknown";

const getParams = (n: Node): Record<string, any> =>
  (n.data as any)?.params ?? {};

const getCategory = (n: Node): string =>
  (n.data as any)?.category ?? (n.data as any)?._def?.category ?? "";

// ─── Topological sort (Kahn's algorithm) ─────────────────────────────────────
function topoSort(nodes: Node[], edges: Edge[]): Node[] {
  const byId  = new Map(nodes.map((n) => [n.id, n]));
  const inDeg = new Map(nodes.map((n) => [n.id, 0]));
  const adj   = new Map<string, string[]>(nodes.map((n) => [n.id, []]));

  for (const e of edges) {
    adj.get(e.source)?.push(e.target);
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
  }

  const queue  = nodes.filter((n) => (inDeg.get(n.id) ?? 0) === 0);
  const result: Node[] = [];

  while (queue.length > 0) {
    const n = queue.shift()!;
    result.push(n);
    for (const nxt of adj.get(n.id) ?? []) {
      const d = (inDeg.get(nxt) ?? 0) - 1;
      inDeg.set(nxt, d);
      if (d === 0) { const nn = byId.get(nxt); if (nn) queue.push(nn); }
    }
  }

  // Append isolated / cycle nodes
  for (const n of nodes) {
    if (!result.find((r) => r.id === n.id)) result.push(n);
  }
  return result;
}

const getSources = (targetId: string, edges: Edge[]): string[] =>
  edges.filter((e) => e.target === targetId).map((e) => e.source);

// ─── __init__ line per node type ──────────────────────────────────────────────
function genInitLine(nodeId: string, type: string, p: Record<string, any>): string | null {
  const a = pyAttr(nodeId);
  switch (type) {
    case "dense":
      return `        self.${a} = nn.Linear(${p.inputSize ?? 128}, ${p.outputSize ?? 256})`;
    case "conv2d": {
      const pad = p.padding === "same" ? `"same"` : (p.padding ?? 1);
      return `        self.${a} = nn.Conv2d(${p.inChannels ?? 3}, ${p.outChannels ?? 64}, kernel_size=${p.kernelSize ?? 3}, stride=${p.stride ?? 1}, padding=${pad}, groups=${p.groups ?? 1})`;
    }
    case "embedding":
      return `        self.${a} = nn.Embedding(${p.vocabSize ?? 50000}, ${p.embeddingDim ?? 512}, padding_idx=${p.paddingIdx ?? 0})`;
    case "lstm": {
      const drop = Number(p.numLayers ?? 2) > 1 ? (p.dropout ?? 0.1) : 0;
      return `        self.${a} = nn.LSTM(${p.inputSize ?? 256}, ${p.hiddenSize ?? 512}, num_layers=${p.numLayers ?? 2}, bidirectional=${p.bidirectional ? "True" : "False"}, dropout=${drop}, batch_first=True)`;
    }
    case "attention":
      return `        self.${a} = nn.MultiheadAttention(${p.embedDim ?? 512}, ${p.numHeads ?? 8}, dropout=${p.dropout ?? 0.1}, batch_first=True)`;
    case "transformer_block":
      return `        self.${a} = nn.TransformerEncoderLayer(${p.embedDim ?? 512}, ${p.numHeads ?? 8}, dim_feedforward=${p.ffnDim ?? 2048}, dropout=${p.dropout ?? 0.1}, batch_first=True, norm_first=True)`;
    case "layernorm":
      return `        self.${a} = nn.LayerNorm(${p.normalizedShape ?? 512}, eps=${p.eps ?? 1e-5})`;
    case "batchnorm":
      return `        self.${a} = nn.BatchNorm1d(${p.numFeatures ?? 64})`;
    case "dropout":
      return `        self.${a} = nn.Dropout(p=${p.p ?? 0.1})`;
    case "relu":       return `        self.${a} = nn.ReLU()`;
    case "gelu":       return `        self.${a} = nn.GELU()`;
    case "sigmoid":    return `        self.${a} = nn.Sigmoid()`;
    case "softmax":    return `        self.${a} = nn.Softmax(dim=${p.dim ?? -1})`;
    case "tanh":       return `        self.${a} = nn.Tanh()`;
    case "leaky_relu": return `        self.${a} = nn.LeakyReLU(negative_slope=${p.negativeSlope ?? 0.01})`;
    case "silu":       return `        self.${a} = nn.SiLU()`;
    default:           return null;
  }
}

// ─── forward() line per node type ────────────────────────────────────────────
function genForwardLine(
  nodeId: string, type: string, p: Record<string, any>, srcIds: string[]
): string | null {
  const v0 = srcIds.length > 0 ? pyVar(srcIds[0]) : "x";
  const v1 = srcIds.length > 1 ? pyVar(srcIds[1]) : v0;
  return genForwardLineWithInputs(nodeId, type, p, v0, v1);
}

function genForwardLineWithInputs(
  nodeId: string, type: string, p: Record<string, any>, v0: string, v1: string
): string | null {
  const a  = pyAttr(nodeId);
  const ov = pyVar(nodeId);

  switch (type) {
    case "dense": case "batchnorm": case "dropout": case "relu": case "gelu":
    case "sigmoid": case "softmax": case "tanh": case "leaky_relu": case "silu":
    case "layernorm": case "conv2d": case "transformer_block":
      return `        ${ov} = self.${a}(${v0})`;
    case "embedding":
      return `        ${ov} = self.${a}(${v0}.long())`;
    case "lstm":
      return `        ${ov}_seq, _ = self.${a}(${v0})\n        ${ov} = ${ov}_seq[:, -1, :]`;
    case "attention":
      return `        ${ov}, _ = self.${a}(${v0}, ${v0}, ${v0})`;
    case "add_node":
      return `        ${ov} = ${v0} + ${v1}`;
    case "multiply_node":
      return `        ${ov} = ${v0} * ${v1}`;
    case "matmul":
      return `        ${ov} = torch.matmul(${v0}, ${v1})`;
    case "normalize":
      return `        ${ov} = F.normalize(${v0}, p=${p.p ?? 2}, dim=${p.dim ?? -1})`;
    case "reshape": {
      const shp = String(p.shape ?? "-1, 512").replace(/\s/g, "").replace(/^-1,/, "");
      return `        ${ov} = ${v0}.reshape(${v0}.size(0), ${shp})`;
    }
    case "transpose":
      return `        ${ov} = ${v0}.transpose(${p.dim0 ?? -2}, ${p.dim1 ?? -1})`;
    case "merge":
      return `        ${ov} = torch.cat([${v0}, ${v1}], dim=${p.dim ?? -1})`;
    case "split_node":
      return `        ${ov}_parts = torch.chunk(${v0}, ${p.chunks ?? 2}, dim=${p.dim ?? -1})\n        ${ov} = ${ov}_parts[0]`;
    case "pool":
      if ((p.type ?? "global_avg").includes("max")) return `        ${ov} = F.adaptive_max_pool2d(${v0}, 1).flatten(1)`;
      if (p.type === "avg_2d")                       return `        ${ov} = F.avg_pool2d(${v0}, 2, stride=${p.stride ?? 2})`;
      if (p.type === "max_2d")                       return `        ${ov} = F.max_pool2d(${v0}, 2, stride=${p.stride ?? 2})`;
      return `        ${ov} = F.adaptive_avg_pool2d(${v0}, 1).flatten(1)`;
    default:
      return null;
  }
}

/**
 * Phase 2: Dynamic forward — execution order follows graph topology, not just a linear chain.
 */
function genDynamicForwardLines(nodes: Node[], edges: Edge[]): { lines: string[]; lastVar: string } {
  const archNodes = nodes.filter((n) => {
    const c = getCategory(n);
    return c !== "data" && c !== "training";
  });
  const archIds = new Set(archNodes.map((n) => n.id));
  const archEdges = edges.filter((e) => archIds.has(e.source) && archIds.has(e.target));
  const order = computeExecutionOrder(archNodes, archEdges);

  const lines: string[] = [];
  let lastVar = "x";

  for (const nodeId of order) {
    const n = archNodes.find((nn) => nn.id === nodeId);
    if (!n) continue;
    const type = getNodeType(n);
    const p = getParams(n);
    const ov = pyVar(nodeId);
    const srcs = getSources(nodeId, edges).filter((id) => archIds.has(id));

    if (type === "input") {
      lines.push(`        ${ov} = x`);
      lastVar = ov;
      continue;
    }

    const v0 = srcs.length > 0 ? pyVar(srcs[0]) : "x";
    const v1 = srcs.length > 1 ? pyVar(srcs[1]) : v0;
    const line = genForwardLineWithInputs(nodeId, type, p, v0, v1);
    if (line) {
      lines.push(line);
      lastVar = ov;
    }
  }

  if (lines.length === 0) {
    lines.push("        pass");
  }
  return { lines, lastVar };
}

/**
 * Fix 1.4: Vollständiges Script zurückgeben (inkl. train() + __main__),
 * damit canvas_model.py direkt mit `python canvas_model.py` ausführbar ist.
 */
export function extractCanvasModelClass(script: string): string {
  const start = script.indexOf("class CanvasModel");
  if (start < 0) return script;
  return script.trim();
}

// ─── Dataset loader code ──────────────────────────────────────────────────────
function genDatasetCode(
  dataNodeType: string, dataParams: Record<string, any>,
  batchSize: number, numClasses: number, inputFeatures: number
): string {
  if (dataNodeType === "image_loader") {
    const sz  = dataParams.imageSize ?? 224;
    const ch  = parseInt(String(dataParams.channels ?? "3"), 10);
    const nrm = dataParams.normalize
      ? `transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),`
      : "";
    return `
def get_dataloaders(batch_size):
    dsp = os.environ.get("DATASET_PATH","")
    if dsp and os.path.isdir(dsp):
        try:
            from torchvision import datasets, transforms
            tfm = transforms.Compose([transforms.Resize((${sz},${sz})), transforms.ToTensor(), ${nrm}])
            ds  = datasets.ImageFolder(dsp, transform=tfm)
            n   = int(len(ds)*0.8)
            tr, va = random_split(ds, [n, len(ds)-n])
            return DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0), \
                   DataLoader(va, batch_size=batch_size, num_workers=0)
        except Exception as e:
            print(f"[Status] Bild-Dataset-Fehler: {e}. Nutze Dummy-Daten.", flush=True)
    print("[Status] Nutze synthetische Bilddaten (${ch}x${sz}x${sz})", flush=True)
    X = torch.randn(256, ${ch}, ${sz}, ${sz}); y = torch.randint(0, ${numClasses}, (256,))
    ds = TensorDataset(X, y)
    return DataLoader(Subset(ds,range(200)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(ds,range(200,256)), batch_size=batch_size)
`;
  }
  if (dataNodeType === "csv_loader") {
    const tgt = dataParams.targetCol ?? "label";
    return `
def get_dataloaders(batch_size):
    dsp = os.environ.get("DATASET_PATH","")
    if dsp and os.path.isfile(dsp):
        try:
            import pandas as pd
            df = pd.read_csv(dsp)
            if "${tgt}" in df.columns:
                X = torch.tensor(df.drop(columns=["${tgt}"]).values.astype("float32"))
                y = torch.tensor(df["${tgt}"].values, dtype=torch.long)
            else:
                X = torch.tensor(df.values[:,:-1].astype("float32"))
                y = torch.tensor(df.values[:,-1].astype("int64"), dtype=torch.long)
            ds = TensorDataset(X, y); n = int(len(ds)*0.8)
            tr, va = random_split(ds, [n, len(ds)-n])
            return DataLoader(tr, batch_size=batch_size, shuffle=True), \
                   DataLoader(va, batch_size=batch_size)
        except Exception as e:
            print(f"[Status] CSV-Fehler: {e}. Nutze Dummy-Daten.", flush=True)
    print("[Status] Nutze synthetische Tabellendaten (${inputFeatures} Features)", flush=True)
    X = torch.randn(512, ${inputFeatures}); y = torch.randint(0, ${numClasses}, (512,))
    ds = TensorDataset(X, y)
    return DataLoader(Subset(ds,range(400)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(ds,range(400,512)), batch_size=batch_size)
`;
  }
  return `
def get_dataloaders(batch_size):
    print("[Status] Nutze Dummy-Daten (${inputFeatures} Features)", flush=True)
    X = torch.randn(512, ${inputFeatures}); y = torch.randint(0, ${numClasses}, (512,))
    ds = TensorDataset(X, y)
    return DataLoader(Subset(ds,range(400)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(ds,range(400,512)), batch_size=batch_size)
`;
}

// ─── Main export ──────────────────────────────────────────────────────────────
export function generateTrainingScript(
  nodes: Node[], edges: Edge[], config: TrainingConfig
): string {
  const archNodes  = nodes.filter((n) => { const c = getCategory(n); return c !== "data" && c !== "training"; });
  const dataNodes  = nodes.filter((n) => getCategory(n) === "data");
  const trainNodes = nodes.filter((n) => getCategory(n) === "training");

  const archEdges = edges.filter(
    (e) => archNodes.find((n) => n.id === e.source) && archNodes.find((n) => n.id === e.target)
  );
  const sorted = topoSort(archNodes, archEdges);

  const initLines: string[] = [];
  for (const n of sorted) {
    const line = genInitLine(n.id, getNodeType(n), getParams(n));
    if (line) initLines.push(line);
  }
  if (initLines.length === 0) initLines.push("        pass  # No learnable layers");

  const { lines: fwdLines, lastVar } = genDynamicForwardLines(nodes, edges);

  const dataNode     = dataNodes[0];
  const dataNodeType = dataNode ? getNodeType(dataNode) : "default";
  const dataParams   = dataNode ? getParams(dataNode) : {};

  const firstNode  = sorted[0];
  const firstType  = firstNode ? getNodeType(firstNode) : "";
  const firstP     = firstNode ? getParams(firstNode) : {};
  let inputFeatures = 128;
  if (firstType === "dense")          inputFeatures = Number(firstP.inputSize ?? 128);
  else if (firstType === "lstm")      inputFeatures = Number(firstP.inputSize ?? 256);
  else if (firstType === "embedding") inputFeatures = 32;

  const optimNode  = trainNodes.find((n) => getNodeType(n) === "optimizer");
  const lossNode   = trainNodes.find((n) => getNodeType(n) === "loss");
  const schedNode  = trainNodes.find((n) => getNodeType(n) === "scheduler");
  const outputNode = trainNodes.find((n) => getNodeType(n) === "output_node");
  const op = optimNode  ? getParams(optimNode)  : {};
  const lp = lossNode   ? getParams(lossNode)   : {};
  const sp = schedNode  ? getParams(schedNode)  : {};
  const xp = outputNode ? getParams(outputNode) : {};

  const numClasses = Number(xp.numClasses ?? 10);
  const taskType   = String(xp.taskType ?? "classification");
  const isClf      = taskType !== "regression";

  const lr        = config.learningRate ?? Number(op.lr ?? 0.001);
  const wd        = Number(op.weightDecay ?? 0.01);
  const epochs    = config.epochs;
  const batchSize = config.batchSize;
  const device    = config.gpu ?? "cpu";

  const optimMap: Record<string, string> = {
    adamw:   `torch.optim.AdamW(model.parameters(), lr=${lr}, weight_decay=${wd})`,
    adam:    `torch.optim.Adam(model.parameters(), lr=${lr}, weight_decay=${wd})`,
    sgd:     `torch.optim.SGD(model.parameters(), lr=${lr}, momentum=0.9, weight_decay=${wd})`,
    rmsprop: `torch.optim.RMSprop(model.parameters(), lr=${lr}, weight_decay=${wd})`,
    adagrad: `torch.optim.Adagrad(model.parameters(), lr=${lr}, weight_decay=${wd})`,
  };
  const optimExpr = optimMap[op.type ?? "adamw"] ?? optimMap.adamw;

  const lossMap: Record<string, string> = {
    cross_entropy: "nn.CrossEntropyLoss()",
    mse: "nn.MSELoss()", mae: "nn.L1Loss()", bce: "nn.BCEWithLogitsLoss()",
    huber: "nn.HuberLoss()", nll: "nn.NLLLoss()",
  };
  const lossExpr = lossMap[lp.type ?? "cross_entropy"] ?? "nn.CrossEntropyLoss()";

  const schedMap: Record<string, string> = {
    cosine:      `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${epochs})`,
    linear:      `torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=${epochs})`,
    exponential: `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)`,
    constant:    "None",
    one_cycle:   `torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=${lr * 10}, total_steps=${epochs})`,
  };
  const schedExpr = schedMap[sp.type ?? "cosine"] ?? schedMap.cosine;

  const lossStep    = isClf ? "criterion(out, batch_y)" : "criterion(out.squeeze(-1), batch_y.float())";
  const valLossStep = isClf ? "criterion(vo, vy).item()" : "criterion(vo.squeeze(-1), vy.float()).item()";

  const datasetCode = genDatasetCode(dataNodeType, dataParams, batchSize, numClasses, inputFeatures);

  const graphMetaJson = JSON.stringify({
    nodeCount: nodes.length, edgeCount: edges.length,
    architecture: sorted.map((n) => ({ type: getNodeType(n), params: getParams(n) })),
  });

  return [
    `#!/usr/bin/env python3`,
    `"""canvas_model.py — Generated by FrameTrain Synapse Builder"""`,
    `import torch, torch.nn as nn, torch.nn.functional as F`,
    `from torch.utils.data import DataLoader, TensorDataset, Subset, random_split`,
    `import json, os, sys, re, traceback, shutil`,
    ``,
    `# ── Generated Model ──────────────────────────────────────────────────────────`,
    `class CanvasModel(nn.Module):`,
    `    def __init__(self):`,
    `        super().__init__()`,
    ...initLines,
    ``,
    `    def forward(self, x):`,
    ...fwdLines,
    `        return ${lastVar}`,
    ``,
    `# ── Diagnostic helper ────────────────────────────────────────────────────────`,
    `def emit_diag(etype, err, **kw):`,
    `    print("[DIAGNOSTIC_JSON] " + json.dumps({"error_type":etype,"raw_error":str(err),**kw}) + " [/DIAGNOSTIC_JSON]", flush=True)`,
    ``,
    `# ── Dataset ──────────────────────────────────────────────────────────────────`,
    datasetCode,
    `# ── Training ─────────────────────────────────────────────────────────────────`,
    `def train():`,
    `    print("[Status] Training startet...", flush=True)`,
    `    _d = "${device}"`,
    `    if _d.startswith("cuda") and not torch.cuda.is_available(): _d="cpu"; print("[Status] CUDA nicht verfuegbar, nutze CPU",flush=True)`,
    `    elif _d=="mps" and not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()): _d="cpu"; print("[Status] MPS nicht verfuegbar, nutze CPU",flush=True)`,
    `    device = torch.device(_d)`,
    `    print(f"[Status] Device: {device}", flush=True)`,
    ``,
    `    model = CanvasModel().to(device)`,
    `    total     = sum(p.numel() for p in model.parameters())`,
    `    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)`,
    `    print(f"[Status] Total Parameters: {total:,} Trainable: {trainable:,}", flush=True)`,
    ``,
    `    train_loader, val_loader = get_dataloaders(${batchSize})`,
    ``,
    `    try:`,
    `        with torch.no_grad():`,
    `            sample = next(iter(train_loader))[0][:2].to(device)`,
    `            out    = model(sample)`,
    `        print(f"[Status] Shape Test: {list(sample.shape)} -> {list(out.shape)}", flush=True)`,
    `    except RuntimeError as _e:`,
    `        _es = str(_e)`,
    `        _m  = re.search(r'\\((\\d+)x(\\d+) and (\\d+)x(\\d+)\\)', _es)`,
    `        if _m: emit_diag("shape_mismatch",_e,actual_output_features=int(_m.group(2)),expected_input_features=int(_m.group(3)))`,
    `        print(f"[ERROR] Shape Test fehlgeschlagen: {_e}", flush=True); sys.exit(1)`,
    `    except Exception as _e:`,
    `        print(f"[Status] Shape Warnung: {_e}", flush=True)`,
    ``,
    `    optimizer = ${optimExpr}`,
    `    criterion = ${lossExpr}`,
    `    scheduler = ${schedExpr}`,
    `    best_val  = float("inf")`,
    ``,
    `    for epoch in range(1, ${epochs}+1):`,
    `        model.train()`,
    `        total_loss, correct, n_samples = 0.0, 0, 0`,
    `        for batch_x, batch_y in train_loader:`,
    `            batch_x, batch_y = batch_x.to(device), batch_y.to(device)`,
    `            optimizer.zero_grad()`,
    `            try:`,
    `                out  = model(batch_x)`,
    `                loss = ${lossStep}`,
    `                loss.backward()`,
    `                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`,
    `                optimizer.step()`,
    `                total_loss += loss.item()`,
    isClf
      ? `                correct += (out.argmax(dim=-1)==batch_y).sum().item()`
      : `                pass  # regression`,
    `                n_samples += batch_y.size(0)`,
    `            except RuntimeError as _e:`,
    `                _es = str(_e)`,
    `                if "shapes cannot be multiplied" in _es or "size mismatch" in _es:`,
    `                    _m = re.search(r'\\((\\d+)x(\\d+) and (\\d+)x(\\d+)\\)', _es)`,
    `                    if _m: emit_diag("shape_mismatch",_e,actual_output_features=int(_m.group(2)),expected_input_features=int(_m.group(3)))`,
    `                    print(f"[ERROR] Shape Fehler: {_e}", flush=True); sys.exit(1)`,
    `                raise`,
    ``,
    `        avg_loss = total_loss / max(len(train_loader), 1)`,
    `        acc      = correct / max(n_samples, 1)`,
    `        model.eval()`,
    `        val_sum  = 0.0`,
    `        with torch.no_grad():`,
    `            for vx, vy in val_loader:`,
    `                vx, vy = vx.to(device), vy.to(device); vo = model(vx)`,
    `                val_sum += ${valLossStep}`,
    `        val_loss = val_sum / max(len(val_loader), 1)`,
    `        if scheduler is not None: scheduler.step()`,
    `        lr_now = optimizer.param_groups[0]["lr"]`,
    `        print(f"[Metric] epoch={epoch} loss={avg_loss:.6f} val_loss={val_loss:.6f} accuracy={acc:.6f} lr={lr_now:.8f}", flush=True)`,
    `        if val_loss < best_val: best_val = val_loss`,
    ``,
    `    out_dir = os.environ.get("OUTPUT_DIR", "/tmp/synapse_output")`,
    `    os.makedirs(out_dir, exist_ok=True)`,
    `    torch.save({`,
    `        "model_state_dict": model.state_dict(),`,
    `        "graph_metadata":   json.loads("""${graphMetaJson}"""),`,
    `        "training_config":  {"epochs":${epochs},"batchSize":${batchSize},"lr":${lr}},`,
    `    }, os.path.join(out_dir, "model.pt"))`,
    `    with open(os.path.join(out_dir, "metrics.json"), "w") as f:`,
    `        json.dump({"total_parameters":total,"trainable_parameters":trainable,"epochs_trained":${epochs},"final_loss":avg_loss,"final_val_loss":val_loss,"best_val_loss":best_val,"num_classes":${numClasses},"task_type":"${taskType}"}, f, indent=2)`,
    `    shutil.copy2(__file__, os.path.join(out_dir, "canvas_model.py"))`,
    `    print(f"[Status] model.pt -> {out_dir}/model.pt", flush=True)`,
    `    print("[Status] TRAINING_COMPLETE", flush=True)`,
    ``,
    `if __name__ == "__main__":`,
    `    try:`,
    `        train()`,
    `    except SystemExit:`,
    `        pass`,
    `    except Exception as _e:`,
    `        print(f"[ERROR] {_e}", flush=True); traceback.print_exc(); sys.exit(1)`,
  ].join("\n");
}

// ─── Compact graph summary for AI prompts ────────────────────────────────────
export function buildCompactGraphSummary(nodes: Node[], edges: Edge[]): string {
  const arch   = nodes.filter((n) => { const c = getCategory(n); return c !== "data" && c !== "training"; });
  const sorted = topoSort(arch, edges);
  return sorted.map((n) => {
    const t = getNodeType(n);
    const p = getParams(n);
    switch (t) {
      case "dense":             return `Dense(${p.inputSize}→${p.outputSize})`;
      case "conv2d":            return `Conv2D(${p.inChannels}→${p.outChannels},k=${p.kernelSize})`;
      case "lstm":              return `LSTM(${p.inputSize},h=${p.hiddenSize},L=${p.numLayers})`;
      case "attention":         return `Attention(dim=${p.embedDim},h=${p.numHeads})`;
      case "transformer_block": return `Transformer(dim=${p.embedDim},h=${p.numHeads})`;
      case "relu":              return "ReLU";
      case "gelu":              return "GELU";
      case "softmax":           return `Softmax(${p.dim})`;
      case "dropout":           return `Dropout(${p.p})`;
      case "layernorm":         return `LN(${p.normalizedShape})`;
      case "add_node":          return "Add";
      case "merge":             return `Cat(dim=${p.dim})`;
      case "pool":              return `Pool(${p.type})`;
      default:                  return t;
    }
  }).join(" → ");
}
