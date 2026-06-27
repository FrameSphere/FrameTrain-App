// ─────────────────────────────────────────────────────────────────────────────
// Synapse Builder — Node Type Definitions
// ─────────────────────────────────────────────────────────────────────────────

export type NodeCategory = 'data' | 'layer' | 'activation' | 'training' | 'math' | 'logic';

export interface PortDef {
  id: string;
  label: string;
  portType: 'tensor' | 'dataset' | 'scalar' | 'model' | 'any';
}

export interface ParamDef {
  key: string;
  label: string;
  type: 'number' | 'string' | 'boolean' | 'select' | 'code';
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  description?: string;
  default: any;
}

export interface NodeDef {
  type: string;
  label: string;
  category: NodeCategory;
  icon: string;
  description: string;
  color: string;
  inputs: PortDef[];
  outputs: PortDef[];
  params: ParamDef[];
}

// ── Aliases for the component files ───────────────────────────────────────────
export type NodeDefinition = NodeDef;
export type ParamDefinition = ParamDef;
export type PortDefinition = PortDef;

// ── Category Metadata ──────────────────────────────────────────────────────
export const CATEGORY_META: Record<NodeCategory, {
  label: string;
  color: string;
  bg: string;
  border: string;
  glow: string;
}> = {
  data:       { label: 'Data',        color: '#60a5fa', bg: 'rgba(59,130,246,0.10)',  border: 'rgba(59,130,246,0.35)',  glow: 'rgba(59,130,246,0.25)' },
  layer:      { label: 'Layers',      color: '#c084fc', bg: 'rgba(168,85,247,0.10)',  border: 'rgba(168,85,247,0.35)',  glow: 'rgba(168,85,247,0.25)' },
  activation: { label: 'Activation',  color: '#34d399', bg: 'rgba(16,185,129,0.10)',  border: 'rgba(16,185,129,0.35)',  glow: 'rgba(16,185,129,0.25)' },
  training:   { label: 'Training',    color: '#fb923c', bg: 'rgba(249,115,22,0.10)',  border: 'rgba(249,115,22,0.35)',  glow: 'rgba(249,115,22,0.25)' },
  math:       { label: 'Math',        color: '#facc15', bg: 'rgba(234,179,8,0.10)',   border: 'rgba(234,179,8,0.35)',   glow: 'rgba(234,179,8,0.25)' },
  logic:      { label: 'Logic',       color: '#f87171', bg: 'rgba(239,68,68,0.10)',   border: 'rgba(239,68,68,0.35)',   glow: 'rgba(239,68,68,0.25)' },
};

// ── Layout Constants (for handle positioning) ──────────────────────────────
export const LAYOUT = {
  HEADER_H: 44,
  BODY_PAD_T: 10,
  BODY_PAD_B: 10,
  ROW_H: 26,
  PARAM_H: 20,
  GAP: 8,
  MAX_PREVIEW_PARAMS: 3,
} as const;

export function calcInputHandleTop(idx: number): number {
  return LAYOUT.HEADER_H + LAYOUT.BODY_PAD_T + idx * LAYOUT.ROW_H + LAYOUT.ROW_H / 2;
}

export function calcOutputHandleTop(idx: number, numInputs: number, numPreviewParams: number): number {
  let top = LAYOUT.HEADER_H + LAYOUT.BODY_PAD_T;
  top += numInputs * LAYOUT.ROW_H;
  if (numInputs > 0 && numPreviewParams > 0) top += LAYOUT.GAP;
  if (numPreviewParams > 0) top += numPreviewParams * LAYOUT.PARAM_H + LAYOUT.GAP;
  top += idx * LAYOUT.ROW_H + LAYOUT.ROW_H / 2;
  return top;
}

// ── All Node Definitions ───────────────────────────────────────────────────
export const NODE_DEFINITIONS: NodeDef[] = [

  // ── DATA ──────────────────────────────────────────────────────────────────
  {
    type: 'input', label: 'Input', category: 'data', icon: '➤',
    description: 'Model input placeholder for raw data',
    color: '#60a5fa',
    inputs: [],
    outputs: [{ id: 'out', label: 'Input', portType: 'tensor' }],
    params: [
      { key: 'shape', label: 'Input Shape (e.g., -1, 3, 224, 224)', type: 'string', default: '-1, 3, 224, 224' },
      { key: 'dtype', label: 'Data Type', type: 'select', options: ['float32', 'float64', 'int32', 'int64'], default: 'float32' },
    ],
  },
  {
    type: 'csv_loader', label: 'CSV Loader', category: 'data', icon: '📊',
    description: 'Load tabular data from a CSV file',
    color: '#60a5fa',
    inputs: [],
    outputs: [{ id: 'out', label: 'Dataset', portType: 'dataset' }],
    params: [
      { key: 'separator',  label: 'Separator',    type: 'select',  options: [',', ';', '\\t', '|'], default: ',' },
      { key: 'hasHeader',  label: 'Has Header',   type: 'boolean', default: true },
      { key: 'targetCol',  label: 'Target Column',type: 'string',  default: 'label' },
      { key: 'normalize',  label: 'Normalize',    type: 'boolean', default: false },
    ],
  },
  {
    type: 'image_loader', label: 'Image Loader', category: 'data', icon: '🖼️',
    description: 'Load images from a folder structure',
    color: '#60a5fa',
    inputs: [],
    outputs: [{ id: 'out', label: 'Dataset', portType: 'dataset' }],
    params: [
      { key: 'imageSize',  label: 'Image Size',   type: 'number', min: 32, max: 1024, step: 32, default: 224 },
      { key: 'channels',   label: 'Channels',     type: 'select', options: ['1', '3', '4'], default: '3' },
      { key: 'normalize',  label: 'Normalize',    type: 'boolean', default: true },
      { key: 'cacheRam',   label: 'Cache in RAM', type: 'boolean', default: false },
    ],
  },
  {
    type: 'tokenizer', label: 'Tokenizer', category: 'data', icon: '✂️',
    description: 'Tokenize raw text into IDs',
    color: '#60a5fa',
    inputs:  [{ id: 'text', label: 'Text', portType: 'dataset' }],
    outputs: [{ id: 'tokens', label: 'Tokens', portType: 'tensor' }],
    params: [
      { key: 'maxLength',  label: 'Max Length', type: 'number', min: 16, max: 8192, step: 16, default: 512 },
      { key: 'vocabSize',  label: 'Vocab Size', type: 'number', min: 100, max: 200000, step: 1000, default: 50000 },
      { key: 'padding',    label: 'Padding',    type: 'boolean', default: true },
      { key: 'truncation', label: 'Truncation', type: 'boolean', default: true },
    ],
  },
  {
    type: 'dataset_split', label: 'Dataset Split', category: 'data', icon: '🔀',
    description: 'Split dataset into train / val / test',
    color: '#60a5fa',
    inputs:  [{ id: 'in', label: 'Dataset', portType: 'dataset' }],
    outputs: [
      { id: 'train', label: 'Train', portType: 'dataset' },
      { id: 'val',   label: 'Val',   portType: 'dataset' },
      { id: 'test',  label: 'Test',  portType: 'dataset' },
    ],
    params: [
      { key: 'trainRatio', label: 'Train %', type: 'number', min: 0.1, max: 0.9, step: 0.05, default: 0.8 },
      { key: 'valRatio',   label: 'Val %',   type: 'number', min: 0.05, max: 0.5, step: 0.05, default: 0.1 },
      { key: 'shuffle',    label: 'Shuffle', type: 'boolean', default: true },
      { key: 'seed',       label: 'Seed',    type: 'number', min: 0, max: 99999, step: 1, default: 42 },
    ],
  },
  {
    type: 'augmentation', label: 'Augmentation', category: 'data', icon: '🎲',
    description: 'Apply random data augmentations',
    color: '#60a5fa',
    inputs:  [{ id: 'in', label: 'Dataset', portType: 'dataset' }],
    outputs: [{ id: 'out', label: 'Dataset', portType: 'dataset' }],
    params: [
      { key: 'flip',       label: 'Horiz. Flip',    type: 'boolean', default: true },
      { key: 'rotate',     label: 'Max Rotation °', type: 'number', min: 0, max: 180, step: 5, default: 15 },
      { key: 'brightness', label: 'Brightness',     type: 'number', min: 0, max: 1, step: 0.05, default: 0.2 },
      { key: 'noise',      label: 'Gaussian Noise', type: 'boolean', default: false },
    ],
  },

  // ── LAYERS ────────────────────────────────────────────────────────────────
  {
    type: 'dense', label: 'Dense', category: 'layer', icon: '▦',
    description: 'Fully connected (linear) layer',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'inputSize',   label: 'Input Size',   type: 'number', min: 1, max: 65536, step: 1, default: 128 },
      { key: 'outputSize',  label: 'Output Size',  type: 'number', min: 1, max: 65536, step: 1, default: 256 },
      { key: 'bias',        label: 'Bias',         type: 'boolean', default: true },
      { key: 'initializer', label: 'Initializer',  type: 'select', options: ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'zeros', 'ones'], default: 'xavier_uniform' },
    ],
  },
  {
    type: 'conv2d', label: 'Conv2D', category: 'layer', icon: '◫',
    description: '2D Convolutional layer',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'inChannels',  label: 'In Channels',  type: 'number', min: 1, max: 2048, step: 1, default: 3 },
      { key: 'outChannels', label: 'Out Channels', type: 'number', min: 1, max: 2048, step: 1, default: 64 },
      { key: 'kernelSize',  label: 'Kernel Size',  type: 'select', options: ['1', '3', '5', '7', '11'], default: '3' },
      { key: 'stride',      label: 'Stride',       type: 'number', min: 1, max: 8, step: 1, default: 1 },
      { key: 'padding',     label: 'Padding',      type: 'select', options: ['0', '1', '2', 'same'], default: '1' },
      { key: 'groups',      label: 'Groups',       type: 'number', min: 1, max: 64, step: 1, default: 1 },
    ],
  },
  {
    type: 'embedding', label: 'Embedding', category: 'layer', icon: '⌘',
    description: 'Learnable token embedding table',
    color: '#c084fc',
    inputs:  [{ id: 'tokens', label: 'Tokens', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Embeddings', portType: 'tensor' }],
    params: [
      { key: 'vocabSize',    label: 'Vocab Size',    type: 'number', min: 100, max: 200000, step: 1000, default: 50000 },
      { key: 'embeddingDim', label: 'Embedding Dim', type: 'number', min: 8, max: 4096, step: 8, default: 512 },
      { key: 'paddingIdx',   label: 'Padding Idx',   type: 'number', min: -1, max: 10, step: 1, default: 0 },
    ],
  },
  {
    type: 'lstm', label: 'LSTM', category: 'layer', icon: '↺',
    description: 'Long Short-Term Memory layer',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [
      { id: 'out',    label: 'Output', portType: 'tensor' },
      { id: 'hidden', label: 'Hidden', portType: 'tensor' },
    ],
    params: [
      { key: 'inputSize',    label: 'Input Size',    type: 'number', min: 1, max: 65536, step: 1, default: 256 },
      { key: 'hiddenSize',   label: 'Hidden Size',   type: 'number', min: 1, max: 65536, step: 1, default: 512 },
      { key: 'numLayers',    label: 'Num Layers',    type: 'number', min: 1, max: 32, step: 1, default: 2 },
      { key: 'bidirectional',label: 'Bidirectional', type: 'boolean', default: false },
      { key: 'dropout',      label: 'Dropout',       type: 'number', min: 0, max: 0.9, step: 0.05, default: 0.1 },
    ],
  },
  {
    type: 'attention', label: 'Attention', category: 'layer', icon: '◎',
    description: 'Multi-head self-attention',
    color: '#c084fc',
    inputs: [
      { id: 'query', label: 'Query', portType: 'tensor' },
      { id: 'key',   label: 'Key',   portType: 'tensor' },
      { id: 'value', label: 'Value', portType: 'tensor' },
    ],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'embedDim', label: 'Embed Dim', type: 'number', min: 8, max: 4096, step: 8, default: 512 },
      { key: 'numHeads', label: 'Num Heads', type: 'number', min: 1, max: 64, step: 1, default: 8 },
      { key: 'dropout',  label: 'Dropout',   type: 'number', min: 0, max: 0.9, step: 0.05, default: 0.1 },
      { key: 'causal',   label: 'Causal Mask', type: 'boolean', default: false },
    ],
  },
  {
    type: 'transformer_block', label: 'Transformer Block', category: 'layer', icon: '⬡',
    description: 'Full transformer encoder block (Attention + FFN)',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'embedDim',     label: 'Embed Dim',      type: 'number', min: 8, max: 4096, step: 8, default: 512 },
      { key: 'numHeads',     label: 'Num Heads',      type: 'number', min: 1, max: 64, step: 1, default: 8 },
      { key: 'ffnDim',       label: 'FFN Hidden Dim', type: 'number', min: 8, max: 16384, step: 8, default: 2048 },
      { key: 'dropout',      label: 'Dropout',        type: 'number', min: 0, max: 0.9, step: 0.05, default: 0.1 },
      { key: 'contextLength',label: 'Context Length', type: 'number', min: 8, max: 32768, step: 8, default: 1024 },
    ],
  },
  {
    type: 'layernorm', label: 'LayerNorm', category: 'layer', icon: '≡',
    description: 'Layer normalization',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'normalizedShape', label: 'Normalized Shape', type: 'number', min: 1, max: 65536, step: 1, default: 512 },
      { key: 'eps',             label: 'Epsilon',          type: 'number', min: 0.0000001, max: 0.01, step: 0.0000001, default: 0.00001 },
      { key: 'affine',          label: 'Affine',           type: 'boolean', default: true },
    ],
  },
  {
    type: 'batchnorm', label: 'BatchNorm', category: 'layer', icon: '≋',
    description: 'Batch normalization',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'numFeatures', label: 'Num Features', type: 'number', min: 1, max: 65536, step: 1, default: 64 },
      { key: 'eps',         label: 'Epsilon',      type: 'number', min: 0.0000001, max: 0.01, step: 0.0000001, default: 0.00001 },
      { key: 'momentum',    label: 'Momentum',     type: 'number', min: 0, max: 1, step: 0.01, default: 0.1 },
      { key: 'affine',      label: 'Affine',       type: 'boolean', default: true },
    ],
  },
  {
    type: 'dropout', label: 'Dropout', category: 'layer', icon: '∿',
    description: 'Dropout regularization',
    color: '#c084fc',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'p',       label: 'Drop Prob.', type: 'number', min: 0, max: 0.99, step: 0.05, default: 0.1 },
      { key: 'inplace', label: 'In-place',   type: 'boolean', default: false },
    ],
  },

  // ── ACTIVATIONS ───────────────────────────────────────────────────────────
  {
    type: 'relu', label: 'ReLU', category: 'activation', icon: '⌐',
    description: 'Rectified Linear Unit — max(0, x)',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'inplace', label: 'In-place', type: 'boolean', default: false },
    ],
  },
  {
    type: 'gelu', label: 'GELU', category: 'activation', icon: '⌇',
    description: 'Gaussian Error Linear Unit',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'approximate', label: 'Approximation', type: 'select', options: ['none', 'tanh'], default: 'none' },
    ],
  },
  {
    type: 'sigmoid', label: 'Sigmoid', category: 'activation', icon: 'σ',
    description: 'Sigmoid — maps to (0, 1)',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },
  {
    type: 'softmax', label: 'Softmax', category: 'activation', icon: '∑',
    description: 'Softmax — probability distribution',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'dim', label: 'Dim', type: 'number', min: -3, max: 3, step: 1, default: -1 },
    ],
  },
  {
    type: 'tanh', label: 'Tanh', category: 'activation', icon: '∩',
    description: 'Hyperbolic tangent — maps to (−1, 1)',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },
  {
    type: 'leaky_relu', label: 'Leaky ReLU', category: 'activation', icon: '⌐',
    description: 'ReLU with small negative slope',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'negativeSlope', label: 'Negative Slope', type: 'number', min: 0, max: 0.5, step: 0.01, default: 0.01 },
    ],
  },
  {
    type: 'silu', label: 'SiLU / Swish', category: 'activation', icon: '~',
    description: 'Sigmoid Linear Unit — used in LLaMA, GPT-4',
    color: '#34d399',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },

  // ── TRAINING ──────────────────────────────────────────────────────────────
  {
    type: 'optimizer', label: 'Optimizer', category: 'training', icon: '⚙',
    description: 'Gradient-based optimizer configuration',
    color: '#fb923c',
    inputs:  [],
    outputs: [{ id: 'optimizer', label: 'Optimizer', portType: 'any' }],
    params: [
      { key: 'type',        label: 'Type',          type: 'select', options: ['adamw', 'adam', 'sgd', 'rmsprop', 'adagrad', 'adafactor'], default: 'adamw' },
      { key: 'lr',          label: 'Learning Rate', type: 'number', min: 0.000001, max: 1, step: 0.00001, default: 0.001 },
      { key: 'weightDecay', label: 'Weight Decay',  type: 'number', min: 0, max: 0.5, step: 0.001, default: 0.01 },
      { key: 'clipGrad',    label: 'Gradient Clip', type: 'number', min: 0, max: 10, step: 0.1, default: 1.0 },
    ],
  },
  {
    type: 'loss', label: 'Loss', category: 'training', icon: '↘',
    description: 'Loss function for training',
    color: '#fb923c',
    inputs: [
      { id: 'pred',    label: 'Predictions', portType: 'tensor' },
      { id: 'targets', label: 'Targets',     portType: 'tensor' },
    ],
    outputs: [{ id: 'loss', label: 'Loss', portType: 'scalar' }],
    params: [
      { key: 'type',           label: 'Loss Type',       type: 'select', options: ['cross_entropy', 'mse', 'mae', 'bce', 'huber', 'nll', 'kl_div', 'focal'], default: 'cross_entropy' },
      { key: 'reduction',      label: 'Reduction',       type: 'select', options: ['mean', 'sum', 'none'], default: 'mean' },
      { key: 'labelSmoothing', label: 'Label Smoothing', type: 'number', min: 0, max: 0.5, step: 0.01, default: 0 },
    ],
  },
  {
    type: 'scheduler', label: 'LR Scheduler', category: 'training', icon: '📉',
    description: 'Learning rate schedule',
    color: '#fb923c',
    inputs:  [{ id: 'optimizer', label: 'Optimizer', portType: 'any' }],
    outputs: [{ id: 'scheduler', label: 'Scheduler', portType: 'any' }],
    params: [
      { key: 'type',        label: 'Schedule',     type: 'select', options: ['cosine', 'linear', 'constant', 'one_cycle', 'exponential', 'polynomial'], default: 'cosine' },
      { key: 'warmupSteps', label: 'Warmup Steps', type: 'number', min: 0, max: 10000, step: 10, default: 100 },
      { key: 'minLr',       label: 'Min LR',       type: 'number', min: 0, max: 0.01, step: 0.0000001, default: 0.000001 },
    ],
  },
  {
    type: 'output_node', label: 'Output', category: 'training', icon: '🏁',
    description: 'Model output — final prediction head',
    color: '#fb923c',
    inputs:  [{ id: 'logits', label: 'Logits', portType: 'tensor' }],
    outputs: [],
    params: [
      { key: 'numClasses', label: 'Num Classes', type: 'number', min: 1, max: 100000, step: 1, default: 10 },
      { key: 'taskType',   label: 'Task Type',   type: 'select', options: ['classification', 'regression', 'generation', 'segmentation', 'detection'], default: 'classification' },
    ],
  },

  // ── MATH ──────────────────────────────────────────────────────────────────
  {
    type: 'add_node', label: 'Add', category: 'math', icon: '+',
    description: 'Element-wise addition (e.g. skip connections)',
    color: '#facc15',
    inputs: [
      { id: 'a', label: 'A', portType: 'tensor' },
      { id: 'b', label: 'B', portType: 'tensor' },
    ],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },
  {
    type: 'multiply_node', label: 'Multiply', category: 'math', icon: '×',
    description: 'Element-wise (Hadamard) multiplication',
    color: '#facc15',
    inputs: [
      { id: 'a', label: 'A', portType: 'tensor' },
      { id: 'b', label: 'B', portType: 'tensor' },
    ],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },
  {
    type: 'matmul', label: 'MatMul', category: 'math', icon: '⋅',
    description: 'Matrix multiplication (batch-aware)',
    color: '#facc15',
    inputs: [
      { id: 'a', label: 'A', portType: 'tensor' },
      { id: 'b', label: 'B', portType: 'tensor' },
    ],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [],
  },
  {
    type: 'normalize', label: 'Normalize', category: 'math', icon: '∥',
    description: 'L-p normalize along a dimension',
    color: '#facc15',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'p',   label: 'P-norm', type: 'number', min: 1, max: 4, step: 1, default: 2 },
      { key: 'dim', label: 'Dim',    type: 'number', min: -3, max: 3, step: 1, default: -1 },
    ],
  },
  {
    type: 'reshape', label: 'Reshape', category: 'math', icon: '⬚',
    description: 'Reshape tensor to new dimensions',
    color: '#facc15',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'shape', label: 'Shape  e.g. -1, 512', type: 'string', default: '-1, 512' },
    ],
  },
  {
    type: 'transpose', label: 'Transpose', category: 'math', icon: 'ᵀ',
    description: 'Transpose two dimensions',
    color: '#facc15',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'dim0', label: 'Dim 0', type: 'number', min: -3, max: 3, step: 1, default: -2 },
      { key: 'dim1', label: 'Dim 1', type: 'number', min: -3, max: 3, step: 1, default: -1 },
    ],
  },

  // ── LOGIC ─────────────────────────────────────────────────────────────────
  {
    type: 'merge', label: 'Merge', category: 'logic', icon: '⤵',
    description: 'Concatenate tensors along a dimension',
    color: '#f87171',
    inputs: [
      { id: 'a', label: 'A', portType: 'tensor' },
      { id: 'b', label: 'B', portType: 'tensor' },
    ],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'dim', label: 'Concat Dim', type: 'number', min: -3, max: 3, step: 1, default: -1 },
    ],
  },
  {
    type: 'split_node', label: 'Split', category: 'logic', icon: '⤴',
    description: 'Split tensor into equal chunks',
    color: '#f87171',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [
      { id: 'a', label: 'A', portType: 'tensor' },
      { id: 'b', label: 'B', portType: 'tensor' },
    ],
    params: [
      { key: 'chunks', label: 'Chunks', type: 'number', min: 2, max: 16, step: 1, default: 2 },
      { key: 'dim',    label: 'Dim',    type: 'number', min: -3, max: 3, step: 1, default: -1 },
    ],
  },
  {
    type: 'pool', label: 'Pooling', category: 'logic', icon: '▼',
    description: 'Global or adaptive pooling',
    color: '#f87171',
    inputs:  [{ id: 'in', label: 'Input', portType: 'tensor' }],
    outputs: [{ id: 'out', label: 'Output', portType: 'tensor' }],
    params: [
      { key: 'type',   label: 'Pool Type', type: 'select', options: ['global_avg', 'global_max', 'avg_2d', 'max_2d', 'adaptive_avg'], default: 'global_avg' },
      { key: 'stride', label: 'Stride',    type: 'number', min: 1, max: 8, step: 1, default: 2 },
    ],
  },
];

// ── Convenience lookups ────────────────────────────────────────────────────
export const CATEGORY_NODES: Record<NodeCategory, NodeDef[]> = {
  data:       NODE_DEFINITIONS.filter(n => n.category === 'data'),
  layer:      NODE_DEFINITIONS.filter(n => n.category === 'layer'),
  activation: NODE_DEFINITIONS.filter(n => n.category === 'activation'),
  training:   NODE_DEFINITIONS.filter(n => n.category === 'training'),
  math:       NODE_DEFINITIONS.filter(n => n.category === 'math'),
  logic:      NODE_DEFINITIONS.filter(n => n.category === 'logic'),
};

// Capitalized-key map used by NodeLibrary (display names)
export const NODE_CATEGORIES: Record<string, NodeDef[]> = {
  Data:        CATEGORY_NODES.data,
  Layers:      CATEGORY_NODES.layer,
  Activations: CATEGORY_NODES.activation,
  Training:    CATEGORY_NODES.training,
  Math:        CATEGORY_NODES.math,
  Logic:       CATEGORY_NODES.logic,
};

export function getNodeDef(type: string): NodeDef | undefined {
  return NODE_DEFINITIONS.find(n => n.type === type);
}

export function buildDefaultParams(def: NodeDef): Record<string, any> {
  const params: Record<string, any> = {};
  for (const p of def.params) params[p.key] = p.default;
  return params;
}

export type TemplateId = 'empty' | 'mlp' | 'cnn' | 'transformer' | 'lstm_cls' | 'resnet_skip';

export const TEMPLATES: Record<TemplateId, { label: string; icon: string; description: string }> = {
  empty:       { label: 'Empty Canvas',        icon: '⬜', description: 'Start from scratch' },
  mlp:         { label: 'MLP Classifier',      icon: '▦', description: 'Dense → ReLU → Dense → Softmax' },
  cnn:         { label: 'CNN Classifier',      icon: '◫', description: 'Conv2D blocks → Dense head' },
  transformer: { label: 'Transformer Encoder', icon: '⬡', description: 'Embedding → Transformer Block → Output' },
  lstm_cls:    { label: 'LSTM Classifier',     icon: '↺', description: 'Embedding → LSTM → Dense → Output' },
  resnet_skip: { label: 'ResNet Skip Block',   icon: '+', description: 'Conv + Skip connection via Add' },
};
