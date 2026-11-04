// Sammelt die Werkzeuge aller Domaenen und baut das Manifest fuer Rust.

import type { McpTool, Tool } from './types';
import { systemTools } from './tools/system';
import { modelsTools } from './tools/models';
import { datasetsTools } from './tools/datasets';
import { studioTools } from './tools/studio';
import { trainingTools } from './tools/training';
import { synapseTools } from './tools/synapse';
import { devtrainTools } from './tools/devtrain';
import { labTools } from './tools/lab';
import { testsTools } from './tools/tests';
import { analysisTools } from './tools/analysis';
import { hostingTools } from './tools/hosting';

export const ALL_TOOLS: McpTool[] = [
  ...systemTools,
  ...modelsTools,
  ...datasetsTools,
  ...studioTools,
  ...trainingTools,
  ...synapseTools,
  ...devtrainTools,
  ...labTools,
  ...testsTools,
  ...analysisTools,
  ...hostingTools,
];

export interface ToolRegistry {
  tools: McpTool[];
  get(name: string): McpTool | undefined;
  manifest(): Tool[];
}

export function createRegistry(tools: McpTool[] = ALL_TOOLS): ToolRegistry {
  const byName = new Map(tools.map((t) => [t.name, t]));
  return {
    tools,
    get: (name) => byName.get(name),
    // Nur die Vertragsfelder — `run` und `requiresUser` bleiben in der Webview.
    manifest: () => tools.map(({ name, title, description, inputSchema, annotations, _meta }) => ({
      name, title, description, inputSchema, annotations, _meta,
    })),
  };
}

export const registry = createRegistry();

/** Stabiler Hash des Manifests (fuer Heartbeat/manifestHash auf Rust-Seite). */
export function manifestHash(tools: Tool[]): string {
  const s = JSON.stringify(tools);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
