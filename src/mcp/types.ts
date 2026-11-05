// MCP-Werkzeuge in der Webview — gemeinsame Typen.
//
// Format verbindlich nach docs/mcp/CONTRACT.md, Abschnitt 1 und 3:
// ein Werkzeug ist eine MCP-Tool-Definition plus `_meta` mit Domaene und
// Freigabestufe. Die Rust-Seite (Strang B) reicht Definitionen und Ergebnisse
// unveraendert durch.

export type ApprovalLevel = 'none' | 'once' | 'always';

export const MCP_DOMAINS = [
  'system', 'models', 'datasets', 'studio', 'training', 'synapse',
  'devtrain', 'lab', 'tests', 'analysis', 'hosting',
] as const;
export type McpDomain = typeof MCP_DOMAINS[number];

/** Teilmenge von JSON Schema, die die Werkzeuge benutzen. */
export interface JsonSchema {
  type?: 'object' | 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'null';
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean | JsonSchema;
  items?: JsonSchema;
  enum?: readonly (string | number | boolean)[];
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  minLength?: number;
  maxLength?: number;
  default?: unknown;
}

export interface ToolAnnotations {
  title?: string;
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

export interface ToolMeta {
  'frametrain/domain': McpDomain;
  'frametrain/approval': ApprovalLevel;
}

/** Definition, wie sie im Manifest an Rust geht (ohne `run`). */
export interface Tool {
  name: string;
  title: string;
  description: string;
  inputSchema: JsonSchema & { type: 'object' };
  annotations: ToolAnnotations;
  _meta: ToolMeta;
}

export interface TextContent {
  type: 'text';
  text: string;
}

/** MCP CallToolResult. `structuredContent` ist laut Spezifikation ein Objekt. */
export interface CallToolResult {
  content: TextContent[];
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
}

export interface McpClientInfo {
  name: string;
  version?: string;
}

export type McpCallSource = 'local' | 'remote';

/** Payload des Events `mcp://call` (Vertrag Abschnitt 3). */
export interface McpCallEvent {
  callId: string;
  tool: string;
  args: Record<string, unknown>;
  source: McpCallSource;
  client: McpClientInfo;
}

export interface McpCallContext {
  callId: string;
  client: McpClientInfo;
  source: McpCallSource;
  /** Angemeldeter Nutzer der App, null vor dem Login. */
  userId: string | null;
}

export type ToolArgs = Record<string, unknown>;

/** Werkzeug samt Ausfuehrung — lebt nur in der Webview. */
export interface McpTool extends Tool {
  /** false: Werkzeug laeuft auch ohne angemeldeten Nutzer (nur system_*). */
  requiresUser: boolean;
  run(args: ToolArgs, ctx: McpCallContext): Promise<CallToolResult>;
}

/** Antwort aus dem Freigabe-Dialog. */
export type ApprovalDecision = 'allow' | 'deny' | 'always';

export interface ApprovalRequest {
  id: string;
  tool: Tool;
  args: ToolArgs;
  client: McpClientInfo;
  source: McpCallSource;
  /** Dialog bietet "immer erlauben" an (nicht bei destruktiven Werkzeugen). */
  canRemember: boolean;
}
