// Bausteine fuer die Werkzeug-Dateien: Definition, Ergebnisse, Argumente,
// Schema-Pruefung. Bewusst ohne Abhaengigkeiten, damit die Domaenen-Dateien
// kurz bleiben.

import { invoke } from '@tauri-apps/api/core';
import type {
  ApprovalLevel, CallToolResult, JsonSchema, McpCallContext, McpDomain, McpTool,
  ToolAnnotations, ToolArgs,
} from './types';
import { MCP_DOMAINS } from './types';

// ── Fehler ──────────────────────────────────────────────────────────────────

/** Falsche oder fehlende Argumente — landet als isError beim Client. */
export class ToolArgError extends Error {}

/** Fachlicher Fehler mit verstaendlichem Text (kein Stacktrace). */
export class ToolFailure extends Error {}

// ── Ergebnisse ──────────────────────────────────────────────────────────────

/** Strukturierte Antwort: Arrays kommen unter `items`, weil MCP ein Objekt verlangt. */
export function ok(text: string, data?: unknown): CallToolResult {
  const result: CallToolResult = { content: [{ type: 'text', text }] };
  if (data !== undefined) result.structuredContent = toStructured(data);
  return result;
}

export function fail(text: string, data?: unknown): CallToolResult {
  const result: CallToolResult = { content: [{ type: 'text', text }], isError: true };
  if (data !== undefined) result.structuredContent = toStructured(data);
  return result;
}

export function toStructured(data: unknown): Record<string, unknown> {
  if (data && typeof data === 'object' && !Array.isArray(data)) return data as Record<string, unknown>;
  if (Array.isArray(data)) return { items: data, count: data.length };
  return { value: data ?? null };
}

export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  try { return JSON.stringify(e); } catch { return String(e); }
}

// ── Tauri ───────────────────────────────────────────────────────────────────

/** invoke mit lesbarer Fehlermeldung (Tauri liefert Strings als Fehler). */
export async function call<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(cmd, args);
  } catch (e) {
    throw new ToolFailure(`${cmd}: ${errorText(e)}`);
  }
}

// ── Definition ──────────────────────────────────────────────────────────────

export interface ToolSpec {
  name: string;
  title: string;
  description: string;
  /** Eigenschaften des Eingabe-Objekts; leer = keine Argumente. */
  properties?: Record<string, JsonSchema>;
  required?: string[];
  approval?: ApprovalLevel;
  annotations?: ToolAnnotations;
  requiresUser?: boolean;
  run: (args: ToolArgs, ctx: McpCallContext) => Promise<CallToolResult>;
}

export function domainOf(name: string): McpDomain {
  const prefix = name.split('_')[0] as McpDomain;
  if (!(MCP_DOMAINS as readonly string[]).includes(prefix)) {
    throw new Error(`Werkzeugname ohne gueltiges Domaenen-Praefix: ${name}`);
  }
  return prefix;
}

/**
 * Baut ein Werkzeug nach Vertrag. Standard: lesend, keine Freigabe.
 * `annotations` werden mit sinnvollen Vorgaben aus der Freigabestufe gefuellt.
 */
export function defineTool(spec: ToolSpec): McpTool {
  const domain = domainOf(spec.name);
  const approval = spec.approval ?? 'none';
  const readOnly = approval === 'none';
  return {
    name: spec.name,
    title: spec.title,
    description: spec.description,
    inputSchema: {
      type: 'object',
      properties: spec.properties ?? {},
      ...(spec.required && spec.required.length ? { required: spec.required } : {}),
      additionalProperties: false,
    },
    annotations: {
      readOnlyHint: readOnly,
      destructiveHint: false,
      openWorldHint: false,
      ...spec.annotations,
    },
    _meta: { 'frametrain/domain': domain, 'frametrain/approval': approval },
    requiresUser: spec.requiresUser ?? domain !== 'system',
    run: spec.run,
  };
}

// ── Schema-Kuerzel ──────────────────────────────────────────────────────────

export const S = {
  str: (description: string, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'string', description, ...extra }),
  num: (description: string, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'number', description, ...extra }),
  int: (description: string, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'integer', description, ...extra }),
  bool: (description: string, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'boolean', description, ...extra }),
  strArr: (description: string, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'array', items: { type: 'string' }, description, ...extra }),
  enumOf: (description: string, values: readonly string[], extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'string', enum: values, description, ...extra }),
  obj: (description: string, properties?: Record<string, JsonSchema>, extra: Partial<JsonSchema> = {}): JsonSchema => (
    properties ? { type: 'object', description, properties, ...extra } : { type: 'object', description, ...extra }
  ),
  arr: (description: string, items: JsonSchema, extra: Partial<JsonSchema> = {}): JsonSchema => ({ type: 'array', description, items, ...extra }),
};

// ── Argumente lesen ─────────────────────────────────────────────────────────

export function reqStr(args: ToolArgs, key: string): string {
  const v = args[key];
  if (typeof v !== 'string' || v.trim() === '') throw new ToolArgError(`Argument "${key}" fehlt oder ist leer.`);
  return v;
}

export function optStr(args: ToolArgs, key: string): string | undefined {
  const v = args[key];
  return typeof v === 'string' && v !== '' ? v : undefined;
}

export function optNum(args: ToolArgs, key: string, fallback?: number): number | undefined {
  const v = args[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

export function reqNum(args: ToolArgs, key: string): number {
  const v = optNum(args, key);
  if (v === undefined) throw new ToolArgError(`Argument "${key}" muss eine Zahl sein.`);
  return v;
}

export function optBool(args: ToolArgs, key: string, fallback = false): boolean {
  const v = args[key];
  return typeof v === 'boolean' ? v : fallback;
}

export function strArr(args: ToolArgs, key: string, required = false): string[] {
  const v = args[key];
  if (v === undefined && !required) return [];
  if (!Array.isArray(v) || v.some((x) => typeof x !== 'string')) {
    throw new ToolArgError(`Argument "${key}" muss eine Liste von Texten sein.`);
  }
  if (required && v.length === 0) throw new ToolArgError(`Argument "${key}" darf nicht leer sein.`);
  return v as string[];
}

export function optObj(args: ToolArgs, key: string): Record<string, unknown> | undefined {
  const v = args[key];
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

// ── Schema-Pruefung ─────────────────────────────────────────────────────────

/**
 * Prueft Argumente gegen das Eingabe-Schema (die Teilmenge, die die Werkzeuge
 * benutzen). Rueckgabe: Liste lesbarer Fehler, leer = gueltig.
 */
export function validateArgs(schema: JsonSchema, value: unknown, path = 'args'): string[] {
  const errors: string[] = [];
  const t = schema.type;
  if (t) {
    const actual = Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value;
    const matches =
      t === 'integer' ? typeof value === 'number' && Number.isInteger(value)
        : t === 'number' ? typeof value === 'number' && Number.isFinite(value)
          : t === actual;
    if (!matches) {
      errors.push(`${path}: erwartet ${t}, bekommen ${actual}`);
      return errors;
    }
  }
  if (schema.enum && !schema.enum.includes(value as never)) {
    errors.push(`${path}: erlaubt sind ${schema.enum.map((e) => JSON.stringify(e)).join(', ')}`);
  }
  if (typeof value === 'number') {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${path}: mindestens ${schema.minimum}`);
    if (schema.maximum !== undefined && value > schema.maximum) errors.push(`${path}: hoechstens ${schema.maximum}`);
  }
  if (typeof value === 'string') {
    if (schema.minLength !== undefined && value.length < schema.minLength) errors.push(`${path}: mindestens ${schema.minLength} Zeichen`);
    if (schema.maxLength !== undefined && value.length > schema.maxLength) errors.push(`${path}: hoechstens ${schema.maxLength} Zeichen`);
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${path}: mindestens ${schema.minItems} Eintraege`);
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${path}: hoechstens ${schema.maxItems} Eintraege`);
    if (schema.items) value.forEach((v, i) => errors.push(...validateArgs(schema.items!, v, `${path}[${i}]`)));
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (obj[key] === undefined) errors.push(`${path}.${key}: fehlt`);
    }
    const props = schema.properties ?? {};
    for (const [key, v] of Object.entries(obj)) {
      if (v === undefined) continue;
      const sub = props[key];
      if (sub) errors.push(...validateArgs(sub, v, `${path}.${key}`));
      else if (schema.additionalProperties === false) errors.push(`${path}.${key}: unbekanntes Argument`);
      else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
        errors.push(...validateArgs(schema.additionalProperties, v, `${path}.${key}`));
      }
    }
  }
  return errors;
}

// ── Kleinkram ───────────────────────────────────────────────────────────────

export function formatBytes(n: number | undefined | null): string {
  if (!n || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Kuerzt lange Texte fuer `text`-Zusammenfassungen. */
export function clip(s: string, max = 4000): string {
  return s.length > max ? `${s.slice(0, max)}\n… (${s.length - max} more characters)` : s;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
