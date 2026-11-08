// Manifest nach Vertrag (docs/mcp/CONTRACT.md, Abschnitt 1):
// eindeutige snake_case-Namen mit Domaenen-Praefix, gueltige Schemas,
// _meta mit Domaene + Freigabe, jede Domaene vertreten.

import { describe, expect, it } from 'vitest';
import { ALL_TOOLS, createRegistry, manifestHash, registry } from '../registry';
import { MCP_DOMAINS, type JsonSchema } from '../types';
import { validateArgs } from '../util';

const TYPES = new Set(['object', 'string', 'number', 'integer', 'boolean', 'array', 'null']);

/** Prueft ein Schema rekursiv auf die Teilmenge, die wir benutzen. */
function schemaProblems(schema: JsonSchema, path: string): string[] {
  const out: string[] = [];
  if (schema.type && !TYPES.has(schema.type)) out.push(`${path}: unknown type ${schema.type}`);
  if (schema.type === 'array' && !schema.items) out.push(`${path}: array without items`);
  if (schema.enum && (!Array.isArray(schema.enum) || schema.enum.length === 0)) out.push(`${path}: empty enum`);
  if (schema.required) {
    for (const r of schema.required) if (!schema.properties?.[r]) out.push(`${path}: required "${r}" not in properties`);
  }
  for (const [k, sub] of Object.entries(schema.properties ?? {})) out.push(...schemaProblems(sub, `${path}.${k}`));
  if (schema.items) out.push(...schemaProblems(schema.items, `${path}[]`));
  return out;
}

describe('MCP-Manifest', () => {
  const manifest = registry.manifest();

  it('hat eindeutige Namen nach Vertrag', () => {
    const names = manifest.map((t) => t.name);
    expect(new Set(names).size).toBe(names.length);
    for (const n of names) {
      expect(n).toMatch(/^[a-z][a-z0-9_]*$/);
      expect(n.length).toBeLessThanOrEqual(64);
    }
  });

  it('jedes Werkzeug traegt Domaenen-Praefix und passendes _meta', () => {
    for (const t of manifest) {
      const domain = t._meta['frametrain/domain'];
      expect(MCP_DOMAINS).toContain(domain);
      expect(t.name.startsWith(`${domain}_`)).toBe(true);
      expect(['none', 'once', 'always']).toContain(t._meta['frametrain/approval']);
    }
  });

  it('jede Domaene ist vertreten', () => {
    const seen = new Set(manifest.map((t) => t._meta['frametrain/domain']));
    for (const d of MCP_DOMAINS) expect(seen.has(d), `Domaene ${d}`).toBe(true);
  });

  it('Schemas sind gueltig und Objekt-Schemas ohne Zusatzfelder', () => {
    for (const t of manifest) {
      expect(t.inputSchema.type).toBe('object');
      expect(t.inputSchema.additionalProperties).toBe(false);
      expect(schemaProblems(t.inputSchema, t.name)).toEqual([]);
      // Ein leerer Aufruf scheitert nur an fehlenden Pflichtfeldern, nie am Schema selbst.
      const errs = validateArgs(t.inputSchema, {});
      expect(errs.every((e) => e.endsWith('missing'))).toBe(true);
    }
  });

  it('Beschreibungen sind englisch und aussagekraeftig, Titel vorhanden', () => {
    for (const t of manifest) {
      expect(t.title.length).toBeGreaterThan(2);
      expect(t.description.length).toBeGreaterThan(40);
      // Grober Sprach-Check: keine Umlaute in LLM-Beschreibungen
      expect(t.description).not.toMatch(/[äöüÄÖÜß]/);
    }
  });

  it('ueberschreibende Werkzeuge sind fuer Claude/ChatGPT destruktiv, nur Loeschen ist unwiderruflich', () => {
    const overwriting = ['studio_project_update', 'studio_edit_text', 'studio_set_annotation', 'synapse_graph_set',
      'synapse_graph_edit', 'synapse_open_model', 'synapse_save', 'devtrain_script_set', 'devtrain_test_stop',
      'devtrain_stop', 'tests_stop', 'training_stop', 'system_job_cancel'];
    for (const n of overwriting) {
      const t = manifest.find((x) => x.name === n);
      expect(t?.annotations.destructiveHint, n).toBe(true);
      expect(t?._meta['frametrain/irreversible'], n).toBeUndefined();
    }
    for (const t of manifest) {
      if (t._meta['frametrain/irreversible']) expect(t.name).toMatch(/_delete/);
    }
  });

  it('loeschende Werkzeuge fragen immer und sind als destruktiv markiert', () => {
    for (const t of manifest.filter((x) => /_delete/.test(x.name))) {
      expect(t._meta['frametrain/approval']).toBe('always');
      expect(t.annotations.destructiveHint).toBe(true);
      expect(t._meta['frametrain/irreversible']).toBe(true);
    }
  });

  it('lesende Werkzeuge sind readOnly, Startwerkzeuge fragen', () => {
    for (const t of manifest) {
      if (t._meta['frametrain/approval'] === 'none') expect(t.annotations.readOnlyHint).toBe(true);
    }
    for (const n of ['training_start', 'synapse_train', 'devtrain_start', 'tests_start']) {
      expect(registry.get(n)?._meta['frametrain/approval']).toBe('always');
    }
  });

  it('Manifest enthaelt nur Vertragsfelder (kein run)', () => {
    for (const t of manifest) {
      expect(Object.keys(t).sort()).toEqual(['_meta', 'annotations', 'description', 'inputSchema', 'name', 'title']);
    }
    expect(() => JSON.stringify(manifest)).not.toThrow();
  });

  it('Hash ist stabil und reagiert auf Aenderungen', () => {
    expect(manifestHash(manifest)).toBe(manifestHash(registry.manifest()));
    const smaller = createRegistry(ALL_TOOLS.slice(1)).manifest();
    expect(manifestHash(smaller)).not.toBe(manifestHash(manifest));
  });

  it('Schwerpunkte des Vertrags sind abgedeckt', () => {
    const want = [
      'system_ping', 'system_info', 'models_list', 'models_download_hf', 'datasets_list', 'datasets_preview',
      'studio_project_create', 'studio_add_texts', 'studio_set_annotation', 'studio_export', 'studio_fetch_web',
      'training_start', 'training_status', 'synapse_node_catalog', 'synapse_graph_get', 'synapse_graph_set',
      'synapse_graph_edit', 'synapse_validate', 'synapse_generate_code', 'synapse_save', 'synapse_train',
      'devtrain_context', 'devtrain_script_get', 'devtrain_script_set', 'devtrain_start', 'devtrain_status',
      'lab_server_start', 'lab_infer', 'tests_start', 'tests_results', 'analysis_runs_compare', 'hosting_infer',
    ];
    for (const n of want) expect(registry.get(n), n).toBeDefined();
  });
});
