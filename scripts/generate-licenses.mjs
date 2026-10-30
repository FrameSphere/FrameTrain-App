#!/usr/bin/env node
// Erzeugt public/third-party-notices.txt: Lizenztexte aller mit der App
// ausgelieferten npm- und Rust-Pakete (Einstellungen > About zeigt die Datei).
//
// Gleiche Lizenztexte stehen nur einmal drin, darueber die Pakete, fuer die sie
// gelten. Python-Pakete fehlen bewusst: Die App liefert nur eigene Skripte aus,
// PyTorch & Co. laedt erst der Nutzer per pip nach.
//
// Bricht ab, wenn ein Paket nur unter GPL/AGPL/LGPL zu haben ist — das vertraegt
// sich nicht mit der proprietaeren Lizenz der App.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = join(ROOT, 'public', 'third-party-notices.txt');
const LICENSE_FILE = /^(licen[cs]e|copying|notice|unlicense)/i;

function licenseTexts(dir) {
  if (!dir || !existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true })
    .filter(e => e.isFile() && LICENSE_FILE.test(e.name))
    .sort((a, b) => a.name.localeCompare(b.name))
    .map(e => readFileSync(join(dir, e.name), 'utf8').trim())
    .filter(Boolean);
}

function npmPackages() {
  const lock = JSON.parse(readFileSync(join(ROOT, 'package-lock.json'), 'utf8'));
  return Object.entries(lock.packages)
    .filter(([path, p]) => path && !p.dev && !p.devOptional)
    .map(([path, p]) => {
      const dir = join(ROOT, path);
      const manifest = join(dir, 'package.json');
      const license = p.license ?? (existsSync(manifest) ? JSON.parse(readFileSync(manifest, 'utf8')).license : undefined);
      return { name: path.replace(/^.*node_modules\//, ''), version: p.version, license, dir, authors: [] };
    });
}

function rustPackages() {
  const out = execFileSync(
    'cargo',
    ['metadata', '--format-version', '1', '--manifest-path', join(ROOT, 'src-tauri', 'Cargo.toml')],
    { encoding: 'utf8', maxBuffer: 1024 * 1024 * 256, stdio: ['ignore', 'pipe', 'ignore'] },
  );
  const meta = JSON.parse(out);
  const nodes = new Map(meta.resolve.nodes.map(n => [n.id, n]));
  const packages = new Map(meta.packages.map(p => [p.id, p]));
  const seen = new Set();
  const stack = [meta.resolve.root];
  while (stack.length) {
    const id = stack.pop();
    if (seen.has(id)) continue;
    seen.add(id);
    for (const dep of nodes.get(id).deps) {
      // Nur was in die App kommt: normale und Build-Abhaengigkeiten, keine dev-deps
      if (dep.dep_kinds.some(k => k.kind === null || k.kind === 'build')) stack.push(dep.pkg);
    }
  }
  seen.delete(meta.resolve.root);
  return [...seen].map(id => {
    const p = packages.get(id);
    return { name: p.name, version: p.version, license: p.license, dir: dirname(p.manifest_path), authors: p.authors ?? [] };
  });
}

/** true, wenn jede Wahlmoeglichkeit des SPDX-Ausdrucks unter die GPL-Familie faellt. */
function isCopyleftOnly(license) {
  if (!license) return false;
  return license.split(/\s+OR\s+|\//).every(option => /\b(A|L)?GPL\b/.test(option));
}

const all = [...npmPackages(), ...rustPackages()]
  .sort((a, b) => a.name.localeCompare(b.name) || String(a.version).localeCompare(String(b.version)));

const forbidden = all.filter(p => isCopyleftOnly(p.license));
if (forbidden.length) {
  console.error('GPL-Pakete gefunden, die App darf sie so nicht ausliefern:');
  for (const p of forbidden) console.error(`  ${p.name} ${p.version}: ${p.license}`);
  process.exit(1);
}

const unknown = all.filter(p => !p.license);
if (unknown.length) {
  console.error(`Pakete ohne Lizenzangabe: ${unknown.map(p => p.name).join(', ')}`);
  process.exit(1);
}

const groups = new Map();
const withoutText = [];
for (const pkg of all) {
  const texts = licenseTexts(pkg.dir);
  if (!texts.length) {
    withoutText.push(pkg);
    continue;
  }
  const text = texts.join('\n\n');
  const key = createHash('sha256').update(text).digest('hex');
  if (!groups.has(key)) groups.set(key, { text, packages: [] });
  groups.get(key).packages.push(pkg);
}

const label = p => `${p.name} ${p.version} (${p.license})`;
const rule = '='.repeat(72);
const parts = [
  'FrameTrain Desktop - Third-Party Notices',
  'FrameTrain enthaelt die folgenden Open-Source-Komponenten.',
  'FrameTrain includes the following open-source components.',
  'Pakete mit Wahl zwischen mehreren Lizenzen werden unter einer der genannten Lizenzen genutzt, nicht unter (L)GPL.',
  'Packages offering a choice of licenses are used under one of the listed licenses, not under the (L)GPL.',
  'MPL-2.0-Komponenten sind unveraendert; ihr Quellcode steht auf crates.io und in den jeweiligen Projekten bereit.',
  'MPL-2.0 components are unmodified; their source code is available on crates.io and from the respective projects.',
];
for (const { text, packages } of groups.values()) {
  parts.push(rule, packages.map(label).join('\n'), '', text);
}
if (withoutText.length) {
  parts.push(
    rule,
    'Ohne eigene Lizenzdatei im Paket / No license file shipped in the package.',
    'Standardtexte der genannten Lizenzen / Standard license texts: https://spdx.org/licenses/',
    '',
  );
  for (const p of withoutText) {
    parts.push(`${label(p)}${p.authors.length ? ` - ${p.authors.join(', ')}` : ''}`);
  }
}

writeFileSync(OUTPUT, parts.join('\n\n') + '\n');
console.log(`${all.length} Pakete, ${groups.size} Lizenztexte, ${withoutText.length} ohne Lizenzdatei -> public/third-party-notices.txt`);
