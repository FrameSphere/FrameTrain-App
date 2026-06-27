#!/usr/bin/env node

import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const appRoot = resolve(scriptDir, '..')
const repoRoot = appRoot
const outFormat = process.env.CHANGELOG_OUTPUT || 'json'
const baseRef = process.env.CHANGELOG_BASE_REF || process.env.LAST_CHANGELOG_REF || ''
const headRef = process.env.CHANGELOG_HEAD_REF || 'HEAD'

function runGit(args) {
  return execFileSync('git', args, {
    cwd: repoRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim()
}

function safeRunGit(args) {
  try {
    return runGit(args)
  } catch {
    return ''
  }
}

function readVersion() {
  const packageJsonPath = resolve(appRoot, 'package.json')
  const tauriPath = resolve(appRoot, 'src-tauri', 'tauri.conf.json')
  const cargoPath = resolve(appRoot, 'src-tauri', 'Cargo.toml')

  const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'))
  const tauri = JSON.parse(readFileSync(tauriPath, 'utf8'))
  const cargo = readFileSync(cargoPath, 'utf8')
  const cargoVersion = (cargo.match(/^version\s*=\s*"([^"]+)"/m) || [])[1] || packageJson.version

  return {
    package: packageJson.version || null,
    tauri: tauri.version || null,
    cargo: cargoVersion || null,
  }
}

function shortStatList(statOutput) {
  return statOutput
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .filter(line => /(\.tsx?|\.rs|\.json|\.md|\.css|\.toml)$/.test(line))
    .slice(0, 40)
}

function classifyCommit(subject) {
  const s = subject.toLowerCase()
  if (/fix|bug|hotfix|patch/.test(s)) return 'fix'
  if (/refactor|improv|clean|perf|optimi|stabil/.test(s)) return 'improvement'
  if (/security|auth|sanitize|escape/.test(s)) return 'security'
  if (/breaking|migrate|remove/.test(s)) return 'breaking'
  return 'feature'
}

function buildRange() {
  if (baseRef) return `${baseRef}..${headRef}`
  const fallbackBase = safeRunGit(['describe', '--tags', '--abbrev=0', '--match', 'v*', headRef]) || safeRunGit(['rev-list', '--max-count=1', `${headRef}`]) || ''
  return fallbackBase ? `${fallbackBase}..${headRef}` : `${headRef}`
}

const version = readVersion()
const range = buildRange()
const lastTag = safeRunGit(['describe', '--tags', '--abbrev=0', '--match', 'v*', headRef]) || null
const effectiveRange = range.includes('..') ? range : `${headRef}`
const commitCount = safeRunGit(['rev-list', '--count', effectiveRange]) || '0'
const commits = safeRunGit([
  'log',
  '--no-merges',
  '--format=%H%x09%s',
  '--reverse',
  effectiveRange,
])
  .split('\n')
  .filter(Boolean)
  .map(line => {
    const [hash, subject] = line.split('\t')
    return { hash, subject, type: classifyCommit(subject || '') }
  })

const latestCommit = commits.length ? commits[commits.length - 1].hash : (safeRunGit(['rev-parse', headRef]) || null)
const files = shortStatList(
  safeRunGit([
    'diff',
    '--name-only',
    effectiveRange,
  ])
)

const summary = {
  source: 'desktop-app',
  version,
  lastTag,
  range: effectiveRange,
  commitCount: Number.parseInt(commitCount || '0', 10) || commits.length,
  latestCommit,
  commits,
  files,
}

if (outFormat === 'markdown') {
  const lines = []
  lines.push(`### Was wurde in desktop-app geändert`)
  lines.push(`- **Versionen**: package ${version.package || 'n/a'}, Tauri ${version.tauri || 'n/a'}, Cargo ${version.cargo || 'n/a'}`)
  lines.push(`- **Referenz**: ${summary.range}`)
  lines.push(`- **Commits**: ${summary.commitCount}`)
  if (summary.lastTag) lines.push(`- **Letzter Tag**: ${summary.lastTag}`)
  if (commits.length) {
    lines.push('')
    lines.push('### Relevante Änderungen')
    for (const commit of commits.slice(0, 12)) {
      lines.push(`- **${commit.type}**: ${commit.subject}`)
    }
  }
  if (files.length) {
    lines.push('')
    lines.push('### Betroffene Dateien')
    for (const file of files.slice(0, 20)) {
      lines.push(`- ${file}`)
    }
  }
  process.stdout.write(lines.join('\n') + '\n')
} else {
  process.stdout.write(JSON.stringify(summary, null, 2) + '\n')
}
