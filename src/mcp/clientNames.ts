// Lesbare Namen fuer KI-Programme. Clients melden beim Verbinden technische
// Kennungen (clientInfo.name), z. B. "claude-ai" oder "local-agent-mode-frametrain".
// Der Freigabe-Dialog und die Freigaben-Liste zeigen stattdessen den Produktnamen.
// "Immer erlauben" gilt damit fuer alle Varianten desselben Programms.

const KNOWN: [RegExp, string][] = [
  [/^claude[-_ ]?code\b/i, 'Claude Code'],
  [/^local-agent-mode\b/i, 'Claude Code'],
  [/^claude[-_ ]?(ai|desktop)\b/i, 'Claude'],
  [/^(claude|anthropic)\b/i, 'Claude'],
  [/^(openai[-_ ]?mcp|chatgpt|openai)\b/i, 'ChatGPT'],
  [/^cursor\b/i, 'Cursor'],
  [/^(visual studio code|vscode)\b/i, 'VS Code'],
  [/^windsurf\b/i, 'Windsurf'],
  [/^zed\b/i, 'Zed'],
  [/^mcp[-_ ]?inspector\b/i, 'MCP Inspector'],
];

export const UNKNOWN_CLIENT = 'Unknown client';

export function friendlyClientName(raw: unknown): string {
  if (typeof raw !== 'string') return UNKNOWN_CLIENT;
  const name = raw.trim();
  if (!name) return UNKNOWN_CLIENT;
  for (const [re, label] of KNOWN) if (re.test(name)) return label;
  return name;
}
