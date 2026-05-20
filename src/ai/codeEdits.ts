export type CodeEdit = {
  id: string;
  find: string;
  replace: string;
  applied?: boolean;
  failed?: boolean;
  strategy?: string;
  confidence?: number; // 0..1
};

function stripPythonFences(s: string) {
  return s.replace(/^```python\n?/, '').replace(/\n?```$/, '');
}

export function parseEdits(text: string): CodeEdit[] {
  const edits: CodeEdit[] = [];

  // Erst vollständige Blöcke (mit ##EDIT_END##)
  const regexComplete = /##EDIT_START##\s*FIND:\s*([\s\S]*?)\s*REPLACE:\s*([\s\S]*?)\s*##EDIT_END##/g;
  let match: RegExpExecArray | null;
  while ((match = regexComplete.exec(text)) !== null) {
    let find = stripPythonFences(match[1].trim());
    let replace = stripPythonFences(match[2].trim());
    edits.push({ id: `edit_${Date.now()}_${edits.length}`, find, replace });
  }

  // Fallback: unvollständige Blöcke (KI hat ##EDIT_END## vergessen), greedy bis Ende
  if (edits.length === 0) {
    const regexIncomplete = /##EDIT_START##\s*FIND:\s*([\s\S]*?)\s*REPLACE:\s*([\s\S]*)/g;
    while ((match = regexIncomplete.exec(text)) !== null) {
      let find = stripPythonFences(match[1].trim());
      let replace = stripPythonFences(match[2].trim());
      if (find && replace) {
        edits.push({ id: `edit_${Date.now()}_${edits.length}`, find, replace });
      }
    }
  }

  return edits;
}

function normalizeTabs(s: string) {
  return s.replace(/\t/g, '  ');
}

function collapseSpaces(s: string) {
  return s.replace(/[ \t]+/g, ' ');
}

function splitLines(s: string) {
  return s.split('\n');
}

function trimEndLines(lines: string[]) {
  return lines.map(l => l.trimEnd());
}

function findLineWindow(
  scriptLines: string[],
  findLines: string[],
  normalize: (l: string) => string,
): number | null {
  const fl = findLines.map(normalize);
  for (let i = 0; i <= scriptLines.length - findLines.length; i++) {
    let ok = true;
    for (let j = 0; j < findLines.length; j++) {
      if (normalize(scriptLines[i + j]) !== fl[j]) { ok = false; break; }
    }
    if (ok) return i;
  }
  return null;
}

function replaceLineWindow(scriptLines: string[], start: number, findLen: number, replaceLines: string[]) {
  const before = scriptLines.slice(0, start).join('\n');
  const after = scriptLines.slice(start + findLen).join('\n');
  return [...(before ? [before] : []), ...replaceLines, ...(after ? [after] : [])].join('\n');
}

function nonEmptyLines(lines: string[]) {
  return lines.map(l => l.trim()).filter(Boolean);
}

function anchorReplace(script: string, find: string, replace: string): { result: string; success: boolean } {
  const findLines = splitLines(find);
  const anchors = nonEmptyLines(findLines);
  if (anchors.length < 2) return { result: script, success: false };

  const first = anchors[0];
  const last = anchors[anchors.length - 1];
  const startIdx = script.indexOf(first);
  if (startIdx < 0) return { result: script, success: false };
  const endIdx = script.indexOf(last, startIdx + first.length);
  if (endIdx < 0) return { result: script, success: false };

  // Expand end to end-of-line of last anchor
  const endLineEnd = script.indexOf('\n', endIdx);
  const endExclusive = endLineEnd >= 0 ? endLineEnd : script.length;

  // Expand start to start-of-line of first anchor
  const startLineStart = script.lastIndexOf('\n', startIdx);
  const startInclusive = startLineStart >= 0 ? startLineStart + 1 : 0;

  const result = script.slice(0, startInclusive) + replace + script.slice(endExclusive);
  return { result, success: true };
}

export function applyEdit(script: string, edit: CodeEdit): { result: string; success: boolean; strategy: string; confidence: number } {
  // Tier 1: exact substring
  if (script.includes(edit.find)) {
    return { result: script.replace(edit.find, edit.replace), success: true, strategy: 'exact', confidence: 1.0 };
  }

  const scriptLines = splitLines(script);
  const findLines = splitLines(edit.find);
  const replaceLines = splitLines(edit.replace);

  // Tier 2: tabs->spaces, exact line window
  {
    const idx = findLineWindow(scriptLines, findLines, (l) => normalizeTabs(l));
    if (idx != null) {
      return { result: replaceLineWindow(scriptLines, idx, findLines.length, replaceLines), success: true, strategy: 'tabs-normalized', confidence: 0.9 };
    }
  }

  // Tier 3: trimEnd + tabs->spaces
  {
    const idx = findLineWindow(trimEndLines(scriptLines), trimEndLines(findLines), (l) => normalizeTabs(l));
    if (idx != null) {
      return { result: replaceLineWindow(scriptLines, idx, findLines.length, replaceLines), success: true, strategy: 'trimend', confidence: 0.8 };
    }
  }

  // Tier 4: collapse multiple spaces (more permissive)
  {
    const idx = findLineWindow(scriptLines, findLines, (l) => collapseSpaces(normalizeTabs(l.trimEnd())));
    if (idx != null) {
      return { result: replaceLineWindow(scriptLines, idx, findLines.length, replaceLines), success: true, strategy: 'space-collapsed', confidence: 0.7 };
    }
  }

  // Tier 5: anchor-based (first/last non-empty lines)
  {
    const anchored = anchorReplace(script, edit.find, edit.replace);
    if (anchored.success) return { result: anchored.result, success: true, strategy: 'anchors', confidence: 0.6 };
  }

  return { result: script, success: false, strategy: 'no-match', confidence: 0 };
}

export function applyAllEdits(script: string, edits: CodeEdit[]) {
  let current = script;
  const results = edits.map(e => {
    const r = applyEdit(current, e);
    if (r.success) current = r.result;
    return r;
  });
  return { result: current, results };
}

export function removeEditBlocks(text: string) {
  // Vollständige Blöcke entfernen
  let result = text.replace(/##EDIT_START##[\s\S]*?##EDIT_END##/g, '');
  // Unvollständige Blöcke (KI hat ##EDIT_END## vergessen) – alles ab ##EDIT_START## entfernen
  result = result.replace(/##EDIT_START##[\s\S]*$/g, '');
  return result;
}

export function extractFullPythonCode(text: string) {
  return text.match(/```python\n([\s\S]*?)```/)?.[1] ?? null;
}

