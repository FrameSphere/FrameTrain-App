// Abgeschnittene Code-Antworten fortsetzen.
// Regression aus Karols Test vom 27.10.2026: das Chronos-2-Skript endete am
// Token-Limit mitten im Code und landete deshalb nicht in der Datei.

import { describe, it, expect } from 'vitest';
import {
  continueTruncated, hasOpenCodeBlock, joinContinuation, MAX_CONTINUATIONS, shouldContinue,
} from '../continueTruncated';
import { extractFullPythonCode } from '../codeEdits';

const head = 'Plan: Fenster pro id.\n\n```python\nimport os\nimport torch\n\ndef evaluate():\n    model.eval()\n    los';

describe('hasOpenCodeBlock / shouldContinue', () => {
  it('erkennt einen offenen Codeblock', () => {
    expect(hasOpenCodeBlock(head)).toBe(true);
    expect(hasOpenCodeBlock(head + 'ses = []\n```')).toBe(false);
    expect(hasOpenCodeBlock('nur Text')).toBe(false);
  });

  it('setzt fort bei offenem Code oder angekuendigtem Rewrite ohne Code', () => {
    expect(shouldContinue(head, 'rewrite')).toBe(true);
    expect(shouldContinue('Ich schreibe das Skript so:', 'rewrite')).toBe(true);
    expect(shouldContinue('Ich schreibe das Skript so:', 'chat')).toBe(false);
  });
});

describe('joinContinuation', () => {
  it('fuegt nahtlos an', () => {
    expect(joinContinuation(head, 'ses = []\n    return 0\n```')).toBe(head + 'ses = []\n    return 0\n```');
  });

  it('entfernt Steuerblock und neuen python-Fence', () => {
    const next = '```ft_action\n{"mode":"rewrite","rationale":"weiter"}\n```\n```python\nses = []\n```';
    expect(joinContinuation(head, next)).toBe(head + 'ses = []\n```');
  });

  it('wiederholte halbe Zeile wird nicht doppelt geschrieben', () => {
    const joined = joinContinuation(head, '    losses = []\n    return 0\n```');
    expect(joined).toContain('    model.eval()\n    losses = []\n    return 0');
    expect(joined).not.toContain('los    losses');
  });

  it('Ueberlappung am Ende wird entfernt', () => {
    const joined = joinContinuation(head, 'def evaluate():\n    model.eval()\n    losses = []\n```');
    expect(joined.match(/def evaluate/g)).toHaveLength(1);
  });
});

describe('continueTruncated', () => {
  it('fragt nach, bis der Codeblock geschlossen ist, und liefert vollstaendigen Code', async () => {
    const parts = [
      { text: 'ses = []\n    for x in range(3):\n', cut: true },
      { text: '        losses.append(x)\n    return losses\n```\nFertig.', cut: false },
    ];
    let calls = 0;
    const res = await continueTruncated(head, 'rewrite', async () => parts[calls++]);
    expect(calls).toBe(2);
    expect(res.cut).toBe(false);
    const code = extractFullPythonCode(res.text)!;
    expect(code).toContain('import torch');
    expect(code).toContain('    return losses');
  });

  it('ergaenzt einen vergessenen schliessenden Fence', async () => {
    const res = await continueTruncated(head, 'rewrite', async () => ({ text: 'ses = []\n', cut: false }));
    expect(res.cut).toBe(false);
    expect(extractFullPythonCode(res.text)).toContain('losses = []');
  });

  it('gibt nach MAX_CONTINUATIONS auf und meldet weiter abgeschnitten', async () => {
    let calls = 0;
    const res = await continueTruncated(head, 'rewrite', async () => { calls++; return { text: `x${calls} = 1\n`, cut: true }; });
    expect(calls).toBe(MAX_CONTINUATIONS);
    expect(res.cut).toBe(true);
  });

  it('fragt bei normalen Chat-Antworten nicht nach', async () => {
    let calls = 0;
    const res = await continueTruncated('Kurze Erklaerung ohne Code', 'chat', async () => { calls++; return { text: '', cut: false }; });
    expect(calls).toBe(0);
    expect(res.cut).toBe(false);
  });
});
