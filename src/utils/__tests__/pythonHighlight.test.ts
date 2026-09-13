// Regression aus dem App-Durchgang vom 13.09.2026: Der Dev-Test-Editor zeigte
// `indent=class="tok-num">2` statt `indent=2`.
import { describe, it, expect } from 'vitest';
import { highlightPythonToHtml } from '../pythonHighlight';

const visibleText = (html: string) => html.replace(/<[^>]*>/g, '').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

describe('highlightPythonToHtml', () => {
  it('Zahlen neben Keywords bleiben lesbar', () => {
    const code = 'json.dump(report, f, ensure_ascii=False, indent=2)\nclass A: pass\n';
    expect(visibleText(highlightPythonToHtml(code)).trimEnd()).toBe(code.trimEnd());
    expect(highlightPythonToHtml(code)).not.toContain('class="tok-num">2)');
  });
});
