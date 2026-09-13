// Python-Syntax-Highlighting fuer die Editoren in Dev Train und Dev Test.
//
// Beide Panels hatten eine eigene Kopie. Dev Train war schon repariert, Dev Test
// nicht: dort lief der Keyword-Durchlauf ueber das eigene Markup
// (<span class="tok-num"> enthaelt das Keyword `class`) und der Editor zeigte
// `indent=class="tok-num">2` statt `indent=2`.

export function escapeHtml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function highlightPythonToHtml(code: string) {
  // Lightweight highlighter: strings + comments first, then keywords/numbers in remaining code.
  const KEYWORDS = new Set([
    'False','None','True','and','as','assert','async','await','break','class','continue','def','del','elif','else','except','finally',
    'for','from','global','if','import','in','is','lambda','nonlocal','not','or','pass','raise','return','try','while','with','yield',
  ]);
  const BUILTINS = new Set([
    'print','len','range','enumerate','zip','map','filter','list','dict','set','tuple','str','int','float','bool','open','sum','min','max',
    'sorted','any','all','isinstance','type','super','dir','vars','getattr','setattr','hasattr','Exception','ValueError','TypeError',
  ]);

  type Seg = { t: 'code' | 'str' | 'cmt'; v: string };
  const segs: Seg[] = [];
  let i = 0;
  let cur = '';
  let state: 'code' | 'str' | 'cmt' = 'code';
  let quote: "'" | '"' | "'''" | '"""' | null = null;

  const flush = () => {
    if (!cur) return;
    segs.push({ t: state, v: cur });
    cur = '';
  };

  while (i < code.length) {
    const ch = code[i];
    const next3 = code.slice(i, i + 3);

    if (state === 'code') {
      if (next3 === "'''" || next3 === '"""') {
        flush();
        state = 'str';
        quote = next3 as "'''" | '"""';
        cur += next3;
        i += 3;
        continue;
      }
      if (ch === "'" || ch === '"') {
        flush();
        state = 'str';
        quote = ch as "'" | '"';
        cur += ch;
        i += 1;
        continue;
      }
      if (ch === '#') {
        flush();
        state = 'cmt';
        quote = null;
        cur += ch;
        i += 1;
        continue;
      }
      cur += ch;
      i += 1;
      continue;
    }

    if (state === 'cmt') {
      cur += ch;
      i += 1;
      if (ch === '\n') {
        flush();
        state = 'code';
      }
      continue;
    }

    // string state
    cur += ch;
    i += 1;
    if (quote === "'" || quote === '"') {
      if (ch === '\\' && i < code.length) {
        cur += code[i];
        i += 1;
        continue;
      }
      if (ch === quote) {
        flush();
        state = 'code';
        quote = null;
      }
      continue;
    }
    if (quote === "'''" || quote === '"""') {
      if (code.slice(i - 1, i - 1 + 3) === quote) {
        cur += quote.slice(1);
        i += 2;
        flush();
        state = 'code';
        quote = null;
      }
    }
  }
  flush();

  /**
   * Wendet ein Replace nur auf Text AUSSERHALB bereits eingefügter HTML-Tags an.
   *
   * Ohne diesen Schutz lief der Identifier-Durchlauf über das eigene Markup:
   * In `<span class="tok-num">` steckt das Python-Keyword `class`, das prompt
   * ein zweites Mal umschlossen wurde. Das Ergebnis war kaputtes HTML, das im
   * Editor als literaler Text `class="tok-num">2` auftauchte und den Code
   * praktisch unlesbar machte.
   */
  const replaceOutsideTags = (
    input: string,
    pattern: RegExp,
    replacer: (...args: any[]) => string,
  ): string =>
    input
      .split(/(<[^>]*>)/g)
      .map(part => (part.startsWith('<') && part.endsWith('>') ? part : part.replace(pattern, replacer as any)))
      .join('');

  const highlightCode = (s: string) => {
    let out = escapeHtml(s);
    out = replaceOutsideTags(out, /\b\d+(\.\d+)?\b/g, (m: string) => `<span class="tok-num">${m}</span>`);
    out = replaceOutsideTags(out, /\b(def)\s+([A-Za-z_][A-Za-z0-9_]*)/g,
      (_m: string, kw: string, name: string) => `<span class="tok-kw">${kw}</span> <span class="tok-fn">${name}</span>`);
    out = replaceOutsideTags(out, /\b(class)\s+([A-Za-z_][A-Za-z0-9_]*)/g,
      (_m: string, kw: string, name: string) => `<span class="tok-kw">${kw}</span> <span class="tok-cl">${name}</span>`);
    out = replaceOutsideTags(out, /(^|\n)(\s*)(@[\w.]+)/g,
      (_m: string, pre: string, ws: string, dec: string) => `${pre}${ws}<span class="tok-de">${dec}</span>`);
    out = replaceOutsideTags(out, /\b([A-Za-z_][A-Za-z0-9_]*)\b/g, (_m: string, w: string) => {
      if (KEYWORDS.has(w)) return `<span class="tok-kw">${w}</span>`;
      if (BUILTINS.has(w)) return `<span class="tok-bi">${w}</span>`;
      return w;
    });
    return out;
  };

  const html = segs.map(seg => {
    if (seg.t === 'str') return `<span class="tok-str">${escapeHtml(seg.v)}</span>`;
    if (seg.t === 'cmt') return `<span class="tok-cmt">${escapeHtml(seg.v)}</span>`;
    return highlightCode(seg.v);
  }).join('');

  return html.endsWith('\n') ? html + ' ' : html;
}
