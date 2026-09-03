// Markdown-Darstellung fuer KI-Antworten.
//
// Unterstuetzt die Teilmenge, die die Modelle in dieser App tatsaechlich
// liefern: Fettschrift, Kursiv, Inline-Code, Code-Bloecke, Aufzaehlungen,
// nummerierte Listen und kleine Ueberschriften. Bewusst keine Bibliothek —
// das haelt die Ausgabe unter unserer Kontrolle und den Bundle klein.
//
// Lag vorher lokal im FloatingAICoach; seit die Startseite dasselbe Format
// rendert, liegt der Renderer hier, damit es nur EINE Darstellung gibt.

import React from 'react';

export function renderInline(str: string, key?: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match;
  let i = 0;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`t${i++}`}>{str.slice(lastIndex, match.index)}</span>);
    }
    if (match[0].startsWith('**')) {
      parts.push(<strong key={`b${i++}`} className="font-semibold text-white">{match[2]}</strong>);
    } else if (match[0].startsWith('*')) {
      parts.push(<em key={`em${i++}`} className="italic">{match[3]}</em>);
    } else if (match[0].startsWith('`')) {
      parts.push(
        <code key={`c${i++}`} className="px-1.5 py-0.5 bg-white/10 rounded text-[11px] font-mono text-purple-300">
          {match[4]}
        </code>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < str.length) {
    parts.push(<span key={`t${i++}`}>{str.slice(lastIndex)}</span>);
  }
  return parts.length > 0 ? parts : str;
}

export function MarkdownText({ text, className = '' }: { text: string; className?: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // ── Code Block (```...```) ──────────────────────────────────────────────
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim(); // z.B. "python", "bash", ""
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // schließendes ``` überspringen
      // Ein alleinstehender Fence-Rest (z. B. wenn das Modell den Block nicht
      // geschlossen hat) erzeugte bisher eine leere Code-Box im Chat.
      if (codeLines.join('').trim() === '') continue;
      elements.push(
        // min-w-0 ist nötig, damit das <pre> in den Flex-Bubbles tatsächlich
        // scrollen kann statt rechts abgeschnitten zu werden.
        <div key={`cb-${i}`} className="my-2 rounded-xl overflow-hidden border border-white/10 min-w-0 max-w-full">
          {lang && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.06] border-b border-white/10">
              <span className="text-[10px] font-mono text-purple-300 font-medium">{lang}</span>
            </div>
          )}
          <pre className="px-3 py-2.5 bg-black/40 overflow-x-auto max-w-full">
            <code className="block text-[11px] font-mono text-emerald-300 leading-relaxed whitespace-pre">
              {codeLines.join('\n')}
            </code>
          </pre>
        </div>
      );
      continue;
    }

    if (!trimmed) {
      elements.push(<div key={i} className="h-1.5" />);
      i++;
      continue;
    }

    if (trimmed.match(/^[-*•]\s/)) {
      const items: string[] = [];
      while (i < lines.length) {
        if (lines[i].trim().match(/^[-*•]\s/)) {
          items.push(lines[i].trim().replace(/^[-*•]\s+/, ''));
          i++;
          continue;
        }
        if (lines[i].trim() === '') {
          let k = i;
          while (k < lines.length && lines[k].trim() === '') k++;
          if (k < lines.length && lines[k].trim().match(/^[-*•]\s/)) { i = k; continue; }
        }
        break;
      }
      elements.push(
        <ul key={`ul-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 mt-0.5 flex-shrink-0 text-xs">•</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (trimmed.match(/^\d+\.\s/)) {
      const items: string[] = [];
      while (i < lines.length) {
        if (lines[i].trim().match(/^\d+\.\s/)) {
          items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
          i++;
          continue;
        }
        // Leerzeilen zwischen zwei Punkten beenden die Liste nicht. Sonst
        // entstand pro Block ein eigenes <ol> und die Nummerierung fing
        // jedes Mal wieder bei 1 an (1., 2., 1., 1., 1.).
        if (lines[i].trim() === '') {
          let k = i;
          while (k < lines.length && lines[k].trim() === '') k++;
          if (k < lines.length && lines[k].trim().match(/^\d+\.\s/)) { i = k; continue; }
        }
        break;
      }
      elements.push(
        <ol key={`ol-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 flex-shrink-0 font-medium text-xs w-4">{j + 1}.</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    if (trimmed.startsWith('###')) {
      elements.push(
        <div key={i} className="font-semibold text-white mt-2 mb-1 text-sm">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else if (trimmed.startsWith('##')) {
      elements.push(
        <div key={i} className="font-bold text-white mt-2 mb-1">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else {
      elements.push(
        <p key={i} className="leading-relaxed">
          {renderInline(line)}
        </p>
      );
    }
    i++;
  }

  return <div className={`text-sm space-y-0.5 min-w-0 max-w-full ${className}`}>{elements}</div>;
}
