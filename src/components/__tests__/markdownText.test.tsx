// Der Renderer lag bis zur Auslagerung ungetestet im FloatingAICoach. Beim
// Umzug in ein gemeinsames Modul (Coach + Startseiten-Briefing nutzen ihn
// jetzt beide) sind die Faelle abgesichert, die die Kommentare im Code als
// frueher aufgetretene Fehler nennen.

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MarkdownText, unwrapBoldHeading } from '../ui/MarkdownText';

describe('MarkdownText', () => {
  it('rendert Fettschrift, Kursiv und Inline-Code als echte Elemente', () => {
    const { container } = render(<MarkdownText text="Der **Loss** ist *gut*, siehe `train_loss`." />);
    expect(container.querySelector('strong')?.textContent).toBe('Loss');
    expect(container.querySelector('em')?.textContent).toBe('gut');
    expect(container.querySelector('code')?.textContent).toBe('train_loss');
  });

  it('macht aus "- "-Zeilen eine Liste', () => {
    const { container } = render(<MarkdownText text={'- erster Punkt\n- zweiter Punkt'} />);
    expect(container.querySelectorAll('ul li')).toHaveLength(2);
    expect(screen.getByText('erster Punkt')).toBeTruthy();
  });

  it('haelt eine nummerierte Liste ueber Leerzeilen hinweg zusammen', () => {
    // Regression laut Code-Kommentar: pro Block entstand ein eigenes <ol>,
    // die Nummerierung fing jedes Mal wieder bei 1 an.
    const { container } = render(<MarkdownText text={'1. eins\n\n2. zwei\n\n3. drei'} />);
    expect(container.querySelectorAll('ol')).toHaveLength(1);
    expect(container.querySelectorAll('ol li')).toHaveLength(3);
  });

  it('rendert einen Code-Block mit Sprachangabe', () => {
    const { container } = render(<MarkdownText text={'```python\nprint(1)\n```'} />);
    expect(container.querySelector('pre code')?.textContent).toBe('print(1)');
    expect(screen.getByText('python')).toBeTruthy();
  });

  it('erzeugt fuer einen leeren Code-Block keine leere Box', () => {
    // Regression laut Code-Kommentar: ein nicht geschlossener Fence hinterliess
    // im Chat eine leere Code-Box.
    const { container } = render(<MarkdownText text={'Text\n```\n\n```'} />);
    expect(container.querySelector('pre')).toBeNull();
  });

  it('behandelt schlichten Text als Absaetze', () => {
    const { container } = render(<MarkdownText text={'Erster Absatz.\nZweiter Absatz.'} />);
    expect(container.querySelectorAll('p')).toHaveLength(2);
  });

  it('laesst Text ohne Markdown unveraendert', () => {
    render(<MarkdownText text="Ganz normaler Satz." />);
    expect(screen.getByText('Ganz normaler Satz.')).toBeTruthy();
  });

  // Eine einzelne Raute fehlte im Renderer: "# Analyse der Konfiguration"
  // stand woertlich samt Raute im Dialog des Metrik-Assistenten.
  it('rendert auch eine einzelne Raute als Ueberschrift', () => {
    render(<MarkdownText text={'# Analyse der Konfiguration\nText darunter.'} />);
    expect(screen.getByText('Analyse der Konfiguration')).toBeTruthy();
    expect(screen.queryByText(/^#/)).toBeNull();
  });

  it('haelt eine Raute ohne Leerzeichen fuer normalen Text', () => {
    render(<MarkdownText text="#1 im Ranking" />);
    expect(screen.getByText('#1 im Ranking')).toBeTruthy();
  });
});

// Groq/compound schreibt Ueberschriften als "**## Titel**" — die Zeile beginnt
// dann mit ** statt mit #, die Ueberschriften-Erkennung griff nicht und die
// Rauten standen sichtbar im Trainingsbericht.
describe('unwrapBoldHeading', () => {
  it('loest die Fettschrift um eine Ueberschrift', () => {
    expect(unwrapBoldHeading('**## Gesamtbewertung**')).toBe('## Gesamtbewertung');
    expect(unwrapBoldHeading('  **# Titel**  ')).toBe('# Titel');
  });

  it('laesst normale Fettschrift in Ruhe', () => {
    expect(unwrapBoldHeading('**Wichtig**')).toBe('**Wichtig**');
    expect(unwrapBoldHeading('## Titel')).toBe('## Titel');
    expect(unwrapBoldHeading('Text mit **fett** drin')).toBe('Text mit **fett** drin');
  });

  it('rendert eine fett gesetzte Ueberschrift als Ueberschrift', () => {
    render(<MarkdownText text={'**## Was lief gut**\nInhalt.'} />);
    expect(screen.getByText('Was lief gut')).toBeTruthy();
    expect(screen.queryByText(/##/)).toBeNull();
  });
});
