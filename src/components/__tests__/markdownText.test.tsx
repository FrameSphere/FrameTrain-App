// Der Renderer lag bis zur Auslagerung ungetestet im FloatingAICoach. Beim
// Umzug in ein gemeinsames Modul (Coach + Startseiten-Briefing nutzen ihn
// jetzt beide) sind die Faelle abgesichert, die die Kommentare im Code als
// frueher aufgetretene Fehler nennen.

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MarkdownText } from '../ui/MarkdownText';

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
});
