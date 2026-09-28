// Ergebnis-Ansichten im Labor fuer NER, Embeddings und freien Text.
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { LanguageProvider } from '../../contexts/LanguageContext';
import { EntityText, PairView, SimilarityView, compareEntities, sameText, similarityPercent } from '../LabTaskViews';

describe('EntityText', () => {
  it('markiert Entitaeten an ihren Zeichenpositionen mit Typ', () => {
    const { container } = render(
      <EntityText text="EU lehnt deutsche Idee ab" entities={[{ label: 'ORG', start: 0, end: 2 }, { label: 'MISC', start: 9, end: 17 }]} />,
    );
    const marks = [...container.querySelectorAll('mark')].map(m => m.textContent);
    expect(marks).toEqual(['EUORG', 'deutscheMISC']);
    expect(container.textContent).toBe('EUORG lehnt deutscheMISC Idee ab');
  });

  it('ignoriert Spannen ausserhalb des Textes und Ueberlappungen', () => {
    const { container } = render(
      <EntityText text="abc" entities={[{ label: 'A', start: 0, end: 2 }, { label: 'B', start: 1, end: 3 }, { label: 'C', start: 2, end: 9 }]} />,
    );
    expect(container.querySelectorAll('mark')).toHaveLength(1);
  });
});

describe('compareEntities', () => {
  it('zaehlt Treffer nur bei gleicher Spanne und gleichem Typ', () => {
    const exp = [{ text: 'EU', label: 'ORG', start: 0, end: 2 }, { text: 'Peter', label: 'PER', start: 5, end: 10 }];
    const got = [{ label: 'ORG', start: 0, end: 2 }, { label: 'LOC', start: 5, end: 10 }];
    expect(compareEntities(exp, got)).toEqual({ hits: 1, missing: 1, extra: 1 });
  });
});

describe('SimilarityView / PairView', () => {
  it('zeigt den Wert und das Soll aus dem Dataset', () => {
    const { container } = render(<LanguageProvider><SimilarityView similarity={0.8123} expected="4.2" /></LanguageProvider>);
    expect(container.textContent).toContain('0.812');
    expect(container.textContent).toContain('4.2');
  });

  // Live-Test 1.4.2: Soll 0.8 stand bei 80 % des Balkens, der aber von -1
  // bis 1 reicht — richtig sind 90 %.
  it('Soll-Markierung auf derselben Skala wie der Balken', () => {
    const { container } = render(<LanguageProvider><SimilarityView similarity={0.5} expected="0.8" /></LanguageProvider>);
    const marker = container.querySelector('[title]') as HTMLElement;
    expect(marker.style.left).toBe('90%');
    expect(similarityPercent(-1)).toBe(0);
    expect(similarityPercent(0)).toBe(50);
  });

  it('zerlegt "A ||| B" in zwei Zeilen', () => {
    const { container } = render(<PairView text="Ein Hund ||| Ein Tier" />);
    expect([...container.querySelectorAll('p')].map(p => p.textContent)).toEqual(['AEin Hund', 'BEin Tier']);
  });
});

describe('sameText', () => {
  it('Gross/klein, Leerraum und Schlusspunkt egal', () => {
    expect(sameText('  Werkzeug. ', 'werkzeug')).toBe(true);
    expect(sameText('Tier', 'Werkzeug')).toBe(false);
  });
});
