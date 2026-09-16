// Die Boxen ueber dem Vorschaubild im Labor.
//
// Der Punkt der Anzeige: "Tree 80%" sagt nicht, WO das Modell den Baum sieht.
// Damit die Boxen passen, muss das SVG in Bildkoordinaten rechnen und
// deckungsgleich ueber dem object-contain-Bild liegen.

import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { DetectionOverlay } from '../LaboratoryPanel';

const BOX = { label: 'Tree', confidence: 0.802, x1: 321, y1: 232, x2: 512, y2: 390 };

describe('DetectionOverlay', () => {
  it('rechnet in Bildkoordinaten statt in Anzeigepixeln', () => {
    const { container } = render(<DetectionOverlay boxes={[BOX]} width={512} height={512} />);
    const svg = container.querySelector('svg')!;
    expect(svg.getAttribute('viewBox')).toBe('0 0 512 512');
    // object-contain im Bild, meet im SVG – sonst laufen die Boxen weg.
    expect(svg.getAttribute('preserveAspectRatio')).toBe('xMidYMid meet');

    const rect = container.querySelector('rect')!;
    expect(rect.getAttribute('x')).toBe('321');
    expect(rect.getAttribute('width')).toBe('191');
    expect(rect.getAttribute('height')).toBe('158');
  });

  it('beschriftet mit Klasse und gerundeter Konfidenz', () => {
    const { container } = render(<DetectionOverlay boxes={[BOX]} width={512} height={512} />);
    expect(container.querySelector('text')?.textContent).toBe('Tree 80%');
  });

  it('haelt die Beschriftung im Bild, wenn die Box oben klebt', () => {
    const top = { ...BOX, y1: 0, y2: 120 };
    const { container } = render(<DetectionOverlay boxes={[top]} width={512} height={512} />);
    const y = Number(container.querySelector('text')!.getAttribute('y'));
    expect(y).toBeGreaterThan(0);
  });

  it('zeichnet nichts ohne Boxen oder ohne Bildmasse', () => {
    expect(render(<DetectionOverlay boxes={[]} width={512} height={512} />)
      .container.querySelector('svg')).toBeNull();
    expect(render(<DetectionOverlay boxes={[BOX]} width={0} height={0} />)
      .container.querySelector('svg')).toBeNull();
  });

  it('skaliert Linien und Schrift mit der Bildgroesse', () => {
    // Bei einem 4000px-Foto waeren 12px Schrift unsichtbar.
    const big = render(<DetectionOverlay boxes={[BOX]} width={4000} height={3000} />);
    const small = render(<DetectionOverlay boxes={[BOX]} width={512} height={512} />);
    const fontOf = (c: HTMLElement) => Number(c.querySelector('text')!.getAttribute('font-size'));
    expect(fontOf(big.container)).toBeGreaterThan(fontOf(small.container));
  });
});
