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

  it('faerbt jede Klasse eigen und nimmt dafuer die Klassenliste des Modells', () => {
    const classes = ['Tree', 'Sky'];
    const { container } = render(
      <DetectionOverlay
        boxes={[BOX, { ...BOX, label: 'Sky' }]}
        classes={classes}
        width={512}
        height={512}
      />,
    );
    const strokes = [...container.querySelectorAll('rect')].map(r => r.getAttribute('stroke'));
    expect(strokes[0]).not.toBe(strokes[1]);
    // Beschriftung in derselben Farbe wie ihre Box – sonst ist die Zuordnung Raten.
    expect(container.querySelector('text')?.getAttribute('fill')).toBe(strokes[0]);
  });

  it('zeichnet Soll gestrichelt und Erkennung durchgezogen, beide in Klassenfarbe', () => {
    const truth = [{ label: 'Tree', x1: 10, y1: 10, x2: 60, y2: 60 }];
    const { container } = render(
      <DetectionOverlay boxes={[BOX]} truthBoxes={truth} classes={['Tree']} width={512} height={512} />,
    );
    const rects = [...container.querySelectorAll('rect')];
    expect(rects[0].getAttribute('stroke-dasharray')).toBeTruthy();
    expect(rects[1].getAttribute('stroke-dasharray')).toBeNull();
    expect(rects[0].getAttribute('stroke')).toBe(rects[1].getAttribute('stroke'));
  });

  it('zeigt Soll-Boxen auch ohne Erkennung', () => {
    const truth = [{ label: 'Tree', x1: 10, y1: 10, x2: 60, y2: 60 }];
    const { container } = render(
      <DetectionOverlay boxes={[]} truthBoxes={truth} width={512} height={512} />,
    );
    expect(container.querySelectorAll('rect')).toHaveLength(1);
  });
});

// Live-Test 1.4.0: Segmentierung, Pose und gedrehte Boxen erschienen als
// gerade Rechtecke — Masken, Keypoints und Drehung lieferte der YOLO-Server
// zwar, gezeichnet wurden sie nie.
describe('DetectionOverlay: Masken, Keypoints, gedrehte Boxen', () => {
  it('zeichnet eine Maske als gefuelltes Polygon statt als Rechteck', () => {
    const seg = { ...BOX, polygon: [[330, 240], [500, 250], [480, 380]] as [number, number][] };
    const { container } = render(<DetectionOverlay boxes={[seg]} width={512} height={512} />);
    const poly = container.querySelector('polygon')!;
    expect(poly.getAttribute('points')).toBe('330,240 500,250 480,380');
    expect(poly.getAttribute('fill-opacity')).toBe('0.25');
    expect(container.querySelector('rect')).toBeNull();
  });

  it('zeichnet sichere Keypoints und bei 17 Punkten das Skelett', () => {
    const kpts = Array.from({ length: 17 }, (_, i) => [100 + i, 200, i === 16 ? 0.1 : 0.9] as [number, number, number]);
    const { container } = render(<DetectionOverlay boxes={[{ ...BOX, keypoints: kpts }]} width={512} height={512} />);
    // Punkt 16 ist unsicher (0.1) → 16 Kreise; die Skelett-Linien zu ihm fallen weg.
    expect(container.querySelectorAll('circle')).toHaveLength(16);
    expect(container.querySelectorAll('line').length).toBeGreaterThan(10);
  });

  it('Soll-Umrisse und Soll-Keypoints gestrichelt bzw. hohl', () => {
    const truth = [{ label: 'Tree', x1: 0, y1: 0, x2: 10, y2: 10, polygon: [[0, 0], [10, 0], [5, 10]] as [number, number][],
      keypoints: [[5, 5, 2]] as [number, number, number][] }];
    const { container } = render(<DetectionOverlay boxes={[]} truthBoxes={truth} width={512} height={512} />);
    expect(container.querySelector('polygon')?.getAttribute('stroke-dasharray')).toBeTruthy();
    expect(container.querySelector('circle')?.getAttribute('fill')).toBe('none');
  });
});
