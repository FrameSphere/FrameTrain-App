// Regression aus dem App-Durchgang mit 1.2.73: Im Chat unter der KI-Analyse
// stand "Empfohlener Parameter: `p = 0.30`" mit sichtbaren Backticks — der
// Bericht-Renderer kannte nur Fett und Kursiv.
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { ReportText } from '../AnalysisPanel';

describe('ReportText', () => {
  it('rendert Inline-Code ohne Backticks, auch in Fett und mit Sternchen', () => {
    const { container } = render(<ReportText text={'- Empfohlener Parameter: `p = 0.30` und **`warmupSteps`**, siehe `a*b*c`.'} />);
    const codes = [...container.querySelectorAll('code')].map(c => c.textContent);
    expect(codes).toEqual(['p = 0.30', 'warmupSteps', 'a*b*c']);
    expect(container.querySelector('strong code')?.textContent).toBe('warmupSteps');
    expect(container.textContent).not.toContain('`');
    expect(container.querySelector('em')).toBeNull();
  });

  it('escaped Markup im Code', () => {
    const { container } = render(<ReportText text={'Nutze `<script>`'} />);
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('code')?.textContent).toBe('<script>');
  });
});
