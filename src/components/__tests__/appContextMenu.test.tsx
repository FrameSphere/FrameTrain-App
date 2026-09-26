// Das app-weite Rechtsklick-Menue zeigt, was die Seite anbietet: je Gruppe
// ein Abschnitt, das Tastenkuerzel rechts daneben, Untermenues aufklappbar —
// und die Seite erfaehrt, worauf geklickt wurde.

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

vi.mock('../../ai/aiCoachEvents', () => ({ openAICoach: vi.fn() }));

import AppContextMenu from '../ui/AppContextMenu';
import { useContextMenuActions } from '../../ui/contextMenuRegistry';

function Seite({ onZiel, onKlasse }: { onZiel: (id: string | null) => void; onKlasse: (k: string) => void }) {
  useContextMenuActions(({ target }) => {
    onZiel(target?.closest('[data-karte]')?.getAttribute('data-karte') ?? null);
    return [
      { id: 'a', group: 'Box', label: 'Duplizieren', shortcut: '⌘D', onSelect: () => {} },
      { id: 'b', group: 'Box', label: 'Klasse', onSelect: () => {},
        submenu: [{ id: 'b1', label: 'Sky', shortcut: '2', onSelect: () => onKlasse('Sky') }] },
      { id: 'c', group: 'Projekt', label: 'Exportieren …', onSelect: () => {} },
    ];
  });
  return <div data-karte="k1"><span>Karte</span></div>;
}

describe('AppContextMenu', () => {
  it('zeigt Gruppen, Kuerzel und Untermenue und reicht das Ziel weiter', () => {
    const onZiel = vi.fn();
    const onKlasse = vi.fn();
    render(<><Seite onZiel={onZiel} onKlasse={onKlasse} /><AppContextMenu /></>);

    fireEvent.contextMenu(screen.getByText('Karte'), { clientX: 10, clientY: 10 });

    expect(onZiel).toHaveBeenCalledWith('k1');
    expect(screen.getByText('Box')).toBeInTheDocument();
    expect(screen.getByText('Projekt')).toBeInTheDocument();
    expect(screen.getByText('⌘D')).toBeInTheDocument();

    fireEvent.mouseEnter(screen.getByText('Klasse').closest('div')!);
    fireEvent.click(screen.getByText('Sky'));
    expect(onKlasse).toHaveBeenCalledWith('Sky');
  });
});
