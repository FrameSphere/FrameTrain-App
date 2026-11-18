// Team-Funktionen sind nur fuer Business-Konten freigeschaltet: Einzelkonten
// sehen keinen Team-Eintrag, und die Team-Seite faellt fuer sie auf Start zurueck.

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(() => Promise.resolve(null)) }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn(() => Promise.resolve(() => {})) }));

import Sidebar from '../../components/Sidebar';
import { LanguageProvider } from '../../contexts/LanguageContext';
import { TrainingContextProvider } from '../../contexts/TrainingContext';
import type { TeamInfo } from '../team';

const team: TeamInfo = { id: 't1', name: 'Muster GmbH', seatLimit: 5, role: 'member', isOwner: false, seatId: 's1' };

function renderSidebar(t: TeamInfo | null, memberName: string | null = null) {
  return render(
    <LanguageProvider>
      <TrainingContextProvider>
        <Sidebar currentView="home" onViewChange={() => {}} userEmail="karol@muster.de" memberName={memberName} team={t} onLogout={() => {}} />
      </TrainingContextProvider>
    </LanguageProvider>,
  );
}

describe('Business-Freischaltung', () => {
  it('Einzelkonto: kein Team-Eintrag, kein Team-Name', () => {
    renderSidebar(null);
    expect(screen.queryByRole('button', { name: 'Team' })).toBeNull();
    expect(screen.queryByText('Muster GmbH')).toBeNull();
    expect(screen.getByText('karol@muster.de')).toBeTruthy();
  });

  it('Business-Konto: Team-Eintrag und Name der Person', () => {
    renderSidebar(team, 'Anna Muster');
    expect(screen.getByRole('button', { name: 'Team' })).toBeTruthy();
    expect(screen.getByText('Muster GmbH')).toBeTruthy();
    expect(screen.getByText('Anna Muster')).toBeTruthy();
  });
});
