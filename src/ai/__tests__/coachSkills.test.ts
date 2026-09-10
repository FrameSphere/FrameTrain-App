// Coach-Skills — die "/"-Kurzbefehle im Eingabefeld.
//
// Sie loesen das leere Eingabefeld: der Coach kann viel, aber nichts davon ist
// sichtbar, bevor man die richtige Frage stellt. Wichtig ist, dass die Liste
// seitenbezogen kurz bleibt und ein Schraegstrich MITTEN im Text (etwa in
// einem Pfad) nichts aufklappt.
//
// Ausfuehren: npx vitest run src/ai/__tests__/coachSkills.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { COACH_SKILLS, matchSkills, skillsForPage, skillPrompt } from '../coachSkills';
import { PAGE_KNOWLEDGE_PAGES } from './__helpers__/pages';

describe('matchSkills', () => {
  it('oeffnet die Liste bei einem einzelnen Schraegstrich', () => {
    const m = matchSkills('/', 'training');
    expect(m).not.toBeNull();
    expect(m!.length).toBeGreaterThan(3);
  });

  it('filtert nach dem Getippten', () => {
    const m = matchSkills('/ra', 'training');
    expect(m!.map(s => s.id)).toContain('ram');
    expect(m!.map(s => s.id)).not.toContain('config');
  });

  it('reagiert NICHT auf einen Schraegstrich mitten im Text', () => {
    expect(matchSkills('Der Pfad /Users/karol/model', 'training')).toBeNull();
    expect(matchSkills('was bedeutet 1/2', 'training')).toBeNull();
    expect(matchSkills('normale Frage', 'training')).toBeNull();
  });

  it('gibt eine leere Liste zurueck, wenn nichts passt — statt zu oeffnen', () => {
    expect(matchSkills('/gibtsnicht', 'training')).toEqual([]);
  });
});

describe('skillsForPage', () => {
  it('bietet die RAM-Frage im Training an, nicht im Dataset', () => {
    expect(skillsForPage('training').map(s => s.id)).toContain('ram');
    expect(skillsForPage('dataset').map(s => s.id)).not.toContain('ram');
  });

  it('bietet die seitenlosen Skills ueberall an', () => {
    for (const page of PAGE_KNOWLEDGE_PAGES) {
      const ids = skillsForPage(page).map(s => s.id);
      expect(ids, `fehlt auf ${page}`).toContain('hier');
      expect(ids, `fehlt auf ${page}`).toContain('weiter');
    }
  });

  it('haelt die Liste pro Seite ueberschaubar', () => {
    for (const page of PAGE_KNOWLEDGE_PAGES) {
      expect(skillsForPage(page).length, `zu viele auf ${page}`).toBeLessThanOrEqual(8);
    }
  });
});

describe('Skill-Definitionen', () => {
  it('hat eindeutige Kurzbefehle', () => {
    const ids = COACH_SKILLS.map(s => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('liefert in beiden Sprachen eine echte Frage', () => {
    for (const skill of COACH_SKILLS) {
      expect(skillPrompt(skill, 'de').length, skill.id).toBeGreaterThan(30);
      expect(skillPrompt(skill, 'en').length, skill.id).toBeGreaterThan(30);
    }
  });

  it('benutzt keine Emojis — die UI ist emoji-frei', () => {
    const emoji = /\p{Extended_Pictographic}/u;
    for (const skill of COACH_SKILLS) {
      expect(emoji.test(skill.label.de + skill.hint.de + skill.prompt.de), skill.id).toBe(false);
      expect(emoji.test(skill.label.en + skill.hint.en + skill.prompt.en), skill.id).toBe(false);
    }
  });
});
