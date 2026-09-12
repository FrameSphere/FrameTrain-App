// Das Tool-Protokoll des Coaches — seitenabhaengig statt pauschal.
//
// Vorher ging die vollstaendige Liste an jede Seite. Das kostete rund 570
// Token pro Runde und verleitete das Modell dazu, Buttons anzubieten, die auf
// der aktuellen Seite nichts bewirken (etwa [[set:…]] auf der Dataset-Seite).
//
// Ausfuehren: npx vitest run src/ai/__tests__/coachTools.test.ts --config vitest.config.ts

import { describe, it, expect } from 'vitest';
import { buildCoachSystemPrompt, pageKnowledge, parseCoachActions, dropUnavailableSetFields, type PageId } from '../coachContext';
import { expandHiddenFields } from '../coachToolEvents';
import { PAGE_KNOWLEDGE_PAGES } from './__helpers__/pages';

const prompt = (pageId: PageId | null, automation = false) =>
  buildCoachSystemPrompt({
    language: 'de',
    pageId,
    pageContent: '',
    isFirstMessage: true,
    pageChanged: false,
    includePageKnowledge: false,
    automation,
  });

describe('Tool-Protokoll pro Seite', () => {
  it('bietet Navigation, Rueckfrage und Hilfe-Link ueberall an', () => {
    for (const page of ['training', 'dataset', 'settings', 'synapse'] as PageId[]) {
      const p = prompt(page);
      expect(p).toContain('[[go:SEITE]]');
      expect(p).toContain('[[ask:');
      expect(p).toContain('[[link:KEY]]');
    }
  });

  it('bietet die Trainings-Config nur im Training an', () => {
    expect(prompt('training')).toContain('[[set:key=wert');
    for (const page of ['dataset', 'analysis', 'settings', 'models'] as PageId[]) {
      expect(prompt(page)).not.toContain('[[set:key=wert');
    }
  });

  it('bietet Split nur auf der Dataset-Seite und HF-Suche nicht im Training an', () => {
    expect(prompt('dataset')).toContain('[[split:');
    expect(prompt('training')).not.toContain('[[split:');
    expect(prompt('models')).toContain('[[hf:');
    expect(prompt('settings')).not.toContain('[[hf:');
  });

  it('bietet die RAM-Schaetzung nur dort an, wo eine Config steht', () => {
    expect(prompt('training')).toContain('[[estimate:ram]]');
    expect(prompt('analysis')).not.toContain('[[estimate:ram]]');
  });

  it('schaltet Start/Stop nur mit Automation UND nur im Training frei', () => {
    expect(prompt('training', true)).toContain('[[train:start]]');
    expect(prompt('training', false)).not.toContain('[[train:start]]');
    expect(prompt('dataset', true)).not.toContain('[[train:start]]');
  });

  it('faellt ohne bekannte Seite auf die volle Liste zurueck', () => {
    const p = prompt(null);
    expect(p).toContain('[[set:key=wert');
    expect(p).toContain('[[split:');
    expect(p).toContain('[[hf:');
  });

  it('ist auf einer Seite ohne Sonderwerkzeuge deutlich kuerzer als im Training', () => {
    // Verglichen wird nur der Tool-Block; Persona und Ueberblick sind gleich.
    const toolsOnly = (page: PageId) =>
      buildCoachSystemPrompt({
        language: 'de', pageId: page, pageContent: '', isFirstMessage: false,
        pageChanged: false, includePageKnowledge: false, repeatTools: true,
      }).length;
    expect(toolsOnly('settings')).toBeLessThan(toolsOnly('training') * 0.8);
  });
});

describe('Seiten-Wissen', () => {
  // Die Startseite meldet einen Live-Zustand, hatte aber als einzige solche
  // Seite kein Wissen hinterlegt — ausgerechnet der Einstiegspunkt.
  it('jede Seite, die einen Live-Zustand meldet, bringt auch Wissen mit', () => {
    for (const page of PAGE_KNOWLEDGE_PAGES) {
      expect(pageKnowledge(page, 'de'), `Seiten-Wissen fehlt: ${page}`).not.toBe('');
      expect(pageKnowledge(page, 'en'), `page knowledge missing: ${page}`).not.toBe('');
    }
  });
});

describe('Ueberschrift des Live-Blocks', () => {
  const withState = (opts: { pageChanged: boolean; stateChanged: boolean }) =>
    buildCoachSystemPrompt({
      language: 'de',
      pageId: 'training',
      pageContent: 'batch_size=32',
      isFirstMessage: false,
      pageChanged: opts.pageChanged,
      stateChanged: opts.stateChanged,
      includePageKnowledge: false,
    });

  it('nennt einen echten Seitenwechsel beim Namen', () => {
    expect(withState({ pageChanged: true, stateChanged: true })).toContain('Seite gewechselt');
  });

  it('nennt eine reine Zustandsaenderung NICHT Seitenwechsel', () => {
    const p = withState({ pageChanged: false, stateChanged: true });
    expect(p).not.toContain('Seite gewechselt');
    expect(p).toContain('Zustand hat sich');
  });

  it('bleibt neutral, wenn sich nichts geaendert hat', () => {
    const p = withState({ pageChanged: false, stateChanged: false });
    expect(p).toContain('Aktuelle Seite (Live-Zustand)');
  });
});

// Praxisfall 1.2.61: Fuer YOLO bot der Coach "Uebernehmen: Gradient
// Checkpointing an" an — ein Schalter, den Ultralytics nicht kennt.
describe('Nicht verfuegbare Felder', () => {
  it('loest die Gruppe "lora" in die einzelnen Felder auf', () => {
    const fields = expandHiddenFields(['lora', 'gradient_checkpointing']);
    for (const f of ['use_lora', 'lora_r', 'load_in_4bit', 'gradient_checkpointing']) {
      expect(fields.has(f), f).toBe(true);
    }
  });

  it('wirft nicht verfuegbare Felder aus einem Button und behaelt den Rest', () => {
    const { actions } = parseCoachActions('Text [[set:gradient_checkpointing=true;batch_size=16]]');
    const filtered = dropUnavailableSetFields(actions, new Set(['gradient_checkpointing']));
    expect(filtered).toHaveLength(1);
    expect(filtered[0]).toMatchObject({ type: 'set', patch: { batch_size: 16 } });
    expect((filtered[0] as { summary: string }).summary).not.toMatch(/Checkpointing/);
  });

  it('entfernt einen Button ganz, wenn nichts Verfuegbares uebrig bleibt', () => {
    const { actions } = parseCoachActions('Text [[set:use_lora=true]] [[go:analysis]]');
    const filtered = dropUnavailableSetFields(actions, expandHiddenFields(['lora']));
    expect(filtered.map(a => a.type)).toEqual(['navigate']);
  });

  it('laesst ohne Einschraenkung alles durch', () => {
    const { actions } = parseCoachActions('Text [[set:gradient_checkpointing=true]]');
    expect(dropUnavailableSetFields(actions, new Set())).toEqual(actions);
  });
});
