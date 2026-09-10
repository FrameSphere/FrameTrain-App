// ============================================================================
// Coach-Skills — vorbereitete Fragen, per "/" im Eingabefeld
// ----------------------------------------------------------------------------
// Das Problem, das sie loesen: der Coach kann viel, aber das leere Eingabefeld
// verraet nichts davon. Wer nicht weiss, dass er nach der RAM-Schaetzung fragen
// KANN, fragt nicht danach. Die Tool-Chips helfen erst NACH einer Antwort.
//
// Ein Skill ist bewusst kein neues Koennen, sondern ein guter Einstieg: ein
// Kurzbefehl, der eine praezise formulierte Frage einsetzt. Das kostet keinen
// einzigen zusaetzlichen Token im Prompt — der Kontext ist derselbe, nur die
// Frage ist besser gestellt als "hilf mir mal".
//
// Seiten-gebunden: auf der Dataset-Seite steht kein /ram, im Training kein
// /testskript. Die Liste bleibt dadurch kurz genug, um sie zu ueberfliegen.
// ============================================================================

import type { Language } from '../contexts/LanguageContext';
import type { PageId } from './coachContext';

type Bilingual = { de: string; en: string };

export interface CoachSkill {
  /** Kurzbefehl ohne Schraegstrich, klein geschrieben. */
  id: string;
  /** Anzeigename in der Liste. */
  label: Bilingual;
  /** Eine Zeile: was der Skill tut. */
  hint: Bilingual;
  /** Auf welchen Seiten der Skill angeboten wird. Leer = ueberall. */
  pages?: PageId[];
  /** Die Frage, die tatsaechlich abgeschickt wird. */
  prompt: Bilingual;
}

export const COACH_SKILLS: CoachSkill[] = [
  // ── Ueberall ─────────────────────────────────────────────────────────────
  {
    id: 'hier',
    label: { de: '/hier', en: '/here' },
    hint: { de: 'Was sehe ich gerade und was kann ich hier tun?', en: 'What am I looking at and what can I do here?' },
    prompt: {
      de: 'Erkläre mir kurz, was auf dieser Seite gerade zu sehen ist — anhand meines konkreten Zustands, nicht allgemein — und was der sinnvollste nächste Schritt wäre.',
      en: 'Briefly explain what is on this page right now — based on my concrete state, not in general — and what the most sensible next step would be.',
    },
  },
  {
    id: 'weiter',
    label: { de: '/weiter', en: '/next' },
    hint: { de: 'Was ist der nächste sinnvolle Schritt?', en: 'What is the next sensible step?' },
    prompt: {
      de: 'Was ist auf Basis meines aktuellen Zustands der nächste sinnvolle Schritt? Nenne genau einen und begründe ihn in zwei Sätzen.',
      en: 'Based on my current state, what is the next sensible step? Name exactly one and justify it in two sentences.',
    },
  },

  // ── Start ────────────────────────────────────────────────────────────────
  {
    id: 'stand',
    label: { de: '/stand', en: '/status' },
    hint: { de: 'Wie steht mein Projekt?', en: 'How is my project doing?' },
    pages: ['home'],
    prompt: {
      de: 'Bewerte den Stand meines Projekts anhand der Kennzahlen und des Loss-Trends. Was läuft gut, wo hakt es? Deute die Zahlen, lies sie nicht nur vor.',
      en: 'Assess the state of my project from the key figures and the loss trend. What is going well, where is it stuck? Interpret the numbers, do not just read them out.',
    },
  },

  // ── Models ───────────────────────────────────────────────────────────────
  {
    id: 'modellwahl',
    label: { de: '/modellwahl', en: '/pick-model' },
    hint: { de: 'Welches Modell passt zu meiner Aufgabe?', en: 'Which model fits my task?' },
    pages: ['models'],
    prompt: {
      de: 'Ich beschreibe gleich meine Aufgabe. Frage mich zuerst kurz nach Aufgabentyp und Datenmenge, wenn das aus dem Zustand nicht hervorgeht, und empfiehl dann zwei konkrete Modelle mit Begründung und ungefährem Speicherbedarf.',
      en: 'I will describe my task. First ask briefly about task type and data volume if that is not clear from the state, then recommend two concrete models with reasoning and rough memory requirements.',
    },
  },

  // ── Training ─────────────────────────────────────────────────────────────
  {
    id: 'config',
    label: { de: '/config', en: '/config' },
    hint: { de: 'Meine Einstellungen prüfen', en: 'Review my settings' },
    pages: ['training'],
    prompt: {
      de: 'Prüfe meine aktuelle Trainings-Config gegen Modell und Dataset. Nenne konkret, was du ändern würdest und warum — und biete die Änderung zum Übernehmen an. Wenn alles passt, sag das klar, statt etwas zu erfinden.',
      en: 'Review my current training config against model and dataset. Name concretely what you would change and why — and offer the change for applying. If everything fits, say so clearly instead of inventing something.',
    },
  },
  {
    id: 'ram',
    label: { de: '/ram', en: '/ram' },
    hint: { de: 'Passt der Lauf auf meine Maschine?', en: 'Will this run fit on my machine?' },
    pages: ['training'],
    prompt: {
      de: 'Reicht mein Arbeitsspeicher für diesen Lauf? Nutze die Schätzung aus dem Kontext, vergleiche sie mit meinem System-RAM und sag klar: passt, knapp oder reicht nicht. Falls es knapp wird, nenne die wirksamste Sparmaßnahme zuerst.',
      en: 'Is my memory enough for this run? Use the estimate from the context, compare it with my system RAM and say clearly: fits, tight or not enough. If it gets tight, name the most effective saving measure first.',
    },
  },
  {
    id: 'oom',
    label: { de: '/oom', en: '/oom' },
    hint: { de: 'Out-of-Memory vermeiden oder beheben', en: 'Avoid or fix out-of-memory' },
    pages: ['training', 'training-dev'],
    prompt: {
      de: 'Mein Training läuft in Out-of-Memory oder ich befürchte es. Nenne die Maßnahmen in der Reihenfolge ihrer Wirkung, jeweils mit dem konkreten Wert für meine Config — und was jede Maßnahme an Qualität oder Zeit kostet.',
      en: 'My training runs out of memory or I fear it will. Name the measures in order of effect, each with the concrete value for my config — and what each one costs in quality or time.',
    },
  },
  {
    id: 'schneller',
    label: { de: '/schneller', en: '/faster' },
    hint: { de: 'Training beschleunigen', en: 'Speed up training' },
    pages: ['training'],
    prompt: {
      de: 'Wie mache ich diesen Lauf schneller, ohne die Ergebnisqualität nennenswert zu opfern? Konkrete Werte für meine Config, und sag dazu, was der jeweilige Preis ist.',
      en: 'How do I make this run faster without sacrificing result quality noticeably? Concrete values for my config, and say what the price of each is.',
    },
  },

  // ── Dev-Modi ─────────────────────────────────────────────────────────────
  {
    id: 'fehler',
    label: { de: '/fehler', en: '/error' },
    hint: { de: 'Fehler im Output erklären', en: 'Explain the error in the output' },
    pages: ['training-dev', 'tests-dev', 'training', 'tests'],
    prompt: {
      de: 'Erkläre den Fehler, der gerade im Output steht: erst die Ursache in einem Satz, dann der konkrete Fix. Keine allgemeine Fehlerkunde.',
      en: 'Explain the error currently in the output: first the cause in one sentence, then the concrete fix. No general error theory.',
    },
  },
  {
    id: 'review',
    label: { de: '/review', en: '/review' },
    hint: { de: 'Mein Skript durchsehen', en: 'Review my script' },
    pages: ['training-dev', 'tests-dev'],
    prompt: {
      de: 'Sieh mein Skript durch: Läuft es so? Nenne die Stellen, die brechen oder unsauber sind, mit Zeilenbezug — die wichtigste zuerst. Wenn es solide ist, sag das.',
      en: 'Review my script: will it run as is? Name the places that break or are sloppy, with line references — the most important first. If it is solid, say so.',
    },
  },

  // ── Dataset ──────────────────────────────────────────────────────────────
  {
    id: 'daten',
    label: { de: '/daten', en: '/data' },
    hint: { de: 'Ist mein Dataset trainingsbereit?', en: 'Is my dataset ready for training?' },
    pages: ['dataset'],
    prompt: {
      de: 'Ist mein Dataset so trainingsbereit? Prüfe Format, Split-Status und Größe und sag mir, was noch fehlt — mit dem Klick, der es behebt.',
      en: 'Is my dataset ready for training as is? Check format, split status and size and tell me what is missing — with the click that fixes it.',
    },
  },

  // ── Analyse ──────────────────────────────────────────────────────────────
  {
    id: 'bewerten',
    label: { de: '/bewerten', en: '/assess' },
    hint: { de: 'Ist dieser Lauf gut geworden?', en: 'Did this run turn out well?' },
    pages: ['analysis'],
    prompt: {
      de: 'Ist dieser Lauf gut geworden? Nenne die konkreten Zahlen aus dem Kontext (Loss-Reduktion, Overfitting-Gap) und ordne sie ein — gut, brauchbar oder Problem. Keine erfundenen Werte; was du nicht hast, sagst du.',
      en: 'Did this run turn out well? Name the concrete numbers from the context (loss reduction, overfitting gap) and classify them — good, usable or a problem. No invented values; say what you do not have.',
    },
  },
  {
    id: 'naechsterlauf',
    label: { de: '/naechsterlauf', en: '/next-run' },
    hint: { de: 'Was ändere ich im nächsten Lauf?', en: 'What do I change for the next run?' },
    pages: ['analysis'],
    prompt: {
      de: 'Was ändere ich für den nächsten Lauf? Höchstens drei Parameter mit konkreten Werten, begründet aus den Kurven — und biete sie zum Übernehmen an.',
      en: 'What do I change for the next run? At most three parameters with concrete values, justified from the curves — and offer them for applying.',
    },
  },
  {
    id: 'kurven',
    label: { de: '/kurven', en: '/curves' },
    hint: { de: 'Was sagen mir die Kurven?', en: 'What do the curves tell me?' },
    pages: ['analysis'],
    prompt: {
      de: 'Lies mir die Loss-Kurven: Was ist der Verlauf, was bedeutet er, und woran erkenne ich das? Beziehe dich auf meine Zahlen, nicht auf Lehrbuchfälle.',
      en: 'Read the loss curves for me: what is the shape, what does it mean, and how do I recognise it? Refer to my numbers, not textbook cases.',
    },
  },

  // ── Tests ────────────────────────────────────────────────────────────────
  {
    id: 'ergebnis',
    label: { de: '/ergebnis', en: '/result' },
    hint: { de: 'Testergebnis deuten', en: 'Interpret the test result' },
    pages: ['tests', 'tests-dev'],
    prompt: {
      de: 'Deute mein Testergebnis: Ist das gut? Woran liegt es, wenn nicht — am Modell, an der Version oder am Eingabeformat?',
      en: 'Interpret my test result: is it good? If not, what is the cause — the model, the version or the input format?',
    },
  },

  // ── Versionen ────────────────────────────────────────────────────────────
  {
    id: 'version',
    label: { de: '/version', en: '/version' },
    hint: { de: 'Welche Version ist die beste?', en: 'Which version is the best?' },
    pages: ['versions'],
    prompt: {
      de: 'Welche meiner Versionen ist die beste und woran machst du das fest? Sag auch, welche ich gefahrlos aufräumen kann.',
      en: 'Which of my versions is the best and what do you base that on? Also say which ones I can safely clean up.',
    },
  },

  // ── Einstellungen ────────────────────────────────────────────────────────
  {
    id: 'budget',
    label: { de: '/budget', en: '/budget' },
    hint: { de: 'Welches Token-Budget passt zu mir?', en: 'Which token budget suits me?' },
    pages: ['settings'],
    prompt: {
      de: 'Erkläre mir die Token-Budgets: Was ändert sich zwischen Minimal und Unlimited konkret an den Antworten, und welches passt zu meinem aktuellen Anbieter?',
      en: 'Explain the token budgets: what concretely changes in the answers between Minimal and Unlimited, and which one suits my current provider?',
    },
  },
  {
    id: 'anbieter',
    label: { de: '/anbieter', en: '/provider' },
    hint: { de: 'Welcher KI-Anbieter passt zu mir?', en: 'Which AI provider suits me?' },
    pages: ['settings'],
    prompt: {
      de: 'Welcher der vier Anbieter passt zu mir? Nenne für jeden den entscheidenden Vor- und Nachteil in einem Satz — Kosten, Tempo, Datenschutz — und gib eine Empfehlung.',
      en: 'Which of the four providers suits me? Name the decisive pro and con for each in one sentence — cost, speed, privacy — and give a recommendation.',
    },
  },

  // ── Laboratory / Synapse ─────────────────────────────────────────────────
  {
    id: 'experiment',
    label: { de: '/experiment', en: '/experiment' },
    hint: { de: 'Ein sinnvolles Experiment aufsetzen', en: 'Set up a meaningful experiment' },
    pages: ['laboratory'],
    prompt: {
      de: 'Hilf mir, ein sinnvolles Experiment aufzusetzen: Was variiere ich, was halte ich fest, und woran erkenne ich hinterher, welche Variante besser war?',
      en: 'Help me set up a meaningful experiment: what do I vary, what do I hold fixed, and how will I recognise afterwards which variant was better?',
    },
  },
  {
    id: 'graph',
    label: { de: '/graph', en: '/graph' },
    hint: { de: 'Meinen Node-Graphen erklären', en: 'Explain my node graph' },
    pages: ['synapse'],
    prompt: {
      de: 'Erkläre meinen aktuellen Graphen: Was macht er, wo sind offene Enden oder unplausible Verbindungen?',
      en: 'Explain my current graph: what does it do, where are loose ends or implausible connections?',
    },
  },
];

/** Skills, die auf dieser Seite angeboten werden (seitenlose zuerst). */
export function skillsForPage(pageId: PageId | null): CoachSkill[] {
  const global = COACH_SKILLS.filter(s => !s.pages || s.pages.length === 0);
  if (!pageId) return global;
  const scoped = COACH_SKILLS.filter(s => s.pages?.includes(pageId));
  return [...global, ...scoped];
}

/**
 * Filtert die Skill-Liste nach dem, was hinter dem "/" getippt wurde.
 * Gibt null zurueck, wenn die Eingabe gar kein Skill-Aufruf ist.
 */
export function matchSkills(input: string, pageId: PageId | null): CoachSkill[] | null {
  // Nur ein "/" am ANFANG oeffnet die Liste — ein Schraegstrich mitten im Text
  // (etwa in einem Pfad) ist normaler Inhalt.
  const m = /^\/([a-zA-Z-]*)$/.exec(input);
  if (!m) return null;
  const query = m[1].toLowerCase();
  const available = skillsForPage(pageId);
  if (!query) return available;
  return available.filter(s =>
    s.id.startsWith(query) || s.label.en.replace('/', '').startsWith(query),
  );
}

export const skillLabel = (s: CoachSkill, language: Language) => (language === 'en' ? s.label.en : s.label.de);
export const skillHint = (s: CoachSkill, language: Language) => (language === 'en' ? s.hint.en : s.hint.de);
export const skillPrompt = (s: CoachSkill, language: Language) => (language === 'en' ? s.prompt.en : s.prompt.de);
