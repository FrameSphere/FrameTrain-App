// ============================================================================
// Coach-Tool-Events — client-seitige Aktionen, die der AI Coach auslösen kann.
// ----------------------------------------------------------------------------
// Der Coach schlägt Aktionen vor (Tokens wie [[set:…]], [[open:…]] …); der User
// bestätigt per Klick auf einen Chip; dieser dispatcht hier ein Event, das die
// zuständige Seite abfängt. Nichts passiert ohne User-Klick.
//
// Für Aktionen auf ANDEREN Seiten: der Chip navigiert zuerst dorthin und ruft
// dann runCoachCommand() — die Zielseite verarbeitet das Kommando entweder live
// (schon gemountet) oder beim Mounten über consumePendingCoachCommand().
// ============================================================================

// ── 1. Config-Patch (Training) ──────────────────────────────────────────────

/** Ein Patch auf die Trainings-Config (Feldname → Wert). */
export type CoachConfigPatch = Record<string, number | boolean | string>;

const APPLY_CONFIG_EVENT = 'ft_coach_apply_config';

/**
 * Wie viele Seiten gerade auf Config-Patches hoeren.
 *
 * Ohne diesen Zaehler ging jeder Patch verloren, der ausgeloest wurde, waehrend
 * das Training-Panel nicht gemountet war — also im Normalfall, denn den Coach
 * fragt man von der Seite aus, auf der man gerade ist. Der Chip meldete
 * trotzdem "Übernommen".
 */
let applyListeners = 0;

/** Zuletzt vorgemerkter Patch, den das Training beim Öffnen übernimmt. */
let pendingPatch: { patch: CoachConfigPatch; at: number } | null = null;

/** So lange bleibt ein vorgemerkter Patch gueltig (Zeit zum Hinnavigieren). */
const PENDING_PATCH_TTL_MS = 10 * 60 * 1000;

/**
 * Übergibt einen Config-Patch ans Training.
 *
 * @returns true, wenn eine gemountete Seite ihn sofort übernommen hat;
 *          false, wenn er nur vorgemerkt wurde (Training noch nicht offen).
 */
export function applyCoachConfig(patch: CoachConfigPatch): boolean {
  if (applyListeners === 0) {
    pendingPatch = { patch, at: Date.now() };
    return false;
  }
  try {
    window.dispatchEvent(new CustomEvent<CoachConfigPatch>(APPLY_CONFIG_EVENT, { detail: patch }));
    return true;
  } catch {
    pendingPatch = { patch, at: Date.now() };
    return false;
  }
}

export function onApplyCoachConfig(handler: (patch: CoachConfigPatch) => void) {
  const listener = (e: Event) => handler((e as CustomEvent<CoachConfigPatch>).detail || {});
  window.addEventListener(APPLY_CONFIG_EVENT, listener as EventListener);
  applyListeners += 1;
  return () => {
    window.removeEventListener(APPLY_CONFIG_EVENT, listener as EventListener);
    applyListeners = Math.max(0, applyListeners - 1);
  };
}

/**
 * Beim Mounten des Trainings: einen vorgemerkten Patch abholen.
 * Gibt jeden Patch nur einmal heraus.
 */
export function consumePendingCoachConfig(): CoachConfigPatch | null {
  if (pendingPatch && Date.now() - pendingPatch.at < PENDING_PATCH_TTL_MS) {
    const p = pendingPatch.patch;
    pendingPatch = null;
    return p;
  }
  pendingPatch = null;
  return null;
}

// ── 2. Allgemeine Kommandos (seitenspezifische Aktionen) ────────────────────

export type CoachCommand =
  | { kind: 'openDialog'; target: string }   // templates | ai-assistant | add-model | ram
  | { kind: 'splitDataset'; name?: string }
  | { kind: 'hfSearch'; query: string }
  | { kind: 'applyRecommended' }
  | { kind: 'startTraining' }
  | { kind: 'stopTraining' };

const COMMAND_EVENT = 'ft_coach_command';
const PENDING_TTL_MS = 8000;
let pending: { cmd: CoachCommand; at: number } | null = null;

/** Löst ein Kommando aus (und merkt es kurz vor, falls die Zielseite erst noch lädt). */
export function runCoachCommand(cmd: CoachCommand) {
  pending = { cmd, at: Date.now() };
  try {
    window.dispatchEvent(new CustomEvent<CoachCommand>(COMMAND_EVENT, { detail: cmd }));
  } catch { /* ignore */ }
}

export function onCoachCommand(handler: (cmd: CoachCommand) => void) {
  const listener = (e: Event) => handler((e as CustomEvent<CoachCommand>).detail);
  window.addEventListener(COMMAND_EVENT, listener as EventListener);
  return () => window.removeEventListener(COMMAND_EVENT, listener as EventListener);
}

/**
 * Beim Mounten einer Seite: ein noch offenes Kommando abholen, das diese Seite
 * verarbeiten kann (z.B. nach Navigation ausgelöst, bevor die Seite gemountet war).
 */
export function consumePendingCoachCommand(canHandle: (cmd: CoachCommand) => boolean): CoachCommand | null {
  if (pending && Date.now() - pending.at < PENDING_TTL_MS && canHandle(pending.cmd)) {
    const c = pending.cmd;
    pending = null;
    return c;
  }
  return null;
}

// ── 3. Empfohlene Parameter (Brücke Analysis → Training) ────────────────────
// Analysis berechnet KI-Empfehlungen; das Training kann sie via [[apply:recommended]]
// übernehmen. Wir halten den letzten Satz modulweit vor (kurzlebige Brücke).

let lastRecommendedParams: Record<string, unknown> | null = null;

export function setRecommendedParams(params: Record<string, unknown> | null) {
  lastRecommendedParams = params;
}

export function getRecommendedParams(): Record<string, unknown> | null {
  return lastRecommendedParams;
}
