// Zugriffsschicht auf das Dev-Train-Skript ausserhalb des Panels.
//
// Das Skript lebt im React-State von DevTrainPanel. Ist das Panel offen,
// lesen/schreiben Werkzeuge direkt dort (der Editor aktualisiert sich).
// Sonst landet ein geschriebenes Skript als Entwurf im localStorage; das
// Panel uebernimmt ihn beim naechsten Oeffnen in den Editor.

export interface DevTrainPanelHandle {
  userId: string;
  getScript(): string;
  setScript(script: string, origin?: string): void;
  /** Was im Panel gerade gewaehlt ist (Modell, Datensaetze, Version). */
  getSelection(): { modelId: string | null; datasetIds: string[]; versionId: string | null };
}

export interface SavedDevScript { id: string; name: string; script: string; savedAt: string }

let panel: DevTrainPanelHandle | null = null;

export function attachDevTrainPanel(handle: DevTrainPanelHandle): () => void {
  panel = handle;
  return () => { if (panel === handle) panel = null; };
}

export function getAttachedDevTrainPanel(userId: string): DevTrainPanelHandle | null {
  return panel && panel.userId === userId ? panel : null;
}

export const draftKey = (userId: string) => `ft_devtrain_mcp_draft_${userId}`;
/** Gleicher Schluessel wie die Skript-Bibliothek in DevTrainPanel. */
export const libraryKey = (userId: string) => `ft_saved_scripts_${userId}`;

export function readDevDraft(userId: string): { script: string; savedAt: string; origin?: string } | null {
  try {
    const raw = localStorage.getItem(draftKey(userId));
    const d = raw ? JSON.parse(raw) : null;
    return d && typeof d.script === 'string' ? d : null;
  } catch {
    return null;
  }
}

export function clearDevDraft(userId: string) {
  try { localStorage.removeItem(draftKey(userId)); } catch { /* egal */ }
}

/** Aktuelles Skript: offener Editor, sonst Entwurf, sonst null. */
export function readDevScript(userId: string): { script: string; source: 'panel' | 'draft' } | null {
  const p = getAttachedDevTrainPanel(userId);
  if (p) return { script: p.getScript(), source: 'panel' };
  const d = readDevDraft(userId);
  return d ? { script: d.script, source: 'draft' } : null;
}

export function writeDevScript(userId: string, script: string, origin?: string): 'panel' | 'draft' {
  const p = getAttachedDevTrainPanel(userId);
  if (p) {
    p.setScript(script, origin);
    return 'panel';
  }
  localStorage.setItem(draftKey(userId), JSON.stringify({ script, savedAt: new Date().toISOString(), origin }));
  return 'draft';
}

export function listSavedDevScripts(userId: string): SavedDevScript[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(libraryKey(userId)) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((s): s is SavedDevScript => !!s && typeof s.script === 'string') : [];
  } catch {
    return [];
  }
}

/** Legt ein Skript in der Bibliothek ab (neueste zuerst, max. 50 wie im Panel). */
export function saveDevScriptToLibrary(userId: string, name: string, script: string): SavedDevScript {
  const entry: SavedDevScript = { id: `sc_${Date.now()}`, name, script, savedAt: new Date().toISOString() };
  const all = [entry, ...listSavedDevScripts(userId)].slice(0, 50);
  localStorage.setItem(libraryKey(userId), JSON.stringify(all));
  return entry;
}
