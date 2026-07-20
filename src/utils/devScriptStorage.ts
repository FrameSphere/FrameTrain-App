// Zentrale Helfer für gespeicherte Dev-Scripts (DevTrain + DevTest + Lab).
//
// Historie: DevTrain speichert seit dem User-Scoping unter
// `ft_saved_scripts_<userId>`, DevTest speicherte aber weiterhin GLOBAL unter
// `ft_saved_test_scripts` — damit sahen alle Accounts auf demselben Gerät
// dieselben Test-Scripts. Diese Utility scoped beides pro User und migriert
// die globalen Alt-Keys einmalig in den Key des ersten angemeldeten Users,
// der sie liest (danach werden die globalen Keys entfernt).

export interface StoredDevScript {
  id: string;
  name: string;
  script: string;
  savedAt: string;
}

export const trainScriptsKey = (userId?: string) =>
  userId ? `ft_saved_scripts_${userId}` : 'ft_saved_scripts';

export const testScriptsKey = (userId?: string) =>
  userId ? `ft_saved_test_scripts_${userId}` : 'ft_saved_test_scripts';

function readKey(key: string): StoredDevScript[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) ?? '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Migriert die globalen Legacy-Keys in die user-spezifischen Keys.
 * Idempotent — nach dem ersten Lauf sind die globalen Keys weg.
 */
export function migrateLegacyDevScripts(userId?: string): void {
  if (!userId) return;
  const pairs: Array<[string, string]> = [
    ['ft_saved_scripts', trainScriptsKey(userId)],
    ['ft_saved_test_scripts', testScriptsKey(userId)],
  ];
  for (const [legacyKey, userKey] of pairs) {
    try {
      const legacyRaw = localStorage.getItem(legacyKey);
      if (!legacyRaw) continue;
      const legacy = JSON.parse(legacyRaw);
      if (Array.isArray(legacy) && legacy.length > 0) {
        const existing = readKey(userKey);
        const ids = new Set(existing.map((s) => s.id));
        const merged = [
          ...existing,
          ...legacy.filter((s: StoredDevScript) => s?.id && s?.script && !ids.has(s.id)),
        ].slice(0, 50);
        localStorage.setItem(userKey, JSON.stringify(merged));
      }
      localStorage.removeItem(legacyKey);
    } catch { /* defekter Legacy-Eintrag — ignorieren */ }
  }
}

/** Liest NUR die Scripts des angegebenen Users (inkl. vorheriger Migration). */
export function readUserDevScripts(userId?: string): {
  train: StoredDevScript[];
  test: StoredDevScript[];
} {
  migrateLegacyDevScripts(userId);
  if (!userId) return { train: [], test: [] };
  return {
    train: readKey(trainScriptsKey(userId)),
    test: readKey(testScriptsKey(userId)),
  };
}
