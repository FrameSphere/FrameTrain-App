/**
 * Angemeldeter Nutzer fuer Module ohne React-Props (localStorage-Helfer,
 * MCP-Freigaben). App.tsx setzt ihn synchron beim Login und leert ihn beim
 * Logout — vor dem Rendern der Seiten, damit deren erste Lesezugriffe schon
 * im richtigen Namensraum landen.
 */
let currentUserId: string | null = null;

export function setCurrentUser(userId: string | null): void {
  currentUserId = userId || null;
}

export function getCurrentUser(): string | null {
  return currentUserId;
}

/**
 * localStorage-Schluessel pro Account. Alte gemeinsame Eintraege (ohne
 * Suffix) werden bewusst NICHT uebernommen: Es ist nicht feststellbar, von
 * welchem Account sie stammen.
 */
export function userScopedKey(base: string): string {
  return `${base}_${currentUserId ?? 'anon'}`;
}
