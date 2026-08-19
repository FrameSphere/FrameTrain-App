// Gemeinsame Token-Erkennung für Modellnamen.
//
// Wortgrenzen sind entscheidend: ohne sie matcht "bert" in "albert" und
// "resnet" in "detr-resnet-50". Genau daran sind frühere Erkennungen
// gescheitert — Modelle galten als trainierbar und wurden erst beim
// Trainingsstart abgewiesen, nach dem vollständigen Download.

export function containsToken(haystack: string, token: string): boolean {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i').test(haystack);
}

export function normalizePath(modelPathOrId: string): string {
  return modelPathOrId.toLowerCase().replace(/\\/g, '/');
}

/** Verzeichnisnamen, die kein Modellname sind. */
const OPAQUE_DIR = /^(ver_[a-z0-9]+|hf_[a-z0-9]+|snapshots|blobs|refs|model|models|versions|original|latest|[0-9a-f]{12,})$/;

/** Reduziert Pfad oder Repo-ID auf das Segment mit dem Modellnamen. */
export function modelNameSegment(normalized: string): string {
  const segments = normalized.split('/').filter(Boolean);
  if (segments.length === 0) return normalized;
  for (let i = segments.length - 1; i >= 0; i--) {
    if (!OPAQUE_DIR.test(segments[i])) return segments[i];
  }
  return segments[segments.length - 1];
}
