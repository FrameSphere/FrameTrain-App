// Regression aus dem Library-Test (23.08.2026): Beide Skripte in der oeffentlichen
// Bibliothek waren von der Pruefung abgelehnt worden ("AI: Das Skript enthaelt
// keinen gueltigen Python-Code") — die App zeigte sie trotzdem nur als
// "Ungeprueft" an und bot sie zum Download an.

import { describe, it, expect } from 'vitest';
import { isRejected } from '../OpenLibraryModal';

describe('isRejected', () => {
  it('erkennt ein abgelehntes Skript am Zeitstempel', () => {
    expect(isRejected({ rejectedAt: '2026-05-25T00:40:31.907Z' })).toBe(true);
  });

  it('haelt ein ungeprueftes Skript nicht faelschlich fuer abgelehnt', () => {
    expect(isRejected({ rejectedAt: null })).toBe(false);
    expect(isRejected({ rejectedAt: undefined })).toBe(false);
    expect(isRejected({})).toBe(false);
  });

  it('wertet einen leeren Zeitstempel nicht als Ablehnung', () => {
    expect(isRejected({ rejectedAt: '' })).toBe(false);
  });
});
