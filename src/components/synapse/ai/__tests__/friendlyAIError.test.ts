// Regression aus dem Synapse-Test vom 23.08.2026: Im KI-Panel stand der rohe
// Groq-Fehler inklusive Organisations-ID — "Request too large for model
// 'openai/gpt-oss-120b' in organization 'org_01kp341...' service tier
// 'on_demand' on tokens per minute (TPM): Limit 8000, Requested 8191 ...".
// Damit konnte niemand etwas anfangen, und die Org-ID gehoert nicht in die UI.

import { describe, it, expect } from 'vitest';
import { friendlyAIError } from '../synapseAgent';

const GROQ_TOO_LARGE =
  "Request too large for model `openai/gpt-oss-120b` in organization `org_01kp341qx5fxbtqqcffgqx9sqs` " +
  "service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 8191, " +
  "please reduce your message size and try again.";

describe('friendlyAIError', () => {
  it('nennt bei zu grosser Anfrage das Limit und den Ausweg', () => {
    const de = friendlyAIError(GROQ_TOO_LARGE, 'de');
    expect(de).toContain('8000');
    expect(de).toContain('Token-Budget');
    expect(de).not.toContain('org_01kp341qx5fxbtqqcffgqx9sqs');
    expect(de).not.toContain('gpt-oss-120b');
  });

  it('antwortet auf Englisch, wenn die Oberflaeche englisch ist', () => {
    const en = friendlyAIError(GROQ_TOO_LARGE, 'en');
    expect(en).toContain('token limit');
    expect(en).toContain('Balanced');
    expect(en).not.toContain('organization');
  });

  it('unterscheidet ein normales Rate-Limit von einer zu grossen Anfrage', () => {
    const out = friendlyAIError('Rate limit reached, please try again in 4.2s', 'de');
    expect(out).toContain('Limit');
    expect(out).not.toContain('Token-Budget');
  });

  it('kuerzt unbekannte Fehler auf die erste Zeile', () => {
    const raw = 'Connection refused\n  at fetch (native)\n  at callAI (aiClient.ts:42)';
    expect(friendlyAIError(raw, 'de')).toBe('Connection refused');
  });

  it('schneidet sehr lange Einzeiler ab', () => {
    const out = friendlyAIError('x'.repeat(500), 'de');
    expect(out.length).toBeLessThanOrEqual(201);
    expect(out.endsWith('…')).toBe(true);
  });
});
