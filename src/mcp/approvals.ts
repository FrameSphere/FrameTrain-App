// Schnittstelle fuer den Einstellungs-Tab "KI-Verbindungen" (Strang B).
// Strang B laedt dieses Modul per import.meta.glob('../../mcp/approvals.ts').
// Die Logik liegt in approval.ts; hier nur die vereinbarten Namen.

import { listGrants, revokeGrant, revokeClient, revokeAllGrants, subscribeGrants, type GrantKind } from './approval';

export interface McpApproval {
  /** client.name aus mcp://call */
  client: string;
  /** Werkzeugname */
  tool: string;
  /** ISO-Zeit der Freigabe */
  grantedAt?: string;
  /** 'once' = erste Zustimmung zu einem once-Werkzeug, 'always' = "immer erlauben" */
  kind?: GrantKind;
}

export function listApprovals(): McpApproval[] {
  return listGrants().map((g) => ({ client: g.client, tool: g.tool, grantedAt: g.grantedAt, kind: g.kind }));
}

export function revokeApproval(client: string, tool: string): void {
  revokeGrant(client, tool);
}

/** Alle Freigaben eines Clients widerrufen. */
export function revokeClientApprovals(client: string): void {
  revokeClient(client);
}

export function revokeAllApprovals(): void {
  revokeAllGrants();
}

export function onApprovalsChanged(cb: () => void): () => void {
  return subscribeGrants(cb);
}
