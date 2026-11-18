// FrameTrain Business: Team, Plaetze und Anmeldefehler eines Platzes.

export type TeamRole = 'admin' | 'member' | 'viewer';

/** Was der Server bei der Anmeldung ueber das Team liefert (null = Einzelkonto). */
export interface TeamInfo {
  id: string;
  name: string;
  seatLimit: number;
  role: TeamRole;
  isOwner: boolean;
  seatId: string | null;
}

export interface TeamMember {
  id: string;
  name: string;
  role: TeamRole;
  isOwner: boolean;
  isYou: boolean;
  isActive: boolean;
  deviceName: string | null;
  lastSeenAt: string | null;
}

export interface TeamOverview extends TeamInfo {
  freeSeats: number;
  accessUntil: string | null;
  members: TeamMember[];
}

export type SeatEndReason = 'locked' | 'other_device' | 'no_access';

export interface TeamStatus {
  active: boolean;
  reason: SeatEndReason | null;
  team: TeamOverview | null;
}

/** Anmeldefehler, auf die die Maske gezielt reagiert (vom Rust-Backend als JSON). */
export type SeatLoginError =
  | { code: 'seat_password'; message: string; seatName?: string | null; teamName?: string | null }
  | { code: 'seat_in_use'; message: string; deviceName?: string | null }
  | { code: 'seat_locked'; message: string };

export function parseSeatLoginError(raw: unknown): SeatLoginError | null {
  const text = typeof raw === 'string' ? raw : raw instanceof Error ? raw.message : '';
  if (!text.startsWith('{')) return null;
  try {
    const v = JSON.parse(text);
    if (v && (v.code === 'seat_password' || v.code === 'seat_in_use' || v.code === 'seat_locked') && typeof v.message === 'string') {
      return v as SeatLoginError;
    }
  } catch { /* kein JSON */ }
  return null;
}

export function isTeamInfo(v: unknown): v is TeamInfo {
  if (!v || typeof v !== 'object') return false;
  const t = v as Record<string, unknown>;
  return typeof t.id === 'string' && typeof t.name === 'string' && typeof t.seatLimit === 'number';
}

/** Alle 5 Minuten: Platz gesperrt oder auf einem anderen Geraet uebernommen? */
export const TEAM_HEARTBEAT_MS = 5 * 60 * 1000;

/**
 * Ob die App abmelden muss. Netzwerkfehler (status null) zaehlen nicht —
 * offline weiterarbeiten bleibt moeglich.
 */
export function seatEnded(status: TeamStatus | null): SeatEndReason | null {
  if (!status || status.active) return null;
  return status.reason ?? 'no_access';
}

/** Zeitangabe "zuletzt aktiv" grob und lesbar. */
export function lastSeenBucket(iso: string | null, now = Date.now()): { key: 'never' | 'now' | 'minutes' | 'hours' | 'yesterday' | 'days'; n: number } {
  if (!iso) return { key: 'never', n: 0 };
  const diff = Math.max(0, now - new Date(iso).getTime());
  const min = Math.floor(diff / 60000);
  if (min < 10) return { key: 'now', n: 0 };
  if (min < 60) return { key: 'minutes', n: min };
  const h = Math.floor(min / 60);
  if (h < 24) return { key: 'hours', n: h };
  const d = Math.floor(h / 24);
  return d === 1 ? { key: 'yesterday', n: 1 } : { key: 'days', n: d };
}

export function initials(name: string): string {
  const parts = name.trim().split(/[\s.@_-]+/).filter(Boolean);
  const s = parts.length >= 2 ? parts[0][0] + parts[1][0] : (parts[0] ?? '?').slice(0, 2);
  return s.toUpperCase();
}
