import { describe, expect, it } from 'vitest';
import { initials, isTeamInfo, lastSeenBucket, parseSeatLoginError, seatEnded } from '../team';

describe('parseSeatLoginError', () => {
  it('reads the JSON errors of the Rust backend', () => {
    expect(parseSeatLoginError('{"code":"seat_in_use","message":"m","deviceName":"Annas Mac"}'))
      .toEqual({ code: 'seat_in_use', message: 'm', deviceName: 'Annas Mac' });
    expect(parseSeatLoginError(new Error('{"code":"seat_password","message":"m","teamName":"Muster"}'))?.code)
      .toBe('seat_password');
  });
  it('ignores plain errors and unknown codes', () => {
    expect(parseSeatLoginError('Falsches Passwort')).toBeNull();
    expect(parseSeatLoginError('{"code":"other","message":"m"}')).toBeNull();
    expect(parseSeatLoginError('{kaputt')).toBeNull();
    expect(parseSeatLoginError(undefined)).toBeNull();
  });
});

describe('seatEnded', () => {
  it('network errors keep the session', () => {
    expect(seatEnded(null)).toBeNull();
  });
  it('active seats keep the session', () => {
    expect(seatEnded({ active: true, reason: null, team: null })).toBeNull();
  });
  it('reports why the seat ended', () => {
    expect(seatEnded({ active: false, reason: 'other_device', team: null })).toBe('other_device');
    expect(seatEnded({ active: false, reason: null, team: null })).toBe('no_access');
  });
});

describe('helpers', () => {
  it('isTeamInfo', () => {
    expect(isTeamInfo({ id: 't', name: 'Muster', seatLimit: 5, role: 'admin', isOwner: true, seatId: null })).toBe(true);
    expect(isTeamInfo(null)).toBe(false);
    expect(isTeamInfo({ id: 't' })).toBe(false);
  });
  it('lastSeenBucket', () => {
    const now = Date.parse('2026-11-17T12:00:00Z');
    expect(lastSeenBucket(null, now).key).toBe('never');
    expect(lastSeenBucket('2026-11-17T11:55:00Z', now).key).toBe('now');
    expect(lastSeenBucket('2026-11-17T11:30:00Z', now)).toEqual({ key: 'minutes', n: 30 });
    expect(lastSeenBucket('2026-11-17T09:00:00Z', now)).toEqual({ key: 'hours', n: 3 });
    expect(lastSeenBucket('2026-11-16T10:00:00Z', now)).toEqual({ key: 'yesterday', n: 1 });
    expect(lastSeenBucket('2026-11-14T12:00:00Z', now)).toEqual({ key: 'days', n: 3 });
  });
  it('initials', () => {
    expect(initials('Anna Muster')).toBe('AM');
    expect(initials('karol@firma.de')).toBe('KF');
    expect(initials('Ben')).toBe('BE');
  });
});
