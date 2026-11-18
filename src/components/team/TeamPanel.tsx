// FrameTrain Business: Team-Seite. Zeigt das Team, die Plaetze und wer
// zuletzt aktiv war. Plaetze vergibt und sperrt der Inhaber im Web-Dashboard.

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import { ExternalLink, Loader2, MonitorSmartphone, RefreshCw, UserPlus } from 'lucide-react';
import TeamCatalog from './TeamCatalog';
import type { CatalogActivity } from '../../team/catalog';
import { useLanguage } from '../../contexts/LanguageContext';
import { initials, lastSeenBucket, type TeamInfo, type TeamMember, type TeamStatus } from '../../team/team';

interface TeamPanelProps {
  apiKey: string;
  /** Was die Anmeldung schon ueber das Team weiss (zeigt sofort etwas an) */
  team: TeamInfo;
}

export default function TeamPanel({ apiKey, team }: TeamPanelProps) {
  const { t, language } = useLanguage();
  const [status, setStatus] = useState<TeamStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'catalog' | 'members' | 'activity'>('catalog');
  const [activity, setActivity] = useState<CatalogActivity[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setStatus(await invoke<TeamStatus>('team_overview', { apiKey }));
    } catch (e) {
      setError(typeof e === 'string' ? e : t('team.loadError'));
    } finally {
      setLoading(false);
    }
  }, [apiKey, t]);
  useEffect(() => { void load(); }, [load]);

  const overview = status?.team ?? null;
  const members = overview?.members ?? [];
  const seatLimit = overview?.seatLimit ?? team.seatLimit;
  const freeSeats = overview?.freeSeats ?? 0;
  const role = overview?.role ?? team.role;
  const isOwner = overview?.isOwner ?? team.isOwner;
  const dashboardUrl = `https://frame-train.com/${language === 'de' ? 'de' : 'en'}/dashboard`;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="ft-h1">{t('team.title')}</h1>
          <p className="text-gray-400 mt-1">
            {overview?.name ?? team.name} · {t('team.seatCount', { used: members.length || 1, limit: seatLimit })}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button onClick={() => { void load(); }} disabled={loading} className="ft-icon-btn" aria-label={t('team.refresh')} title={t('team.refresh')}>
            {loading ? <Loader2 className="w-[17px] h-[17px] animate-spin" /> : <RefreshCw className="w-[17px] h-[17px]" strokeWidth={1.8} />}
          </button>
          {isOwner && (
            <button onClick={() => { void openUrl(dashboardUrl).catch(() => {}); }}
              className="px-4 py-2 rounded-xl text-sm font-medium text-white flex items-center gap-2 border border-white/10 hover:bg-white/5">
              <ExternalLink className="w-4 h-4" />{t('team.manage')}
            </button>
          )}
        </div>
      </div>

      <div role="tablist" aria-label={t('team.title')} className="flex gap-5 border-b border-white/10">
        {(['catalog', 'members', 'activity'] as const).map(id => (
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
            className={`pb-2 -mb-px text-sm border-b-2 transition-colors ${tab === id ? 'text-white border-current' : 'text-gray-400 border-transparent hover:text-white'}`}>
            {t(`team.tabs.${id}`)}
          </button>
        ))}
      </div>

      {/* Katalog bleibt geladen, damit Abgleich und Kopien weiterlaufen */}
      <div hidden={tab !== 'catalog'}>
        <TeamCatalog apiKey={apiKey} team={team} onActivity={setActivity} />
      </div>

      {tab === 'activity' && <ActivityList items={activity} />}

      {tab === 'members' && (<>
      <p className="ft-intro">{t(`team.roleIntro.${role}`)}</p>

      {error && (
        <div role="alert" className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">{error}</div>
      )}

      <section aria-label={t('team.membersLabel')} className="rounded-xl border border-white/10">
        {members.length === 0 && loading && (
          <div className="px-4 py-6 text-sm text-gray-400 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" />{t('team.loading')}</div>
        )}
        {members.map((m, i) => (
          <MemberRow key={m.id} member={m} first={i === 0} />
        ))}
        {members.length > 0 && freeSeats > 0 && (
          <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-white/10 text-sm text-gray-400">
            <span className="flex items-center gap-2"><UserPlus className="w-4 h-4" />{t('team.freeSeats', { n: freeSeats })}</span>
            {isOwner && (
              <button onClick={() => { void openUrl(dashboardUrl).catch(() => {}); }} className="text-gray-300 hover:text-white underline-offset-2 hover:underline">
                {t('team.addInDashboard')}
              </button>
            )}
          </div>
        )}
      </section>

      </>)}
    </div>
  );
}

function ActivityList({ items }: { items: CatalogActivity[] }) {
  const { t } = useLanguage();
  if (!items.length) return <p className="text-sm text-gray-500">{t('team.activity.empty')}</p>;
  return (
    <ul className="rounded-xl border border-white/10">
      {items.map((a, i) => {
        const seen = lastSeenBucket(a.createdAt);
        const when = seen.key === 'never' ? '' : seen.key === 'now' ? t('team.activity.justNow') : t(`team.seen.${seen.key}`, { n: seen.n });
        return (
          <li key={a.id} className={`flex items-baseline justify-between gap-4 px-4 py-2.5 text-sm ${i ? 'border-t border-white/10' : ''}`}>
            <span className="min-w-0 text-gray-300">
              {t(`team.activity.${a.action}`, { name: a.actorName, title: a.title })}
            </span>
            <span className="flex-shrink-0 text-xs text-gray-500">{when}</span>
          </li>
        );
      })}
    </ul>
  );
}

function MemberRow({ member, first }: { member: TeamMember; first: boolean }) {
  const { t } = useLanguage();
  const seen = lastSeenBucket(member.lastSeenAt);
  const seenText = seen.key === 'never' ? t('team.seen.never') : t(`team.seen.${seen.key}`, { n: seen.n });
  return (
    <div className={`grid grid-cols-[32px_minmax(0,1fr)_auto] sm:grid-cols-[32px_minmax(0,1fr)_110px_140px] items-center gap-3 px-4 py-3 ${first ? '' : 'border-t border-white/10'} ${member.isActive ? '' : 'opacity-60'}`}>
      <span aria-hidden="true" className="w-8 h-8 rounded-full flex items-center justify-center text-[12px] font-semibold text-white" style={{ background: member.isYou ? 'var(--ft-sel)' : 'var(--ft-hover)' }}>
        {initials(member.name)}
      </span>
      <div className="min-w-0">
        <div className="text-sm text-white truncate">
          {member.name}
          {member.isYou && <span className="text-gray-500"> · {t('team.you')}</span>}
          {member.isOwner && <span className="text-gray-500"> · {t('team.owner')}</span>}
        </div>
        {member.deviceName && (
          <div className="text-xs text-gray-500 flex items-center gap-1 truncate"><MonitorSmartphone className="w-3 h-3 flex-shrink-0" />{member.deviceName}</div>
        )}
      </div>
      <span className="hidden sm:block text-sm text-gray-400">{t(`team.roles.${member.role}`)}</span>
      <span className="text-sm text-gray-400 text-right">{member.isActive ? seenText : t('team.locked')}</span>
    </div>
  );
}
