import {
  Home,
  Play,
  Database,
  BarChart3,
  GitBranch,
  LogOut,
  Layers,
  Settings as SettingsIcon,
  FlaskConical,
  Microscope,
  Network,
  Server,
  type LucideIcon,
} from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { useTrainingContext } from '../contexts/TrainingContext';
import BrandMark from './ui/BrandMark';

interface SidebarProps {
  currentView: string;
  onViewChange: (view: any) => void;
  userEmail: string;
  onLogout: () => void;
}

/** macOS: Fenster mit Overlay-Titelleiste, die Ampel liegt oben in der Seitenleiste. */
export const IS_MAC = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent);

interface NavItem { id: string; label: string; icon: LucideIcon; beta?: boolean }

export default function Sidebar({ currentView, onViewChange, userEmail, onLogout }: SidebarProps) {
  const { t } = useLanguage();
  const { state: training } = useTrainingContext();

  // Laufendes Training steht direkt am Eintrag, in der Ergebnis-Farbe
  const job = training.currentJob;
  const livePct = job && (job.status === 'running' || job.status === 'pending')
    ? Math.round(job.progress?.progress_percent ?? 0)
    : null;

  // Gruppiert nach dem Weg eines Modells: vorbereiten, trainieren, pruefen, nutzen
  const groups: { title?: string; items: NavItem[] }[] = [
    { items: [
      { id: 'home',     label: t('sidebar.home'),            icon: Home },
      { id: 'models',   label: t('sidebar.nav.models'),      icon: Layers },
      { id: 'versions', label: t('sidebar.nav.versions'),    icon: GitBranch },
    ] },
    { title: t('sidebar.groups.prepare'), items: [
      { id: 'dataset',  label: t('sidebar.nav.datasets'),    icon: Database },
    ] },
    { title: t('sidebar.groups.train'), items: [
      { id: 'training', label: t('sidebar.nav.training'),    icon: Play },
      { id: 'analysis', label: t('sidebar.nav.analysis'),    icon: BarChart3 },
    ] },
    { title: t('sidebar.groups.check'), items: [
      { id: 'tests',      label: t('sidebar.nav.tests'),      icon: FlaskConical },
      { id: 'laboratory', label: t('sidebar.nav.laboratory'), icon: Microscope, beta: true },
    ] },
    { title: t('sidebar.groups.use'), items: [
      { id: 'hosting',  label: t('sidebar.nav.hosting'),     icon: Server },
      { id: 'synapse',  label: t('sidebar.nav.synapse'),     icon: Network, beta: true },
    ] },
  ];

  const initial = (userEmail.trim()[0] ?? '?').toUpperCase();

  return (
    <aside className="ft-glass ft-sidebar flex-none w-[244px] rounded-[18px] flex flex-col min-h-0">
      {/* Kopf: Ampel-Platz (macOS) und Marke. Ziehbar wie eine Titelleiste. */}
      <div data-tauri-drag-region className={`flex items-center gap-2.5 px-4 ${IS_MAC ? 'pt-[50px]' : 'pt-4'} pb-3`}>
        <BrandMark size={26} />
        <div data-tauri-drag-region className="min-w-0 leading-tight">
          <div data-tauri-drag-region className="text-[15px] font-semibold tracking-[-0.02em] text-white">FrameTrain</div>
          <div data-tauri-drag-region className="text-[11.5px] text-gray-500">{t('sidebar.tagline')}</div>
        </div>
      </div>

      <nav aria-label="FrameTrain" className="flex-1 min-h-0 overflow-y-auto px-2.5 pb-2">
        {groups.map((group, gi) => (
          <div key={gi} className="flex flex-col gap-px">
            {group.title && <div className="ft-nav-group">{group.title}</div>}
            {group.items.map(item => {
              const Icon = item.icon;
              const active = currentView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onViewChange(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`ft-nav ${active ? 'is-active' : ''}`}
                >
                  <Icon className="ft-nav-icon" strokeWidth={1.8} />
                  <span className="truncate">{item.label}</span>
                  {item.id === 'training' && livePct !== null ? (
                    <span className="ml-auto font-mono text-[11px] tabular-nums" style={{ color: 'var(--ft-out-text)' }}>
                      {livePct}%
                    </span>
                  ) : item.beta ? (
                    <span className="ml-auto font-mono text-[9.5px] tracking-[0.08em] text-gray-500">BETA</span>
                  ) : null}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Fuss: Konto, Einstellungen, Abmelden */}
      <div className="flex items-center gap-2 px-3 py-3 border-t" style={{ borderColor: 'var(--ft-line)' }}>
        <span
          className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-semibold text-white flex-none"
          style={{ background: 'var(--ft-sel)' }}
          aria-hidden="true"
        >
          {initial}
        </span>
        <span className="flex-1 min-w-0 truncate text-[12.5px] text-gray-400" title={userEmail}>{userEmail}</span>
        <button
          onClick={() => onViewChange('settings')}
          aria-label={t('sidebar.settings')}
          title={t('sidebar.settings')}
          aria-current={currentView === 'settings' ? 'page' : undefined}
          className={`ft-icon-btn ${currentView === 'settings' ? 'is-active' : ''}`}
        >
          <SettingsIcon className="w-[17px] h-[17px]" strokeWidth={1.8} />
        </button>
        <button
          onClick={onLogout}
          aria-label={t('sidebar.logout')}
          title={t('sidebar.logout')}
          className="ft-icon-btn ft-icon-btn-danger"
        >
          <LogOut className="w-[17px] h-[17px]" strokeWidth={1.8} />
        </button>
      </div>
    </aside>
  );
}
