import { 
  LayoutDashboard, 
  Home,
  Play, 
  Upload, 
  BarChart3, 
  GitBranch, 
  LogOut,
  Layers,
  Settings as SettingsIcon,
  User,
  FlaskConical,
  Microscope,
  Zap,
  Network,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';

interface SidebarProps {
  currentView: string;
  onViewChange: (view: any) => void;
  userEmail: string;
  onLogout: () => void;
}

export default function Sidebar({ currentView, onViewChange, userEmail, onLogout }: SidebarProps) {
  const { currentTheme } = useTheme();
  const { t } = useLanguage();

  // Wenn der Synapse Builder aktiv ist, übernimmt die Sidebar dessen dunkles
  // Design (Navy + Violet/Indigo-Akzente) mit sanftem Übergang. Verlässt man
  // den Synapse Builder wieder, geht es zurück zur normalen Theme-Farbe.
  const isSynapse = currentView === 'synapse';
  const synapseActiveGradient = 'linear-gradient(to right, #6366f1, #a78bfa)';

  const menuItems = [
    { id: 'models',     label: t('sidebar.nav.models'),    icon: Layers },
    { id: 'dataset',    label: t('sidebar.nav.datasets'),  icon: Upload },
    { id: 'training',   label: t('sidebar.nav.training'),  icon: Play },
    { id: 'analysis',   label: t('sidebar.nav.analysis'),  icon: BarChart3 },
    { id: 'tests',      label: t('sidebar.nav.tests'),     icon: FlaskConical },
    { id: 'laboratory', label: t('sidebar.nav.laboratory'),icon: Microscope },
    { id: 'synapse',    label: t('sidebar.nav.synapse'),   icon: Network },
    { id: 'versions',   label: t('sidebar.nav.versions'),  icon: GitBranch },
  ];

  return (
    <div
      className="w-64 backdrop-blur-lg flex flex-col transition-colors duration-500 ease-in-out"
      style={{
        backgroundColor: isSynapse ? '#0a0e17' : 'rgba(0, 0, 0, 0.2)',
        borderRight: `1px solid ${isSynapse ? '#1e293b' : 'rgba(255, 255, 255, 0.1)'}`,
      }}
    >
      {/* Header */}
      <div
        className="p-6 transition-colors duration-500 ease-in-out"
        style={{ borderBottom: `1px solid ${isSynapse ? '#1e293b' : 'rgba(255, 255, 255, 0.1)'}` }}
      >
        <div className="flex items-center gap-3">
          {/* Home-Button — Rücksprung zur Startseite von jeder Ansicht aus.
              Zeigt den Aktiv-Zustand, damit "ich bin auf Start" sichtbar bleibt. */}
          <button
            onClick={() => onViewChange('home')}
            title={t('sidebar.home')}
            aria-label={t('sidebar.home')}
            aria-current={currentView === 'home' ? 'page' : undefined}
            className={`p-2 rounded-xl flex-shrink-0 transition-all duration-500 ease-in-out ${
              currentView === 'home'
                ? isSynapse
                  ? 'text-white shadow-lg'
                  : `bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow-lg`
                : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white border border-white/10'
            }`}
            style={currentView === 'home' && isSynapse ? { backgroundImage: synapseActiveGradient } : undefined}
          >
            <Home className="w-5 h-5" />
          </button>
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-white leading-tight">FrameTrain</h1>
            <p className="text-gray-400 text-sm">{t('sidebar.tagline')}</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-500 ease-in-out ${
                isActive
                  ? isSynapse
                    ? 'text-white shadow-lg'
                    : `bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow-lg`
                  : 'text-gray-300 hover:bg-white/5 hover:text-white'
              }`}
              style={isActive && isSynapse ? { backgroundImage: synapseActiveGradient } : undefined}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium flex items-center gap-1.5">
                {item.label}
                {(item.id === 'synapse' || item.id === 'laboratory') && (
                  <span className={`text-[8px] font-bold px-1 py-0.5 rounded tracking-wide leading-none ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  }`}>BETA</span>
                )}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Bottom Section */}
      <div
        className="p-4 space-y-2 transition-colors duration-500 ease-in-out"
        style={{ borderTop: `1px solid ${isSynapse ? '#1e293b' : 'rgba(255, 255, 255, 0.1)'}` }}
      >
        {/* User Info */}
        <div className="px-4 py-2 mb-2">
          <div className="flex items-center space-x-2 text-gray-400">
            <User className="w-4 h-4" />
            <span className="text-sm truncate">{userEmail}</span>
          </div>
        </div>

        {/* Settings Button */}
        <button
          onClick={() => onViewChange('settings')}
          className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-500 ease-in-out ${
            currentView === 'settings'
              ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow-lg`
              : 'text-gray-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <SettingsIcon className="w-5 h-5" />
          <span className="font-medium">{t('sidebar.settings')}</span>
        </button>

        {/* Logout Button */}
        <button
          onClick={onLogout}
          className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-red-500/20 hover:text-red-300 transition-all"
        >
          <LogOut className="w-5 h-5" />
          <span className="font-medium">{t('sidebar.logout')}</span>
        </button>
      </div>
    </div>
  );
}
