import { 
  LayoutDashboard, 
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
    <div className="w-64 bg-black/20 backdrop-blur-lg border-r border-white/10 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <h1 className="text-2xl font-bold text-white">FrameTrain</h1>
        <p className="text-gray-400 text-sm mt-1">{t('sidebar.tagline')}</p>
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
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                isActive
                  ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow-lg`
                  : 'text-gray-300 hover:bg-white/5 hover:text-white'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom Section */}
      <div className="p-4 border-t border-white/10 space-y-2">
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
          className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
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
