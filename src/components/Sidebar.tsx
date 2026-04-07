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
  FlaskConical
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface SidebarProps {
  currentView: string;
  onViewChange: (view: any) => void;
  userEmail: string;
  onLogout: () => void;
}

export default function Sidebar({ currentView, onViewChange, userEmail, onLogout }: SidebarProps) {
  const { currentTheme } = useTheme();
  
  const menuItems = [
    { id: 'models', label: 'Modelle', icon: Layers },
    { id: 'dataset', label: 'Datensätze', icon: Upload },
    { id: 'training', label: 'Training', icon: Play },
    { id: 'analysis', label: 'Analyse', icon: BarChart3 },
    { id: 'tests', label: 'Tests', icon: FlaskConical },
    { id: 'versions', label: 'Versionen', icon: GitBranch },
  ];

  return (
    <div className="w-64 bg-black/20 backdrop-blur-lg border-r border-white/10 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <h1 className="text-2xl font-bold text-white">FrameTrain</h1>
        <p className="text-gray-400 text-sm mt-1">Local ML Training</p>
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
          <span className="font-medium">Einstellungen</span>
        </button>

        {/* Logout Button */}
        <button
          onClick={onLogout}
          className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-red-500/20 hover:text-red-300 transition-all"
        >
          <LogOut className="w-5 h-5" />
          <span className="font-medium">Abmelden</span>
        </button>
      </div>
    </div>
  );
}
