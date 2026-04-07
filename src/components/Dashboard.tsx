import { useState } from 'react';
import Sidebar from './Sidebar';
import ModelManager from './ModelManager';
import TrainingPanel from './TrainingPanel';
import DatasetUpload from './DatasetUpload';
import AnalysisPanel from './AnalysisPanel';
import TestPanel from './TestPanel';
import VersionManager from './VersionManager';
import Settings from './Settings';
import { useTheme } from '../contexts/ThemeContext';

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

interface DashboardProps {
  userData: UserData;
  onLogout: () => void;
}

type View = 'models' | 'training' | 'dataset' | 'analysis' | 'tests' | 'versions' | 'settings';

export default function Dashboard({ userData, onLogout }: DashboardProps) {
  const [currentView, setCurrentView] = useState<View>('models');
  const { currentTheme } = useTheme();

  const renderView = () => {
    switch (currentView) {
      case 'models':
        return <ModelManager />;
      case 'training':
        return <TrainingPanel />;
      case 'dataset':
        return <DatasetUpload />;
      case 'analysis':
        return <AnalysisPanel />;
      case 'tests':
        return <TestPanel />;
      case 'versions':
        return <VersionManager />;
      case 'settings':
        return <Settings userData={userData} onLogout={onLogout} />;
      default:
        return <ModelManager />;
    }
  };

  return (
    <div className={`flex h-screen bg-gradient-to-br ${currentTheme.colors.background}`}>
      <Sidebar 
        currentView={currentView} 
        onViewChange={setCurrentView}
        userEmail={userData.email}
        onLogout={onLogout}
      />
      
      <main className="flex-1 overflow-auto p-8">
        <div className="max-w-7xl mx-auto">
          {renderView()}
        </div>
      </main>
    </div>
  );
}
