import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import Sidebar from './Sidebar';
import ModelManager from './ModelManager';
import TrainingPanel from './TrainingPanel';
import DatasetUpload from './DatasetUpload';
import AnalysisPanel from './AnalysisPanel';
import TestPanel from './TestPanel';
import VersionManager from './VersionManager';
import LaboratoryPanel from './LaboratoryPanel';
import Settings from './Settings';
import FloatingAICoach from './FloatingAICoach';
import GlobalTrainingProgress from './GlobalTrainingProgress';
import TrainingDashboard from './TrainingDashboard';
import { useTheme } from '../contexts/ThemeContext';
import { useTrainingContext } from '../contexts/TrainingContext';

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

type View = 'models' | 'training' | 'dataset' | 'analysis' | 'tests' | 'versions' | 'settings' | 'laboratory';

export default function Dashboard({ userData, onLogout }: DashboardProps) {
  const [currentView, setCurrentView] = useState<View>('models');
  const [initialAnalysisVersionId, setInitialAnalysisVersionId] = useState<string | null>(null);
  const { currentTheme } = useTheme();

  const {
    state: trainingState,
    setShowDashboard,
    setIsDashMinimized,
    setCurrentJob,
    setCompletedVersionId,
    clearTraining,
  } = useTrainingContext();

  const renderView = () => {
    switch (currentView) {
      case 'models':
        return <ModelManager />;
      case 'training':
        return (
          <TrainingPanel
            onNavigateToAnalysis={(versionId) => {
              setInitialAnalysisVersionId(versionId);
              setCurrentView('analysis');
            }}
          />
        );
      case 'dataset':
        return <DatasetUpload />;
      case 'analysis':
        return <AnalysisPanel initialVersionId={initialAnalysisVersionId} />;
      case 'tests':
        return <TestPanel />;
      case 'laboratory':
        return <LaboratoryPanel />;
      case 'versions':
        return <VersionManager />;
      case 'settings':
        return <Settings userData={userData} onLogout={onLogout} />;
      default:
        return <ModelManager />;
    }
  };

  const handleStopFromGlobal = async () => {
    try { await invoke('stop_training'); } catch { /* ignore */ }
  };

  const handleNavigateToAnalysis = (versionId: string) => {
    setInitialAnalysisVersionId(versionId);
    setCurrentView('analysis');
    setShowDashboard(false);
    setIsDashMinimized(false);
    setCurrentJob(null);
    setCompletedVersionId(null);
    clearTraining();
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

      {/* Floating AI Coach */}
      <FloatingAICoach />

      {/* Globales Mini-Training-Widget — sichtbar auf JEDER Seite solange minimiert */}
      {trainingState.isDashMinimized && (
        <GlobalTrainingProgress
          onNavigateToTraining={() => {
            setShowDashboard(true);
            setIsDashMinimized(false);
          }}
        />
      )}

      {/* Globales TrainingDashboard — overlay über dem aktuellen Inhalt, egal auf welcher Seite */}
      {trainingState.showDashboard && !trainingState.isDashMinimized && (
        <TrainingDashboard
          isOpen={true}
          isMinimized={false}
          onMinimize={() => {
            setIsDashMinimized(true);
          }}
          onMaximize={() => {
            setIsDashMinimized(false);
          }}
          onClose={() => {
            setShowDashboard(false);
            setIsDashMinimized(false);
            setCurrentJob(null);
            setCompletedVersionId(null);
            clearTraining();
          }}
          mode={trainingState.mode}
          modelName={trainingState.modelName}
          datasetName={trainingState.datasetName}
          config={trainingState.config as any}
          job={trainingState.currentJob}
          lossPoints={trainingState.lossPoints}
          sessionId={trainingState.sessionId}
          startedAt={trainingState.dashStartedAt}
          onStop={handleStopFromGlobal}
          completedVersionId={trainingState.completedVersionId}
          onNavigateToAnalysis={handleNavigateToAnalysis}
        />
      )}
    </div>
  );
}
