import { createContext, useContext, useState, ReactNode, useCallback } from 'react';

export interface LossPoint {
  step: number;
  epoch: number;
  train_loss: number;
  val_loss?: number;
}

export interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  train_loss: number;
  val_loss: number | null;
  learning_rate: number;
  progress_percent: number;
}

export interface TrainingJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  progress: TrainingProgress;
  error: string | null;
}

export interface TrainingState {
  // Minimized dashboard state
  isDashMinimized: boolean;
  showDashboard: boolean;
  currentJob: TrainingJob | null;
  lossPoints: LossPoint[];
  sessionId: string;
  dashStartedAt: number;
  completedVersionId: string | null;
  
  // Training info
  mode: 'standard' | 'dev';
  modelName: string;
  datasetName: string;
  config: unknown | null;
}

interface TrainingContextType {
  state: TrainingState;
  
  // Minimize/Maximize
  setIsDashMinimized: (minimized: boolean) => void;
  setShowDashboard: (show: boolean) => void;
  
  // Training management
  setCurrentJob: (job: TrainingJob | null) => void;
  updateJobStatus: (status: TrainingJob['status'], error?: string) => void;
  addLossPoint: (point: LossPoint) => void;
  setLossPoints: (points: LossPoint[]) => void;
  
  // Session management
  setSessionId: (id: string) => void;
  setDashStartedAt: (time: number) => void;
  setCompletedVersionId: (id: string | null) => void;
  
  // Training info
  setTrainingInfo: (mode: 'standard' | 'dev', modelName: string, datasetName: string) => void;
  setTrainingConfig: (config: unknown | null) => void;
  
  // Clear all
  clearTraining: () => void;
}

const TrainingContext = createContext<TrainingContextType | undefined>(undefined);

const defaultState: TrainingState = {
  isDashMinimized: false,
  showDashboard: false,
  currentJob: null,
  lossPoints: [],
  sessionId: '',
  dashStartedAt: 0,
  completedVersionId: null,
  mode: 'standard',
  modelName: '',
  datasetName: '',
  config: null,
};

export function TrainingContextProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<TrainingState>(defaultState);

  const setIsDashMinimized = useCallback((minimized: boolean) => {
    setState(s => ({ ...s, isDashMinimized: minimized }));
  }, []);

  const setShowDashboard = useCallback((show: boolean) => {
    setState(s => ({ ...s, showDashboard: show }));
  }, []);

  const setCurrentJob = useCallback((job: TrainingJob | null) => {
    setState(s => ({ ...s, currentJob: job }));
  }, []);

  const updateJobStatus = useCallback((status: TrainingJob['status'], error?: string) => {
    setState(s => ({
      ...s,
      currentJob: s.currentJob
        ? { ...s.currentJob, status, error: error ?? s.currentJob.error ?? null }
        : null,
    }));
  }, []);

  const addLossPoint = useCallback((point: LossPoint) => {
    setState(s => ({ ...s, lossPoints: [...s.lossPoints, point] }));
  }, []);

  const setLossPoints = useCallback((points: LossPoint[]) => {
    setState(s => ({ ...s, lossPoints: points }));
  }, []);

  const setSessionId = useCallback((id: string) => {
    setState(s => ({ ...s, sessionId: id }));
  }, []);

  const setCompletedVersionId = useCallback((id: string | null) => {
    setState(s => ({ ...s, completedVersionId: id }));
  }, []);

  const setDashStartedAt = useCallback((time: number) => {
    setState(s => ({ ...s, dashStartedAt: time }));
  }, []);

  const setTrainingInfo = useCallback((mode: 'standard' | 'dev', modelName: string, datasetName: string) => {
    setState(s => ({ ...s, mode, modelName, datasetName }));
  }, []);

  const setTrainingConfig = useCallback((config: unknown | null) => {
    setState(s => ({ ...s, config }));
  }, []);

  const clearTraining = useCallback(() => {
    setState(defaultState);
  }, []);

  const value: TrainingContextType = {
    state,
    setIsDashMinimized,
    setShowDashboard,
    setCurrentJob,
    updateJobStatus,
    addLossPoint,
    setLossPoints,
    setSessionId,
    setDashStartedAt,
    setCompletedVersionId,
    setTrainingInfo,
    setTrainingConfig,
    clearTraining,
  };

  return (
    <TrainingContext.Provider value={value}>
      {children}
    </TrainingContext.Provider>
  );
}

export function useTrainingContext() {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error('useTrainingContext must be used within TrainingContextProvider');
  }
  return context;
}
