import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';

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

  // ── Globale Event-Listener ────────────────────────────────────────────────
  // Der Provider ist IMMER gemountet — damit bleiben Dashboard-Overlay und
  // Mini-Widget aktuell, auch wenn das TrainingPanel (Trainings-Seite) beim
  // Seitenwechsel unmountet wurde. Vorher froren Status/Progress dann ein und
  // Stop/Complete/Error kamen nie im UI an.
  // Dev-Trainings (job_id "dev_…") verwaltet das DevTrainPanel selbst.
  useEffect(() => {
    const isDevJob = (jobId?: string) => jobId?.startsWith('dev_') ?? false;
    const unlisteners: Array<() => void> = [];
    let disposed = false;
    const add = (p: Promise<() => void>) =>
      p.then((fn) => { if (disposed) fn(); else unlisteners.push(fn); });

    add(listen<{ job_id?: string; data?: TrainingProgress }>('training-progress', (e) => {
      if (isDevJob(e.payload.job_id)) return;
      const d = e.payload.data;
      if (!d) return;
      setState((s) => {
        if (!s.currentJob) return s;
        const next: TrainingState = {
          ...s,
          currentJob: { ...s.currentJob, status: 'running', progress: d },
        };
        if (d.train_loss != null) {
          next.lossPoints = [
            ...s.lossPoints,
            { step: d.step, epoch: d.epoch, train_loss: d.train_loss, val_loss: d.val_loss ?? undefined },
          ];
        }
        return next;
      });
    }));

    add(listen<{ job_id?: string; new_version_id?: string }>('training-complete', (e) => {
      if (isDevJob(e.payload.job_id)) return;
      setState((s) => ({
        ...s,
        currentJob: s.currentJob ? { ...s.currentJob, status: 'completed' } : s.currentJob,
        completedVersionId: e.payload.new_version_id ?? s.completedVersionId,
      }));
    }));

    add(listen<{ job_id?: string; data?: { error?: string; details?: string } }>('training-error', (e) => {
      if (isDevJob(e.payload.job_id)) return;
      const err = e.payload.data?.error ?? 'Unbekannter Fehler';
      const det = e.payload.data?.details;
      setState((s) => ({
        ...s,
        currentJob: s.currentJob
          ? { ...s.currentJob, status: 'failed', error: det ? `${err}\n${det}` : err }
          : s.currentJob,
      }));
    }));

    add(listen<{ job_id?: string; version_id?: string }>('training-stopped-with-checkpoint', (e) => {
      if (isDevJob(e.payload.job_id)) return;
      setState((s) => ({
        ...s,
        currentJob: s.currentJob ? { ...s.currentJob, status: 'stopped' } : s.currentJob,
        completedVersionId: e.payload.version_id ?? s.completedVersionId,
      }));
    }));

    // Stop OHNE Checkpoint — hatte vorher app-weit KEINEN Listener:
    // das Dashboard blieb nach Stop für immer auf "Training läuft".
    add(listen<{ job_id?: string }>('training-stopped', (e) => {
      if (isDevJob(e.payload.job_id)) return;
      setState((s) => ({
        ...s,
        currentJob: s.currentJob ? { ...s.currentJob, status: 'stopped' } : s.currentJob,
      }));
    }));

    return () => {
      disposed = true;
      unlisteners.forEach((u) => u());
    };
  }, []);

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
