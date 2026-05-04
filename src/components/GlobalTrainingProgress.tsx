import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, CheckCircle, AlertCircle, Square, Maximize2, TrendingDown } from 'lucide-react';
import { useTrainingContext } from '../contexts/TrainingContext';

interface ActiveTraining {
  training_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  progress_percentage: number;
  train_loss: number;
  val_loss: number | null;
  learning_rate: number;
  elapsed_time_seconds: number;
  estimated_time_remaining_seconds: number | null;
}

interface GlobalTrainingProgressProps {
  onNavigateToTraining?: () => void;
}

export default function GlobalTrainingProgress({ onNavigateToTraining }: GlobalTrainingProgressProps) {
  const { state: trainingState } = useTrainingContext();
  const [training, setTraining] = useState<ActiveTraining | null>(null);
  const [nowTick, setNowTick] = useState(() => Date.now());

  useEffect(() => {
    const hasSomethingToShow = (trainingState.isDashMinimized && trainingState.currentJob) || training;
    if (!hasSomethingToShow) return;
    const id = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(id);
  }, [training, trainingState.currentJob, trainingState.isDashMinimized]);

  const contextTraining = useMemo<ActiveTraining | null>(() => {
    if (!trainingState.currentJob || !trainingState.isDashMinimized) return null;
    const p = trainingState.currentJob.progress;
    return {
      training_id: trainingState.currentJob.id,
      status: trainingState.currentJob.status,
      current_epoch: p.epoch,
      total_epochs: p.total_epochs,
      current_step: p.step,
      total_steps: p.total_steps,
      progress_percentage: p.progress_percent,
      train_loss: p.train_loss,
      val_loss: p.val_loss ?? null,
      learning_rate: p.learning_rate,
      elapsed_time_seconds: Math.max(0, Math.floor((nowTick - (trainingState.dashStartedAt || nowTick)) / 1000)),
      estimated_time_remaining_seconds: null,
    };
  }, [trainingState.currentJob, trainingState.dashStartedAt, trainingState.isDashMinimized, nowTick]);

  useEffect(() => {
    // Wenn wir die Daten aus dem Context haben (normaler Flow), kein Backend-Polling nötig.
    if (contextTraining) return;

    const check = async () => {
      try {
        const list = await invoke<ActiveTraining[]>('list_active_trainings');
        setTraining(list.length > 0 ? list[0] : null);
      } catch {
        setTraining(null);
      }
    };
    check();
    const id = setInterval(check, 2000);
    return () => clearInterval(id);
  }, [contextTraining]);

  const effectiveTraining = contextTraining ?? training;
  if (!effectiveTraining) return null;

  const isRunning   = effectiveTraining.status === 'running' || effectiveTraining.status === 'pending';
  const isCompleted = effectiveTraining.status === 'completed';
  const isFailed    = effectiveTraining.status === 'failed';
  const isStopped   = effectiveTraining.status === 'stopped';

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    if (h > 0) return `${h}h ${m % 60}m`;
    if (m > 0) return `${m}m ${s % 60}s`;
    return `${s}s`;
  };

  return (
    <div
      className="fixed bottom-5 right-5 z-50 flex items-center gap-3 px-4 py-3 rounded-2xl bg-slate-900 border border-white/10 shadow-2xl cursor-pointer hover:bg-slate-800 transition-all group"
      onClick={onNavigateToTraining}
      title="Zum Training"
    >
      {isRunning   && <Loader2     className="w-4 h-4 text-emerald-400 animate-spin flex-shrink-0" />}
      {isCompleted && <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />}
      {isFailed    && <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />}
      {isStopped   && <Square      className="w-4 h-4 text-gray-400 flex-shrink-0" />}

      <div className="min-w-0">
        <p className="text-white text-xs font-semibold">
          {isRunning   ? 'Training läuft…'
           : isCompleted ? 'Training abgeschlossen ✓'
           : isFailed    ? 'Training fehlgeschlagen'
           : 'Training gestoppt'}
        </p>
        <p className="text-gray-500 text-[10px]">
          E{effectiveTraining.current_epoch}/{effectiveTraining.total_epochs}
          {' · '}
          <TrendingDown className="w-2.5 h-2.5 inline text-emerald-400" />{' '}
          {effectiveTraining.train_loss?.toFixed(4)}
          {' · '}
          {formatTime(effectiveTraining.elapsed_time_seconds)}
        </p>
      </div>

      {/* Mini-Progressbar */}
      <div className="w-20 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-teal-500 transition-all duration-500"
          style={{ width: `${Math.min(effectiveTraining.progress_percentage, 100)}%` }}
        />
      </div>

      <Maximize2 className="w-3.5 h-3.5 text-gray-500 group-hover:text-white transition-all flex-shrink-0" />
    </div>
  );
}
