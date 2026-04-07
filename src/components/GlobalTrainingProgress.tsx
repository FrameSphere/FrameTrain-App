import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { X, TrendingUp, Clock, CheckCircle } from 'lucide-react';

interface TrainingProgress {
  training_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  progress_percentage: number;
  train_loss: number;
  train_accuracy: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  learning_rate: number;
  elapsed_time_seconds: number;
  estimated_time_remaining_seconds: number | null;
  last_updated: string;
}

export default function GlobalTrainingProgress() {
  const [activeTrainings, setActiveTrainings] = useState<TrainingProgress[]>([]);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Prüfe alle 2 Sekunden nach aktiven Trainings
    const interval = setInterval(checkActiveTrainings, 2000);
    
    // Initial check
    checkActiveTrainings();
    
    return () => clearInterval(interval);
  }, []);

  const checkActiveTrainings = async () => {
    try {
      // Hole alle aktiven Trainings vom Backend
      const trainings = await invoke<TrainingProgress[]>('list_active_trainings');
      
      setActiveTrainings(trainings);
      setIsVisible(trainings.length > 0);
      
    } catch (error) {
      // Noch keine aktiven Trainings oder Fehler
      setActiveTrainings([]);
      setIsVisible(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  if (!isVisible || activeTrainings.length === 0) {
    return null;
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 pointer-events-none">
      <div className="container mx-auto px-4 pb-4 pointer-events-auto">
        <div className="space-y-2">
          {activeTrainings.map((training) => (
            <div
              key={training.training_id}
              className="bg-gradient-to-r from-purple-900/95 to-pink-900/95 backdrop-blur-lg border-2 border-purple-500/50 rounded-xl shadow-2xl overflow-hidden"
            >
              <div className="p-4">
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <h3 className="text-white font-bold text-lg">
                      Training läuft
                    </h3>
                    <span className="text-sm text-purple-200">
                      Epoch {training.current_epoch}/{training.total_epochs}
                    </span>
                  </div>
                  <button
                    onClick={() => setIsVisible(false)}
                    className="p-1 hover:bg-white/10 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-white" />
                  </button>
                </div>

                {/* Progress Bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-sm text-purple-200 mb-1">
                    <span>Fortschritt</span>
                    <span className="font-semibold">{training.progress_percentage.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-green-400 to-emerald-500 h-full transition-all duration-500 relative overflow-hidden"
                      style={{ width: `${training.progress_percentage}%` }}
                    >
                      {/* Animated shine effect */}
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
                    </div>
                  </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-white/5 rounded-lg p-2">
                    <div className="flex items-center space-x-1 mb-1">
                      <TrendingUp className="w-3 h-3 text-blue-400" />
                      <span className="text-xs text-gray-300">Loss</span>
                    </div>
                    <p className="text-sm font-bold text-white">
                      {training.train_loss.toFixed(4)}
                    </p>
                  </div>

                  {training.train_accuracy !== null && (
                    <div className="bg-white/5 rounded-lg p-2">
                      <div className="flex items-center space-x-1 mb-1">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        <span className="text-xs text-gray-300">Accuracy</span>
                      </div>
                      <p className="text-sm font-bold text-white">
                        {(training.train_accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}

                  <div className="bg-white/5 rounded-lg p-2">
                    <div className="flex items-center space-x-1 mb-1">
                      <Clock className="w-3 h-3 text-yellow-400" />
                      <span className="text-xs text-gray-300">Zeit</span>
                    </div>
                    <p className="text-sm font-bold text-white">
                      {formatTime(training.elapsed_time_seconds)}
                    </p>
                  </div>

                  {training.estimated_time_remaining_seconds !== null && (
                    <div className="bg-white/5 rounded-lg p-2">
                      <div className="flex items-center space-x-1 mb-1">
                        <Clock className="w-3 h-3 text-purple-400" />
                        <span className="text-xs text-gray-300">Verbleibend</span>
                      </div>
                      <p className="text-sm font-bold text-white">
                        {formatTime(training.estimated_time_remaining_seconds)}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Shimmer animation keyframes */}
      <style>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </div>
  );
}
