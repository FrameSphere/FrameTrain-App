import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import GlobalTrainingProgress from './components/GlobalTrainingProgress';
import LoadingScreen from './components/LoadingScreen';
import FirstLaunchSetup from './components/FirstLaunchSetup';
import { UpdateChecker } from './components/UpdateChecker';
import { ThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { AISettingsProvider } from './contexts/AISettingsContext';
import { PageContextProvider } from './contexts/PageContext';
import './App.css';

interface ApiKeyValidation {
  user_id: string;
  email: string;
  is_valid: boolean;
}

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

// ============ Close Confirmation Dialog ============

interface CloseDialogProps {
  isTraining: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

function CloseConfirmDialog({ isTraining, onConfirm, onCancel }: CloseDialogProps) {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}>
      <div className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-sm mx-4 overflow-hidden">
        {/* Roter Warnstreifen oben */}
        <div className="h-1 bg-gradient-to-r from-red-500 to-orange-500" />
        <div className="p-6">
          <div className="flex items-start gap-4 mb-5">
            {/* Icon */}
            <div className="w-10 h-10 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
            </div>
            <div>
              <h2 className="text-white font-semibold text-lg leading-tight">
                {isTraining ? 'Training läuft noch' : 'App schließen?'}
              </h2>
              <p className="text-gray-400 text-sm mt-1.5 leading-relaxed">
                {isTraining
                  ? 'Ein Training ist noch aktiv. Wenn du die App jetzt schließt, wird das Training abgebrochen und der Fortschritt geht verloren.'
                  : 'Möchtest du FrameTrain wirklich schließen?'}
              </p>
            </div>
          </div>

          {isTraining && (
            <div className="mb-5 px-4 py-3 bg-amber-500/10 border border-amber-500/30 rounded-xl">
              <p className="text-amber-300 text-xs">
                ⚠️ Tipp: Stoppe das Training zuerst über den „Training stoppen“-Button, um den Fortschritt zu sichern.
              </p>
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 py-2.5 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white text-sm font-medium transition-all"
            >
              Abbrechen
            </button>
            <button
              onClick={onConfirm}
              className="flex-1 py-2.5 px-4 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-xl text-red-300 text-sm font-medium transition-all"
            >
              {isTraining ? 'Training abbrechen & schließen' : 'Schließen'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [isFirstLaunch, setIsFirstLaunch] = useState<boolean | null>(null);
  const [showFirstLaunch, setShowFirstLaunch] = useState(false);

  // Close-Dialog State
  const [showCloseDialog, setShowCloseDialog] = useState(false);
  const [isTrainingActive, setIsTrainingActive] = useState(false);

  // App-Close abfangen
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    listen('app-close-requested', async () => {
      // Prüfe ob Training aktiv ist
      try {
        const currentJob = await invoke<{ status: string } | null>('get_current_training');
        const training = currentJob?.status === 'running' || currentJob?.status === 'pending';
        setIsTrainingActive(training);
      } catch {
        setIsTrainingActive(false);
      }
      setShowCloseDialog(true);
    }).then(fn => { unlisten = fn; });

    return () => { if (unlisten) unlisten(); };
  }, []);

  const handleConfirmClose = async () => {
    // Training stoppen falls aktiv
    if (isTrainingActive) {
      try { await invoke('stop_training'); } catch { /* ignore */ }
    }
    // Sleep-Prevention deaktivieren
    try { await invoke('disable_prevent_sleep'); } catch { /* ignore */ }
    // App schließen
    await invoke('force_quit_app');
  };

  const handleCancelClose = () => {
    setShowCloseDialog(false);
  };

  useEffect(() => {
    checkFirstLaunch();
  }, []);

  const checkFirstLaunch = async () => {
    try {
      const firstLaunch = await invoke<boolean>('check_first_launch');
      console.log('[App] First launch check:', firstLaunch);
      setIsFirstLaunch(firstLaunch);
      setShowFirstLaunch(firstLaunch);
      
      // If not first launch, proceed with auth check
      if (!firstLaunch) {
        await checkAuth();
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error('[App] Failed to check first launch:', error);
      // On error, assume not first launch and proceed
      setIsFirstLaunch(false);
      await checkAuth();
    }
  };

  const checkAuth = async () => {
    try {
      // Versuche gespeicherte Credentials zu laden
      const savedConfig = await invoke<string>('load_config');
      const config = JSON.parse(savedConfig);
      
      if (config.api_key && config.password) {
        // Login user and establish session
        const session = await invoke<{
          user_id: string;
          email: string;
          logged_in_at: string;
        }>('login_user', {
          apiKey: config.api_key,
          password: config.password
        });
        
        setUserData({
          apiKey: config.api_key,
          password: config.password,
          userId: session.user_id,
          email: session.email
        });
        setIsAuthenticated(true);
        
        console.log('✅ Auto-login successful:', session.email);
      }
    } catch (error) {
      console.log('Keine gültige Authentifizierung gefunden:', error);
      // Config löschen falls vorhanden aber ungültig
      try {
        await invoke('clear_config');
      } catch (e) {
        // Ignore
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (apiKey: string, password: string) => {
    try {
      // Login user and set current user in database
      const session = await invoke<{
        user_id: string;
        email: string;
        logged_in_at: string;
      }>('login_user', {
        apiKey,
        password
      });

      // Speichere Credentials
      const config = {
        api_key: apiKey,
        password: password
      };
      
      await invoke('save_config', { 
        apiKey: apiKey,  // Benötigt für backward compatibility
        config: JSON.stringify(config)
      });

      setUserData({
        apiKey,
        password,
        userId: session.user_id,
        email: session.email
      });
      setIsAuthenticated(true);
      
      console.log('✅ User logged in and session established:', session.email);
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const handleLogout = async () => {
    try {
      // Clear user session in database
      await invoke('logout_user');
      
      // Lösche gespeicherte Config
      await invoke('clear_config');
      
      console.log('✅ User logged out');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsAuthenticated(false);
      setUserData(null);
    }
  };

  const handleFirstLaunchComplete = () => {
    console.log('[App] First launch complete, checking auth...');
    setShowFirstLaunch(false);
    setIsFirstLaunch(false);
    checkAuth();
  };

  if (loading) {
    return <LoadingScreen />;
  }

  // Show first launch setup if needed
  if (isFirstLaunch === null) {
    return <LoadingScreen />;
  }

  if (showFirstLaunch) {
    return (
      <ThemeProvider>
        <NotificationProvider>
          <AISettingsProvider>
            <PageContextProvider>
              <FirstLaunchSetup onComplete={handleFirstLaunchComplete} />
            </PageContextProvider>
          </AISettingsProvider>
        </NotificationProvider>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <NotificationProvider>
        <AISettingsProvider>
          <PageContextProvider>
            <div className="app">
            {isAuthenticated && userData ? (
              <>
                <Dashboard 
                  userData={userData}
                  onLogout={handleLogout} 
                />
                <GlobalTrainingProgress />
                <UpdateChecker />
              </>
            ) : (
              <Login onLogin={handleLogin} />
            )}
          </div>

            {/* App-Close-Dialog — über allem */}
            {showCloseDialog && (
              <CloseConfirmDialog
                isTraining={isTrainingActive}
                onConfirm={handleConfirmClose}
                onCancel={handleCancelClose}
              />
            )}
          </PageContextProvider>
        </AISettingsProvider>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;
