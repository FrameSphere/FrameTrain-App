import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import GlobalTrainingProgress from './components/GlobalTrainingProgress';
import LoadingScreen from './components/LoadingScreen';
import FirstLaunchSetup from './components/FirstLaunchSetup';
import { UpdateChecker } from './components/UpdateChecker';
import { TauriAutoUpdater } from './components/TauriAutoUpdater';
import { ThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { AISettingsProvider } from './contexts/AISettingsContext';
import { PageContextProvider } from './contexts/PageContext';
import { TrainingContextProvider } from './contexts/TrainingContext';
import { LanguageProvider } from './contexts/LanguageContext';
import TermsGate from './components/TermsGate';
import { applyAccountDiagnosticsConsent } from './utils/diagnosticsConsent';
import ApprovalDialog from './mcp/ApprovalDialog';
import { setMcpUser, startMcpBridge } from './mcp/bridge';
import { setCurrentUser } from './utils/currentUser';
import { runAutoShare } from './team/teamApi';
import { isTeamInfo, parseSeatLoginError, seatEnded, TEAM_HEARTBEAT_MS, type SeatEndReason, type TeamInfo, type TeamStatus } from './team/team';
import './App.css';

interface ApiKeyValidation {
  user_id: string;
  email: string;
  is_valid: boolean;
}

interface LoginSession {
  user_id: string;
  email: string;
  logged_in_at: string;
  /** null = Server ohne das Feld (aeltere Version) */
  terms_accepted: boolean | null;
  /** Frueher zugestimmt, aber einer aelteren Fassung */
  terms_updated: boolean | null;
  diagnostics_consent: boolean | null;
  /** Business: Team und Rolle; null = Einzelkonto */
  team?: unknown;
  member_name?: string | null;
}

/** Einwilligungen aus dem Konto uebernehmen; Rueckgabe: ob und warum der AGB-Dialog kommt. */
async function applyAccountConsents(apiKey: string, session: LoginSession): Promise<null | 'missing' | 'updated'> {
  await applyAccountDiagnosticsConsent(apiKey, session.diagnostics_consent);
  if (session.terms_accepted !== false) return null;
  return session.terms_updated ? 'updated' : 'missing';
}

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
  /** Business: Team und eigene Rolle; null = Einzelkonto */
  team: TeamInfo | null;
  /** Name der Person auf einem Business-Platz (sonst null) */
  memberName: string | null;
}

/** Login-Antwort in die Nutzerdaten der App uebernehmen. */
function toUserData(apiKey: string, password: string, session: LoginSession): UserData {
  return {
    apiKey,
    password,
    userId: session.user_id,
    email: session.email,
    team: isTeamInfo(session.team) ? session.team : null,
    memberName: session.member_name ?? null,
  };
}

// ============ Close Confirmation Dialog ============

interface CloseDialogProps {
  isTraining: boolean;
  /** Hosting hat Modelle geladen: statt Beenden im Hintergrund weiterlaufen. */
  hostingActive?: boolean;
  onBackground?: () => void;
  onConfirm: () => void;
  onCancel: () => void;
}

function CloseConfirmDialog({ isTraining, hostingActive = false, onBackground, onConfirm, onCancel }: CloseDialogProps) {
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

          {hostingActive && !isTraining && onBackground && (
            <div className="mb-4 space-y-2">
              <p className="text-gray-400 text-xs leading-relaxed">
                Es sind Modelle gehostet. Im Hintergrund bleiben Schnell-Chat, Tastenkürzel, Menüleisten-Symbol und API erreichbar.
              </p>
              <button
                onClick={onBackground}
                className="w-full py-2.5 px-4 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 rounded-xl text-emerald-300 text-sm font-medium transition-all"
              >
                Im Hintergrund weiterlaufen
              </button>
            </div>
          )}

          {isTraining && (
            <div className="mb-5 px-4 py-3 bg-amber-500/10 border border-amber-500/30 rounded-xl">
              <p className="text-amber-300 text-xs">
                Tipp: Stoppe das Training zuerst über den „Training stoppen“-Button, um den Fortschritt zu sichern.
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
  // Konto ohne Zustimmung zur aktuellen Fassung von AGB & Datenschutz
  const [needsTerms, setNeedsTerms] = useState<null | 'missing' | 'updated'>(null);
  // Business: Anmeldung beim Start brauchte eine Entscheidung (Platz uebernehmen,
  // Passwort festlegen) — die Anmeldemaske zeigt sie mit den gespeicherten Daten.
  const [resumeLogin, setResumeLogin] = useState<{ apiKey: string; password: string; error: string } | null>(null);
  // Business: warum die App abgemeldet hat (Platz gesperrt, anderes Geraet ...)
  const [logoutNotice, setLogoutNotice] = useState<SeatEndReason | null>(null);
  const [isFirstLaunch, setIsFirstLaunch] = useState<boolean | null>(null);
  const [showFirstLaunch, setShowFirstLaunch] = useState(false);

  // KI-Clients (MCP): Werkzeug-Manifest melden und Aufrufe annehmen.
  // Ohne die Rust-Seite laeuft die App unveraendert weiter (Aufrufe abgesichert).
  useEffect(() => { void startMcpBridge(); }, []);
  useEffect(() => { setMcpUser(isAuthenticated ? userData?.userId ?? null : null); }, [isAuthenticated, userData?.userId]);

  // Close-Dialog State
  const [showCloseDialog, setShowCloseDialog] = useState(false);
  const [isTrainingActive, setIsTrainingActive] = useState(false);
  const [hostingActive, setHostingActive] = useState(false);

  // App-Close abfangen
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let disposed = false;

    // listen() rejects if window.__TAURI_INTERNALS__ isn't ready yet (e.g. Vite HMR
    // reloading this module while a previous listen() call is still in flight), and
    // that rejection previously had no .catch(), surfacing as an unhandled promise
    // rejection ("Cannot read properties of undefined (reading 'transformCallback')").
    // Same root cause already handled in contexts/TrainingContext.tsx — swallow it here too.
    listen('app-close-requested', async () => {
      // Prüfe ob Training aktiv ist
      try {
        const currentJob = await invoke<{ status: string } | null>('get_current_training');
        const training = currentJob?.status === 'running' || currentJob?.status === 'pending';
        setIsTrainingActive(training);
      } catch {
        setIsTrainingActive(false);
      }
      try {
        const st = await invoke<{ any_loaded: boolean }>('hosting_desktop_status');
        setHostingActive(!!st?.any_loaded);
      } catch {
        setHostingActive(false);
      }
      setShowCloseDialog(true);
    }).then(fn => {
      if (disposed) {
        try { fn(); } catch { /* already unregistered */ }
      } else {
        unlisten = fn;
      }
    }).catch(() => { /* listen() itself failed — nothing to unlisten */ });

    return () => {
      disposed = true;
      if (unlisten) { try { unlisten(); } catch { /* already unregistered */ } }
    };
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

  const handleBackground = async () => {
    setShowCloseDialog(false);
    try { await invoke('hosting_hide_main'); } catch { /* ignore */ }
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
        let session: LoginSession;
        try {
          session = await invoke<LoginSession>('login_user', {
            apiKey: config.api_key,
            password: config.password,
          });
        } catch (error) {
          // Business-Platz auf einem anderen Geraet / Passwort fehlt: nicht
          // vergessen, sondern in der Anmeldemaske nachfragen.
          const seat = parseSeatLoginError(error);
          if (seat && seat.code !== 'seat_locked') {
            setResumeLogin({ apiKey: config.api_key, password: config.password, error: String(error) });
            return;
          }
          throw error;
        }
        
        setCurrentUser(session.user_id);
        setUserData(toUserData(config.api_key, config.password, session));
        setNeedsTerms(await applyAccountConsents(config.api_key, session));
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

  const handleLogin = async (apiKey: string, password: string, takeover = false) => {
    try {
      // Login user and set current user in database
      const session = await invoke<LoginSession>('login_user', {
        apiKey,
        password,
        takeover,
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

      setCurrentUser(session.user_id);
      setUserData(toUserData(apiKey, password, session));
      setResumeLogin(null);
      setLogoutNotice(null);
      setNeedsTerms(await applyAccountConsents(apiKey, session));
      setIsAuthenticated(true);
      
      console.log('✅ User logged in and session established:', session.email);
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  // Business: Lebenszeichen des Platzes. Gesperrt, auf einem anderen Geraet
  // uebernommen oder Zugang abgelaufen -> abmelden und Grund zeigen.
  // Netzwerkfehler zaehlen nicht (offline weiterarbeiten bleibt moeglich).
  const teamId = userData?.team?.id ?? null;
  const teamRef = useRef<TeamInfo | null>(null);
  teamRef.current = userData?.team ?? null;
  const heartbeatKey = userData?.apiKey ?? null;
  useEffect(() => {
    if (!isAuthenticated || !teamId || !heartbeatKey) return;
    let stopped = false;
    const beat = async () => {
      const status = await invoke<TeamStatus>('team_overview', { apiKey: heartbeatKey }).catch(() => null);
      // Stufe 2: fertige Trainingslaeufe nebenbei im Team-Katalog zeigen
      if (status?.active && teamRef.current) void runAutoShare(heartbeatKey, teamRef.current).catch(() => 0);
      const reason = seatEnded(status);
      if (reason && !stopped) {
        stopped = true;
        await handleLogout();
        setLogoutNotice(reason);
      }
    };
    const id = window.setInterval(() => { void beat(); }, TEAM_HEARTBEAT_MS);
    return () => { stopped = true; window.clearInterval(id); };
    // handleLogout ist pro Render neu, haengt aber nur an userData
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, teamId, heartbeatKey]);

  const handleLogout = async () => {
    try {
      // Clear user session in database
      await invoke('logout_user');
      
      // Lösche gespeicherte Config
      await invoke('clear_config');

      // FIX: localStorage beim Logout leeren (nur Auth-bezogene Daten, nicht User-Daten wie Sessions)
      if (userData) {
        localStorage.removeItem(`ft_ai_settings_${userData.userId}`);
        localStorage.removeItem(`ft_lab_sessions_${userData.userId}`);
        // NOTE: synapse_sessions_${userId} NICHT löschen - Sessions sind User-Daten und müssen persistent sein
      }
      // Legacy-Keys ohne userId ebenfalls entfernen
      localStorage.removeItem('ft_ai_settings');
      
      console.log('✅ User logged out');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsAuthenticated(false);
      setCurrentUser(null);
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
      <LanguageProvider>
        <NotificationProvider>
          <AISettingsProvider userId={userData?.userId}>
            <PageContextProvider>
              <TrainingContextProvider>
                <FirstLaunchSetup onComplete={handleFirstLaunchComplete} />
              </TrainingContextProvider>
            </PageContextProvider>
          </AISettingsProvider>
        </NotificationProvider>
      </LanguageProvider>
    </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <LanguageProvider>
        <NotificationProvider>
          <AISettingsProvider userId={userData?.userId}>
            <PageContextProvider>
              <TrainingContextProvider>
                <div className="app">
              {isAuthenticated && userData ? (
                <>
                  <Dashboard 
                    key={userData.userId}
                    userData={userData}
                    onLogout={handleLogout} 
                  />
                  <UpdateChecker />
                  <TauriAutoUpdater />
                  {needsTerms && (
                    <TermsGate
                      apiKey={userData.apiKey}
                      reason={needsTerms}
                      onAccepted={() => setNeedsTerms(null)}
                      onDecline={() => { setNeedsTerms(null); void handleLogout(); }}
                    />
                  )}
                </>
              ) : (
                <Login
                  onLogin={handleLogin}
                  resume={resumeLogin}
                  notice={logoutNotice}
                />
              )}
            </div>

              {/* Freigabe fuer Werkzeug-Aufrufe von KI-Clients (MCP) */}
              <ApprovalDialog />

              {/* App-Close-Dialog — über allem */}
              {showCloseDialog && (
                <CloseConfirmDialog
                  isTraining={isTrainingActive}
                  hostingActive={hostingActive}
                  onBackground={handleBackground}
                  onConfirm={handleConfirmClose}
                  onCancel={handleCancelClose}
                />
              )}
            </TrainingContextProvider>
          </PageContextProvider>
        </AISettingsProvider>
      </NotificationProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}

export default App;
