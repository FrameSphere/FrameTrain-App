import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import GlobalTrainingProgress from './components/GlobalTrainingProgress';
import LoadingScreen from './components/LoadingScreen';
import FirstLaunchSetup from './components/FirstLaunchSetup';
import { UpdateChecker } from './components/UpdateChecker';
import { ThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';
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

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [isFirstLaunch, setIsFirstLaunch] = useState<boolean | null>(null);
  const [showFirstLaunch, setShowFirstLaunch] = useState(false);

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
          <FirstLaunchSetup onComplete={handleFirstLaunchComplete} />
        </NotificationProvider>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <NotificationProvider>
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
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;
