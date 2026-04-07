import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Key, Lock, AlertCircle } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface LoginProps {
  onLogin: (apiKey: string, password: string) => Promise<void>;
}

export default function Login({ onLogin }: LoginProps) {
  const [apiKey, setApiKey] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState('');
  const { currentTheme } = useTheme();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setDebugInfo('');
    setLoading(true);

    try {
      // Validiere Format
      if (!apiKey.startsWith('ft_')) {
        throw new Error('API-Key muss mit "ft_" beginnen');
      }

      if (apiKey.length < 24) {
        throw new Error('API-Key ist zu kurz');
      }

      if (!password || password.length < 6) {
        throw new Error('Passwort muss mindestens 6 Zeichen lang sein');
      }

      console.log('🔐 Attempting login...');
      console.log('API Key length:', apiKey.length);
      console.log('Password length:', password.length);

      await onLogin(apiKey, password);
      
      console.log('✅ Login successful!');
    } catch (err: any) {
      console.error('❌ Login error:', err);
      
      let errorMessage = err.message || 'Login fehlgeschlagen';
      let debugMessage = '';

      // Detaillierte Error-Informationen
      if (typeof err === 'string') {
        errorMessage = err;
      } else if (err.message) {
        errorMessage = err.message;
        debugMessage = `Type: ${typeof err}, Full: ${JSON.stringify(err)}`;
      }

      setError(errorMessage);
      setDebugInfo(debugMessage);
      
      console.log('Error details:', {
        message: errorMessage,
        type: typeof err,
        full: err
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br ${currentTheme.colors.background} p-4`}>
      <div className="w-full max-w-md">
        {/* Logo & Header */}
        <div className="text-center mb-8">
          <div className={`inline-block p-4 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-2xl mb-4`}>
            <svg
              className="w-12 h-12 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
              />
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">FrameTrain</h1>
          <p className="text-gray-400">Lokales ML-Training auf deinem Desktop</p>
        </div>

        {/* Login Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 shadow-2xl">
          <h2 className="text-2xl font-bold text-white mb-6">Anmelden</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* API Key Input */}
            <div>
              <label htmlFor="apiKey" className="block text-sm font-medium text-gray-300 mb-2">
                API-Key
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Key className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="apiKey"
                  type="text"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="ft_xxxxxxxxxxxxxxxx"
                  disabled={loading}
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                  required
                />
              </div>
              <p className="mt-1 text-xs text-gray-400">
                Dein API-Key aus dem FrameTrain Dashboard
              </p>
            </div>

            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Passwort
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Dein Passwort"
                  disabled={loading}
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
                  required
                />
              </div>
              <p className="mt-1 text-xs text-gray-400">
                Das Passwort deines FrameTrain-Accounts
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="space-y-2">
                <div className="flex items-start space-x-2 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-red-300">{error}</p>
                    {debugInfo && (
                      <p className="text-xs text-red-400 mt-1 font-mono">{debugInfo}</p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !apiKey || !password}
              className={`w-full py-3 px-4 bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold rounded-lg hover:opacity-90 focus:outline-none focus:ring-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]`}
              style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Authentifiziere...
                </span>
              ) : (
                'Anmelden'
              )}
            </button>
          </form>

          {/* Help Text */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-400">
              Noch kein Account?{' '}
              <a
                href="https://frametrain.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium transition-colors"
                style={{ color: currentTheme.colors.accent }}
              >
                Registriere dich hier
              </a>
            </p>
          </div>

        </div>

        {/* Footer */}
        <p className="mt-8 text-center text-sm text-gray-500">
          Sichere Authentifizierung mit verschlüsselter Verbindung
        </p>
      </div>
    </div>
  );
}
