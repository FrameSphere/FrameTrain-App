import { useState } from 'react';
import { Key, Lock, AlertCircle, Eye, EyeOff, ExternalLink, Info } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface LoginProps {
  onLogin: (apiKey: string, password: string) => Promise<void>;
}

export default function Login({ onLogin }: LoginProps) {
  const [apiKey, setApiKey] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [needsDesktopPassword, setNeedsDesktopPassword] = useState(false);
  const { currentTheme } = useTheme();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setNeedsDesktopPassword(false);

    if (!apiKey.startsWith('ft_') || apiKey.length < 24) {
      setError('API-Key muss mit „ft_" beginnen und mindestens 24 Zeichen lang sein');
      return;
    }
    if (!password || password.length < 6) {
      setError('Passwort muss mindestens 6 Zeichen lang sein');
      return;
    }

    setLoading(true);
    try {
      await onLogin(apiKey, password);
    } catch (err: any) {
      const msg: string = err?.message || err || 'Anmeldung fehlgeschlagen';
      // Hinweis auf fehlendes Desktop-Passwort hervorheben
      if (msg.includes('Kein Desktop-Passwort') || msg.includes('desktop-password') || msg.includes('needsDesktopPassword')) {
        setNeedsDesktopPassword(true);
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const Spinner = () => (
    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );

  return (
    <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br ${currentTheme.colors.background} p-4`}>
      <div className="w-full max-w-md">

        {/* Logo & Header */}
        <div className="text-center mb-8">
          <div
            className="inline-flex items-center justify-center mb-4 rounded-[18px] shadow-2xl"
            style={{
              background: 'linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%)',
              width: 72,
              height: 72,
              boxShadow: '0 0 32px rgba(168,85,247,0.45), 0 8px 32px rgba(0,0,0,0.4)',
            }}
          >
            <span style={{ fontFamily: 'Arial, sans-serif', fontSize: 40, fontWeight: 900, color: 'white', lineHeight: 1, userSelect: 'none' }}>
              F
            </span>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">FrameTrain</h1>
          <p className="text-gray-400">Lokales ML-Training auf deinem Desktop</p>
        </div>

        {/* Login Card */}
        <form onSubmit={handleSubmit} className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 shadow-2xl">
          <h2 className="text-2xl font-bold text-white mb-6">Anmelden</h2>

          {/* API-Key */}
          <div className="mb-5">
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
                onChange={e => { setApiKey(e.target.value); setError(''); setNeedsDesktopPassword(false); }}
                placeholder="ft_xxxxxxxxxxxxxxxx"
                disabled={loading}
                className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              />
            </div>
            <p className="mt-1 text-xs text-gray-400">Dein API-Key aus dem FrameTrain Dashboard</p>
          </div>

          {/* Passwort */}
          <div className="mb-5">
            <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
              Passwort
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={e => { setPassword(e.target.value); setError(''); setNeedsDesktopPassword(false); }}
                placeholder="Dein Passwort"
                disabled={loading}
                className="w-full pl-10 pr-12 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
              />
              <button
                type="button"
                onClick={() => setShowPassword(p => !p)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white transition-colors"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            {/* Hover-Tooltip */}
            <div className="mt-1 flex items-center gap-1 group relative w-fit">
              <Info className="h-3.5 w-3.5 text-gray-500 cursor-help" />
              <span className="text-xs text-gray-500 cursor-help">Was muss ich hier eingeben?</span>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-10 w-72">
                <div className="bg-gray-900 border border-white/10 rounded-lg p-3 shadow-xl text-xs text-gray-300 leading-relaxed">
                  <p className="font-semibold text-white mb-1">Welches Passwort?</p>
                  <p><span className="text-purple-400">E-Mail-Account:</span> Dein normales FrameTrain-Passwort.</p>
                  <p className="mt-1"><span className="text-blue-400">Google / GitHub:</span> Das Desktop-Passwort, das du im Dashboard unter „Desktop-App Passwort“ gesetzt hast.</p>
                </div>
                <div className="w-2.5 h-2.5 bg-gray-900 border-r border-b border-white/10 rotate-45 ml-3 -mt-1.5" />
              </div>
            </div>
          </div>

          {/* Fehlermeldung */}
          {error && (
            <div className="mb-4 flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm text-red-300">{error}</p>
                {needsDesktopPassword && (
                  <a
                    href="https://frame-train.vercel.app/dashboard"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 mt-1.5 text-xs text-purple-400 hover:text-purple-300 underline"
                  >
                    <ExternalLink className="w-3 h-3" />
                    Zum Dashboard → Desktop-App Passwort setzen
                  </a>
                )}
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !apiKey || !password}
            className={`w-full py-3 px-4 bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold rounded-lg hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
          >
            {loading ? <><Spinner />Authentifiziere...</> : 'Anmelden'}
          </button>
        </form>

        <p className="mt-8 text-center text-sm text-gray-500">
          Sichere Authentifizierung über FrameTrain-Server
        </p>
      </div>
    </div>
  );
}
