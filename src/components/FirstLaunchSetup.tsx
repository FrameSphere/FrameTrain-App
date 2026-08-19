import React, { useState, useEffect } from 'react';
import { invoke } from "@tauri-apps/api/core";
import { listen } from '@tauri-apps/api/event';
import { Check, Download, Package, Clock, HardDrive, Loader2, AlertCircle, XCircle, Globe, ShieldCheck, Cpu, Database, Palette, Brain, ScanEye } from 'lucide-react';
import { useTheme, type Theme, type ThemeId } from '../contexts/ThemeContext';
import { useLanguage, LANGUAGE_META, type Language } from '../contexts/LanguageContext';
import AIAssistantSettingsPanel from './AIAssistantSettingsPanel';

interface PluginInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  built_in: boolean;
  required_packages: string[];
  estimated_size_mb: number;
  install_time_minutes: number;
  is_selected?: boolean;
  is_installed?: boolean;
}

/** Symbol je Plugin — lucide statt Emoji (das Backend liefert nur eine Kategorie). */
function pluginIcon(plugin: PluginInfo, className: string) {
  return plugin.category === 'Vision'
    ? <ScanEye className={className} />
    : <Brain className={className} />;
}

/** Ohne diesen Stack kann die App nichts trainieren — er bleibt gesetzt. */
const REQUIRED_PLUGIN_ID = 'seq_classification';

interface InstallProgress {
  plugin_id: string;
  status: string;
  message: string;
  progress?: number;
}

interface DependencyStatus {
  package: string;
  installed: boolean;
  version?: string;
}

interface GpuInfo {
  has_nvidia_gpu: boolean;
  cuda_available: boolean;
  cuda_version?: string;
  gpu_name?: string;
  recommended_torch_index: string;
}

interface PreFlightCheck {
  ok: boolean;
  python_found: boolean;
  python_version?: string;
  python_version_ok: boolean;
  pip_found: boolean;
  free_gb: number;
  free_gb_ok: boolean;
  gpu_info: GpuInfo;
  platform: string;
  errors: string[];
  warnings: string[];
}

const FirstLaunchSetup: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const { currentTheme, setTheme, themes: allThemes } = useTheme();
  const { language, setLanguage, t } = useLanguage();

  // Screen 0: Sprachauswahl — kommt vor Python-Check
  const [languageSelected, setLanguageSelected] = useState(false);

  const handleLanguageConfirm = (lang: Language) => {
    setLanguage(lang);
    setLanguageSelected(true);
  };

  // Screen 0.5: Design-Auswahl — direkt nach der Sprachauswahl
  const [themeSelected, setThemeSelected] = useState(false);

  // Screen 0.75: KI-Einstellungen — direkt nach der Design-Auswahl (überspringbar)
  const [aiSetupDone, setAiSetupDone] = useState(false);

  // Screen 1: Pre-Flight-Check
  const [preFlightDone, setPreFlightDone] = useState(false);
  const [preFlightResult, setPreFlightResult] = useState<PreFlightCheck | null>(null);
  const [preFlightLoading, setPreFlightLoading] = useState(false);

  const runPreFlight = async () => {
    setPreFlightLoading(true);
    try {
      const result = await invoke<PreFlightCheck>('run_preflight_check');
      setPreFlightResult(result);
    } catch (e) {
      setPreFlightResult({
        ok: false, python_found: false, python_version_ok: false,
        pip_found: false, free_gb: 0, free_gb_ok: false,
        platform: '', errors: [String(e)], warnings: [],
        gpu_info: { has_nvidia_gpu: false, cuda_available: false, recommended_torch_index: 'cpu' }
      });
    } finally {
      setPreFlightLoading(false);
    }
  };

  // Pre-Flight starten sobald Sprache, Design und KI-Setup abgeschlossen sind
  useEffect(() => {
    if (aiSetupDone && !preFlightResult) {
      runPreFlight();
    }
  }, [aiSetupDone]);
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [pluginsLoadError, setPluginsLoadError] = useState<string | null>(null);
  const [pluginsLoading, setPluginsLoading] = useState(false);
  const [selectedPlugins, setSelectedPlugins] = useState<Set<string>>(new Set([REQUIRED_PLUGIN_ID]));
  const [installing, setInstalling] = useState(false);
  const [installProgress, setInstallProgress] = useState<Map<string, InstallProgress>>(new Map());
  
  // Python setup state
  const [pythonSetupPhase, setPythonSetupPhase] = useState<'checking' | 'installing' | 'complete' | 'error'>('checking');
  const [dependencyStatus, setDependencyStatus] = useState<DependencyStatus[]>([]);
  const [pythonError, setPythonError] = useState<string>('');
  
  // Computed values — alle selected Plugins zählen (auch built_in haben Größe)
  const totalSize = Array.from(selectedPlugins)
    .map(id => plugins.find(p => p.id === id))
    .filter((p): p is PluginInfo => !!p)
    .reduce((sum, p) => sum + (p.estimated_size_mb || 0), 0);
  
  const totalTime = Array.from(selectedPlugins)
    .map(id => plugins.find(p => p.id === id))
    .filter((p): p is PluginInfo => !!p)
    .reduce((sum, p) => sum + (p.install_time_minutes || 0), 0);
  
  useEffect(() => {
    if (preFlightDone) {
      checkPythonDependencies();
    }
  }, [preFlightDone]);
  
  const checkPythonDependencies = async () => {
    console.log('[Setup] Checking Python dependencies...');
    try {
      const status = await invoke<DependencyStatus[]>('check_dependency_status');
      setDependencyStatus(status);
      
      const allInstalled = status.every(s => s.installed);
      if (allInstalled) {
        console.log('[Setup] ✅ All Python dependencies installed');
        setPythonSetupPhase('complete');
        await loadPlugins();
      } else {
        const missing = status.filter(s => !s.installed).map(s => s.package);
        console.log('[Setup] ⚠️ Missing Python packages:', missing);
        setPythonSetupPhase('installing');
        await installPythonDependencies();
      }
    } catch (error) {
      const errorMsg = String(error);
      console.error('[Setup] Error checking dependencies:', errorMsg);
      
      // Bessere Fehlermeldungen basierend auf dem Fehlertyp
      let userFriendlyError = errorMsg;
      if (errorMsg.includes('not installed') || errorMsg.includes('nicht installiert')) {
        userFriendlyError = t('firstLaunch.python.pythonNotInstalled');
      }
      
      setPythonSetupPhase('error');
      setPythonError(userFriendlyError);
    }
  };
  
  const installPythonDependencies = async () => {
    try {
      console.log('[Setup] Installing Python dependencies...');
      await invoke('install_plugins', { pluginIds: [] });
    } catch (error) {
      console.error('[Setup] Error installing Python dependencies:', error);
      setPythonSetupPhase('error');
      setPythonError(String(error));
    }
  };
  
  useEffect(() => {
    if (pythonSetupPhase !== 'installing') return;
    
    const setupListenersForPython = async () => {
      const unlistenProgress = await listen<InstallProgress>('plugin-install-progress', (event) => {
        const progress = event.payload;
        setInstallProgress(prev => new Map(prev).set(progress.plugin_id, progress));
      });
      
      const unlistenComplete = await listen('plugin-install-complete', async () => {
        console.log('[Setup] Python dependencies installed successfully');
        setPythonSetupPhase('complete');
        await loadPlugins();
      });
      
      return () => {
        unlistenProgress();
        unlistenComplete();
      };
    };
    
    setupListenersForPython();
  }, [pythonSetupPhase]);
  
  const loadPlugins = async () => {
    setPluginsLoading(true);
    setPluginsLoadError(null);
    try {
      const pluginList = await invoke<PluginInfo[]>('get_available_plugins');
      setPlugins(pluginList);
      
      // Vorauswahl kommt aus dem Backend (is_selected). Frueher war zusaetzlich
      // ein Phantom-Eintrag "text" gesetzt und jedes built_in-Plugin vorgewaehlt —
      // dadurch stand im Zaehler "3 Plugins" bei zwei sichtbaren, und YOLO liess
      // sich nicht abwaehlen.
      const preSelected = new Set<string>([REQUIRED_PLUGIN_ID]);
      pluginList.forEach(p => {
        if (p.is_selected) preSelected.add(p.id);
      });
      setSelectedPlugins(preSelected);
    } catch (error) {
      console.error('Failed to load plugins:', error);
      setPluginsLoadError(String(error));
    } finally {
      setPluginsLoading(false);
    }
  };
  
  const setupListeners = async (): Promise<() => void> => {
    const unlistenProgress = await listen<InstallProgress>('plugin-install-progress', (event) => {
      const progress = event.payload;
      setInstallProgress(prev => new Map(prev).set(progress.plugin_id, progress));
    });

    const unlistenComplete = await listen('plugin-install-complete', () => {
      setInstalling(false);
      onComplete();
    });

    return () => {
      unlistenProgress();
      unlistenComplete();
    };
  };
  
  // Listener werden NICHT mehr ans Laden der Plugins gekoppelt -- sie wuerden sonst
  // erst NACH dem install_plugins()-Aufruf registriert, und ein sehr schneller
  // Installationsdurchlauf (z.B. alle Pakete schon vorhanden) kann das
  // plugin-install-complete Event verpassen, bevor der Listener aktiv ist.
  // Stattdessen registrieren wir die Listener einmalig sobald die Plugin-Liste
  // geladen wurde, unabhaengig vom Installationsstart.
  useEffect(() => {
    if (pythonSetupPhase !== 'complete' || plugins.length === 0) return;
    let cleanup: (() => void) | undefined;
    setupListeners().then(fn => { cleanup = fn; });
    return () => { cleanup?.(); };
  }, [pythonSetupPhase, plugins.length]);
  
  const togglePlugin = (pluginId: string) => {
    if (pluginId === REQUIRED_PLUGIN_ID) return;
    
    setSelectedPlugins(prev => {
      const next = new Set(prev);
      if (next.has(pluginId)) {
        next.delete(pluginId);
      } else {
        next.add(pluginId);
      }
      return next;
    });
  };
  
  const startInstallation = async () => {
    setInstalling(true);
    setInstallProgress(new Map());
    
    try {
      await invoke('install_plugins', {
        pluginIds: Array.from(selectedPlugins)
      });
    } catch (error) {
      console.error('Installation failed:', error);
      setInstalling(false);
    }
  };
  
  const skipSetup = async () => {
    try {
      await invoke('install_plugins', { pluginIds: ['text'] });
      onComplete();
    } catch (error) {
      console.error('Skip setup failed:', error);
    }
  };
  
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'complete': return 'text-green-400';
      case 'failed': return 'text-red-400';
      default: return 'text-blue-400';
    }
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      text: 'from-purple-500 to-pink-500',
      vision: 'from-blue-500 to-cyan-500',
      audio: 'from-green-500 to-teal-500',
      tabular: 'from-orange-500 to-red-500',
      graph: 'from-indigo-500 to-purple-500',
      multimodal: 'from-pink-500 to-rose-500',
      rl: 'from-yellow-500 to-orange-500'
    };
    return colors[category] || 'from-gray-500 to-gray-600';
  };
  
  return (
    <div className={`h-screen flex items-center justify-center bg-gradient-to-br ${currentTheme.colors.background} p-6`}>
      <div className="w-full max-w-5xl h-[90vh] bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-2xl flex flex-col overflow-hidden">

        {/* Screen 0: Sprachauswahl */}
        {!languageSelected && (
          <LanguageSelectScreen
            currentLanguage={language}
            currentTheme={currentTheme}
            onConfirm={handleLanguageConfirm}
          />
        )}

        {/* Screen 0.5: Design-Auswahl */}
        {languageSelected && !themeSelected && (
          <ThemeSelectScreen
            currentTheme={currentTheme}
            allThemes={allThemes}
            onSelectTheme={setTheme}
            onContinue={() => setThemeSelected(true)}
          />
        )}

        {/* Screen 0.75: KI-Einstellungen (überspringbar) */}
        {languageSelected && themeSelected && !aiSetupDone && (
          <AISetupScreen
            currentTheme={currentTheme}
            onContinue={() => setAiSetupDone(true)}
            onSkip={() => setAiSetupDone(true)}
          />
        )}

        {/* Screen 1: Pre-Flight-Check */}
        {languageSelected && themeSelected && aiSetupDone && !preFlightDone && (
          <PreFlightScreen
            loading={preFlightLoading}
            result={preFlightResult}
            currentTheme={currentTheme}
            onContinue={() => setPreFlightDone(true)}
            onRetry={runPreFlight}
          />
        )}

        {/* Screen 2: Python Setup */}
        {languageSelected && themeSelected && aiSetupDone && preFlightDone && pythonSetupPhase !== 'complete' && (
          <PythonSetupScreen 
            phase={pythonSetupPhase}
            dependencyStatus={dependencyStatus}
            error={pythonError}
            currentTheme={currentTheme}
            installProgress={installProgress}
          />
        )}
        
        {/* Screen 3: Plugin-Auswahl */}
        {languageSelected && themeSelected && aiSetupDone && preFlightDone && pythonSetupPhase === 'complete' && (
          <PluginSelectionScreen
            plugins={plugins}
            pluginsLoading={pluginsLoading}
            pluginsLoadError={pluginsLoadError}
            onRetryLoadPlugins={loadPlugins}
            selectedPlugins={selectedPlugins}
            installing={installing}
            installProgress={installProgress}
            totalSize={totalSize}
            totalTime={totalTime}
            currentTheme={currentTheme}
            onTogglePlugin={togglePlugin}
            onStartInstallation={startInstallation}
            onSkipSetup={skipSetup}
            getCategoryColor={getCategoryColor}
            getStatusColor={getStatusColor}
          />
        )}
      </div>
    </div>
  );
};

// ============ Sub-Component: Pre-Flight Screen ============
interface PreFlightScreenProps {
  loading: boolean;
  result: PreFlightCheck | null;
  currentTheme: any;
  onContinue: () => void;
  onRetry: () => void;
}

const PreFlightScreen: React.FC<PreFlightScreenProps> = ({
  loading, result, currentTheme, onContinue, onRetry
}) => {
  const { t } = useLanguage();
  const hasErrors   = (result?.errors?.length ?? 0) > 0;
  const hasWarnings = (result?.warnings?.length ?? 0) > 0;

  const CheckRow: React.FC<{ label: string; ok: boolean; detail?: string }> = ({ label, ok, detail }) => (
    <div className={`flex items-start gap-3 p-3 rounded-lg border ${
      ok ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'
    }`}>
      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5 ${
        ok ? 'bg-green-500/30' : 'bg-red-500/30'
      }`}>
        {ok
          ? <Check className="w-3.5 h-3.5 text-green-400" strokeWidth={3} />
          : <XCircle className="w-3.5 h-3.5 text-red-400" />}
      </div>
      <div>
        <span className={`font-medium ${ ok ? 'text-green-300' : 'text-red-300'}`}>{label}</span>
        {detail && <p className="text-xs text-gray-400 mt-0.5">{detail}</p>}
      </div>
    </div>
  );

  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <ShieldCheck className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{t('firstLaunch.preflight.title')}</h1>
            <p className="text-gray-300">{t('firstLaunch.preflight.subtitle')}</p>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-y-auto">
        <div className="w-full max-w-xl">

          {/* Loading */}
          {loading && (
            <div className="text-center">
              <Loader2 className="w-14 h-14 text-blue-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-300">{t('firstLaunch.preflight.scanning')}</p>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="space-y-3">
              <CheckRow
                label={result.python_found
                  ? t('firstLaunch.preflight.pythonFound', { version: result.python_version ?? '' })
                  : t('firstLaunch.preflight.pythonMissing')}
                ok={result.python_found && result.python_version_ok}
                detail={!result.python_found
                  ? t('firstLaunch.preflight.pythonInstallHint')
                  : !result.python_version_ok
                  ? t('firstLaunch.preflight.pythonTooOld', { version: result.python_version ?? '' })
                  : undefined}
              />
              <CheckRow
                label={result.pip_found ? t('firstLaunch.preflight.pipFound') : t('firstLaunch.preflight.pipMissing')}
                ok={result.pip_found}
                detail={!result.pip_found ? t('firstLaunch.preflight.pipHint') : undefined}
              />
              <CheckRow
                label={result.free_gb > 0
                  ? t('firstLaunch.preflight.diskFree', { gb: result.free_gb.toFixed(1) })
                  : t('firstLaunch.preflight.diskUnknown')}
                ok={result.free_gb_ok || result.free_gb === 0}
                detail={!result.free_gb_ok && result.free_gb > 0
                  ? t('firstLaunch.preflight.diskHint')
                  : undefined}
              />
              <CheckRow
                label={result.gpu_info.has_nvidia_gpu
                  ? t('firstLaunch.preflight.gpuFound', {
                      name:  result.gpu_info.gpu_name ?? 'NVIDIA',
                      cuda:  result.gpu_info.cuda_version ?? '?',
                      index: result.gpu_info.recommended_torch_index,
                    })
                  : t('firstLaunch.preflight.gpuNone')}
                ok={true}
                detail={!result.gpu_info.has_nvidia_gpu
                  ? t('firstLaunch.preflight.gpuNoneHint')
                  : undefined}
              />

              {/* Errors */}
              {hasErrors && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <p className="text-red-300 font-semibold mb-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" /> {t('firstLaunch.preflight.errorsTitle')}
                  </p>
                  {result.errors.map((e, i) => (
                    <p key={i} className="text-red-300 text-sm mt-1">{e}</p>
                  ))}
                </div>
              )}

              {/* Warnings */}
              {hasWarnings && (
                <div className="mt-2 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-xl">
                  {result.warnings.map((w, i) => (
                    <p key={i} className="text-yellow-300 text-sm">{w}</p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
        <div className="flex justify-between items-center">
          <button
            onClick={onRetry}
            disabled={loading}
            className="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20 disabled:opacity-40"
          >
            {t('firstLaunch.preflight.recheck')}
          </button>
          <button
            onClick={onContinue}
            disabled={loading || hasErrors}
            className={`px-8 py-3 rounded-xl font-semibold text-white transition-all flex items-center gap-2 ${
              loading || hasErrors
                ? 'bg-gray-600 opacity-40 cursor-not-allowed'
                : `bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90 shadow-lg`
            }`}
          >
            {t('firstLaunch.preflight.continue')} →
          </button>
        </div>
      </div>
    </>
  );
};

// ============ Sub-Component: Language Select Screen ============
interface LanguageSelectScreenProps {
  currentLanguage: Language;
  currentTheme: any;
  onConfirm: (lang: Language) => void;
}

const LanguageSelectScreen: React.FC<LanguageSelectScreenProps> = ({
  currentLanguage,
  currentTheme,
  onConfirm,
}) => {
  const [selected, setSelected] = useState<Language>(currentLanguage);
  const { t } = useLanguage();

  const labels = {
    de: {
      headline: 'Willkommen bei FrameTrain',
      sub: 'Wähle deine Sprache, um zu beginnen.',
      btn: 'Weiter',
    },
    en: {
      headline: 'Welcome to FrameTrain',
      sub: 'Choose your language to get started.',
      btn: 'Continue',
    },
  };
  const ui = labels[selected];

  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <Globe className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{ui.headline}</h1>
            <p className="text-gray-300 mt-1">{ui.sub}</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="w-full max-w-sm space-y-4">
          {(Object.entries(LANGUAGE_META) as [Language, typeof LANGUAGE_META[Language]][]).map(
            ([lang, meta]) => {
              const active = selected === lang;
              return (
                <button
                  key={lang}
                  onClick={() => setSelected(lang)}
                  className={`w-full flex items-center gap-5 px-6 py-5 rounded-2xl border-2 transition-all duration-200 ${
                    active
                      ? `bg-gradient-to-r ${currentTheme.colors.gradient} border-transparent shadow-lg scale-[1.02]`
                      : 'bg-white/5 border-white/15 hover:bg-white/10 hover:border-white/30'
                  }`}
                >
                  <span className="text-4xl">{meta.flag}</span>
                  <div className="text-left">
                    <div className="text-white font-bold text-lg">{meta.nativeLabel}</div>
                    {active && (
                      <div className="text-white/70 text-sm">{t('firstLaunch.language.selectedLabel')}</div>
                    )}
                  </div>
                  {active && (
                    <div className="ml-auto w-6 h-6 rounded-full bg-white/30 flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" strokeWidth={3} />
                    </div>
                  )}
                </button>
              );
            },
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
        <div className="flex justify-end">
          <button
            onClick={() => onConfirm(selected)}
            className={`px-8 py-3 rounded-xl font-semibold text-white bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90 transition-all shadow-lg flex items-center gap-2`}
          >
            {ui.btn}
            <span className="text-lg">→</span>
          </button>
        </div>
      </div>
    </>
  );
};

// ============ Sub-Component: Theme/Design Select Screen ============
interface ThemeSelectScreenProps {
  currentTheme: Theme;
  allThemes: Record<ThemeId, Theme>;
  onSelectTheme: (id: ThemeId) => void;
  onContinue: () => void;
}

const ThemeSelectScreen: React.FC<ThemeSelectScreenProps> = ({
  currentTheme,
  allThemes,
  onSelectTheme,
  onContinue,
}) => {
  const { t } = useLanguage();

  const isLightTheme = (themeId: string) => themeId === 'light-gray' || themeId === 'pure-white';

  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <Palette className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{t('firstLaunch.theme.headline')}</h1>
            <p className="text-gray-300 mt-1">{t('firstLaunch.theme.sub')}</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="grid grid-cols-3 gap-4 max-w-4xl mx-auto">
          {Object.values(allThemes).map((theme) => {
            const isLight = isLightTheme(theme.id);
            const textColor = isLight ? 'text-slate-900' : 'text-white';
            const descColor = isLight ? 'text-slate-600' : 'text-gray-400';
            const active = currentTheme.id === theme.id;

            return (
              <button
                key={theme.id}
                onClick={() => onSelectTheme(theme.id)}
                className={`relative p-5 bg-gradient-to-br ${theme.colors.background} border-2 rounded-xl transition-all hover:scale-105 ${
                  active
                    ? 'border-white/40 shadow-lg ring-2 ring-white/20'
                    : 'border-white/10 hover:border-white/20'
                }`}
              >
                {active && (
                  <div className="absolute top-2 right-2 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-lg">
                    <Check className="w-4 h-4 text-slate-900" />
                  </div>
                )}

                <div className="flex justify-center mb-3 space-x-2">
                  <div className={`w-7 h-7 rounded-full bg-gradient-to-br ${theme.colors.gradient} shadow-md`} />
                  <div className="w-7 h-7 rounded-full shadow-md" style={{ backgroundColor: theme.colors.accent }} />
                </div>

                <div className="text-center">
                  <div className={`${textColor} font-semibold text-sm mb-1`}>{theme.name}</div>
                  <div className={`text-xs ${descColor}`}>{theme.description}</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
        <div className="flex justify-end">
          <button
            onClick={onContinue}
            className={`px-8 py-3 rounded-xl font-semibold text-white bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90 transition-all shadow-lg flex items-center gap-2`}
          >
            {t('firstLaunch.theme.btn')}
            <span className="text-lg">→</span>
          </button>
        </div>
      </div>
    </>
  );
};

// ============ Sub-Component: AI Setup Screen (überspringbar) ============
interface AISetupScreenProps {
  currentTheme: Theme;
  onContinue: () => void;
  onSkip: () => void;
}

const AISetupScreen: React.FC<AISetupScreenProps> = ({ currentTheme, onContinue, onSkip }) => {
  const { t } = useLanguage();

  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <Brain className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{t('firstLaunch.aiSetup.headline')}</h1>
            <p className="text-gray-300 mt-1">{t('firstLaunch.aiSetup.sub')}</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-3xl mx-auto">
          <AIAssistantSettingsPanel />
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
        <div className="flex justify-between items-center">
          <button
            onClick={onSkip}
            className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20"
          >
            {t('firstLaunch.aiSetup.skip')}
          </button>
          <button
            onClick={onContinue}
            className={`px-8 py-3 rounded-xl font-semibold text-white bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90 transition-all shadow-lg flex items-center gap-2`}
          >
            {t('firstLaunch.aiSetup.btn')}
            <span className="text-lg">→</span>
          </button>
        </div>
      </div>
    </>
  );
};

// ============ Sub-Component: Python Setup Screen ============
interface PythonSetupScreenProps {
  phase: 'checking' | 'installing' | 'error';
  dependencyStatus: DependencyStatus[];
  error: string;
  currentTheme: any;
  installProgress: Map<string, InstallProgress>;
}

const PythonSetupScreen: React.FC<PythonSetupScreenProps> = ({
  phase,
  dependencyStatus,
  error,
  currentTheme,
  installProgress
}) => {
  const progress = installProgress.get('seq_classification');
  const { t } = useLanguage();
  
  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <Package className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{t('firstLaunch.python.setupTitle')}</h1>
            <p className="text-gray-300">{t('firstLaunch.python.setupSubtitle')}</p>
          </div>
        </div>
      </div>
      
      {/* Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-y-auto">
        <div className="w-full max-w-2xl">
          {phase === 'checking' && (
            <>
              <div className="text-center mb-8">
                <Loader2 className="w-16 h-16 text-blue-400 animate-spin mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">{t('firstLaunch.python.checkingTitle')}</h2>
                <p className="text-gray-400">{t('firstLaunch.python.checkingDesc')}</p>
              </div>
            </>
          )}
          
          {phase === 'installing' && (
            <>
              <div className="text-center mb-8">
                <Loader2 className="w-16 h-16 text-blue-400 animate-spin mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">{t('firstLaunch.python.installingTitle')}</h2>
                <p className="text-gray-400">{t('firstLaunch.python.installingDesc')}</p>
              </div>
              
              <div className="space-y-4">
                {dependencyStatus.map(dep => (
                  <div key={dep.package} className="bg-white/5 rounded-xl p-4 border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-white">{dep.package}</span>
                      {dep.installed ? (
                        <span className="text-green-400 flex items-center gap-1">
                          <Check className="w-4 h-4" /> {dep.version || 'installed'}
                        </span>
                      ) : (
                        <span className="text-gray-400">{t('firstLaunch.python.installingPackage')}</span>
                      )}
                    </div>
                    
                    {!dep.installed && (
                      <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-full transition-all duration-300 animate-pulse"
                          style={{ width: '100%' }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              {progress && progress.message && (
                <div className="mt-6 p-4 bg-white/5 rounded-xl border border-white/10">
                  <p className="text-sm text-gray-300">{progress.message}</p>
                </div>
              )}
              
              <p className="text-sm text-gray-500 text-center mt-8">
                {t('firstLaunch.python.installDontClose')}
              </p>
            </>
          )}
          
          {phase === 'error' && (
            <>
              <div className="text-center mb-8">
                <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">{t('firstLaunch.python.errorTitle')}</h2>
                <p className="text-gray-400">{t('firstLaunch.python.errorSubtitle')}</p>
              </div>
              
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-8">
                <p className="text-red-300 text-sm whitespace-pre-wrap font-mono">{error}</p>
              </div>
              
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 mb-6">
                <h3 className="text-blue-300 font-semibold mb-2">{t('firstLaunch.python.nextSteps')}</h3>
                <ul className="text-blue-300 text-sm space-y-1 list-disc list-inside">
                  <li>{t('firstLaunch.python.step1')}</li>
                  <li>{t('firstLaunch.python.step2')}</li>
                  <li>{t('firstLaunch.python.step3')}</li>
                </ul>
              </div>
              
              <button
                onClick={() => window.location.reload()}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl transition-all font-semibold"
              >
                {t('firstLaunch.python.restart')}
              </button>
            </>
          )}
        </div>
      </div>
    </>
  );
};

// ============ Sub-Component: Plugin Selection Screen ============
interface PluginSelectionScreenProps {
  plugins: PluginInfo[];
  pluginsLoading: boolean;
  pluginsLoadError: string | null;
  onRetryLoadPlugins: () => void;
  selectedPlugins: Set<string>;
  installing: boolean;
  installProgress: Map<string, InstallProgress>;
  totalSize: number;
  totalTime: number;
  currentTheme: any;
  onTogglePlugin: (id: string) => void;
  onStartInstallation: () => void;
  onSkipSetup: () => void;
  getCategoryColor: (category: string) => string;
  getStatusColor: (status: string) => string;
}

const PluginSelectionScreen: React.FC<PluginSelectionScreenProps> = ({
  plugins,
  pluginsLoading,
  pluginsLoadError,
  onRetryLoadPlugins,
  selectedPlugins,
  installing,
  installProgress,
  totalSize,
  totalTime,
  currentTheme,
  onTogglePlugin,
  onStartInstallation,
  onSkipSetup,
  getCategoryColor,
  getStatusColor
}) => {
  const { t } = useLanguage();
  return (
    <>
      {/* Header */}
      <div className="flex-shrink-0 p-8 border-b border-white/10">
        <div className="flex items-center gap-4 mb-4">
          <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
            <Package className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">{t('firstLaunch.plugins.title')}</h1>
            <p className="text-gray-300">{t('firstLaunch.plugins.subtitle')}</p>
          </div>
        </div>
      </div>
      
      {!installing ? (
        <>
          {/* Plugin List - Scrollable */}
          <div className="flex-1 overflow-y-auto p-6">
            {pluginsLoading ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
                <p className="text-gray-400 text-sm">Loading available plugins...</p>
              </div>
            ) : pluginsLoadError ? (
              <div className="flex flex-col items-center justify-center py-16 gap-4 px-8 text-center">
                <AlertCircle className="w-10 h-10 text-red-400" />
                <div>
                  <p className="text-red-300 font-medium mb-1">Could not load plugins</p>
                  <p className="text-gray-500 text-xs max-w-md font-mono break-words">{pluginsLoadError}</p>
                </div>
                <button
                  onClick={onRetryLoadPlugins}
                  className="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20 text-sm"
                >
                  Retry
                </button>
              </div>
            ) : plugins.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
                <Package className="w-10 h-10 text-gray-500" />
                <p className="text-gray-400 text-sm">No plugins found.</p>
                <button
                  onClick={onRetryLoadPlugins}
                  className="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20 text-sm"
                >
                  Retry
                </button>
              </div>
            ) : (
            <div className="grid gap-3">
              {plugins.map(plugin => {
                const isSelected = selectedPlugins.has(plugin.id);
                const isDisabled = plugin.id === REQUIRED_PLUGIN_ID;
                
                return (
                  <div
                    key={plugin.id}
                    onClick={() => !isDisabled && onTogglePlugin(plugin.id)}
                    className={`
                      group relative p-4 rounded-xl border transition-all cursor-pointer
                      ${isSelected 
                        ? 'bg-white/20 border-white/40 shadow-lg' 
                        : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                      }
                      ${isDisabled ? 'opacity-60 cursor-default' : ''}
                    `}
                  >
                    <div className="flex items-start gap-4">
                      {/* Checkbox */}
                      <div className="flex-shrink-0 mt-1">
                        <div className={`
                          w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all
                          ${isSelected 
                            ? 'bg-gradient-to-br from-blue-500 to-purple-500 border-transparent' 
                            : 'border-white/30 group-hover:border-white/50'
                          }
                        `}>
                          {isSelected && <Check className="w-4 h-4 text-white" strokeWidth={3} />}
                        </div>
                      </div>
                      
                      {/* Category Badge */}
                      <div className="flex-shrink-0">
                        <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getCategoryColor(plugin.category)} flex items-center justify-center`}>
                          {pluginIcon(plugin, 'w-6 h-6 text-white')}
                        </div>
                      </div>
                      
                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-lg font-semibold text-white">{t(plugin.name)}</h3>
                          {plugin.built_in && (
                            <span className="px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded-full border border-green-400/30">
                              {t('firstLaunch.plugins.builtInBadge')}
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-300 mb-2">{t(plugin.description)}</p>
                        
                        {(
                          <div className="flex items-center gap-3 text-xs text-gray-400">
                            <span className="flex items-center gap-1">
                              <HardDrive className="w-3 h-3" />
                              {plugin.estimated_size_mb >= 1024 
                                ? `${(plugin.estimated_size_mb / 1024).toFixed(1)} GB`
                                : `${plugin.estimated_size_mb} MB`
                              }
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              ~{plugin.install_time_minutes} min
                            </span>
                            <span className="flex items-center gap-1">
                              <Package className="w-3 h-3" />
                              {plugin.required_packages.length} packages
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            )}
          </div>
          
          {/* Footer with Summary and Actions */}
          <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
            <div className="flex items-center justify-between mb-4">
              <div className="flex gap-8">
                <div>
                  <div className="text-2xl font-bold text-white">{selectedPlugins.size}</div>
                  <div className="text-xs text-gray-400">{t('firstLaunch.plugins.selected')}</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">
                    {totalSize >= 1024 
                      ? `${(totalSize / 1024).toFixed(1)} GB`
                      : `${totalSize} MB`
                    }
                  </div>
                  <div className="text-xs text-gray-400">{t('firstLaunch.plugins.downloadSize')}</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">~{totalTime} min</div>
                  <div className="text-xs text-gray-400">{t('firstLaunch.plugins.installTime')}</div>
                </div>
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={onSkipSetup}
                  className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20"
                >
                  {t('firstLaunch.plugins.skip')}
                </button>
                <button
                  onClick={onStartInstallation}
                  disabled={selectedPlugins.size === 0}
                  className={`
                    px-6 py-3 rounded-xl font-semibold transition-all flex items-center gap-2
                    ${selectedPlugins.size === 0 
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                      : `bg-gradient-to-r ${currentTheme.colors.gradient} text-white hover:shadow-lg hover:shadow-blue-500/50`
                    }
                  `}
                >
                  <Download className="w-5 h-5" />
                  {t('firstLaunch.plugins.install').replace('{count}', String(selectedPlugins.size))}
                </button>
              </div>
            </div>
            
            <p className="text-xs text-gray-400 text-center">
              {t('firstLaunch.plugins.footerNote')}
            </p>
          </div>
        </>
      ) : (
        /* Installation Progress */
        <div className="flex-1 flex flex-col p-8 overflow-y-auto">
          <div className="w-full max-w-3xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8">
              <Loader2 className="w-16 h-16 text-blue-400 animate-spin mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-white mb-2">{t('firstLaunch.plugins.installing')}</h2>
              <p className="text-gray-300">{t('firstLaunch.plugins.installingDesc')}</p>
              <p className="text-sm text-gray-400 mt-4">{t('firstLaunch.plugins.estimatedTime')}</p>
            </div>

            {/* Globale Progress Bar (aus system-Event) */}
            {(() => {
              const sys = installProgress.get('system');
              const pct = sys?.progress ?? 0;
              const isGlobalComplete = sys?.status === 'complete';
              const isGlobalFailed   = sys?.status === 'failed';
              return (
                <div className="mb-8">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-300">
                      {isGlobalComplete
                        ? t('firstLaunch.plugins.progressComplete')
                        : isGlobalFailed
                        ? t('firstLaunch.plugins.progressFailed')
                        : sys?.message || t('firstLaunch.plugins.progressInstalling')}
                    </span>
                    <span className={`text-sm font-bold ${
                      isGlobalComplete ? 'text-green-400' :
                      isGlobalFailed   ? 'text-red-400'   :
                      'text-blue-300'
                    }`}>{pct}%</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-4 overflow-hidden shadow-inner">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ease-out bg-gradient-to-r ${
                        isGlobalComplete ? 'from-green-500 to-green-400' :
                        isGlobalFailed   ? 'from-red-500 to-red-400'   :
                        'from-blue-600 to-purple-500'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })()}

            {/* Pakete-Liste */}
            <div className="space-y-3 mb-8">
              {Array.from(selectedPlugins).map(pluginId => {
                const plugin = plugins.find(p => p.id === pluginId);
                const progress = installProgress.get(pluginId);
                if (!plugin) return null;
                
                const isComplete = progress?.status === 'complete' || progress?.status === 'package_complete';
                const isFailed = progress?.status === 'failed';
                const isInstalling = progress && !isComplete && !isFailed;
                
                return (
                  <div 
                    key={pluginId} 
                    className={`rounded-xl p-4 border transition-all ${
                      isComplete ? 'bg-green-500/10 border-green-500/30' 
                      : isFailed  ? 'bg-red-500/10 border-red-500/30'
                      : 'bg-white/5 border-white/10'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getCategoryColor(plugin.category)} flex items-center justify-center flex-shrink-0`}>
                        {pluginIcon(plugin, 'w-5 h-5 text-white')}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold text-white">{t(plugin.name)}</span>
                          {isComplete  && <span className="text-green-400 flex items-center gap-1 flex-shrink-0"><Check className="w-4 h-4" />{t('firstLaunch.plugins.progressComplete')}</span>}
                          {isFailed    && <span className="text-red-400 flex items-center gap-1 flex-shrink-0"><XCircle className="w-4 h-4" />{t('firstLaunch.plugins.progressFailed')}</span>}
                          {isInstalling && <span className="text-blue-400 flex items-center gap-1 flex-shrink-0"><Loader2 className="w-3 h-3 animate-spin" />{t('firstLaunch.plugins.progressInstalling')}</span>}
                        </div>
                        {progress?.message && (
                          <p className={`text-sm mt-1 break-words ${
                            isFailed ? 'text-red-300' : isComplete ? 'text-green-300' : 'text-gray-300'
                          }`}>{progress.message}</p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
              <p className="text-sm text-blue-300 text-center">{t('firstLaunch.plugins.doNotClose')}</p>
              <p className="text-xs text-blue-400 text-center mt-2">{t('firstLaunch.plugins.waitNote')}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FirstLaunchSetup;
