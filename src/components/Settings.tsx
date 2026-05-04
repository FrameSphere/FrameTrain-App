import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { User, Key, Shield, Bell, Palette, Info, ExternalLink, LogOut, AlertCircle, CheckCircle, Check, Download, BookOpen, Loader2, Zap, MessageCircle, Send, ChevronDown, Plus, RefreshCw, Star, AlertTriangle, Inbox, Edit, Wrench, FileText, Lightbulb, MailX, Brain, Monitor } from 'lucide-react';
import { useTheme, ThemeId } from '../contexts/ThemeContext';
import { useAISettings, type AIProvider } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';
import { getVersion } from '@tauri-apps/api/app';
import { open as openUrl } from '@tauri-apps/plugin-shell';

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

// Support-related types
interface SupportMessage {
  id: number;
  sender: 'user' | 'admin';
  message: string;
  created_at: string;
}

interface StoredTicket {
  ticket_id: number;
  user_token: string;
  subject: string;
}

interface SupportTicket {
  id: number;
  subject: string;
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  created_at: string;
  updated_at: string;
}

interface InstallProgress {
  plugin_id: string;
  status: string;
  message: string;
  progress?: number;
}

interface SettingsProps {
  userData: UserData;
  onLogout: () => void;
}

type SettingsTab = 'account' | 'appearance' | 'notifications' | 'updates' | 'docs' | 'support' | 'ai-assistant' | 'about' | 'system';

// Status helpers for Support tickets
const STATUS_LABEL: Record<string, string> = {
  open: 'Offen',
  in_progress: 'In Bearbeitung',
  resolved: 'Gelöst',
  closed: 'Geschlossen',
};

const STATUS_COLOR: Record<string, string> = {
  open: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  in_progress: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
  resolved: 'text-green-400 bg-green-500/10 border-green-500/20',
  closed: 'text-gray-400 bg-gray-500/10 border-gray-500/20',
};

const PROVIDER_META: Record<AIProvider, {
  label: string; emoji: string; needsKey: boolean;
  keyPlaceholder: string; keyHint: string; keyLink: string;
  models: string[];
}> = {
  anthropic: {
    label: 'Claude (Anthropic)',
    emoji: '🤖',
    needsKey: true,
    keyPlaceholder: 'sk-ant-api03-...',
    keyHint: 'Kostenlos testen: console.anthropic.com',
    keyLink: 'https://console.anthropic.com',
    models: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  },
  openai: {
    label: 'GPT-4o (OpenAI)',
    emoji: '🟢',
    needsKey: true,
    keyPlaceholder: 'sk-...',
    keyHint: 'platform.openai.com/api-keys',
    keyLink: 'https://platform.openai.com/api-keys',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
  },
  groq: {
    label: 'Groq (Kostenlos)',
    emoji: '⚡',
    needsKey: true,
    keyPlaceholder: 'gsk_...',
    keyHint: '✅ Kostenloser Account — console.groq.com',
    keyLink: 'https://console.groq.com',
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
  },
  ollama: {
    label: 'Ollama (Lokal, kein Key)',
    emoji: '🦙',
    needsKey: false,
    keyPlaceholder: '',
    keyHint: '✅ Kein Account nötig — ollama.com installieren',
    keyLink: 'https://ollama.com',
    models: ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5'],
  },
};

const MANAGER_API = 'https://webcontrol-hq-api.karol-paschek.workers.dev';

// Support hook – persists ticket list in localStorage
function useStoredTickets(userId: string) {
  const key = `ft_tickets_${userId || 'anon'}`;

  const getAll = useCallback((): StoredTicket[] => {
    try {
      return JSON.parse(localStorage.getItem(key) || '[]');
    } catch {
      return [];
    }
  }, [key]);

  const add = useCallback((t: StoredTicket) => {
    const list = getAll().filter(x => x.ticket_id !== t.ticket_id);
    localStorage.setItem(key, JSON.stringify([t, ...list]));
  }, [key, getAll]);

  return { getAll, add };
}

export default function Settings({ userData, onLogout }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('account');
  const [showApiKey, setShowApiKey] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);
  const { currentTheme, setTheme, themes: allThemes } = useTheme();
  const { settings: aiSettings, updateSettings: updateAISettings } = useAISettings();
  const { setCurrentPageContent } = usePageContext();
  const [appVersion, setAppVersion] = useState<string>('Loading...');
  const [latestVersion, setLatestVersion] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<'checking' | 'up-to-date' | 'update-available' | 'error'>('checking');
  const [checkingUpdates, setCheckingUpdates] = useState(false);

  // Support state
  const [supportOpen, setSupportOpen] = useState(false);
  const [supportView, setSupportView] = useState<'list' | 'new' | 'thread'>('list');
  const [storedTickets, setStoredTickets] = useState<StoredTicket[]>([]);
  const [activeTicket, setActiveTicket] = useState<StoredTicket | null>(null);
  const [ticketInfo, setTicketInfo] = useState<SupportTicket | null>(null);
  const [messages, setMessages] = useState<SupportMessage[]>([]);
  const [threadLoading, setThreadLoading] = useState(false);
  const [newSubject, setNewSubject] = useState('');
  const [newMessage, setNewMessage] = useState('');
  const [replyText, setReplyText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [sendingReply, setSendingReply] = useState(false);
  const [supportBadge, setSupportBadge] = useState(0);
  const [showApiKeyField, setShowApiKeyField] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // System check state
  const [systemDeps, setSystemDeps] = useState<{package: string; installed: boolean; version?: string}[] | null>(null);
  const [systemReqs, setSystemReqs] = useState<{python_installed: boolean; python_version: string; torch_installed: boolean; torch_version: string; transformers_installed: boolean; transformers_version: string; cuda_available: boolean; mps_available: boolean; peft_installed: boolean; peft_version: string; ready: boolean} | null>(null);
  const [systemLoading, setSystemLoading] = useState(false);
  const [preventSleepActive, setPreventSleepActive] = useState(false);
  const [systemInstalling, setSystemInstalling] = useState(false);
  const [systemInstallProgress, setSystemInstallProgress] = useState<Map<string, InstallProgress>>(new Map());
  const [systemInstallError, setSystemInstallError] = useState<string>('');

  const { getAll, add } = useStoredTickets(userData.userId);

  const loadSystemInfo = useCallback(async () => {
    setSystemLoading(true);
    try {
      const [deps, reqs, sleep] = await Promise.all([
        invoke<{ package: string; installed: boolean; version?: string }[]>('check_dependency_status'),
        invoke<{ python_installed: boolean; python_version: string; torch_installed: boolean; torch_version: string; transformers_installed: boolean; transformers_version: string; cuda_available: boolean; mps_available: boolean; peft_installed: boolean; peft_version: string; ready: boolean }>('check_training_requirements'),
        invoke<boolean>('get_prevent_sleep_status'),
      ]);
      setSystemDeps(deps);
      setSystemReqs(reqs);
      setPreventSleepActive(sleep);
    } catch {
      // ignore
    } finally {
      setSystemLoading(false);
    }
  }, []);

  const installMissingDeps = useCallback(async () => {
    setSystemInstallError('');
    setSystemInstallProgress(new Map());
    setSystemInstalling(true);
    try {
      await invoke('install_plugins', { pluginIds: [] });
    } catch (e) {
      setSystemInstallError(String(e));
      setSystemInstalling(false);
    }
  }, []);

  useEffect(() => {
    loadAppVersion();
  }, []);

  // Page context for AI coach
  useEffect(() => {
    const providerInfo = PROVIDER_META[aiSettings.provider];
    const tabs: Record<string, string> = {
      account:      'Account (E-Mail, API-Key, Passwort, Abmelden)',
      appearance:   'Erscheinungsbild (Farbthemen für die App)',
      notifications: 'Benachrichtigungen',
      updates:      'Updates (Versionsinformationen, Update-Prüfung)',
      docs:         'Dokumentation (Anleitungen, Tipps)',
      support:      'Support (Tickets erstellen, mit Team kommunizieren)',
      'ai-assistant': 'KI-Assistent (Provider, API-Key, Modell konfigurieren)',
      about:        'Informationen (Über FrameTrain)',
      system:       'System (Training-Pakete, Hardware, Anti-Sleep)',
    };
    const lines = [
      '=== FrameTrain Einstellungen (Settings) ===',
      '',
      '--- Aktueller Tab ---',
      `Aktiver Tab: "${tabs[activeTab] || activeTab}"`,
      '',
      '--- Alle verfügbaren Tabs ---',
      ...Object.entries(tabs).map(([k, v]) => `\u2022 ${v}`),
      '',
      '--- KI-Assistent Konfiguration ---',
      `Status: ${aiSettings.enabled ? '\u2705 Aktiviert' : '\u274c Deaktiviert'}`,
      `Anbieter: ${providerInfo.label} (${aiSettings.provider})`,
      providerInfo.needsKey
        ? `API-Key: ${aiSettings.apiKey ? '\u2705 eingetragen' : '\u274c fehlt! (' + providerInfo.keyHint + ')'}`
        : `API-Key: nicht benötigt (${providerInfo.keyHint})`,
      `Modell: ${aiSettings.selectedModel || providerInfo.models[0]}`,
      aiSettings.provider === 'ollama' ? `Ollama-Modell: ${aiSettings.ollamaModel || 'llama3.2'}` : '',
      '',
      '--- Anbieter-Vergleich ---',
      '\u2022 Anthropic (Claude): Bezahlt, bestes Modell, sk-ant-... Key',
      '\u2022 OpenAI (GPT-4o): Bezahlt, sehr gut, sk-... Key',
      '\u2022 Groq: KOSTENLOS, schnell, gsk_... Key von console.groq.com',
      '\u2022 Ollama: KOSTENLOS, lokal, kein Key — ollama.com installieren',
      '',
      '--- App-Version ---',
      `Version: ${appVersion}`,
      updateStatus === 'update-available' ? `\u26a0\ufe0f Update verfügbar: ${latestVersion}` :
      updateStatus === 'up-to-date' ? '\u2705 App ist aktuell' : '',
      '',
      '--- Account ---',
      `E-Mail: ${userData.email || 'nicht gesetzt'}`,
      `User-ID: ${userData.userId || 'nicht gesetzt'}`,
      '',
      '--- Mögliche Aktionen ---',
      '\u2022 KI-Assistent aktivieren/deaktivieren (Toggle im KI-Assistent-Tab)',
      '\u2022 Anbieter wechseln: KI-Assistent-Tab > Anbieter auswählen',
      '\u2022 API-Key eintragen: KI-Assistent-Tab > API-Key Feld',
      '\u2022 Farbthema ändern: Erscheinungsbild-Tab',
      '\u2022 Support-Ticket erstellen: Support-Tab',
      '\u2022 Abmelden: Account-Tab > Abmelden-Button',
    ].filter(Boolean);
    setCurrentPageContent(lines.join('\n'));
  }, [activeTab, aiSettings, appVersion, updateStatus, latestVersion, userData, setCurrentPageContent]);

  // Check for unread admin replies
  useEffect(() => {
    async function checkBadge() {
      const tickets = getAll();
      if (!tickets.length) return;
      let unread = 0;
      for (const t of tickets) {
        const lastSeenKey = `ft_ticket_seen_${t.ticket_id}`;
        const lastSeen = parseInt(localStorage.getItem(lastSeenKey) || '0', 10);
        try {
          const res = await fetch(`${MANAGER_API}/api/support/${t.ticket_id}/thread?token=${t.user_token}`);
          if (!res.ok) continue;
          const data = await res.json();
          const msgs: SupportMessage[] = data.messages || [];
          const adminMsgs = msgs.filter((m: SupportMessage) => m.sender === 'admin');
          if (adminMsgs.length > 0) {
            const lastAdmin = new Date(adminMsgs[adminMsgs.length - 1].created_at).getTime();
            if (lastAdmin > lastSeen) unread++;
          }
        } catch {
          /* ignore */
        }
      }
      setSupportBadge(unread);
    }
    checkBadge();
  }, [supportOpen, getAll]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load stored tickets when support tab opens
  useEffect(() => {
    if (activeTab === 'support' && supportOpen) {
      setStoredTickets(getAll());
    }
  }, [activeTab, supportOpen, getAll]);

  useEffect(() => {
    if (activeTab === 'system') {
      loadSystemInfo();
    }
  }, [activeTab, loadSystemInfo]);

  // Anti-Sleep Status aktuell halten (z.B. wenn Training startet während System-Tab offen ist)
  useEffect(() => {
    if (activeTab !== 'system') return;
    const id = setInterval(() => {
      invoke<boolean>('get_prevent_sleep_status')
        .then(setPreventSleepActive)
        .catch(() => {});
    }, 1500);
    return () => clearInterval(id);
  }, [activeTab]);

  // Dependencies/Requirements im System-Tab automatisch aktualisieren (ohne manuelles "Neu prüfen")
  useEffect(() => {
    if (activeTab !== 'system') return;
    const id = setInterval(() => {
      if (systemLoading || systemInstalling) return;
      loadSystemInfo();
    }, 10000);
    return () => clearInterval(id);
  }, [activeTab, loadSystemInfo, systemInstalling, systemLoading]);

  useEffect(() => {
    if (!systemInstalling) return;

    let unlistenProgress: (() => void) | undefined;
    let unlistenComplete: (() => void) | undefined;

    (async () => {
      unlistenProgress = await listen<InstallProgress>('plugin-install-progress', (event) => {
        const progress = event.payload;
        setSystemInstallProgress(prev => new Map(prev).set(progress.plugin_id, progress));
        if (progress.status === 'failed') {
          setSystemInstallError(progress.message || 'Installation fehlgeschlagen');
          setSystemInstalling(false);
        }
      });

      unlistenComplete = await listen('plugin-install-complete', async () => {
        setSystemInstalling(false);
        setNotification({ type: 'success', message: 'Dependencies installiert ✓' });
        setTimeout(() => setNotification(null), 3000);
        await loadSystemInfo();
      });
    })();

    return () => {
      unlistenProgress?.();
      unlistenComplete?.();
    };
  }, [systemInstalling, loadSystemInfo]);

  const loadAppVersion = async () => {
    try {
      const version = await getVersion();
      setAppVersion(version);
      checkForUpdates(version);
    } catch (error) {
      console.error('Failed to load app version:', error);
      setAppVersion('Unknown');
    }
  };

  const checkForUpdates = async (currentVersion: string) => {
    setCheckingUpdates(true);
    setUpdateStatus('checking');
    
    try {
      let version: string = '';

      // Methode 1: GitHub API
      try {
        const response = await fetch(
          'https://api.github.com/repos/FrameSphere/FrameTrain-App/releases/latest',
          { headers: { 'Accept': 'application/json' }, cache: 'no-store' }
        );

        if (response.ok) {
          const data = await response.json();
          version = (data.tag_name as string)?.replace(/^v/, '') ?? '';
        }
      } catch (err) {
        console.warn('GitHub API failed:', err);
      }

      // Methode 2: Fallback zu latest.json
      if (!version) {
        try {
          const response = await fetch(
            'https://github.com/FrameSphere/FrameTrain-App/releases/latest/download/latest.json',
            { headers: { 'Accept': 'application/json' }, cache: 'no-store' }
          );

          if (response.ok) {
            const data = await response.json();
            version = (data.version as string)?.replace(/^v/, '') ?? '';
          }
        } catch (err) {
          console.warn('latest.json failed:', err);
        }
      }

      if (!version) {
        setUpdateStatus('error');
        setLatestVersion(null);
      } else {
        setLatestVersion(version);
        if (compareVersions(version, currentVersion) > 0) {
          setUpdateStatus('update-available');
        } else {
          setUpdateStatus('up-to-date');
        }
      }
    } catch (error) {
      console.error('Error checking updates:', error);
      setUpdateStatus('error');
    } finally {
      setCheckingUpdates(false);
    }
  };

  const compareVersions = (v1: string, v2: string): number => {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);
    for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
      const p1 = parts1[i] || 0;
      const p2 = parts2[i] || 0;
      if (p1 > p2) return 1;
      if (p1 < p2) return -1;
    }
    return 0;
  };

  const handleCheckUpdates = () => {
    checkForUpdates(appVersion);
  };

  const handleOpenGitHub = () => {
    openUrl('https://github.com/FrameSphere/FrameTrain-App/releases/latest').catch(() => {
      window.open('https://github.com/FrameSphere/FrameTrain-App/releases/latest', '_blank');
    });
  };

  // Support API functions
  const submitTicket = async () => {
    if (!newSubject.trim() || !newMessage.trim()) {
      setNotification({ type: 'error', message: 'Bitte Betreff und Nachricht ausfüllen' });
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch(`${MANAGER_API}/api/support/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userData.userId,
          name: userData.email?.split('@')[0] || 'FrameTrain User',
          email: userData.email || '',
          subject: newSubject.trim(),
          message: newMessage.trim(),
        }),
      });
      const data = await res.json();
      if (!data.success) throw new Error();

      const stored: StoredTicket = {
        ticket_id: data.ticket_id,
        user_token: data.user_token,
        subject: newSubject.trim(),
      };
      add(stored);
      setStoredTickets(getAll());
      setNewSubject('');
      setNewMessage('');
      openThread(stored);
      setNotification({ type: 'success', message: 'Ticket erfolgreich eingereicht!' });
      setTimeout(() => setNotification(null), 3000);
    } catch {
      setNotification({ type: 'error', message: 'Fehler beim Einreichen des Tickets' });
    } finally {
      setSubmitting(false);
    }
  };

  const openThread = async (stored: StoredTicket) => {
    setActiveTicket(stored);
    setSupportView('thread');
    setThreadLoading(true);
    try {
      const res = await fetch(`${MANAGER_API}/api/support/${stored.ticket_id}/thread?token=${stored.user_token}`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setTicketInfo(data.ticket);
      setMessages(data.messages);
    } catch {
      setTicketInfo(null);
      setMessages([]);
    } finally {
      setThreadLoading(false);
    }
  };

  const sendReply = async () => {
    if (!replyText.trim() || !activeTicket) return;
    setSendingReply(true);
    const text = replyText.trim();
    setReplyText('');
    try {
      const res = await fetch(`${MANAGER_API}/api/support/${activeTicket.ticket_id}/reply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: activeTicket.user_token, message: text }),
      });
      if (!res.ok) throw new Error();
      setMessages(prev => [...prev, { id: Date.now(), sender: 'user', message: text, created_at: new Date().toISOString() }]);
      if (ticketInfo) setTicketInfo({ ...ticketInfo, status: 'in_progress' });
    } catch {
      setNotification({ type: 'error', message: 'Senden fehlgeschlagen' });
      setReplyText(text);
    } finally {
      setSendingReply(false);
    }
  };

  const tabs = [
    { id: 'account' as SettingsTab, label: 'Konto', icon: User },
    { id: 'appearance' as SettingsTab, label: 'Darstellung', icon: Palette },
    { id: 'notifications' as SettingsTab, label: 'Benachrichtigungen', icon: Bell },
    { id: 'ai-assistant' as SettingsTab, label: 'KI-Assistent', icon: Brain },
    { id: 'system' as SettingsTab, label: 'System', icon: Monitor },
    { id: 'updates' as SettingsTab, label: 'Updates', icon: Download },
    { id: 'docs' as SettingsTab, label: 'Dokumentation', icon: BookOpen },
    { id: 'support' as SettingsTab, label: 'Support', icon: MessageCircle },
    { id: 'about' as SettingsTab, label: 'Über', icon: Info },
  ];

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setNotification({ type: 'success', message: 'In Zwischenablage kopiert!' });
      setTimeout(() => setNotification(null), 3000);
    } catch (error) {
      setNotification({ type: 'error', message: 'Kopieren fehlgeschlagen' });
      setTimeout(() => setNotification(null), 3000);
    }
  };

  const renderAccountTab = () => (
    <div className="space-y-6">
      {/* User Info Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Benutzerinformationen</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">E-Mail</label>
            <div className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
              {userData.email}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">User ID</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm truncate">
                {userData.userId}
              </div>
              <button
                onClick={() => copyToClipboard(userData.userId)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Kopieren
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* API Key Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">API-Key</h3>
          <Key className="w-5 h-5 text-purple-400" />
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Dein API-Key</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm">
                {showApiKey ? userData.apiKey : '••••••••••••••••••••'}
              </div>
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors border border-white/10"
              >
                {showApiKey ? 'Verbergen' : 'Anzeigen'}
              </button>
              <button
                onClick={() => copyToClipboard(userData.apiKey)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Kopieren
              </button>
            </div>
          </div>

          <div className="flex items-start space-x-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <Shield className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-yellow-300">
              Teile deinen API-Key niemals mit anderen. Er gewährt vollen Zugriff auf deinen Account.
            </p>
          </div>
        </div>
      </div>

      {/* Account Management */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Kontoverwaltung</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.vercel.app/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Dashboard öffnen</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.vercel.app/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Passwort ändern</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <button
            onClick={onLogout}
            className="w-full flex items-center justify-between px-4 py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-300 hover:text-red-200 transition-colors"
          >
            <span>Abmelden</span>
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );

  const renderAIAssistantTab = () => {
    const meta = PROVIDER_META[aiSettings.provider];
    return (
      <div className="space-y-6">
        {/* Enable/Disable Toggle */}
        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-white">KI-Assistent aktivieren</h3>
              <p className="text-sm text-gray-400 mt-1">Globale Einstellung für den KI-Coach in der App</p>
            </div>
            <button
              onClick={() => updateAISettings({ enabled: !aiSettings.enabled })}
              className={`w-11 h-6 rounded-full transition-all ${
                aiSettings.enabled ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'bg-white/10'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full bg-white shadow-lg transform transition-transform ${
                  aiSettings.enabled ? 'translate-x-5' : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>
          {aiSettings.enabled && (
            <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-sm text-green-300 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
              KI-Assistent ist aktiv und verfügbar
            </div>
          )}
          {!aiSettings.enabled && (
            <div className="p-3 bg-gray-500/10 border border-gray-500/20 rounded-lg text-sm text-gray-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              KI-Assistent ist deaktiviert
            </div>
          )}
        </div>

        {aiSettings.enabled && (
          <>
            {/* Provider Selection */}
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">KI-ANBIETER WÄHLEN</h3>
              <div className="grid grid-cols-2 gap-3">
                {(Object.entries(PROVIDER_META) as [AIProvider, typeof PROVIDER_META[AIProvider]][]).map(([key, m]) => (
                  <button
                    key={key}
                    onClick={() => updateAISettings({ provider: key, selectedModel: m.models[0] })}
                    className={`flex items-center gap-3 p-4 rounded-xl border text-left transition-all ${
                      aiSettings.provider === key
                        ? 'bg-purple-500/20 border-purple-500/50 text-white'
                        : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    <span className="text-3xl flex-shrink-0">{m.emoji}</span>
                    <div className="min-w-0">
                      <div className="text-sm font-semibold">{m.label}</div>
                      <div className="text-xs opacity-60 mt-0.5">{m.needsKey ? 'API-Key nötig' : '✅ Kein Key'}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Provider-specific Configuration */}
            {meta.needsKey ? (
              <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">API-Key</h3>
                  <a
                    href={meta.keyLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
                  >
                    Key holen ↗
                  </a>
                </div>
                
                <div>
                  <label className="block text-sm text-gray-400 mb-2">{meta.keyHint}</label>
                  <div className="flex items-center gap-2">
                    <input
                      type={showApiKeyField ? 'text' : 'password'}
                      value={aiSettings.apiKey}
                      onChange={(e) => updateAISettings({ apiKey: e.target.value })}
                      placeholder={meta.keyPlaceholder}
                      className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                    />
                    <button
                      onClick={() => setShowApiKeyField(!showApiKeyField)}
                      className="px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-400 hover:text-white transition-all text-sm"
                    >
                      {showApiKeyField ? 'Verbergen' : 'Anzeigen'}
                    </button>
                  </div>
                </div>

                <p className="text-xs text-gray-500">Key wird nur lokal gespeichert, nie an FrameTrain übertragen.</p>

                {/* Model Selection */}
                <div>
                  <label className="block text-sm font-semibold text-white mb-2">Modell</label>
                  <div className="flex flex-wrap gap-2">
                    {meta.models.map(m => (
                      <button
                        key={m}
                        onClick={() => updateAISettings({ selectedModel: m })}
                        className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                          aiSettings.selectedModel === m
                            ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                            : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              /* Ollama Configuration */
              <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">Ollama-Konfiguration</h3>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Modell-Name (muss lokal installiert sein)</label>
                  <input
                    type="text"
                    value={aiSettings.ollamaModel}
                    onChange={(e) => updateAISettings({ ollamaModel: e.target.value, selectedModel: e.target.value })}
                    placeholder="llama3.2"
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-white mb-2">Beliebte Modelle</label>
                  <div className="flex flex-wrap gap-2">
                    {PROVIDER_META.ollama.models.map(m => (
                      <button
                        key={m}
                        onClick={() => updateAISettings({ ollamaModel: m, selectedModel: m })}
                        className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                          aiSettings.ollamaModel === m
                            ? 'bg-green-500/20 border-green-500/50 text-green-300'
                            : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-4 bg-white/[0.03] rounded-lg border border-white/10 text-xs text-gray-400 space-y-2">
                  <div className="font-semibold text-gray-300">🦙 Ollama noch nicht installiert?</div>
                  <div>1. <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:underline">ollama.com</a> — kostenlos herunterladen</div>
                  <div>2. Im Terminal: <code className="bg-black/30 px-1 py-0.5 rounded font-mono text-xs">ollama pull llama3.2</code></div>
                  <div>3. Ollama läuft dann im Hintergrund</div>
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-sm text-blue-300 flex items-start gap-3">
              <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold mb-1">Diese Einstellungen gelten global</div>
                <p className="text-xs text-blue-200">Der KI-Anbieter und das Modell werden für alle KI-Funktionen verwendet: Quick Chat, Training-Analysen und Labor-Assistenz.</p>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  const handleThemeChange = async (themeId: ThemeId) => {
    setTheme(themeId);
    setNotification({ type: 'success', message: 'Theme erfolgreich geändert!' });
    setTimeout(() => setNotification(null), 3000);
  };

  const renderAppearanceTab = () => {
    // Helper function to determine if a theme is light
    const isLightTheme = (themeId: string) => {
      return themeId === 'light-gray' || themeId === 'pure-white';
    };

    return (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Farbschema</h3>
        <p className="text-gray-400 mb-6">Wähle dein bevorzugtes Farbschema für die Desktop-App</p>
        
        <div className="grid grid-cols-3 gap-4 max-h-[500px] overflow-y-auto pr-2">
          {Object.values(allThemes).map((theme) => {
            const isLight = isLightTheme(theme.id);
            const textColor = isLight ? 'text-slate-900' : 'text-white';
            const descColor = isLight ? 'text-slate-600' : 'text-gray-400';
            
            return (
            <button
              key={theme.id}
              onClick={() => handleThemeChange(theme.id)}
              className={`relative p-5 bg-gradient-to-br ${theme.colors.background} border-2 rounded-xl transition-all hover:scale-105 ${
                currentTheme.id === theme.id
                  ? 'border-white/40 shadow-lg ring-2 ring-white/20'
                  : 'border-white/10 hover:border-white/20'
              }`}
            >
              {/* Checkmark for active theme */}
              {currentTheme.id === theme.id && (
                <div className="absolute top-2 right-2 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-lg">
                  <Check className="w-4 h-4 text-slate-900" />
                </div>
              )}
              
              {/* Color preview */}
              <div className="flex justify-center mb-3 space-x-2">
                <div className={`w-7 h-7 rounded-full bg-gradient-to-br ${theme.colors.gradient} shadow-md`} />
                <div className="w-7 h-7 rounded-full shadow-md" style={{ backgroundColor: theme.colors.accent }} />
              </div>
              
              {/* Theme info */}
              <div className="text-center">
                <div className={`${textColor} font-semibold text-sm mb-1`}>{theme.name}</div>
                <div className={`text-xs ${descColor}`}>{theme.description}</div>
              </div>
            </button>
            );
          })}
        </div>
      </div>

      {/* Preview Section */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Vorschau</h3>
        <div className={`p-6 bg-gradient-to-br ${currentTheme.colors.background} rounded-xl border border-white/10`}>
          <div className="flex items-center space-x-4 mb-4">
            <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${currentTheme.colors.gradient} flex items-center justify-center`}>
              <Palette className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-white font-semibold">Beispiel Button</div>
              <div className="text-gray-400 text-sm">So sieht dein Theme aus</div>
            </div>
          </div>
          <button className={`w-full py-3 px-4 bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold rounded-lg hover:opacity-90 transition-opacity`}>
            Beispiel Button
          </button>
        </div>
      </div>
    </div>
    );
  };

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Desktop-Benachrichtigungen</h3>
        
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Training abgeschlossen</div>
              <div className="text-sm text-gray-400">Benachrichtigung wenn Training fertig ist</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Fehler und Warnungen</div>
              <div className="text-sm text-gray-400">Benachrichtigung bei Problemen</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Updates verfügbar</div>
              <div className="text-sm text-gray-400">Benachrichtigung über neue Versionen</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>
        </div>
      </div>
    </div>
  );

  const renderUpdatesTab = () => (
    <div className="space-y-6">
      {/* Update Status Card */}
      <div className={`rounded-xl p-6 border ${
        updateStatus === 'update-available'
          ? 'bg-red-500/10 border-red-500/30'
          : updateStatus === 'up-to-date'
          ? 'bg-green-500/10 border-green-500/30'
          : 'bg-white/5 border-white/10'
      }`}>
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {updateStatus === 'checking' && (
              <Loader2 className="w-6 h-6 text-gray-400 animate-spin" />
            )}
            {updateStatus === 'update-available' && (
              <AlertCircle className="w-6 h-6 text-red-400" />
            )}
            {updateStatus === 'up-to-date' && (
              <CheckCircle className="w-6 h-6 text-green-400" />
            )}
            {updateStatus === 'error' && (
              <AlertCircle className="w-6 h-6 text-gray-400" />
            )}
            <h3 className={`text-lg font-semibold ${
              updateStatus === 'update-available'
                ? 'text-red-300'
                : updateStatus === 'up-to-date'
                ? 'text-green-300'
                : 'text-white'
            }`}>
              {updateStatus === 'checking' && 'Auf Updates prüfen...'}
              {updateStatus === 'up-to-date' && 'Du bist auf dem neuesten Stand!'}
              {updateStatus === 'update-available' && 'Neues Update verfügbar'}
              {updateStatus === 'error' && 'Update-Prüfung fehlgeschlagen'}
            </h3>
          </div>
          <button
            onClick={handleCheckUpdates}
            disabled={checkingUpdates}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg text-sm font-semibold transition-all"
          >
            {checkingUpdates ? 'Wird geprüft...' : 'Neu prüfen'}
          </button>
        </div>

        {/* Version Comparison */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="bg-black/20 rounded-lg p-4">
            <p className="text-gray-400 text-xs mb-1">Installiert</p>
            <p className="text-white font-mono font-semibold text-lg">v{appVersion}</p>
          </div>
          <div className="flex items-center justify-center">
            <Zap className="w-5 h-5 text-gray-400" />
          </div>
          <div className={`${
            updateStatus === 'update-available'
              ? 'bg-red-500/20 border-red-500/30'
              : 'bg-black/20'
          } rounded-lg p-4 border`}>
            <p className="text-gray-400 text-xs mb-1">Verfügbar</p>
            <p className={`font-mono font-semibold text-lg ${
              latestVersion ? 'text-white' : 'text-gray-500'
            }`}>
              v{latestVersion || '—'}
            </p>
          </div>
        </div>

        {/* Status Message */}
        {updateStatus === 'update-available' && (
          <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 mb-4">
            <p className="text-red-300 text-sm">
              Eine neuere Version ist verfügbar! Klick auf "Zu GitHub Releases" um die neue Version herunterzuladen.
            </p>
          </div>
        )}

        {updateStatus === 'error' && (
          <div className="bg-gray-500/20 border border-gray-500/30 rounded-lg p-4 mb-4">
            <p className="text-gray-300 text-sm">
              Konnte nicht auf Updates prüfen. Prüfe deine Internetverbindung oder klick "Neu prüfen".
            </p>
          </div>
        )}
      </div>

      {/* GitHub Releases Link */}
      <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-6">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center flex-shrink-0">
            <Download className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold text-white mb-2">Neue Version herunterladen</h3>
            <p className="text-gray-300 mb-4">
              Besuche die GitHub Releases-Seite, um die neueste Version von FrameTrain herunterzuladen.
            </p>
            
            <button
              onClick={handleOpenGitHub}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-semibold transition-all"
            >
              <Download className="w-5 h-5" />
              <span>Zu GitHub Releases</span>
              <ExternalLink className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Installation Instructions */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-5 h-5 text-gray-400" />
          <h3 className="text-lg font-semibold text-white">Update-Installation</h3>
        </div>
        <div className="space-y-3 text-gray-400 text-sm">
          <p>
            <span className="font-semibold text-white">1.</span> Lade die neue Version herunter
          </p>
          <p>
            <span className="font-semibold text-white">2.</span> Deinstalliere die alte FrameTrain App komplett:
          </p>
          <ul className="ml-6 space-y-1 list-disc">
            <li>macOS: <span className="text-white">Applications</span> → FrameTrain → <span className="text-white">Move to Trash</span></li>
            <li>Windows: <span className="text-white">Control Panel</span> → <span className="text-white">Uninstall</span></li>
            <li>Linux: <span className="text-white">sudo apt remove frametrain</span> oder entsprechend für deine Distribution</li>
          </ul>
          <p>
            <span className="font-semibold text-white">3.</span> Installiere die neue Version
          </p>
          <p>
            <span className="font-semibold text-white">4.</span> Starte FrameTrain neu (alle Einstellungen und Daten bleiben erhalten)
          </p>
        </div>
      </div>

      {/* Auto-Update Info */}
      <div className="bg-blue-500/10 rounded-xl p-6 border border-blue-500/20">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-white font-semibold mb-1">Automatische Update-Benachrichtigung</h3>
            <p className="text-blue-300 text-sm">
              FrameTrain prüft automatisch beim Start auf neue Versionen. 
              Wenn ein Update verfügbar ist, wird dir ein Modal angezeigt. 
              Du kannst auch hier jederzeit manuell prüfen.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDocsTab = () => (
    <div className="space-y-6">
      {/* Docs Header Card */}
      <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-3">
          <BookOpen className="w-6 h-6 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">FrameTrain Dokumentation</h3>
        </div>
        <p className="text-sm text-gray-300">
          Lerne alles über FrameTrain – von den Grundlagen bis zu erweiterten Funktionen.
        </p>
      </div>

      {/* Main Docs Link */}
      <a
        href="https://frame-train.vercel.app/docs"
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/30 rounded-xl p-6 transition-all hover:shadow-lg"
      >
        <div className="flex items-start justify-between">
          <div>
            <h4 className="text-base font-semibold text-white mb-2">Komplette Dokumentation</h4>
            <p className="text-sm text-gray-400">
              Vollständige Anleitung mit allen Features, Tutorials und Best Practices
            </p>
          </div>
          <ExternalLink className="w-5 h-5 text-blue-400 flex-shrink-0 mt-1" />
        </div>
      </a>

      {/* AI Training Guide */}
      <a
        href="https://frame-train.vercel.app/docs/ai-training-guide"
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/30 rounded-xl p-6 transition-all hover:shadow-lg"
      >
        <div className="flex items-start justify-between">
          <div>
            <h4 className="text-base font-semibold text-white mb-2">KI-Training Guide</h4>
            <p className="text-sm text-gray-400">
              Schritt-für-Schritt Anleitung zum Training deines eigenen KI-Modells
            </p>
          </div>
          <ExternalLink className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-1" />
        </div>
      </a>

      {/* Quick Tips Card */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h4 className="text-base font-semibold text-white mb-4">Schnelle Tipps</h4>
        <ul className="space-y-3">
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">Starte mit dem Training Guide um ein neues Modell zu trainieren</span>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">Nutze die Dokumentation zum Troubleshooting von Fehlern</span>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">In den Docs findest du Video-Tutorials und Code-Beispiele</span>
          </li>
        </ul>
      </div>

      {/* Support Info */}
      <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl p-6">
        <div className="flex items-center justify-center gap-2 mb-2">
          <BookOpen className="w-4 h-4 text-blue-400" />
          <p className="text-sm text-gray-400 text-center">
            Die Dokumentation wird regelmäßig aktualisiert.
          </p>
        </div>
        <p className="text-sm text-gray-400 text-center">
          Haben Sie Fragen? Schau in den Docs vorbei!
        </p>
      </div>
    </div>
  );

  const renderSupportTab = () => (
    <div className="space-y-6">
      {/* Support Header */}
      <div className="glass-strong rounded-2xl shadow-lg border border-white/10 overflow-hidden">
        {/* Header */}
        <button
          onClick={() => {
            const opening = !supportOpen;
            setSupportOpen(opening);
            if (opening) {
              // Mark all tickets seen
              getAll().forEach((t) => {
                localStorage.setItem(`ft_ticket_seen_${t.ticket_id}`, Date.now().toString());
              });
              setSupportBadge(0);
            }
          }}
          className="w-full flex items-center justify-between px-8 py-6 hover:bg-white/5 transition-colors relative"
        >
          {/* Unread admin reply badge */}
          {supportBadge > 0 && !supportOpen && (
            <span className="absolute top-3 right-16 flex items-center justify-center w-5 h-5 rounded-full bg-red-500 text-white text-[11px] font-black shadow-lg shadow-red-500/40 animate-pulse">
              {supportBadge}
            </span>
          )}
          <div className="flex items-center gap-3">
            <MessageCircle className="w-6 h-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-white">Support</h2>
            {storedTickets.length > 0 && (
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {storedTickets.length} Ticket{storedTickets.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform duration-200 ${supportOpen ? 'rotate-180' : ''}`} />
        </button>

        {supportOpen && (
          <div className="border-t border-white/10">
            {/* Sub-nav */}
            <div className="flex border-b border-white/10">
              {[
                { id: 'list' as const, label: 'Meine Tickets', icon: Inbox },
                { id: 'new' as const, label: 'Neues Ticket', icon: Edit },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => {
                    setSupportView(tab.id);
                    setActiveTicket(null);
                  }}
                  className={`px-6 py-3 text-sm font-semibold transition-colors flex items-center gap-2 ${
                    supportView === tab.id || (supportView === 'thread' && tab.id === 'list')
                      ? 'text-purple-400 border-b-2 border-purple-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="p-8">
              {/* New ticket form */}
              {supportView === 'new' && (
                <div className="max-w-2xl">
                  <h3 className="text-lg font-bold text-white mb-6">Neues Support-Ticket</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Betreff</label>
                      <input
                        value={newSubject}
                        onChange={(e) => setNewSubject(e.target.value)}
                        placeholder="Kurze Beschreibung deines Problems..."
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Nachricht</label>
                      <textarea
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        placeholder="Beschreibe dein Anliegen so detailliert wie möglich..."
                        rows={5}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-colors resize-none"
                      />
                    </div>
                    <div className="flex items-center gap-3 pt-2">
                      <p className="text-xs text-gray-500 flex-1">
                        Deine User-ID <code className="text-purple-400 bg-white/5 px-1 rounded">{userData.userId}</code> wird automatisch
                        mitgeschickt.
                      </p>
                      <button
                        onClick={submitTicket}
                        disabled={submitting || !newSubject.trim() || !newMessage.trim()}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {submitting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                        {submitting ? 'Wird gesendet...' : 'Ticket einreichen'}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Ticket list */}
              {supportView === 'list' && !activeTicket && (
                <div>
                  {storedTickets.length === 0 ? (
                    <div className="text-center py-12">
                      <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4">
                        <MailX className="w-8 h-8 text-gray-500" />
                      </div>
                      <p className="text-gray-400 mb-2">Du hast noch keine Support-Tickets.</p>
                      <p className="text-gray-500 text-sm mb-6">Hast du ein Problem oder eine Frage? Wir helfen gerne.</p>
                      <button
                        onClick={() => setSupportView('new')}
                        className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors mx-auto text-sm font-semibold"
                      >
                        <Plus className="w-4 h-4" /> Erstes Ticket erstellen
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold text-white">Deine Tickets</h3>
                        <button
                          onClick={() => setSupportView('new')}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded-lg text-sm transition-colors border border-purple-500/20"
                        >
                          <Plus className="w-3.5 h-3.5" /> Neues Ticket
                        </button>
                      </div>
                      {storedTickets.map((t) => (
                        <button
                          key={t.ticket_id}
                          onClick={() => openThread(t)}
                          className="w-full flex items-center justify-between glass rounded-xl px-5 py-4 border border-white/10 hover:border-purple-500/30 hover:bg-white/5 transition-all text-left"
                        >
                          <div>
                            <p className="text-white font-semibold text-sm">{t.subject}</p>
                            <p className="text-gray-500 text-xs mt-0.5">Ticket #{t.ticket_id}</p>
                          </div>
                          <MessageCircle className="w-4 h-4 text-gray-500" />
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Thread view */}
              {supportView === 'thread' && activeTicket && (
                <div className="max-w-2xl">
                  {/* Back */}
                  <button
                    onClick={() => {
                      setSupportView('list');
                      setActiveTicket(null);
                    }}
                    className="flex items-center gap-1.5 text-gray-400 hover:text-white text-sm mb-5 transition-colors"
                  >
                    ← Zurück zur Übersicht
                  </button>

                  {threadLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
                    </div>
                  ) : (
                    <>
                      {/* Ticket meta */}
                      <div className="glass rounded-xl px-5 py-4 border border-white/10 mb-5">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <h3 className="text-white font-bold">{activeTicket.subject}</h3>
                            <p className="text-gray-500 text-xs mt-0.5">Ticket #{activeTicket.ticket_id}</p>
                          </div>
                          {ticketInfo && (
                            <span className={`text-xs font-bold px-3 py-1 rounded-full border flex-shrink-0 ${STATUS_COLOR[ticketInfo.status] || STATUS_COLOR.open}`}>
                              {STATUS_LABEL[ticketInfo.status] || ticketInfo.status}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Messages */}
                      <div className="space-y-4 mb-5 max-h-96 overflow-y-auto pr-1">
                        {messages.length === 0 && <p className="text-center text-gray-500 text-sm py-8">Noch keine Nachrichten</p>}
                        {messages.map((m) => (
                          <div key={m.id} className={`flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div
                              className={`max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                                m.sender === 'user'
                                  ? 'bg-gradient-to-br from-purple-600 to-pink-600 text-white rounded-br-sm'
                                  : 'glass border border-white/10 text-gray-200 rounded-bl-sm'
                              }`}
                            >
                              <p style={{ whiteSpace: 'pre-wrap' }}>{m.message}</p>
                              <p className={`text-xs mt-1.5 flex items-center gap-1 ${m.sender === 'user' ? 'text-purple-200' : 'text-gray-500'}`}>
                                {m.sender === 'user' ? 'Du' : <>
                                  <Wrench className="w-3 h-3" />
                                  Support
                                </>} · {new Date(m.created_at).toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}
                              </p>
                            </div>
                          </div>
                        ))}
                        <div ref={messagesEndRef} />
                      </div>

                      {/* Reply box – nur wenn nicht geschlossen */}
                      {ticketInfo?.status !== 'closed' && ticketInfo?.status !== 'resolved' ? (
                        <div className="flex gap-3">
                          <textarea
                            value={replyText}
                            onChange={(e) => setReplyText(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendReply();
                            }}
                            placeholder="Nachricht schreiben... (Strg+Enter zum Senden)"
                            rows={3}
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-colors resize-none text-sm"
                          />
                          <button
                            onClick={sendReply}
                            disabled={sendingReply || !replyText.trim()}
                            className="self-end flex items-center gap-1.5 px-5 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-colors disabled:opacity-50 font-semibold text-sm"
                          >
                            {sendingReply ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                          </button>
                        </div>
                      ) : (
                        <div className="glass rounded-xl px-4 py-3 border border-white/10 text-center text-gray-500 text-sm">
                          Dieses Ticket ist {STATUS_LABEL[ticketInfo.status]?.toLowerCase()} – du kannst keine weitere Nachricht schreiben.
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderSystemTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Python-Pakete (Training)</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={installMissingDeps}
              disabled={systemLoading || systemInstalling || !(systemDeps?.some(d => !d.installed))}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-lg text-emerald-300 hover:text-emerald-200 text-sm transition-all disabled:opacity-50 disabled:hover:bg-emerald-500/10"
              title="Installiert fehlende Python-Pakete via pip"
            >
              {systemInstalling ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              Fehlende installieren
            </button>
            <button
              onClick={loadSystemInfo}
              disabled={systemLoading || systemInstalling}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-400 hover:text-white text-sm transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${systemLoading ? 'animate-spin' : ''}`} />
              Neu prüfen
            </button>
          </div>
        </div>

        {!systemDeps ? (
          <div className="text-gray-500 text-sm">Status noch nicht geladen.</div>
        ) : (
          <div className="space-y-2">
            {systemDeps.map(dep => (
              <div key={dep.package} className="flex items-center justify-between px-4 py-3 bg-white/[0.03] rounded-xl border border-white/10">
                <div className="flex items-center gap-3">
                  {dep.installed
                    ? <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                    : <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />}
                  <span className="text-white font-mono text-sm">{dep.package}</span>
                </div>
                <span className={`text-xs font-mono ${dep.installed ? 'text-emerald-400' : 'text-red-400'}`}>
                  {dep.installed ? (dep.version ?? 'installiert') : 'fehlt'}
                </span>
              </div>
            ))}
          </div>
        )}

        {systemInstallError && (
          <div className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{systemInstallError}</span>
          </div>
        )}

        {systemInstalling && (
          <div className="mt-4 p-4 rounded-xl bg-white/[0.03] border border-white/10 space-y-2">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
                Installation läuft…
              </span>
              <span className="font-mono">
                {(systemInstallProgress.get('seq_classification')?.progress ?? 0)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-teal-500 transition-all"
                style={{ width: `${Math.min(systemInstallProgress.get('seq_classification')?.progress ?? 0, 100)}%` }}
              />
            </div>
            <div className="text-[11px] text-gray-500 font-mono break-words">
              {systemInstallProgress.get('seq_classification')?.message ?? 'pip wird gestartet…'}
            </div>
          </div>
        )}
      </div>

      {systemReqs && (
        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">Hardware & Beschleunigung</h3>
          <div className="space-y-2">
            {[
              { label: 'Python', value: systemReqs.python_version, ok: systemReqs.python_installed },
              { label: 'PyTorch', value: systemReqs.torch_version, ok: systemReqs.torch_installed },
              { label: 'Transformers', value: systemReqs.transformers_version, ok: systemReqs.transformers_installed },
              { label: 'PEFT (LoRA)', value: systemReqs.peft_version, ok: systemReqs.peft_installed },
              { label: 'CUDA (GPU)', value: systemReqs.cuda_available ? 'Verfügbar ✓' : 'Nicht verfügbar', ok: systemReqs.cuda_available },
              { label: 'Apple MPS', value: systemReqs.mps_available ? 'Verfügbar ✓' : 'Nicht verfügbar', ok: systemReqs.mps_available },
            ].map(row => (
              <div key={row.label} className="flex items-center justify-between px-4 py-3 bg-white/[0.03] rounded-xl border border-white/10">
                <span className="text-gray-400 text-sm">{row.label}</span>
                <span className={`text-xs font-mono ${row.ok ? 'text-emerald-400' : 'text-gray-500'}`}>{row.value}</span>
              </div>
            ))}
          </div>
          <div className={`mt-4 p-3 rounded-xl border text-sm flex items-center gap-2 ${systemReqs.ready ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300' : 'bg-red-500/10 border-red-500/20 text-red-300'}`}>
            {systemReqs.ready ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            {systemReqs.ready ? 'System ist bereit für KI-Trainings' : 'System nicht bereit — Pakete fehlen'}
          </div>
        </div>
      )}

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Anti-Sleep Status</h3>
        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${preventSleepActive ? 'bg-blue-500/10 border-blue-500/20' : 'bg-white/[0.03] border-white/10'}`}>
          <div className={`w-2.5 h-2.5 rounded-full ${preventSleepActive ? 'bg-blue-400 animate-pulse' : 'bg-gray-600'}`} />
          <span className={`text-sm ${preventSleepActive ? 'text-blue-300' : 'text-gray-500'}`}>
            {preventSleepActive ? 'Aktiv — Gerät bleibt während Training wach' : 'Inaktiv — kein Training läuft'}
          </span>
        </div>
      </div>
    </div>
  );

  const renderAboutTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10 text-center">
      <div
      className="inline-flex items-center justify-center mb-4 rounded-[18px]"
      style={{
      background: 'linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%)',
      width: 72,
      height: 72,
        boxShadow: '0 0 32px rgba(168,85,247,0.45), 0 8px 32px rgba(0,0,0,0.4)',
      }}
      >
      <span
        style={{
            fontFamily: 'Arial, sans-serif',
              fontSize: 40,
              fontWeight: 900,
              color: 'white',
              lineHeight: 1,
              userSelect: 'none',
            }}
          >
            F
          </span>
        </div>
        <h3 className="text-2xl font-bold text-white mb-2">FrameTrain Desktop</h3>
        <p className="text-gray-400 mb-4">Version {appVersion}</p>
        <p className="text-sm text-gray-400 max-w-md mx-auto">
          Trainiere Machine Learning Modelle lokal auf deinem Computer mit der Leistung von PyTorch.
        </p>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Links</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Website</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.vercel.app/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Dokumentation</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://github.com/KarolP-tech/FrameTrain/releases"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>GitHub</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>
        </div>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <p className="text-sm text-gray-400 text-center">
          © 2025 FrameTrain. Alle Rechte vorbehalten.
        </p>
      </div>
    </div>
  );

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Einstellungen</h2>
        <p className="text-gray-400">Verwalte dein Konto und App-Einstellungen</p>
      </div>

      {/* Notification */}
      {notification && (
        <div className={`mb-6 flex items-start space-x-2 p-4 rounded-lg border ${
          notification.type === 'success'
            ? 'bg-green-500/10 border-green-500/20'
            : 'bg-red-500/10 border-red-500/20'
        }`}>
          {notification.type === 'success' ? (
            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
          ) : (
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          )}
          <p className={`text-sm ${
            notification.type === 'success' ? 'text-green-300' : 'text-red-300'
          }`}>
            {notification.message}
          </p>
        </div>
      )}

      <div className="grid grid-cols-4 gap-6">
        {/* Sidebar Tabs */}
        <div className="space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg'
                    : 'bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white border border-white/10'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content Area */}
        <div className="col-span-3">
          {activeTab === 'account' && renderAccountTab()}
          {activeTab === 'appearance' && renderAppearanceTab()}
          {activeTab === 'notifications' && renderNotificationsTab()}
          {activeTab === 'ai-assistant' && renderAIAssistantTab()}
          {activeTab === 'system' && renderSystemTab()}
          {activeTab === 'updates' && renderUpdatesTab()}
          {activeTab === 'docs' && renderDocsTab()}
          {activeTab === 'support' && renderSupportTab()}
          {activeTab === 'about' && renderAboutTab()}
        </div>
      </div>
    </div>
  );
}
