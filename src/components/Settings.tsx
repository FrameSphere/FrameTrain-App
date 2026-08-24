import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { User, Key, Shield, Bell, Palette, Info, ExternalLink, LogOut, AlertCircle, CheckCircle, Check, Download, BookOpen, Loader2, Zap, MessageCircle, Send, ChevronDown, Plus, RefreshCw, Star, AlertTriangle, Inbox, Edit, Wrench, FileText, Lightbulb, MailX, Brain, Monitor, Pencil, Globe, Sparkles, X, Flame, Leaf, Scale } from 'lucide-react';
import { useTheme, ThemeId } from '../contexts/ThemeContext';
import { useLanguage, LANGUAGE_META, type Language } from '../contexts/LanguageContext';
import { useEscapeKey } from '../hooks/useEscapeKey';
import { useAISettings, type AIProvider, type TokenBudget, TOKEN_BUDGET_CONFIG } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';
import { buildPageContext, kv } from '../ai/coachContext';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../plugins/hf-encoder/detect';
import { getVersion } from '@tauri-apps/api/app';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import { PROVIDER_META } from '../ai/providerMeta';
import { getStoredAuthorName, saveAuthorName } from './OpenLibraryModal';
import { dateLocale } from '../utils/dateLocale';

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

type SettingsTab = 'account' | 'appearance' | 'language' | 'notifications' | 'updates' | 'docs' | 'support' | 'ai-assistant' | 'about' | 'system';

const STATUS_COLOR: Record<string, string> = {
  open: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  in_progress: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
  resolved: 'text-green-400 bg-green-500/10 border-green-500/20',
  closed: 'text-gray-400 bg-gray-500/10 border-gray-500/20',
};

// PROVIDER_META ist zentral definiert in src/ai/providerMeta.ts

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

// ── Duplicate Community Name Error Modal ──────────────────────────────────

function CommunityNameErrorModal({ name, onClose }: { name: string; onClose: () => void }) {
  const { t } = useLanguage();
  useEscapeKey(onClose);
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl border border-red-500/20 max-w-sm w-full p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-red-500/20 border border-red-500/30 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{t('settings.account.duplicateName.title')}</h3>
            <p className="text-sm text-gray-400 mt-1">{t('settings.account.duplicateName.subtitle')}</p>
          </div>
        </div>
        
        <div className="bg-red-500/8 border border-red-500/20 rounded-lg p-3">
          <p className="text-sm text-red-200">
            {t('settings.account.duplicateName.message').replace('{name}', name)}
          </p>
        </div>

        <div className="bg-violet-500/8 border border-violet-500/20 rounded-lg p-3">
          <p className="text-xs text-violet-200">
            <span className="inline-flex items-center gap-2">
              <Lightbulb className="w-3.5 h-3.5" />
              <span>{t('settings.account.duplicateName.tip')}</span>
            </span>
          </p>
        </div>

        <button
          onClick={onClose}
          className="w-full py-2.5 rounded-lg bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/30 text-violet-300 font-medium transition-all"
        >
          {t('settings.account.duplicateName.retry')}
        </button>
      </div>
    </div>
  );
}

export default function Settings({ userData, onLogout }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('account');
  const [showApiKey, setShowApiKey] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);
  const { currentTheme, setTheme, themes: allThemes } = useTheme();
  const { language, setLanguage, t } = useLanguage();
  // Status helpers for Support tickets
  const STATUS_LABEL: Record<string, string> = {
    open: t('settings.support.statusOpen'),
    in_progress: t('settings.support.statusInProgress'),
    resolved: t('settings.support.statusResolved'),
    closed: t('settings.support.statusClosed'),
  };
  const { settings: aiSettings, updateSettings: updateAISettings } = useAISettings();
  const { setCurrentPageContent } = usePageContext();
  const [appVersion, setAppVersion] = useState<string>('Loading...');
  const [latestVersion, setLatestVersion] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<'checking' | 'up-to-date' | 'update-available' | 'error'>('checking');
  const [checkingUpdates, setCheckingUpdates] = useState(false);
  // Community-Name
  const [communityName, setCommunityName]         = useState(() => getStoredAuthorName(userData.userId) || '');
  const [communityNameInput, setCommunityNameInput] = useState(() => getStoredAuthorName(userData.userId) || '');
  const [editingCommunity, setEditingCommunity]   = useState(false);
  const [communitySaved, setCommunitySaved]       = useState(false);
  const [savingCommunity, setSavingCommunity]     = useState(false);
  const [duplicateNameError, setDuplicateNameError] = useState<string | null>(null);

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
    setCurrentPageContent(buildPageContext({
      pageId: 'settings',
      // Der Titel landet im Kontext-Chip des Coaches. Fest verdrahtet stand
      // dort doppelt „Einstellungen (Settings)“ — jetzt in der UI-Sprache.
      title: t('settings.title'),
      purpose: 'Konfiguration der App: KI-Provider (dieser Coach), Erscheinungsbild, Account, Support, Updates, System.',
      state: [
        kv('Aktiver Tab', tabs[activeTab] || activeTab),
        kv('KI-Assistent', aiSettings.enabled ? 'aktiviert' : 'deaktiviert'),
        kv('Anbieter', `${providerInfo.label} (${aiSettings.provider})`),
        providerInfo.needsKey
          ? kv('API-Key', aiSettings.apiKey ? 'eingetragen' : `FEHLT! (${providerInfo.keyHint})`)
          : kv('API-Key', `nicht ben\u00f6tigt (${providerInfo.keyHint})`),
        kv('Modell', aiSettings.selectedModel || providerInfo.models[0]),
        aiSettings.provider === 'ollama' ? kv('Ollama-Modell', aiSettings.ollamaModel || 'llama3.2') : '',
        kv('App-Version', appVersion),
        updateStatus === 'update-available' ? kv('Update', `verf\u00fcgbar: ${latestVersion}`)
          : updateStatus === 'up-to-date' ? kv('Update', 'App ist aktuell') : '',
        kv('E-Mail', userData.email || 'nicht gesetzt'),
      ],
      actions: [
        'KI-Assistent aktivieren/deaktivieren (Toggle im KI-Assistent-Tab)',
        'Anbieter wechseln: KI-Assistent-Tab > Anbieter ausw\u00e4hlen',
        'API-Key eintragen: KI-Assistent-Tab > API-Key Feld',
        'Farbthema \u00e4ndern: Erscheinungsbild-Tab',
        'Support-Ticket erstellen: Support-Tab',
        'Abmelden: Account-Tab > Abmelden-Button',
      ],
      sections: [
        { heading: 'VERF\u00dcGBARE TABS', lines: Object.values(tabs) },
        {
          heading: 'ANBIETER-VERGLEICH',
          lines: [
            'Anthropic (Claude): bezahlt, bestes Modell, sk-ant-\u2026 Key',
            'OpenAI (GPT-4o): bezahlt, sehr gut, sk-\u2026 Key',
            'Groq: KOSTENLOS, schnell, gsk_\u2026 Key von console.groq.com',
            'Ollama: KOSTENLOS, lokal, kein Key \u2014 ollama.com installieren',
          ],
        },
      ],
    }), 'settings');
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

  // Support-Tab: Immer ausgeklappt halten wenn auf Support-Sektion
  useEffect(() => {
    if (activeTab === 'support') {
      setSupportOpen(true);
    }
  }, [activeTab]);

  useEffect(() => {
    if (!systemInstalling) return;

    let unlistenProgress: (() => void) | undefined;
    let unlistenComplete: (() => void) | undefined;

    (async () => {
      unlistenProgress = await listen<InstallProgress>('plugin-install-progress', (event) => {
        const progress = event.payload;
        setSystemInstallProgress(prev => new Map(prev).set(progress.plugin_id, progress));
        if (progress.status === 'failed') {
          setSystemInstallError(progress.message || t('settings.system.installFailed'));
          setSystemInstalling(false);
        }
      });

      unlistenComplete = await listen('plugin-install-complete', async () => {
        setSystemInstalling(false);
        setNotification({ type: 'success', message: t('settings.system.depsInstalled') });
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
      setNotification({ type: 'error', message: t('settings.support.fillRequired') });
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
      setNotification({ type: 'success', message: t('settings.support.submitSuccess') });
      setTimeout(() => setNotification(null), 3000);
    } catch {
      setNotification({ type: 'error', message: t('settings.support.submitError') });
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
      setNotification({ type: 'error', message: t('settings.support.sendFailed') });
      setReplyText(text);
    } finally {
      setSendingReply(false);
    }
  };

  const tabs = [
    { id: 'account'      as SettingsTab, label: t('settings.tabs.account'),              icon: User },
    { id: 'appearance'   as SettingsTab, label: t('settings.tabs.appearance'),         icon: Palette },
    { id: 'language'     as SettingsTab, label: t('settings.tabs.language'),             icon: Globe },
    { id: 'notifications'as SettingsTab, label: t('settings.tabs.notifications'),  icon: Bell },
    { id: 'ai-assistant' as SettingsTab, label: t('settings.tabs.aiAssistant'),        icon: Brain },
    { id: 'system'       as SettingsTab, label: t('settings.tabs.system'),              icon: Monitor },
    { id: 'updates'      as SettingsTab, label: t('settings.tabs.updates'),             icon: Download },
    { id: 'docs'         as SettingsTab, label: t('settings.tabs.docs'),       icon: BookOpen },
    { id: 'support'      as SettingsTab, label: t('settings.tabs.support'),             icon: MessageCircle },
    { id: 'about'        as SettingsTab, label: t('settings.tabs.about'),               icon: Info },
  ];

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setNotification({ type: 'success', message: t('settings.account.copied') });
      setTimeout(() => setNotification(null), 3000);
    } catch (error) {
      setNotification({ type: 'error', message: t('settings.account.copyFailed') });
      setTimeout(() => setNotification(null), 3000);
    }
  };

  const renderAccountTab = () => (
    <div className="space-y-6">
      {/* User Info Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.account.title')}</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">{t('settings.account.email')}</label>
            <div className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
              {userData.email}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">{t('settings.account.userId')}</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm truncate">
                {userData.userId}
              </div>
              <button
                onClick={() => copyToClipboard(userData.userId)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                {t('settings.account.copy')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* API Key Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">{t('settings.account.apiKeySection.title')}</h3>
          <Key className="w-5 h-5 text-purple-400" />
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">{t('settings.account.apiKeySection.label')}</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm">
                {showApiKey ? userData.apiKey : '••••••••••••••••••••'}
              </div>
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors border border-white/10"
              >
                {showApiKey ? t('settings.account.apiKeySection.hide') : t('settings.account.apiKeySection.show')}
              </button>
              <button
                onClick={() => copyToClipboard(userData.apiKey)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                {t('settings.account.copy')}
              </button>
            </div>
          </div>

          <div className="flex items-start space-x-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <Shield className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-yellow-300">
              {t('settings.account.apiKeySection.warning')}
            </p>
          </div>
        </div>
      </div>

      {/* Community Name Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Globe className="w-5 h-5 text-violet-400" />
            <h3 className="text-lg font-semibold text-white">{t('settings.account.communityName.title')}</h3>
          </div>
          {communityName && !editingCommunity && (
            <button onClick={() => { setEditingCommunity(true); setCommunitySaved(false); }} className="p-2 rounded-lg hover:bg-white/5 text-gray-500 hover:text-violet-400 transition-colors">
              <Pencil className="w-4 h-4" />
            </button>
          )}
        </div>
        <p className="text-sm text-gray-400 mb-4">{t('settings.account.communityName.subtitle')}</p>

        {communitySaved && (
          <div className="flex items-center gap-2 p-3 mb-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
            <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <p className="text-sm text-emerald-300">{t('settings.account.communityName.savedMsg').replace('{name}', communityName)}</p>
          </div>
        )}

        {!communityName || editingCommunity ? (
          <div className="space-y-3">
            <div className="flex items-start gap-2 p-3 bg-violet-500/8 border border-violet-500/20 rounded-lg">
              <Globe className="w-4 h-4 text-violet-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-violet-300">{!communityName ? t('settings.account.communityName.firstTime') : t('settings.account.communityName.change')}</p>
            </div>
            <div className="flex gap-2">
              <input
                value={communityNameInput}
                onChange={e => setCommunityNameInput(e.target.value.replace(/[^a-z0-9_\-. ]/gi, ''))}
                placeholder={t('settings.account.communityName.placeholder')}
                maxLength={40}
                className="flex-1 px-4 py-2.5 bg-white/5 border border-violet-500/30 rounded-lg text-white placeholder:text-gray-600 focus:outline-none focus:border-violet-500/60"
              />
              <button
                onClick={async () => {
                  if (!communityNameInput.trim() || savingCommunity) return;
                  
                  setSavingCommunity(true);
                  console.log('[Settings] Saving community name:', communityNameInput.trim());
                  
                  try {
                    // Prüfe auf Duplikate
                    try {
                      const res = await fetch(`https://frame-train.com/api/library/authors/${encodeURIComponent(communityNameInput.trim())}/exists`);
                      const data = await res.json();
                      if (data.exists && communityNameInput.trim() !== communityName) {
                        console.warn('[Settings] Community name already exists');
                        setDuplicateNameError(communityNameInput.trim());
                        setSavingCommunity(false);
                        return;
                      }
                    } catch (err) {
                      console.error('[Settings] Duplicate check failed:', err);
                    }
                    
                    // Update User.communityName in DB
                    const response = await fetch(`https://frame-train.com/api/user/community-name`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ 
                        userId: userData.userId, 
                        communityName: communityNameInput.trim() 
                      }),
                    });
                    
                    if (!response.ok) {
                      const error = await response.json();
                      console.error('[Settings] PATCH failed:', error);
                      alert(t('settings.account.communityName.saveError').replace('{error}', error.error));
                      setSavingCommunity(false);
                      return;
                    }
                    
                    const result = await response.json();
                    console.log('[Settings] PATCH succeeded:', result);
                    
                    // Speichere auch lokal
                    saveAuthorName(userData.userId, communityNameInput);
                    setCommunityName(communityNameInput.trim());
                    setEditingCommunity(false);
                    setCommunitySaved(true);
                    
                    // Show success message für 3 Sekunden
                    setTimeout(() => setCommunitySaved(false), 3000);
                  } catch (err) {
                    console.error('[Settings] Error saving community name:', err);
                    alert(t('settings.account.communityName.saveError').replace('{error}', String(err)));
                  } finally {
                    setSavingCommunity(false);
                  }
                }}
                disabled={!communityNameInput.trim() || savingCommunity}
                className="px-4 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:opacity-40 text-white rounded-lg transition-colors flex items-center gap-1.5 text-sm"
              >
                {savingCommunity ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> {t('settings.account.communityName.saving')}
                  </>
                ) : (
                  <>
                    <Check className="w-4 h-4" /> {t('settings.account.communityName.save')}
                  </>
                )}
              </button>
              {editingCommunity && (
                <button
                  onClick={() => { setEditingCommunity(false); setCommunityNameInput(communityName); }}
                  className="px-3 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <p className="text-[11px] text-gray-600">{t('settings.account.communityName.hint')}</p>
          </div>
        ) : (
          <div className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/10 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-violet-500/20 border border-violet-500/30 flex items-center justify-center">
              <User className="w-4 h-4 text-violet-300" />
            </div>
            <div>
              <p className="text-white font-medium">@{communityName}</p>
              <p className="text-gray-500 text-xs">{t('settings.account.communityName.publicLabel')}</p>
            </div>
          </div>
        )}
      </div>

      {/* Account Management */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.account.management.title')}</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.com/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>{t('settings.account.management.openDashboard')}</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.com/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>{t('settings.account.management.changePassword')}</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <button
            onClick={onLogout}
            className="w-full flex items-center justify-between px-4 py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-300 hover:text-red-200 transition-colors"
          >
            <span>{t('settings.account.management.logout')}</span>
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );

  const renderAIAssistantTab = () => {
    const providerIcon = (provider: AIProvider) => {
      const common = 'w-7 h-7';
      switch (provider) {
        case 'anthropic':
          return <Brain className={`${common} text-purple-300`} />;
        case 'openai':
          return <Sparkles className={`${common} text-emerald-300`} />;
        case 'groq':
          return <Zap className={`${common} text-amber-300`} />;
        case 'ollama':
          return <Monitor className={`${common} text-blue-300`} />;
        default:
          return <Sparkles className={`${common} text-gray-300`} />;
      }
    };

    const meta = PROVIDER_META[aiSettings.provider];
    return (
      <div className="space-y-6">
        {/* Enable/Disable Toggle */}
        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-white">{t('settings.ai.title')}</h3>
              <p className="text-sm text-gray-400 mt-1">{t('settings.ai.subtitle')}</p>
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
              {t('settings.ai.active')}
            </div>
          )}
          {!aiSettings.enabled && (
            <div className="p-3 bg-gray-500/10 border border-gray-500/20 rounded-lg text-sm text-gray-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {t('settings.ai.inactive')}
            </div>
          )}
        </div>

        {aiSettings.enabled && (
          <>
            {/* Provider Selection */}
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">{t('settings.ai.providerTitle')}</h3>
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
                    <span className="flex-shrink-0">{providerIcon(key)}</span>
                    <div className="min-w-0">
                      <div className="text-sm font-semibold">{m.labelKey ? t(m.labelKey, m.label) : m.label}</div>
                      <div className="text-xs opacity-60 mt-0.5 flex items-center gap-1.5">
                        {m.needsKey ? (
                          <>{t('settings.ai.keyNeeded')}</>
                        ) : (
                          <>
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                            {t('settings.ai.noKeyNeeded')}
                          </>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Provider-specific Configuration */}
            {meta.needsKey ? (
              <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">{t('settings.ai.apiKeyTitle')}</h3>
                  <a
                    href={meta.keyLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
                  >
                    {t('settings.ai.getKey')}
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
                      {showApiKeyField ? t('settings.ai.hide') : t('settings.ai.show')}
                    </button>
                  </div>
                </div>

                <p className="text-xs text-gray-500">{t('settings.ai.keyLocalOnly')}</p>

                {/* Model Selection */}
                <div>
                  <label className="block text-sm font-semibold text-white mb-2">{t('settings.ai.model')}</label>
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
                  <h3 className="text-lg font-semibold text-white">{t('settings.ai.ollamaTitle')}</h3>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">{t('settings.ai.ollamaModelLabel')}</label>
                  <input
                    type="text"
                    value={aiSettings.ollamaModel}
                    onChange={(e) => updateAISettings({ ollamaModel: e.target.value, selectedModel: e.target.value })}
                    placeholder="llama3.2"
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-white mb-2">{t('settings.ai.ollamaPopular')}</label>
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
                  <div className="font-semibold text-gray-300 flex items-center gap-2">
                    <Download className="w-4 h-4 text-blue-300" />
                    {t('settings.ai.ollamaInstallTitle')}
                  </div>
                  <div>{t('settings.ai.ollamaInstallStep1')} <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:underline">ollama.com</a>{t('settings.ai.ollamaInstallStep1Suffix')}</div>
                  <div>{t('settings.ai.ollamaInstallStep2')} <code className="bg-black/30 px-1 py-0.5 rounded font-mono text-xs">{t('settings.ai.ollamaInstallCommand')}</code></div>
                  <div>{t('settings.ai.ollamaInstallStep3')}</div>
                </div>
              </div>
            )}

            {/* Token Budget */}
            <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-1">{t('settings.ai.tokenBudget.title')}</h3>
                <p className="text-sm text-gray-400">{t('settings.ai.tokenBudget.subtitle')}</p>
              </div>

              {/* Budget Stufen */}
              <div className="grid grid-cols-2 gap-3">
                {(Object.entries(TOKEN_BUDGET_CONFIG) as [TokenBudget, typeof TOKEN_BUDGET_CONFIG[TokenBudget]][]).map(([key, cfg]) => {
                  const active = (aiSettings.tokenBudget ?? 'balanced') === key;
                  const colors: Record<TokenBudget, string> = {
                    minimal:  'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
                    balanced: 'border-blue-500/50 bg-blue-500/10 text-blue-300',
                    quality:  'border-purple-500/50 bg-purple-500/10 text-purple-300',
                    max:      'border-amber-500/50 bg-amber-500/10 text-amber-300',
                  };
                  const iconComponents: Record<TokenBudget, React.ReactNode> = {
                    minimal:  <Leaf      className="w-4 h-4 text-emerald-400" />,
                    balanced: <Scale     className="w-4 h-4 text-blue-400" />,
                    quality:  <Star      className="w-4 h-4 text-purple-400" />,
                    max:      <Flame     className="w-4 h-4 text-amber-400" />,
                  };
                  return (
                    <button
                      key={key}
                      onClick={() => updateAISettings({ tokenBudget: key })}
                      className={`flex flex-col gap-2 p-4 rounded-xl border-2 text-left transition-all ${
                        active ? colors[key] : 'border-white/10 bg-white/[0.03] text-gray-400 hover:bg-white/5 hover:border-white/20'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{iconComponents[key]}</span>
                        {active && <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-white/20">{t('settings.ai.tokenBudget.active')}</span>}
                      </div>
                      <div>
                        <div className="font-semibold text-sm">{cfg.label}</div>
                        <div className="text-xs opacity-70 mt-0.5 leading-relaxed">{t(`settings.ai.tokenBudget.${key}Desc`)}</div>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Live-Vorschau */}
              {(() => {
                const budget = TOKEN_BUDGET_CONFIG[aiSettings.tokenBudget ?? 'balanced'];
                const provMeta = PROVIDER_META[aiSettings.provider];

                // Kosten-Schätzung pro Nachricht (Input + Output)
                // Grobe Durchschnittswerte ($ per 1M tokens)
                const pricing: Record<string, { input: number; output: number }> = {
                  'claude-opus-4-5':         { input: 15,   output: 75 },
                  'claude-sonnet-4-5':       { input: 3,    output: 15 },
                  'claude-haiku-4-5':        { input: 0.8,  output: 4  },
                  'gpt-4o':                  { input: 2.5,  output: 10 },
                  'gpt-4o-mini':             { input: 0.15, output: 0.6 },
                  'llama-3.3-70b-versatile': { input: 0,    output: 0  }, // Groq Free
                  'llama-3.1-8b-instant':    { input: 0,    output: 0  }, // Groq Free
                  'ollama':                  { input: 0,    output: 0  },
                };
                const model = aiSettings.provider === 'ollama'
                  ? 'ollama'
                  : (aiSettings.selectedModel || provMeta.models[0]);
                const p = pricing[model] ?? { input: 1, output: 5 };

                // Geschätzte Input-Tokens pro Nachricht: System (~300) + History-Budget + User-Message (~80)
                const estimatedInput  = 300 + budget.historyTokenBudget + 80;
                const estimatedOutput = budget.maxTokens * 0.7; // ~70% Ausnutzung
                const costPerMsg = ((estimatedInput * p.input) + (estimatedOutput * p.output)) / 1_000_000;

                // Groq Rate-Limit-Indikator (6000 TPM Free Tier)
                const isGroq = aiSettings.provider === 'groq';
                const groqTpm = estimatedInput + estimatedOutput;
                const groqMsgsPerMinute = Math.floor(6000 / groqTpm);

                const budgetBarWidth: Record<TokenBudget, string> = {
                  minimal: 'w-1/4', balanced: 'w-2/4', quality: 'w-3/4', max: 'w-full'
                };
                const budgetBarColor: Record<TokenBudget, string> = {
                  minimal: 'bg-emerald-400', balanced: 'bg-blue-400',
                  quality: 'bg-purple-400',  max: 'bg-amber-400',
                };

                return (
                  <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-3">
                    <p className="text-xs font-semibold text-gray-300 uppercase tracking-wider">{t('settings.ai.tokenBudget.previewTitle')}</p>

                    {/* Token-Balken */}
                    <div className="space-y-2">
                      {([
                        [t('settings.ai.tokenBudget.previewCoach'),   budget.maxTokens,        1500],
                        [t('settings.ai.tokenBudget.previewSynapse'), budget.synapseMaxTokens, 8000],
                        [t('settings.ai.tokenBudget.previewHistory'), budget.historyTokenBudget, 4000],
                      ] as [string, number, number][]).map(([label, val, max]) => (
                        <div key={label}>
                          <div className="flex justify-between text-xs text-gray-400 mb-1">
                            <span>{label}</span>
                            <span className="font-mono text-gray-300">{val.toLocaleString()} tokens</span>
                          </div>
                          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${budgetBarColor[aiSettings.tokenBudget ?? 'balanced']}`}
                              style={{ width: `${Math.min((val / max) * 100, 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Kosten + Rate-Limit */}
                    <div className="grid grid-cols-2 gap-2 pt-1">
                      <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewCostLabel')}</p>
                        <p className="text-sm font-semibold text-white">
                          {costPerMsg < 0.0001
                            ? t('settings.ai.tokenBudget.previewCostFree')
                            : `~${costPerMsg.toFixed(4)}`}
                        </p>
                        <p className="text-[10px] text-gray-500 mt-0.5">{t('settings.ai.tokenBudget.previewCostPerMsg')}</p>
                      </div>
                      {isGroq ? (
                        <div className={`rounded-lg p-3 ${
                          groqMsgsPerMinute >= 3 ? 'bg-emerald-500/10' : groqMsgsPerMinute >= 1 ? 'bg-amber-500/10' : 'bg-red-500/10'
                        }`}>
                          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewGroqLimit')}</p>
                          <p className={`text-sm font-semibold ${
                            groqMsgsPerMinute >= 3 ? 'text-emerald-300' : groqMsgsPerMinute >= 1 ? 'text-amber-300' : 'text-red-300'
                          }`}>
                            ~{groqMsgsPerMinute} {t('settings.ai.tokenBudget.previewGroqMsgs')}
                          </p>
                          <p className="text-[10px] text-gray-500 mt-0.5">{t('settings.ai.tokenBudget.previewGroqPerMin')}</p>
                        </div>
                      ) : (
                        <div className="bg-white/5 rounded-lg p-3">
                          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewQuality')}</p>
                          <p className="text-sm font-semibold text-white">
                            {(aiSettings.tokenBudget ?? 'balanced') === 'minimal'  ? t('settings.ai.tokenBudget.qualityConcise') :
                             (aiSettings.tokenBudget ?? 'balanced') === 'balanced' ? t('settings.ai.tokenBudget.qualityBalanced') :
                             (aiSettings.tokenBudget ?? 'balanced') === 'quality'  ? t('settings.ai.tokenBudget.qualityDetailed') :
                                                                                     t('settings.ai.tokenBudget.qualityMax')}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Groq-Warning */}
                    {isGroq && (aiSettings.tokenBudget === 'quality' || aiSettings.tokenBudget === 'max') && (
                      <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                        <p className="text-xs text-amber-300">{t('settings.ai.tokenBudget.groqWarning')}</p>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>

            {/* Info Box */}
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-sm text-blue-300 flex items-start gap-3">
              <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold mb-1">{t('settings.ai.globalNote')}</div>
                <p className="text-xs text-blue-200">{t('settings.ai.globalNoteDesc')}</p>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  const handleThemeChange = async (themeId: ThemeId) => {
    setTheme(themeId);
    setNotification({ type: 'success', message: t('settings.appearance.themeChanged') });
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
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.appearance.title')}</h3>
        <p className="text-gray-400 mb-6">{t('settings.appearance.subtitle')}</p>
        
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
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.appearance.preview')}</h3>
        <div className={`p-6 bg-gradient-to-br ${currentTheme.colors.background} rounded-xl border border-white/10`}>
          <div className="flex items-center space-x-4 mb-4">
            <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${currentTheme.colors.gradient} flex items-center justify-center`}>
              <Palette className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-white font-semibold">{t('settings.appearance.exampleButton')}</div>
              <div className="text-gray-400 text-sm">{t('settings.appearance.exampleSubtitle')}</div>
            </div>
          </div>
          <button className={`w-full py-3 px-4 bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold rounded-lg hover:opacity-90 transition-opacity`}>
            {t('settings.appearance.exampleButton')}
          </button>
        </div>
      </div>
    </div>
    );
  };

  const renderLanguageTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center gap-3 mb-2">
          <Globe className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">{t('settings.language.title')}</h3>
        </div>
        <p className="text-sm text-gray-400 mb-6">
          {t('settings.language.subtitle')}
        </p>

        <div className="space-y-3 max-w-sm">
          {(Object.entries(LANGUAGE_META) as [Language, typeof LANGUAGE_META[Language]][]).map(
            ([lang, meta]) => {
              const active = language === lang;
              return (
                <button
                  key={lang}
                  onClick={() => {
                    setLanguage(lang);
                    setNotification({ type: 'success', message: t('settings.language.changed').replace('{lang}', meta.nativeLabel) });
                    setTimeout(() => setNotification(null), 2500);
                  }}
                  className={`w-full flex items-center gap-4 px-5 py-4 rounded-2xl border-2 transition-all duration-200 ${
                    active
                      ? `bg-gradient-to-r ${currentTheme.colors.gradient} border-transparent shadow-lg scale-[1.01]`
                      : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                  }`}
                >
                  <span className="w-10 h-10 flex items-center justify-center rounded-lg bg-white/10 border border-white/15 text-sm font-semibold tracking-wide text-white">
                    {meta.code}
                  </span>
                  <div className="flex-1 text-left">
                    <div className="text-white font-semibold">{meta.nativeLabel}</div>
                    <div className={`text-xs mt-0.5 ${active ? 'text-white/70' : 'text-gray-500'}`}>
                      {lang === 'de' ? 'Deutsch' : 'English'}
                    </div>
                  </div>
                  {active && (
                    <div className="w-6 h-6 rounded-full bg-white/25 flex items-center justify-center flex-shrink-0">
                      <Check className="w-4 h-4 text-white" strokeWidth={3} />
                    </div>
                  )}
                </button>
              );
            },
          )}
        </div>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-5 flex items-start gap-3">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-blue-300 mb-1">{t('settings.language.moreComingSoon')}</p>
          <p className="text-xs text-blue-200/70">
            {t('settings.language.moreComingSoonDesc')}
          </p>
        </div>
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.notifications.title')}</h3>
        
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">{t('settings.notifications.trainingComplete')}</div>
              <div className="text-sm text-gray-400">{t('settings.notifications.trainingCompleteDesc')}</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">{t('settings.notifications.errors')}</div>
              <div className="text-sm text-gray-400">{t('settings.notifications.errorsDesc')}</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">{t('settings.notifications.updates')}</div>
              <div className="text-sm text-gray-400">{t('settings.notifications.updatesDesc')}</div>
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
              {updateStatus === 'checking' && t('settings.updates.checking')}
              {updateStatus === 'up-to-date' && t('settings.updates.upToDate')}
              {updateStatus === 'update-available' && t('settings.updates.updateAvailable')}
              {updateStatus === 'error' && t('settings.updates.error')}
            </h3>
          </div>
          <button
            onClick={handleCheckUpdates}
            disabled={checkingUpdates}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg text-sm font-semibold transition-all"
          >
            {checkingUpdates ? t('settings.updates.rechecking') : t('settings.updates.recheck')}
          </button>
        </div>

        {/* Version Comparison */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="bg-black/20 rounded-lg p-4">
            <p className="text-gray-400 text-xs mb-1">{t('settings.updates.installed')}</p>
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
            <p className="text-gray-400 text-xs mb-1">{t('settings.updates.available')}</p>
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
              {t('settings.updates.updateMsg')}
            </p>
          </div>
        )}

        {updateStatus === 'error' && (
          <div className="bg-gray-500/20 border border-gray-500/30 rounded-lg p-4 mb-4">
            <p className="text-gray-300 text-sm">
              {t('settings.updates.errorMsg')}
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
            <h3 className="text-xl font-bold text-white mb-2">{t('settings.updates.downloadTitle')}</h3>
            <p className="text-gray-300 mb-4">
              {t('settings.updates.downloadDesc')}
            </p>
            
            <button
              onClick={handleOpenGitHub}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-semibold transition-all"
            >
              <Download className="w-5 h-5" />
              <span>{t('settings.updates.githubReleases')}</span>
              <ExternalLink className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Installation Instructions */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-5 h-5 text-gray-400" />
          <h3 className="text-lg font-semibold text-white">{t('settings.updates.installTitle')}</h3>
        </div>
        <div className="space-y-3 text-gray-400 text-sm">
          <p>
            <span className="font-semibold text-white">1.</span> {t('settings.updates.step1')}
          </p>
          <p>
            <span className="font-semibold text-white">2.</span> {t('settings.updates.step2')}
          </p>
          <ul className="ml-6 space-y-1 list-disc">
            <li>{t('settings.updates.uninstallMac').replace('{app}', 'FrameTrain')}</li>
            <li>{t('settings.updates.uninstallWindows')}</li>
            <li>{t('settings.updates.uninstallLinux')}</li>
          </ul>
          <p>
            <span className="font-semibold text-white">3.</span> {t('settings.updates.step3')}
          </p>
          <p>
            <span className="font-semibold text-white">4.</span> {t('settings.updates.step4')}
          </p>
        </div>
      </div>

      {/* Auto-Update Info */}
      <div className="bg-blue-500/10 rounded-xl p-6 border border-blue-500/20">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-white font-semibold mb-1">{t('settings.updates.autoUpdateTitle')}</h3>
            <p className="text-blue-300 text-sm">
              {t('settings.updates.autoUpdateDesc')}
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
          <h3 className="text-lg font-semibold text-white">{t('settings.docs.title')}</h3>
        </div>
        <p className="text-sm text-gray-300">
          {t('settings.docs.subtitle')}
        </p>
      </div>

      {/* Main Docs Link */}
      <a
        href="https://frame-train.com/docs"
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/30 rounded-xl p-6 transition-all hover:shadow-lg"
      >
        <div className="flex items-start justify-between">
          <div>
            <h4 className="text-base font-semibold text-white mb-2">{t('settings.docs.fullDocsTitle')}</h4>
            <p className="text-sm text-gray-400">
              {t('settings.docs.fullDocsDesc')}
            </p>
          </div>
          <ExternalLink className="w-5 h-5 text-blue-400 flex-shrink-0 mt-1" />
        </div>
      </a>

      {/* AI Training Guide */}
      <a
        href="https://frame-train.com/docs/ai-training-guide"
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/30 rounded-xl p-6 transition-all hover:shadow-lg"
      >
        <div className="flex items-start justify-between">
          <div>
            <h4 className="text-base font-semibold text-white mb-2">{t('settings.docs.guideTitle')}</h4>
            <p className="text-sm text-gray-400">
              {t('settings.docs.guideDesc')}
            </p>
          </div>
          <ExternalLink className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-1" />
        </div>
      </a>

      {/* Quick Tips Card */}
      <div className="bg-white/5 rounded-xl border border-white/10 p-6">
        <h4 className="text-base font-semibold text-white mb-4">{t('settings.docs.tipsTitle')}</h4>
        <ul className="space-y-3">
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">{t('settings.docs.tip1')}</span>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">{t('settings.docs.tip2')}</span>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-2 flex-shrink-0" />
            <span className="text-sm text-gray-300">{t('settings.docs.tip3')}</span>
          </li>
        </ul>
      </div>

      {/* Support Info */}
      <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl p-6">
        <div className="flex items-center justify-center gap-2 mb-2">
          <BookOpen className="w-4 h-4 text-blue-400" />
          <p className="text-sm text-gray-400 text-center">
            {t('settings.docs.footerNote')}
          </p>
        </div>
        <p className="text-sm text-gray-400 text-center">
          {t('settings.docs.footerQuestion')}
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
            // Wenn im Support-Tab: nicht zuklappbar, nur öffnen wenn tickets geladen
            if (activeTab === 'support') {
              getAll().forEach((t) => {
                localStorage.setItem(`ft_ticket_seen_${t.ticket_id}`, Date.now().toString());
              });
              setSupportBadge(0);
              return;
            }
            // Sonst: normales Toggle
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
            <h2 className="text-2xl font-bold text-white">{t('settings.support.title')}</h2>
            {storedTickets.length > 0 && (
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {storedTickets.length === 1
                  ? t('settings.support.ticketCount').replace('{count}', String(storedTickets.length))
                  : t('settings.support.ticketCountPlural').replace('{count}', String(storedTickets.length))
                }
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
                { id: 'list' as const, label: t('settings.support.myTickets'), icon: Inbox },
                { id: 'new' as const, label: t('settings.support.newTicket'), icon: Edit },
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
                  <h3 className="text-lg font-bold text-white mb-6">{t('settings.support.newTicketTitle')}</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">{t('settings.support.subject')}</label>
                      <input
                        value={newSubject}
                        onChange={(e) => setNewSubject(e.target.value)}
                        placeholder={t('settings.support.subjectPlaceholder')}
                        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">{t('settings.support.message')}</label>
                      <textarea
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        placeholder={t('settings.support.messagePlaceholder')}
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
                        {submitting ? t('settings.support.submitting') : t('settings.support.submit')}
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
                      <p className="text-gray-400 mb-2">{t('settings.support.noTickets')}</p>
                      <p className="text-gray-500 text-sm mb-6">{t('settings.support.noTicketsHint')}</p>
                      <button
                        onClick={() => setSupportView('new')}
                        className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors mx-auto text-sm font-semibold"
                      >
                        <Plus className="w-4 h-4" /> {t('settings.support.createFirst')}
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold text-white">{t('settings.support.yourTickets')}</h3>
                        <button
                          onClick={() => setSupportView('new')}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded-lg text-sm transition-colors border border-purple-500/20"
                        >
                          <Plus className="w-3.5 h-3.5" /> {t('settings.support.newTicket')}
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
                    {t('settings.support.back')}
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
                        {messages.length === 0 && <p className="text-center text-gray-500 text-sm py-8">{t('settings.support.noMessages')}</p>}
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
                                {m.sender === 'user' ? t('settings.support.sender') : <>
                                  <Wrench className="w-3 h-3" />
                                  {t('settings.support.senderSupport')}
                                </>} · {new Date(m.created_at).toLocaleString(dateLocale(language), { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}
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
                            placeholder={t('settings.support.sendPlaceholder')}
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
                            {t('settings.support.ticketClosed').replace(
                              '{status}',
                              ({
                                open: t('settings.support.statusOpen'),
                                in_progress: t('settings.support.statusInProgress'),
                                resolved: t('settings.support.statusResolved'),
                                closed: t('settings.support.statusClosed'),
                              }[ticketInfo.status] ?? '').toLowerCase()
                            )
                          }
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
          <h3 className="text-lg font-semibold text-white">{t('settings.system.packagesTitle')}</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={installMissingDeps}
              disabled={systemLoading || systemInstalling || !(systemDeps?.some(d => !d.installed))}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-lg text-emerald-300 hover:text-emerald-200 text-sm transition-all disabled:opacity-50 disabled:hover:bg-emerald-500/10"
              title={t('settings.system.installMissingTooltip')}
            >
              {systemInstalling ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              {t('settings.system.installMissing')}
            </button>
            <button
              onClick={loadSystemInfo}
              disabled={systemLoading || systemInstalling}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-400 hover:text-white text-sm transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${systemLoading ? 'animate-spin' : ''}`} />
              {t('settings.system.recheck')}
            </button>
          </div>
        </div>

        {!systemDeps ? (
          <div className="text-gray-500 text-sm">{t('settings.system.statusNotLoaded')}</div>
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
                  {dep.installed ? (dep.version ?? t('settings.system.installed')) : t('settings.system.missing')}
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
                {t('settings.system.installRunning')}
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
              {systemInstallProgress.get('seq_classification')?.message ?? t('settings.system.pipStarting')}
            </div>
          </div>
        )}
      </div>

      {systemReqs && (
        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">{t('settings.system.hardwareTitle')}</h3>
          <div className="space-y-2">
            {[
              { label: 'Python', value: systemReqs.python_version, ok: systemReqs.python_installed },
              { label: 'PyTorch', value: systemReqs.torch_version, ok: systemReqs.torch_installed },
              { label: 'Transformers', value: systemReqs.transformers_version, ok: systemReqs.transformers_installed },
              { label: 'PEFT (LoRA)', value: systemReqs.peft_version, ok: systemReqs.peft_installed },
              { label: 'CUDA (GPU)', value: systemReqs.cuda_available ? t('settings.system.cudaAvailable') : t('settings.system.cudaUnavailable'), ok: systemReqs.cuda_available },
              { label: 'Apple MPS', value: systemReqs.mps_available ? t('settings.system.cudaAvailable') : t('settings.system.cudaUnavailable'), ok: systemReqs.mps_available },
            ].map(row => (
              <div key={row.label} className="flex items-center justify-between px-4 py-3 bg-white/[0.03] rounded-xl border border-white/10">
                <span className="text-gray-400 text-sm">{row.label}</span>
                <span className={`text-xs font-mono ${row.ok ? 'text-emerald-400' : 'text-gray-500'}`}>{row.value}</span>
              </div>
            ))}
          </div>
          <div className={`mt-4 p-3 rounded-xl border text-sm flex items-center gap-2 ${systemReqs.ready ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300' : 'bg-red-500/10 border-red-500/20 text-red-300'}`}>
            {systemReqs.ready ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            {systemReqs.ready ? t('settings.system.systemReady') : t('settings.system.systemNotReady')}
          </div>
        </div>
      )}

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.system.antiSleepTitle')}</h3>
        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${preventSleepActive ? 'bg-blue-500/10 border-blue-500/20' : 'bg-white/[0.03] border-white/10'}`}>
          <div className={`w-2.5 h-2.5 rounded-full ${preventSleepActive ? 'bg-blue-400 animate-pulse' : 'bg-gray-600'}`} />
          <span className={`text-sm ${preventSleepActive ? 'text-blue-300' : 'text-gray-500'}`}>
            {preventSleepActive ? t('settings.system.antiSleepActive') : t('settings.system.antiSleepInactive')}
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
        <h3 className="text-2xl font-bold text-white mb-2">{t('settings.about.desktopTitle')}</h3>
        <p className="text-gray-400 mb-4">{t('settings.about.versionLabel').replace('{version}', appVersion)}</p>
        <p className="text-sm text-gray-400 max-w-md mx-auto">
          {t('settings.about.subtitle')}
        </p>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-2">{t('settings.about.supportedModels')}</h3>
        <p className="text-sm text-gray-400 mb-4">
          {t('settings.about.supportedModelsDesc')}
        </p>
        <div className="flex flex-wrap gap-2">
          {HF_ENCODER_SUPPORTED_MODEL_TYPES.map((t) => (
            <span
              key={t}
              className="px-2.5 py-1 rounded-full text-xs font-mono bg-white/5 border border-white/10 text-gray-200"
              title={`HuggingFace model_type: ${t}`}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">{t('settings.about.links')}</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>{t('settings.about.website')}</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.com/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>{t('settings.about.docs')}</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://github.com/FrameSphere/FrameTrain/releases"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>{t('settings.about.github')}</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>
        </div>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <p className="text-sm text-gray-400 text-center">
          {t('settings.about.copyright')}
        </p>
      </div>
    </div>
  );

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">{t('settings.title')}</h2>
        <p className="text-gray-400">{t('settings.subtitle')}</p>
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
          {activeTab === 'account'       && renderAccountTab()}
          {activeTab === 'appearance'     && renderAppearanceTab()}
          {activeTab === 'language'       && renderLanguageTab()}
          {activeTab === 'notifications'  && renderNotificationsTab()}
          {activeTab === 'ai-assistant'   && renderAIAssistantTab()}
          {activeTab === 'system'         && renderSystemTab()}
          {activeTab === 'updates'        && renderUpdatesTab()}
          {activeTab === 'docs'           && renderDocsTab()}
          {activeTab === 'support'        && renderSupportTab()}
          {activeTab === 'about'          && renderAboutTab()}
        </div>
      </div>

      {/* Community Name Error Modal */}
      {duplicateNameError && (
        <CommunityNameErrorModal 
          name={duplicateNameError} 
          onClose={() => setDuplicateNameError(null)} 
        />
      )}
    </div>
  );
}
