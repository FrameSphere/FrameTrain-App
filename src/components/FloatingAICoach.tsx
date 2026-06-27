import { useState, useRef, useEffect, useCallback } from 'react';
import {
  X, Send, Loader2, AlertCircle, CheckCircle, Maximize2, Minimize2,
  MessageSquare, Plus, Trash2, ChevronDown, ChevronRight, Brain,
  FileSearch, Cpu, Sparkles, ArrowLeft
} from 'lucide-react';
import { useAISettings, type AIProvider } from '../contexts/AISettingsContext';
import { useTheme } from '../contexts/ThemeContext';
import { usePageContext } from '../contexts/PageContext';
import { useLanguage } from '../contexts/LanguageContext';
import { callAI as callAIClient } from '../ai/aiClient';
import { PROVIDER_META } from '../ai/providerMeta';
import { onOpenAICoach } from '../ai/aiCoachEvents';

// ============ Types ============

interface ThinkingStep {
  id: string;
  label: string;
  detail?: string;
  icon: 'search' | 'brain' | 'cpu' | 'sparkles' | 'check' | 'error';
  status: 'pending' | 'active' | 'done' | 'error';
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  thinkingSteps?: ThinkingStep[];
  thinkingCollapsed?: boolean;
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

// PROVIDER_META kommt aus src/ai/providerMeta.ts (zentral für die App)

const STORAGE_KEY = 'ft_ai_chats_v2';
const MAX_CHATS = 50;

// ============ Helpers ============

function createFreshChat(title: string): Chat {
  return {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title,
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function generateTitle(firstMessage: string): string {
  return firstMessage.slice(0, 40) + (firstMessage.length > 40 ? '…' : '');
}

type TFn = (key: string, fallback?: string) => string;
function formatRelativeTime(ts: number, t: TFn): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (mins < 1) return t('aiCoach.justNow');
  if (mins < 60) return t('aiCoach.minutesAgo').replace('{n}', String(mins));
  if (hours < 24) return t('aiCoach.hoursAgo').replace('{n}', String(hours));
  return (days === 1 ? t('aiCoach.daysAgo') : t('aiCoach.daysAgoPlural')).replace('{n}', String(days));
}

function formatPageContextTitle(pageContent: string): string {
  const firstLine = pageContent.split('\n').find(line => line.trim())?.trim() ?? '';
  const clean = firstLine
    .replace(/^=+\s*/, '')
    .replace(/\s*=+$/, '')
    .replace(/^FrameTrain\s+/i, '')
    .trim();
  return clean || '–';
}

// ============ Markdown Renderer ============

function renderInline(str: string, key?: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match;
  let i = 0;

  while ((match = regex.exec(str)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`t${i++}`}>{str.slice(lastIndex, match.index)}</span>);
    }
    if (match[0].startsWith('**')) {
      parts.push(<strong key={`b${i++}`} className="font-semibold text-white">{match[2]}</strong>);
    } else if (match[0].startsWith('*')) {
      parts.push(<em key={`em${i++}`} className="italic">{match[3]}</em>);
    } else if (match[0].startsWith('`')) {
      parts.push(
        <code key={`c${i++}`} className="px-1.5 py-0.5 bg-white/10 rounded text-[11px] font-mono text-purple-300">
          {match[4]}
        </code>
      );
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < str.length) {
    parts.push(<span key={`t${i++}`}>{str.slice(lastIndex)}</span>);
  }
  return parts.length > 0 ? parts : str;
}

function MarkdownText({ text, className = '' }: { text: string; className?: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      elements.push(<div key={i} className="h-1.5" />);
      i++;
      continue;
    }

    if (trimmed.match(/^[-*•]\s/)) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^[-*•]\s/)) {
        items.push(lines[i].trim().replace(/^[-*•]\s+/, ''));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 mt-0.5 flex-shrink-0 text-xs">•</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (trimmed.match(/^\d+\.\s/)) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^\d+\.\s/)) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${i}`} className="space-y-1 my-1.5">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2">
              <span className="text-purple-400 flex-shrink-0 font-medium text-xs w-4">{j + 1}.</span>
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    if (trimmed.startsWith('###')) {
      elements.push(
        <div key={i} className="font-semibold text-white mt-2 mb-1 text-sm">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else if (trimmed.startsWith('##')) {
      elements.push(
        <div key={i} className="font-bold text-white mt-2 mb-1">
          {renderInline(trimmed.replace(/^#+\s*/, ''))}
        </div>
      );
    } else {
      elements.push(
        <p key={i} className="leading-relaxed">
          {renderInline(line)}
        </p>
      );
    }
    i++;
  }

  return <div className={`text-sm space-y-0.5 ${className}`}>{elements}</div>;
}

// ============ Light-Color Detection (Fix für Monochrome / Arctic White) ============

function hexLuminance(hex: string): number {
  try {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const toL = (c: number) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    return 0.2126 * toL(r) + 0.7152 * toL(g) + 0.0722 * toL(b);
  } catch { return 0.5; }
}

function darkenHex(hex: string, amount: number): string {
  try {
    const r = Math.max(0, parseInt(hex.slice(1, 3), 16) - amount);
    const g = Math.max(0, parseInt(hex.slice(3, 5), 16) - amount);
    const b = Math.max(0, parseInt(hex.slice(5, 7), 16) - amount);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  } catch { return hex; }
}

const COACH_MAGIC_CSS = `
  @keyframes ftCoachButtonFloat { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-3px) scale(1.02); } }
  @keyframes ftCoachPanelIn { from { opacity:0; transform:translate3d(10px,14px,0) scale(.975); filter:blur(8px); } to { opacity:1; transform:none; filter:blur(0); } }
  @keyframes ftCoachAuraSweep { 0%,100% { opacity:.28; transform:translateX(-32%); } 50% { opacity:.82; transform:translateX(32%); } }
  @keyframes ftCoachMessageIn { from { opacity:0; transform:translateY(8px) scale(.985); } to { opacity:1; transform:none; } }
  @keyframes ftCoachThinkingSweep { from { transform:translateX(-80%); opacity:.18; } 50% { opacity:.72; } to { transform:translateX(180%); opacity:.18; } }
  @keyframes ftCoachSoftPulse { 0%,100% { box-shadow:0 0 0 rgba(255,255,255,0); } 50% { box-shadow:0 0 26px rgba(255,255,255,.12); } }
  .ft-coach-button { animation:ftCoachButtonFloat 4.5s ease-in-out infinite; }
  .ft-coach-shell {
    animation:ftCoachPanelIn .26s cubic-bezier(.2,.85,.22,1);
    backdrop-filter:blur(22px) saturate(1.28);
    -webkit-backdrop-filter:blur(22px) saturate(1.28);
  }
  .ft-coach-shell::before {
    content:"";
    position:absolute;
    inset:0;
    pointer-events:none;
    border-radius:16px;
    background:
      radial-gradient(circle at 18% 4%, rgba(255,255,255,.10), transparent 28%),
      radial-gradient(circle at 88% 10%, rgba(255,255,255,.08), transparent 30%);
    opacity:.75;
  }
  .ft-coach-shell::after {
    content:"";
    position:absolute;
    top:0;
    left:14px;
    right:14px;
    height:1px;
    pointer-events:none;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
    animation:ftCoachAuraSweep 5.2s ease-in-out infinite;
  }
  .ft-coach-message { animation:ftCoachMessageIn .22s ease both; }
  .ft-coach-scroll::-webkit-scrollbar { width:6px; }
  .ft-coach-scroll::-webkit-scrollbar-track { background:transparent; }
  .ft-coach-scroll::-webkit-scrollbar-thumb { background:rgba(148,163,184,.32); border-radius:999px; }
  .ft-coach-thinking-card { position:relative; overflow:hidden; }
  .ft-coach-thinking-card::before {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent);
    animation:ftCoachThinkingSweep 1.7s ease-in-out infinite;
  }
`;

// ============ Thinking Block ============

function ThinkingBlock({
  steps,
  isActive,
  collapsed,
  onToggle,
}: {
  steps: ThinkingStep[];
  isActive: boolean;
  collapsed: boolean;
  onToggle: () => void;
}) {
  // Fix 4: Theme-Farben im ThinkingBlock
  const { currentTheme } = useTheme();
  const { t } = useLanguage();
  const tPrimary = hexLuminance(currentTheme.colors.primary) > 0.5
    ? darkenHex(currentTheme.colors.primary, 100)
    : currentTheme.colors.primary;
  const borderColor = `${tPrimary}33`;
  const bgColor     = `${tPrimary}1a`;
  const bgHover     = `${tPrimary}26`;
  const textColor   = `${tPrimary}cc`;
  const chevronColor = `${tPrimary}99`;

  const iconMap: Record<ThinkingStep['icon'], React.ReactNode> = {
    search: <FileSearch className="w-3 h-3" />,
    brain: <Brain className="w-3 h-3" />,
    cpu: <Cpu className="w-3 h-3" />,
    sparkles: <Sparkles className="w-3 h-3" />,
    check: <CheckCircle className="w-3 h-3 text-green-400" />,
    error: <AlertCircle className="w-3 h-3 text-red-400" />,
  };

  const activeStep = steps.find(s => s.status === 'active');
  const doneCount = steps.filter(s => s.status === 'done').length;

  if (isActive) {
    return (
      <div className="ft-coach-thinking-card mb-3 rounded-2xl border border-white/10 bg-white/[0.045] p-2.5">
        <div className="flex items-center gap-1.5 text-xs mb-1.5" style={{ color: textColor }}>
          <Loader2 className="w-3 h-3 animate-spin" style={{ color: tPrimary }} />
          <span>{activeStep?.label || t('aiCoach.thinking')}</span>
        </div>
        <div className="pl-1 space-y-1 ml-1.5" style={{ borderLeft: `2px solid ${borderColor}` }}>
          {steps.map(step => (
            <div
              key={step.id}
              className={`flex items-start gap-2 py-0.5 transition-all ${
                step.status === 'active' ? 'opacity-100' : 'opacity-60'
              }`}
            >
              <div className={`flex-shrink-0 mt-0.5 ${
                step.status === 'done' ? 'text-green-400' : ''
              }`} style={step.status === 'active' ? { color: tPrimary } : undefined}>
                {step.status === 'active'
                  ? <Loader2 className="w-3 h-3 animate-spin" />
                  : iconMap[step.icon]
                }
              </div>
              <div className="min-w-0">
                <div className={`text-xs ${
                  step.status === 'active' ? 'text-white' : 'text-gray-400'
                }`}>
                  {step.label}
                </div>
                {step.detail && (
                  <div className="text-[10px] text-gray-600 mt-0.5 break-words">{step.detail}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl transition-all group hover:scale-[1.015]"
        style={{ background: bgColor, border: `1px solid ${borderColor}` }}
        onMouseEnter={e => (e.currentTarget.style.background = bgHover)}
        onMouseLeave={e => (e.currentTarget.style.background = bgColor)}
      >
        <Brain className="w-3 h-3 flex-shrink-0" style={{ color: tPrimary }} />
        <span className="text-[11px] font-medium" style={{ color: textColor }}>
          {t('aiCoach.hadThought')}
        </span>
        <span className="text-[10px] ml-0.5" style={{ color: chevronColor }}>· {doneCount} {t('aiCoach.steps')}</span>
        <ChevronDown
          className={`w-3 h-3 ml-auto transition-transform ${collapsed ? '' : 'rotate-180'}`}
          style={{ color: chevronColor }}
        />
      </button>

      {!collapsed && (
        <div className="mt-1.5 pl-1 space-y-1 ml-1.5 pb-1" style={{ borderLeft: `2px solid ${borderColor}` }}>
          {steps.map(step => (
            <div key={step.id} className="flex items-start gap-2 py-0.5 opacity-60">
              <div className="flex-shrink-0 mt-0.5 text-green-400">
                {iconMap[step.icon]}
              </div>
              <div className="min-w-0">
                <div className="text-xs text-gray-400">{step.label}</div>
                {step.detail && (
                  <div className="text-[10px] text-gray-600 mt-0.5 break-words">{step.detail}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============ Main Component ============

interface FloatingAICoachProps {
  currentPageContent?: string;
  userId?: string;
}

function getChatStorageKey(userId?: string) {
  if (!userId) return STORAGE_KEY;
  return `${STORAGE_KEY}_${userId}`;
}

function loadChatsForUser(userId?: string): Chat[] {
  // Migration: legacy (global) -> user scoped (only if userId present and scoped is empty)
  const scopedKey = getChatStorageKey(userId);
  try {
    const scopedRaw = localStorage.getItem(scopedKey);
    if (scopedRaw) return JSON.parse(scopedRaw);
  } catch { /* ignore */ }

  if (!userId) return [];

  try {
    const legacyRaw = localStorage.getItem(STORAGE_KEY);
    if (!legacyRaw) return [];
    const parsed = JSON.parse(legacyRaw) as Chat[];
    // Only migrate if scoped key is missing/empty
    localStorage.setItem(scopedKey, JSON.stringify(parsed));
    localStorage.removeItem(STORAGE_KEY);
    return parsed;
  } catch {
    return [];
  }
}

function saveChatsForUser(chats: Chat[], userId?: string): void {
  try {
    const limited = chats.slice(0, MAX_CHATS);
    localStorage.setItem(getChatStorageKey(userId), JSON.stringify(limited));
  } catch { /* ignore */ }
}

export default function FloatingAICoach({ currentPageContent, userId }: FloatingAICoachProps) {
  const { settings } = useAISettings();
  const { language } = useLanguage();
  const { currentTheme } = useTheme();
  const { currentPageContent: ctxPageContent } = usePageContext();
  const { t } = useLanguage();

  // Theme-Farben mit Light-Color-Detection (Fix für Monochrome / Arctic White)
  const safePrimary   = hexLuminance(currentTheme.colors.primary)   > 0.5 ? darkenHex(currentTheme.colors.primary,   100) : currentTheme.colors.primary;
  const safeSecondary = hexLuminance(currentTheme.colors.secondary) > 0.5 ? darkenHex(currentTheme.colors.secondary, 100) : currentTheme.colors.secondary;
  const themeGradient       = `linear-gradient(135deg, ${safePrimary}, ${safeSecondary})`;
  const themeGradientSubtle = `linear-gradient(to right, ${safePrimary}1a, ${safeSecondary}0d)`;

  const pageContent = currentPageContent || ctxPageContent || '';

  // Modal state
  const [isOpen, setIsOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [view, setView] = useState<'chat' | 'chatList'>('chat');

  // ── Chat state ──
  // `chats` = only persisted chats (have at least one message)
  // `currentChat` = the currently displayed chat (may be unsaved/empty)
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  // Track if currentChat is already in `chats` (persisted)
  const currentChatPersistedRef = useRef(false);

  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);

  // Draggable/resizable state
  const [position, setPosition] = useState({ x: window.innerWidth - 390, y: window.innerHeight - 560 });
  const [size, setSize] = useState({ width: 370, height: 520 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, w: 0, h: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const prevMsgCountRef = useRef(0);

  // Load chats from localStorage on mount (only persisted ones)
  useEffect(() => {
    const loaded = loadChatsForUser(userId);
    setChats(loaded);
  }, [userId]);

  // Fix 3: Scroll nur wenn neue Nachricht hinzukommt, nicht bei Collapse-Toggle
  useEffect(() => {
    const count = currentChat?.messages.length ?? 0;
    if (count > prevMsgCountRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    prevMsgCountRef.current = count;
  }, [currentChat?.messages]);

  // ── Drag / resize ──
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newX = Math.max(0, Math.min(window.innerWidth - size.width, e.clientX - dragOffset.x));
        const newY = Math.max(0, Math.min(window.innerHeight - size.height, e.clientY - dragOffset.y));
        setPosition({ x: newX, y: newY });
      } else if (isResizing) {
        const dx = e.clientX - resizeStart.x;
        const dy = e.clientY - resizeStart.y;
        setSize({ width: Math.max(300, resizeStart.w + dx), height: Math.max(320, resizeStart.h + dy) });
      }
    };
    const handleMouseUp = () => { setIsDragging(false); setIsResizing(false); };
    if (isDragging || isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, isResizing, dragOffset, resizeStart, size]);

  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    setIsDragging(true);
    setDragOffset({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, w: size.width, h: size.height });
  };

  // ── Opens modal with a fresh empty chat (never saved until message sent) ──
  const openModal = useCallback(() => {
    const fresh = createFreshChat(t('aiCoach.newChat'));
    setCurrentChat(fresh);
    currentChatPersistedRef.current = false;
    setView('chat');
    setError('');
    setInputText('');
    setThinkingSteps([]);
    setIsOpen(true);
  }, []);

  // ── Start a new chat (from chat list or header button) ──
  // Creates a fresh unsaved chat and switches to it, without touching `chats`
  const handleNewChat = useCallback(() => {
    const fresh = createFreshChat(t('aiCoach.newChat'));
    setCurrentChat(fresh);
    currentChatPersistedRef.current = false;
    setView('chat');
    setError('');
    setInputText('');
    setThinkingSteps([]);
  }, []);

  // External open (z.B. aus Analysis/Training/Dev Panels)
  useEffect(() => {
    return onOpenAICoach(({ prefill, newChat }) => {
      if (!isOpen) {
        openModal();
      } else if (newChat) {
        handleNewChat();
      }
      if (typeof prefill === 'string' && prefill.trim()) {
        setInputText(prefill);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    });
  }, [handleNewChat, isOpen, openModal]);

  // ── Switch to an existing persisted chat ──
  const switchToChat = useCallback((chatId: string) => {
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentChat(chat);
      currentChatPersistedRef.current = true;
    }
    setView('chat');
    setError('');
  }, [chats]);

  // ── Delete a persisted chat ──
  const deleteChat = useCallback((chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = chats.filter(c => c.id !== chatId);
    setChats(updated);
    saveChatsForUser(updated, userId);
    // If we're currently viewing this chat, open a fresh one
      if (currentChat?.id === chatId) {
      const fresh = createFreshChat(t('aiCoach.newChat'));
      setCurrentChat(fresh);
      currentChatPersistedRef.current = false;
    }
  }, [chats, currentChat, t]);

  // ── Update currentChat and sync to `chats` if already persisted ──
  const applyToCurrentChat = useCallback((updater: (c: Chat) => Chat) => {
    setCurrentChat(prev => {
      if (!prev) return prev;
      const updated = updater(prev);
      // Also update in chats list if this chat is persisted
      if (currentChatPersistedRef.current) {
        setChats(prevChats => {
          const newChats = prevChats.map(c => c.id === updated.id ? updated : c);
          saveChatsForUser(newChats, userId);
          return newChats;
        });
      }
      return updated;
    });
  }, []);

  // ── Ensure currentChat is persisted (call before first message) ──
  const ensurePersisted = useCallback((chat: Chat) => {
    if (!currentChatPersistedRef.current) {
      currentChatPersistedRef.current = true;
      setChats(prev => {
        const newChats = [chat, ...prev];
        saveChatsForUser(newChats, userId);
        return newChats;
      });
    }
  }, []);

  // ── Build system prompt ──
  const buildSystemPrompt = (): string => {
    let prompt = `Du bist ein hilfreicher KI-Assistent in der FrameTrain Desktop-Anwendung für Machine Learning Training.\n\nAKTUELLE SEITE UND KONTEXT:`;
    if (pageContent) {
      prompt += `\n${pageContent}`;
    } else {
      prompt += `\nKein spezifischer Seitenkontext verfügbar.`;
    }
    prompt += `\n\nANWEISUNGEN:\n- Antworte auf Deutsch, prägnant und hilfreich\n- Erkläre ML-Konzepte verständlich\n- Wenn du Fehler siehst, erkläre ihre Ursache und Lösung\n- Nutze Markdown-Formatierung: **fett** für wichtige Begriffe, Listen für Schritte\n- Beziehe dich konkret auf den Seiteninhalt wenn relevant`;
    return prompt;
  };

  // Fix 2: Schritte sequenziell aufdecken — nur den aktuellen Schritt zeigen
  const runThinkingAnimation = async (hasPageContent: boolean): Promise<ThinkingStep[]> => {
    const steps: ThinkingStep[] = [
      { id: 's1', label: 'Seite analysieren', icon: 'search', status: 'pending',
        detail: hasPageContent ? pageContent.slice(0, 80) + '...' : 'Kein Kontext verfügbar' },
      { id: 's2', label: 'Kontext verarbeiten', icon: 'brain', status: 'pending', detail: undefined },
      { id: 's3', label: 'Antwort generieren', icon: 'sparkles', status: 'pending', detail: undefined },
    ];

    // Nur Step 1 sichtbar + aktiv
    setThinkingSteps([{ ...steps[0], status: 'active' }]);
    await new Promise(r => setTimeout(r, 400));

    // Step 1 fertig, Step 2 wird sichtbar + aktiv
    setThinkingSteps([
      { ...steps[0], status: 'done' },
      { ...steps[1], status: 'active' },
    ]);
    await new Promise(r => setTimeout(r, 350));

    // Step 2 fertig, Step 3 wird sichtbar + aktiv
    setThinkingSteps([
      { ...steps[0], status: 'done' },
      { ...steps[1], status: 'done' },
      { ...steps[2], status: 'active' },
    ]);

    return steps.map(s => ({ ...s, status: 'done' as const }));
  };

  // ── Send message ──
  const sendMessage = async () => {
    const text = inputText.trim();
    if (!text || isLoading || !currentChat) return;

    if (!settings.enabled) {
      setError(t('aiCoach.disabledError'));
      return;
    }

    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    // Snapshot of currentChat before any async
    const chatSnapshot = currentChat;

    setInputText('');
    setError('');

    // Build updated chat with user message
    const isFirstMessage = chatSnapshot.messages.length === 0;
    const chatWithUserMsg: Chat = {
      ...chatSnapshot,
      messages: [...chatSnapshot.messages, userMsg],
      title: isFirstMessage ? generateTitle(text) : chatSnapshot.title,
      updatedAt: Date.now(),
    };

    // Persist if this is the first message
    if (isFirstMessage) {
      ensurePersisted(chatWithUserMsg);
    }

    // Update local state immediately
    setCurrentChat(chatWithUserMsg);
    if (!isFirstMessage && currentChatPersistedRef.current) {
      setChats(prev => {
        const updated = prev.map(c => c.id === chatWithUserMsg.id ? chatWithUserMsg : c);
        saveChatsForUser(updated, userId);
        return updated;
      });
    }

    setIsLoading(true);

    // Run thinking animation — get the final done-steps for later
    const finalThinkingSteps = await runThinkingAnimation(!!pageContent);

    try {
      const meta = PROVIDER_META[settings.provider];
      if (meta.needsKey && !settings.apiKey) {
        throw new Error(t('aiCoach.apiKeyMissing'));
      }

      const history = chatWithUserMsg.messages.slice(-10);
      const systemPrompt = buildSystemPrompt();
      const responseText = await callAIClient(settings, {
        system: systemPrompt,
        messages: history.map(m => ({ role: m.role, content: m.content })),
        maxTokens: 1500,
        temperature: 0.7,
        responseLanguage: language,
      });

      setThinkingSteps([]);

      const assistantMsg: Message = {
        id: `msg-${Date.now()}-ai`,
        role: 'assistant',
        content: responseText,
        timestamp: Date.now(),
        // Use the locally captured finalThinkingSteps (no stale closure!)
        thinkingSteps: finalThinkingSteps,
        thinkingCollapsed: true,
      };

      setCurrentChat(prev => {
        if (!prev) return prev;
        const updated: Chat = { ...prev, messages: [...prev.messages, assistantMsg], updatedAt: Date.now() };
        if (currentChatPersistedRef.current) {
          setChats(prevChats => {
            const newChats = prevChats.map(c => c.id === updated.id ? updated : c);
            saveChatsForUser(newChats, userId);
            return newChats;
          });
        }
        return updated;
      });

    } catch (e: any) {
      setThinkingSteps([]);
      setError(e?.message || t('aiCoach.unknownError'));
    } finally {
      setIsLoading(false);
    }
  };

  // ── Toggle thinking collapse for a specific message ──
  const toggleMessageThinking = (msgId: string) => {
    applyToCurrentChat(c => ({
      ...c,
      messages: c.messages.map(m =>
        m.id === msgId ? { ...m, thinkingCollapsed: !m.thinkingCollapsed } : m
      ),
    }));
  };

  if (!settings.enabled) return null;

  // ── Closed: floating button ──
  if (!isOpen) {
    return (
      <>
        <style>{COACH_MAGIC_CSS}</style>
        <button
          onClick={openModal}
          className="ft-coach-button fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-2xl hover:shadow-white/20 hover:scale-110 transition-all flex items-center justify-center z-40"
          style={{
            background: themeGradient,
            boxShadow: `0 18px 48px ${safePrimary}33, 0 0 0 1px rgba(255,255,255,.14), inset 0 1px 0 rgba(255,255,255,.28)`,
          }}
          title={t('aiCoach.openTitle')}
        >
          <Sparkles className="absolute w-3 h-3 text-white/70 -top-0.5 right-1.5" />
          <Brain className="w-6 h-6 text-white" />
        </button>
      </>
    );
  }

  // ── Chat list view ──
  const renderChatListView = () => (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-white/10 flex-shrink-0 select-none cursor-move"
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <MessageSquare className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-semibold text-white">{t('aiCoach.chatHistory')}</span>
          <span className="text-xs text-gray-500">({chats.length})</span>
        </div>
        <div className="pointer-events-auto flex items-center gap-1">
          <button
            onClick={handleNewChat}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all"
            title={t('aiCoach.newChat')}
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={() => setView('chat')}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title={t('aiCoach.back')}
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="ft-coach-scroll flex-1 overflow-y-auto p-2 space-y-1">
        {chats.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center p-4">
            <MessageSquare className="w-8 h-8 text-gray-600" />
            <p className="text-gray-500 text-sm">{t('aiCoach.noChats')}</p>
            <p className="text-gray-600 text-xs">{t('aiCoach.noChatsHint')}</p>
            <button
              onClick={handleNewChat}
              className="px-3 py-2 text-xs font-medium bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg border border-purple-500/30 transition-all flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('aiCoach.newChatStart')}
            </button>
          </div>
        ) : (
          <>
            <button
              onClick={handleNewChat}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-white/10 hover:border-purple-500/30 hover:bg-purple-500/5 text-gray-400 hover:text-purple-300 transition-all text-xs font-medium"
            >
              <Plus className="w-3.5 h-3.5" />
              {t('aiCoach.newChat')}
            </button>
            {chats.map(chat => (
              <div key={chat.id} className="ft-coach-message flex items-center gap-1.5 group">
                <button
                  onClick={() => switchToChat(chat.id)}
                  className={`flex-1 flex items-start justify-between gap-2 px-3 py-2.5 rounded-xl border text-left transition-all ${
                    chat.id === currentChat?.id
                      ? 'bg-white/[0.10] border-white/20 text-white shadow-lg shadow-black/20'
                      : 'bg-white/[0.035] border-white/5 hover:bg-white/[0.07] hover:border-white/10 text-gray-300 hover:translate-x-0.5'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium truncate">{chat.title}</div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[10px] text-gray-600">{formatRelativeTime(chat.updatedAt, t)}</span>
                      {chat.messages.length > 0 && (
                        <span className="text-[10px] text-gray-600">{chat.messages.length} {t('aiCoach.messagesCount')}</span>
                      )}
                    </div>
                  </div>
                </button>
                <button
                  onClick={(e) => deleteChat(chat.id, e)}
                  className="p-1.5 rounded-lg hover:bg-red-500/20 text-gray-600 hover:text-red-400 transition-all flex-shrink-0 hover:scale-110"
                  title={t('aiCoach.deleteChat')}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );

  // ── Chat view ──
  const renderChatView = () => (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 flex-shrink-0 select-none cursor-move"
        style={{ background: themeGradientSubtle }}
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none min-w-0">
          <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: themeGradient }}>
            <Brain className="w-3.5 h-3.5 text-white" />
          </div>
          <div className="min-w-0">
            <div className="text-xs font-bold text-white truncate max-w-[160px]">
              {currentChat?.title || t('aiCoach.openTitle')}
            </div>
            {pageContent && (
              <div className="text-[10px] text-gray-500 truncate max-w-[160px]">
                {formatPageContextTitle(pageContent).slice(0, 35)}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 pointer-events-auto flex-shrink-0">
          <button onClick={() => setView('chatList')} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all" title={t('aiCoach.chatHistory')}>
            <MessageSquare className="w-3.5 h-3.5" />
          </button>
          <button onClick={handleNewChat} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all" title={t('aiCoach.newChat')}>
            <Plus className="w-3.5 h-3.5" />
          </button>
          {!isMaximized && (
            <button onClick={() => setIsMaximized(true)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
          {isMaximized && (
            <button onClick={() => setIsMaximized(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
              <Minimize2 className="w-3.5 h-3.5" />
            </button>
          )}
          <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="ft-coach-scroll flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {(!currentChat || currentChat.messages.length === 0) && !isLoading && (
          <div className="ft-coach-message flex flex-col items-center justify-center h-full text-center gap-3 py-8">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center"
              style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}33, ${currentTheme.colors.secondary}1a)`, boxShadow: `0 14px 34px ${safePrimary}22, inset 0 1px 0 rgba(255,255,255,.12)` }}>
              <Brain className="w-6 h-6" style={{ color: safePrimary }} />
            </div>
            <div>
              <p className="text-gray-300 text-sm font-medium">{t('aiCoach.greeting')}</p>
              <p className="text-gray-600 text-xs mt-1">
                {pageContent
                  ? t('aiCoach.greetingWithContext')
                  : t('aiCoach.greetingNoContext')}
              </p>
            </div>
            {pageContent && (
              <div className="flex justify-center w-full">
                <div className="max-w-[220px] px-3 py-2 bg-white/[0.03] border border-white/5 rounded-xl space-y-1">
                  <div className="flex items-center gap-1 leading-none">
                    <FileSearch className="w-3 h-3 flex-shrink-0" style={{ color: safePrimary }} />
                    <span className="text-[10px] font-medium" style={{ color: `${safePrimary}cc` }}>{t('aiCoach.loadedContext')}</span>
                  </div>
                  <p className="text-[10px] text-gray-600 leading-relaxed truncate">
                    {formatPageContextTitle(pageContent)}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {currentChat?.messages.map(msg => (
          <div key={msg.id} className={`ft-coach-message flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' ? (
              <div className="max-w-[88%] space-y-1">
                {/* Thinking block — shown above the message when finished */}
                {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                  <ThinkingBlock
                    steps={msg.thinkingSteps}
                    isActive={false}
                    collapsed={msg.thinkingCollapsed ?? true}
                    onToggle={() => toggleMessageThinking(msg.id)}
                  />
                )}
                <div className="px-3 py-2.5 rounded-2xl rounded-tl-sm bg-white/[0.065] border border-white/[0.10] text-gray-200 shadow-lg shadow-black/20">
                  <MarkdownText text={msg.content} />
                </div>
              </div>
            ) : (
              <div className="max-w-[85%] px-3 py-2.5 rounded-2xl rounded-tr-sm text-white text-sm leading-relaxed shadow-lg shadow-black/20"
                style={{ background: themeGradient, boxShadow: `0 12px 28px ${safePrimary}20, inset 0 1px 0 rgba(255,255,255,.16)` }}>
                {msg.content}
              </div>
            )}
          </div>
        ))}

        {/* Active thinking steps (while loading) */}
        {isLoading && thinkingSteps.length > 0 && (
          <div className="ft-coach-message flex justify-start">
            <div className="max-w-[88%]">
              <ThinkingBlock
                steps={thinkingSteps}
                isActive={true}
                collapsed={false}
                onToggle={() => {}}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="ft-coach-message flex justify-start">
            <div className="max-w-[88%] px-3 py-2.5 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-start gap-2">
              <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
              <span className="text-red-300 text-xs leading-relaxed break-words">{error}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 pb-3 pt-2 flex-shrink-0 border-t border-white/5 bg-black/10">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
            }}
            placeholder={t('aiCoach.inputPlaceholder')}
            disabled={isLoading}
            rows={1}
            className="flex-1 px-3 py-2 bg-white/[0.055] border border-white/10 rounded-xl text-white text-xs placeholder-gray-600 focus:outline-none disabled:opacity-50 resize-none leading-relaxed transition-all focus:bg-white/[0.075]"
            onFocus={e => {
              e.currentTarget.style.borderColor = `${safePrimary}66`;
              e.currentTarget.style.boxShadow = `0 0 0 3px ${safePrimary}1f`;
            }}
            onBlur={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.10)';
              e.currentTarget.style.boxShadow = 'none';
            }}
            style={{ maxHeight: '80px' }}
            onInput={e => {
              const el = e.target as HTMLTextAreaElement;
              el.style.height = 'auto';
              el.style.height = Math.min(el.scrollHeight, 80) + 'px';
            }}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputText.trim()}
            className="p-2 rounded-xl text-white flex-shrink-0 transition-all disabled:opacity-40 hover:opacity-95 hover:scale-105 active:scale-95"
            style={{ background: themeGradient, boxShadow: `0 10px 26px ${safePrimary}25, inset 0 1px 0 rgba(255,255,255,.20)` }}
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );

  const content = view === 'chatList' ? renderChatListView() : renderChatView();

  // ── Maximized overlay ──
  if (isMaximized) {
    return (
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
        <style>{COACH_MAGIC_CSS}</style>
        <div className="ft-coach-shell relative bg-slate-900/90 rounded-2xl border border-white/10 w-full max-w-2xl h-[85vh] overflow-hidden flex flex-col shadow-2xl">
          {content}
        </div>
      </div>
    );
  }

  // ── Floating window ──
  return (
    <div
      className="ft-coach-shell fixed bg-slate-900/90 rounded-2xl border border-white/[0.10] shadow-2xl overflow-hidden flex flex-col z-50"
      style={{
        width: `${size.width}px`,
        height: `${size.height}px`,
        left: `${position.x}px`,
        top: `${position.y}px`,
        boxShadow: `0 25px 70px rgba(0,0,0,0.52), 0 0 0 1px rgba(255,255,255,0.06), 0 0 48px ${safePrimary}14`,
      }}
    >
      <style>{COACH_MAGIC_CSS}</style>
      {content}
      <div
        className="absolute bottom-0 right-0 w-5 h-5 cursor-se-resize"
        onMouseDown={handleResizeMouseDown}
        style={{
          background: `linear-gradient(135deg, transparent 50%, ${currentTheme.colors.primary}4d 100%)`,
          borderRadius: '0 0 16px 0',
        }}
      />
    </div>
  );
}
