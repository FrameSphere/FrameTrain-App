import { useState, useRef, useEffect, useCallback } from 'react';
import {
  X, Send, Loader2, AlertCircle, CheckCircle, Maximize2, Minimize2,
  MessageSquare, Plus, Trash2, ChevronDown, ChevronRight, Brain,
  FileSearch, Cpu, Sparkles, ArrowLeft
} from 'lucide-react';
import { useAISettings, AIProvider } from '../contexts/AISettingsContext';
import { useTheme } from '../contexts/ThemeContext';
import { usePageContext } from '../contexts/PageContext';

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

const PROVIDER_META: Record<AIProvider, {
  label: string; emoji: string; needsKey: boolean; models: string[];
}> = {
  anthropic: {
    label: 'Claude (Anthropic)', emoji: '🤖', needsKey: true,
    models: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  },
  openai: {
    label: 'GPT-4o (OpenAI)', emoji: '🟢', needsKey: true,
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
  },
  groq: {
    label: 'Groq', emoji: '⚡', needsKey: true,
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
  },
  ollama: {
    label: 'Ollama (Lokal)', emoji: '🦙', needsKey: false,
    models: ['llama3.2', 'llama3.1', 'mistral', 'gemma2'],
  },
};

const STORAGE_KEY = 'ft_ai_chats_v2';
const MAX_CHATS = 50;

// ============ Helpers ============

function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveChats(chats: Chat[]): void {
  try {
    const limited = chats.slice(0, MAX_CHATS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(limited));
  } catch { /* ignore */ }
}

function createChat(): Chat {
  return {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title: 'Neuer Chat',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function generateTitle(firstMessage: string): string {
  return firstMessage.slice(0, 40) + (firstMessage.length > 40 ? '…' : '');
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (mins < 1) return 'Gerade eben';
  if (mins < 60) return `vor ${mins} Min.`;
  if (hours < 24) return `vor ${hours} Std.`;
  return `vor ${days} Tag${days !== 1 ? 'en' : ''}`;
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

    // Bullet list
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

    // Numbered list
    if (trimmed.match(/^\d+\.\s/)) {
      const items: string[] = [];
      let num = 1;
      while (i < lines.length && lines[i].trim().match(/^\d+\.\s/)) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
        num++;
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

    // Heading (### or ##)
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

  return (
    <div className="mb-3">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors group"
      >
        {isActive ? (
          <Loader2 className="w-3 h-3 animate-spin text-purple-400" />
        ) : (
          <Brain className="w-3 h-3 text-purple-400" />
        )}
        <span className="text-purple-300/70">
          {isActive
            ? (activeStep?.label || 'Denkt nach...')
            : `Denkprozess (${doneCount} Schritte)`
          }
        </span>
        {!isActive && (
          <ChevronDown className={`w-3 h-3 transition-transform ${collapsed ? '' : 'rotate-180'}`} />
        )}
      </button>

      {(!collapsed || isActive) && (
        <div className="mt-1.5 pl-1 space-y-1 border-l-2 border-purple-500/20 ml-1.5">
          {steps.map(step => (
            <div
              key={step.id}
              className={`flex items-start gap-2 py-0.5 transition-all ${
                step.status === 'pending' ? 'opacity-30' :
                step.status === 'active' ? 'opacity-100' : 'opacity-60'
              }`}
            >
              <div className={`flex-shrink-0 mt-0.5 ${
                step.status === 'active' ? 'text-purple-300' :
                step.status === 'done' ? 'text-green-400' :
                'text-gray-600'
              }`}>
                {step.status === 'active'
                  ? <Loader2 className="w-3 h-3 animate-spin" />
                  : iconMap[step.icon]
                }
              </div>
              <div className="min-w-0">
                <div className={`text-xs ${
                  step.status === 'active' ? 'text-white' :
                  step.status === 'done' ? 'text-gray-400' :
                  'text-gray-600'
                }`}>
                  {step.label}
                </div>
                {step.detail && step.status !== 'pending' && (
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
}

export default function FloatingAICoach({ currentPageContent }: FloatingAICoachProps) {
  const { settings } = useAISettings();
  const { currentTheme } = useTheme();
  const { currentPageContent: ctxPageContent } = usePageContext();

  // Theme-abhängige Farben (reagieren auf Design-Wechsel in Einstellungen)
  const themeGradient = `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`;
  const themeGradientSubtle = `linear-gradient(to right, ${currentTheme.colors.primary}1a, ${currentTheme.colors.secondary}0d)`;
  const themeAccentAlpha = `${currentTheme.colors.primary}4d`; // ~30% opacity

  const pageContent = currentPageContent || ctxPageContent || '';

  // Modal state
  const [isOpen, setIsOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [view, setView] = useState<'chat' | 'chatList'>('chat');

  // Chat state
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [thinkingCollapsed, setThinkingCollapsed] = useState(false);

  // Draggable/resizable state
  const [position, setPosition] = useState({ x: window.innerWidth - 390, y: window.innerHeight - 560 });
  const [size, setSize] = useState({ width: 370, height: 520 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, w: 0, h: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load chats from localStorage on mount
  useEffect(() => {
    const loaded = loadChats();
    setChats(loaded);
    if (loaded.length > 0) {
      setActiveChatId(loaded[0].id);
    }
  }, []);

  const activeChat = chats.find(c => c.id === activeChatId) || null;

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeChat?.messages]);

  // Drag/resize mouse handling
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newX = Math.max(0, Math.min(window.innerWidth - size.width, e.clientX - dragOffset.x));
        const newY = Math.max(0, Math.min(window.innerHeight - size.height, e.clientY - dragOffset.y));
        setPosition({ x: newX, y: newY });
      } else if (isResizing) {
        const dx = e.clientX - resizeStart.x;
        const dy = e.clientY - resizeStart.y;
        setSize({
          width: Math.max(300, resizeStart.w + dx),
          height: Math.max(320, resizeStart.h + dy),
        });
      }
    };
    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };
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

  // Chat management
  const createNewChat = useCallback(() => {
    const chat = createChat();
    const updated = [chat, ...chats];
    setChats(updated);
    saveChats(updated);
    setActiveChatId(chat.id);
    setView('chat');
    setError('');
  }, [chats]);

  const switchToChat = useCallback((chatId: string) => {
    setActiveChatId(chatId);
    setView('chat');
    setError('');
  }, []);

  const deleteChat = useCallback((chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = chats.filter(c => c.id !== chatId);
    setChats(updated);
    saveChats(updated);
    if (activeChatId === chatId) {
      setActiveChatId(updated.length > 0 ? updated[0].id : null);
    }
  }, [chats, activeChatId]);

  const updateChat = useCallback((chatId: string, updater: (c: Chat) => Chat) => {
    setChats(prev => {
      const updated = prev.map(c => c.id === chatId ? updater(c) : c);
      saveChats(updated);
      return updated;
    });
  }, []);

  // Build system prompt with page context
  const buildSystemPrompt = (): string => {
    let prompt = `Du bist ein hilfreicher KI-Assistent in der FrameTrain Desktop-Anwendung für Machine Learning Training.

AKTUELLE SEITE UND KONTEXT:`;

    if (pageContent) {
      prompt += `\n${pageContent}`;
    } else {
      prompt += `\nKein spezifischer Seitenkontext verfügbar.`;
    }

    prompt += `\n\nANWEISUNGEN:
- Antworte auf Deutsch, prägnant und hilfreich
- Erkläre ML-Konzepte verständlich  
- Wenn du Fehler siehst, erkläre ihre Ursache und Lösung
- Nutze Markdown-Formatierung: **fett** für wichtige Begriffe, Listen für Schritte
- Beziehe dich konkret auf den Seiteninhalt wenn relevant`;

    return prompt;
  };

  // Thinking steps helper
  const runThinkingAnimation = async (hasPageContent: boolean): Promise<ThinkingStep[]> => {
    const steps: ThinkingStep[] = [
      { id: 's1', label: 'Seite analysieren', icon: 'search', status: 'pending',
        detail: hasPageContent ? pageContent.slice(0, 80) + '...' : 'Kein Kontext verfügbar' },
      { id: 's2', label: 'Kontext verarbeiten', icon: 'brain', status: 'pending', detail: undefined },
      { id: 's3', label: 'Antwort generieren', icon: 'sparkles', status: 'pending', detail: undefined },
    ];

    const setSteps = (updater: (prev: ThinkingStep[]) => ThinkingStep[]) => {
      setThinkingSteps(prev => updater(prev));
      return new Promise<void>(r => setTimeout(r, 0));
    };

    setThinkingSteps(steps);
    await new Promise(r => setTimeout(r, 50));

    // Step 1 active
    setThinkingSteps(s => s.map((st, i) => i === 0 ? { ...st, status: 'active' } : st));
    await new Promise(r => setTimeout(r, 400));

    // Step 1 done, step 2 active
    setThinkingSteps(s => s.map((st, i) =>
      i === 0 ? { ...st, status: 'done' } :
      i === 1 ? { ...st, status: 'active' } : st
    ));
    await new Promise(r => setTimeout(r, 350));

    // Step 2 done, step 3 active
    setThinkingSteps(s => s.map((st, i) =>
      i === 1 ? { ...st, status: 'done' } :
      i === 2 ? { ...st, status: 'active' } : st
    ));

    return steps;
  };

  const completeThinkingSteps = () => {
    setThinkingSteps(s => s.map(st => ({ ...st, status: 'done' as const })));
    setThinkingCollapsed(true);
  };

  // Send message
  const sendMessage = async () => {
    const text = inputText.trim();
    if (!text || isLoading) return;

    if (!settings.enabled) {
      setError('KI-Assistent ist deaktiviert. Bitte in Einstellungen aktivieren.');
      return;
    }

    // Ensure we have a chat
    let currentChatId = activeChatId;
    if (!currentChatId) {
      const chat = createChat();
      const updated = [chat, ...chats];
      setChats(updated);
      saveChats(updated);
      setActiveChatId(chat.id);
      currentChatId = chat.id;
    }

    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    setInputText('');
    setError('');
    setThinkingCollapsed(false);

    // Add user message and update title
    updateChat(currentChatId, c => ({
      ...c,
      messages: [...c.messages, userMsg],
      title: c.messages.length === 0 ? generateTitle(text) : c.title,
      updatedAt: Date.now(),
    }));

    setIsLoading(true);

    // Animate thinking steps
    const hasContent = !!pageContent;
    await runThinkingAnimation(hasContent);

    try {
      const meta = PROVIDER_META[settings.provider];
      if (meta.needsKey && !settings.apiKey) {
        throw new Error('API-Key fehlt. Bitte in Einstellungen → KI-Assistent konfigurieren.');
      }

      // Get current chat messages for context (up to last 10)
      const currentChat = chats.find(c => c.id === currentChatId);
      const history = currentChat?.messages.slice(-10) || [];
      const systemPrompt = buildSystemPrompt();
      let responseText = '';

      if (settings.provider === 'ollama') {
        const conversationText = [
          systemPrompt,
          ...history.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`),
          `User: ${text}`,
          'Assistant:',
        ].join('\n\n');

        const res = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: settings.ollamaModel || 'llama3.2',
            prompt: conversationText,
            stream: false,
            options: { temperature: 0.7, num_ctx: 4096 },
          }),
        });
        if (!res.ok) throw new Error('Ollama nicht erreichbar (http://localhost:11434). Läuft Ollama?');
        const data = await res.json();
        responseText = data.response || '';

      } else if (settings.provider === 'groq') {
        const messages = [
          { role: 'system', content: systemPrompt },
          ...history.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: text },
        ];
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${settings.apiKey}` },
          body: JSON.stringify({
            model: settings.selectedModel || 'llama-3.3-70b-versatile',
            max_tokens: 1500,
            temperature: 0.7,
            messages,
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';

      } else if (settings.provider === 'anthropic') {
        const messages = [
          ...history.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: text },
        ];
        const res = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': settings.apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          body: JSON.stringify({
            model: settings.selectedModel || 'claude-haiku-4-5',
            max_tokens: 1500,
            system: systemPrompt,
            messages,
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.content?.[0]?.text || '';

      } else {
        // OpenAI
        const messages = [
          { role: 'system', content: systemPrompt },
          ...history.map(m => ({ role: m.role, content: m.content })),
          { role: 'user', content: text },
        ];
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${settings.apiKey}` },
          body: JSON.stringify({
            model: settings.selectedModel || 'gpt-4o-mini',
            max_tokens: 1500,
            temperature: 0.7,
            messages,
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';
      }

      completeThinkingSteps();

      const assistantMsg: Message = {
        id: `msg-${Date.now()}-ai`,
        role: 'assistant',
        content: responseText,
        timestamp: Date.now(),
        thinkingSteps: thinkingSteps.map(s => ({ ...s, status: 'done' as const })),
        thinkingCollapsed: true,
      };

      updateChat(currentChatId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        updatedAt: Date.now(),
      }));
      setThinkingSteps([]);

    } catch (e: any) {
      completeThinkingSteps();
      setThinkingSteps([]);
      setError(e?.message || 'Unbekannter Fehler');
    } finally {
      setIsLoading(false);
    }
  };

  // Toggle thinking steps for a specific message
  const toggleMessageThinking = (msgId: string) => {
    updateChat(activeChatId!, c => ({
      ...c,
      messages: c.messages.map(m =>
        m.id === msgId ? { ...m, thinkingCollapsed: !m.thinkingCollapsed } : m
      ),
    }));
  };

  if (!settings.enabled) return null;

  // ── Closed state: floating button ──
  if (!isOpen) {
    return (
      <button
        onClick={() => {
          setIsOpen(true);
          if (!activeChatId && chats.length === 0) createNewChat();
        }}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-2xl hover:shadow-purple-500/30 hover:scale-110 transition-all flex items-center justify-center z-40"
        style={{ background: themeGradient }}
        title="KI-Coach öffnen"
      >
        <Brain className="w-6 h-6 text-white" />
      </button>
    );
  }

  // ── Content for both floating and maximized ──
  const renderChatListView = () => (
    <div className="flex flex-col h-full">
      {/* Chat list header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-white/10 flex-shrink-0 select-none cursor-move"
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <MessageSquare className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-semibold text-white">Chatverläufe</span>
          <span className="text-xs text-gray-500">({chats.length})</span>
        </div>
        <div className="pointer-events-auto flex items-center gap-1">
          <button
            onClick={createNewChat}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all"
            title="Neuer Chat"
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={() => setView('chat')}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Zurück"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {chats.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center p-4">
            <MessageSquare className="w-8 h-8 text-gray-600" />
            <p className="text-gray-500 text-sm">Noch keine Chats</p>
            <button
              onClick={createNewChat}
              className="px-3 py-2 text-xs font-medium bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg border border-purple-500/30 transition-all flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              Ersten Chat starten
            </button>
          </div>
        ) : (
          <>
            <button
              onClick={createNewChat}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-white/10 hover:border-purple-500/30 hover:bg-purple-500/5 text-gray-400 hover:text-purple-300 transition-all text-xs font-medium"
            >
              <Plus className="w-3.5 h-3.5" />
              Neuer Chat
            </button>
            {chats.map(chat => (
              <button
                key={chat.id}
                onClick={() => switchToChat(chat.id)}
                className={`w-full flex items-start justify-between gap-2 px-3 py-2.5 rounded-xl border text-left transition-all group ${
                  chat.id === activeChatId
                    ? 'bg-purple-500/15 border-purple-500/30 text-white'
                    : 'bg-white/[0.03] border-white/5 hover:bg-white/[0.06] hover:border-white/10 text-gray-300'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium truncate">{chat.title}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px] text-gray-600">{formatRelativeTime(chat.updatedAt)}</span>
                    {chat.messages.length > 0 && (
                      <span className="text-[10px] text-gray-600">{chat.messages.length} Nachrichten</span>
                    )}
                  </div>
                </div>
                <button
                  onClick={(e) => deleteChat(chat.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 text-gray-600 hover:text-red-400 transition-all flex-shrink-0"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </button>
            ))}
          </>
        )}
      </div>
    </div>
  );

  const renderChatView = () => (
    <div className="flex flex-col h-full">
      {/* Chat header */}
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
              {activeChat?.title || 'KI-Coach'}
            </div>
            {pageContent && (
              <div className="text-[10px] text-gray-500 truncate max-w-[160px]">
                {pageContent.split('\n')[0]?.slice(0, 35)}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 pointer-events-auto flex-shrink-0">
          <button
            onClick={() => setView('chatList')}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all"
            title="Chatverläufe"
          >
            <MessageSquare className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={createNewChat}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-purple-300 transition-all"
            title="Neuer Chat"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
          {!isMaximized && (
            <button
              onClick={() => setIsMaximized(true)}
              className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
          {isMaximized && (
            <button
              onClick={() => setIsMaximized(false)}
              className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            >
              <Minimize2 className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={() => setIsOpen(false)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {(!activeChat || activeChat.messages.length === 0) && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 py-8">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center"
              style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}33, ${currentTheme.colors.secondary}1a)` }}>
              <Brain className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <p className="text-gray-300 text-sm font-medium">Hallo! Ich bin dein KI-Coach.</p>
              <p className="text-gray-600 text-xs mt-1">
                {pageContent
                  ? 'Ich kenne deinen aktuellen Seiteninhalt und helfe dir gerne weiter.'
                  : 'Stelle mir eine Frage zu FrameTrain.'}
              </p>
            </div>
            {pageContent && (
              <div className="w-full max-w-xs px-3 py-2 bg-white/[0.03] border border-white/5 rounded-xl">
                <div className="flex items-center gap-1.5 mb-1">
                  <FileSearch className="w-3 h-3 text-purple-400" />
                  <span className="text-[10px] text-purple-300/70 font-medium">Geladener Kontext</span>
                </div>
                <p className="text-[10px] text-gray-600 leading-relaxed truncate">
                  {pageContent.split('\n')[0]}
                </p>
              </div>
            )}
          </div>
        )}

        {activeChat?.messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' ? (
              <div className="max-w-[88%] space-y-1">
                {/* Thinking steps for this message */}
                {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                  <ThinkingBlock
                    steps={msg.thinkingSteps}
                    isActive={false}
                    collapsed={msg.thinkingCollapsed ?? true}
                    onToggle={() => toggleMessageThinking(msg.id)}
                  />
                )}
                {/* Message content */}
                <div className="px-3 py-2.5 rounded-2xl rounded-tl-sm bg-white/[0.06] border border-white/[0.08] text-gray-200">
                  <MarkdownText text={msg.content} />
                </div>
              </div>
            ) : (
              <div className="max-w-[85%] px-3 py-2.5 rounded-2xl rounded-tr-sm text-white text-sm leading-relaxed"
                style={{ background: themeGradient }}>
                {msg.content}
              </div>
            )}
          </div>
        ))}

        {/* Active thinking steps */}
        {isLoading && thinkingSteps.length > 0 && (
          <div className="flex justify-start">
            <div className="max-w-[88%]">
              <ThinkingBlock
                steps={thinkingSteps}
                isActive={true}
                collapsed={thinkingCollapsed}
                onToggle={() => setThinkingCollapsed(c => !c)}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-start">
            <div className="max-w-[88%] px-3 py-2.5 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-start gap-2">
              <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
              <span className="text-red-300 text-xs leading-relaxed break-words">{error}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 pb-3 pt-2 flex-shrink-0 border-t border-white/5">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Frage stellen... (Enter senden)"
            disabled={isLoading}
            rows={1}
            className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-xs placeholder-gray-600 focus:outline-none focus:border-purple-500/40 disabled:opacity-50 resize-none leading-relaxed"
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
            className="p-2 rounded-xl text-white flex-shrink-0 transition-all disabled:opacity-40 hover:opacity-90 active:scale-95"
            style={{ background: themeGradient }}
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
        <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl h-[85vh] overflow-hidden flex flex-col shadow-2xl">
          {content}
        </div>
      </div>
    );
  }

  // ── Floating window ──
  return (
    <div
      className="fixed bg-slate-900/95 backdrop-blur-md rounded-2xl border border-white/[0.08] shadow-2xl overflow-hidden flex flex-col z-50"
      style={{
        width: `${size.width}px`,
        height: `${size.height}px`,
        left: `${position.x}px`,
        top: `${position.y}px`,
        boxShadow: '0 25px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)',
      }}
    >
      {content}

      {/* Resize handle */}
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
