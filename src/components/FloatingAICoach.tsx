import { useState, useRef, useEffect } from 'react';
import { X, Send, Loader2, AlertCircle, CheckCircle, Maximize2, Minimize2, Brain } from 'lucide-react';
import { useAISettings, AIProvider } from '../contexts/AISettingsContext';
import { usePageContext } from '../contexts/PageContext';
import { useTheme } from '../contexts/ThemeContext';
import { 
  AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS,
  getRelevantKnowledge,
  formatKnowledgeForContext 
} from '../contexts/AIKnowledgeBaseSmart';

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

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface FloatingAICoachProps {
  currentPageContent?: string; // Kontext der aktuellen Seite
}

export default function FloatingAICoach({ currentPageContent }: FloatingAICoachProps) {
  const { settings } = useAISettings();
  const { currentTheme } = useTheme();
  const { currentPageContent: contextPageContent } = usePageContext();
  
  // Nutze Page Context wenn verfügbar, ansonsten Props
  const pageContent = contextPageContent || currentPageContent;
  const [isOpen, setIsOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Load initial position from localStorage, otherwise calculate safe default
  const getInitialPosition = () => {
    try {
      const saved = localStorage.getItem('aiCoachPosition');
      if (saved) {
        const parsed = JSON.parse(saved);
        return parsed;
      }
    } catch (e) {
      // Fall through to default
    }
    // Safe default: bottom right with padding to stay visible
    return { x: window.innerWidth - 400, y: window.innerHeight - 540 };
  };

  // Floating position state
  const [position, setPosition] = useState(getInitialPosition());
  const [size, setSize] = useState({ width: 360, height: 500 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, startWidth: 0, startHeight: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle mouse move for dragging/resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y,
        });
      } else if (isResizing) {
        const deltaX = e.clientX - resizeStart.x;
        const deltaY = e.clientY - resizeStart.y;
        setSize({
          width: Math.max(280, resizeStart.startWidth + deltaX),
          height: Math.max(300, resizeStart.startHeight + deltaY),
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
  }, [isDragging, isResizing, dragOffset, resizeStart]);

  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return; // Dont drag if clicking a button
    setIsDragging(true);
    setDragOffset({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    });
  };

  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({
      x: e.clientX,
      y: e.clientY,
      startWidth: size.width,
      startHeight: size.height,
    });
  };

  // Save position to localStorage when it changes
  useEffect(() => {
    if (isOpen && !isMaximized) {
      localStorage.setItem('aiCoachPosition', JSON.stringify(position));
    }
  }, [position, isOpen, isMaximized]);

  const buildSystemPrompt = (userMessage?: string): string => {
    let prompt = AI_SYSTEM_PROMPT_WITH_INSTRUCTIONS;
    
    // Wenn Nutzer eine Frage hat: Relevante Dokumentation laden
    if (userMessage?.trim()) {
      const relevantSections = getRelevantKnowledge(userMessage);
      const formattedKnowledge = formatKnowledgeForContext(relevantSections);
      prompt += formattedKnowledge;
    }
    
    // Aktuelle Page-Context hinzufügen
    if (pageContent) {
      prompt += `\n\n### Aktueller App-Kontext:\n${pageContent.slice(0, 500)}`;
    }

    return prompt;
  };

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    if (!settings.enabled) {
      setError('KI-Assistent ist deaktiviert. Bitte aktiviere ihn in den Einstellungen.');
      return;
    }

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: inputText.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError('');

    try {
      const meta = PROVIDER_META[settings.provider];
      if (meta.needsKey && !settings.apiKey) {
        throw new Error('API-Key nicht konfiguriert. Bitte in Einstellungen einstellen.');
      }

      let responseText = '';

      if (settings.provider === 'ollama') {
        const model = settings.ollamaModel || 'llama3.2';
        const res = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            prompt: buildSystemPrompt(userMessage.content) + '\n\nUser: ' + userMessage.content + '\n\nAssistant:',
            stream: false,
            options: { temperature: 0.7, num_ctx: 2048 },
          }),
        });
        if (!res.ok) throw new Error(`Ollama nicht erreichbar. Läuft Ollama? (http://localhost:11434)`);
        const data = await res.json();
        responseText = data.response || '';

      } else if (settings.provider === 'groq') {
        const model = settings.selectedModel || 'llama-3.3-70b-versatile';
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${settings.apiKey}` },
          body: JSON.stringify({
            model,
            max_tokens: 1000,
            temperature: 0.7,
            messages: [
              { role: 'system', content: buildSystemPrompt(userMessage.content) },
              { role: 'user', content: userMessage.content },
            ],
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';

      } else if (settings.provider === 'anthropic') {
        const model = settings.selectedModel || 'claude-opus-4-5';
        const res = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': settings.apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          body: JSON.stringify({
            model,
            max_tokens: 1000,
            system: buildSystemPrompt(userMessage.content),
            messages: [{ role: 'user', content: userMessage.content }],
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
        const model = settings.selectedModel || 'gpt-4o';
        const res = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${settings.apiKey}` },
          body: JSON.stringify({
            model,
            max_tokens: 1000,
            temperature: 0.7,
            messages: [
              { role: 'system', content: buildSystemPrompt(userMessage.content) },
              { role: 'user', content: userMessage.content },
            ],
          }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e?.error?.message || `HTTP ${res.status}`);
        }
        const data = await res.json();
        responseText = data.choices?.[0]?.message?.content || '';
      }

      const assistantMessage: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: responseText,
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (e: any) {
      setError('Fehler: ' + (e?.message || 'Unbekannter Fehler'));
    } finally {
      setIsLoading(false);
    }
  };

  // If not enabled, don't show the button
  if (!settings.enabled) {
    return null;
  }

  // Minimized button (bottom right)
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className={`fixed bottom-6 right-6 px-4 py-3 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white shadow-lg hover:shadow-xl hover:scale-105 transition-all flex items-center gap-2 font-semibold z-40`}
        title="KI-Coach öffnen"
      >
        <Brain className="w-5 h-5" />
        <span>AI Coach</span>
      </button>
    );
  }

  // Maximized modal - overlay the entire app
  if (isMaximized) {
    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div
          className="bg-slate-900 rounded-2xl border border-white/10 w-full h-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
          ref={modalRef}
        >
          {/* Header */}
          <div className={`flex items-center justify-between p-5 border-b border-white/10 flex-shrink-0 bg-gradient-to-r ${currentTheme.colors.gradient} opacity-10`}>
            <div className="flex items-center gap-3">
              <Brain className="w-6 h-6 text-white" />
              <div>
                <h2 className="text-xl font-bold text-white">KI-Coach</h2>
                <p className="text-xs text-gray-400">Frag mich anything!</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsMaximized(false)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
                title="Minimieren"
              >
                <Minimize2 className="w-5 h-5" />
              </button>
              <button
                onClick={() => {
                  setIsOpen(false);
                  setMessages([]);
                  // Save final position
                  localStorage.setItem('aiCoachPosition', JSON.stringify(position));
                }}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Messages area */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Brain className="w-16 h-16 text-purple-400 mb-4" />
                <p className="text-gray-400 text-lg">Hallo! Ich bin dein KI-Coach.</p>
                <p className="text-gray-500 text-sm mt-2">Stelle mir eine Frage zu dem, was du gerade machst.</p>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-xs px-4 py-2 rounded-lg ${
                    msg.role === 'user'
                      ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white`
                      : 'bg-white/10 text-gray-100 border border-white/10'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white/10 text-gray-300 px-4 py-3 rounded-lg border border-white/10 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">KI denkt nach...</span>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-start">
                <div className="bg-red-500/10 text-red-300 px-4 py-3 rounded-lg border border-red-500/20 flex items-center gap-2 text-sm">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-white/10 bg-slate-800/50 flex-shrink-0">
            <div className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                placeholder="Deine Frage..."
                disabled={isLoading}
                className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:opacity-50"
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !inputText.trim()}
                className={`p-2 bg-gradient-to-r ${currentTheme.colors.gradient} text-white rounded-lg hover:opacity-90 transition-all disabled:opacity-50`}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Floating window
  return (
    <div
      className="fixed bg-slate-900 rounded-2xl border border-white/10 shadow-2xl overflow-hidden flex flex-col z-50"
      style={{
        width: `${size.width}px`,
        height: `${size.height}px`,
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
      ref={modalRef}
    >
      {/* Header - draggable */}
      <div
        className={`bg-gradient-to-r ${currentTheme.colors.gradient} opacity-10 border-b border-white/10 p-3 cursor-move flex items-center justify-between flex-shrink-0 select-none`}
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2 pointer-events-none">
          <Brain className="w-5 h-5 text-white" />
          <div>
            <div className="text-sm font-bold text-white">KI-Coach</div>
            <div className="text-xs text-gray-400">Quick Chat</div>
          </div>
        </div>
        <div className="flex items-center gap-1 pointer-events-auto">
          <button
            onClick={() => setIsMaximized(true)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all text-center"
            title="Maximieren"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => {
              setIsOpen(false);
              setMessages([]);
            }}
            className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all"
            title="Schließen"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Brain className="w-8 h-8 text-purple-400 mb-2" />
            <p className="text-gray-400 text-xs">Hallo! Stell mir eine Frage.</p>
          </div>
        )}
        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[75%] px-3 py-2 rounded-lg text-xs ${
                msg.role === 'user'
                  ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white`
                  : 'bg-white/10 text-gray-100 border border-white/10'
              }`}
            >
              <p className="leading-relaxed break-words">{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white/10 text-gray-300 px-3 py-2 rounded-lg border border-white/10 flex items-center gap-2 text-xs">
              <Loader2 className="w-3 h-3 animate-spin" />
              Denken...
            </div>
          </div>
        )}
        {error && (
          <div className="flex justify-start">
            <div className="bg-red-500/10 text-red-300 px-3 py-2 rounded-lg border border-red-500/20 flex items-center gap-1 text-xs col-span-full">
              <AlertCircle className="w-3 h-3 flex-shrink-0" />
              <span className="break-words">{error}</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-white/10 bg-slate-800/50 flex-shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Frage..."
            disabled={isLoading}
            className="flex-1 px-2 py-1.5 bg-white/5 border border-white/10 rounded text-white text-xs placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputText.trim()}
            className={`p-1.5 bg-gradient-to-r ${currentTheme.colors.gradient} text-white rounded hover:opacity-90 transition-all disabled:opacity-50 flex items-center justify-center`}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Resize handle (bottom right) */}
      <div
        className="absolute bottom-0 right-0 w-4 h-4 bg-gradient-to-tl from-purple-500/50 to-transparent cursor-se-resize rounded-tl"
        onMouseDown={handleResizeMouseDown}
      />
    </div>
  );
}
