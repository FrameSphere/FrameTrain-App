import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

/**
 * Zentrale KI-Einstellungen für die gesamte App
 * - Gibt den AI-Provider vor (Anthropic, OpenAI, Groq, Ollama)
 * - Wird von TrainingPanel, AnalysisPanel, LaboratoryPanel und FloatingAICoach genutzt
 */

export type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';

export interface AISettings {
  enabled: boolean;
  provider: AIProvider;
  apiKey: string;
  selectedModel: string;
  ollamaModel: string; // Nur für Ollama
}

interface AISettingsContextType {
  settings: AISettings;
  updateSettings: (updates: Partial<AISettings>) => void;
  resetSettings: () => void;
}

const AISettingsContext = createContext<AISettingsContextType | undefined>(undefined);

const DEFAULT_SETTINGS: AISettings = {
  enabled: false,
  provider: 'ollama',
  apiKey: '',
  selectedModel: 'llama3.2',
  ollamaModel: 'llama3.2',
};

export function AISettingsProvider({ children, userId }: { children: ReactNode; userId?: string }) {
  const [settings, setSettings] = useState<AISettings>(DEFAULT_SETTINGS);

  // FIX: Key pro User, damit AI-Keys nicht zwischen Accounts geteilt werden
  const storageKey = userId ? `ft_ai_settings_${userId}` : 'ft_ai_settings';

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try {
        setSettings(JSON.parse(stored));
      } catch {
        setSettings(DEFAULT_SETTINGS);
      }
    } else {
      // Fallback: legacy key ohne userId migrieren
      const legacy = localStorage.getItem('ft_ai_settings');
      if (legacy && userId) {
        try {
          const parsed = JSON.parse(legacy);
          setSettings(parsed);
          localStorage.setItem(storageKey, legacy);
          localStorage.removeItem('ft_ai_settings');
        } catch { /* ignore */ }
      } else {
        setSettings(DEFAULT_SETTINGS);
      }
    }
  }, [storageKey]);

  const updateSettings = (updates: Partial<AISettings>) => {
    setSettings(prev => {
      const updated = { ...prev, ...updates };
      localStorage.setItem(storageKey, JSON.stringify(updated));
      return updated;
    });
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    localStorage.setItem(storageKey, JSON.stringify(DEFAULT_SETTINGS));
  };

  return (
    <AISettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </AISettingsContext.Provider>
  );
}

export function useAISettings() {
  const context = useContext(AISettingsContext);
  if (!context) {
    throw new Error('useAISettings must be used within AISettingsProvider');
  }
  return context;
}
