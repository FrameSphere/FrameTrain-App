import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

/**
 * Zentrale KI-Einstellungen für die gesamte App
 * - Gibt den AI-Provider vor (Anthropic, OpenAI, Groq, Ollama)
 * - Wird von TrainingPanel, AnalysisPanel, LaboratoryPanel und FloatingAICoach genutzt
 */

export type AIProvider = 'anthropic' | 'openai' | 'groq' | 'ollama';
export type TokenBudget = 'minimal' | 'balanced' | 'quality' | 'max';

export const TOKEN_BUDGET_CONFIG: Record<TokenBudget, {
  label: string;
  maxTokens: number;
  historyTokenBudget: number;
  synapseMaxTokens: number;
  description: string;
}> = {
  minimal: {
    label: 'Minimal',
    maxTokens: 400,
    historyTokenBudget: 800,
    synapseMaxTokens: 1500,
    description: 'Sehr kurze Antworten, minimaler Verbrauch. Ideal für Groq Free Tier.',
  },
  balanced: {
    label: 'Balanced',
    maxTokens: 800,
    historyTokenBudget: 1500,
    synapseMaxTokens: 3000,
    description: 'Gute Balance aus Qualität und Token-Verbrauch. Empfohlen für die meisten User.',
  },
  quality: {
    label: 'Quality',
    maxTokens: 1500,
    historyTokenBudget: 2500,
    synapseMaxTokens: 5000,
    description: 'Ausführliche Antworten mit mehr Kontext. Für Claude / GPT-4 empfohlen.',
  },
  max: {
    label: 'Maximum',
    maxTokens: 3000,
    historyTokenBudget: 4000,
    synapseMaxTokens: 8000,
    description: 'Maximale Qualität und Tiefe. Nur für bezahlte APIs mit hohem Rate-Limit.',
  },
};

export interface AISettings {
  enabled: boolean;
  provider: AIProvider;
  apiKey: string;
  selectedModel: string;
  ollamaModel: string;
  tokenBudget: TokenBudget;
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
  tokenBudget: 'balanced',
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
