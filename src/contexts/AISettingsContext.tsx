import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * Zentrale KI-Einstellungen für die gesamte App
 * - Gibt den AI-Provider vor (Anthropic, OpenAI, Groq, Ollama)
 * - Wird von TrainingPanel, AnalysisPanel, LaboratoryPanel und FloatingAICoach genutzt
 *
 * Zwei-Ebenen-Modell (Draft + Committed):
 *   - `settings`  = gespeicherter Stand. NUR dieser wird von KI-Aufrufen genutzt.
 *   - `draft`     = Bearbeitungsstand in den Einstellungen. Änderungen greifen
 *                   erst nach `saveSettings()` — nichts wird ohne Speichern aktiv.
 *
 * Sicherheit:
 *   - Der API-Key liegt NICHT im localStorage, sondern im OS-Schlüsselbund
 *     (macOS Keychain / Windows Credential Manager) über die Tauri-Commands
 *     `secret_get/set/delete`. Im localStorage steht nur noch Nicht-Geheimes
 *     (Provider, Modell, Token-Budget, an/aus).
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
  /** Gespeicherter Stand — von KI-Aufrufen genutzt. */
  settings: AISettings;
  /** Bearbeitungsstand in den Einstellungen (noch nicht wirksam). */
  draft: AISettings;
  /** true, wenn der Draft vom gespeicherten Stand abweicht. */
  isDirty: boolean;
  /** true, solange der Key beim Start aus dem Schlüsselbund geladen wird. */
  keyLoading: boolean;
  /** false, wenn der OS-Schlüsselbund nicht erreichbar ist (z.B. außerhalb Tauri). */
  keychainAvailable: boolean;
  /** Ändert nur den Draft — wird erst mit saveSettings() wirksam. */
  updateDraft: (updates: Partial<AISettings>) => void;
  /** Alias auf updateDraft (Rückwärtskompatibilität). */
  updateSettings: (updates: Partial<AISettings>) => void;
  /** Übernimmt den Draft: persistiert Nicht-Geheimes + Key in den Schlüsselbund. */
  saveSettings: () => Promise<void>;
  /** Verwirft den Draft und stellt den gespeicherten Stand wieder her. */
  discardDraft: () => void;
  /** Setzt alles auf Standardwerte zurück (inkl. Key-Löschung) und speichert. */
  resetSettings: () => Promise<void>;
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

/** Konto-Name im Schlüsselbund, pro Nutzer getrennt. */
const secretAccount = (userId?: string) => `ft_ai_key_${userId || 'anon'}`;

async function keychainGet(account: string): Promise<{ ok: true; value: string | null } | { ok: false }> {
  try {
    const value = await invoke<string | null>('secret_get', { key: account });
    return { ok: true, value: value ?? null };
  } catch (e) {
    console.warn('[AISettings] Schlüsselbund-Lesen fehlgeschlagen:', e);
    return { ok: false };
  }
}

async function keychainWrite(account: string, value: string): Promise<boolean> {
  try {
    if (value) await invoke('secret_set', { key: account, value });
    else await invoke('secret_delete', { key: account });
    return true;
  } catch (e) {
    console.warn('[AISettings] Schlüsselbund-Schreiben fehlgeschlagen:', e);
    return false;
  }
}

export function AISettingsProvider({ children, userId }: { children: ReactNode; userId?: string }) {
  const [settings, setSettings] = useState<AISettings>(DEFAULT_SETTINGS);
  const [draft, setDraft] = useState<AISettings>(DEFAULT_SETTINGS);
  const [keyLoading, setKeyLoading] = useState(true);
  const [keychainAvailable, setKeychainAvailable] = useState(true);

  // Key pro User, damit AI-Keys nicht zwischen Accounts geteilt werden
  const storageKey = userId ? `ft_ai_settings_${userId}` : 'ft_ai_settings';
  const account = secretAccount(userId);

  /** Persistiert ausschließlich Nicht-Geheimes (Key wird geleert abgelegt). */
  const persistNonSecret = useCallback((s: AISettings) => {
    const { apiKey: _drop, ...rest } = s;
    void _drop;
    localStorage.setItem(storageKey, JSON.stringify({ ...rest, apiKey: '' }));
  }, [storageKey]);

  // Laden beim Mount / Nutzerwechsel
  useEffect(() => {
    let cancelled = false;
    setKeyLoading(true);

    // 1) Nicht-geheime Einstellungen aus localStorage (inkl. Legacy-Migration)
    let base: AISettings = { ...DEFAULT_SETTINGS };
    let legacyKey = '';
    const readBlob = (raw: string | null) => {
      if (!raw) return false;
      try {
        const parsed = JSON.parse(raw) as Partial<AISettings>;
        legacyKey = (parsed.apiKey || '').trim();
        base = { ...DEFAULT_SETTINGS, ...parsed, apiKey: '' };
        return true;
      } catch {
        return false;
      }
    };

    if (!readBlob(localStorage.getItem(storageKey))) {
      // Fallback: Legacy-Key ohne userId migrieren
      const legacy = localStorage.getItem('ft_ai_settings');
      if (legacy && userId && readBlob(legacy)) {
        localStorage.removeItem('ft_ai_settings');
      }
    }

    (async () => {
      // 2) Key aus dem Schlüsselbund holen
      const got = await keychainGet(account);
      if (cancelled) return;

      let apiKey = '';
      if (got.ok) {
        setKeychainAvailable(true);
        if (got.value) {
          apiKey = got.value;
        } else if (legacyKey) {
          // 3) Alt-Key aus localStorage in den Schlüsselbund migrieren
          const migrated = await keychainWrite(account, legacyKey);
          if (cancelled) return;
          apiKey = legacyKey;
          setKeychainAvailable(migrated);
        }
      } else {
        // Schlüsselbund nicht verfügbar → Key nur für diese Sitzung im Speicher
        setKeychainAvailable(false);
        apiKey = legacyKey;
      }

      const loaded: AISettings = { ...base, apiKey };
      // localStorage von jeglichem Klartext-Key säubern
      persistNonSecret(loaded);

      if (cancelled) return;
      setSettings(loaded);
      setDraft(loaded);
      setKeyLoading(false);
    })();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey, account]);

  const updateDraft = useCallback((updates: Partial<AISettings>) => {
    setDraft(prev => ({ ...prev, ...updates }));
  }, []);

  const saveSettings = useCallback(async () => {
    const next = draft;
    persistNonSecret(next);
    const wrote = await keychainWrite(account, next.apiKey.trim());
    setKeychainAvailable(wrote || !next.apiKey.trim());
    setSettings({ ...next, apiKey: next.apiKey.trim() });
    setDraft({ ...next, apiKey: next.apiKey.trim() });
  }, [draft, account, persistNonSecret]);

  const discardDraft = useCallback(() => {
    setDraft(settings);
  }, [settings]);

  const resetSettings = useCallback(async () => {
    persistNonSecret(DEFAULT_SETTINGS);
    await keychainWrite(account, '');
    setSettings(DEFAULT_SETTINGS);
    setDraft(DEFAULT_SETTINGS);
  }, [account, persistNonSecret]);

  const isDirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(settings),
    [draft, settings],
  );

  const value = useMemo<AISettingsContextType>(() => ({
    settings,
    draft,
    isDirty,
    keyLoading,
    keychainAvailable,
    updateDraft,
    updateSettings: updateDraft,
    saveSettings,
    discardDraft,
    resetSettings,
  }), [settings, draft, isDirty, keyLoading, keychainAvailable, updateDraft, saveSettings, discardDraft, resetSettings]);

  return (
    <AISettingsContext.Provider value={value}>
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
