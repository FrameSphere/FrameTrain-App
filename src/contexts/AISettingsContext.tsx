import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef, ReactNode } from 'react';
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
export type TokenBudget = 'minimal' | 'balanced' | 'quality' | 'max' | 'unlimited';

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
  unlimited: {
    label: 'Unlimited',
    // Bewusst sehr hoch: lässt die KI ohne künstliche Kürzung arbeiten. Bei
    // Anthropic wird zusätzlich effort:'max' gesetzt (siehe aiClient). Nur für
    // bezahlte APIs / Abo-Token mit hohem Limit sinnvoll — höchste Kosten/Zeit.
    maxTokens: 16000,
    historyTokenBudget: 8000,
    synapseMaxTokens: 16000,
    description: 'Keine Restriktionen — maximale Länge, Tiefe und Reasoning-Effort. Nur für bezahlte APIs / Abo-Token; höchster Verbrauch.',
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
  /**
   * Anbieter, für die bereits ein Key hinterlegt ist. Der Wechsel zwischen
   * Anbietern verliert keinen Key mehr — die Oberfläche zeigt damit an,
   * wo schon einer liegt.
   */
  providersWithKey: KeyProvider[];
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

/**
 * Konto-Name im Schlüsselbund — pro Nutzer UND pro Anbieter getrennt.
 *
 * Vorher teilten sich alle Anbieter ein Konto: wer von Claude auf Groq
 * wechselte und dort seinen Groq-Key speicherte, hatte den Claude-Key
 * verloren und musste ihn beim Zurueckwechseln neu eintippen. Jetzt behaelt
 * jeder Anbieter seinen eigenen Key, und das Umschalten holt ihn zurueck.
 */
const secretAccount = (userId: string | undefined, provider: AIProvider) =>
  `ft_ai_key_${userId || 'anon'}_${provider}`;

/** Altes gemeinsames Konto — wird beim ersten Start auf den Anbieter umgezogen. */
const legacyAccount = (userId?: string) => `ft_ai_key_${userId || 'anon'}`;

/** Anbieter, fuer die ueberhaupt ein Key im Schluesselbund liegt. */
export type KeyProvider = Exclude<AIProvider, 'ollama'>;

const KEY_PROVIDERS: KeyProvider[] = ['anthropic', 'openai', 'groq'];

function needsKey(provider: AIProvider): provider is KeyProvider {
  return provider !== 'ollama';
}

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

  /**
   * Bereits geladene / eingetippte Keys je Anbieter.
   *
   * Haelt den Wechsel zwischen Anbietern verlustfrei: wer von Claude auf Groq
   * schaltet, findet beim Zurueckwechseln seinen Claude-Key wieder vor —
   * auch dann, wenn er zwischendurch nichts gespeichert hat.
   */
  const [keysByProvider, setKeysByProvider] = useState<Partial<Record<KeyProvider, string>>>({});

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
      // 2) Nur Key-Provider brauchen den Schlüsselbund. Ollama (der Default!)
      //    nie — dadurch sehen Standard-Nutzer den macOS-Schlüsselbund-Dialog
      //    gar nicht erst. Der Zugriff wird erst ausgelöst, wenn wirklich ein
      //    Key-Provider (Claude/OpenAI/Groq) gespeichert ist.
      if (base.provider === 'ollama') {
        const loaded: AISettings = { ...base, apiKey: '' };
        persistNonSecret(loaded);
        if (cancelled) return;
        setSettings(loaded);
        setDraft(loaded);
        setKeyLoading(false);
        return;
      }

      const provider = base.provider as KeyProvider;
      // Key aus dem Schlüsselbund holen — aus dem Konto DIESES Anbieters.
      const got = await keychainGet(secretAccount(userId, provider));
      if (cancelled) return;

      let apiKey = '';
      if (got.ok) {
        setKeychainAvailable(true);
        if (got.value) {
          apiKey = got.value;
        } else {
          // 3a) Umzug vom frueheren gemeinsamen Konto auf das Anbieter-Konto.
          //     Der alte Eintrag bleibt vorerst liegen: aeltere App-Versionen
          //     auf demselben Rechner lesen ihn noch.
          const shared = await keychainGet(legacyAccount(userId));
          if (cancelled) return;
          if (shared.ok && shared.value) {
            apiKey = shared.value;
            await keychainWrite(secretAccount(userId, provider), shared.value);
            if (cancelled) return;
          } else if (legacyKey) {
            // 3b) Alt-Key aus localStorage in den Schlüsselbund migrieren
            const migrated = await keychainWrite(secretAccount(userId, provider), legacyKey);
            if (cancelled) return;
            apiKey = legacyKey;
            setKeychainAvailable(migrated);
          }
        }
      } else {
        // Schlüsselbund nicht verfügbar → Key nur für diese Sitzung im Speicher
        setKeychainAvailable(false);
        apiKey = legacyKey;
      }
      if (apiKey) setKeysByProvider(prev => ({ ...prev, [provider]: apiKey }));

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
  }, [storageKey, userId]);

  const updateDraft = useCallback((updates: Partial<AISettings>) => {
    setDraft(prev => {
      const next = { ...prev, ...updates };
      // Der eingetippte Key gehoert zum AKTUELLEN Anbieter — merken, bevor
      // umgeschaltet wird.
      if (needsKey(prev.provider) && typeof updates.apiKey === 'string') {
        const typed = updates.apiKey;
        setKeysByProvider(cache => ({ ...cache, [prev.provider as KeyProvider]: typed }));
      }
      // Anbieterwechsel: den Key des NEUEN Anbieters einsetzen (aus dem
      // Zwischenspeicher; sonst holt ihn der Effekt unten aus dem
      // Schluesselbund). Der Key des alten Anbieters bleibt erhalten.
      if (updates.provider && updates.provider !== prev.provider) {
        if (needsKey(prev.provider) && prev.apiKey.trim()) {
          const previous = prev.apiKey;
          setKeysByProvider(cache => ({ ...cache, [prev.provider as KeyProvider]: previous }));
        }
        next.apiKey = needsKey(updates.provider)
          ? (keysByProvider[updates.provider as KeyProvider] ?? '')
          : '';
      }
      return next;
    });
  }, [keysByProvider]);

  /**
   * Key des gewaehlten Anbieters nachladen.
   *
   * Beim Start wird der Schluesselbund bewusst nur angefasst, wenn der
   * GESPEICHERTE Anbieter einen Key braucht (sonst saehe jeder Ollama-Nutzer
   * den macOS-Dialog). Wer danach im Dialog auf Claude oder Groq umstellt,
   * stand deshalb vor einem leeren Key-Feld, obwohl der Key im Schluesselbund
   * lag — jetzt wird er pro Anbieter genau einmal nachgeholt.
   */
  const fetchedAccounts = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (keyLoading) return;
    const provider = draft.provider;
    if (!needsKey(provider)) return;
    if (draft.apiKey.trim()) return;
    if (keysByProvider[provider] !== undefined) return;
    const acc = secretAccount(userId, provider);
    if (fetchedAccounts.current.has(acc)) return;
    fetchedAccounts.current.add(acc);
    let cancelled = false;
    (async () => {
      const got = await keychainGet(acc);
      if (cancelled || !got.ok) return;
      const value = got.value ?? '';
      setKeysByProvider(cache => ({ ...cache, [provider]: value }));
      if (!value) return;
      setDraft(prev => (prev.provider === provider && !prev.apiKey.trim() ? { ...prev, apiKey: value } : prev));
    })();
    return () => { cancelled = true; };
  }, [draft.provider, draft.apiKey, keysByProvider, userId, keyLoading]);

  const saveSettings = useCallback(async () => {
    const next = draft;
    persistNonSecret(next);
    const trimmed = next.apiKey.trim();
    // Gespeichert wird immer nur das Konto des GEWAEHLTEN Anbieters. Die Keys
    // der anderen Anbieter bleiben unberuehrt im Schluesselbund liegen.
    //
    // Ein leeres Feld loeschte den Key frueher bedingungslos — auch dann,
    // wenn nie einer geladen war (etwa direkt nach einem Anbieterwechsel).
    // Geloescht wird jetzt nur, wenn ein geladener Key bewusst geleert wurde.
    let wrote = true;
    if (needsKey(next.provider)) {
      const acc = secretAccount(userId, next.provider);
      const known = keysByProvider[next.provider];
      if (trimmed) {
        wrote = await keychainWrite(acc, trimmed);
      } else if ((known ?? '').trim() || (settings.provider === next.provider && settings.apiKey.trim())) {
        wrote = await keychainWrite(acc, '');
      }
      setKeysByProvider(cache => ({ ...cache, [next.provider as KeyProvider]: trimmed }));
    }
    setKeychainAvailable(wrote || !trimmed);
    setSettings({ ...next, apiKey: trimmed });
    setDraft({ ...next, apiKey: trimmed });
  }, [draft, userId, persistNonSecret, settings.apiKey, settings.provider, keysByProvider]);

  const discardDraft = useCallback(() => {
    setDraft(settings);
  }, [settings]);

  const resetSettings = useCallback(async () => {
    persistNonSecret(DEFAULT_SETTINGS);
    // Zuruecksetzen heisst: ALLE hinterlegten Keys weg, nicht nur der des
    // gerade gewaehlten Anbieters.
    for (const provider of KEY_PROVIDERS) {
      await keychainWrite(secretAccount(userId, provider), '');
    }
    await keychainWrite(legacyAccount(userId), '');
    setKeysByProvider({});
    fetchedAccounts.current.clear();
    setSettings(DEFAULT_SETTINGS);
    setDraft(DEFAULT_SETTINGS);
  }, [userId, persistNonSecret]);

  const isDirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(settings),
    [draft, settings],
  );

  /** Anbieter, fuer die bereits ein Key hinterlegt ist (fuer die Anzeige). */
  const providersWithKey = useMemo<KeyProvider[]>(() => {
    const set = new Set<KeyProvider>();
    for (const p of KEY_PROVIDERS) if ((keysByProvider[p] ?? '').trim()) set.add(p);
    if (needsKey(draft.provider) && draft.apiKey.trim()) set.add(draft.provider);
    if (needsKey(settings.provider) && settings.apiKey.trim()) set.add(settings.provider);
    return KEY_PROVIDERS.filter(p => set.has(p));
  }, [keysByProvider, draft.provider, draft.apiKey, settings.provider, settings.apiKey]);

  const value = useMemo<AISettingsContextType>(() => ({
    settings,
    draft,
    isDirty,
    keyLoading,
    keychainAvailable,
    providersWithKey,
    updateDraft,
    updateSettings: updateDraft,
    saveSettings,
    discardDraft,
    resetSettings,
  }), [settings, draft, isDirty, keyLoading, keychainAvailable, providersWithKey, updateDraft, saveSettings, discardDraft, resetSettings]);

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
