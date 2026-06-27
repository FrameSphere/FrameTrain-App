import { createContext, useContext, useState, ReactNode } from 'react'
import de from '../locales/de.json'
import en from '../locales/en.json'

// ── Typen ────────────────────────────────────────────────────────
export type Language = 'de' | 'en'

export const LANGUAGE_META: Record<Language, { label: string; flag: string; nativeLabel: string }> = {
  de: { label: 'Deutsch',  flag: '🇩🇪', nativeLabel: 'Deutsch' },
  en: { label: 'English',  flag: '🇬🇧', nativeLabel: 'English' },
}

const LS_KEY = 'ft_language'

const LOCALES: Record<Language, Record<string, unknown>> = { de, en }

function replacePlaceholders(template: string, params: Record<string, string | number>): string {
  return Object.entries(params).reduce(
    (acc, [placeholder, replacement]) => acc.split(`{${placeholder}}`).join(String(replacement)),
    template,
  )
}

// ── Context ──────────────────────────────────────────────────────
interface LanguageContextValue {
  language: Language
  setLanguage: (lang: Language) => void
  /**
   * t(key) – Übersetzungsfunktion.
   * Punkt-separierter Key, z. B. 'sidebar.nav.training'.
   * Zweites Argument kann entweder ein Fallback-String oder ein Platzhalter-Objekt sein.
   */
  t: (key: string, paramsOrFallback?: string | Record<string, string | number>) => string
}

const LanguageContext = createContext<LanguageContextValue>({
  language: 'de',
  setLanguage: () => {},
  t: (_key, paramsOrFallback) => typeof paramsOrFallback === 'string' ? paramsOrFallback : _key,
})

// ── Provider ─────────────────────────────────────────────────────
export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() => {
    const stored = localStorage.getItem(LS_KEY) as Language | null
    return stored === 'en' ? 'en' : 'de'
  })

  const setLanguage = (lang: Language) => {
    localStorage.setItem(LS_KEY, lang)
    setLanguageState(lang)
  }

  const t = (key: string, paramsOrFallback?: string | Record<string, string | number>): string => {
    const value = key
      .split('.')
      .reduce((obj: unknown, k: string) =>
        obj != null && typeof obj === 'object' ? (obj as Record<string, unknown>)[k] : undefined,
        LOCALES[language] as unknown,
      )
    if (typeof value === 'string') {
      if (paramsOrFallback && typeof paramsOrFallback === 'object') {
        return replacePlaceholders(value, paramsOrFallback)
      }
      return value
    }
    // Fallback zur deutschen Version wenn EN-Key fehlt
    if (language !== 'de') {
      const deFallback = key
        .split('.')
        .reduce((obj: unknown, k: string) =>
          obj != null && typeof obj === 'object' ? (obj as Record<string, unknown>)[k] : undefined,
          LOCALES['de'] as unknown,
      )
      if (typeof deFallback === 'string') {
        if (paramsOrFallback && typeof paramsOrFallback === 'object') {
          return replacePlaceholders(deFallback, paramsOrFallback)
        }
        return deFallback
      }
    }
    return typeof paramsOrFallback === 'string' ? paramsOrFallback : key
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  )
}

// ── Hook ─────────────────────────────────────────────────────────
export function useLanguage() {
  return useContext(LanguageContext)
}
