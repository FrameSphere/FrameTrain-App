import type { AIProvider } from '../contexts/AISettingsContext';

export type ProviderMeta = {
  /** Fallback-Anzeige. Reine Markennamen brauchen keine Übersetzung. */
  label: string;
  /** Gesetzt, wenn das Label übersetzbare Anteile enthält (z. B. „Lokal“). */
  labelKey?: string;
  needsKey: boolean;
  keyPlaceholder: string;
  keyHint: string;
  keyLink: string;
  models: string[];
  defaultModel: string;
};

export const PROVIDER_META: Record<AIProvider, ProviderMeta> = {
  anthropic: {
    label: 'Claude (Anthropic)',
    needsKey: true,
    keyPlaceholder: 'sk-ant-api03-...',
    keyHint: 'Kostenlos testen: console.anthropic.com',
    keyLink: 'https://console.anthropic.com',
    // Aktuelle Modell-IDs (Stand 2026): opus-4-5/sonnet-4-5 sind ausgelaufen.
    models: ['claude-opus-5', 'claude-sonnet-5', 'claude-haiku-4-5'],
    defaultModel: 'claude-haiku-4-5',
  },
  openai: {
    label: 'GPT-4o (OpenAI)',
    needsKey: true,
    keyPlaceholder: 'sk-...',
    keyHint: 'platform.openai.com/api-keys',
    keyLink: 'https://platform.openai.com/api-keys',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
    defaultModel: 'gpt-4o-mini',
  },
  groq: {
    label: 'Groq',
    needsKey: true,
    keyPlaceholder: 'gsk_...',
    keyHint: 'console.groq.com',
    keyLink: 'https://console.groq.com',
    // Groq mustert Modelle regelmäßig aus. Deshalb hier ausschliesslich
    // PRODUKTIONS-Modelle (nicht die Preview-Modelle wie qwen3.6-27b, die
    // laut Groq nur "for evaluation purposes" laufen und jederzeit verschwinden
    // — genau das legte frueher die gesamte KI lahm). Falls Groq doch eines
    // ausmustert, kann der Nutzer im Freitextfeld ein aktuelles Modell setzen.
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'groq/compound-mini'],
    defaultModel: 'openai/gpt-oss-20b',
  },
  ollama: {
    label: 'Ollama (Lokal)',
    labelKey: 'settings.ai.providerLabelOllama',
    needsKey: false,
    keyPlaceholder: '',
    keyHint: 'Kein Account nötig — ollama.com installieren',
    keyLink: 'https://ollama.com',
    models: ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5'],
    defaultModel: 'llama3.2',
  },
};

export function resolveModel(provider: AIProvider, selectedModel?: string, ollamaModel?: string): string {
  if (provider === 'ollama') return (ollamaModel || selectedModel || PROVIDER_META.ollama.defaultModel).trim();
  return (selectedModel || PROVIDER_META[provider].defaultModel).trim();
}
