import type { AIProvider } from '../contexts/AISettingsContext';

export type ProviderMeta = {
  label: string;
  emoji: string;
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
    emoji: '🤖',
    needsKey: true,
    keyPlaceholder: 'sk-ant-api03-...',
    keyHint: 'Kostenlos testen: console.anthropic.com',
    keyLink: 'https://console.anthropic.com',
    models: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
    defaultModel: 'claude-haiku-4-5',
  },
  openai: {
    label: 'GPT-4o (OpenAI)',
    emoji: '🟢',
    needsKey: true,
    keyPlaceholder: 'sk-...',
    keyHint: 'platform.openai.com/api-keys',
    keyLink: 'https://platform.openai.com/api-keys',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
    defaultModel: 'gpt-4o-mini',
  },
  groq: {
    label: 'Groq',
    emoji: '⚡',
    needsKey: true,
    keyPlaceholder: 'gsk_...',
    keyHint: 'console.groq.com',
    keyLink: 'https://console.groq.com',
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
    defaultModel: 'llama-3.3-70b-versatile',
  },
  ollama: {
    label: 'Ollama (Lokal)',
    emoji: '🦙',
    needsKey: false,
    keyPlaceholder: '',
    keyHint: '✅ Kein Account nötig — ollama.com installieren',
    keyLink: 'https://ollama.com',
    models: ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5'],
    defaultModel: 'llama3.2',
  },
};

export function resolveModel(provider: AIProvider, selectedModel?: string, ollamaModel?: string): string {
  if (provider === 'ollama') return (ollamaModel || selectedModel || PROVIDER_META.ollama.defaultModel).trim();
  return (selectedModel || PROVIDER_META[provider].defaultModel).trim();
}
