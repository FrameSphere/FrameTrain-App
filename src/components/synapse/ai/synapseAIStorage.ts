export type SynapseAIMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  ts: number;
};

export type SynapseAIChat = {
  id: string;
  title: string;
  messages: SynapseAIMessage[];
  createdAt: number;
  updatedAt: number;
};

const BASE_KEY = 'ft_synapse_ai_chats_v1';
const MAX_CHATS = 40;

function storageKey(userId?: string) {
  return userId ? `${BASE_KEY}_${userId}` : BASE_KEY;
}

export function loadSynapseAIChats(userId?: string): SynapseAIChat[] {
  try {
    const raw = localStorage.getItem(storageKey(userId));
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveSynapseAIChats(chats: SynapseAIChat[], userId?: string) {
  try {
    localStorage.setItem(storageKey(userId), JSON.stringify(chats.slice(0, MAX_CHATS)));
  } catch {
    // ignore
  }
}

export function freshSynapseChat(): SynapseAIChat {
  return {
    id: `syn_chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title: 'Synapse Chat',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

