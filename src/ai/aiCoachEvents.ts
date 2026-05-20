export type AICoachOpenDetail = {
  prefill?: string;
  newChat?: boolean;
  titleHint?: string;
};

const EVENT_NAME = 'ft_ai_coach_open';

export function openAICoach(detail: AICoachOpenDetail = {}) {
  try {
    window.dispatchEvent(new CustomEvent<AICoachOpenDetail>(EVENT_NAME, { detail }));
  } catch {
    // ignore
  }
}

export function onOpenAICoach(handler: (detail: AICoachOpenDetail) => void) {
  const listener = (e: Event) => handler((e as CustomEvent<AICoachOpenDetail>).detail || {});
  window.addEventListener(EVENT_NAME, listener as EventListener);
  return () => window.removeEventListener(EVENT_NAME, listener as EventListener);
}

