// GradientChatInput – gemeinsames Chat-Eingabefeld für alle KI-Chats
// (FloatingAICoach, SynapseAIPanel, AnalysisPanel).
//
// Design nach dem "gradient chat input"-Pattern, angepasst an FrameTrain:
//  - Karte mit Innenring (p-1) und weichem Schatten
//  - Send-Button "aktiviert" sich mit Theme-Gradient sobald Text vorhanden ist
//  - Loading-Zustand: Stop-Button (wenn onStop übergeben) oder Spinner
//  - Auto-Grow-Textarea, Enter = senden, Shift+Enter = Zeilenumbruch
//  - Keine externen Dependencies (kein motion/radix) – CSS-Transitions genügen

import { forwardRef, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Loader2, Send, Square } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

export interface GradientChatInputProps {
  /** Kontrollierter Wert des Eingabefelds. */
  value: string;
  onChange: (value: string) => void;
  /** Wird bei Enter oder Klick auf Senden ausgelöst (nur wenn Text vorhanden). */
  onSend: () => void;
  /** Wenn gesetzt und loading=true, wird der Button zum Stop-Button. */
  onStop?: () => void;
  /** KI antwortet gerade – Eingabe wird gesperrt, Button zeigt Spinner/Stop. */
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  /** Kompakte Variante (FloatingAICoach) oder Standard. */
  size?: 'sm' | 'md';
  /** CSS-Gradient für den aktiven Send-Button. Fallback: aktuelles Theme. */
  gradient?: string;
  /** Hex-Farbe für den Fokus-Glow. Fallback: aktuelles Theme. */
  primaryColor?: string;
  /** Tooltip-Texte (i18n von außen). */
  sendTitle?: string;
  stopTitle?: string;
  autoFocus?: boolean;
  className?: string;
}

/** Relative Luminanz einer Hex-Farbe (0..1) – helle Theme-Farben abdunkeln. */
function hexLuminance(hex: string): number {
  const h = hex.replace('#', '');
  if (h.length < 6) return 0;
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function darkenHex(hex: string, amount: number): string {
  const h = hex.replace('#', '');
  if (h.length < 6) return hex;
  const d = (v: number) => Math.max(0, v - amount).toString(16).padStart(2, '0');
  return `#${d(parseInt(h.slice(0, 2), 16))}${d(parseInt(h.slice(2, 4), 16))}${d(parseInt(h.slice(4, 6), 16))}`;
}

const GradientChatInput = forwardRef<HTMLTextAreaElement, GradientChatInputProps>(
  function GradientChatInput(
    {
      value, onChange, onSend, onStop,
      loading = false, disabled = false,
      placeholder, size = 'md',
      gradient, primaryColor,
      sendTitle, stopTitle,
      autoFocus = false, className = '',
    },
    ref,
  ) {
    const { currentTheme } = useTheme();
    const [focused, setFocused] = useState(false);
    const innerRef = useRef<HTMLTextAreaElement | null>(null);

    const setTextareaRef = useCallback((el: HTMLTextAreaElement | null) => {
      innerRef.current = el;
      if (typeof ref === 'function') ref(el);
      else if (ref) (ref as React.MutableRefObject<HTMLTextAreaElement | null>).current = el;
    }, [ref]);

    // Theme-Farben mit Luminanz-Schutz (helle Themes wie "Ice" abdunkeln)
    const safe = useCallback((hex: string) =>
      hexLuminance(hex) > 0.5 ? darkenHex(hex, 100) : hex, []);

    const primary = primaryColor ?? safe(currentTheme.colors.primary);
    const activeGradient = useMemo(
      () => gradient ?? `linear-gradient(135deg, ${safe(currentTheme.colors.primary)}, ${safe(currentTheme.colors.secondary)})`,
      [gradient, currentTheme, safe],
    );

    const hasText = value.trim().length > 0;
    const showStop = loading && !!onStop;
    const canSend = hasText && !loading && !disabled;

    const sm = size === 'sm';
    const maxHeight = sm ? 80 : 120;

    // Auto-Grow bei jeder Wertänderung — auch wenn `value` programmatisch
    // gesetzt wird (Prefill) oder nach dem Senden geleert wird (Höhe-Reset).
    useLayoutEffect(() => {
      const el = innerRef.current;
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, maxHeight) + 'px';
    }, [value, maxHeight]);

    const handleSend = () => { if (canSend) onSend(); };

    return (
      <div
        className={`rounded-2xl border bg-white/[0.04] p-1 transition-all duration-200 ${className}`}
        style={{
          borderColor: focused ? `${primary}66` : 'rgba(255,255,255,0.10)',
          boxShadow: focused
            ? `0 0 0 3px ${primary}1f, 0 10px 24px -8px rgba(0,0,0,0.35)`
            : '0 10px 24px -8px rgba(0,0,0,0.25)',
        }}
      >
        <div className={`flex items-end gap-1.5 rounded-xl bg-white/[0.035] ${sm ? 'p-1' : 'p-1.5'}`}>
          <textarea
            ref={setTextareaRef}
            value={value}
            onChange={e => onChange(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={placeholder}
            disabled={disabled || loading}
            rows={1}
            autoFocus={autoFocus}
            className={`flex-1 resize-none bg-transparent text-white placeholder-gray-600 focus:outline-none disabled:opacity-50 leading-relaxed ${
              sm ? 'px-2 py-1.5 text-xs' : 'px-2.5 py-2 text-sm'
            }`}
            style={{ maxHeight }}
          />
          <button
            type="button"
            onClick={() => (showStop ? onStop!() : handleSend())}
            onMouseDown={e => e.preventDefault()}
            disabled={!showStop && !canSend}
            title={showStop ? stopTitle : sendTitle}
            aria-label={showStop ? (stopTitle ?? 'Stop') : (sendTitle ?? 'Senden')}
            className={`flex items-center justify-center rounded-xl flex-shrink-0 transition-all duration-200 active:scale-95 ${
              sm ? 'w-8 h-8' : 'w-9 h-9'
            } ${
              showStop
                ? 'bg-red-500/20 border border-red-500/40 text-red-300 hover:bg-red-500/30'
                : canSend
                ? 'text-white hover:scale-105'
                : 'bg-white/[0.06] text-gray-500 cursor-not-allowed'
            }`}
            style={
              !showStop && canSend
                ? { background: activeGradient, boxShadow: `0 8px 20px ${primary}30, inset 0 1px 0 rgba(255,255,255,.18)` }
                : undefined
            }
          >
            {showStop
              ? <Square className={sm ? 'w-3 h-3' : 'w-3.5 h-3.5'} fill="currentColor" />
              : loading
              ? <Loader2 className={`animate-spin ${sm ? 'w-3.5 h-3.5' : 'w-4 h-4'}`} />
              : <Send className={sm ? 'w-3.5 h-3.5' : 'w-4 h-4'} strokeWidth={2.25} />}
          </button>
        </div>
      </div>
    );
  },
);

export default GradientChatInput;
