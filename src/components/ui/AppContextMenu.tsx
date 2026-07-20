// AppContextMenu — app-weites Rechtsklick-Menü (Hybrid-Ansatz):
//
// - In Eingabefeldern (input/textarea/contenteditable) bleibt das NATIVE
//   macOS-Menü aktiv — inklusive Apples Rechtschreib-Vorschlägen, die sich
//   nicht nachbauen lassen (keine Web-API dafür).
// - Überall sonst wird das nutzlose WebView-Menü ("Neu laden" …) ersetzt:
//   bei markiertem Text erscheint ein App-gestyltes Menü (Kopieren /
//   Mit AI Coach besprechen), ohne Markierung erscheint gar kein Menü —
//   wie in einer nativen Desktop-App.

import { useCallback, useEffect, useRef, useState } from 'react';
import { Copy, Check, Sparkles } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { openAICoach } from '../../ai/aiCoachEvents';

interface MenuState {
  x: number;
  y: number;
  selection: string;
}

const MENU_WIDTH = 230;
const MENU_HEIGHT = 92;

export default function AppContextMenu() {
  const { t, language } = useLanguage();
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [copied, setCopied] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const close = useCallback(() => { setMenu(null); setCopied(false); }, []);

  useEffect(() => {
    const onContextMenu = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      // Eingabefelder: natives Apple-Menü (mit Vorschlägen) durchlassen
      if (target?.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]')) {
        close();
        return;
      }
      e.preventDefault();
      const selection = window.getSelection()?.toString().trim() ?? '';
      if (!selection) {
        // Ohne Markierung: gar kein Menü — kein Web-Junk ("Neu laden" etc.)
        close();
        return;
      }
      // Position im Viewport halten
      const x = Math.min(e.clientX, window.innerWidth - MENU_WIDTH - 8);
      const y = Math.min(e.clientY, window.innerHeight - MENU_HEIGHT - 8);
      setCopied(false);
      setMenu({ x: Math.max(8, x), y: Math.max(8, y), selection });
    };

    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };

    document.addEventListener('contextmenu', onContextMenu);
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('blur', close);
    window.addEventListener('resize', close);
    document.addEventListener('scroll', close, true);
    return () => {
      document.removeEventListener('contextmenu', onContextMenu);
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('blur', close);
      window.removeEventListener('resize', close);
      document.removeEventListener('scroll', close, true);
    };
  }, [close]);

  // Klick außerhalb schließt (mousedown, damit auch Klicks ohne click-Event greifen)
  useEffect(() => {
    if (!menu) return;
    const onMouseDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) close();
    };
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, [menu, close]);

  if (!menu) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(menu.selection);
    } catch {
      // Fallback für WebViews ohne Clipboard-Permission
      try { document.execCommand('copy'); } catch { /* ignore */ }
    }
    setCopied(true);
    setTimeout(close, 450);
  };

  const handleAskCoach = () => {
    const en = (language ?? '').toLowerCase().startsWith('en');
    openAICoach({
      newChat: false,
      prefill: en
        ? `Regarding this from FrameTrain:\n\n"${menu.selection.slice(0, 1500)}"\n\n`
        : `Zu diesem Inhalt aus FrameTrain:\n\n"${menu.selection.slice(0, 1500)}"\n\n`,
    });
    close();
  };

  return (
    <div
      ref={menuRef}
      className="fixed z-[10000] py-1.5 rounded-xl border border-white/10 bg-slate-900/95 backdrop-blur-md shadow-2xl select-none"
      style={{ left: menu.x, top: menu.y, width: MENU_WIDTH, animation: 'ft-ctx-in 0.1s ease-out' }}
      onContextMenu={(e) => e.preventDefault()}
    >
      <style>{`@keyframes ft-ctx-in { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: scale(1); } }`}</style>
      <button
        onClick={handleCopy}
        className="w-full flex items-center gap-2.5 px-3 py-2 text-left text-sm text-gray-200 hover:bg-white/10 transition-colors"
      >
        {copied
          ? <Check className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
          : <Copy className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />}
        {copied ? t('contextMenu.copied') : t('contextMenu.copy')}
      </button>
      <div className="my-1 border-t border-white/10" />
      <button
        onClick={handleAskCoach}
        className="w-full flex items-center gap-2.5 px-3 py-2 text-left text-sm text-violet-300 hover:bg-violet-500/15 transition-colors"
      >
        <Sparkles className="w-3.5 h-3.5 flex-shrink-0" />
        {t('contextMenu.askCoach')}
      </button>
    </div>
  );
}
