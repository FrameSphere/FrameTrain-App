// AppContextMenu — app-weites Rechtsklick-Menü.
//
// Struktur (nur nicht-leere Abschnitte werden gezeigt):
//   1. Auswahl-Aktionen   — nur wenn Text markiert ist (Kopieren, AI Coach)
//   2. Seiten-Aktionen    — von der aktuell gemounteten Seite (Registry)
//   3. Globale Aktionen   — AI Coach öffnen
//   4. Navigation         — "Gehe zu …" (Untermenü)
//
// In Eingabefeldern (input/textarea/contenteditable) bleibt das native
// macOS-Menü aktiv — inkl. Apples Rechtschreib-Vorschlägen, die keine
// Web-API nachbauen kann.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Copy, Check, Sparkles, MessageSquarePlus, Compass, ChevronRight,
  Layers, Upload, Play, BarChart3, FlaskConical, Microscope, Network, GitBranch,
  type LucideIcon,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { openAICoach } from '../../ai/aiCoachEvents';
import { navigateTo, type AppView } from '../../ui/navigationEvents';
import { collectContextMenuActions, type ContextMenuAction } from '../../ui/contextMenuRegistry';

interface MenuState {
  x: number;
  y: number;
  selection: string;
  pageActions: ContextMenuAction[];
}

interface Row {
  id: string;
  label: string;
  icon?: LucideIcon;
  onSelect: () => void;
  disabled?: boolean;
  danger?: boolean;
  accent?: boolean;
  submenu?: Row[];
}

interface Section {
  rows: Row[];
  heading?: string;
}

const MENU_WIDTH = 248;

const NAV_ITEMS: { id: AppView; icon: LucideIcon; key: string }[] = [
  { id: 'models',     icon: Layers,       key: 'sidebar.nav.models' },
  { id: 'dataset',    icon: Upload,       key: 'sidebar.nav.datasets' },
  { id: 'training',   icon: Play,         key: 'sidebar.nav.training' },
  { id: 'analysis',   icon: BarChart3,    key: 'sidebar.nav.analysis' },
  { id: 'tests',      icon: FlaskConical, key: 'sidebar.nav.tests' },
  { id: 'laboratory', icon: Microscope,   key: 'sidebar.nav.laboratory' },
  { id: 'synapse',    icon: Network,      key: 'sidebar.nav.synapse' },
  { id: 'versions',   icon: GitBranch,    key: 'sidebar.nav.versions' },
];

export default function AppContextMenu() {
  const { t, language } = useLanguage();
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [copied, setCopied] = useState(false);
  const [openSub, setOpenSub] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const close = useCallback(() => { setMenu(null); setCopied(false); setOpenSub(null); }, []);

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
      const pageActions = collectContextMenuActions();
      setCopied(false);
      setOpenSub(null);
      setMenu({ x: e.clientX, y: e.clientY, selection, pageActions });
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

  useEffect(() => {
    if (!menu) return;
    const onMouseDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) close();
    };
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, [menu, close]);

  const handleCopy = useCallback(async (text: string) => {
    try { await navigator.clipboard.writeText(text); }
    catch { try { document.execCommand('copy'); } catch { /* ignore */ } }
    setCopied(true);
    setTimeout(close, 450);
  }, [close]);

  // ── Menüzeilen in Sektionen aufbauen ──────────────────────────────────────
  const sections = useMemo<Section[]>(() => {
    if (!menu) return [];
    const en = (language ?? '').toLowerCase().startsWith('en');
    const result: Section[] = [];

    // 1. Auswahl
    if (menu.selection) {
      const sel = menu.selection;
      result.push({ rows: [
        {
          id: 'copy',
          label: copied ? t('contextMenu.copied') : t('contextMenu.copy'),
          icon: copied ? Check : Copy,
          onSelect: () => handleCopy(sel),
        },
        {
          id: 'ask-coach-sel',
          label: t('contextMenu.askCoach'),
          icon: Sparkles,
          accent: true,
          onSelect: () => {
            openAICoach({
              prefill: en
                ? `Regarding this from FrameTrain:\n\n"${sel.slice(0, 1500)}"\n\n`
                : `Zu diesem Inhalt aus FrameTrain:\n\n"${sel.slice(0, 1500)}"\n\n`,
            });
            close();
          },
        },
      ] });
    }

    // 2. Seiten-Aktionen (mit Gruppen-Überschrift der aktuellen Seite)
    if (menu.pageActions.length > 0) {
      result.push({
        heading: menu.pageActions[0].group,
        rows: menu.pageActions.map((a) => ({
          id: a.id,
          label: a.label,
          icon: a.icon,
          disabled: a.disabled,
          onSelect: () => { if (!a.disabled) { a.onSelect(); close(); } },
        })),
      });
    }

    // 3. Globale Aktion + Navigation
    result.push({ rows: [
      {
        id: 'coach-open',
        label: t('contextMenu.openCoach'),
        icon: MessageSquarePlus,
        onSelect: () => { openAICoach({ newChat: true }); close(); },
      },
      {
        id: 'nav',
        label: t('contextMenu.goto'),
        icon: Compass,
        onSelect: () => {},
        submenu: NAV_ITEMS.map((n) => ({
          id: `nav-${n.id}`,
          label: t(n.key),
          icon: n.icon,
          onSelect: () => { navigateTo(n.id); close(); },
        })),
      },
    ] });

    return result;
  }, [menu, copied, language, t, handleCopy, close]);

  if (!menu) return null;

  // Menühöhe grob schätzen für Viewport-Klemmung
  const rowCount = sections.reduce((n, s) => n + s.rows.length + (s.heading ? 1 : 0), 0) + sections.length;
  const estH = rowCount * 32 + 12;
  const x = Math.max(8, Math.min(menu.x, window.innerWidth - MENU_WIDTH - 8));
  const y = Math.max(8, Math.min(menu.y, window.innerHeight - estH - 8));
  const subLeft = x + MENU_WIDTH + 4 > window.innerWidth - 8;

  return (
    <div
      ref={menuRef}
      className="fixed z-[10000] py-1.5 rounded-xl border border-white/10 bg-slate-900/95 backdrop-blur-md shadow-2xl select-none"
      style={{ left: x, top: y, width: MENU_WIDTH, animation: 'ft-ctx-in 0.1s ease-out' }}
      onContextMenu={(e) => e.preventDefault()}
    >
      <style>{`@keyframes ft-ctx-in { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: scale(1); } }`}</style>
      {sections.map((section, si) => (
        <div key={si}>
          {si > 0 && <div className="my-1 border-t border-white/10" />}
          {section.heading && (
            <div className="px-3 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wide text-gray-500">
              {section.heading}
            </div>
          )}
          {section.rows.map((row) => (
            <div
              key={row.id}
              className="relative"
              onMouseEnter={() => setOpenSub(row.submenu ? row.id : null)}
            >
              <button
                onClick={() => { if (!row.submenu) row.onSelect(); }}
                disabled={row.disabled}
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-left text-sm transition-colors ${
                  row.disabled
                    ? 'text-gray-600 cursor-default'
                    : row.accent
                      ? 'text-violet-300 hover:bg-violet-500/15'
                      : row.danger
                        ? 'text-red-300 hover:bg-red-500/15'
                        : 'text-gray-200 hover:bg-white/10'
                }`}
              >
                {row.icon && <row.icon className="w-3.5 h-3.5 flex-shrink-0" />}
                <span className="flex-1 truncate">{row.label}</span>
                {row.submenu && <ChevronRight className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />}
              </button>

              {row.submenu && openSub === row.id && (
                <div
                  className="absolute top-0 py-1.5 rounded-xl border border-white/10 bg-slate-900/95 backdrop-blur-md shadow-2xl"
                  style={{ width: MENU_WIDTH - 30, [subLeft ? 'right' : 'left']: MENU_WIDTH - 2 } as React.CSSProperties}
                >
                  {row.submenu.map((sub) => (
                    <button
                      key={sub.id}
                      onClick={sub.onSelect}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-left text-sm text-gray-200 hover:bg-white/10 transition-colors"
                    >
                      {sub.icon && <sub.icon className="w-3.5 h-3.5 flex-shrink-0 text-gray-400" />}
                      <span className="flex-1 truncate">{sub.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
