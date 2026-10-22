import { Check } from 'lucide-react';
import { useTheme, themeVars, type ThemeId, type Theme } from '../../contexts/ThemeContext';
import { useLanguage } from '../../contexts/LanguageContext';

/**
 * Theme-Auswahl: jede Karte zeigt ein Mini-Fenster in den Farben des Themes
 * (Wand, Glas-Seitenleiste, Flaeche, Akzent-Knopf, laufende Kurve in Orange).
 * Die Vorschau setzt ihre eigenen Tokens ueber .ft-scope, deshalb stimmt sie
 * auch, wenn gerade ein anderes Theme aktiv ist.
 */
export default function ThemePicker({ onSelect, columns = 3 }: {
  onSelect?: (id: ThemeId) => void;
  columns?: 2 | 3 | 4;
}) {
  const { currentTheme, setTheme, themes } = useTheme();
  const { t } = useLanguage();
  const list = Object.values(themes);
  const night = list.filter(th => th.base === 'night');
  const day = list.filter(th => th.base === 'day');

  const pick = (id: ThemeId) => {
    if (onSelect) onSelect(id);
    else setTheme(id);
  };

  const grid = columns === 4 ? 'grid-cols-2 lg:grid-cols-4' : columns === 2 ? 'grid-cols-2' : 'grid-cols-2 lg:grid-cols-3';

  const section = (label: string, items: Theme[]) => (
    <div>
      <div className="ft-label mb-2.5">{label}</div>
      <div className={`grid ${grid} gap-3`}>
        {items.map(theme => {
          const active = currentTheme.id === theme.id;
          return (
            <button
              key={theme.id}
              onClick={() => pick(theme.id)}
              aria-pressed={active}
              className={`ft-theme-card text-left rounded-[14px] p-2 transition-colors ${active ? 'is-active' : ''}`}
            >
              <ThemePreview theme={theme} />
              <div className="flex items-start justify-between gap-2 px-1 pt-2.5 pb-0.5">
                <div className="min-w-0">
                  <div className="text-[13.5px] font-semibold text-white truncate">{theme.name}</div>
                  <div className="text-[12px] text-gray-500 truncate">{t(`themes.${theme.id}`, theme.description)}</div>
                </div>
                {active && (
                  <span className="w-5 h-5 rounded-full flex items-center justify-center flex-none mt-0.5" style={{ background: 'var(--ft-accent)', color: 'var(--ft-on-accent)' }}>
                    <Check className="w-3 h-3" strokeWidth={3} />
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="space-y-5">
      {section(t('settings.appearance.groupNight'), night)}
      {section(t('settings.appearance.groupDay'), day)}
    </div>
  );
}

function ThemePreview({ theme }: { theme: Theme }) {
  return (
    <div
      className={`ft-scope ${theme.base === 'day' ? 'ft-day' : 'ft-night'} rounded-[10px] p-1.5 flex gap-1.5 h-[96px]`}
      style={{ ...themeVars(theme), background: 'var(--ft-wall)' } as React.CSSProperties}
      aria-hidden="true"
    >
      <div className="w-[30%] rounded-[7px] p-1.5 flex flex-col gap-1" style={{ background: 'var(--ft-glass)', border: '1px solid var(--ft-glass-edge)' }}>
        <span className="h-1.5 w-3/4 rounded-full" style={{ background: 'var(--ft-ink)', opacity: 0.55 }} />
        <span className="h-2.5 rounded-[3px] mt-1" style={{ background: 'var(--ft-sel)' }} />
        <span className="h-1 w-2/3 rounded-full" style={{ background: 'var(--ft-muted)', opacity: 0.6 }} />
        <span className="h-1 w-1/2 rounded-full" style={{ background: 'var(--ft-muted)', opacity: 0.6 }} />
      </div>
      <div className="flex-1 rounded-[7px] p-2 flex flex-col gap-1.5 min-w-0" style={{ background: 'var(--ft-panel)', border: '1px solid var(--ft-line)' }}>
        <div className="flex items-center justify-between">
          <span className="h-1.5 w-1/2 rounded-full" style={{ background: 'var(--ft-ink)', opacity: 0.8 }} />
          <span className="h-3 w-9 rounded-full" style={{ background: 'var(--ft-accent)' }} />
        </div>
        <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="w-full flex-1">
          <polyline points="0,6 15,14 30,19 50,23 70,25 100,26" fill="none" stroke="var(--ft-out)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <polyline points="0,10 25,18 50,22 75,24 100,25" fill="none" stroke="var(--ft-accent-text)" strokeWidth="1.2" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
}
