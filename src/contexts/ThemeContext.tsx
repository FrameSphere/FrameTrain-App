import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * FrameTrain Theme System (Neuron)
 *
 * Ein Theme ist eine Akzentfarbe auf einer Basis: Night (dunkel) oder Day
 * (hell). Flaechen, Linien, Text und das Orange fuer laufende Ergebnisse
 * kommen aus src/styles/tokens.css und sind in jedem Theme gleich, nur der
 * Akzent wechselt. So bleibt jedes Theme lesbar und ruhig.
 *
 * Die IDs sind dieselben wie vor dem Redesign, gespeicherte Einstellungen
 * gelten weiter. Neues Theme = ein Eintrag in ACCENTS.
 */

export type ThemeId = 'purple-passion' | 'ocean-breeze' | 'forest-green' | 'sunset-orange' | 'midnight-blue' | 'rose-gold' | 'dark-minimal' | 'cyber-yellow' | 'arctic-white' | 'neon-pink' | 'blood-red' | 'toxic-green' | 'golden-hour' | 'monochrome' | 'lavender-dream' | 'fire-ice' | 'light-gray' | 'pure-white';

export type ThemeBase = 'night' | 'day';

export interface ThemeAccent {
  /** Flaeche von Primaerknoepfen und aktivem Zustand */
  fill: string;
  hover: string;
  /** Akzent als Textfarbe (Links, aktive Icons), auf der Basis lesbar */
  text: string;
  /** Schrift auf der Akzentflaeche */
  on: string;
}

export interface Theme {
  id: ThemeId;
  name: string;
  description: string;
  base: ThemeBase;
  accent: ThemeAccent;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    gradient: string;
  };
}

interface ThemeSpec {
  name: string;
  description: string;
  base: ThemeBase;
  accent: ThemeAccent;
}

const WHITE = '#ffffff';
const INK = '#111114';

const ACCENTS: Record<ThemeId, ThemeSpec> = {
  'purple-passion': { name: 'Indigo',     description: 'FrameTrain-Standard, Indigo auf Night', base: 'night', accent: { fill: '#4f53f0', hover: '#5d61ff', text: '#a3a6ff', on: WHITE } },
  'ocean-breeze':   { name: 'Ocean',      description: 'Ruhiges Blau',                          base: 'night', accent: { fill: '#1f6fd1', hover: '#2f80e6', text: '#8fb8ff', on: WHITE } },
  'forest-green':   { name: 'Forest',     description: 'Tiefes Gruen',                          base: 'night', accent: { fill: '#1a7f5e', hover: '#1f9470', text: '#6fd3a8', on: WHITE } },
  'sunset-orange':  { name: 'Clay',       description: 'Warmer Ton, gedaempft',                 base: 'night', accent: { fill: '#b4532a', hover: '#c45f33', text: '#f2a27c', on: WHITE } },
  'midnight-blue':  { name: 'Midnight',   description: 'Dunkles Blau',                          base: 'night', accent: { fill: '#3451b2', hover: '#4060c8', text: '#9db0ff', on: WHITE } },
  'rose-gold':      { name: 'Rose',       description: 'Gedaempftes Rosa',                      base: 'night', accent: { fill: '#c2336e', hover: '#d23f7c', text: '#ff8fb8', on: WHITE } },
  'dark-minimal':   { name: 'Graphite',   description: 'Neutral, ohne Farbe',                   base: 'night', accent: { fill: '#55555f', hover: '#63636e', text: '#d4d4da', on: WHITE } },
  'cyber-yellow':   { name: 'Citrine',    description: 'Gelber Akzent',                         base: 'night', accent: { fill: '#e2b21c', hover: '#efc130', text: '#f2cd5c', on: INK } },
  'arctic-white':   { name: 'Arctic',     description: 'Kuehles Hellblau',                      base: 'night', accent: { fill: '#7cc4ec', hover: '#93d0f2', text: '#a9dcf7', on: INK } },
  'neon-pink':      { name: 'Fuchsia',    description: 'Kraeftiges Pink',                       base: 'night', accent: { fill: '#b5389f', hover: '#c847b0', text: '#f09ae0', on: WHITE } },
  'blood-red':      { name: 'Crimson',    description: 'Dunkles Rot',                           base: 'night', accent: { fill: '#b3262e', hover: '#c7333b', text: '#ff8a8f', on: WHITE } },
  'toxic-green':    { name: 'Lime',       description: 'Frisches Gelbgruen',                    base: 'night', accent: { fill: '#8cc22a', hover: '#9bd23a', text: '#b8e27a', on: INK } },
  'golden-hour':    { name: 'Amber',      description: 'Warmes Gold',                           base: 'night', accent: { fill: '#d4951a', hover: '#e3a42a', text: '#f5c867', on: INK } },
  'monochrome':     { name: 'Mono',       description: 'Schwarz und Weiss',                     base: 'night', accent: { fill: '#ececf0', hover: '#ffffff', text: '#ececf0', on: INK } },
  'lavender-dream': { name: 'Lavender',   description: 'Weiches Violett',                       base: 'night', accent: { fill: '#6c5ce7', hover: '#7a6bf0', text: '#c4b8ff', on: WHITE } },
  'fire-ice':       { name: 'Ice',        description: 'Eisblau, Orange fuer Laufendes',         base: 'night', accent: { fill: '#0b7ea0', hover: '#0e8fb3', text: '#7fd6ef', on: WHITE } },
  'light-gray':     { name: 'Day Graphite', description: 'Hell, neutraler Akzent',              base: 'day',   accent: { fill: '#2b2b33', hover: '#3a3a44', text: '#2b2b33', on: WHITE } },
  'pure-white':     { name: 'Day',        description: 'Hell, Indigo-Akzent',                   base: 'day',   accent: { fill: '#3b3ed6', hover: '#2f33c0', text: '#3b3ed6', on: WHITE } },
};

/** Vollfarbe ueber die Verlaufs-Klassen, die die Komponenten schon nutzen. */
const ACCENT_GRADIENT = 'from-[var(--ft-accent)] to-[var(--ft-accent)]';
const WALL_GRADIENT = 'from-[var(--ft-wall)] to-[var(--ft-wall)]';

function buildTheme(id: ThemeId, spec: ThemeSpec): Theme {
  const day = spec.base === 'day';
  return {
    id,
    name: spec.name,
    description: spec.description,
    base: spec.base,
    accent: spec.accent,
    colors: {
      primary: spec.accent.fill,
      secondary: spec.accent.hover,
      accent: spec.accent.text,
      background: WALL_GRADIENT,
      surface: 'bg-white/5',
      text: day ? INK : '#f4f4f6',
      textSecondary: day ? '#55555f' : '#a3a3ad',
      border: 'border-white/10',
      gradient: ACCENT_GRADIENT,
    },
  };
}

export const themes = Object.fromEntries(
  (Object.keys(ACCENTS) as ThemeId[]).map(id => [id, buildTheme(id, ACCENTS[id])]),
) as Record<ThemeId, Theme>;

export function isLightTheme(id: string): boolean {
  return themes[id as ThemeId]?.base === 'day';
}

/** CSS-Variablen eines Themes, fuer :root oder eine Vorschau mit .ft-scope */
export function themeVars(theme: Theme): Record<string, string> {
  return {
    '--ft-accent': theme.accent.fill,
    '--ft-accent-hover': theme.accent.hover,
    '--ft-accent-text': theme.accent.text,
    '--ft-on-accent': theme.accent.on,
  };
}

interface ThemeContextType {
  currentTheme: Theme;
  setTheme: (themeId: ThemeId) => void;
  themes: Record<ThemeId, Theme>;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [currentThemeId, setCurrentThemeId] = useState<ThemeId>('purple-passion');

  useEffect(() => {
    // Load saved theme
    loadSavedTheme();
  }, []);

  // Helle Themes brauchen eigene Textfarben (index.css, [data-theme="light"]),
  // und das Fenster dasselbe Erscheinungsbild: macOS zeichnet Auswahllisten
  // (<select>) als System-Menue — bei hellem System erschien es sonst hell
  // lila auf der dunklen App.
  useEffect(() => {
    const light = isLightTheme(currentThemeId);
    const root = document.documentElement;
    for (const [k, v] of Object.entries(themeVars(themes[currentThemeId]))) root.style.setProperty(k, v);
    void import('@tauri-apps/api/window')
      .then(({ getCurrentWindow }) => {
        const win = getCurrentWindow();
        // Der Schnell-Chat hat sein eigenes Glas-Design und bleibt, wie er ist.
        if (win.label === 'quickchat') return;
        document.documentElement.dataset.theme = light ? 'light' : 'dark';
        return win.setTheme(light ? 'light' : 'dark');
      })
      .catch(() => { document.documentElement.dataset.theme = light ? 'light' : 'dark'; });
  }, [currentThemeId]);

  const loadSavedTheme = async () => {
    try {
      const config = await invoke<string>('load_config');
      const parsedConfig = JSON.parse(config);
      if (parsedConfig.theme && themes[parsedConfig.theme as ThemeId]) {
        setCurrentThemeId(parsedConfig.theme as ThemeId);
      }
    } catch (error) {
      // Use default theme
      console.log('No saved theme, using default');
    }
  };

  const setTheme = async (themeId: ThemeId) => {
    setCurrentThemeId(themeId);
    
    // Save theme preference
    try {
      const config = await invoke<string>('load_config').catch(() => '{}');
      const parsedConfig = JSON.parse(config);
      parsedConfig.theme = themeId;
      
      await invoke('save_config', {
        apiKey: parsedConfig.api_key || '',
        config: JSON.stringify(parsedConfig),
      });
    } catch (error) {
      console.error('Failed to save theme:', error);
    }
  };

  return (
    <ThemeContext.Provider
      value={{
        currentTheme: themes[currentThemeId],
        setTheme,
        themes,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
