import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * FrameTrain Theme System
 * 
 * 16 handverlesene Farbthemes für die Desktop-App:
 * 
 * ORIGINAL COLLECTION:
 * - Purple Passion: Klassisches lila-pink Design (Standard)
 * - Ocean Breeze: Beruhigendes Blau-Türkis
 * - Forest Green: Natürliches Grün
 * - Sunset Orange: Warmes Orange-Rot
 * - Midnight Blue: Tiefes Dunkelblau
 * - Rose Gold: Elegantes Rosa-Gold
 * 
 * DARK & MINIMAL:
 * - Dark Minimal: Minimalistisches schwarzes Design
 * - Monochrome: Klassisches Schwarz-Weiß
 * 
 * NEON COLLECTION:
 * - Cyber Yellow: Schwarz mit neon-gelbem Akzent
 * - Neon Pink: Elektrisierendes Neon-Pink
 * - Toxic Green: Giftiges Neon-Grün
 * 
 * INTENSE COLLECTION:
 * - Blood Red: Intensives dunkles Rot
 * - Golden Hour: Warmes Gold und Bernstein
 * - Fire & Ice: Orange-Rot trifft Eisblau
 * 
 * PREMIUM COLLECTION:
 * - Arctic White: Eisige weiß-blaue Töne
 * - Lavender Dream: Sanftes Lavendel und Violett
 */

export type ThemeId = 'purple-passion' | 'ocean-breeze' | 'forest-green' | 'sunset-orange' | 'midnight-blue' | 'rose-gold' | 'dark-minimal' | 'cyber-yellow' | 'arctic-white' | 'neon-pink' | 'blood-red' | 'toxic-green' | 'golden-hour' | 'monochrome' | 'lavender-dream' | 'fire-ice' | 'light-gray' | 'pure-white';

export interface Theme {
  id: ThemeId;
  name: string;
  description: string;
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

export const themes: Record<ThemeId, Theme> = {
  'purple-passion': {
    id: 'purple-passion',
    name: 'Purple Passion',
    description: 'Klassisches lila-pink Design',
    colors: {
      primary: '#9333ea',
      secondary: '#ec4899',
      accent: '#a855f7',
      background: 'from-slate-900 via-purple-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-purple-600 to-pink-600',
    },
  },
  'ocean-breeze': {
    id: 'ocean-breeze',
    name: 'Ocean Breeze',
    description: 'Beruhigendes Blau-Türkis Theme',
    colors: {
      primary: '#0891b2',
      secondary: '#06b6d4',
      accent: '#22d3ee',
      background: 'from-slate-900 via-cyan-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-cyan-600 to-blue-600',
    },
  },
  'forest-green': {
    id: 'forest-green',
    name: 'Forest Green',
    description: 'Natürliches Grün Theme',
    colors: {
      primary: '#059669',
      secondary: '#10b981',
      accent: '#34d399',
      background: 'from-slate-900 via-emerald-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-emerald-600 to-green-600',
    },
  },
  'sunset-orange': {
    id: 'sunset-orange',
    name: 'Sunset Orange',
    description: 'Warmes Orange-Rot Theme',
    colors: {
      primary: '#ea580c',
      secondary: '#f59e0b',
      accent: '#fb923c',
      background: 'from-slate-900 via-orange-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-orange-600 to-red-600',
    },
  },
  'midnight-blue': {
    id: 'midnight-blue',
    name: 'Midnight Blue',
    description: 'Tiefes dunkelblaues Theme',
    colors: {
      primary: '#1e40af',
      secondary: '#3b82f6',
      accent: '#60a5fa',
      background: 'from-slate-900 via-blue-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-blue-600 to-indigo-600',
    },
  },
  'rose-gold': {
    id: 'rose-gold',
    name: 'Rose Gold',
    description: 'Elegantes Rosa-Gold Theme',
    colors: {
      primary: '#be123c',
      secondary: '#e11d48',
      accent: '#fb7185',
      background: 'from-slate-900 via-rose-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#9ca3af',
      border: 'border-white/10',
      gradient: 'from-rose-600 to-pink-600',
    },
  },
  'dark-minimal': {
    id: 'dark-minimal',
    name: 'Dark Minimal',
    description: 'Minimalistisches schwarzes Design',
    colors: {
      primary: '#18181b',
      secondary: '#27272a',
      accent: '#52525b',
      background: 'from-black via-zinc-900 to-black',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#71717a',
      border: 'border-white/5',
      gradient: 'from-zinc-700 to-zinc-800',
    },
  },
  'cyber-yellow': {
    id: 'cyber-yellow',
    name: 'Cyber Yellow',
    description: 'Schwarz mit neon-gelbem Akzent',
    colors: {
      primary: '#eab308',
      secondary: '#facc15',
      accent: '#fde047',
      background: 'from-black via-yellow-900/20 to-black',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#a1a1aa',
      border: 'border-yellow-500/20',
      gradient: 'from-yellow-500 to-amber-500',
    },
  },
  'arctic-white': {
    id: 'arctic-white',
    name: 'Arctic White',
    description: 'Eisige weiß-blaue Töne',
    colors: {
      primary: '#e0f2fe',
      secondary: '#bae6fd',
      accent: '#7dd3fc',
      background: 'from-slate-800 via-sky-900 to-slate-800',
      surface: 'bg-white/10',
      text: '#ffffff',
      textSecondary: '#cbd5e1',
      border: 'border-white/20',
      gradient: 'from-sky-400 to-cyan-300',
    },
  },
  'neon-pink': {
    id: 'neon-pink',
    name: 'Neon Pink',
    description: 'Elektrisierendes Neon-Pink',
    colors: {
      primary: '#f0abfc',
      secondary: '#e879f9',
      accent: '#d946ef',
      background: 'from-black via-fuchsia-950 to-black',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#d4d4d8',
      border: 'border-fuchsia-500/20',
      gradient: 'from-fuchsia-500 to-pink-500',
    },
  },
  'blood-red': {
    id: 'blood-red',
    name: 'Blood Red',
    description: 'Intensives dunkles Rot',
    colors: {
      primary: '#dc2626',
      secondary: '#ef4444',
      accent: '#f87171',
      background: 'from-black via-red-950 to-black',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#a1a1aa',
      border: 'border-red-500/20',
      gradient: 'from-red-600 to-rose-600',
    },
  },
  'toxic-green': {
    id: 'toxic-green',
    name: 'Toxic Green',
    description: 'Giftiges Neon-Grün',
    colors: {
      primary: '#84cc16',
      secondary: '#a3e635',
      accent: '#bef264',
      background: 'from-black via-lime-950 to-black',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#a1a1aa',
      border: 'border-lime-500/20',
      gradient: 'from-lime-500 to-green-500',
    },
  },
  'golden-hour': {
    id: 'golden-hour',
    name: 'Golden Hour',
    description: 'Warmes Gold und Bernstein',
    colors: {
      primary: '#f59e0b',
      secondary: '#fbbf24',
      accent: '#fcd34d',
      background: 'from-amber-950 via-yellow-900 to-amber-950',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#d4d4d8',
      border: 'border-amber-500/20',
      gradient: 'from-amber-500 to-yellow-500',
    },
  },
  'monochrome': {
    id: 'monochrome',
    name: 'Monochrome',
    description: 'Klassisches Schwarz-Weiß',
    colors: {
      primary: '#ffffff',
      secondary: '#e5e5e5',
      accent: '#a3a3a3',
      background: 'from-neutral-950 via-neutral-900 to-neutral-950',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#a3a3a3',
      border: 'border-white/10',
      gradient: 'from-neutral-600 to-neutral-700',
    },
  },
  'lavender-dream': {
    id: 'lavender-dream',
    name: 'Lavender Dream',
    description: 'Sanftes Lavendel und Violett',
    colors: {
      primary: '#a78bfa',
      secondary: '#c4b5fd',
      accent: '#ddd6fe',
      background: 'from-slate-900 via-violet-900 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#c4b5fd',
      border: 'border-violet-500/20',
      gradient: 'from-violet-500 to-purple-500',
    },
  },
  'fire-ice': {
    id: 'fire-ice',
    name: 'Fire & Ice',
    description: 'Orange-Rot trifft Eisblau',
    colors: {
      primary: '#f97316',
      secondary: '#06b6d4',
      accent: '#38bdf8',
      background: 'from-slate-900 via-orange-950 via-cyan-950 to-slate-900',
      surface: 'bg-white/5',
      text: '#ffffff',
      textSecondary: '#94a3b8',
      border: 'border-white/10',
      gradient: 'from-orange-500 via-red-500 to-cyan-500',
    },
  },
  'light-gray': {
    id: 'light-gray',
    name: 'Light Gray',
    description: 'Helles neutrales Grau',
    colors: {
      primary: '#1e293b',
      secondary: '#475569',
      accent: '#0f172a',
      background: 'from-gray-50 via-slate-100 to-gray-50',
      surface: 'bg-white',
      text: '#0f172a',
      textSecondary: '#334155',
      border: 'border-slate-300',
      gradient: 'from-slate-700 to-gray-800',
    },
  },
  'pure-white': {
    id: 'pure-white',
    name: 'Pure White',
    description: 'Klares helles Weiß Theme',
    colors: {
      primary: '#1e40af',
      secondary: '#2563eb',
      accent: '#1e3a8a',
      background: 'from-white via-slate-50 to-white',
      surface: 'bg-white',
      text: '#0f172a',
      textSecondary: '#334155',
      border: 'border-gray-300',
      gradient: 'from-blue-700 to-indigo-700',
    },
  },
};

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
