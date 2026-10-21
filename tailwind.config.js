/** @type {import('tailwindcss').Config} */

// Farben laufen ueber die Tokens aus src/styles/tokens.css. So stellen Theme
// (Akzent) und Basis (Night/Day) die ganze Oberflaeche um, ohne dass die
// Komponenten ihre Klassen aendern muessen:
// - white = Textfarbe der Basis (in Day dunkel), white/5 also immer "leichte Flaeche"
// - gray/slate/zinc = neutrale Stufen der Basis
// - purple/violet/fuchsia/indigo/pink = Akzent des Themes
const tok = (name) => `color-mix(in srgb, var(${name}) calc(<alpha-value> * 100%), transparent)`;

const ramp = (prefix) => ({
  50: tok(`${prefix}-50`),
  100: tok(`${prefix}-100`),
  200: tok(`${prefix}-200`),
  300: tok(`${prefix}-300`),
  400: tok(`${prefix}-400`),
  500: tok(`${prefix}-500`),
  600: tok(`${prefix}-600`),
  700: tok(`${prefix}-700`),
  800: tok(`${prefix}-800`),
  900: tok(`${prefix}-900`),
  950: tok(`${prefix}-950`),
});

const neutral = ramp('--ft-n');
const accent = ramp('--ft-a');

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        white: tok('--ft-ink'),
        gray: neutral,
        slate: neutral,
        zinc: neutral,
        neutral,
        purple: accent,
        violet: accent,
        fuchsia: accent,
        indigo: accent,
        pink: accent,
        // Ergebnis-Farbe aus dem Logo: nur fuer Laufendes und Ergebnisse
        output: tok('--ft-out'),
        paper: '#ffffff',
      },
      fontFamily: {
        sans: ['"Hanken Grotesk Variable"', '"Hanken Grotesk"', 'system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
