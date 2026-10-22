import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { installGlobalErrorReporting } from './utils/errorReport';
import { AppErrorBoundary } from './components/AppErrorBoundary';
import { getCurrentWindow } from '@tauri-apps/api/window';
import QuickChatApp from './components/hosting/QuickChatApp';
import { ThemeProvider } from './contexts/ThemeContext';
import { LanguageProvider } from './contexts/LanguageContext';

// Dasselbe Bundle laeuft im Hauptfenster und im Schnell-Chat-Fenster des
// Hostings; das Fenster-Label entscheidet, welche Oberflaeche erscheint.
function isQuickChatWindow(): boolean {
  try { return getCurrentWindow().label === 'quickchat'; } catch { return false; }
}

const IS_MAC = /Mac/i.test(navigator.platform || navigator.userAgent);

// Globales Auto-Error-Reporting an den Manager (speist die Auto-Fix-Pipeline).
installGlobalErrorReporting();

// Einzeilige Felder sind hier Modellnamen, Repo-IDs, Pfade und Suchen — die
// macOS-Autokorrektur machte aus "distilbert" beim Wegklicken "distillery".
// Mehrzeilige Felder (Chats, Notizen) behalten Rechtschreibhilfe.
document.addEventListener('focusin', e => {
  const el = e.target;
  if (!(el instanceof HTMLInputElement) || el.dataset.ftTyping === 'natural') return;
  el.setAttribute('autocorrect', 'off');
  el.setAttribute('autocapitalize', 'off');
  el.spellcheck = false;
}, true);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      {isQuickChatWindow() ? (
        <ThemeProvider>
          <LanguageProvider>
            <QuickChatApp />
          </LanguageProvider>
        </ThemeProvider>
      ) : (
        <>
          {/* macOS: Overlay-Titelleiste (tauri.conf.json). Der obere Rand bleibt
              auf jeder Seite ziehbar; die Ampel liegt links darueber. */}
          {IS_MAC && <div data-tauri-drag-region className="fixed top-0 inset-x-0 h-7 z-[45]" aria-hidden="true" />}
          <App />
        </>
      )}
    </AppErrorBoundary>
  </React.StrictMode>,
);
