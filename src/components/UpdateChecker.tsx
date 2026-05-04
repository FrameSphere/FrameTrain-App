import { useEffect, useState } from 'react';
import { getVersion } from '@tauri-apps/api/app';
import { open as openUrl } from '@tauri-apps/plugin-shell';

interface UpdateInfo {
  latestVersion: string;
  currentVersion: string;
}

export function UpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [showDialog, setShowDialog] = useState(false);

  useEffect(() => {
    checkForUpdates();
  }, []);

  async function checkForUpdates() {
    try {
      const currentVersion = await getVersion();
      console.log('[UpdateChecker] Current version:', currentVersion);

      let latestVersion: string = '';

      // Methode 1: Versuche GitHub API für Releases zu nutzen
      try {
        console.log('[UpdateChecker] Trying GitHub API...');
        const response = await fetch(
          'https://api.github.com/repos/FrameSphere/FrameTrain-App/releases/latest',
          {
            headers: { 'Accept': 'application/json' },
            cache: 'no-store'
          }
        );

        if (response.ok) {
          const data = await response.json();
          latestVersion = (data.tag_name as string)?.replace(/^v/, '') ?? '';
          console.log('[UpdateChecker] Latest version from GitHub API:', latestVersion);
        }
      } catch (err) {
        console.warn('[UpdateChecker] GitHub API fehlgeschlagen, versuche alternatives Verfahren:', err);
      }

      // Methode 2: Fallback zu latest.json
      if (!latestVersion) {
        try {
          console.log('[UpdateChecker] Trying latest.json...');
          const response = await fetch(
            'https://github.com/FrameSphere/FrameTrain-App/releases/latest/download/latest.json',
            {
              headers: { 'Accept': 'application/json' },
              cache: 'no-store'
            }
          );

          if (response.ok) {
            const data = await response.json();
            latestVersion = (data.version as string)?.replace(/^v/, '') ?? '';
            console.log('[UpdateChecker] Latest version from latest.json:', latestVersion);
          }
        } catch (err) {
          console.warn('[UpdateChecker] latest.json nicht erreichbar:', err);
        }
      }

      if (!latestVersion) {
        console.warn('[UpdateChecker] Keine Versionsinformation verfügbar');
        return;
      }

      if (compareVersions(latestVersion, currentVersion) > 0) {
        console.log('[UpdateChecker] Update verfügbar:', latestVersion, '>', currentVersion);
        setUpdateInfo({ latestVersion, currentVersion });
        setShowDialog(true);
      } else {
        console.log('[UpdateChecker] App ist aktuell:', currentVersion);
      }
    } catch (error) {
      console.error('[UpdateChecker] Fehler beim Update-Check:', error);
    }
  }

  function compareVersions(v1: string, v2: string): number {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);
    for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
      const p1 = parts1[i] || 0;
      const p2 = parts2[i] || 0;
      if (p1 > p2) return 1;
      if (p1 < p2) return -1;
    }
    return 0;
  }

  function openDashboard() {
    openUrl('https://frametrain.vercel.app/dashboard').catch((err) => {
      console.error('Failed to open URL:', err);
      // Fallback: Versuche window.open als Backup
      window.open('https://frametrain.vercel.app/dashboard', '_blank');
    });
  }

  if (!showDialog || !updateInfo) return null;

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-[9999]"
      style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}
    >
      <div className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Warnstreifen */}
        <div className="h-1 bg-gradient-to-r from-orange-500 to-red-500" />

        <div className="p-6">
          {/* Header */}
          <div className="flex items-start gap-4 mb-5">
            <div className="w-10 h-10 rounded-full bg-orange-500/20 border border-orange-500/40 flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
            </div>
            <div>
              <h2 className="text-white font-semibold text-lg leading-tight">
                Neues Update verfügbar 🚀
              </h2>
              <p className="text-gray-400 text-sm mt-1">
                Version <span className="font-mono text-orange-400">v{updateInfo.latestVersion}</span> ist verfügbar
              </p>
            </div>
          </div>

          {/* Versions-Info */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4 mb-4 flex justify-between text-sm">
            <div className="text-center">
              <p className="text-gray-500 text-xs mb-1">Installiert</p>
              <p className="font-mono text-gray-300">v{updateInfo.currentVersion}</p>
            </div>
            <div className="flex items-center text-gray-600">→</div>
            <div className="text-center">
              <p className="text-gray-500 text-xs mb-1">Neu</p>
              <p className="font-mono text-green-400 font-semibold">v{updateInfo.latestVersion}</p>
            </div>
          </div>

          {/* Nachricht */}
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-5">
            <p className="text-red-300 text-sm leading-relaxed">
              ⚠️ <strong>Bitte lade dir die neue Version herunter</strong> und deinstalliere
              umgehend die alte Version. Diese könnte Sicherheitslücken aufweisen und wird
              nicht mehr unterstützt.
            </p>
          </div>

          {/* Update-Anleitung */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 mb-5">
            <p className="text-blue-300 font-semibold text-sm mb-3">📋 Update-Anleitung:</p>
            <ol className="text-gray-300 text-xs space-y-1.5 ml-4 list-decimal">
              <li>Klick auf <strong>"Zum Dashboard"</strong> um die neue Version herunterzuladen</li>
              <li>Deinstalliere die alte FrameTrain App komplett:
                <div className="mt-1 p-2 bg-black/30 rounded text-gray-400 font-mono text-[10px]">
                  Gehe zu <strong>Applications</strong> → FrameTrain → <strong>Move to Trash</strong>
                </div>
              </li>
              <li>Installiere die neue Version:
                <div className="mt-1 p-2 bg-black/30 rounded text-gray-400 font-mono text-[10px]">
                  Die heruntergeladene <strong>.dmg</strong> öffnen und FrameTrain in <strong>Applications</strong> ziehen
                </div>
              </li>
              <li>Starte die neue FrameTrain App</li>
            </ol>
          </div>

          {/* Download Link */}
          <div className="bg-white/5 rounded-xl p-4 mb-5 text-center">
            <p className="text-gray-400 text-xs mb-2">Download verfügbar unter:</p>
            <p
              className="text-blue-400 underline cursor-pointer hover:text-blue-300 transition-colors text-sm font-semibold"
              onClick={openDashboard}
            >
              frametrain.vercel.app/dashboard
            </p>
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              onClick={openDashboard}
              className="flex-1 py-2.5 px-4 bg-blue-600 hover:bg-blue-500 rounded-xl text-white text-sm font-semibold transition-all"
            >
              Zum Dashboard
            </button>
            <button
              onClick={() => setShowDialog(false)}
              className="flex-1 py-2.5 px-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-gray-300 text-sm font-medium transition-all"
            >
              Schließen
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
