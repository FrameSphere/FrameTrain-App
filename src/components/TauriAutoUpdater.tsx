import { useEffect } from 'react';
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { ask } from '@tauri-apps/plugin-dialog';
import { platform } from '@tauri-apps/plugin-os';

/**
 * Echter In-App-Updater fuer Windows/Linux (signierte Updates via
 * tauri-plugin-updater: Download + Installation + Neustart).
 *
 * Auf macOS bewusst INAKTIV: Ohne Apple-Notarisierung wuerde die per
 * Auto-Update getauschte .app beim Neustart von Gatekeeper blockiert. Dort
 * bleibt der manuelle UpdateChecker-Hinweis (frame-train.com/dashboard)
 * zustaendig. Sobald ein Apple-Developer-Account vorliegt, kann macOS hier
 * ergaenzt werden.
 *
 * Rendert nichts; laeuft einmalig beim Mount. Fehler duerfen die App nie
 * blockieren und werden nur geloggt.
 */
export function TauriAutoUpdater() {
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const os = platform(); // 'windows' | 'linux' | 'macos' | ...
        if (os !== 'windows' && os !== 'linux') return;

        const update = await check();
        if (cancelled || !update) return;

        console.log('[TauriAutoUpdater] Update verfuegbar:', update.version, 'installiert:', update.currentVersion);

        const proceed = await ask(
          `Version ${update.version} ist verfuegbar (installiert: ${update.currentVersion}).\n\n` +
            'Jetzt herunterladen und installieren? Die App startet danach neu.',
          {
            title: 'Update verfuegbar',
            kind: 'info',
            okLabel: 'Aktualisieren',
            cancelLabel: 'Spaeter',
          }
        );
        if (cancelled || !proceed) return;

        await update.downloadAndInstall();
        await relaunch();
      } catch (err) {
        console.error('[TauriAutoUpdater]', err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
