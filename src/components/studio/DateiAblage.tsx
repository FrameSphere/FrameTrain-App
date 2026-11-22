// Dateien und Ordner ins Fenster ziehen.
//
// Tauri faengt den Drop des Betriebssystems ab, bevor der Webview ihn sieht:
// HTML5-Drop-Ereignisse kommen nie mit Dateien an. Die Pfade kommen ueber
// onDragDropEvent. Sortiert werden sie im Backend: nur dort ist bekannt, was
// ein Ordner ist.

import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { Upload } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import ModalPortal from '../ui/ModalPortal';
import { useNotification } from '../../contexts/NotificationContext';

export interface Ablage {
  folders: string[];
  images: string[];
  audio: string[];
  videos: string[];
  texts: string[];
  other: number;
}

type Art = 'image' | 'audio' | 'video' | 'text';

const DATEIEN: Record<Art, keyof Omit<Ablage, 'other'>> = {
  image: 'images', audio: 'audio', video: 'videos', text: 'texts',
};

/**
 * Hoert auf Drops, solange `aktiv` ist. `onAblage` bekommt nur, was zur Art des
 * Projekts passt; ist nichts dabei, sagt der Hook das selbst.
 */
export default function DateiAblage({ art, aktiv, auchVideos = false, onAblage }: {
  art: Art;
  aktiv: boolean;
  /** Box-Projekte nehmen auch Videos: daraus werden Einzelbilder. */
  auchVideos?: boolean;
  onAblage: (a: Ablage) => void | Promise<void>;
}) {
  const { t } = useLanguage();
  const { error, warning } = useNotification();
  const [ueber, setUeber] = useState(false);

  const ablegen = async (paths: string[]) => {
    try {
      const a = await invoke<Ablage>('studio_sort_drop', { paths });
      const passend = a.folders.length + a[DATEIEN[art]].length + (auchVideos ? a.videos.length : 0);
      if (passend === 0) {
        warning(t('studio.drop.nothingTitle'), t(`studio.drop.${art}`));
        return;
      }
      await onAblage(a);
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    }
  };
  const jetzt = useRef({ aktiv, ablegen });
  jetzt.current = { aktiv, ablegen };

  useEffect(() => {
    let un: (() => void) | undefined;
    let weg = false;
    try {
      getCurrentWebview().onDragDropEvent(ev => {
        const p = ev.payload;
        if (!jetzt.current.aktiv) { setUeber(false); return; }
        // Nur echte Dateien: gezogener Text oder ein Bild aus der App selbst
        // bringen keine Pfade mit.
        if (p.type === 'enter') setUeber((p.paths?.length ?? 0) > 0);
        else if (p.type === 'leave') setUeber(false);
        else if (p.type === 'drop') {
          setUeber(false);
          if (p.paths?.length) void jetzt.current.ablegen(p.paths);
        }
      }).then(fn => { if (weg) fn(); else un = fn; }).catch(() => { /* dann ohne */ });
    } catch { /* ausserhalb der App (Tests, Vorschau) */ }
    return () => { weg = true; un?.(); };
  }, []);

  if (!ueber || !aktiv) return null;
  return (
    <ModalPortal><div data-testid="datei-ablage"
      className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center bg-black/50 p-8">
      <div className="rounded-2xl border-2 border-dashed border-white/40 bg-[#101218] px-10 py-8 text-center">
        <Upload className="w-8 h-8 text-gray-300 mx-auto mb-3" />
        <p className="text-white text-sm font-medium">{t('studio.drop.title')}</p>
        <p className="text-gray-400 text-xs mt-1">{t(`studio.drop.${art}`)}</p>
      </div>
    </div></ModalPortal>
  );
}
