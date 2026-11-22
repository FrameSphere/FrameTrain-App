// Videoabspieler der Werkbank.
//
// AVI, MKV und MPEG-4 Part 2 spielt keine Webview ab: statt des Videos stand
// dort ein durchgestrichenes Symbol, labeln liess es sich nicht. Meldet das
// Element einen Fehler, kommt eine Wiedergabekopie aus dem Backend — trainiert
// und exportiert wird weiter mit dem Original.

import { useEffect, useRef, useState, type VideoHTMLAttributes } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import type { StudioSample } from './studioTypes';

type Props = Pick<VideoHTMLAttributes<HTMLVideoElement>, 'onLoadedMetadata' | 'onTimeUpdate'> & {
  projectId: string;
  sample: StudioSample;
  src: string;
  videoRef: (el: HTMLVideoElement | null) => void;
};

/// Kopien dieser Sitzung je Mediendatei: alle Abschnitte eines Videos teilen
/// sich eine, und ein laufender Auftrag wird nicht zweimal gestartet.
const kopien = new Map<string, Promise<string>>();

function kopieFuer(projectId: string, sample: StudioSample): Promise<string> {
  const key = `${projectId}/${sample.media}`;
  const da = kopien.get(key);
  if (da) return da;
  const neu = invoke<string>('studio_playable_video', { projectId, sampleId: sample.id });
  kopien.set(key, neu);
  neu.catch(() => kopien.delete(key));
  return neu;
}

export default function StudioVideo({ projectId, sample, src, videoRef, onLoadedMetadata, onTimeUpdate }: Props) {
  const { t } = useLanguage();
  const [kopie, setKopie] = useState<string | null>(null);
  const [laeuft, setLaeuft] = useState(false);
  const [fehler, setFehler] = useState<string | null>(null);
  const lebt = useRef(true);
  useEffect(() => { lebt.current = true; return () => { lebt.current = false; }; }, []);

  const ausweichen = async () => {
    // Auch die Kopie spielt nicht: dann hilft kein zweiter Versuch.
    if (kopie) { setFehler(t('studio.media.videoUnplayable')); return; }
    if (laeuft) return;
    setLaeuft(true);
    try {
      const pfad = await kopieFuer(projectId, sample);
      if (lebt.current) setKopie(pfad);
    } catch (err: unknown) {
      if (lebt.current) setFehler(String(err));
    } finally {
      if (lebt.current) setLaeuft(false);
    }
  };

  return (
    <div className="relative">
      <video ref={videoRef} controls={!laeuft && !fehler} src={kopie ? convertFileSrc(kopie) : src}
        className="h-[420px] w-full rounded-lg bg-black object-contain"
        onError={() => void ausweichen()}
        onLoadedMetadata={onLoadedMetadata} onTimeUpdate={onTimeUpdate} />
      {(laeuft || fehler) && (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-black/80 p-6 text-center">
          {fehler ? (
            <p className="text-amber-200/90 text-xs inline-flex items-start gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" /> {fehler}
            </p>
          ) : (
            <p className="text-gray-300 text-sm inline-flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> {t('studio.media.videoPreparing')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
