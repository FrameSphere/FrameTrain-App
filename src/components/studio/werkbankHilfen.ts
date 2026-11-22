// Was die drei Werkbaenke gemeinsam haben, ohne gleich auszusehen.

import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import type { SampleFilter, StudioSample } from './studioTypes';

/** Die Filter ueber der Liste; "doubt" nur, wo ein Modell pruefen kann. */
export function filterListe(mitZweifel: boolean): SampleFilter[] {
  return ['all', 'open', 'confirmed', 'skipped', 'uncertain', ...(mitZweifel ? ['doubt' as const] : [])];
}

/** Haelt das ausgewaehlte Sample in der Liste sichtbar — beim Durchgehen mit
 *  der Tastatur lief die Auswahl sonst unten aus dem Bild. */
export function useSichtbareAuswahl(index: number, anzahl: number) {
  const liste = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = liste.current?.querySelector<HTMLElement>(`[data-index="${index}"]`);
    el?.scrollIntoView?.({ block: 'nearest' });
  }, [index, anzahl]);
  return liste;
}

/**
 * Ein KI-Client (MCP) hat dieses Projekt geaendert: neu laden. Mehrere Aufrufe
 * kurz hintereinander werden zusammengefasst; waehrend `beschaeftigt` (eine Box
 * wird gezogen, ein Text getippt) wird gewartet statt weggenommen.
 */
export function useMcpAbgleich(projectId: string, neuLaden: () => void | Promise<void>, beschaeftigt: () => boolean) {
  const laden = useRef(neuLaden);
  laden.current = neuLaden;
  const besetzt = useRef(beschaeftigt);
  besetzt.current = beschaeftigt;
  useEffect(() => {
    let timer: number | undefined;
    const versuchen = () => {
      if (besetzt.current()) { timer = window.setTimeout(versuchen, 1500); return; }
      timer = undefined;
      void laden.current();
    };
    const onChange = (e: Event) => {
      const d = (e as CustomEvent<{ domain?: string; projectId?: string }>).detail;
      if (d?.domain !== 'studio' || d.projectId !== projectId) return;
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(versuchen, 400);
    };
    window.addEventListener('ft-mcp-changed', onChange);
    return () => { window.removeEventListener('ft-mcp-changed', onChange); if (timer) window.clearTimeout(timer); };
  }, [projectId]);
}

/** Mehrfachauswahl in der Liste: ⌘-Klick schaltet um, ⇧-Klick waehlt bis hierher. */
export function useMehrfachauswahl(samples: StudioSample[]) {
  const [ids, setIds] = useState<Set<string>>(new Set());
  const anker = useRef<number | null>(null);

  // Was nicht mehr in der Liste steht (entfernt, anderer Filter), faellt heraus.
  useEffect(() => {
    setIds(prev => {
      if (prev.size === 0) return prev;
      const da = new Set(samples.map(s => s.id));
      const n = new Set([...prev].filter(id => da.has(id)));
      return n.size === prev.size ? prev : n;
    });
  }, [samples]);

  /** true: der Klick galt der Auswahl, nicht dem Oeffnen. */
  const klick = useCallback((e: MouseEvent, i: number, aktuell: number): boolean => {
    if (e.shiftKey) {
      const von = anker.current ?? aktuell;
      const [a, b] = von <= i ? [von, i] : [i, von];
      setIds(prev => new Set([...prev, ...samples.slice(a, b + 1).map(s => s.id)]));
      return true;
    }
    if (e.metaKey || e.ctrlKey) {
      anker.current = i;
      const id = samples[i]?.id;
      if (!id) return true;
      setIds(prev => {
        const n = new Set(prev);
        // Der erste ⌘-Klick nimmt das gerade offene Sample mit — sonst stuende
        // es sichtbar markiert da und gehoerte doch nicht dazu.
        if (n.size === 0 && aktuell !== i && samples[aktuell]) n.add(samples[aktuell].id);
        if (n.has(id)) n.delete(id); else n.add(id);
        return n;
      });
      return true;
    }
    anker.current = i;
    setIds(prev => (prev.size === 0 ? prev : new Set()));
    return false;
  }, [samples]);

  return { ids, klick, leeren: useCallback(() => setIds(new Set()), []), alle: useCallback(() => setIds(new Set(samples.map(s => s.id))), [samples]) };
}

/** Herkunft lesbar und in der Sprache der Oberflaeche. Das Backend schreibt
 *  feste deutsche Vermerke, wo es keine Datei und keine Adresse gibt. */
export function herkunftText(src: StudioSample['src'], t: (k: string) => string): string {
  if (src.kind === 'record') return t('studio.workbench.recording');
  const fest: Record<string, string> = {
    'von Hand angelegt': 'studio.origin.byHand', 'Zwischenablage': 'studio.origin.clipboard',
    'Labor': 'studio.origin.lab', 'Aufnahme': 'studio.workbench.recording', 'mcp': 'studio.origin.assistant',
  };
  const o = src.origin ?? '';
  return fest[o] ? t(fest[o]) : o.split(/[\\/]/).filter(Boolean).pop() ?? '';
}
