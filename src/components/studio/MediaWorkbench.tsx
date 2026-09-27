// Medien-Werkbank des Dataset Studio: eine Klasse je Sample (oder ein
// Transkript) fuer Audio, Bilder und Videos.
//
// Frueher gab es sie nur fuer Audio. Bild-Klassifikation und Video brauchen
// genau dasselbe — Liste links, Sample in der Mitte, Klassen rechts, 1 bis 9
// weist zu — nur ein anderes Element in der Mitte und andere Wege hinein:
//
//   * Audio: aufnehmen (als 16-kHz-WAV, siehe wavEncode.ts) oder einlesen.
//   * Bild:  Ordner einlesen (Ordnername = Klasse), Cmd+V, aus dem Netz.
//   * Video: Videos einlesen, auf Wunsch in Abschnitte zerlegt. Ein Abschnitt
//            spielt nur seinen Teil; X teilt ihn an der aktuellen Stelle.
//
// Auf macOS braucht das Mikrofon NSMicrophoneUsageDescription in der
// Info.plist, sonst scheitert die Anfrage stumm.

import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import {
  ArrowLeft, FolderOpen, Download, Loader2, Check, SkipForward, Plus, AlertTriangle,
  ChevronDown, Mic, Square, AudioLines, Tag, ListEnd, Trash2, Play, Globe, Wand2,
  ShieldQuestion, CopyCheck, Film, Image as ImageIcon, Scissors,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { classColor } from '../labGroundTruth';
import { useStudioModels, StudioModelSelect } from './studioModels';
import { statsNachAenderung, balanceWarnungen } from './studioStats';
import { nextOpenIndex } from './studioBoxes';
import type {
  StudioProject, StudioSample, SamplePage, ImportReport, StudioStats, SampleStatus,
  SampleFilter, ExportResult,
} from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import RemoveSampleButton from './RemoveSampleButton';
import { useContextMenuActions, type ContextMenuAction } from '../../ui/contextMenuRegistry';
import { useEscape } from './useEscape';
import { toWav16kMono, brauchtWav } from './wavEncode';
import FetchDialog from './FetchDialog';
import ModelRunDialog from './ModelRunDialog';
import NearDupDialog from './NearDupDialog';
import ExportReportView from './ExportReportView';

const PAGE = 200;

/// Welche Endung zu dem gehoert, was der Recorder tatsaechlich aufgenommen hat.
///
/// Nur noch der Rueckfall, wenn das Umwandeln in WAV scheitert: MediaRecorder
/// liefert je nach Webview etwas anderes (WKWebView MP4, Chromium WebM). Wer
/// "webm" fest hineinschreibt, legt auf dem Mac ein MP4 unter falschem Namen
/// ab. Das Backend prueft die Bytes zusaetzlich.
export function audioExtForMime(mime: string): string {
  const basis = (mime || '').split(';')[0].trim().toLowerCase();
  switch (basis) {
    case 'audio/mp4': case 'video/mp4': case 'audio/aac': case 'audio/x-m4a': return 'm4a';
    case 'audio/webm': case 'video/webm':                                      return 'webm';
    case 'audio/ogg': case 'video/ogg': case 'audio/opus':                     return 'ogg';
    case 'audio/wav': case 'audio/wave': case 'audio/x-wav':                   return 'wav';
    case 'audio/mpeg': case 'audio/mp3':                                       return 'mp3';
    case 'audio/flac': case 'audio/x-flac':                                    return 'flac';
    default:                                                                   return 'webm';
  }
}

/** Abschnitt als "2,0–6,0 s". */
export function abschnittText(start?: number | null, end?: number | null): string | null {
  if (start == null && end == null) return null;
  const f = (x: number) => x.toFixed(1).replace('.', ',');
  return `${f(start ?? 0)}–${end != null ? f(end) : '…'} s`;
}

type Medium = 'audio' | 'image' | 'video';

interface Props {
  project: StudioProject;
  onBack: () => void;
  onProjectChanged: (p: StudioProject) => void;
  /** Fuer Tests austauschbar: Aufnahme -> WAV. */
  convertToWav?: (data: ArrayBuffer) => Promise<Uint8Array>;
}

export default function MediaWorkbench({ project, onBack, onProjectChanged, convertToWav = toWav16kMono }: Props) {
  const { t } = useLanguage();
  const { success, error, warning, info } = useNotification();
  const medium: Medium = project.modality === 'image' ? 'image' : project.modality === 'video' ? 'video' : 'audio';

  const [samples, setSamples] = useState<StudioSample[]>([]);
  const [total, setTotal]     = useState(0);
  const [loading, setLoading] = useState(true);
  const [index, setIndex]     = useState(0);
  const [filter, setFilter]   = useState<SampleFilter>('all');
  const [stats, setStats]     = useState<StudioStats | null>(null);
  const [newClass, setNewClass] = useState('');
  const [importing, setImporting] = useState<{ cur: number; total: number } | null>(null);
  const [dialog, setDialog] = useState<null | 'export' | 'fetch' | 'suggest' | 'review' | 'neardup' | 'videos'>(null);
  const [transcript, setTranscript] = useState('');
  const [showKeys, setShowKeys] = useState(false);
  const [recording, setRecording] = useState(false);
  const [wartetAufMikrofon, setWartetAufMikrofon] = useState(false);
  const [seconds, setSeconds] = useState(0);

  const recorder = useRef<MediaRecorder | null>(null);
  const player   = useRef<HTMLAudioElement | HTMLVideoElement | null>(null);
  const chunks   = useRef<Blob[]>([]);
  const ticker   = useRef<number | null>(null);

  const classes  = project.classes;
  const current  = samples[index];
  const transkript = project.task === 'transcript';

  // ── Laden ───────────────────────────────────────────────────────────────
  const loadSamples = useCallback(async (offset: number, replace: boolean) => {
    const page = await invoke<SamplePage>('studio_list_samples', {
      projectId: project.id, status: filter, offset, limit: PAGE,
    });
    setTotal(page.total);
    setSamples(prev => (replace ? page.items : [...prev, ...page.items]));
    return page.items;
  }, [project.id, filter]);

  const loadStats = useCallback(async () => {
    try { setStats(await invoke<StudioStats>('studio_stats', { projectId: project.id })); }
    catch { /* Zahlen sind Beiwerk */ }
  }, [project.id]);

  const [ladeStand, setLadeStand] = useState(0);
  const neuLaden = async () => {
    await loadSamples(0, true);
    setLadeStand(n => n + 1);
    await loadStats();
  };

  // Alte Aufnahmen (M4A, WebM) einmal je Oeffnen nach WAV umwandeln — sonst
  // liest das Training sie unter Windows gar nicht.
  const umgewandelt = useRef(false);
  const alteAufnahmenUmwandeln = useCallback(async (items: StudioSample[]) => {
    if (umgewandelt.current || medium !== 'audio') return;
    umgewandelt.current = true;
    const alt = items.filter(s => s.src.kind === 'record' && brauchtWav(s.media));
    let n = 0;
    for (const s of alt) {
      try {
        const res = await fetch(convertFileSrc(s.abs_path));
        const wav = await convertToWav(await res.arrayBuffer());
        await invoke('studio_replace_audio', { projectId: project.id, sampleId: s.id, bytes: Array.from(wav) });
        n += 1;
      } catch { /* bleibt, wie es ist — abspielen geht weiterhin */ }
    }
    if (n > 0) {
      info(t('studio.media.wavConvertedTitle'), t('studio.media.wavConverted', { count: n }));
      await loadSamples(0, true);
      setLadeStand(x => x + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [medium, project.id]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setIndex(0);
    loadSamples(0, true)
      .then(items => { if (active) void alteAufnahmenUmwandeln(items); })
      .catch(err => { if (active) error(t('studio.notifications.loadError'), String(err)); })
      .finally(() => { if (active) setLoading(false); });
    void loadStats();
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadSamples, loadStats]);

  useEffect(() => {
    if (samples.length >= total) return;
    if (index < samples.length - 20) return;
    void loadSamples(samples.length, false).catch(() => { /* still */ });
  }, [index, samples.length, total, loadSamples]);

  // Was gezeigt wird, haengt an Sample *und* Ladestand. Die ID allein reicht
  // nicht: nach einem Modelllauf kommt dasselbe Sample mit neuem Vorschlag
  // zurueck, und ein Enter haette sonst den alten bestaetigt.
  const shownKey = current ? `${current.id}#${ladeStand}` : null;
  const [shownId, setShownId] = useState<string | null>(null);
  if (current && shownId !== shownKey) {
    setShownId(shownKey);
    setTranscript(current.ann.target ?? '');
  }

  const refreshProject = async () => {
    const list = await invoke<StudioProject[]>('studio_list_projects');
    const fresh = list.find(x => x.id === project.id);
    if (fresh) onProjectChanged(fresh);
  };

  // ── Speichern ───────────────────────────────────────────────────────────
  const persist = useCallback(async (
    status: SampleStatus, sample: StudioSample, label: string | null, ziel: string | null,
  ) => {
    try {
      await invoke('studio_set_annotation', {
        projectId: project.id, sampleId: sample.id, boxes: [], status, label, target: ziel,
      });
      setStats(prev => prev ? statsNachAenderung(prev, project.classes,
        { status: sample.status, boxes: [], label: sample.ann.label },
        { status, boxes: [], label }, false) : prev);
      setSamples(prev => prev.map(s => s.id === sample.id
        ? { ...s, status, ann: { ...s.ann, label: label ?? undefined, target: ziel ?? undefined } } : s));
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, project.classes, t]);

  const [entfernenScharf, setEntfernenScharf] = useState(0);

  const goTo = (i: number) => {
    if (samples.length === 0) return;
    setIndex(Math.min(Math.max(i, 0), samples.length - 1));
  };
  const goToNextOpen = () => {
    const next = nextOpenIndex(samples.map(x => x.status), index);
    if (next >= 0) goTo(next);
  };
  const advance = (statusOfCurrent: SampleStatus) => {
    const statuses = samples.map((s, i) => (i === index ? statusOfCurrent : s.status));
    const next = nextOpenIndex(statuses, index);
    goTo(next >= 0 ? next : index + 1);
  };

  const assign = async (label: string) => {
    if (!current) return;
    await persist('confirmed', current, label, null);
    advance('confirmed');
  };

  const confirmTranscript = async () => {
    if (!current) return;
    if (!transcript.trim()) {
      warning(t('studio.audio.needsTextTitle'), t('studio.audio.needsTextDetail'));
      return;
    }
    await persist('confirmed', current, null, transcript.trim());
    advance('confirmed');
  };

  const skip = async () => {
    if (!current) return;
    await persist('skipped', current, current.ann.label ?? null, current.ann.target ?? null);
    advance('skipped');
  };

  const removeCurrent = async () => {
    if (!current) return;
    const warLetztes = index >= samples.length - 1;
    try {
      await invoke('studio_delete_samples', { projectId: project.id, sampleIds: [current.id] });
      if (warLetztes) setIndex(i => Math.max(0, i - 1));
      await neuLaden();
    } catch (err: unknown) {
      error(t('studio.remove.errorTitle'), String(err));
    }
  };

  // ── Abspielen ───────────────────────────────────────────────────────────
  const playPause = () => {
    const a = player.current;
    if (!a) return;
    if (a.paused) {
      // Ein Abschnitt beginnt an seinem Start, nicht am Anfang des Videos.
      const s = current?.meta.start ?? null;
      const e = current?.meta.end ?? null;
      if (s != null && (a.currentTime < s - 0.05 || (e != null && a.currentTime >= e - 0.05))) a.currentTime = s;
      void a.play();
    } else a.pause();
  };
  const seek = (delta: number) => {
    const a = player.current;
    if (!a) return;
    const lo = current?.meta.start ?? 0;
    const hi = current?.meta.end ?? (Number.isFinite(a.duration) ? a.duration : Infinity);
    a.currentTime = Math.min(Math.max(a.currentTime + delta, lo), hi);
  };

  const splitHere = async () => {
    const v = player.current as HTMLVideoElement | null;
    if (!current || medium !== 'video' || !v) return;
    try {
      await invoke('studio_split_segment', {
        projectId: project.id, sampleId: current.id, at: v.currentTime,
        duration: Number.isFinite(v.duration) ? v.duration : null,
      });
      success(t('studio.media.splitDone'), abschnittText(current.meta.start ?? 0, v.currentTime) ?? '');
      await neuLaden();
    } catch (err: unknown) {
      error(t('studio.media.splitError'), String(err));
    }
  };

  // ── Aufnehmen ───────────────────────────────────────────────────────────
  const stopTicker = () => {
    if (ticker.current) { window.clearInterval(ticker.current); ticker.current = null; }
  };

  const startRecording = async () => {
    if (medium !== 'audio' || recording || wartetAufMikrofon) return;
    // Beim ersten Mal fragt macOS nach dem Mikrofon; bis dahin sah es aus, als
    // haette der Knopf nicht reagiert.
    setWartetAufMikrofon(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setWartetAufMikrofon(false);
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = e => { if (e.data.size > 0) chunks.current.push(e.data); };
      rec.onstop = async () => {
        stream.getTracks().forEach(tr => tr.stop());
        stopTicker();
        setRecording(false);
        setSeconds(0);
        const typ = rec.mimeType || 'audio/mp4';
        const blob = new Blob(chunks.current, { type: typ });
        if (blob.size === 0) {
          warning(t('studio.audio.emptyTitle'), t('studio.audio.emptyDetail'));
          return;
        }
        try {
          const roh = await blob.arrayBuffer();
          // WAV, damit jede Plattform und jede Bibliothek die Aufnahme liest.
          // Scheitert das Dekodieren, bleibt das Original — besser als nichts.
          let bytes: number[];
          let ext: string;
          try {
            bytes = Array.from(await convertToWav(roh));
            ext = 'wav';
          } catch {
            bytes = Array.from(new Uint8Array(roh));
            ext = audioExtForMime(typ);
          }
          const report = await invoke<ImportReport>('studio_add_audio', {
            projectId: project.id, bytes, ext, label: null,
          });
          if (report.duplicates > 0) {
            info(t('studio.audio.recordedTitle'), t('studio.audio.duplicate'));
            return;
          }
          success(t('studio.audio.recordedTitle'),
            t('studio.write.doneDetail', { added: report.added, duplicates: report.duplicates }));
          await neuLaden();
          await refreshProject();
        } catch (err: unknown) {
          error(t('studio.audio.errorTitle'), String(err));
        }
      };
      rec.start();
      recorder.current = rec;
      setRecording(true);
      setSeconds(0);
      ticker.current = window.setInterval(() => setSeconds(s => s + 1), 1000);
    } catch (err: unknown) {
      setWartetAufMikrofon(false);
      error(t('studio.audio.micErrorTitle'), t('studio.audio.micErrorDetail', { detail: String(err) }));
      setRecording(false);
    }
  };

  const stopRecording = () => {
    if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop();
  };

  useEffect(() => () => {
    stopTicker();
    if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop();
  }, []);

  const uhrzeit = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;

  /** Derselbe Knopf oben und im leeren Zustand — beide muessen stoppen koennen. */
  const aufnahmeKnopf = (form: string) => (
    <button onClick={recording ? stopRecording : () => void startRecording()}
      disabled={wartetAufMikrofon}
      className={`${form} border text-sm transition-all inline-flex items-center gap-2 disabled:opacity-70 ${recording
        ? 'bg-red-500/20 hover:bg-red-500/30 border-red-500/40 text-red-200'
        : 'bg-white/5 hover:bg-white/10 border-white/10 text-gray-200'}`}>
      {wartetAufMikrofon ? <Loader2 className="w-4 h-4 animate-spin" />
        : recording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
      {wartetAufMikrofon ? t('studio.audio.waitingForMic')
        : recording ? t('studio.audio.stop', { time: uhrzeit }) : t('studio.audio.record')}
    </button>
  );

  // ── Einlesen ────────────────────────────────────────────────────────────
  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-import-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setImporting(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  const nachImport = async (report: ImportReport) => {
    success(t('studio.audio.doneTitle'), t('studio.audio.doneDetail', {
      added: report.added, duplicates: report.duplicates, labels: report.with_labels,
    }));
    if (report.classes_added.length > 0) await refreshProject();
    await neuLaden();
  };

  const importFolder = async () => {
    try {
      const sel = await open({ directory: true, multiple: false,
        title: t(medium === 'image' ? 'studio.import.dialogTitle' : 'studio.audio.pickFolder') });
      if (!sel || typeof sel !== 'string') return;
      setImporting({ cur: 0, total: 0 });
      const report = medium === 'image'
        ? await invoke<ImportReport>('studio_import_folder', {
          projectId: project.id, sourcePath: sel, labelClasses: null, ignoreLabels: false })
        : await invoke<ImportReport>('studio_import_audio', {
          projectId: project.id, sourcePath: sel, ignoreLabels: false });
      await nachImport(report);
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    } finally {
      setImporting(null);
    }
  };

  const importVideos = async (paths: string[], clipSeconds: number | null) => {
    setDialog(null);
    setImporting({ cur: 0, total: 0 });
    try {
      const report = await invoke<ImportReport>('studio_import_videos', {
        projectId: project.id, paths, clipSeconds, ignoreLabels: false,
      });
      success(t('studio.media.videoImport.doneTitle'), t('studio.media.videoImport.doneDetail', {
        added: report.added, duplicates: report.duplicates, labels: report.with_labels,
      }));
      if (report.classes_added.length > 0) await refreshProject();
      await neuLaden();
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    } finally {
      setImporting(null);
    }
  };

  // Bilder aus der Zwischenablage.
  useEffect(() => {
    if (medium !== 'image') return;
    const onPaste = async (e: ClipboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) return;
      const file = Array.from(e.clipboardData?.files ?? []).find(f => f.type.startsWith('image/'));
      if (!file) return;
      e.preventDefault();
      try {
        const bytes = Array.from(new Uint8Array(await file.arrayBuffer()));
        const report = await invoke<ImportReport>('studio_add_image', { projectId: project.id, bytes, origin: null });
        if (report.duplicates > 0) info(t('studio.media.pastedTitle'), t('studio.audio.duplicate'));
        else success(t('studio.media.pastedTitle'), file.name || '');
        await neuLaden();
      } catch (err: unknown) {
        error(t('studio.import.errorTitle'), String(err));
      }
    };
    const handler = (e: ClipboardEvent) => { void onPaste(e); };
    window.addEventListener('paste', handler);
    return () => window.removeEventListener('paste', handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [medium, project.id]);

  const addClass = async () => {
    const name = newClass.trim();
    if (!name) return;
    try {
      onProjectChanged(await invoke<StudioProject>('studio_update_project', {
        projectId: project.id, classes: [...classes, name],
      }));
      setNewClass('');
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    }
  };

  // ── Rechtsklick ─────────────────────────────────────────────────────────
  useContextMenuActions(() => {
    if (dialog) return [];
    const gSample = t(medium === 'audio' ? 'studio.menu.recording' : medium === 'video' ? 'studio.menu.video' : 'studio.menu.image');
    const gProjekt = t('studio.menu.project');
    const aktionen: ContextMenuAction[] = [];
    if (current) {
      if (medium !== 'image') {
        aktionen.push({ id: 'st-med-play', group: gSample, label: t('studio.menu.playPause'), icon: Play,
          shortcut: t('studio.menu.spaceKey'), onSelect: playPause });
      }
      if (medium === 'video') {
        aktionen.push({ id: 'st-med-split', group: gSample, label: t('studio.menu.splitSegment'), icon: Scissors,
          shortcut: 'X', onSelect: () => { void splitHere(); } });
      }
      if (!transkript && classes.length > 0) {
        aktionen.push({
          id: 'st-med-class', group: gSample, label: t('studio.menu.assignClass'), icon: Tag, onSelect: () => {},
          submenu: classes.map((name, i) => ({
            id: `st-med-class-${i}`, label: name, shortcut: i < 9 ? String(i + 1) : undefined,
            onSelect: () => { void assign(name); },
          })),
        });
      }
      aktionen.push(
        { id: 'st-med-skip', group: gSample, label: t('studio.workbench.skip'), icon: SkipForward, shortcut: 'S',
          onSelect: () => { void skip(); } },
        { id: 'st-med-next', group: gSample, label: t('studio.menu.nextOpen'), icon: ListEnd, shortcut: 'N',
          onSelect: goToNextOpen },
        { id: 'st-med-remove', group: gSample, label: t('studio.menu.removeSample'), icon: Trash2, danger: true,
          onSelect: () => setEntfernenScharf(n => n + 1) },
      );
    }
    if (medium === 'audio') {
      aktionen.push({ id: 'st-med-record', group: gProjekt,
        label: recording ? t('studio.menu.stopRecording') : t('studio.menu.startRecording'),
        icon: recording ? Square : Mic, shortcut: 'R', disabled: wartetAufMikrofon,
        onSelect: () => { if (recording) stopRecording(); else void startRecording(); } });
    }
    aktionen.push(
      medium === 'video'
        ? { id: 'st-med-videos', group: gProjekt, label: t('studio.menu.importVideos'), icon: Film,
          disabled: !!importing, onSelect: () => setDialog('videos') }
        : { id: 'st-med-folder', group: gProjekt, label: t('studio.menu.importFolder'), icon: FolderOpen,
          disabled: !!importing || recording, onSelect: () => { void importFolder(); } },
      { id: 'st-med-web', group: gProjekt, label: t('studio.menu.getFromWeb'), icon: Globe,
        onSelect: () => setDialog('fetch') },
      { id: 'st-med-suggest', group: gProjekt, label: t('studio.menu.suggest'), icon: Wand2,
        disabled: samples.length === 0, onSelect: () => setDialog('suggest') },
    );
    if (!transkript) {
      aktionen.push({ id: 'st-med-review', group: gProjekt, label: t('studio.menu.review'), icon: ShieldQuestion,
        disabled: (stats?.confirmed ?? 0) === 0, onSelect: () => setDialog('review') });
    }
    if (medium === 'image') {
      aktionen.push({ id: 'st-med-neardup', group: gProjekt, label: t('studio.menu.nearDup'), icon: CopyCheck,
        disabled: samples.length < 2, onSelect: () => setDialog('neardup') });
    }
    aktionen.push({ id: 'st-med-export', group: gProjekt, label: t('studio.menu.export'), icon: Download,
      disabled: (stats?.confirmed ?? 0) === 0, onSelect: () => setDialog('export') });
    return aktionen;
  });

  // ── Tastatur ────────────────────────────────────────────────────────────
  // Ein einziger Listener, der immer den Handler des aktuellen Renderdurchgangs
  // ruft — ein per useEffect neu angemeldeter Handler kam erst nach dem Zeichnen an.
  const tastenRef = useRef<(e: KeyboardEvent) => void>(() => {});
  tastenRef.current = (e: KeyboardEvent) => {
    const el = e.target as HTMLElement | null;
    const imFeld = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    if (dialog) return;
    if (imFeld) {
      if (current && transkript && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); void confirmTranscript();
      }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const taste = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    // R geht auch im leeren Projekt — dort faengt man ja mit Aufnehmen an.
    if (taste === 'r' && medium === 'audio') {
      e.preventDefault();
      if (recording) stopRecording(); else void startRecording();
      return;
    }
    if (!current) return;
    if (e.key === ' ' && medium !== 'image') {
      // Sonst loest die Leertaste den zuletzt geklickten Knopf noch einmal aus.
      e.preventDefault();
      playPause();
      return;
    }
    if ((e.key === '[' || e.key === ']') && medium !== 'image') {
      e.preventDefault();
      const schritt = medium === 'video' ? 1 : 5;
      seek(e.key === ']' ? schritt : -schritt);
      return;
    }
    if (taste === 'x' && medium === 'video') { e.preventDefault(); void splitHere(); return; }
    if (taste === 'n') { e.preventDefault(); goToNextOpen(); return; }
    if (e.key >= '1' && e.key <= '9' && !transkript) {
      const i = Number(e.key) - 1;
      if (i >= classes.length) return;
      e.preventDefault();
      void assign(classes[i]);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (transkript) void confirmTranscript();
      else if (current.ann.label) void assign(current.ann.label);
      return;
    }
    if (taste === 's') { e.preventDefault(); void skip(); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); goTo(index + 1); return; }
    if (e.key === 'ArrowLeft' || e.key === 'Backspace') { e.preventDefault(); goTo(index - 1); }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => tastenRef.current(e);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // ── Anzeige ─────────────────────────────────────────────────────────────
  const confirmed = stats?.confirmed ?? 0;
  const statusDot = (s: SampleStatus) =>
    s === 'confirmed' ? 'bg-emerald-400' : s === 'skipped' ? 'bg-gray-500'
      : s === 'suggested' ? 'bg-amber-400' : 'bg-white/20';
  const warnungen = transkript ? [] : balanceWarnungen(stats?.per_class ?? [], classes);
  const EmptyIcon = medium === 'image' ? ImageIcon : medium === 'video' ? Film : AudioLines;

  const listenText = (s: StudioSample) => {
    const teil = abschnittText(s.meta.start, s.meta.end);
    const name = s.ann.label ?? ((s.ann.target ?? '').slice(0, 26) || t('studio.audio.noLabel'));
    return teil ? `${name} · ${teil}` : name;
  };

  const mitte = (s: StudioSample) => {
    const src = convertFileSrc(s.abs_path);
    if (medium === 'image') {
      return <img key={s.id} src={src} alt={s.src.origin ?? s.id}
        className="max-h-[440px] w-auto max-w-full mx-auto rounded-lg object-contain" />;
    }
    if (medium === 'video') {
      return (
        <video key={s.id} ref={el => { player.current = el; }} controls src={src}
          className="max-h-[420px] w-full rounded-lg bg-black"
          onLoadedMetadata={e => { if (s.meta.start != null) e.currentTarget.currentTime = s.meta.start; }}
          onTimeUpdate={e => {
            // Ein Abschnitt endet an seinem Ende — sonst hoert man das naechste mit.
            const v = e.currentTarget;
            if (s.meta.end != null && v.currentTime >= s.meta.end && !v.paused) {
              v.pause();
              v.currentTime = s.meta.end;
            }
          }} />
      );
    }
    return <audio key={s.id} ref={el => { player.current = el; }} controls src={src} className="w-full" />;
  };

  const sourceButtons = (gross: boolean) => {
    const form = gross ? 'px-4 py-2.5 rounded-xl' : 'px-3 py-2 rounded-lg';
    const knopf = `${form} bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50 whitespace-nowrap`;
    return (
      <>
        {medium === 'audio' && aufnahmeKnopf(form)}
        {medium === 'video' ? (
          <button onClick={() => setDialog('videos')} disabled={!!importing} className={knopf}>
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Film className="w-4 h-4" />}
            {importing && importing.total > 0
              ? t('studio.import.progress', { current: importing.cur, total: importing.total })
              : t('studio.media.videoImportButton')}
          </button>
        ) : (
          <button onClick={() => void importFolder()} disabled={!!importing || recording} className={knopf}>
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
            {importing && importing.total > 0
              ? t('studio.import.progress', { current: importing.cur, total: importing.total })
              : t('studio.audio.importFolder')}
          </button>
        )}
        <button onClick={() => setDialog('fetch')} className={knopf}>
          <Globe className="w-4 h-4" /> {t('studio.media.getFromWeb')}
        </button>
      </>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={onBack}
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 transition-all"
          aria-label={t('studio.workbench.back')}>
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="min-w-0">
          <h2 className="text-white font-semibold text-lg truncate">{project.name}</h2>
          <p className="text-gray-500 text-xs">
            {t('studio.workbench.subtitle', { confirmed, total: stats?.total ?? total })}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2 flex-wrap justify-end">
          {sourceButtons(false)}
          <button onClick={() => setDialog('suggest')} disabled={samples.length === 0}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-40 whitespace-nowrap">
            <Wand2 className="w-4 h-4" /> {t('studio.workbench.suggestButton')}
          </button>
          <button onClick={() => setDialog('export')} disabled={confirmed === 0}
            className="px-3 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-40 whitespace-nowrap">
            <Download className="w-4 h-4" /> {t('studio.workbench.exportButton')}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20 text-gray-500 gap-2">
          <Loader2 className="w-5 h-5 animate-spin" /> {t('common.loading', 'Lädt…')}
        </div>
      ) : samples.length === 0 && filter === 'all' ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-12 text-center">
          <EmptyIcon className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{t(`studio.media.empty.${medium}Title`)}</p>
          <p className="text-gray-500 text-sm mt-1 mb-5 max-w-md mx-auto">{t(`studio.media.empty.${medium}Detail`)}</p>
          <div className="flex items-center justify-center gap-2 flex-wrap">{sourceButtons(true)}</div>
        </div>
      ) : (
        <div className="grid grid-cols-[200px_minmax(0,1fr)_220px] gap-4">
          <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden flex flex-col">
            <div className="grid grid-cols-3 text-[11px] border-b border-white/10">
              {([['all', t('studio.filter.all')], ['open', t('studio.filter.open')],
                 ['confirmed', t('studio.filter.confirmed')], ['uncertain', t('studio.filter.uncertain')],
                 ...(transkript ? [] : [['doubt', stats?.doubts ? `${t('studio.filter.doubt')} ${stats.doubts}` : t('studio.filter.doubt')]])] as [SampleFilter, string][])
                .map(([val, label]) => (
                  <button key={val} onClick={() => setFilter(val)} title={val === 'uncertain' ? t('studio.filter.uncertainHint') : undefined}
                    className={`py-2 px-0.5 whitespace-nowrap transition-all ${filter === val ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                    {label}
                  </button>
                ))}
            </div>
            <div className="overflow-y-auto max-h-[520px]">
              {samples.length === 0 && (
                <p className="text-gray-600 text-[11px] text-center py-4 px-3">
                  {t(filter === 'uncertain' ? 'studio.filter.uncertainEmpty' : 'studio.filter.empty')}
                </p>
              )}
              {samples.map((s, i) => (
                <button key={s.id} onClick={() => goTo(i)}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-all ${i === index ? 'bg-white/10' : 'hover:bg-white/[0.04]'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${statusDot(s.status)}`} />
                  <span className="text-gray-300 text-xs truncate flex-1">{listenText(s)}</span>
                  {s.status === 'suggested' && s.ann.confidence != null && (
                    <span className="text-[10px] text-amber-300/80 tabular-nums">{Math.round(s.ann.confidence * 100)} %</span>
                  )}
                </button>
              ))}
              {samples.length < total && (
                <p className="text-gray-600 text-[11px] text-center py-2">
                  {t('studio.workbench.moreLoading', { count: total - samples.length })}
                </p>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5 flex flex-col gap-4 min-w-0">
            {current && (
              <>
                <div className="rounded-lg bg-black/30 border border-white/10 p-4">{mitte(current)}</div>

                {current.status === 'suggested' && (
                  <p className="text-amber-300/80 text-xs">
                    {t('studio.media.suggestedHint', {
                      conf: current.ann.confidence != null ? `${Math.round(current.ann.confidence * 100)} %` : '—',
                    })}
                  </p>
                )}
                {current.doubt && (
                  <p className="text-orange-300/90 text-xs flex items-start gap-1.5">
                    <ShieldQuestion className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    {t('studio.media.doubt', { model: current.doubt.missing.join(', '), label: current.doubt.extra.join(', ') })}
                  </p>
                )}

                {transkript ? (
                  <label className="block">
                    <span className="text-gray-400 text-xs">{t('studio.audio.transcriptLabel')}</span>
                    <textarea value={transcript} onChange={e => setTranscript(e.target.value)} rows={4}
                      placeholder={t('studio.audio.transcriptPlaceholder')}
                      className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none" />
                    <span className="text-gray-600 text-[11px]">{t('studio.text.targetHint')}</span>
                  </label>
                ) : classes.length === 0 ? (
                  <p className="text-amber-300/80 text-xs flex items-start gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    {t('studio.text.noClasses')}
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {classes.map((name, i) => (
                      <button key={i} onClick={() => void assign(name)}
                        className={`px-3 py-2 rounded-lg border text-sm inline-flex items-center gap-2 transition-all ${current.ann.label === name ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-300 hover:bg-white/10'}`}>
                        <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                          style={{ background: classColor(name, classes) }} />
                        {name}
                        {i < 9 && (
                          <span className="text-[10px] font-mono text-gray-500 border border-white/10 rounded px-1">{i + 1}</span>
                        )}
                      </button>
                    ))}
                  </div>
                )}

                <div className="flex items-center gap-3 text-[11px] text-gray-500">
                  <span className="tabular-nums">{index + 1} / {total}</span>
                  {abschnittText(current.meta.start, current.meta.end) && (
                    <span className="tabular-nums">{abschnittText(current.meta.start, current.meta.end)}</span>
                  )}
                  <span className="truncate max-w-xs">{current.src.origin?.split(/[\\/]/).pop()}</span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {(transkript || (current.status === 'suggested' && current.ann.label)) && (
                    <button onClick={() => void (transkript ? confirmTranscript() : assign(current.ann.label as string))}
                      className="px-4 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm whitespace-nowrap inline-flex items-center gap-2">
                      <Check className="w-4 h-4" /> {t('studio.workbench.confirm')}
                    </button>
                  )}
                  <button onClick={() => void skip()}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm whitespace-nowrap inline-flex items-center gap-2">
                    <SkipForward className="w-4 h-4" /> {t('studio.workbench.skip')}
                  </button>
                  {medium === 'video' && (
                    <button onClick={() => void splitHere()} title={t('studio.media.splitHint')}
                      className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm whitespace-nowrap inline-flex items-center gap-2">
                      <Scissors className="w-4 h-4" /> {t('studio.media.splitHere')}
                    </button>
                  )}
                  <span className="ml-auto" />
                  <RemoveSampleButton key={current.id} onRemove={() => void removeCurrent()} armSignal={entfernenScharf} />
                </div>
              </>
            )}
          </div>

          <div className="space-y-3 pb-20">
            {!transkript && (
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                <p className="text-gray-500 text-xs mb-2">{t('studio.workbench.classesTitle')}</p>
                <div className="space-y-1 max-h-56 overflow-y-auto">
                  {classes.map((name, i) => (
                    <div key={i} className="flex items-center gap-2 px-2 py-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                        style={{ background: classColor(name, classes) }} />
                      {i < 9 && (
                        <span className="text-[10px] font-mono text-gray-500 border border-white/10 rounded px-1">{i + 1}</span>
                      )}
                      <span className="text-gray-200 text-xs truncate">{name}</span>
                      <span className="ml-auto text-gray-500 text-xs tabular-nums"
                        title={t('studio.workbench.classCountHint')}>
                        {stats?.per_class?.[i] ?? 0}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="flex gap-1.5 mt-2">
                  <input value={newClass} onChange={e => setNewClass(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') void addClass(); }}
                    placeholder={t('studio.workbench.newClassPlaceholder')}
                    className="flex-1 min-w-0 px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25" />
                  <button onClick={() => void addClass()}
                    className="px-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300"
                    aria-label={t('studio.workbench.addClass')}>
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}

            <BalanceHinweise warnungen={warnungen} />

            {stats && (
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 space-y-1.5">
                <p className="text-gray-500 text-xs mb-1">{t('studio.stats.title')}</p>
                {([['confirmed', stats.confirmed], ['open', stats.new + stats.suggested],
                   ['skipped', stats.skipped]] as const).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">{t(`studio.stats.${key}`)}</span>
                    <span className="text-gray-200 tabular-nums">{val}</span>
                  </div>
                ))}
                <div className="flex gap-1.5 pt-1.5">
                  {!transkript && (
                    <button onClick={() => setDialog('review')} disabled={confirmed === 0}
                      className="flex-1 px-2 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[11px] inline-flex items-center justify-center gap-1 disabled:opacity-40">
                      <ShieldQuestion className="w-3.5 h-3.5" /> {t('studio.workbench.reviewButton')}
                    </button>
                  )}
                  {medium === 'image' && (
                    <button onClick={() => setDialog('neardup')} disabled={samples.length < 2}
                      className="flex-1 px-2 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-[11px] inline-flex items-center justify-center gap-1 disabled:opacity-40">
                      <CopyCheck className="w-3.5 h-3.5" /> {t('studio.media.nearDupButton')}
                    </button>
                  )}
                </div>
              </div>
            )}

            <div className="rounded-xl border border-white/10 bg-white/[0.03]">
              <button onClick={() => setShowKeys(v => !v)}
                className="w-full flex items-center gap-2 p-3 text-left">
                <span className="text-gray-500 text-xs flex-1">{t('studio.shortcuts.title')}</span>
                <ChevronDown className={`w-3.5 h-3.5 text-gray-500 transition-transform ${showKeys ? 'rotate-180' : ''}`} />
              </button>
              {showKeys && (
                <div className="space-y-1 text-[11px] text-gray-400 px-3 pb-3">
                  <p>{t(transkript ? 'studio.text.shortcutPair' : 'studio.shortcuts.classes')}</p>
                  <p>{t('studio.shortcuts.skip')}</p>
                  <p>{t('studio.shortcuts.nextOpen')}</p>
                  <p>{t('studio.shortcuts.back')}</p>
                  {medium !== 'image' && <p>{t('studio.shortcuts.play')}</p>}
                  {medium === 'audio' && <p>{t('studio.shortcuts.seek')}</p>}
                  {medium === 'audio' && <p>{t('studio.shortcuts.record')}</p>}
                  {medium === 'video' && <p>{t('studio.shortcuts.seekVideo')}</p>}
                  {medium === 'video' && <p>{t('studio.shortcuts.split')}</p>}
                  {medium === 'image' && <p>{t('studio.shortcuts.paste')}</p>}
                  <p>{t('studio.shortcuts.menu')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {dialog === 'export' && (
        <MediaExportDialog project={project} medium={medium} confirmed={confirmed}
          suggested={stats?.suggested ?? 0}
          onClose={() => setDialog(null)} onDone={() => void loadStats()} />
      )}
      {dialog === 'fetch' && (
        <FetchDialog project={project} onClose={() => setDialog(null)}
          onDone={() => { setDialog(null); void neuLaden(); void refreshProject(); }} />
      )}
      {(dialog === 'suggest' || dialog === 'review') && (
        <ModelRunDialog mode={dialog} project={project} onClose={() => setDialog(null)}
          onDone={() => { setDialog(null); void neuLaden(); void refreshProject(); }} />
      )}
      {dialog === 'neardup' && (
        <NearDupDialog project={project} onClose={() => setDialog(null)}
          onDone={() => { setDialog(null); void neuLaden(); void refreshProject(); }} />
      )}
      {dialog === 'videos' && (
        <VideoImportDialog onClose={() => setDialog(null)} onRun={(p, s) => void importVideos(p, s)} />
      )}
    </div>
  );
}

// ── Balance ───────────────────────────────────────────────────────────────

export function BalanceHinweise({ warnungen }: { warnungen: ReturnType<typeof balanceWarnungen> }) {
  const { t } = useLanguage();
  if (warnungen.length === 0) return null;
  return (
    <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 space-y-1" data-testid="balance-warnings">
      <p className="text-amber-200 text-xs font-medium">{t('studio.balance.title')}</p>
      {warnungen.slice(0, 4).map((w, i) => (
        <p key={i} className="text-amber-200/80 text-[11px]">
          {w.art === 'leer' ? t('studio.balance.leer', { klasse: w.klasse })
            : w.art === 'wenig' ? t('studio.balance.wenig', { klasse: w.klasse, anzahl: w.anzahl })
              : t('studio.balance.schief', { gross: w.gross, grossN: w.grossN, klein: w.klein, kleinN: w.kleinN })}
        </p>
      ))}
    </div>
  );
}

// ── Videos einlesen ───────────────────────────────────────────────────────

function VideoImportDialog({ onClose, onRun }: {
  onClose: () => void; onRun: (paths: string[], clipSeconds: number | null) => void;
}) {
  const { t } = useLanguage();
  const [clip, setClip] = useState<number | null>(null);
  useEscape(onClose, true);

  const waehlen = async (ordner: boolean) => {
    const sel = await open(ordner
      ? { directory: true, multiple: false, title: t('studio.media.videoImport.folder') }
      : { directory: false, multiple: true, title: t('studio.media.videoImport.files'),
        filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'm4v', 'webm', 'mkv', 'avi'] }] });
    if (!sel) return;
    onRun(Array.isArray(sel) ? sel : [sel], clip);
  };

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4" onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.media.videoImport.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.media.videoImport.subtitle')}</p>
        </div>
        <div>
          <span className="text-gray-400 text-xs">{t('studio.media.videoImport.clipLabel')}</span>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {([null, 2, 4, 8, 16] as const).map(n => (
              <button key={String(n)} onClick={() => setClip(n)}
                className={`px-2.5 py-1.5 rounded-lg border text-xs transition-all ${clip === n ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}>
                {n === null ? t('studio.media.videoImport.clipNone') : t('studio.media.videoImport.clipSeconds', { s: n })}
              </button>
            ))}
          </div>
          <p className="text-gray-600 text-[11px] mt-1.5">{t('studio.media.videoImport.clipHint')}</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => void waehlen(true)}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2">
            <FolderOpen className="w-4 h-4" /> {t('studio.media.videoImport.folderButton')}
          </button>
          <button onClick={() => void waehlen(false)}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm inline-flex items-center justify-center gap-2">
            <Film className="w-4 h-4" /> {t('studio.media.videoImport.filesButton')}
          </button>
        </div>
      </div>
    </div></ModalPortal>
  );
}

// ── Export ────────────────────────────────────────────────────────────────

/** Voreinstellungen fuer die Aufteilung: [train, val] — der Rest ist test. */
export const SPLITS: { key: string; train: number; val: number }[] = [
  { key: 'none', train: 0, val: 0 },
  { key: '8020', train: 0.8, val: 0.2 },
  { key: '701515', train: 0.7, val: 0.15 },
];

function MediaExportDialog({ project, medium, confirmed, suggested, onClose, onDone }: {
  project: StudioProject; medium: Medium; confirmed: number; suggested: number;
  onClose: () => void; onDone: () => void;
}) {
  const { t } = useLanguage();
  const { success, error } = useNotification();
  const { models, modelId, setModelId } = useStudioModels(project);
  const [name, setName] = useState(project.name);
  const [busy, setBusy] = useState(false);
  const transkript = project.task === 'transcript';
  const [split, setSplit] = useState(transkript ? 'none' : '8020');
  const [mitVorschlaegen, setMitVorschlaegen] = useState(false);
  const [ergebnis, setErgebnis] = useState<ExportResult | null>(null);

  const run = async () => {
    if (!modelId) return;
    setBusy(true);
    const s = SPLITS.find(x => x.key === split) ?? SPLITS[0];
    try {
      const r = await invoke<ExportResult>('studio_export', {
        projectId: project.id, modelId, datasetName: name,
        includeSuggested: mitVorschlaegen, trainRatio: s.train, valRatio: s.val,
      });
      success(t('studio.export.doneTitle'),
        t(s.train > 0 ? 'studio.export.doneDetailSplit' : 'studio.export.doneDetail', { name }));
      setErgebnis(r);
      onDone();
    } catch (err: unknown) {
      error(t('studio.export.errorTitle'), String(err));
    } finally {
      setBusy(false);
    }
  };

  useEscape(onClose, !busy);
  const hinweis = transkript ? 'studio.audio.exportTranscript'
    : medium === 'video' ? 'studio.media.exportVideo'
      : medium === 'image' ? 'studio.media.exportImageClasses' : 'studio.audio.exportClasses';
  const summe = medium === 'video' ? 'studio.media.summaryVideo'
    : medium === 'image' ? 'studio.media.summaryImage' : 'studio.export.summaryAudio';

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t(ergebnis ? 'studio.report.title' : 'studio.export.title')}</h3>
          {!ergebnis && <p className="text-gray-500 text-xs mt-1">{t(hinweis)}</p>}
        </div>

        {ergebnis ? (
          <>
            <ExportReportView report={ergebnis.report} />
            <button onClick={onClose}
              className="w-full py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm">
              {t('studio.suggest.report.close')}
            </button>
          </>
        ) : (
          <>
            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.export.nameLabel')}</span>
              <input value={name} onChange={e => setName(e.target.value)}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
            </label>

            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.export.modelLabel')}</span>
              <StudioModelSelect project={project} models={models} value={modelId} onChange={setModelId} />
            </label>

            {!transkript && (
              <div>
                <span className="text-gray-400 text-xs">{t('studio.export.splitLabel')}</span>
                <div className="mt-1.5 flex gap-1.5">
                  {SPLITS.map(s => (
                    <button key={s.key} onClick={() => setSplit(s.key)}
                      className={`px-2.5 py-1.5 rounded-lg border text-xs transition-all ${split === s.key ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}>
                      {t(`studio.media.split.${s.key}`)}
                    </button>
                  ))}
                </div>
                <p className="text-gray-600 text-[11px] mt-1.5">{t('studio.media.splitGroupNote')}</p>
              </div>
            )}

            {suggested > 0 && (
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={mitVorschlaegen} onChange={e => setMitVorschlaegen(e.target.checked)} className="mt-0.5" />
                <span className="text-gray-300 text-xs">{t('studio.export.includeSuggested', { count: suggested })}</span>
              </label>
            )}

            <p className="text-gray-500 text-xs">{t(summe, { confirmed })}</p>

            <div className="flex gap-2">
              <button onClick={onClose}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void run()} disabled={busy || !modelId}
                className="flex-1 py-2.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-200 text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                {t('studio.export.confirmButton')}
              </button>
            </div>
          </>
        )}
      </div>
    </div></ModalPortal>
  );
}
