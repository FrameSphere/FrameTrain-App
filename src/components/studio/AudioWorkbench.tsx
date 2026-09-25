// Audio-Werkbank des Dataset Studio: Klassifikation und Transkription.
//
// Gleiche Ablage, gleiche Befehle wie Bild und Text — in der Mitte steht ein
// Abspieler. Neu ist der Weg hinein: aufnehmen statt importieren. Das Mikrofon
// laeuft ueber den Webview (MediaRecorder); auf macOS braucht die App dafuer
// NSMicrophoneUsageDescription in der Info.plist, sonst scheitert die Anfrage
// stumm und die Aufnahme bleibt leer.

import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import {
  ArrowLeft, FolderOpen, Download, Loader2, Check, SkipForward,
  Plus, AlertTriangle, ChevronDown, Mic, Square, AudioLines,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { classColor } from '../labGroundTruth';
import { useStudioModels, StudioModelSelect } from './studioModels';
import { statsNachAenderung } from './studioStats';
import { nextOpenIndex } from './studioBoxes';
import type {
  StudioProject, StudioSample, SamplePage, ImportReport, StudioStats, SampleStatus,
} from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import RemoveSampleButton from './RemoveSampleButton';

const PAGE = 200;

/// Welche Endung zu dem gehoert, was der Recorder tatsaechlich aufgenommen hat.
///
/// MediaRecorder liefert je nach Webview etwas anderes: WKWebView (macOS, also
/// auch diese App) nimmt MP4 auf, Chromium WebM. Wer "webm" fest hineinschreibt,
/// legt auf dem Mac ein MP4 unter falschem Namen ab — der Abspieler bleibt dann
/// stumm bei der eigenen Aufnahme. Das Backend prueft die Bytes zusaetzlich.
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

interface Props {
  project: StudioProject;
  onBack: () => void;
  onProjectChanged: (p: StudioProject) => void;
}

export default function AudioWorkbench({ project, onBack, onProjectChanged }: Props) {
  const { t } = useLanguage();
  const { success, error, warning, info } = useNotification();

  const [samples, setSamples] = useState<StudioSample[]>([]);
  const [total, setTotal]     = useState(0);
  const [loading, setLoading] = useState(true);
  const [index, setIndex]     = useState(0);
  const [filter, setFilter]   = useState<'all' | 'open' | 'confirmed'>('all');
  const [stats, setStats]     = useState<StudioStats | null>(null);
  const [newClass, setNewClass] = useState('');
  const [importing, setImporting] = useState<{ cur: number; total: number } | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [showKeys, setShowKeys] = useState(false);
  const [recording, setRecording] = useState(false);
  const [wartetAufMikrofon, setWartetAufMikrofon] = useState(false);
  const [seconds, setSeconds] = useState(0);

  const recorder = useRef<MediaRecorder | null>(null);
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
  }, [project.id, filter]);

  const loadStats = useCallback(async () => {
    try { setStats(await invoke<StudioStats>('studio_stats', { projectId: project.id })); }
    catch { /* Zahlen sind Beiwerk */ }
  }, [project.id]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setIndex(0);
    loadSamples(0, true)
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

  // Beim Wechsel sofort umstellen, nicht im Effekt — sonst landet ein
  // schneller Tastendruck auf dem Transkript der vorigen Aufnahme.
  // Was gezeigt wird, haengt an Sample *und* Ladestand. Die ID allein reicht
  // nicht: nach einem Modelllauf oder Import kommt dasselbe Sample mit neuen
  // Boxen oder Labels zurueck. Wurde vorher nur zurueckgesetzt und dann
  // geladen, rendert React dazwischen mit den alten Daten, merkt sich die ID —
  // und uebernimmt die neuen nie. Ein Enter bestaetigte dann leere Boxen und
  // loeschte den Vorschlag. Der Zaehler steigt erst, wenn die Daten da sind.
  const [ladeStand, setLadeStand] = useState(0);
  const shownKey = current ? `${current.id}#${ladeStand}` : null;
  const [shownId, setShownId] = useState<string | null>(null);
  if (current && shownId !== shownKey) {
    setShownId(shownKey);
    setTranscript(current.ann.target ?? '');
  }

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

  // ── Entfernen ───────────────────────────────────────────────────────────
  const removeCurrent = async () => {
    if (!current) return;
    const warLetztes = index >= samples.length - 1;
    try {
      await invoke('studio_delete_samples', { projectId: project.id, sampleIds: [current.id] });
      // Das naechste Sample rueckt auf denselben Platz; nur am Ende der Liste
      // muss der Zeiger einen Schritt zurueck.
      if (warLetztes) setIndex(i => Math.max(0, i - 1));
      await loadSamples(0, true);
      setLadeStand(n => n + 1);
      await loadStats();
    } catch (err: unknown) {
      error(t('studio.remove.errorTitle'), String(err));
    }
  };

  const goTo = (i: number) => {
    if (samples.length === 0) return;
    setIndex(Math.min(Math.max(i, 0), samples.length - 1));
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

  // ── Aufnehmen ───────────────────────────────────────────────────────────
  const stopTicker = () => {
    if (ticker.current) { window.clearInterval(ticker.current); ticker.current = null; }
  };

  const startRecording = async () => {
    // Eine zweite Aufnahme neben der laufenden verliert man sonst still.
    if (recording || wartetAufMikrofon) return;
    // Beim ersten Mal fragt macOS nach dem Mikrofon. Bis dahin passierte in der
    // App nichts — es sah aus, als haette der Knopf nicht reagiert.
    setWartetAufMikrofon(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setWartetAufMikrofon(false);
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = e => { if (e.data.size > 0) chunks.current.push(e.data); };
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        stopTicker();
        setRecording(false);
        setSeconds(0);
        // Nicht raten, sondern den Recorder fragen: er weiss, in welchem
        // Format er aufgenommen hat.
        const typ = rec.mimeType || 'audio/mp4';
        const ext = audioExtForMime(typ);
        const blob = new Blob(chunks.current, { type: typ });
        if (blob.size === 0) {
          warning(t('studio.audio.emptyTitle'), t('studio.audio.emptyDetail'));
          return;
        }
        try {
          const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
          const report = await invoke<ImportReport>('studio_add_audio', {
            projectId: project.id, bytes, ext, label: null,
          });
          if (report.duplicates > 0) {
            info(t('studio.audio.recordedTitle'), t('studio.audio.duplicate'));
            return;
          }
          success(t('studio.audio.recordedTitle'),
            t('studio.write.doneDetail', { added: report.added, duplicates: report.duplicates }));
          await loadSamples(0, true);
          setLadeStand(n => n + 1);
          await loadStats();
          // Die Projektliste eigens nachziehen: wer waehrend der Aufnahme
          // zurueck zur Liste geht, stoppt sie damit — gespeichert wird aber
          // erst danach, und die Liste hatte schon mit dem alten Stand geladen.
          const list = await invoke<StudioProject[]>('studio_list_projects');
          const fresh = list.find(x => x.id === project.id);
          if (fresh) onProjectChanged(fresh);
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
      // Auf macOS scheitert die Anfrage ohne Systemfreigabe — das muss man
      // dem Nutzer sagen, sonst sucht er den Fehler in der App.
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

  // ── Tastatur ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const imFeld = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
      if (!current || showExport) return;
      if (imFeld) {
        if (transkript && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
          e.preventDefault(); void confirmTranscript();
        }
        return;
      }
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
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); void skip(); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); goTo(index + 1); return; }
      if (e.key === 'ArrowLeft' || e.key === 'Backspace') { e.preventDefault(); goTo(index - 1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, index, samples, classes, transkript, transcript, showExport]);

  // ── Import ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-import-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setImporting(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  const importFolder = async () => {
    try {
      const sel = await open({ directory: true, multiple: false, title: t('studio.audio.pickFolder') });
      if (!sel || typeof sel !== 'string') return;
      setImporting({ cur: 0, total: 0 });
      const report = await invoke<ImportReport>('studio_import_audio', {
        projectId: project.id, sourcePath: sel, ignoreLabels: false,
      });
      success(t('studio.audio.doneTitle'), t('studio.audio.doneDetail', {
        added: report.added, duplicates: report.duplicates, labels: report.with_labels,
      }));
      if (report.classes_added.length > 0) {
        const list = await invoke<StudioProject[]>('studio_list_projects');
        const fresh = list.find(x => x.id === project.id);
        if (fresh) onProjectChanged(fresh);
      }
      await loadSamples(0, true);
      setLadeStand(n => n + 1);
      await loadStats();
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    } finally {
      setImporting(null);
    }
  };

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

  const confirmed = stats?.confirmed ?? 0;
  const statusDot = (s: SampleStatus) =>
    s === 'confirmed' ? 'bg-emerald-400' : s === 'skipped' ? 'bg-gray-500'
      : s === 'suggested' ? 'bg-amber-400' : 'bg-white/20';
  const uhrzeit = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
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
        <div className="ml-auto flex items-center gap-2">
          {aufnahmeKnopf('px-3 py-2 rounded-lg')}
          <button onClick={() => void importFolder()} disabled={!!importing || recording}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50">
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
            {importing && importing.total > 0
              ? t('studio.import.progress', { current: importing.cur, total: importing.total })
              : t('studio.audio.importFolder')}
          </button>
          <button onClick={() => setShowExport(true)} disabled={confirmed === 0}
            className="px-3 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-40">
            <Download className="w-4 h-4" /> {t('studio.workbench.exportButton')}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20 text-gray-500 gap-2">
          <Loader2 className="w-5 h-5 animate-spin" /> {t('common.loading', 'Lädt…')}
        </div>
      ) : samples.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-12 text-center">
          <AudioLines className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{t('studio.audio.emptyProjectTitle')}</p>
          <p className="text-gray-500 text-sm mt-1 mb-5 max-w-md mx-auto">{t('studio.audio.emptyProjectDetail')}</p>
          <div className="flex items-center justify-center gap-2">
            {aufnahmeKnopf('px-4 py-2.5 rounded-xl')}
            <button onClick={() => void importFolder()}
              className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm inline-flex items-center gap-2">
              <FolderOpen className="w-4 h-4" /> {t('studio.audio.importFolder')}
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-[200px_minmax(0,1fr)_220px] gap-4">
          <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden flex flex-col">
            <div className="flex text-[11px] border-b border-white/10">
              {([['all', t('studio.filter.all')], ['open', t('studio.filter.open')],
                 ['confirmed', t('studio.filter.confirmed')]] as const).map(([val, label]) => (
                <button key={val} onClick={() => setFilter(val)}
                  className={`flex-1 py-2 px-0.5 whitespace-nowrap transition-all ${filter === val ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                  {label}
                </button>
              ))}
            </div>
            <div className="overflow-y-auto max-h-[520px]">
              {samples.map((s, i) => (
                <button key={s.id} onClick={() => goTo(i)}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-all ${i === index ? 'bg-white/10' : 'hover:bg-white/[0.04]'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${statusDot(s.status)}`} />
                  <span className="text-gray-300 text-xs truncate">
                    {s.ann.label ?? ((s.ann.target ?? '').slice(0, 26) || t('studio.audio.noLabel'))}
                  </span>
                </button>
              ))}
              {samples.length < total && (
                <p className="text-gray-600 text-[11px] text-center py-2">
                  {t('studio.workbench.moreLoading', { count: total - samples.length })}
                </p>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5 flex flex-col gap-4">
            {current && (
              <>
                <div className="rounded-lg bg-black/30 border border-white/10 p-4">
                  <audio key={current.id} controls src={convertFileSrc(current.abs_path)} className="w-full" />
                </div>

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
                  <span className="truncate max-w-xs">{current.src.origin?.split(/[\\/]/).pop()}</span>
                </div>

                <div className="flex items-center gap-2">
                  {transkript && (
                    <button onClick={() => void confirmTranscript()}
                      className="px-4 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm inline-flex items-center gap-2">
                      <Check className="w-4 h-4" /> {t('studio.workbench.confirm')}
                    </button>
                  )}
                  <button onClick={() => void skip()}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm inline-flex items-center gap-2">
                    <SkipForward className="w-4 h-4" /> {t('studio.workbench.skip')}
                  </button>
                  <span className="ml-auto" />
                  <RemoveSampleButton key={current.id} onRemove={() => void removeCurrent()} />
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
                      {/* Die Verteilung ist bei einer Klassifikation das, worauf es
                          ankommt: 58 zu 2 trainiert ein Modell, das immer die
                          grosse Klasse sagt. Gezaehlt wird nur Bestaetigtes. */}
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
                  <p>{t('studio.shortcuts.back')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showExport && (
        <AudioExportDialog
          project={project}
          confirmed={confirmed}
          onClose={() => setShowExport(false)}
          onDone={() => { setShowExport(false); void loadStats(); }}
        />
      )}
    </div>
  );
}

// ── Export ────────────────────────────────────────────────────────────────

function AudioExportDialog({ project, confirmed, onClose, onDone }: {
  project: StudioProject; confirmed: number; onClose: () => void; onDone: () => void;
}) {
  const { t } = useLanguage();
  const { success, error } = useNotification();
  const { models, modelId, setModelId } = useStudioModels(project);
  const [name, setName] = useState(project.name);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!modelId) return;
    setBusy(true);
    try {
      await invoke('studio_export', {
        projectId: project.id, modelId, datasetName: name,
        includeSuggested: false, trainRatio: 0, valRatio: 0,
      });
      success(t('studio.export.doneTitle'), t('studio.export.doneDetail', { name }));
      onDone();
    } catch (err: unknown) {
      error(t('studio.export.errorTitle'), String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.export.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">
            {t(project.task === 'transcript' ? 'studio.audio.exportTranscript' : 'studio.audio.exportClasses')}
          </p>
        </div>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.export.nameLabel')}</span>
          <input value={name} onChange={e => setName(e.target.value)}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
        </label>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.export.modelLabel')}</span>
          <StudioModelSelect project={project} models={models} value={modelId} onChange={setModelId} />
        </label>

        <p className="text-gray-500 text-xs">{t('studio.export.summaryAudio', { confirmed })}</p>

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
      </div>
    </div></ModalPortal>
  );
}
