// Bild-Werkbank des Dataset Studio: Boxen setzen, bestaetigen, exportieren.
//
// Die Geometrie (Bildschirmpunkt -> Bildpunkt, Box aus zwei Punkten) kommt aus
// labCorrection.ts, die Klassenfarben aus labGroundTruth.ts — beides ist im
// Labor bereits im Einsatz. Neu ist hier nur der Editor drumherum.
//
// Bedienung ist auf Tastatur ausgelegt: wer 1500 Bilder labelt, greift nicht
// 1500-mal zur Maus. Zahlen waehlen die Klasse, Enter bestaetigt und springt
// zum naechsten offenen Bild.

import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import {
  ArrowLeft, FolderOpen, Download, Loader2, Check, SkipForward,
  Trash2, Plus, AlertTriangle, ImageOff, Info, Wand2, ShieldQuestion, Copy, Undo2, Film,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { clientToImagePoint, boxFromPoints } from '../labCorrection';
import { classColor } from '../labGroundTruth';
import {
  toPixel, toNormalized, isUsable, hitTest, handleAt, resizeTo, movedBy,
  classLabel, summarize, nextOpenIndex,
  type PixelBox, type Handle,
} from './studioBoxes';
import type {
  StudioProject, StudioSample, SamplePage, ImportReport, StudioStats, SampleStatus,
} from './studioTypes';

const PAGE = 200;

type Drag =
  | { kind: 'draw'; from: { x: number; y: number }; to: { x: number; y: number } }
  | { kind: 'move'; idx: number; start: PixelBox; from: { x: number; y: number } }
  | { kind: 'resize'; idx: number; handle: Handle; start: PixelBox };

interface Props {
  project: StudioProject;
  onBack: () => void;
  onProjectChanged: (p: StudioProject) => void;
}

interface ModelInfo { id: string; name: string; }
interface VersionTreeItem { id: string; name: string; version_number: number; }
interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }

interface FolderInspection {
  images:           number;
  with_labels:      number;
  source_classes:   string[];
  max_class_id:     number | null;
  needs_class_list: boolean;
}

interface ReviewReport {
  checked: number;
  doubts:  number;
  agree:   number;
  failed:  number;
}

interface SuggestReport {
  processed:        number;
  with_boxes:       number;
  boxes_total:      number;
  left_confirmed:   number;
  without_boxes:    number;
  unmapped_classes: string[];
  classes_added:    string[];
  model_classes:    string[];
  failed:           number;
}

export default function ImageWorkbench({ project, onBack, onProjectChanged }: Props) {
  const { t } = useLanguage();
  const { success, error, warning, info } = useNotification();

  const [samples, setSamples]   = useState<StudioSample[]>([]);
  const [total, setTotal]       = useState(0);
  const [loading, setLoading]   = useState(true);
  const [index, setIndex]       = useState(0);
  const [filter, setFilter]     = useState<'all' | 'open' | 'confirmed' | 'doubt'>('all');
  const [boxes, setBoxes]       = useState<PixelBox[]>([]);
  const [selected, setSelected] = useState(-1);
  const [activeClass, setActiveClass] = useState(0);
  const [drag, setDrag]         = useState<Drag | null>(null);
  const [stats, setStats]       = useState<StudioStats | null>(null);
  const [importing, setImporting] = useState<{ cur: number; total: number } | null>(null);
  const [newClass, setNewClass] = useState('');
  const [showExport, setShowExport] = useState(false);
  const [modelRun, setModelRun] = useState<'suggest' | 'review' | null>(null);
  const [importPlan, setImportPlan] = useState<{ path: string; inspection: FolderInspection } | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [running, setRunning] = useState<{ cur: number; total: number } | null>(null);

  const [zoom, setZoom] = useState(1);
  const imgRef    = useRef<HTMLImageElement>(null);
  const saveTimer = useRef<number | null>(null);
  /// Boxen-Staende dieses Bildes, juengster zuletzt. Wird beim Bildwechsel geleert.
  const undoStack = useRef<PixelBox[][]>([]);

  const classes = project.classes;
  const current = samples[index];
  const imgW    = current?.meta?.w ?? 0;
  const imgH    = current?.meta?.h ?? 0;

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
    catch { /* Zahlen sind Beiwerk, kein Grund die Werkbank zu blockieren */ }
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

  // Nachladen, bevor das Ende der geladenen Seite erreicht ist.
  useEffect(() => {
    if (samples.length >= total) return;
    if (index < samples.length - 20) return;
    void loadSamples(samples.length, false).catch(() => { /* still */ });
  }, [index, samples.length, total, loadSamples]);

  // Beim Bildwechsel die gespeicherten Boxen in Bildpixel umrechnen.
  //
  // Nur beim echten Wechsel: nach jedem Speichern kommt dasselbe Sample als
  // neues Objekt zurueck. Wuerde hier auch dann zurueckgesetzt, verliert die
  // gerade gezeichnete Box 400 ms spaeter ihre Auswahl — und die Klassentaste
  // greift ins Leere.
  const shownId = useRef<string | null>(null);
  useEffect(() => {
    if (!current) { shownId.current = null; setBoxes([]); setSelected(-1); return; }
    if (shownId.current === current.id) return;
    shownId.current = current.id;
    setBoxes(current.ann.boxes.map(b => toPixel(b, current.meta.w, current.meta.h)));
    setSelected(-1);
    setDrag(null);
    undoStack.current = [];
  }, [current]);

  // ── Speichern ───────────────────────────────────────────────────────────
  const applyDelta = (from: SampleStatus, to: SampleStatus) => {
    if (from === to) return;
    setStats(prev => prev ? { ...prev, [from]: Math.max(0, prev[from] - 1), [to]: prev[to] + 1 } : prev);
  };

  const persist = useCallback(async (status: SampleStatus, px: PixelBox[], sample: StudioSample) => {
    const norm = px.map(b => toNormalized(b, sample.meta.w, sample.meta.h)).filter(isUsable);
    try {
      await invoke('studio_set_annotation', {
        projectId: project.id, sampleId: sample.id, boxes: norm, status,
      });
      applyDelta(sample.status, status);
      setSamples(prev => prev.map(s => s.id === sample.id
        ? { ...s, status, ann: { boxes: norm } } : s));
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, t]);

  /** Jede Box-Aenderung landet verzoegert auf der Platte — kein Speichern-Knopf. */
  const commitBoxes = (next: PixelBox[], merken = true) => {
    // Fuer Rueckgaengig: der Stand vor dieser Aenderung.
    if (merken) undoStack.current.push(boxes);
    setBoxes(next);
    const sample = current;
    if (!sample) return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => { void persist(sample.status, next, sample); }, 400);
  };

  useEffect(() => () => { if (saveTimer.current) window.clearTimeout(saveTimer.current); }, []);

  const undo = () => {
    const vorher = undoStack.current.pop();
    if (!vorher) return;
    commitBoxes(vorher, false);
    setSelected(-1);
  };

  /// Boxen des vorigen Bildes uebernehmen.
  ///
  /// Bei aufeinanderfolgenden Aufnahmen steht fast dasselbe im Bild; eine
  /// verschobene Box ist schneller als eine neu gezogene.
  const copyFromPrevious = () => {
    const vorheriges = samples[index - 1];
    if (!vorheriges || !current || vorheriges.ann.boxes.length === 0) return;
    commitBoxes([
      ...boxes,
      ...vorheriges.ann.boxes.map(b => toPixel(b, current.meta.w, current.meta.h)),
    ]);
  };

  // ── Navigation ──────────────────────────────────────────────────────────
  const goTo = (i: number) => {
    if (samples.length === 0) return;
    setIndex(Math.min(Math.max(i, 0), samples.length - 1));
  };

  const advance = (statusOfCurrent: SampleStatus) => {
    const statuses = samples.map((s, i) => (i === index ? statusOfCurrent : s.status));
    const next = nextOpenIndex(statuses, index);
    goTo(next >= 0 ? next : index + 1);
  };

  const confirmAndNext = async () => {
    if (!current) return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    await persist('confirmed', boxes, current);
    advance('confirmed');
  };

  const skipAndNext = async () => {
    if (!current) return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    await persist('skipped', boxes, current);
    advance('skipped');
  };

  // ── Zeichnen ────────────────────────────────────────────────────────────
  const point = (e: React.PointerEvent) => {
    const el = imgRef.current;
    if (!el) return { x: 0, y: 0 };
    return clientToImagePoint(el.getBoundingClientRect(), imgW, imgH, e.clientX, e.clientY);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!current || imgW <= 0 || classes.length === 0) return;
    e.preventDefault();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const p = point(e);
    const tol = Math.max(6, Math.min(imgW, imgH) * 0.02);
    if (selected >= 0 && boxes[selected]) {
      const h = handleAt(boxes[selected], p.x, p.y, tol);
      if (h) { setDrag({ kind: 'resize', idx: selected, handle: h, start: boxes[selected] }); return; }
    }
    const hit = hitTest(boxes, p.x, p.y);
    if (hit >= 0) {
      setSelected(hit);
      setActiveClass(boxes[hit].cls);
      setDrag({ kind: 'move', idx: hit, start: boxes[hit], from: p });
      return;
    }
    setSelected(-1);
    setDrag({ kind: 'draw', from: p, to: p });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const p = point(e);
    if (drag.kind === 'draw') { setDrag({ ...drag, to: p }); return; }
    if (drag.kind === 'move') {
      setBoxes(prev => prev.map((b, i) => i === drag.idx
        ? movedBy(drag.start, p.x - drag.from.x, p.y - drag.from.y, imgW, imgH) : b));
      return;
    }
    setBoxes(prev => prev.map((b, i) => i === drag.idx
      ? resizeTo(drag.start, drag.handle, p.x, p.y) : b));
  };

  const onPointerUp = () => {
    if (!drag) return;
    if (drag.kind === 'draw') {
      const r = boxFromPoints(drag.from, drag.to, '');
      const px: PixelBox = { cls: activeClass, x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2 };
      // Ein Klick ohne Ziehen ist kein Rechteck, sondern ein Fehlklick.
      if (isUsable(toNormalized(px, imgW, imgH))) {
        commitBoxes([...boxes, px]);
        setSelected(boxes.length);
      }
    } else {
      commitBoxes(boxes);
    }
    setDrag(null);
  };

  // ── Tastatur ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      if (!current || showExport || modelRun || importPlan) return;

      if (e.key >= '1' && e.key <= '9') {
        const cls = Number(e.key) - 1;
        if (cls >= classes.length) return;
        e.preventDefault();
        setActiveClass(cls);
        if (selected >= 0) commitBoxes(boxes.map((b, i) => (i === selected ? { ...b, cls } : b)));
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault(); undo(); return;
      }
      if (e.key === 'v' || e.key === 'V') { e.preventDefault(); copyFromPrevious(); return; }
      if (e.key === '+' || e.key === '=') { e.preventDefault(); setZoom(z => Math.min(z * 1.5, 6)); return; }
      if (e.key === '-') { e.preventDefault(); setZoom(z => Math.max(z / 1.5, 1)); return; }
      if (e.key === '0') { e.preventDefault(); setZoom(1); return; }
      if (e.key === 'Enter')  { e.preventDefault(); void confirmAndNext(); return; }
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); void skipAndNext(); return; }
      if (e.key === 'Escape') { setSelected(-1); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); goTo(index + 1); return; }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); goTo(index - 1); return; }
      // Auf Mac schickt die Entf-Taste 'Backspace'. Ist eine Box ausgewaehlt,
      // ist Loeschen gemeint; sonst der Schritt zurueck.
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        if (selected >= 0) {
          commitBoxes(boxes.filter((_, i) => i !== selected));
          setSelected(-1);
        } else {
          goTo(index - 1);
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, boxes, selected, index, samples, classes.length, showExport, modelRun, importPlan]);

  // ── Import ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-import-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setImporting(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  /// Von der Platte neu einlesen und den gezeigten Bildzustand erzwingen.
  const reloadFromDisk = useCallback(async () => {
    shownId.current = null;
    await loadSamples(0, true);
    await loadStats();
  }, [loadSamples, loadStats]);

  const handleImport = async () => {
    try {
      const sel = await open({ directory: true, multiple: false, title: t('studio.import.dialogTitle') });
      if (!sel || typeof sel !== 'string') return;
      // Bringt der Ordner Labels mit, muss vorher geklaert sein, zu welcher
      // Klassenliste deren Zahlen gehoeren.
      const inspection = await invoke<FolderInspection>('studio_inspect_folder', { sourcePath: sel });
      if (inspection.with_labels > 0) { setImportPlan({ path: sel, inspection }); return; }
      await runImport(sel, null, false);
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    }
  };

  const runImport = async (path: string, labelClasses: string[] | null, ignoreLabels: boolean) => {
    setImportPlan(null);
    try {
      setImporting({ cur: 0, total: 0 });
      const report = await invoke<ImportReport>('studio_import_folder', {
        projectId: project.id, sourcePath: path, labelClasses, ignoreLabels,
      });
      success(
        t('studio.import.doneTitle'),
        t('studio.import.doneDetail', {
          added: report.added, duplicates: report.duplicates,
          labels: report.with_labels, unreadable: report.unreadable,
        }),
      );
      if (report.unknown_ids.length > 0) {
        warning(t('studio.import.unknownIdsTitle'),
          t('studio.import.unknownIdsDetail', { ids: report.unknown_ids.join(', ') }));
      }
      if (report.classes_added.length > 0) {
        info(t('studio.import.classesTitle'), report.classes_added.join(', '));
        const list = await invoke<StudioProject[]>('studio_list_projects');
        const fresh = list.find(p => p.id === project.id);
        if (fresh) onProjectChanged(fresh);
      }
      await reloadFromDisk();
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    } finally {
      setImporting(null);
    }
  };

  useEffect(() => {
    const abos = ['studio-suggest-progress', 'studio-review-progress'].map(name =>
      listen<{ project_id: string; current: number; total: number; done?: boolean }>(name, ev => {
        if (ev.payload.project_id !== project.id) return;
        setRunning(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      }));
    return () => { abos.forEach(a => { void a.then(f => f()); }); };
  }, [project.id]);

  const handlePickVideo = async () => {
    try {
      const sel = await open({
        multiple: false, title: t('studio.video.dialogTitle'),
        filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'mkv', 'm4v', 'webm'] }],
      });
      if (sel && typeof sel === 'string') setVideoPath(sel);
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    }
  };

  const runVideoImport = async (everyN: number, maxFrames: number) => {
    const path = videoPath;
    setVideoPath(null);
    if (!path) return;
    try {
      setImporting({ cur: 0, total: 0 });
      const report = await invoke<ImportReport>('studio_import_video', {
        projectId: project.id, videoPath: path, everyN, maxFrames,
      });
      success(t('studio.video.doneTitle'),
        t('studio.video.doneDetail', { added: report.added, duplicates: report.duplicates }));
      await reloadFromDisk();
    } catch (err: unknown) {
      error(t('studio.video.errorTitle'), String(err));
    } finally {
      setImporting(null);
    }
  };

  const handleAddClass = async () => {
    const name = newClass.trim();
    if (!name) return;
    try {
      const updated = await invoke<StudioProject>('studio_update_project', {
        projectId: project.id, classes: [...classes, name],
      });
      onProjectChanged(updated);
      setNewClass('');
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    }
  };

  // ── Darstellung ─────────────────────────────────────────────────────────
  const stroke = Math.max(1.5, Math.min(imgW || 1, imgH || 1) / 300);
  const fontSize = Math.max(11, Math.min(imgW || 1, imgH || 1) / 28);
  const draftBox = drag?.kind === 'draw'
    ? boxFromPoints(drag.from, drag.to, '')
    : null;
  const confirmed = stats?.confirmed ?? 0;
  const statusDot = (s: SampleStatus) =>
    s === 'confirmed' ? 'bg-emerald-400'
      : s === 'skipped' ? 'bg-gray-500'
        : s === 'suggested' ? 'bg-amber-400' : 'bg-white/20';

  return (
    <div className="space-y-4">
      {/* Kopf */}
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
          <button onClick={handleImport} disabled={!!importing}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50">
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
            {importing && importing.total > 0
              ? t('studio.import.progress', { current: importing.cur, total: importing.total })
              : t('studio.workbench.importButton')}
          </button>
          <button onClick={handlePickVideo} disabled={!!importing}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50">
            <Film className="w-4 h-4" /> {t('studio.workbench.videoButton')}
          </button>
          <button onClick={() => setModelRun('suggest')} disabled={!!running || total === 0}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50">
            {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
            {running && running.total > 0
              ? t('studio.suggest.progress', { current: running.cur, total: running.total })
              : t('studio.workbench.suggestButton')}
          </button>
          <button onClick={() => setModelRun('review')} disabled={!!running || confirmed === 0}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50"
            title={confirmed === 0 ? t('studio.review.needsConfirmed') : undefined}>
            <ShieldQuestion className="w-4 h-4" />
            {t('studio.workbench.reviewButton')}
          </button>
          <button onClick={() => setShowExport(true)} disabled={confirmed === 0}
            className="px-3 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-40"
            title={confirmed === 0 ? t('studio.export.needsConfirmed') : undefined}>
            <Download className="w-4 h-4" />
            {t('studio.workbench.exportButton')}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20 text-gray-500 gap-2">
          <Loader2 className="w-5 h-5 animate-spin" /> {t('common.loading', 'Lädt…')}
        </div>
      ) : samples.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-12 text-center">
          <ImageOff className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{t('studio.workbench.emptyTitle')}</p>
          <p className="text-gray-500 text-sm mt-1 mb-5">{t('studio.workbench.emptyDetail')}</p>
          <button onClick={handleImport}
            className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center gap-2">
            <FolderOpen className="w-4 h-4" /> {t('studio.workbench.importButton')}
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-[200px_minmax(0,1fr)_220px] gap-4">
          {/* Warteschlange */}
          <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden flex flex-col">
            <div className="flex text-[11px] border-b border-white/10">
              {([
                ['all', t('studio.filter.all')],
                ['open', t('studio.filter.open')],
                ['confirmed', t('studio.filter.confirmed')],
                ['doubt', stats?.doubts ? `${t('studio.filter.doubt')} ${stats.doubts}` : t('studio.filter.doubt')],
              ] as const).map(([val, label]) => (
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
                  <span className="text-gray-400 text-[11px] tabular-nums flex-shrink-0">{i + 1}</span>
                  <span className="text-gray-300 text-xs truncate">
                    {summarize(s.ann.boxes, classes) || t('studio.workbench.noBoxes')}
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

          {/* Bild */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-4 flex flex-col items-center gap-3">
            {current && (
              <>
                <div className={`w-full flex justify-center ${zoom > 1 ? 'overflow-auto max-h-[520px]' : ''}`}>
                <div className="relative inline-block">
                  <img
                    ref={imgRef}
                    src={convertFileSrc(current.abs_path)}
                    alt=""
                    role="presentation"
                    draggable={false}
                    style={{ maxHeight: 520 * zoom }}
                    className="max-w-none object-contain block rounded-lg select-none cursor-crosshair"
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                  />
                  {imgW > 0 && imgH > 0 && (
                    <svg viewBox={`0 0 ${imgW} ${imgH}`}
                      className="absolute inset-0 w-full h-full pointer-events-none">
                      {boxes.map((b, i) => {
                        const color = classColor(classLabel(b.cls, classes), classes);
                        const w = b.x2 - b.x1, h = b.y2 - b.y1;
                        const labelY = b.y1 > fontSize * 1.3 ? b.y1 - fontSize * 0.4 : b.y1 + fontSize;
                        return (
                          <g key={i}>
                            <rect x={b.x1} y={b.y1} width={w} height={h}
                              fill={i === selected ? color : 'none'} fillOpacity={i === selected ? 0.12 : 0}
                              stroke={color} strokeWidth={i === selected ? stroke * 1.8 : stroke}
                              strokeDasharray={current.status === 'suggested' ? `${stroke * 5} ${stroke * 3}` : undefined} />
                            <text x={b.x1 + stroke} y={labelY} fill={color}
                              fontSize={fontSize} fontWeight={600}>
                              {classLabel(b.cls, classes)}
                            </text>
                            {i === selected && ([
                              [b.x1, b.y1], [b.x2, b.y1], [b.x1, b.y2], [b.x2, b.y2],
                            ] as const).map(([cx, cy], k) => (
                              <rect key={k} x={cx - stroke * 3} y={cy - stroke * 3}
                                width={stroke * 6} height={stroke * 6} fill={color} />
                            ))}
                          </g>
                        );
                      })}
                      {draftBox && (
                        <rect x={draftBox.x1} y={draftBox.y1}
                          width={draftBox.x2 - draftBox.x1} height={draftBox.y2 - draftBox.y1}
                          fill="none" stroke={classColor(classLabel(activeClass, classes), classes)}
                          strokeWidth={stroke} strokeDasharray={`${stroke * 4} ${stroke * 3}`} />
                      )}
                    </svg>
                  )}
                </div>
                </div>
                {current.doubt && (
                  <div className="flex items-start gap-2 text-[11px] text-orange-200/90 bg-orange-500/10 border border-orange-500/25 rounded-lg px-3 py-1.5 max-w-xl">
                    <ShieldQuestion className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>
                      {current.doubt.missing.length > 0 &&
                        t('studio.review.missing', { classes: current.doubt.missing.join(', ') })}
                      {current.doubt.missing.length > 0 && current.doubt.extra.length > 0 && ' · '}
                      {current.doubt.extra.length > 0 &&
                        t('studio.review.extra', { classes: current.doubt.extra.join(', ') })}
                    </span>
                  </div>
                )}
                {current.status === 'suggested' && (
                  <div className="flex items-center gap-2 text-[11px] text-amber-300/90 bg-amber-500/10 border border-amber-500/25 rounded-lg px-3 py-1.5">
                    <Wand2 className="w-3.5 h-3.5 flex-shrink-0" />
                    {t('studio.suggest.hint')}
                  </div>
                )}
                <div className="flex items-center gap-3 text-[11px] text-gray-500">
                  <span className="tabular-nums">{index + 1} / {total}</span>
                  <span>{imgW} x {imgH}</span>
                  <span className="truncate max-w-xs">{current.src.origin?.split(/[\\/]/).pop()}</span>
                  {zoom > 1 && <span className="tabular-nums">{Math.round(zoom * 100)} %</span>}
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => void confirmAndNext()}
                    className="px-4 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm inline-flex items-center gap-2">
                    <Check className="w-4 h-4" /> {t('studio.workbench.confirm')}
                  </button>
                  <button onClick={() => void skipAndNext()}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm inline-flex items-center gap-2">
                    <SkipForward className="w-4 h-4" /> {t('studio.workbench.skip')}
                  </button>
                  <button onClick={copyFromPrevious} disabled={index === 0 || !samples[index - 1]?.ann.boxes.length}
                    className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm inline-flex items-center gap-2 disabled:opacity-30"
                    title={t('studio.workbench.copyPreviousHint')}>
                    <Copy className="w-4 h-4" /> {t('studio.workbench.copyPrevious')}
                  </button>
                  <button onClick={undo} disabled={undoStack.current.length === 0}
                    className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm inline-flex items-center gap-2 disabled:opacity-30">
                    <Undo2 className="w-4 h-4" /> {t('studio.workbench.undo')}
                  </button>
                  {selected >= 0 && (
                    <button onClick={() => { commitBoxes(boxes.filter((_, i) => i !== selected)); setSelected(-1); }}
                      className="px-4 py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-300 text-sm inline-flex items-center gap-2">
                      <Trash2 className="w-4 h-4" /> {t('studio.workbench.deleteBox')}
                    </button>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Klassen */}
          <div className="space-y-3">
            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <p className="text-gray-500 text-xs mb-2">{t('studio.workbench.classesTitle')}</p>
              {classes.length === 0 && (
                <p className="text-amber-300/80 text-xs mb-2 flex items-start gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                  {t('studio.workbench.noClasses')}
                </p>
              )}
              <div className="space-y-1">
                {classes.map((name, i) => (
                  <button key={i}
                    onClick={() => {
                      setActiveClass(i);
                      if (selected >= 0) commitBoxes(boxes.map((b, k) => (k === selected ? { ...b, cls: i } : b)));
                    }}
                    className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left transition-all ${activeClass === i ? 'bg-white/10' : 'hover:bg-white/[0.04]'}`}>
                    <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                      style={{ background: classColor(name, classes) }} />
                    {i < 9 && (
                      <span className="text-[10px] font-mono text-gray-500 border border-white/10 rounded px-1">{i + 1}</span>
                    )}
                    <span className="text-gray-200 text-xs truncate">{name}</span>
                    <span className="ml-auto text-gray-500 text-[11px] tabular-nums">
                      {stats?.per_class?.[i] ?? 0}
                    </span>
                  </button>
                ))}
              </div>
              <div className="flex gap-1.5 mt-2">
                <input value={newClass} onChange={e => setNewClass(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') void handleAddClass(); }}
                  placeholder={t('studio.workbench.newClassPlaceholder')}
                  className="flex-1 min-w-0 px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25" />
                <button onClick={() => void handleAddClass()}
                  className="px-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300"
                  aria-label={t('studio.workbench.addClass')}>
                  <Plus className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {stats && (
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 space-y-1.5">
                <p className="text-gray-500 text-xs mb-1">{t('studio.stats.title')}</p>
                {([
                  ['confirmed', stats.confirmed], ['open', stats.new + stats.suggested],
                  ['skipped', stats.skipped], ['boxes', stats.boxes_total],
                  ['doubts', stats.doubts],
                ] as const).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">{t(`studio.stats.${key}`)}</span>
                    <span className="text-gray-200 tabular-nums">{val}</span>
                  </div>
                ))}
                {stats.empty_confirmed > 0 && (
                  <p className="text-gray-500 text-[11px] pt-1 flex items-start gap-1.5">
                    <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
                    {t('studio.stats.emptyConfirmed', { count: stats.empty_confirmed })}
                  </p>
                )}
              </div>
            )}

            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <p className="text-gray-500 text-xs mb-2">{t('studio.shortcuts.title')}</p>
              <div className="space-y-1 text-[11px] text-gray-400">
                {['classes', 'confirm', 'skip', 'back', 'delete', 'undo', 'copy', 'zoom'].map(k => (
                  <p key={k}>{t(`studio.shortcuts.${k}`)}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {videoPath && (
        <VideoDialog
          path={videoPath}
          onCancel={() => setVideoPath(null)}
          onRun={(everyN, maxFrames) => void runVideoImport(everyN, maxFrames)}
        />
      )}

      {importPlan && (
        <ImportDialog
          inspection={importPlan.inspection}
          onCancel={() => setImportPlan(null)}
          onRun={(classes, ignore) => void runImport(importPlan.path, classes, ignore)}
        />
      )}

      {modelRun && (
        <ModelRunDialog
          mode={modelRun}
          project={project}
          onClose={() => setModelRun(null)}
          onDone={async () => {
            setModelRun(null);
            const list = await invoke<StudioProject[]>('studio_list_projects');
            const fresh = list.find(p => p.id === project.id);
            if (fresh) onProjectChanged(fresh);
            await reloadFromDisk();
          }}
        />
      )}

      {showExport && (
        <ExportDialog
          project={project}
          confirmed={confirmed}
          suggested={stats?.suggested ?? 0}
          onClose={() => setShowExport(false)}
          onDone={() => { setShowExport(false); void loadStats(); }}
        />
      )}
    </div>
  );
}

// ── Export ────────────────────────────────────────────────────────────────

function ExportDialog({ project, confirmed, suggested, onClose, onDone }: {
  project: StudioProject; confirmed: number; suggested: number;
  onClose: () => void; onDone: () => void;
}) {
  const { t } = useLanguage();
  const { success, error } = useNotification();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState('');
  const [name, setName] = useState(project.name);
  const [includeSuggested, setIncludeSuggested] = useState(false);
  const [split, setSplit] = useState(false);
  const [trainPct, setTrainPct] = useState(70);
  const [valPct, setValPct] = useState(20);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    invoke<ModelInfo[]>('list_models')
      .then(list => { setModels(list); if (list.length > 0) setModelId(list[0].id); })
      .catch(() => { /* Auswahl bleibt leer, der Knopf bleibt gesperrt */ });
  }, []);

  const run = async () => {
    if (!modelId) return;
    setBusy(true);
    try {
      await invoke('studio_export', {
        projectId: project.id, modelId, datasetName: name, includeSuggested,
        trainRatio: split ? trainPct / 100 : 0,
        valRatio:   split ? valPct / 100 : 0,
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.export.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.export.subtitle')}</p>
        </div>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.export.nameLabel')}</span>
          <input value={name} onChange={e => setName(e.target.value)}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
        </label>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.export.modelLabel')}</span>
          <select value={modelId} onChange={e => setModelId(e.target.value)}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
            {models.length === 0 && <option value="">{t('studio.export.noModels')}</option>}
            {models.map(m => <option key={m.id} value={m.id} className="bg-[#101218]">{m.name}</option>)}
          </select>
        </label>

        <label className="flex items-start gap-2 cursor-pointer">
          <input type="checkbox" checked={includeSuggested} disabled={suggested === 0}
            onChange={e => setIncludeSuggested(e.target.checked)} className="mt-0.5" />
          <span className="text-gray-300 text-xs">
            {t('studio.export.includeSuggested', { count: suggested })}
          </span>
        </label>

        <label className="flex items-start gap-2 cursor-pointer">
          <input type="checkbox" checked={split} onChange={e => setSplit(e.target.checked)} className="mt-0.5" />
          <span className="text-gray-300 text-xs">{t('studio.export.splitLabel')}</span>
        </label>

        {split && (
          <div className="pl-6 space-y-2">
            <label className="block">
              <span className="text-gray-400 text-xs">
                {t('studio.export.ratios', { train: trainPct, val: valPct, test: Math.max(0, 100 - trainPct - valPct) })}
              </span>
              <input type="range" min={40} max={95} step={5} value={trainPct}
                onChange={e => {
                  const v = Number(e.target.value);
                  setTrainPct(v);
                  if (v + valPct > 100) setValPct(Math.max(0, 100 - v));
                }}
                className="mt-2 w-full" />
              <input type="range" min={0} max={50} step={5} value={valPct}
                onChange={e => setValPct(Math.min(Number(e.target.value), 100 - trainPct))}
                className="mt-1 w-full" />
            </label>
            <p className="text-gray-500 text-[11px]">{t('studio.export.splitGroupNote')}</p>
          </div>
        )}

        <p className="text-gray-500 text-xs">{t('studio.export.summary', { confirmed })}</p>

        <div className="flex gap-2 pt-1">
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
    </div>
  );
}

// ── Vorschlaege ───────────────────────────────────────────────────────────

function ModelRunDialog({ mode, project, onClose, onDone }: {
  mode: 'suggest' | 'review';
  project: StudioProject; onClose: () => void; onDone: () => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const [tree, setTree] = useState<ModelWithVersionTree[]>([]);
  const [versionId, setVersionId] = useState('');
  const [minConfidence, setMinConfidence] = useState(0.25);
  const [addUnknown, setAddUnknown] = useState(false);
  const [busy, setBusy] = useState(false);
  const [suggestReport, setSuggestReport] = useState<SuggestReport | null>(null);
  const [reviewReport, setReviewReport] = useState<ReviewReport | null>(null);
  const suggest = mode === 'suggest';

  useEffect(() => {
    invoke<ModelWithVersionTree[]>('list_models_with_version_tree')
      .then(list => {
        const withVersions = list.filter(m => m.versions.length > 0);
        setTree(withVersions);
        const first = withVersions[0]?.versions[0];
        if (first) setVersionId(first.id);
      })
      .catch(() => { /* Auswahl bleibt leer, der Knopf bleibt gesperrt */ });
  }, []);

  const run = async () => {
    if (!versionId) return;
    setBusy(true);
    try {
      if (suggest) {
        setSuggestReport(await invoke<SuggestReport>('studio_suggest', {
          projectId: project.id, versionId, minConfidence, addUnknownClasses: addUnknown,
        }));
      } else {
        setReviewReport(await invoke<ReviewReport>('studio_review', {
          projectId: project.id, versionId, minConfidence,
        }));
      }
    } catch (err: unknown) {
      error(t(suggest ? 'studio.suggest.errorTitle' : 'studio.review.errorTitle'), String(err));
    } finally {
      setBusy(false);
    }
  };

  const report = suggestReport || reviewReport;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">
            {t(suggest ? 'studio.suggest.title' : 'studio.review.title')}
          </h3>
          <p className="text-gray-500 text-xs mt-1">
            {t(suggest ? 'studio.suggest.subtitle' : 'studio.review.subtitle')}
          </p>
        </div>

        {report ? (
          <div className="space-y-3">
            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
              {suggestReport && ([
                ['withBoxes', suggestReport.with_boxes],
                ['boxes', suggestReport.boxes_total],
                ['withoutBoxes', suggestReport.without_boxes],
                ['leftConfirmed', suggestReport.left_confirmed],
              ] as const).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t(`studio.suggest.report.${key}`)}</span>
                  <span className="text-gray-200 tabular-nums">{val}</span>
                </div>
              ))}
              {reviewReport && ([
                ['checked', reviewReport.checked],
                ['doubts', reviewReport.doubts],
                ['agree', reviewReport.agree],
              ] as const).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t(`studio.review.report.${key}`)}</span>
                  <span className={`tabular-nums ${key === 'doubts' && val > 0 ? 'text-orange-300' : 'text-gray-200'}`}>{val}</span>
                </div>
              ))}
              {report.failed > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-red-300">{t('studio.suggest.report.failed')}</span>
                  <span className="text-red-300 tabular-nums">{report.failed}</span>
                </div>
              )}
            </div>

            {suggestReport && suggestReport.unmapped_classes.length > 0 && (
              <div className="rounded-lg bg-amber-500/10 border border-amber-500/25 p-3">
                <p className="text-amber-200 text-xs font-medium mb-1">
                  {t('studio.suggest.report.unmappedTitle')}
                </p>
                <p className="text-amber-200/80 text-xs">{suggestReport.unmapped_classes.join(', ')}</p>
                <p className="text-gray-400 text-[11px] mt-1.5">
                  {t('studio.suggest.report.unmappedHint')}
                </p>
              </div>
            )}

            {suggestReport && suggestReport.classes_added.length > 0 && (
              <p className="text-gray-400 text-xs">
                {t('studio.suggest.report.classesAdded')}: {suggestReport.classes_added.join(', ')}
              </p>
            )}

            {reviewReport && reviewReport.doubts > 0 && (
              <p className="text-gray-400 text-xs">{t('studio.review.report.hint')}</p>
            )}

            <button onClick={onDone}
              className="w-full py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm">
              {t('studio.suggest.report.close')}
            </button>
          </div>
        ) : (
          <>
            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.suggest.modelLabel')}</span>
              <select value={versionId} onChange={e => setVersionId(e.target.value)}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
                {tree.length === 0 && <option value="">{t('studio.suggest.noVersions')}</option>}
                {tree.map(m => (
                  <optgroup key={m.id} label={m.name} className="bg-[#101218]">
                    {m.versions.map(v => (
                      <option key={v.id} value={v.id} className="bg-[#101218]">
                        {m.name} · v{v.version_number} {v.name}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="text-gray-400 text-xs">
                {t('studio.suggest.confidenceLabel', { value: Math.round(minConfidence * 100) })}
              </span>
              <input type="range" min={5} max={95} step={5} value={Math.round(minConfidence * 100)}
                onChange={e => setMinConfidence(Number(e.target.value) / 100)}
                className="mt-2 w-full" />
            </label>

            {suggest && (
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={addUnknown}
                  onChange={e => setAddUnknown(e.target.checked)} className="mt-0.5" />
                <span className="text-gray-300 text-xs">{t('studio.suggest.addUnknown')}</span>
              </label>
            )}

            <p className="text-gray-500 text-xs">
              {t(suggest ? 'studio.suggest.safetyNote' : 'studio.review.safetyNote')}
            </p>

            <div className="flex gap-2 pt-1">
              <button onClick={onClose} disabled={busy}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm disabled:opacity-40">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void run()} disabled={busy || !versionId}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" />
                  : suggest ? <Wand2 className="w-4 h-4" /> : <ShieldQuestion className="w-4 h-4" />}
                {t('studio.suggest.runButton')}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Import mit Klassenfrage ───────────────────────────────────────────────

type ClassSource = 'folder' | 'model' | 'manual' | 'ignore';

function ImportDialog({ inspection, onCancel, onRun }: {
  inspection: FolderInspection;
  onCancel: () => void;
  onRun: (classes: string[] | null, ignore: boolean) => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const hasFolderList = inspection.source_classes.length > 0;
  const [source, setSource] = useState<ClassSource>(hasFolderList ? 'folder' : 'model');
  const [tree, setTree] = useState<ModelWithVersionTree[]>([]);
  const [versionId, setVersionId] = useState('');
  const [modelClasses, setModelClasses] = useState<string[] | null>(null);
  const [manual, setManual] = useState('');
  const [loadingClasses, setLoadingClasses] = useState(false);

  useEffect(() => {
    invoke<ModelWithVersionTree[]>('list_models_with_version_tree')
      .then(list => {
        const withVersions = list.filter(m => m.versions.length > 0);
        setTree(withVersions);
        const first = withVersions[0]?.versions[0];
        if (first) setVersionId(first.id);
      })
      .catch(() => { /* dann bleibt nur die Handeingabe */ });
  }, []);

  const loadFromModel = async () => {
    if (!versionId) return;
    setLoadingClasses(true);
    try {
      setModelClasses(await invoke<string[]>('studio_model_classes', { versionId }));
    } catch (err: unknown) {
      error(t('studio.import.classesFromModelError'), String(err));
    } finally {
      setLoadingClasses(false);
    }
  };

  const chosen: string[] | null =
    source === 'folder' ? inspection.source_classes
      : source === 'model' ? modelClasses
        : source === 'manual' ? manual.split(/[,\n]/).map(c => c.trim()).filter(Boolean)
          : null;

  const needed = (inspection.max_class_id ?? 0) + 1;
  const tooShort = source !== 'ignore' && chosen !== null && chosen.length > 0 && chosen.length < needed;
  const ready = source === 'ignore' || (chosen !== null && chosen.length > 0);

  const option = (val: ClassSource, label: string, hint?: string, disabled = false) => (
    <label className={`flex items-start gap-2 p-2 rounded-lg transition-all ${disabled ? 'opacity-40' : 'cursor-pointer hover:bg-white/[0.04]'} ${source === val ? 'bg-white/[0.06]' : ''}`}>
      <input type="radio" name="classSource" checked={source === val} disabled={disabled}
        onChange={() => setSource(val)} className="mt-0.5" />
      <span>
        <span className="text-gray-200 text-xs block">{label}</span>
        {hint && <span className="text-gray-500 text-[11px] block mt-0.5">{hint}</span>}
      </span>
    </label>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onCancel}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.import.classDialogTitle')}</h3>
          <p className="text-gray-400 text-xs mt-1">
            {t('studio.import.classDialogSummary', {
              images: inspection.images, labels: inspection.with_labels,
              ids: needed,
            })}
          </p>
        </div>

        <div className="rounded-lg bg-amber-500/10 border border-amber-500/25 p-3">
          <p className="text-amber-200/90 text-xs">{t('studio.import.classDialogWhy')}</p>
        </div>

        <div className="space-y-1">
          {option('folder',
            t('studio.import.optionFolder', { count: inspection.source_classes.length }),
            hasFolderList ? inspection.source_classes.slice(0, 6).join(', ') : t('studio.import.optionFolderMissing'),
            !hasFolderList)}

          {option('model', t('studio.import.optionModel'), t('studio.import.optionModelHint'))}
          {source === 'model' && (
            <div className="pl-7 pr-2 pb-2 space-y-2">
              <div className="flex gap-2">
                <select value={versionId} onChange={e => { setVersionId(e.target.value); setModelClasses(null); }}
                  className="flex-1 min-w-0 px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs focus:outline-none focus:border-white/25">
                  {tree.length === 0 && <option value="">{t('studio.suggest.noVersions')}</option>}
                  {tree.map(m => (
                    <optgroup key={m.id} label={m.name} className="bg-[#101218]">
                      {m.versions.map(v => (
                        <option key={v.id} value={v.id} className="bg-[#101218]">
                          {m.name} · v{v.version_number} {v.name}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <button onClick={() => void loadFromModel()} disabled={!versionId || loadingClasses}
                  className="px-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-xs inline-flex items-center gap-1.5 disabled:opacity-40">
                  {loadingClasses && <Loader2 className="w-3 h-3 animate-spin" />}
                  {t('studio.import.loadClasses')}
                </button>
              </div>
              {modelClasses && (
                <p className="text-gray-400 text-[11px]">
                  {modelClasses.length}: {modelClasses.join(', ')}
                </p>
              )}
            </div>
          )}

          {option('manual', t('studio.import.optionManual'), t('studio.import.optionManualHint'))}
          {source === 'manual' && (
            <div className="pl-7 pr-2 pb-2">
              <textarea value={manual} onChange={e => setManual(e.target.value)} rows={3}
                placeholder={t('studio.import.manualPlaceholder')}
                className="w-full px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white text-xs placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none" />
            </div>
          )}

          {option('ignore', t('studio.import.optionIgnore'), t('studio.import.optionIgnoreHint'))}
        </div>

        {tooShort && (
          <p className="text-amber-300 text-xs flex items-start gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            {t('studio.import.tooShort', { have: chosen?.length ?? 0, need: needed })}
          </p>
        )}

        <div className="flex gap-2">
          <button onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
            {t('common.cancel', 'Abbrechen')}
          </button>
          <button onClick={() => onRun(source === 'ignore' ? null : chosen, source === 'ignore')}
            disabled={!ready}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
            <FolderOpen className="w-4 h-4" /> {t('studio.import.startButton')}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Video ─────────────────────────────────────────────────────────────────

function VideoDialog({ path, onCancel, onRun }: {
  path: string; onCancel: () => void; onRun: (everyN: number, maxFrames: number) => void;
}) {
  const { t } = useLanguage();
  const [everyN, setEveryN] = useState(15);
  const [maxFrames, setMaxFrames] = useState(500);
  const name = path.split(/[\\/]/).pop() ?? path;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onCancel}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.video.title')}</h3>
          <p className="text-gray-500 text-xs mt-1 truncate">{name}</p>
        </div>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.video.everyNLabel', { n: everyN })}</span>
          <input type="range" min={1} max={120} step={1} value={everyN}
            onChange={e => setEveryN(Number(e.target.value))} className="mt-2 w-full" />
          <span className="text-gray-600 text-[11px]">{t('studio.video.everyNHint')}</span>
        </label>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.video.maxLabel')}</span>
          <input type="number" min={1} max={20000} value={maxFrames}
            onChange={e => setMaxFrames(Math.max(1, Math.min(20000, Number(e.target.value) || 1)))}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
        </label>

        <p className="text-gray-500 text-xs">{t('studio.video.groupNote')}</p>

        <div className="flex gap-2">
          <button onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
            {t('common.cancel', 'Abbrechen')}
          </button>
          <button onClick={() => onRun(everyN, maxFrames)}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2">
            <Film className="w-4 h-4" /> {t('studio.video.runButton')}
          </button>
        </div>
      </div>
    </div>
  );
}
