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
  Trash2, Plus, AlertTriangle, ImageOff, Info,
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

export default function ImageWorkbench({ project, onBack, onProjectChanged }: Props) {
  const { t } = useLanguage();
  const { success, error, warning, info } = useNotification();

  const [samples, setSamples]   = useState<StudioSample[]>([]);
  const [total, setTotal]       = useState(0);
  const [loading, setLoading]   = useState(true);
  const [index, setIndex]       = useState(0);
  const [filter, setFilter]     = useState<'all' | 'open' | 'confirmed'>('all');
  const [boxes, setBoxes]       = useState<PixelBox[]>([]);
  const [selected, setSelected] = useState(-1);
  const [activeClass, setActiveClass] = useState(0);
  const [drag, setDrag]         = useState<Drag | null>(null);
  const [stats, setStats]       = useState<StudioStats | null>(null);
  const [importing, setImporting] = useState<{ cur: number; total: number } | null>(null);
  const [newClass, setNewClass] = useState('');
  const [showExport, setShowExport] = useState(false);

  const imgRef    = useRef<HTMLImageElement>(null);
  const saveTimer = useRef<number | null>(null);

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
  const commitBoxes = (next: PixelBox[]) => {
    setBoxes(next);
    const sample = current;
    if (!sample) return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => { void persist(sample.status, next, sample); }, 400);
  };

  useEffect(() => () => { if (saveTimer.current) window.clearTimeout(saveTimer.current); }, []);

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
      if (!current || showExport) return;

      if (e.key >= '1' && e.key <= '9') {
        const cls = Number(e.key) - 1;
        if (cls >= classes.length) return;
        e.preventDefault();
        setActiveClass(cls);
        if (selected >= 0) commitBoxes(boxes.map((b, i) => (i === selected ? { ...b, cls } : b)));
        return;
      }
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
  }, [current, boxes, selected, index, samples, classes.length, showExport]);

  // ── Import ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-import-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setImporting(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  const handleImport = async () => {
    try {
      const sel = await open({ directory: true, multiple: false, title: t('studio.import.dialogTitle') });
      if (!sel || typeof sel !== 'string') return;
      setImporting({ cur: 0, total: 0 });
      const report = await invoke<ImportReport>('studio_import_folder', {
        projectId: project.id, sourcePath: sel,
      });
      success(
        t('studio.import.doneTitle'),
        t('studio.import.doneDetail', {
          added: report.added, duplicates: report.duplicates,
          labels: report.with_labels, unreadable: report.unreadable,
        }),
      );
      if (report.classes_added.length > 0) {
        info(t('studio.import.classesTitle'), report.classes_added.join(', '));
        const list = await invoke<StudioProject[]>('studio_list_projects');
        const fresh = list.find(p => p.id === project.id);
        if (fresh) onProjectChanged(fresh);
      }
      await loadSamples(0, true);
      await loadStats();
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
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
        <div className="grid grid-cols-[180px_minmax(0,1fr)_220px] gap-4">
          {/* Warteschlange */}
          <div className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden flex flex-col">
            <div className="flex text-xs border-b border-white/10">
              {([
                ['all', t('studio.filter.all')],
                ['open', t('studio.filter.open')],
                ['confirmed', t('studio.filter.confirmed')],
              ] as const).map(([val, label]) => (
                <button key={val} onClick={() => setFilter(val)}
                  className={`flex-1 py-2 transition-all ${filter === val ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
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
                <div className="relative inline-block">
                  <img
                    ref={imgRef}
                    src={convertFileSrc(current.abs_path)}
                    alt=""
                    role="presentation"
                    draggable={false}
                    className="max-h-[520px] max-w-full object-contain block rounded-lg select-none cursor-crosshair"
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
                              stroke={color} strokeWidth={i === selected ? stroke * 1.8 : stroke} />
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
                <div className="flex items-center gap-3 text-[11px] text-gray-500">
                  <span className="tabular-nums">{index + 1} / {total}</span>
                  <span>{imgW} x {imgH}</span>
                  <span className="truncate max-w-xs">{current.src.origin?.split(/[\\/]/).pop()}</span>
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
                {['classes', 'confirm', 'skip', 'back', 'delete'].map(k => (
                  <p key={k}>{t(`studio.shortcuts.${k}`)}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
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
