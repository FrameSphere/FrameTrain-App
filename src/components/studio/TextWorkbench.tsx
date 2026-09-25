// Text-Werkbank des Dataset Studio: Klassifikation und Paare.
//
// Dieselbe Ablage, dieselben Befehle, dieselbe Bedienung wie bei Bildern —
// nur steht in der Mitte Text statt eines Bildes. Wer 2000 Zeilen einsortiert,
// greift genauso wenig zur Maus wie beim Boxenziehen: Zahl waehlt die Klasse,
// Enter bestaetigt und springt weiter.

import { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import {
  ArrowLeft, FolderOpen, Download, Loader2, Check, SkipForward,
  Plus, AlertTriangle, FileText, ChevronDown, Info, Wand2, ShieldQuestion, PenLine, Globe, Sparkles,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { classColor } from '../labGroundTruth';
import { useStudioModels, StudioModelSelect } from './studioModels';
import { statsNachAenderung } from './studioStats';
import { nextOpenIndex } from './studioBoxes';
import ModelRunDialog from './ModelRunDialog';
import FetchDialog from './FetchDialog';
import GenerateDialog from './GenerateDialog';
import type { GeneratedItem } from './generatedTexts';
import type {
  StudioProject, StudioSample, SamplePage, ImportReport, StudioStats, SampleStatus,
} from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import RemoveSampleButton from './RemoveSampleButton';
import { useEscape } from './useEscape';

const PAGE = 200;

interface TextInspection {
  kind:         string;
  rows:         number;
  columns:      string[];
  text_column:  string | null;
  label_column: string | null;
  preview:      string[];
  labels:       string[];
}

interface Props {
  project: StudioProject;
  onBack: () => void;
  onProjectChanged: (p: StudioProject) => void;
}

export default function TextWorkbench({ project, onBack, onProjectChanged }: Props) {
  const { t } = useLanguage();
  const { success, error, warning } = useNotification();

  const [samples, setSamples] = useState<StudioSample[]>([]);
  const [total, setTotal]     = useState(0);
  const [loading, setLoading] = useState(true);
  const [index, setIndex]     = useState(0);
  const [filter, setFilter]   = useState<'all' | 'open' | 'confirmed'>('all');
  const [stats, setStats]     = useState<StudioStats | null>(null);
  const [newClass, setNewClass] = useState('');
  const [importing, setImporting] = useState<{ cur: number; total: number } | null>(null);
  const [plan, setPlan] = useState<{ path: string; inspection: TextInspection } | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [modelRun, setModelRun] = useState<'suggest' | 'review' | null>(null);
  const [showWrite, setShowWrite] = useState(false);
  const [showFetch, setShowFetch] = useState(false);
  const [showSource, setShowSource] = useState(false);
  const [showGenerate, setShowGenerate] = useState(false);
  const [running, setRunning] = useState<{ cur: number; total: number } | null>(null);
  const [target, setTarget] = useState('');
  const [showKeys, setShowKeys] = useState(false);

  const classes = project.classes;
  const current = samples[index];
  const paare   = project.task === 'pairs';
  const saveTimer = useRef<number | null>(null);

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

  // Zieltext des aktuellen Samples — waehrend des Renderns, damit ein
  // schneller Tastendruck nicht auf dem Text des vorigen Samples landet.
  // Was gezeigt wird, haengt an Sample *und* Ladestand. Die ID allein reicht
  // nicht: nach einem Modelllauf oder Import kommt dasselbe Sample mit neuen
  // Boxen oder Labels zurueck. Wurde vorher nur zurueckgesetzt und dann
  // geladen, rendert React dazwischen mit den alten Daten, merkt sich die ID —
  // und uebernimmt die neuen nie. Ein Enter bestaetigte dann leere Boxen und
  // loeschte den Vorschlag. Der Zaehler steigt erst, wenn die Daten da sind.
  const [ladeStand, setLadeStand] = useState(0);
  const shownKey = current ? `${current.id}#${ladeStand}` : null;
  const [shownId, setShownId] = useState<string | null>(null);
  // Entwurf beim Bearbeiten des Textes selbst; null = es wird nicht bearbeitet.
  const [entwurf, setEntwurf] = useState<string | null>(null);
  if (current && shownId !== shownKey) {
    setShownId(shownKey);
    setTarget(current.ann.target ?? '');
    setEntwurf(null);
  }

  // ── Speichern ───────────────────────────────────────────────────────────
  const persist = useCallback(async (
    status: SampleStatus, sample: StudioSample, label: string | null, ziel: string | null,
  ) => {
    try {
      await invoke('studio_set_annotation', {
        projectId: project.id, sampleId: sample.id, boxes: [], status,
        label, target: ziel,
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

  useEffect(() => () => { if (saveTimer.current) window.clearTimeout(saveTimer.current); }, []);

  // ── Text bearbeiten ─────────────────────────────────────────────────────
  // Ein Tippfehler oder ein fast richtiges erzeugtes Beispiel soll nicht
  // geloescht und neu geschrieben werden muessen. Label und Status bleiben.
  const saveEdit = async () => {
    if (!current || entwurf === null) return;
    const neu = entwurf.trim();
    if (neu === (current.content ?? '').trim()) { setEntwurf(null); return; }
    try {
      await invoke('studio_edit_text', { projectId: project.id, sampleId: current.id, content: neu });
      setSamples(prev => prev.map(x => (x.id === current.id ? { ...x, content: neu } : x)));
      setEntwurf(null);
    } catch (err: unknown) {
      error(t('studio.edit.errorTitle'), String(err));
    }
  };

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

  /** Klasse zuweisen und weiter — der Griff, der bei 2000 Zeilen zaehlt. */
  const assign = async (label: string) => {
    if (!current) return;
    await persist('confirmed', current, label, null);
    advance('confirmed');
  };

  const confirmPair = async () => {
    if (!current) return;
    if (!target.trim()) {
      warning(t('studio.text.needsTargetTitle'), t('studio.text.needsTargetDetail'));
      return;
    }
    await persist('confirmed', current, null, target.trim());
    advance('confirmed');
  };

  const skip = async () => {
    if (!current) return;
    await persist('skipped', current, current.ann.label ?? null, current.ann.target ?? null);
    advance('skipped');
  };

  // ── Tastatur ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const imFeld = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
      if (!current || showExport || plan || modelRun || showWrite || showFetch || showSource || showGenerate || entwurf !== null) return;

      // Im Zieltext-Feld gilt nur Cmd+Enter, sonst tippt man Kuerzel in den Text.
      if (imFeld) {
        if (paare && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
          e.preventDefault(); void confirmPair();
        }
        return;
      }
      if (e.key >= '1' && e.key <= '9' && !paare) {
        const i = Number(e.key) - 1;
        if (i >= classes.length) return;
        e.preventDefault();
        void assign(classes[i]);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (paare) void confirmPair();
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
  }, [current, index, samples, classes, paare, target, showExport, plan, modelRun, showWrite, showFetch, showSource, showGenerate, entwurf]);

  // ── Import ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const un = listen<{ project_id: string; current: number; total: number; done?: boolean }>(
      'studio-import-progress', ev => {
        if (ev.payload.project_id !== project.id) return;
        setImporting(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      });
    return () => { void un.then(f => f()); };
  }, [project.id]);

  useEffect(() => {
    const abos = ['studio-suggest-progress', 'studio-review-progress'].map(name =>
      listen<{ project_id: string; current: number; total: number; done?: boolean }>(name, ev => {
        if (ev.payload.project_id !== project.id) return;
        setRunning(ev.payload.done ? null : { cur: ev.payload.current, total: ev.payload.total });
      }));
    return () => { abos.forEach(a => { void a.then(f => f()); }); };
  }, [project.id]);

  const pickSource = async (ordner: boolean) => {
    try {
      const sel = await open(ordner
        ? { directory: true, multiple: false, title: t('studio.text.pickFolder') }
        : { multiple: false, title: t('studio.text.pickFile'),
            filters: [{ name: 'Text', extensions: ['csv', 'tsv', 'jsonl', 'ndjson', 'txt'] }] });
      if (!sel || typeof sel !== 'string') return;
      const inspection = await invoke<TextInspection>('studio_inspect_text', { sourcePath: sel });
      setPlan({ path: sel, inspection });
    } catch (err: unknown) {
      error(t('studio.import.errorTitle'), String(err));
    }
  };

  const runImport = async (textColumn: string | null, labelColumn: string | null, ignore: boolean) => {
    const p = plan;
    setPlan(null);
    if (!p) return;
    try {
      setImporting({ cur: 0, total: 0 });
      const report = await invoke<ImportReport>('studio_import_text', {
        projectId: project.id, sourcePath: p.path,
        textColumn, labelColumn, ignoreLabels: ignore,
      });
      success(t('studio.text.doneTitle'), t('studio.text.doneDetail', {
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

  const addTexts = async (items: GeneratedItem[], origin: string | null) => {
    setShowWrite(false);
    setShowGenerate(false);
    try {
      const report = await invoke<ImportReport>('studio_add_texts', {
        projectId: project.id, items, origin,
      });
      success(t('studio.write.doneTitle'),
        t('studio.write.doneDetail', { added: report.added, duplicates: report.duplicates }));
      const list = await invoke<StudioProject[]>('studio_list_projects');
      const fresh = list.find(x => x.id === project.id);
      if (fresh) onProjectChanged(fresh);
      await loadSamples(0, true);
      setLadeStand(n => n + 1);
      await loadStats();
    } catch (err: unknown) {
      error(t('studio.write.errorTitle'), String(err));
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
          <button onClick={() => setShowSource(true)} disabled={!!importing}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-200 text-sm transition-all inline-flex items-center gap-2 disabled:opacity-50">
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
            {importing && importing.total > 0
              ? t('studio.import.progress', { current: importing.cur, total: importing.total })
              : t('studio.text.addButton')}
          </button>
          {!paare && (
            <>
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
                <ShieldQuestion className="w-4 h-4" /> {t('studio.workbench.reviewButton')}
              </button>
            </>
          )}
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
          <FileText className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{t('studio.text.emptyTitle')}</p>
          <p className="text-gray-500 text-sm mt-1 mb-5 max-w-md mx-auto">{t('studio.text.emptyDetail')}</p>
          <button onClick={() => setShowSource(true)}
            className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center gap-2">
            <FileText className="w-4 h-4" /> {t('studio.text.addButton')}
          </button>
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
                  {/* Der Text selbst, nicht nur die Klasse — bei 62 Zeilen stand
                      sonst 62-mal "Request" untereinander. Die Klasse steht als
                      Farbe daneben. */}
                  <span className="text-gray-300 text-xs truncate flex-1 min-w-0">
                    {(s.content ?? '').replace(/\s+/g, ' ').trim() || t('studio.text.noLabel')}
                  </span>
                  {s.ann.label && (
                    <span className="w-2 h-2 rounded-sm flex-shrink-0"
                      title={s.ann.label}
                      style={{ background: classColor(s.ann.label, classes) }} />
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

          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5 flex flex-col gap-4">
            {current && (
              <>
                {entwurf !== null ? (
                  <div className="space-y-2">
                    <textarea value={entwurf} onChange={e => setEntwurf(e.target.value)} rows={5} autoFocus
                      aria-label={t('studio.edit.fieldLabel')}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void saveEdit(); }
                        if (e.key === 'Escape') { e.preventDefault(); setEntwurf(null); }
                      }}
                      className="w-full px-4 py-3 rounded-lg bg-black/30 border border-white/25 text-gray-100 text-sm leading-relaxed focus:outline-none resize-y" />
                    <div className="flex items-center gap-2">
                      <button onClick={() => void saveEdit()} disabled={!entwurf.trim()}
                        className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/15 text-white text-xs inline-flex items-center gap-1.5 disabled:opacity-40">
                        <Check className="w-3.5 h-3.5" /> {t('studio.edit.save')}
                      </button>
                      <button onClick={() => setEntwurf(null)}
                        className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs">
                        {t('common.cancel', 'Abbrechen')}
                      </button>
                      <span className="text-gray-600 text-[11px]">{t('studio.edit.hint')}</span>
                    </div>
                  </div>
                ) : (
                  <div className="group relative rounded-lg bg-black/30 border border-white/10 p-4 pr-12 max-h-64 overflow-y-auto">
                    <p className="text-gray-100 text-sm whitespace-pre-wrap leading-relaxed">
                      {current.content}
                    </p>
                    <button onClick={() => setEntwurf(current.content ?? '')}
                      title={t('studio.edit.button')} aria-label={t('studio.edit.button')}
                      className="absolute top-2 right-2 p-1.5 rounded-md bg-white/5 hover:bg-white/15 border border-white/10 text-gray-400 hover:text-white transition-all">
                      <PenLine className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}

                {paare ? (
                  <label className="block">
                    <span className="text-gray-400 text-xs">{t('studio.text.targetLabel')}</span>
                    <textarea value={target} onChange={e => setTarget(e.target.value)} rows={4}
                      placeholder={t('studio.text.targetPlaceholder')}
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
                        className={`px-3 py-2 rounded-lg border text-sm inline-flex items-center gap-2 transition-all ${current.ann.label === name
                          ? (current.status === 'suggested'
                            ? 'bg-amber-500/15 border-amber-500/40 text-amber-100'
                            : 'bg-white/15 border-white/25 text-white')
                          : 'bg-white/5 border-white/10 text-gray-300 hover:bg-white/10'}`}>
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

                {current.status === 'suggested' && (
                  <p className="text-amber-300/90 text-[11px] inline-flex items-center gap-1.5">
                    <Wand2 className="w-3.5 h-3.5 flex-shrink-0" /> {t('studio.text.suggestedHint')}
                  </p>
                )}
                {current.doubt && (
                  <p className="text-orange-200/90 text-[11px] inline-flex items-center gap-1.5">
                    <ShieldQuestion className="w-3.5 h-3.5 flex-shrink-0" />
                    {t('studio.text.doubtHint', {
                      model: current.doubt.missing.join(', '),
                      label: current.doubt.extra.join(', '),
                    })}
                  </p>
                )}
                <div className="flex items-center gap-3 text-[11px] text-gray-500">
                  <span className="tabular-nums">{index + 1} / {total}</span>
                  <span className="truncate max-w-xs">{current.src.origin?.split(/[\\/]/).pop()}</span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {paare && (
                    <button onClick={() => void confirmPair()}
                      className="px-4 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-200 text-sm whitespace-nowrap inline-flex items-center gap-2">
                      <Check className="w-4 h-4" /> {t('studio.workbench.confirm')}
                    </button>
                  )}
                  <button onClick={() => void skip()}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm whitespace-nowrap inline-flex items-center gap-2">
                    <SkipForward className="w-4 h-4" /> {t('studio.workbench.skip')}
                  </button>
                  <span className="ml-auto" />
                  <RemoveSampleButton key={current.id} onRemove={() => void removeCurrent()} />
                </div>
              </>
            )}
          </div>

          <div className="space-y-3 pb-20">
            {!paare && (
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
                  <p>{t(paare ? 'studio.text.shortcutPair' : 'studio.shortcuts.classes')}</p>
                  <p>{t('studio.shortcuts.skip')}</p>
                  <p>{t('studio.shortcuts.back')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {plan && (
        <TextImportDialog
          inspection={plan.inspection}
          paare={paare}
          onCancel={() => setPlan(null)}
          onRun={(tc, lc, ig) => void runImport(tc, lc, ig)}
        />
      )}

      {showSource && (
        <TextSourceDialog
          onClose={() => setShowSource(false)}
          onFile={() => { setShowSource(false); void pickSource(false); }}
          onFolder={() => { setShowSource(false); void pickSource(true); }}
          onWeb={() => { setShowSource(false); setShowFetch(true); }}
          onWrite={() => { setShowSource(false); setShowWrite(true); }}
          onGenerate={() => { setShowSource(false); setShowGenerate(true); }}
        />
      )}

      {showFetch && (
        <FetchDialog
          project={project}
          onClose={() => setShowFetch(false)}
          onDone={async () => {
            setShowFetch(false);
            await loadSamples(0, true);
            setLadeStand(n => n + 1);
            await loadStats();
          }}
        />
      )}

      {showWrite && (
        <WriteDialog
          classes={classes}
          paare={paare}
          onCancel={() => setShowWrite(false)}
          onCreate={(texte, label, ziel) => void addTexts(
            texte.map(text => ({ text, label, target: ziel })), null)}
        />
      )}

      {showGenerate && (
        <GenerateDialog
          project={project}
          paare={paare}
          vorhanden={samples.filter(x => x.content).map(x => ({
            text: x.content ?? '', label: x.ann.label, target: x.ann.target,
            bestaetigt: x.status === 'confirmed',
          }))}
          onCancel={() => setShowGenerate(false)}
          onAdd={(items, origin) => void addTexts(items, origin)}
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
            const fresh = list.find(x => x.id === project.id);
            if (fresh) onProjectChanged(fresh);
            await loadSamples(0, true);
            setLadeStand(n => n + 1);
            await loadStats();
          }}
        />
      )}

      {showExport && (
        <TextExportDialog
          project={project}
          confirmed={confirmed}
          onClose={() => setShowExport(false)}
          onDone={() => { setShowExport(false); void loadStats(); }}
        />
      )}
    </div>
  );
}

// ── Import ────────────────────────────────────────────────────────────────

function TextImportDialog({ inspection, paare, onCancel, onRun }: {
  inspection: TextInspection; paare: boolean;
  onCancel: () => void;
  onRun: (textColumn: string | null, labelColumn: string | null, ignore: boolean) => void;
}) {
  const { t } = useLanguage();
  const [textColumn, setTextColumn] = useState(inspection.text_column ?? '');
  const [labelColumn, setLabelColumn] = useState(inspection.label_column ?? '');
  const [ignore, setIgnore] = useState(false);
  const hatSpalten = inspection.columns.length > 0;

  useEscape(onCancel);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onCancel}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.text.importTitle')}</h3>
          <p className="text-gray-400 text-xs mt-1">
            {t('studio.text.importSummary', { rows: inspection.rows, kind: inspection.kind })}
          </p>
        </div>

        {inspection.preview.length > 0 && (
          <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1">
            {inspection.preview.map((p, i) => (
              <p key={i} className="text-gray-400 text-[11px] truncate">{p}</p>
            ))}
          </div>
        )}

        {hatSpalten && (
          <>
            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.text.textColumn')}</span>
              <select value={textColumn} onChange={e => setTextColumn(e.target.value)}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
                {inspection.columns.map(c => <option key={c} value={c} className="bg-[#101218]">{c}</option>)}
              </select>
            </label>

            <label className="block">
              <span className="text-gray-400 text-xs">
                {t(paare ? 'studio.text.targetColumn' : 'studio.text.labelColumn')}
              </span>
              <select value={labelColumn} onChange={e => setLabelColumn(e.target.value)}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
                <option value="" className="bg-[#101218]">{t('studio.text.noColumn')}</option>
                {inspection.columns.map(c => <option key={c} value={c} className="bg-[#101218]">{c}</option>)}
              </select>
            </label>
          </>
        )}

        {inspection.labels.length > 0 && (
          <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3">
            <p className="text-gray-400 text-xs">
              {t('studio.text.foundLabels', { count: inspection.labels.length })}
            </p>
            <p className="text-gray-500 text-[11px] mt-1">{inspection.labels.slice(0, 12).join(', ')}</p>
            <label className="flex items-start gap-2 cursor-pointer mt-2">
              <input type="checkbox" checked={ignore} onChange={e => setIgnore(e.target.checked)} className="mt-0.5" />
              <span className="text-gray-400 text-[11px]">{t('studio.text.ignoreLabels')}</span>
            </label>
          </div>
        )}

        <p className="text-gray-500 text-xs flex items-start gap-1.5">
          <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          {t('studio.text.duplicateNote')}
        </p>

        <div className="flex gap-2">
          <button onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
            {t('common.cancel', 'Abbrechen')}
          </button>
          <button onClick={() => onRun(textColumn || null, labelColumn || null, ignore)}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2">
            <FileText className="w-4 h-4" /> {t('studio.import.startButton')}
          </button>
        </div>
      </div>
    </div></ModalPortal>
  );
}

// ── Export ────────────────────────────────────────────────────────────────

function TextExportDialog({ project, confirmed, onClose, onDone }: {
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

  useEscape(onClose, !busy);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.export.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">
            {t(project.task === 'pairs' ? 'studio.text.exportPairs' : 'studio.text.exportClasses')}
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

        <p className="text-gray-500 text-xs">{t('studio.export.summaryText', { confirmed })}</p>

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

// ── Selbst schreiben ──────────────────────────────────────────────────────

function WriteDialog({ classes, paare, onCancel, onCreate }: {
  classes: string[];
  paare: boolean;
  onCancel: () => void;
  onCreate: (texte: string[], label: string | null, ziel: string | null) => void;
}) {
  const { t } = useLanguage();
  const [text, setText] = useState('');
  const [label, setLabel] = useState<string | null>(null);
  const [ziel, setZiel] = useState('');
  // Eine eingefuegte Liste ist der Normalfall; ein langer Absatz die Ausnahme.
  const [proZeile, setProZeile] = useState(true);

  const zeilen = proZeile
    ? text.split('\n').map(z => z.trim()).filter(Boolean)
    : (text.trim() ? [text.trim()] : []);
  const bereit = zeilen.length > 0 && (!paare || ziel.trim().length > 0);

  useEscape(onCancel);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onCancel}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4 max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.write.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.write.subtitle')}</p>
        </div>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.write.textLabel')}</span>
          <textarea value={text} onChange={e => setText(e.target.value)} rows={7} autoFocus
            placeholder={t('studio.write.textPlaceholder')}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none font-mono" />
        </label>

        <label className="flex items-start gap-2 cursor-pointer">
          <input type="checkbox" checked={proZeile} onChange={e => setProZeile(e.target.checked)} className="mt-0.5" />
          <span className="text-gray-300 text-xs">{t('studio.write.perLine')}</span>
        </label>

        {paare ? (
          <label className="block">
            <span className="text-gray-400 text-xs">{t('studio.text.targetLabel')}</span>
            <textarea value={ziel} onChange={e => setZiel(e.target.value)} rows={3}
              placeholder={t('studio.text.targetPlaceholder')}
              className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none" />
          </label>
        ) : (
          <div>
            <span className="text-gray-400 text-xs">{t('studio.write.classLabel')}</span>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              <button onClick={() => setLabel(null)}
                className={`px-2.5 py-1.5 rounded-lg border text-xs transition-all ${label === null ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}>
                {t('studio.write.noClass')}
              </button>
              {classes.map(name => (
                <button key={name} onClick={() => setLabel(name)}
                  className={`px-2.5 py-1.5 rounded-lg border text-xs inline-flex items-center gap-1.5 transition-all ${label === name ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}>
                  <span className="w-2 h-2 rounded-sm flex-shrink-0"
                    style={{ background: classColor(name, classes) }} />
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        <p className="text-gray-500 text-xs">
          {zeilen.length > 0
            ? t('studio.write.count', { count: zeilen.length })
            : t('studio.write.empty')}
        </p>

        <div className="flex gap-2">
          <button onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
            {t('common.cancel', 'Abbrechen')}
          </button>
          <button onClick={() => onCreate(zeilen, label, paare ? ziel : null)} disabled={!bereit}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
            <PenLine className="w-4 h-4" /> {t('studio.write.confirmButton')}
          </button>
        </div>
      </div>
    </div></ModalPortal>
  );
}

// ── Woher kommen die Texte ────────────────────────────────────────────────

function TextSourceDialog({ onClose, onFile, onFolder, onWeb, onWrite, onGenerate }: {
  onClose: () => void;
  onFile: () => void;
  onFolder: () => void;
  onWeb: () => void;
  onWrite: () => void;
  onGenerate: () => void;
}) {
  const { t } = useLanguage();

  const tile = (icon: React.ReactNode, title: string, hint: string, onClick: () => void) => (
    <button onClick={onClick}
      className="w-full flex items-start gap-3 p-3 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-left transition-all">
      <span className="p-2 rounded-lg bg-white/5 border border-white/10 text-gray-300 flex-shrink-0">{icon}</span>
      <span className="min-w-0">
        <span className="text-white text-sm block">{title}</span>
        <span className="text-gray-500 text-xs block mt-0.5">{hint}</span>
      </span>
    </button>
  );

  useEscape(onClose);

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.text.sourceTitle')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.source.subtitle')}</p>
        </div>

        <div className="space-y-2">
          {tile(<FileText className="w-4 h-4" />, t('studio.text.sourceFile'), t('studio.text.sourceFileHint'), onFile)}
          {tile(<FolderOpen className="w-4 h-4" />, t('studio.text.sourceFolder'), t('studio.text.sourceFolderHint'), onFolder)}
          {tile(<Globe className="w-4 h-4" />, t('studio.source.web'), t('studio.source.webHint'), onWeb)}
          {tile(<PenLine className="w-4 h-4" />, t('studio.text.sourceWrite'), t('studio.text.sourceWriteHint'), onWrite)}
          {tile(<Sparkles className="w-4 h-4" />, t('studio.text.sourceGenerate'), t('studio.text.sourceGenerateHint'), onGenerate)}
        </div>

        <button onClick={onClose}
          className="w-full py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
          {t('common.cancel', 'Abbrechen')}
        </button>
      </div>
    </div></ModalPortal>
  );
}
