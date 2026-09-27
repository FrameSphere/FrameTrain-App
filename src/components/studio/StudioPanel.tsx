// Dataset Studio: Projektliste und Einstieg in die Werkbank.
//
// Ein Projekt ist ein Arbeitsstand, kein fertiger Datensatz. Fertig wird er
// erst beim Export — der legt ein ganz normales FrameTrain-Dataset an, das der
// Dataset-Bereich danach wie jedes andere behandelt.

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  ArrowLeft, Plus, Loader2, Trash2, Boxes, Image as ImageIcon, X, FileText, AudioLines, FolderOpen,
  Film, LayoutGrid,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { navigateTo } from '../../ui/navigationEvents';
import { dateLocale } from '../../utils/dateLocale';
import ImageWorkbench from './ImageWorkbench';
import TextWorkbench from './TextWorkbench';
import MediaWorkbench from './MediaWorkbench';
import type { StudioProject } from './studioTypes';
import ModalPortal from '../ui/ModalPortal';
import { useContextMenuActions, type ContextMenuAction } from '../../ui/contextMenuRegistry';
import { useEscape } from './useEscape';

/// Was auf der Projektkarte steht, haengt an Modalitaet und Aufgabe.
///
/// Frueher gab es nur "Text oder sonst Bild" — ein Audioprojekt zaehlte dann
/// "2 Bilder" und trug ein Bildsymbol. Paare und Transkripte haben ausserdem
/// keine Klassen; "0 Klassen" dort wuerde einen Fehler vermuten lassen, wo keiner ist.
export function cardMetaKey(p: Pick<StudioProject, 'modality' | 'task'>): string {
  if (p.modality === 'audio') {
    return p.task === 'transcript' ? 'studio.projects.cardMetaTranscript' : 'studio.projects.cardMetaAudio';
  }
  if (p.modality === 'text') {
    return p.task === 'pairs' ? 'studio.projects.cardMetaPairs' : 'studio.projects.cardMetaText';
  }
  if (p.modality === 'video') return 'studio.projects.cardMetaVideo';
  return 'studio.projects.cardMeta';
}

function ProjectIcon({ modality, task }: { modality: string; task?: string }) {
  const cls = 'w-4 h-4 text-gray-300';
  if (modality === 'audio') return <AudioLines className={cls} aria-label="audio" />;
  if (modality === 'text') return <FileText className={cls} aria-label="text" />;
  if (modality === 'video') return <Film className={cls} aria-label="video" />;
  if (task === 'classify') return <LayoutGrid className={cls} aria-label="image-classes" />;
  return <ImageIcon className={cls} aria-label="image" />;
}

/// Welche Werkbank ein Projekt oeffnet. Boxen haben ihre eigene; alles, was
/// je Sample eine Klasse oder einen Wortlaut bekommt, teilt sich die Medien-Werkbank.
export function werkbankFuer(p: Pick<StudioProject, 'modality' | 'task'>): 'text' | 'media' | 'boxes' {
  if (p.modality === 'text') return 'text';
  if (p.modality === 'audio' || p.modality === 'video') return 'media';
  return p.task === 'classify' ? 'media' : 'boxes';
}

/// Projektarten im Anlegen-Dialog und was daraus wird.
export const PROJEKTARTEN = {
  image:         { modality: 'image', task: 'bbox',           targetFormat: 'yolo_bbox',        classes: true },
  imageClassify: { modality: 'image', task: 'classify',       targetFormat: 'folder_class',     classes: true },
  video:         { modality: 'video', task: 'classify',       targetFormat: 'folder_class',     classes: true },
  text:          { modality: 'text',  task: 'classification', targetFormat: 'flat_file',        classes: true },
  pairs:         { modality: 'text',  task: 'pairs',          targetFormat: 'flat_file',        classes: false },
  audio:         { modality: 'audio', task: 'classification', targetFormat: 'folder_class',     classes: true },
  transcript:    { modality: 'audio', task: 'transcript',     targetFormat: 'audio_transcript', classes: false },
} as const;
export type Projektart = keyof typeof PROJEKTARTEN;

export default function StudioPanel() {
  const { t, language } = useLanguage();
  const { success, error } = useNotification();

  const [projects, setProjects] = useState<StudioProject[]>([]);
  const [loading, setLoading]   = useState(true);
  const [openId, setOpenId]     = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<StudioProject | null>(null);

  const load = useCallback(async () => {
    try {
      setProjects(await invoke<StudioProject[]>('studio_list_projects'));
    } catch (err: unknown) {
      error(t('studio.notifications.loadError'), String(err));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t]);

  useEffect(() => { void load(); }, [load]);

  const openProject = projects.find(p => p.id === openId) ?? null;
  const listeAktiv = !openProject && !showCreate && !confirmDelete;

  // ⌘N legt ein neues Projekt an — solange die Liste zu sehen ist. In einer
  // Werkbank gehoert ⌘N der Werkbank (Texte schreiben).
  useEffect(() => {
    if (!listeAktiv) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') { e.preventDefault(); setShowCreate(true); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [listeAktiv]);

  // Rechtsklick: auf einer Karte das Projekt, sonst die Liste.
  useContextMenuActions(({ target }) => {
    if (!listeAktiv) return [];
    const aktionen: ContextMenuAction[] = [];
    const id = target?.closest('[data-studio-project]')?.getAttribute('data-studio-project');
    const p = id ? projects.find(x => x.id === id) : undefined;
    if (p) {
      aktionen.push(
        { id: 'st-list-open', group: p.name, label: t('studio.menu.openProject'), icon: FolderOpen,
          onSelect: () => setOpenId(p.id) },
        { id: 'st-list-delete', group: p.name, label: t('studio.menu.deleteProject'), icon: Trash2, danger: true,
          onSelect: () => setConfirmDelete(p) },
      );
    }
    aktionen.push({ id: 'st-list-new', group: t('studio.title'), label: t('studio.projects.newButton'), icon: Plus,
      shortcut: '⌘N', onSelect: () => setShowCreate(true) });
    return aktionen;
  });

  if (openProject) {
    // Die Werkbank richtet sich nach der Modalitaet des Projekts. Fehlt diese
    // Weiche, landet auch ein Textprojekt im Bild-Editor und meldet "Noch
    // keine Bilder" — ohne dass irgendetwas fehlschlaegt.
    const art = werkbankFuer(openProject);
    const Werkbank = art === 'text' ? TextWorkbench : art === 'media' ? MediaWorkbench : ImageWorkbench;
    return (
      <Werkbank
        project={openProject}
        onBack={() => { setOpenId(null); void load(); }}
        // Zusammenfuehren statt ersetzen: studio_update_project liefert das
        // Projekt ohne die live gezaehlten Staende, ein Ersetzen wuerde die
        // Karte auf 0 zuruecksetzen.
        onProjectChanged={p => setProjects(prev => prev.map(x => (x.id === p.id ? { ...x, ...p } : x)))}
      />
    );
  }

  const handleDelete = async (p: StudioProject) => {
    try {
      await invoke('studio_delete_project', { projectId: p.id });
      setConfirmDelete(null);
      success(t('studio.projects.deletedTitle'), t('studio.projects.deletedDetail', { name: p.name }));
      await load();
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigateTo('dataset')}
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 transition-all"
          aria-label={t('studio.projects.backToDatasets')}>
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">{t('studio.title')}</h1>
          <p className="text-gray-500 text-sm">{t('studio.subtitle')}</p>
        </div>
        <button onClick={() => setShowCreate(true)}
          className="ml-auto px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center gap-2 transition-all">
          <Plus className="w-4 h-4" /> {t('studio.projects.newButton')}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20 text-gray-500 gap-2">
          <Loader2 className="w-5 h-5 animate-spin" /> {t('common.loading', 'Lädt…')}
        </div>
      ) : projects.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-12 text-center">
          <Boxes className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-white font-medium">{t('studio.projects.emptyTitle')}</p>
          <p className="text-gray-500 text-sm mt-1 mb-5 max-w-md mx-auto">{t('studio.projects.emptyDetail')}</p>
          <button onClick={() => setShowCreate(true)}
            className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center gap-2">
            <Plus className="w-4 h-4" /> {t('studio.projects.newButton')}
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {projects.map(p => (
            <div key={p.id}
              data-studio-project={p.id}
              className="group rounded-2xl border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] transition-all overflow-hidden">
              <button onClick={() => setOpenId(p.id)} className="w-full text-left p-5">
                <div className="flex items-start gap-3">
                  <span className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <ProjectIcon modality={p.modality} task={p.task} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-white font-medium truncate">{p.name}</p>
                    <p className="text-gray-500 text-xs mt-0.5">
                      {t(cardMetaKey(p), {
                        samples: p.sample_count ?? 0,
                        confirmed: p.confirmed_count ?? 0,
                        classes: p.classes.length,
                      })}
                    </p>
                  </div>
                </div>
                <div className="mt-4 h-1 rounded-full bg-white/5 overflow-hidden">
                  <div className="h-full bg-emerald-400/70 transition-all"
                    style={{ width: `${p.sample_count ? Math.round(((p.confirmed_count ?? 0) / p.sample_count) * 100) : 0}%` }} />
                </div>
                <p className="text-gray-600 text-[11px] mt-2">
                  {t('studio.projects.updated', {
                    date: new Date(p.updated_at).toLocaleDateString(dateLocale(language)),
                  })}
                </p>
              </button>
              <div className="px-5 pb-4 -mt-2 flex justify-end">
                <button onClick={() => setConfirmDelete(p)}
                  className="p-1.5 rounded-lg text-gray-600 hover:text-red-300 hover:bg-red-500/10 transition-all opacity-0 group-hover:opacity-100"
                  aria-label={t('studio.projects.delete')}>
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreate && (
        <CreateDialog
          onClose={() => setShowCreate(false)}
          onCreated={p => { setShowCreate(false); setProjects(prev => [p, ...prev]); setOpenId(p.id); }}
        />
      )}

      {confirmDelete && (
        <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
          onClick={() => setConfirmDelete(null)}>
          <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4"
            onClick={e => e.stopPropagation()}>
            <h3 className="text-white font-semibold">{t('studio.projects.deleteTitle')}</h3>
            <p className="text-gray-400 text-sm">
              {t('studio.projects.deleteDetail', { name: confirmDelete.name })}
            </p>
            <div className="flex gap-2">
              <button onClick={() => setConfirmDelete(null)}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void handleDelete(confirmDelete)}
                className="flex-1 py-2.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 text-sm">
                {t('studio.projects.delete')}
              </button>
            </div>
          </div>
        </div></ModalPortal>
      )}
    </div>
  );
}

function CreateDialog({ onClose, onCreated }: {
  onClose: () => void; onCreated: (p: StudioProject) => void;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const [name, setName] = useState('');
  const [classText, setClassText] = useState('');
  const [kind, setKind] = useState<Projektart>('image');
  const art = PROJEKTARTEN[kind];
  const [busy, setBusy] = useState(false);

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const project = await invoke<StudioProject>('studio_create_project', {
        name: name.trim(),
        modality: art.modality,
        task: art.task,
        targetFormat: art.targetFormat,
        classes: art.classes ? classText.split(/[,\n]/).map(c => c.trim()).filter(Boolean) : [],
      });
      onCreated(project);
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
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
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-white font-semibold">{t('studio.create.title')}</h3>
            <p className="text-gray-500 text-xs mt-1">{t(`studio.create.kindHint.${kind}`)}</p>
          </div>
          <button onClick={onClose} className="p-1 text-gray-500 hover:text-white" aria-label={t('common.cancel', 'Abbrechen')}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div>
          <span className="text-gray-400 text-xs">{t('studio.create.kindLabel')}</span>
          <div className="mt-1.5 grid grid-cols-3 gap-1.5">
            {([
              ['image', t('studio.create.kindImage')],
              ['imageClassify', t('studio.create.kindImageClassify')],
              ['video', t('studio.create.kindVideo')],
              ['text', t('studio.create.kindText')],
              ['pairs', t('studio.create.kindPairs')],
              ['audio', t('studio.create.kindAudio')],
              ['transcript', t('studio.create.kindTranscript')],
            ] as const).map(([val, label]) => (
              <button key={val} onClick={() => setKind(val)}
                className={`px-2 py-2 rounded-lg border text-xs transition-all ${kind === val ? 'bg-white/10 border-white/25 text-white' : 'bg-white/[0.03] border-white/10 text-gray-400 hover:bg-white/[0.06]'}`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.create.nameLabel')}</span>
          <input value={name} onChange={e => setName(e.target.value)} autoFocus
            placeholder={t('studio.create.namePlaceholder')}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25" />
        </label>

        {art.classes && <label className="block">
          <span className="text-gray-400 text-xs">{t('studio.create.classesLabel')}</span>
          <textarea value={classText} onChange={e => setClassText(e.target.value)} rows={3}
            placeholder={t('studio.create.classesPlaceholder')}
            className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm placeholder-gray-600 focus:outline-none focus:border-white/25 resize-none" />
          <span className="text-gray-600 text-[11px]">{t('studio.create.classesHint')}</span>
        </label>}

        <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3">
          <p className="text-gray-400 text-xs">
            {t(kind === 'image' ? 'studio.create.formatNote' : `studio.create.formatNote_${kind}`)}
          </p>
        </div>

        <div className="flex gap-2">
          <button onClick={onClose}
            className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
            {t('common.cancel', 'Abbrechen')}
          </button>
          <button onClick={() => void create()} disabled={busy || !name.trim()}
            className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
            {busy && <Loader2 className="w-4 h-4 animate-spin" />}
            {t('studio.create.confirmButton')}
          </button>
        </div>
      </div>
    </div></ModalPortal>
  );
}
