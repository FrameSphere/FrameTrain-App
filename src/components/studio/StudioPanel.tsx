// Dataset Studio: Projektliste und Einstieg in die Werkbank.
//
// Ein Projekt ist ein Arbeitsstand, kein fertiger Datensatz. Fertig wird er
// erst beim Export — der legt ein ganz normales FrameTrain-Dataset an, das der
// Dataset-Bereich danach wie jedes andere behandelt.

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  ArrowLeft, Plus, Loader2, Trash2, Boxes, Image as ImageIcon, X, FileText,
} from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import { navigateTo } from '../../ui/navigationEvents';
import { dateLocale } from '../../utils/dateLocale';
import ImageWorkbench from './ImageWorkbench';
import TextWorkbench from './TextWorkbench';
import type { StudioProject } from './studioTypes';

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

  if (openProject) {
    // Die Werkbank richtet sich nach der Modalitaet des Projekts. Fehlt diese
    // Weiche, landet auch ein Textprojekt im Bild-Editor und meldet "Noch
    // keine Bilder" — ohne dass irgendetwas fehlschlaegt.
    const Werkbank = openProject.modality === 'text' ? TextWorkbench : ImageWorkbench;
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
              className="group rounded-2xl border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] transition-all overflow-hidden">
              <button onClick={() => setOpenId(p.id)} className="w-full text-left p-5">
                <div className="flex items-start gap-3">
                  <span className="p-2 rounded-lg bg-white/5 border border-white/10">
                    {p.modality === 'text'
                      ? <FileText className="w-4 h-4 text-gray-300" />
                      : <ImageIcon className="w-4 h-4 text-gray-300" />}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-white font-medium truncate">{p.name}</p>
                    <p className="text-gray-500 text-xs mt-0.5">
                      {t(p.modality === 'text' ? 'studio.projects.cardMetaText' : 'studio.projects.cardMeta', {
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
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
        </div>
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
  const [kind, setKind] = useState<'image' | 'text' | 'pairs'>('image');
  const [busy, setBusy] = useState(false);

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const project = await invoke<StudioProject>('studio_create_project', {
        name: name.trim(),
        modality: kind === 'image' ? 'image' : 'text',
        task: kind === 'image' ? 'bbox' : kind === 'pairs' ? 'pairs' : 'classification',
        targetFormat: kind === 'image' ? 'yolo_bbox' : 'flat_file',
        classes: kind === 'pairs' ? [] : classText.split(/[,\n]/).map(c => c.trim()).filter(Boolean),
      });
      onCreated(project);
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
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
              ['text', t('studio.create.kindText')],
              ['pairs', t('studio.create.kindPairs')],
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

        {kind !== 'pairs' && <label className="block">
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
    </div>
  );
}
