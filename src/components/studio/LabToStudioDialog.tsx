// Labor-Ergebnisse in ein Werkstatt-Projekt uebernehmen (aktives Lernen).

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, Boxes } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import ModalPortal from '../ui/ModalPortal';
import { useEscape } from './useEscape';
import type { StudioProject } from './studioTypes';
import { labAuswahl, labItems, labModalitaet, type LabAuswahl, type LabErgebnis } from './labToStudio';

interface Bericht { added: number; confirmed: number; suggested: number; duplicates: number; skipped: number; }

export default function LabToStudioDialog({ name, ergebnisse, bboxModell, onClose }: {
  name: string; ergebnisse: LabErgebnis[];
  /** Objekterkennung im Labor -> Boxenprojekt statt Klassen. */
  bboxModell: boolean;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const { success, error } = useNotification();
  const modalitaet = labModalitaet(ergebnisse);
  const [projekte, setProjekte] = useState<StudioProject[]>([]);
  const [ziel, setZiel] = useState<string>('new');
  const [neuName, setNeuName] = useState(`${name} – Nacharbeit`);
  const [wie, setWie] = useState<LabAuswahl>('wrong');
  const [busy, setBusy] = useState(false);
  const [bericht, setBericht] = useState<Bericht | null>(null);
  const auswahl = labAuswahl(ergebnisse, wie);
  const task = modalitaet === 'image' && bboxModell ? 'bbox'
    : ergebnisse.some(r => r.correction?.kind === 'text') ? (modalitaet === 'audio' ? 'transcript' : 'pairs')
      : modalitaet === 'text' ? 'classification' : 'classify';

  useEffect(() => {
    invoke<StudioProject[]>('studio_list_projects')
      .then(list => setProjekte(list.filter(p => p.modality === modalitaet)))
      .catch(() => setProjekte([]));
  }, [modalitaet]);

  useEscape(onClose, !busy);

  const run = async () => {
    if (auswahl.length === 0) return;
    setBusy(true);
    try {
      let projectId = ziel;
      if (ziel === 'new') {
        const p = await invoke<StudioProject>('studio_create_project', {
          name: neuName.trim() || name, modality: modalitaet,
          task: modalitaet === 'audio' && task === 'classify' ? 'classification' : task,
          targetFormat: task === 'bbox' ? 'yolo_bbox' : modalitaet === 'text' ? 'flat_file'
            : task === 'transcript' ? 'audio_transcript' : 'folder_class',
          classes: [],
        });
        projectId = p.id;
      }
      const r = await invoke<Bericht>('studio_add_from_lab', { projectId, items: labItems(auswahl) });
      setBericht(r);
      success(t('studio.fromLab.doneTitle'), t('studio.fromLab.doneDetail', { added: r.added }));
    } catch (err: unknown) {
      error(t('studio.fromLab.errorTitle'), String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <ModalPortal><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6"
      onClick={busy ? undefined : onClose}>
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#101218] p-6 space-y-4" onClick={e => e.stopPropagation()}>
        <div>
          <h3 className="text-white font-semibold">{t('studio.fromLab.title')}</h3>
          <p className="text-gray-500 text-xs mt-1">{t('studio.fromLab.subtitle')}</p>
        </div>

        {bericht ? (
          <>
            <div className="rounded-lg bg-white/[0.04] border border-white/10 p-3 space-y-1.5">
              {(['added', 'confirmed', 'suggested', 'duplicates', 'skipped'] as const).map(k => (
                <div key={k} className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">{t(`studio.fromLab.report.${k}`)}</span>
                  <span className="text-gray-200 tabular-nums">{bericht[k]}</span>
                </div>
              ))}
            </div>
            <p className="text-gray-500 text-xs">{t('studio.fromLab.next')}</p>
            <button onClick={onClose}
              className="w-full py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm">
              {t('studio.suggest.report.close')}
            </button>
          </>
        ) : (
          <>
            <div>
              <span className="text-gray-400 text-xs">{t('studio.fromLab.whichLabel')}</span>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {(['wrong', 'uncertain', 'all'] as const).map(w => (
                  <button key={w} onClick={() => setWie(w)}
                    className={`px-2.5 py-1.5 rounded-lg border text-xs transition-all ${wie === w ? 'bg-white/15 border-white/25 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}>
                    {t(`studio.fromLab.which.${w}`)} ({labAuswahl(ergebnisse, w).length})
                  </button>
                ))}
              </div>
            </div>

            <label className="block">
              <span className="text-gray-400 text-xs">{t('studio.fromLab.projectLabel')}</span>
              <select value={ziel} onChange={e => setZiel(e.target.value)}
                className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
                <option value="new" className="bg-[#101218]">{t('studio.fromLab.newProject')}</option>
                {projekte.map(p => <option key={p.id} value={p.id} className="bg-[#101218]">{p.name}</option>)}
              </select>
            </label>
            {ziel === 'new' && (
              <input value={neuName} onChange={e => setNeuName(e.target.value)} aria-label={t('studio.create.nameLabel')}
                className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25" />
            )}

            <p className="text-gray-500 text-xs">{t('studio.fromLab.rule')}</p>

            <div className="flex gap-2">
              <button onClick={onClose}
                className="flex-1 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm">
                {t('common.cancel', 'Abbrechen')}
              </button>
              <button onClick={() => void run()} disabled={busy || auswahl.length === 0}
                className="flex-1 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 text-white text-sm inline-flex items-center justify-center gap-2 disabled:opacity-40">
                {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Boxes className="w-4 h-4" />}
                {t('studio.fromLab.runButton', { count: auswahl.length })}
              </button>
            </div>
          </>
        )}
      </div>
    </div></ModalPortal>
  );
}
