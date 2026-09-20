// Modell-Lauf ueber ein Projekt: vorschlagen oder pruefen.
//
// Ein Dialog fuer beides und fuer jede Modalitaet. Die Unterschiede sind klein
// (anderer Befehl, anderer Bericht) — zwei fast gleiche Dialoge waeren teurer
// als ein Schalter, und sie waeren frueher oder spaeter verschieden.

import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, Wand2, ShieldQuestion } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import type { StudioProject } from './studioTypes';

export interface VersionTreeItem { id: string; name: string; version_number: number; }
export interface ModelWithVersionTree { id: string; name: string; versions: VersionTreeItem[]; }

export interface ReviewReport {
  checked: number;
  doubts:  number;
  agree:   number;
  failed:  number;
}

export interface SuggestReport {
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

export default function ModelRunDialog({ mode, project, onClose, onDone }: {
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
