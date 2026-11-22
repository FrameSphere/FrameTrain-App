// Was mit mehreren ausgewaehlten Samples geht: Klasse geben, ueberspringen,
// entfernen. Entfernen braucht wie beim einzelnen Sample einen zweiten Klick.

import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, SkipForward, Tag, X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useNotification } from '../../contexts/NotificationContext';
import RemoveSampleButton from './RemoveSampleButton';

export default function AuswahlLeiste({ projectId, ids, klassen, onLeeren, onFertig }: {
  projectId: string;
  ids: Set<string>;
  /** Leer: im Projekt gibt es keine Klasse je Sample (Boxen, Paare, Transkripte). */
  klassen: string[];
  onLeeren: () => void;
  onFertig: () => void | Promise<void>;
}) {
  const { t } = useLanguage();
  const { error } = useNotification();
  const [lauf, setLauf] = useState<{ cur: number; total: number } | null>(null);
  if (ids.size === 0) return null;

  const jedes = async (tu: (id: string) => Promise<unknown>) => {
    const alle = [...ids];
    try {
      for (const [i, id] of alle.entries()) {
        setLauf({ cur: i + 1, total: alle.length });
        await tu(id);
      }
      onLeeren();
    } catch (err: unknown) {
      error(t('studio.notifications.saveError'), String(err));
    } finally {
      setLauf(null);
      await onFertig();
    }
  };
  // keep: was hier nicht genannt ist, bleibt — Ueberspringen nimmt kein Label weg.
  const setze = (status: string, label?: string) => jedes(id => invoke('studio_set_annotation', {
    projectId, sampleId: id, status, label: label ?? null, target: null, boxes: null, keep: true,
  }));

  return (
    <div className="rounded-xl border border-white/15 bg-white/[0.06] px-3 py-2 flex flex-wrap items-center gap-2" data-testid="auswahl-leiste">
      <span className="text-gray-200 text-xs tabular-nums">
        {lauf ? t('studio.suggest.progress', lauf) : t('studio.selection.count', { count: ids.size })}
      </span>
      {lauf ? <Loader2 className="w-4 h-4 animate-spin text-gray-400" /> : (
        <>
          {klassen.length > 0 && (
            <label className="inline-flex items-center gap-1.5 text-gray-400 text-xs">
              <Tag className="w-3.5 h-3.5" />
              <select value="" aria-label={t('studio.menu.assignClass')}
                onChange={e => { if (e.target.value) void setze('confirmed', e.target.value); }}
                className="px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-white text-xs focus:outline-none focus:border-white/25">
                <option value="" className="bg-[#101218]">{t('studio.menu.assignClass')}</option>
                {klassen.map(k => <option key={k} value={k} className="bg-[#101218]">{k}</option>)}
              </select>
            </label>
          )}
          <button onClick={() => void setze('skipped')}
            className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-sm inline-flex items-center gap-2">
            <SkipForward className="w-4 h-4" /> {t('studio.workbench.skip')}
          </button>
          <RemoveSampleButton key={ids.size} onRemove={() => {
            const alle = [...ids];
            void (async () => {
              try { await invoke('studio_delete_samples', { projectId, sampleIds: alle }); onLeeren(); }
              catch (err: unknown) { error(t('studio.remove.errorTitle'), String(err)); }
              await onFertig();
            })();
          }} />
          <button onClick={onLeeren} aria-label={t('studio.selection.clear')} title={t('studio.selection.clear')}
            className="ml-auto p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10">
            <X className="w-4 h-4" />
          </button>
        </>
      )}
    </div>
  );
}
