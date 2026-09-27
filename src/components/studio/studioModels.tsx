// Welches Modell zu welchem Projekt passt.
//
// Die Auswahl im Export- und Vorschlags-Dialog nahm bisher einfach das erste
// Modell der Liste. Bei einem Textprojekt war das das Bildmodell yolo8n — und
// der fertige Datensatz landete dann auch bei diesem Modell, wo er nie
// trainiert werden kann. Die Erkennung laeuft ueber dieselbe Plugin-Registry
// wie das Training; passende Modelle stehen oben und sind vorausgewaehlt, die
// anderen bleiben waehlbar, falls die Erkennung ein Modell nicht zuordnen kann.

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { detectPluginForModel, type ModelDetectionInfo } from '../../plugins/registry';
import { useLanguage } from '../../contexts/LanguageContext';
import type { StudioProject } from './studioTypes';

export type StudioModel = ModelDetectionInfo;

/** Welche Plugin-Aufgaben aus diesem Projekt trainieren koennen. */
export function passendeAufgaben(project: Pick<StudioProject, 'modality' | 'task'>): string[] {
  if (project.modality === 'text') {
    return project.task === 'pairs' ? ['seq2seq'] : ['seq_classification'];
  }
  if (project.modality === 'video') return ['video_classification'];
  if (project.modality === 'image' && project.task === 'classify') {
    return ['hf_image_classification', 'image_classification'];
  }
  if (project.modality === 'audio') {
    // Transkripte trainieren Sprach-zu-Text-Modelle, die in der Registry als
    // seq2seq gefuehrt sind.
    return project.task === 'transcript' ? ['seq2seq', 'audio_classification'] : ['audio_classification'];
  }
  return ['detect'];
}

export function passtZumProjekt(model: StudioModel, project: Pick<StudioProject, 'modality' | 'task'>): boolean {
  const r = detectPluginForModel(model);
  return r.supported && passendeAufgaben(project).includes(r.plugin.taskType);
}

/** Passende zuerst, Reihenfolge innerhalb der Gruppen bleibt erhalten. */
export function ordneModelle<T extends StudioModel>(
  models: T[], project: Pick<StudioProject, 'modality' | 'task'>,
): { passend: T[]; andere: T[] } {
  const passend: T[] = [];
  const andere: T[] = [];
  for (const m of models) (passtZumProjekt(m, project) ? passend : andere).push(m);
  return { passend, andere };
}

/** Laedt die Modelle und waehlt das erste passende vor. */
export function useStudioModels(project: StudioProject) {
  const [models, setModels] = useState<StudioModel[]>([]);
  const [modelId, setModelId] = useState('');
  useEffect(() => {
    invoke<StudioModel[]>('list_models')
      .then(list => {
        setModels(list);
        const { passend, andere } = ordneModelle(list, project);
        const erstes = passend[0] ?? andere[0];
        if (erstes) setModelId(erstes.id);
      })
      .catch(() => { /* Auswahl bleibt leer, der Knopf bleibt gesperrt */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);
  return { models, modelId, setModelId };
}

/** Ein Modell, das zur Projektart passt und frei verfuegbar ist — als Tipp,
 *  wenn noch keines da ist. Einsteiger wissen sonst nicht, wonach sie suchen sollen. */
export function empfohlenesModell(project: Pick<StudioProject, 'modality' | 'task'>): string {
  if (project.modality === 'video') return 'MCG-NJU/videomae-base';
  if (project.modality === 'audio') return project.task === 'transcript' ? 'openai/whisper-small' : 'facebook/wav2vec2-base';
  if (project.modality === 'text') return project.task === 'pairs' ? 'google/mt5-small' : 'distilbert-base-multilingual-cased';
  if (project.task === 'classify') return 'google/vit-base-patch16-224';
  return 'yolo11n';
}

/** Die Auswahlliste selbst, gruppiert nach passend und andere. */
export function StudioModelSelect({ project, models, value, onChange }: {
  project: StudioProject;
  models: StudioModel[];
  value: string;
  onChange: (id: string) => void;
}) {
  const { t } = useLanguage();
  const { passend, andere } = ordneModelle(models, project);
  const option = (m: StudioModel) =>
    <option key={m.id} value={m.id} className="bg-[#101218]">{m.name}</option>;
  return (
    <>
    <select value={value} onChange={e => onChange(e.target.value)}
      className="mt-1 w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-white/25">
      {models.length === 0 && <option value="">{t('studio.export.noModels')}</option>}
      {passend.length > 0 && andere.length > 0 ? (
        <>
          <optgroup label={t('studio.export.modelsFitting')} className="bg-[#101218]">
            {passend.map(option)}
          </optgroup>
          <optgroup label={t('studio.export.modelsOther')} className="bg-[#101218]">
            {andere.map(option)}
          </optgroup>
        </>
      ) : (
        models.map(option)
      )}
    </select>
    {/* Ohne passendes Modell wuerde der Datensatz an ein Modell gehaengt, das
        ihn nie trainieren kann — das muss man sehen, bevor man exportiert. */}
    {models.length > 0 && passend.length === 0 && (
      <p className="mt-1.5 text-amber-300/90 text-[11px]" data-testid="no-fitting-model">
        {t('studio.export.noFittingModel', { model: empfohlenesModell(project) })}
      </p>
    )}
    </>
  );
}
