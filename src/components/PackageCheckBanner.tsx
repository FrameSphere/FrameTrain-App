// Hinweis vor dem Training/Test: fehlen Python-Pakete fuer diese Aufgabe?
// Mit einem Knopf, der die passende Gruppe in das Python der App installiert.

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, Download, Loader2 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { describeMissing, usePackageInstall, type TaskPackageStatus } from './usePackageInstall';

export default function PackageCheckBanner({ taskType }: { taskType?: string | null }) {
  const { t } = useLanguage();
  const [status, setStatus] = useState<TaskPackageStatus | null>(null);

  const check = useCallback(async () => {
    if (!taskType) { setStatus(null); return; }
    try {
      setStatus(await invoke<TaskPackageStatus>('check_task_packages', { taskType }));
    } catch {
      // Ohne Python kann die Pruefung nichts sagen — der System-Check meldet das.
      setStatus(null);
    }
  }, [taskType]);

  useEffect(() => { void check(); }, [check]);
  const { install, running, progress, error } = usePackageInstall(() => { void check(); });

  if (!status || status.missing.length === 0) return null;
  const group = t(`packageCheck.groups.${status.group}`, status.group);

  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 space-y-2" role="alert">
      <div className="flex items-start gap-2 text-amber-200 text-sm">
        <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <div className="space-y-1">
          <p className="font-medium">{t('packageCheck.title').replace('{group}', group)}</p>
          <p className="text-amber-200/80 text-xs">
            {t('packageCheck.missing').replace('{list}', describeMissing(status.missing, t('packageCheck.tooOld')))}
          </p>
          <p className="text-amber-200/60 text-[11px] font-mono break-all">{t('packageCheck.python').replace('{path}', status.python)}</p>
        </div>
      </div>
      <button
        onClick={() => void install([status.group])}
        disabled={running}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-100 text-xs font-medium disabled:opacity-60"
      >
        {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
        {running ? t('packageCheck.installing') : t('packageCheck.install').replace('{group}', group)}
      </button>
      {running && progress && (
        <div className="space-y-1">
          <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className="h-full bg-amber-400 transition-all" style={{ width: `${Math.min(progress.pct, 100)}%` }} />
          </div>
          <p className="text-[11px] text-amber-200/70 font-mono break-words">{progress.message}</p>
        </div>
      )}
      {error && <p className="text-xs text-red-300 break-words">{error}</p>}
    </div>
  );
}
