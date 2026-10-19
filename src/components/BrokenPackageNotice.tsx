// Ein Python-Paket ist installiert, laesst sich aber nicht laden (z.B. scipy
// unter macOS 27: "dlopen ... zero-fill section"). Statt der rohen Meldung
// sagt der Hinweis, welches Paket es ist, und repariert es auf Knopfdruck.

import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, CheckCircle, Loader2, Wrench } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import type { BrokenPackage } from './usePackageInstall';

export default function BrokenPackageNotice({ broken, onRepaired, compact = false }: {
  broken: BrokenPackage;
  onRepaired?: () => void;
  compact?: boolean;
}) {
  const { t } = useLanguage();
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  const repair = async () => {
    setBusy(true); setErr(null);
    try {
      const version = await invoke<string>('repair_python_package', { package: broken.package });
      setDone(version);
      onRepaired?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`rounded-xl border border-red-500/30 bg-red-500/10 ${compact ? 'p-3' : 'p-4'} space-y-2`} role="alert">
      <div className="flex items-start gap-2 text-red-200 text-sm">
        <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <div className="space-y-1 min-w-0">
          <p className="font-medium">{t('packageCheck.broken.title', { package: broken.package })}</p>
          <p className="text-red-200/80 text-xs">{t('packageCheck.broken.detail', { package: broken.package })}</p>
          <button onClick={() => setShowRaw(v => !v)} className="text-[11px] text-red-200/60 hover:text-red-100 underline decoration-dotted">
            {t(showRaw ? 'packageCheck.broken.hideRaw' : 'packageCheck.broken.showRaw')}
          </button>
          {showRaw && <p className="text-red-200/60 text-[11px] font-mono break-all">{broken.error}</p>}
        </div>
      </div>
      {done ? (
        <p className="flex items-center gap-1.5 text-xs text-emerald-300">
          <CheckCircle className="w-3.5 h-3.5" /> {t('packageCheck.broken.repaired', { package: broken.package, version: done })}
        </p>
      ) : (
        <button
          onClick={() => void repair()}
          disabled={busy}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-100 text-xs font-medium disabled:opacity-60"
        >
          {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wrench className="w-3.5 h-3.5" />}
          {busy ? t('packageCheck.broken.repairing') : t('packageCheck.broken.repair', { package: broken.package })}
        </button>
      )}
      {err && <p className="text-xs text-red-300 break-words font-mono">{err}</p>}
    </div>
  );
}
