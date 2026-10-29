// Zustimmung zu AGB & Datenschutz nachholen.
//
// Erscheint nach dem Login, wenn das Konto der aktuellen Fassung noch nicht
// zugestimmt hat: gar keine Zustimmung ('missing') oder nur einer aelteren
// ('updated'). Ohne Zustimmung geht es nicht weiter — "Ablehnen" meldet ab.

import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import { Loader2 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import ModalPortal from './ui/ModalPortal';

interface Props {
  apiKey: string;
  reason: 'missing' | 'updated';
  onAccepted: () => void;
  onDecline: () => void;
}

export default function TermsGate({ apiKey, reason, onAccepted, onDecline }: Props) {
  const { t, language } = useLanguage();
  const [checked, setChecked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const lang = language === 'de' ? 'de' : 'en';
  const link = (path: string, label: string) => (
    <button
      type="button"
      onClick={() => openUrl(`https://frame-train.com/${lang}/${path}`).catch(() => {})}
      className="text-white underline underline-offset-2"
    >
      {label}
    </button>
  );

  const accept = async () => {
    setBusy(true);
    setFailed(false);
    try {
      await invoke('accept_terms', { apiKey });
      onAccepted();
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <ModalPortal>
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
        <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">{t(reason === 'updated' ? 'termsGate.updatedTitle' : 'termsGate.title')}</h2>
          <p className="text-sm text-gray-400">{t(reason === 'updated' ? 'termsGate.updatedIntro' : 'termsGate.intro')}</p>
          <label className="flex items-start gap-3 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" checked={checked} onChange={e => setChecked(e.target.checked)} className="mt-1" />
            <span>
              {t('termsGate.checkboxBefore')} {link('terms', t('termsGate.terms'))} {t('termsGate.and')}{' '}
              {link('privacy', t('termsGate.privacy'))}{t('termsGate.checkboxAfter')}
            </span>
          </label>
          {failed && <p className="text-xs text-red-300">{t('termsGate.failed')}</p>}
          <div className="flex gap-2 justify-end">
            <button
              type="button"
              onClick={onDecline}
              className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-sm text-gray-300"
            >
              {t('termsGate.decline')}
            </button>
            <button
              type="button"
              onClick={accept}
              disabled={!checked || busy}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-sm text-emerald-200 disabled:opacity-40"
            >
              {busy && <Loader2 className="w-4 h-4 animate-spin" />}
              {t('termsGate.accept')}
            </button>
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}
