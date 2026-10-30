// Lizenzhinweise der mitgelieferten Open-Source-Pakete (Einstellungen > About).
// public/third-party-notices.txt erzeugt scripts/generate-licenses.mjs; die rund
// 2 MB werden erst beim Oeffnen geladen.

import { useEffect, useState } from 'react';
import { Loader2, X } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { useEscapeKey } from '../hooks/useEscapeKey';
import ModalPortal from './ui/ModalPortal';

export default function ThirdPartyNotices({ onClose }: { onClose: () => void }) {
  const { t } = useLanguage();
  useEscapeKey(onClose, true);
  const [text, setText] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch('/third-party-notices.txt')
      .then(res => {
        if (!res.ok) throw new Error(String(res.status));
        return res.text();
      })
      .then(body => { if (!cancelled) setText(body); })
      .catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; };
  }, []);

  return (
    <ModalPortal>
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
        <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-3xl h-[80vh] flex flex-col overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-white">{t('settings.about.licensesTitle')}</h2>
            <button onClick={onClose} aria-label={t('settings.about.licensesClose')} className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white">
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="flex-1 overflow-auto p-6">
            {failed ? (
              <p className="text-sm text-red-300">{t('settings.about.licensesFailed')}</p>
            ) : text === null ? (
              <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />
            ) : (
              <pre className="text-[11px] font-mono text-gray-400 whitespace-pre-wrap break-words">{text}</pre>
            )}
          </div>
        </div>
      </div>
    </ModalPortal>
  );
}
