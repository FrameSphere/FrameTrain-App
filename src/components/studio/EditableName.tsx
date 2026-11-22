// Ein Name, der sich an Ort und Stelle aendern laesst: Projekt oder Klasse.
// Ein Tippfehler in einer Klasse war sonst fuer immer — das Backend konnte
// umbenennen, die Oberflaeche bot es nicht an.

import { useState } from 'react';
import { Pencil } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

export default function EditableName({ value, onSave, className, inputClassName, label }: {
  value: string;
  /** Wirft oder liefert false, wenn der Name nicht gespeichert wurde. */
  onSave: (name: string) => Promise<boolean | void>;
  className?: string;
  inputClassName?: string;
  /** Beschriftung fuer Knopf und Feld. */
  label: string;
}) {
  const { t } = useLanguage();
  const [entwurf, setEntwurf] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const speichern = async () => {
    const neu = (entwurf ?? '').trim();
    if (!neu || neu === value) { setEntwurf(null); return; }
    setBusy(true);
    try {
      if (await onSave(neu) !== false) setEntwurf(null);
    } finally {
      setBusy(false);
    }
  };

  if (entwurf !== null) {
    return (
      <input value={entwurf} autoFocus disabled={busy} aria-label={label}
        onChange={e => setEntwurf(e.target.value)}
        onFocus={e => e.currentTarget.select()}
        onBlur={() => { void speichern(); }}
        onClick={e => e.stopPropagation()}
        onKeyDown={e => {
          e.stopPropagation();
          if (e.key === 'Enter') { e.preventDefault(); void speichern(); }
          if (e.key === 'Escape') { e.preventDefault(); setEntwurf(null); }
        }}
        className={`min-w-0 px-1.5 py-0.5 rounded bg-black/30 border border-white/25 text-white focus:outline-none ${inputClassName ?? 'flex-1 text-xs'}`} />
    );
  }
  return (
    <span className="group/name min-w-0 inline-flex items-center gap-1.5">
      <span className={className ?? 'text-gray-200 text-xs truncate'} title={value}
        onDoubleClick={e => { e.stopPropagation(); setEntwurf(value); }}>{value}</span>
      <button type="button" onClick={e => { e.stopPropagation(); setEntwurf(value); }}
        aria-label={label} title={t('studio.rename.hint')}
        className="p-0.5 rounded text-gray-500 hover:text-white hover:bg-white/10 opacity-0 group-hover/name:opacity-100 focus:opacity-100 transition-opacity flex-shrink-0">
        <Pencil className="w-3 h-3" />
      </button>
    </span>
  );
}
