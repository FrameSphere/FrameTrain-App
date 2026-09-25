// Ein Sample aus dem Projekt nehmen — mit einem zweiten Klick statt eines
// Dialogs.
//
// Beim Selbst-Erstellen entsteht Ausschuss: ein Tippfehler, ein schlechtes
// erzeugtes Beispiel, eine verpatzte Aufnahme. Ein Bestaetigungsdialog fuer
// jedes Stueck waere zu viel; ein einziger Klick zu wenig, denn die Datei ist
// danach weg. Der erste Klick macht den Knopf rot, der zweite entfernt, und
// nach drei Sekunden ohne zweiten Klick ist er wieder harmlos.

import { useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

export default function RemoveSampleButton({ onRemove, disabled }: {
  onRemove: () => void;
  disabled?: boolean;
}) {
  const { t } = useLanguage();
  const [sicher, setSicher] = useState(false);

  useEffect(() => {
    if (!sicher) return;
    const timer = window.setTimeout(() => setSicher(false), 3000);
    return () => window.clearTimeout(timer);
  }, [sicher]);

  return (
    <button
      onClick={() => {
        if (sicher) { setSicher(false); onRemove(); } else setSicher(true);
      }}
      disabled={disabled}
      title={t('studio.remove.hint')}
      className={`px-3 py-2 rounded-lg border text-sm inline-flex items-center gap-2 transition-all disabled:opacity-40 ${sicher
        ? 'bg-red-500/20 border-red-500/40 text-red-200'
        : 'bg-white/5 hover:bg-white/10 border-white/10 text-gray-400'}`}>
      <Trash2 className="w-4 h-4" />
      {sicher ? t('studio.remove.confirm') : t('studio.remove.button')}
    </button>
  );
}
