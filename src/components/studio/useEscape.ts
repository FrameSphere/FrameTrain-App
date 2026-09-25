// Escape schliesst einen Dialog der Werkstatt.
//
// Im Test liess sich "Woher kommen die Bilder?" nur per Klick daneben
// schliessen. Die Werkbaenke fangen Tasten nur ab, solange kein Dialog offen
// ist — Escape kam also bei niemandem an. Solange ein Dialog arbeitet (ein
// Export, ein Modelllauf), bleibt er offen: abbrechen koennte man den Lauf
// damit ohnehin nicht, nur seinen Bericht verlieren.

import { useEffect, useRef } from 'react';

export function useEscape(onClose: () => void, active = true) {
  const schliessen = useRef(onClose);
  schliessen.current = onClose;
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      schliessen.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active]);
}
