// Waehrend ein Datei-Dialog oder ein Bildschirmfoto laeuft, verliert der
// Schnell-Chat den Fokus. Er darf sich dann nicht ausblenden — sonst ist das
// Ergebnis weg, bevor es ankommt.

let active = 0;

/** Blur-Schutz an; die zurueckgegebene Funktion schaltet ihn (verzoegert) wieder ab. */
export function guardBlur(): () => void {
  active += 1;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    // Der Fokus kommt erst kurz nach dem Schliessen des Dialogs zurueck.
    setTimeout(() => { active = Math.max(0, active - 1); }, 400);
  };
}

export function blurGuarded(): boolean {
  return active > 0;
}
