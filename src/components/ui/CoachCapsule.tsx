import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

/**
 * Geschlossene Coach-Kapsel unten rechts.
 *
 * Unten rechts ist sie am besten erreichbar, liegt dort aber auf Knoepfen der
 * Seiten ("Start AI Analysis", Senden im Chat, "Clear" im Hosting). Deshalb:
 * - Liegt unter ihr ein Bedienelement und kommt die Maus, hebt sie sich leicht
 *   an und neigt sich — der Knopf darunter wird frei, die Kapsel bleibt klickbar.
 * - Wird der freigelegte Knopf geklickt, macht sie 4 s ganz Platz.
 * - Ueber den Pfeil laesst sie sich als Lasche an den Rand einklappen.
 */

const INTERACTIVE = 'button, a[href], input, textarea, select, [role="button"], [role="switch"], [contenteditable="true"]';
/** Wie lange die Kapsel nach dem Verlassen noch angehoben bleibt — kurz,
 *  damit sie nach einem versehentlichen Darueberfahren gleich wieder sitzt. */
const SETTLE_MS = 220;
/** Wie lange sie nach einem Klick auf den Knopf darunter weg bleibt. */
const AWAY_MS = 4000;
/** Hubhoehe beim Anheben — nur ein Stueck, gerade genug, um den Knopf darunter
 *  sichtbar und an seiner Unterkante klickbar zu machen. */
const LIFT_PX = 30;

/** Das Bedienelement unter der Kapsel (ohne die Kapsel selbst), sonst null. */
export function findCoveredControl(capsule: HTMLElement): Element | null {
  const r = capsule.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) return null;
  const points: [number, number][] = [
    [r.left + r.width / 2, r.top + r.height / 2],
    [r.left + 8, r.top + r.height / 2],
    [r.right - 8, r.top + r.height / 2],
    [r.left + r.width / 2, r.top + 6],
    [r.left + r.width / 2, r.bottom - 6],
  ];
  for (const [x, y] of points) {
    for (const el of document.elementsFromPoint?.(x, y) ?? []) {
      if (capsule.contains(el)) continue;
      const hit = el.closest(INTERACTIVE);
      if (hit) return hit;
    }
  }
  return null;
}

export default function CoachCapsule({ onOpen, title, children }: {
  onOpen: () => void;
  title: string;
  children: ReactNode;
}) {
  const { t } = useLanguage();
  const capRef = useRef<HTMLButtonElement>(null);
  const coveredRef = useRef<Element | null>(null);
  const settleTimer = useRef<number | undefined>(undefined);
  const awayTimer = useRef<number | undefined>(undefined);
  const [lifted, setLifted] = useState(false);
  const [away, setAway] = useState(false);
  const [tucked, setTucked] = useState(false);

  const clearSettle = () => window.clearTimeout(settleTimer.current);

  // Maus kommt auf die Kapsel: liegt etwas Klickbares darunter? Gemessen wird
  // in der Ruhelage, deshalb nur, solange sie noch nicht angehoben ist.
  const handleEnter = () => {
    clearSettle();
    if (lifted || !capRef.current) return;
    const covered = findCoveredControl(capRef.current);
    if (!covered) return;
    coveredRef.current = covered;
    setLifted(true);
  };

  // Angehoben bleibt sie, solange die Maus auf ihr oder dem freigelegten
  // Knopf ist; danach setzt sie sich mit kurzer Verzoegerung wieder.
  useEffect(() => {
    if (!lifted) return;
    const inside = (el: Element | null, x: number, y: number) => {
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return x >= r.left - 4 && x <= r.right + 4 && y >= r.top - 4 && y <= r.bottom + 4;
    };
    const onMove = (e: MouseEvent) => {
      if (inside(capRef.current, e.clientX, e.clientY) || inside(coveredRef.current, e.clientX, e.clientY)) {
        clearSettle();
        return;
      }
      clearSettle();
      settleTimer.current = window.setTimeout(() => setLifted(false), SETTLE_MS);
    };
    // Klick auf den freigelegten Knopf: kurz ganz Platz machen.
    const onClick = (e: MouseEvent) => {
      const covered = coveredRef.current;
      if (covered && e.target instanceof Node && covered.contains(e.target)) {
        clearSettle();
        setLifted(false);
        setAway(true);
        window.clearTimeout(awayTimer.current);
        awayTimer.current = window.setTimeout(() => setAway(false), AWAY_MS);
      }
    };
    // Maus verlaesst das Fenster: kein mousemove mehr, also sofort absetzen.
    const onLeaveWindow = () => { clearSettle(); setLifted(false); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('click', onClick, true);
    document.documentElement.addEventListener('mouseleave', onLeaveWindow);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('click', onClick, true);
      document.documentElement.removeEventListener('mouseleave', onLeaveWindow);
    };
  }, [lifted]);

  useEffect(() => () => {
    window.clearTimeout(settleTimer.current);
    window.clearTimeout(awayTimer.current);
  }, []);

  const hidden = away || tucked;

  return (
    <>
      {lifted && !hidden && (
        <span className="ft-coach-hint fixed right-6 z-40 text-[11.5px] text-gray-400 pointer-events-none" style={{ bottom: 20 + LIFT_PX + 48 }}>
          {t('aiCoach.capsule.yieldHint')}
        </span>
      )}
      <button
        ref={capRef}
        onClick={onOpen}
        onMouseEnter={handleEnter}
        title={title}
        aria-hidden={hidden || undefined}
        tabIndex={hidden ? -1 : undefined}
        data-state={hidden ? 'hidden' : lifted ? 'lifted' : 'rest'}
        className="ft-coach-capsule group ft-glass ft-press fixed bottom-5 right-5 h-11 pl-3.5 pr-4 rounded-full flex items-center gap-2.5 z-40 text-[13.5px] font-semibold text-white"
        style={{
          transform: hidden ? 'translateX(160px)' : lifted ? `translateY(-${LIFT_PX}px) rotate(-2deg)` : undefined,
          opacity: hidden ? 0 : 1,
          pointerEvents: hidden ? 'none' : undefined,
        }}
      >
        {children}
        <span
          role="button"
          tabIndex={-1}
          aria-label={t('aiCoach.capsule.tuck')}
          title={t('aiCoach.capsule.tuck')}
          onClick={e => { e.stopPropagation(); setLifted(false); setTucked(true); }}
          className="-mr-1.5 w-5 h-5 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <ChevronRight className="w-3.5 h-3.5" />
        </span>
      </button>
      {/* Eingeklappt: schmale Lasche am rechten Rand, bis sie geklickt wird. */}
      <button
        onClick={() => setTucked(false)}
        aria-label={t('aiCoach.capsule.untuck')}
        title={t('aiCoach.capsule.untuck')}
        tabIndex={tucked ? undefined : -1}
        className="ft-coach-tab ft-glass fixed right-0 bottom-6 z-40 w-3.5 h-11 rounded-l-lg flex items-center justify-center"
        style={{ transform: tucked ? 'translateX(0)' : 'translateX(18px)', pointerEvents: tucked ? undefined : 'none' }}
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--ft-out)' }} />
      </button>
    </>
  );
}
