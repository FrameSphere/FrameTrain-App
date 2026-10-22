import { useEffect, useState } from 'react';

// Startbildschirm: das Logo setzt sich zusammen (Eingaben erscheinen,
// Verbindungen zeichnen sich), danach laufen Signale von den drei Eingaben
// zum Ergebnis. Jede Ankunft gibt dem Ergebnis-Knoten einen Impuls, der als
// Ring nach aussen laeuft. Darunter zeichnet sich eine kleine Loss-Kurve.
// Takt: 2,4 s pro Zyklus, Styles in index.css (.ft-load*).

const EDGES = ['M28 32L62 50', 'M28 50H62', 'M28 68L62 50'];

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!mq) return;
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return reduced;
}

export default function LoadingScreen() {
  const reduced = usePrefersReducedMotion();

  return (
    <div
      className="ft-load min-h-screen w-full flex items-center justify-center"
      style={{ background: 'var(--ft-wall)' }}
      role="status"
      aria-label="FrameTrain"
    >
      <div className="flex flex-col items-center">
        <svg viewBox="0 0 100 100" width="84" height="84" className="ft-load-mark" aria-hidden="true">
          <defs>
            <linearGradient id="ft-load-tile" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#5559F2" />
              <stop offset="1" stopColor="#3B3ED6" />
            </linearGradient>
          </defs>

          <rect className="ft-load-tile" width="100" height="100" rx="22.5" fill="url(#ft-load-tile)" />

          {/* Impuls: Ring vom Ergebnis-Knoten nach aussen */}
          <circle className="ft-load-ring" cx="62" cy="50" r="17" fill="none" stroke="#FF7A3D" strokeWidth="2" />

          <g fill="none" stroke="#FFFFFF" strokeOpacity="0.32" strokeWidth="5" strokeLinecap="round">
            {EDGES.map((d, i) => (
              <path key={d} className="ft-load-edge" style={{ animationDelay: `${0.18 + i * 0.07}s` }} d={d} pathLength={1} />
            ))}
          </g>

          {[32, 50, 68].map((cy, i) => (
            <circle key={cy} className="ft-load-in" style={{ animationDelay: `${i * 0.07}s` }} cx="28" cy={cy} r="7" fill="#FFFFFF" />
          ))}

          {/* Signale entlang der Verbindungen (SMIL, im Ruhemodus aus) */}
          {!reduced && EDGES.map((d, i) => (
            <circle key={`p${d}`} r="4.2" fill="#FFFFFF" opacity="0">
              <animateMotion path={d} dur="2.4s" begin={`${0.75 + i * 0.09}s`} repeatCount="indefinite"
                keyPoints="0;0;1;1" keyTimes="0;0.02;0.26;1" calcMode="linear" />
              <animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;0.03;0.22;0.27;1"
                dur="2.4s" begin={`${0.75 + i * 0.09}s`} repeatCount="indefinite" />
            </circle>
          ))}

          <circle className="ft-load-out" cx="62" cy="50" r="17" fill="#FF7A3D" />
        </svg>

        <span className="ft-load-word text-white mt-5" style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.03em' }}>
          FrameTrain
        </span>

        {/* Kleine Loss-Kurve: faellt ab, wie ein Training */}
        <svg viewBox="0 0 120 24" width="120" height="24" className="ft-load-curve mt-3" aria-hidden="true">
          <line x1="0" y1="23.5" x2="120" y2="23.5" stroke="var(--ft-line-strong)" strokeWidth="1" />
          <path
            className="ft-load-loss"
            d="M2 3 C 14 4, 18 12, 30 14 S 46 15, 56 18 S 78 19, 90 20.5 S 110 21, 118 21.2"
            fill="none"
            stroke="var(--ft-out)"
            strokeWidth="1.8"
            strokeLinecap="round"
            pathLength={1}
          />
        </svg>

        <span className="ft-label ft-load-word mt-2" style={{ animationDelay: '0.25s' }}>Local ML Training</span>
      </div>
    </div>
  );
}
