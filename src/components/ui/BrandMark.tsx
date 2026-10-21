import { useId } from 'react';

/**
 * FrameTrain-Logo: drei Eingaben, ein Ergebnis (Neuron). Gleiche Geometrie wie
 * src-tauri/icons/favicon.svg. `live` laesst die Verbindungen nacheinander
 * aufleuchten, z. B. beim Laden.
 */
export default function BrandMark({ size = 28, live = false, className = '' }: {
  size?: number;
  live?: boolean;
  className?: string;
}) {
  const gid = useId().replace(/:/g, '');
  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      className={`${live ? 'ft-mark-live' : ''} ${className}`}
      aria-hidden="true"
      style={{ flex: 'none' }}
    >
      <defs>
        <linearGradient id={`ftm-${gid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5559F2" />
          <stop offset="1" stopColor="#3B3ED6" />
        </linearGradient>
      </defs>
      <rect width="100" height="100" rx="22.5" fill={`url(#ftm-${gid})`} />
      <g fill="none" stroke="#FFFFFF" strokeOpacity="0.55" strokeWidth="5" strokeLinecap="round">
        <path className="ft-mark-edge" d="M28 32L62 50" />
        <path className="ft-mark-edge" d="M28 50H62" />
        <path className="ft-mark-edge" d="M28 68L62 50" />
      </g>
      <circle cx="28" cy="32" r="7" fill="#FFFFFF" />
      <circle cx="28" cy="50" r="7" fill="#FFFFFF" />
      <circle cx="28" cy="68" r="7" fill="#FFFFFF" />
      <circle className="ft-mark-out" cx="62" cy="50" r="17" fill="#FF7A3D" />
    </svg>
  );
}
