import BrandMark from './ui/BrandMark';

// Startbildschirm: ruhig, nur das Logo. Die drei Eingaben leuchten nacheinander
// zum Ergebnis hin auf (siehe .ft-mark-live in index.css).
export default function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center" style={{ background: 'var(--ft-wall)' }}>
      <div className="flex flex-col items-center gap-5">
        <BrandMark size={76} live className="ft-mark-shadow" />
        <div className="flex flex-col items-center gap-1.5">
          <span className="text-white" style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.03em' }}>FrameTrain</span>
          <span className="ft-label">Local ML Training</span>
        </div>
      </div>
    </div>
  );
}
