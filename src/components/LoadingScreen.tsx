export default function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 overflow-hidden relative">
      {/* Ambient background glows */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full opacity-20"
          style={{
            background: 'radial-gradient(circle, #a855f7 0%, #ec4899 50%, transparent 70%)',
            animation: 'ambientPulse 4s ease-in-out infinite',
          }}
        />
      </div>

      <div className="relative flex flex-col items-center gap-10">
        {/* Icon with pulsing rings */}
        <div className="relative flex items-center justify-center">
          {/* Outer ring 3 */}
          <div
            className="absolute rounded-full border border-purple-500/10"
            style={{
              width: 200,
              height: 200,
              animation: 'ringPulse 2.4s ease-out infinite',
              animationDelay: '0.6s',
            }}
          />
          {/* Outer ring 2 */}
          <div
            className="absolute rounded-full border border-purple-500/20"
            style={{
              width: 160,
              height: 160,
              animation: 'ringPulse 2.4s ease-out infinite',
              animationDelay: '0.3s',
            }}
          />
          {/* Outer ring 1 */}
          <div
            className="absolute rounded-full border border-purple-400/30"
            style={{
              width: 120,
              height: 120,
              animation: 'ringPulse 2.4s ease-out infinite',
            }}
          />

          {/* Icon container with glow */}
          <div
            className="relative z-10 rounded-[22px] shadow-2xl"
            style={{
              background: 'linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%)',
              width: 80,
              height: 80,
              animation: 'iconBreath 2.4s ease-in-out infinite',
              boxShadow: '0 0 40px rgba(168, 85, 247, 0.5), 0 0 80px rgba(236, 72, 153, 0.3)',
            }}
          >
            {/* F letter */}
            <div className="absolute inset-0 flex items-center justify-center">
              <span
                style={{
                  fontFamily: 'Arial, sans-serif',
                  fontSize: 42,
                  fontWeight: 900,
                  color: 'white',
                  lineHeight: 1,
                  letterSpacing: '-1px',
                  userSelect: 'none',
                }}
              >
                F
              </span>
            </div>
          </div>
        </div>

        {/* Text section */}
        <div className="flex flex-col items-center gap-3">
          <h1
            className="text-white font-bold tracking-tight"
            style={{ fontSize: 28, letterSpacing: '-0.5px' }}
          >
            FrameTrain
          </h1>

          {/* Animated dots */}
          <div className="flex items-center gap-2">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="rounded-full bg-purple-400"
                style={{
                  width: 6,
                  height: 6,
                  animation: 'dotBounce 1.2s ease-in-out infinite',
                  animationDelay: `${i * 0.2}s`,
                  opacity: 0.5,
                }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Keyframe styles */}
      <style>{`
        @keyframes ringPulse {
          0%   { transform: scale(0.85); opacity: 0.8; }
          50%  { transform: scale(1.05); opacity: 0.3; }
          100% { transform: scale(0.85); opacity: 0.8; }
        }
        @keyframes iconBreath {
          0%   { transform: scale(1);    box-shadow: 0 0 40px rgba(168,85,247,0.5), 0 0 80px rgba(236,72,153,0.3); }
          50%  { transform: scale(1.06); box-shadow: 0 0 60px rgba(168,85,247,0.7), 0 0 120px rgba(236,72,153,0.5); }
          100% { transform: scale(1);    box-shadow: 0 0 40px rgba(168,85,247,0.5), 0 0 80px rgba(236,72,153,0.3); }
        }
        @keyframes dotBounce {
          0%, 80%, 100% { transform: translateY(0);    opacity: 0.4; }
          40%            { transform: translateY(-6px); opacity: 1;   }
        }
        @keyframes ambientPulse {
          0%, 100% { transform: translate(-50%, -50%) scale(1);    opacity: 0.15; }
          50%       { transform: translate(-50%, -50%) scale(1.15); opacity: 0.25; }
        }
      `}</style>
    </div>
  );
}
