// Aufnahmen als 16-kHz-Mono-WAV.
//
// MediaRecorder nimmt auf, was der Webview kann: MP4/AAC auf macOS, WebM/Opus
// auf Windows. WebM liest das Training ohne ffmpeg gar nicht, M4A nur, wo das
// Betriebssystem es dekodiert — eine unter Windows aufgenommene Datei waere
// im Training still verloren gegangen. WAV liest jede Bibliothek, und 16 kHz
// Mono ist das, womit Sprach- und Audiomodelle (Wav2Vec2, Whisper, AST) ohnehin
// rechnen. Dekodieren kann der Webview, was er selbst aufgenommen hat.

export const WAV_RATE = 16000;

/** PCM-16-WAV aus Float-Samples (-1..1). Rein, damit testbar. */
export function encodeWav(samples: Float32Array, rate: number = WAV_RATE): Uint8Array {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const text = (off: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  text(0, 'RIFF');
  v.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  v.setUint32(16, 16, true);        // Groesse des fmt-Blocks
  v.setUint16(20, 1, true);         // PCM
  v.setUint16(22, 1, true);         // Mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);  // Bytes je Sekunde
  v.setUint16(32, 2, true);         // Bytes je Sample
  v.setUint16(34, 16, true);        // Bit je Sample
  text(36, 'data');
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

/** Beliebige Aufnahme -> 16-kHz-Mono-WAV. Wirft, wenn der Webview das Format nicht dekodiert. */
export async function toWav16kMono(data: ArrayBuffer): Promise<Uint8Array> {
  const Ctx = (window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
  const ctx = new Ctx();
  try {
    const decoded = await ctx.decodeAudioData(data.slice(0));
    const frames = Math.max(1, Math.ceil(decoded.duration * WAV_RATE));
    const off = new OfflineAudioContext(1, frames, WAV_RATE);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return encodeWav(rendered.getChannelData(0), WAV_RATE);
  } finally {
    void ctx.close();
  }
}

/** Endungen, die nach WAV gewandelt werden sollten (Aufnahmen aus dem Webview). */
export function brauchtWav(media: string): boolean {
  return /\.(m4a|webm|ogg|aac|mp4)$/i.test(media);
}
