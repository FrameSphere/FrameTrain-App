// Zentrale Quelle für die Tutorial-Videos (YouTube).
//
// Spiegelt bewusst website/src/lib/videos.ts – dieselben IDs, damit App und
// Website nicht auseinanderlaufen. Beim Tausch eines Videos hier UND dort die
// ID anpassen.
//
// In der Desktop-App (Tauri) betten wir keinen Player ein, sondern öffnen den
// Link im Standardbrowser des Systems (@tauri-apps/plugin-shell → open()).

export const VIDEOS = {
  // "How to install FrameTrain"
  install: 'kIgUvcrQbJA',
  // "How to train your first model"
  train: 'WkU9r3TiF74',
} as const

export type VideoKey = keyof typeof VIDEOS

export const youtubeWatchUrl = (id: string) => `https://www.youtube.com/watch?v=${id}`
