// Paketgruppen nachinstallieren (HuggingFace-Stack, YOLO, LLM, Generativ).
//
// Die Gruppen LLM und Generativ sind im Erststart optional. Wer sie dort nicht
// gewaehlt hat, bekam bisher erst im Training einen ImportError — und einen
// "pip install"-Hinweis, der beim Kunden oft in ein anderes Python fuehrt.
// Hier installiert die App selbst, in genau den Interpreter, mit dem sie
// trainiert (install_plugins im Backend).

import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export interface InstallProgressEvent {
  plugin_id: string;
  status: string;
  message: string;
  progress?: number | null;
}

export interface MissingPackage { package: string; installed: boolean; version?: string | null }

export interface BrokenPackage { package: string; error: string }

export interface TaskPackageStatus {
  group: string;
  missing: MissingPackage[];
  python: string;
  /** Installiert, aber nicht ladbar — die "fehlenden" Pakete sind dann oft Folgefehler. */
  broken?: BrokenPackage | null;
}

const MODULE_TO_PIP: Record<string, string> = {
  sklearn: 'scikit-learn', PIL: 'pillow', cv2: 'opencv-python',
  sentence_transformers: 'sentence-transformers', mlx_lm: 'mlx-lm', rouge_score: 'rouge-score',
};

/** Gleiche Erkennung wie broken_culprit im Backend — fuer Fehlertexte, die
 *  direkt aus Python kommen (Hosting, Labor): "dlopen(.../site-packages/scipy/…". */
export function brokenPackageFromError(message: string | null | undefined): BrokenPackage | null {
  if (!message) return null;
  const binary = /dlopen\(|DLL load failed|cannot open shared object|binary incompatibility|undefined symbol|Symbol not found/.test(message);
  if (!binary) return null;
  const m = message.match(/site-packages[\\/]+([A-Za-z0-9_]+)/);
  if (!m) return null;
  return { package: MODULE_TO_PIP[m[1]] ?? m[1], error: message.split('\n')[0].slice(0, 300) };
}

/** "peft (0.7.1 zu alt)" bzw. "diffusers" — fuer Hinweise. */
export function describeMissing(pkgs: MissingPackage[], tooOld: string): string {
  return pkgs.map(p => (p.version ? `${p.package} (${p.version} ${tooOld})` : p.package)).join(', ');
}

export function usePackageInstall(onDone?: () => void) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ pct: number; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (!running) return;
    let offP: (() => void) | undefined;
    let offC: (() => void) | undefined;
    let alive = true;
    (async () => {
      offP = await listen<InstallProgressEvent>('plugin-install-progress', e => {
        if (!alive) return;
        const p = e.payload;
        setProgress(prev => ({ pct: typeof p.progress === 'number' ? p.progress : (prev?.pct ?? 0), message: p.message }));
        if (p.status === 'failed') {
          setError(p.message);
          setRunning(false);
        }
      });
      offC = await listen('plugin-install-complete', () => {
        if (!alive) return;
        setRunning(false);
        setProgress(null);
        doneRef.current?.();
      });
    })();
    return () => { alive = false; offP?.(); offC?.(); };
  }, [running]);

  const install = useCallback(async (groups: string[]) => {
    setError(null);
    setProgress({ pct: 0, message: '' });
    setRunning(true);
    try {
      await invoke('install_plugins', { pluginIds: groups });
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }, []);

  return { install, running, progress, error };
}
