// Liste der gehosteten Modelle, live ueber "hosting-status".

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { DesktopStatus, HostInfo, HostingSettings } from './hostingModel';

export function useHosting() {
  const [hosts, setHosts] = useState<HostInfo[]>([]);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setHosts(await invoke<HostInfo[]>('hosting_list'));
    } catch { /* ohne Backend (Tests) leer lassen */ }
    setLoaded(true);
  }, []);

  useEffect(() => {
    void refresh();
    let un: (() => void) | undefined;
    let disposed = false;
    listen<HostInfo & { removed?: boolean }>('hosting-status', e => {
      const info = e.payload;
      if (info.removed) { void refresh(); return; }
      setHosts(list => {
        const i = list.findIndex(h => h.id === info.id);
        if (i < 0) { void refresh(); return list; }
        const next = list.slice();
        next[i] = info;
        // Nur ein Standard-Modell: die anderen folgen beim naechsten Event, hier gleich angleichen.
        if (info.is_default) return next.map(h => (h.id === info.id ? h : { ...h, is_default: false }));
        return next;
      });
    }).then(fn => { if (disposed) fn(); else un = fn; }).catch(() => {});
    return () => { disposed = true; un?.(); };
  }, [refresh]);

  return { hosts, loaded, refresh };
}

export function useHostingSettings() {
  const [settings, setSettings] = useState<HostingSettings | null>(null);
  const [status, setStatus] = useState<DesktopStatus | null>(null);

  const reloadStatus = useCallback(async () => {
    try { setStatus(await invoke<DesktopStatus>('hosting_desktop_status')); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    invoke<HostingSettings>('hosting_get_settings').then(setSettings).catch(() => {});
    void reloadStatus();
  }, [reloadStatus]);

  const save = useCallback(async (patch: Partial<HostingSettings>) => {
    if (!settings) return;
    const next = { ...settings, ...patch };
    setSettings(next);
    try {
      setSettings(await invoke<HostingSettings>('hosting_save_settings', { settings: next }));
    } catch { /* Zustand bleibt lokal, Status zeigt den Fehler */ }
    await reloadStatus();
  }, [settings, reloadStatus]);

  const rotateToken = useCallback(async () => {
    try {
      const token = await invoke<string>('hosting_rotate_token');
      setSettings(s => (s ? { ...s, api_token: token } : s));
    } catch { /* ignore */ }
  }, []);

  return { settings, status, save, rotateToken, reloadStatus };
}
