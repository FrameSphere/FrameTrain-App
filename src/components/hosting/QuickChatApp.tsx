// Schnell-Chat: eigenes, rahmenloses Glasfenster ("quickchat"). Oeffnet per
// Doppeltipp, Tastenkuerzel oder Tray-Symbol ueber jeder App.
//
// Aufbau: eine Glasflaeche (macOS Vibrancy / Windows Acrylic, sonst deckend),
// oben die Eingabe-Kapsel, darunter Chips bzw. die Antwortkarte.
//
// Verhalten:
// • Jedes Oeffnen beginnt neu — ausser das Fenster war nur kurz zu, eine
//   Antwort laeuft noch oder wurde im Hintergrund fertig (sessionAction).
// • Der vorige Chat ist per Chip oder ⌘↑ sofort wieder da; jede Runde landet
//   zusaetzlich im Verlauf des Modells auf der Hosting-Seite.
// • Esc in Stufen: Modellliste zu → Entwurf leeren → Fenster zu.
// • Tab wechselt das Modell, ⌘N neuer Chat, ⌘C ohne Markierung kopiert die Antwort.
// • Passt die Eingabe nicht zum Modell, wird das passende vorgeschlagen.
//
// Bewegung: per Tastatur geoeffnet → keine Animation (wird dutzendfach am Tag
// ausgeloest). Nur Antworten blenden kurz ein, Knoepfe geben beim Druecken nach.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { ChevronDown, Server, ArrowUpRight, Plus, Check, History, ArrowLeftRight, Lightbulb, X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { Composer, MessageList, requestAnswer, type ComposerSeed } from './HostChat';
import { StatusPill } from './HostingPanel';
import { useHosting } from './useHosting';
import { blurGuarded } from './quickchatGuard';
import {
  agoParts, composerSpec, cycleHost, hostAccepting, newId, sessionAction, taskKey,
  type Attachment, type ChatMsg, type DesktopStatus, type HostInfo, type HostingSettings, type QuickPolicy,
} from './hostingModel';

const IS_MAC = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent);
const IS_WIN = typeof navigator !== 'undefined' && /Win/i.test(navigator.platform || navigator.userAgent);

function Keycap({ children }: { children: React.ReactNode }) {
  return <kbd className="px-1.5 py-px rounded-md bg-white/15 text-[10px] font-sans text-white/80 leading-4">{children}</kbd>;
}

function Chip({ icon: Icon, children, keys, onClick, title }: {
  icon: typeof History; children: React.ReactNode; keys?: string; onClick: () => void; title?: string;
}) {
  return (
    <button type="button" onClick={onClick} title={title}
      className="ft-press inline-flex items-center gap-1.5 max-w-full pl-2.5 pr-2 py-1 rounded-full bg-white/10 hover:bg-white/[0.16] text-[12px] text-white/90">
      <Icon className="w-3.5 h-3.5 text-white/60 flex-shrink-0" />
      <span className="truncate">{children}</span>
      {keys && <Keycap>{keys}</Keycap>}
    </button>
  );
}

function ModelButton({ host, open, onToggle }: { host: HostInfo | null; open: boolean; onToggle: () => void }) {
  if (!host) return null;
  const dot = host.status === 'ready' ? 'bg-emerald-400' : host.status === 'error' ? 'bg-red-400' : host.status === 'loading' ? 'bg-amber-400' : 'bg-white/40';
  return (
    <button type="button" onClick={onToggle} aria-expanded={open}
      className={`ft-press flex-shrink-0 flex items-center gap-1.5 max-w-[180px] h-8 pl-3 pr-2 rounded-full text-[12px] text-white ${open ? 'bg-white/25' : 'bg-white/[0.13] hover:bg-white/20'}`}>
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
      <span className="truncate">{host.name}</span>
      <ChevronDown className={`w-3.5 h-3.5 text-white/55 flex-shrink-0 ${open ? 'rotate-180' : ''}`} />
    </button>
  );
}

/** Modellliste im Fluss statt als Popup — ein Popup wuerde am Fensterrand abgeschnitten. */
function ModelList({ hosts, host, onPick }: { hosts: HostInfo[]; host: HostInfo | null; onPick: (id: string) => void }) {
  const { t } = useLanguage();
  return (
    <div className="mx-1.5 mb-1.5 rounded-[20px] bg-white/[0.07] py-1 max-h-64 overflow-y-auto">
      {hosts.map(h => (
        <button key={h.id} type="button" onClick={() => onPick(h.id)}
          className="w-full flex items-center justify-between gap-2 px-3.5 py-2 text-left hover:bg-white/[0.08]">
          <div className="min-w-0">
            <p className="text-[13px] text-white truncate">{h.name}</p>
            <p className="text-[11px] text-white/50 truncate">{h.modality ? t(`hosting.task.${taskKey(h)}`, h.modality) : t('hosting.page.notLoadedYet')}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <StatusPill status={h.status} busy={h.busy} />
            {h.id === host?.id && <Check className="w-3.5 h-3.5 text-white/80" />}
          </div>
        </button>
      ))}
    </div>
  );
}

interface LastChat { hostId: string; messages: ChatMsg[]; at: number }

/** Abgelegter Chat; die Zeit ist die der letzten Nachricht ("vor 12 min"), nicht die des Ablegens. */
const archive = (hostId: string, messages: ChatMsg[]): LastChat => ({ hostId, messages, at: messages[messages.length - 1]?.at ?? Date.now() });
interface Suggestion { host: HostInfo; file?: Attachment; text?: string }

export default function QuickChatApp() {
  const { t } = useLanguage();
  const { hosts, loaded } = useHosting();
  const [pickedId, setPickedId] = useState<string | null>(null);
  const [focusSignal, setFocusSignal] = useState(1);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [sending, setSending] = useState(false);
  const [lastChat, setLastChat] = useState<LastChat | null>(null);
  /** Antwort wurde fertig, waehrend das Fenster zu war (Zeitpunkt). */
  const [doneWhileHidden, setDoneWhileHidden] = useState<number | null>(null);
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
  const [seed, setSeed] = useState<ComposerSeed>({ nonce: 0 });
  const [glass, setGlass] = useState(IS_MAC || IS_WIN);
  const [now, setNow] = useState(Date.now());
  const rootRef = useRef<HTMLDivElement>(null);

  const host = hosts.find(h => h.id === pickedId) ?? hosts.find(h => h.is_default) ?? hosts.find(h => h.status === 'ready') ?? hosts[0] ?? null;

  // Aktueller Stand fuer Ereignis-Handler, die nicht bei jeder Aenderung neu gebunden werden.
  const live = useRef({ messages, sending, host, hosts, lastChat, pickerOpen, doneWhileHidden });
  live.current = { messages, sending, host, hosts, lastChat, pickerOpen, doneWhileHidden };
  const visible = useRef(false);
  const hiddenAt = useRef<number | null>(null);
  const policy = useRef<QuickPolicy>('smart');
  const draft = useRef<{ text: string; file: Attachment | null }>({ text: '', file: null });

  // Durchsichtiges Fenster: kein Seitenhintergrund
  useLayoutEffect(() => {
    for (const el of [document.documentElement, document.body]) {
      el.style.background = 'transparent';
      el.style.overflow = 'hidden';
    }
  }, []);

  useEffect(() => {
    invoke<DesktopStatus>('hosting_desktop_status').then(s => { if (s && typeof s.glass === 'boolean') setGlass(s.glass); }).catch(() => {});
  }, []);

  // Fensterhoehe folgt dem Inhalt
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const report = () => { void invoke('hosting_quickchat_resize', { height: Math.ceil(el.getBoundingClientRect().height) }).catch(() => {}); };
    report();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(report);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const resetDraft = useCallback((text = '', file: Attachment | null = null) => {
    setSeed(s => ({ nonce: s.nonce + 1, text, file }));
  }, []);

  /** Aktuellen Chat ablegen und leer beginnen. */
  const startNew = useCallback(() => {
    const cur = live.current;
    if (cur.sending) return;
    if (cur.messages.length && cur.host) setLastChat(archive(cur.host.id, cur.messages));
    setMessages([]);
    setDoneWhileHidden(null);
    setSuggestion(null);
    resetDraft();
    setFocusSignal(n => n + 1);
  }, [resetDraft]);

  const restoreLast = useCallback(() => {
    const cur = live.current;
    if (cur.sending || !cur.lastChat) return;
    const back = cur.lastChat;
    setLastChat(cur.messages.length && cur.host ? archive(cur.host.id, cur.messages) : null);
    setPickedId(back.hostId);
    setMessages(back.messages);
    setSuggestion(null);
    setFocusSignal(n => n + 1);
  }, []);

  // Oeffnen / Schliessen
  useEffect(() => {
    const offs: (() => void)[] = [];
    let disposed = false;
    const keep = (p: Promise<() => void>) => p.then(fn => { if (disposed) fn(); else offs.push(fn); }).catch(() => {});

    keep(listen('quickchat-shown', async () => {
      const cur = live.current;
      const wasVisible = visible.current;
      visible.current = true;
      setNow(Date.now());
      setPickerOpen(false);
      void invoke('hosting_set_tray_badge', { on: false }).catch(() => {});
      try {
        const s = await invoke<HostingSettings>('hosting_get_settings');
        policy.current = s?.quick_session ?? 'smart';
      } catch { /* Standard behalten */ }
      // Nur beim echten Wieder-Oeffnen entscheiden (nicht nach Datei-Dialog/Bildschirmfoto).
      if (!wasVisible) {
        const action = sessionAction({
          policy: policy.current,
          pending: cur.sending,
          unread: cur.doneWhileHidden !== null,
          hiddenForMs: hiddenAt.current === null ? null : Date.now() - hiddenAt.current,
          hasMessages: cur.messages.length > 0,
        });
        if (action === 'new') startNew();
      }
      setFocusSignal(n => n + 1);
    }));
    keep(listen('quickchat-hidden', () => {
      visible.current = false;
      hiddenAt.current = Date.now();
    }));
    keep(getCurrentWindow().onFocusChanged(({ payload: focused }) => {
      if (!focused && !blurGuarded()) void invoke('hosting_quickchat_blur').catch(() => {});
    }));
    return () => { disposed = true; offs.forEach(f => f()); };
  }, [startNew]);

  const switchHost = useCallback((id: string, opts: { makeDefault?: boolean } = {}) => {
    const cur = live.current;
    if (cur.sending || id === cur.host?.id) return;
    // Ein Chat gehoert zu einem Modell: Wechsel mit Verlauf beginnt einen neuen.
    if (cur.messages.length && cur.host) {
      setLastChat(archive(cur.host.id, cur.messages));
      setMessages([]);
      setDoneWhileHidden(null);
    }
    setPickedId(id);
    setPickerOpen(false);
    setSuggestion(null);
    if (opts.makeDefault) void invoke('hosting_update_model', { id, makeDefault: true }).catch(() => {});
    setFocusSignal(n => n + 1);
  }, []);

  const send = useCallback(async (text: string, file: Attachment | null) => {
    const cur = live.current;
    const h = cur.host;
    if (!h || cur.sending) return;
    const user: ChatMsg = { id: newId(), role: 'user', text, file: file ?? undefined, at: Date.now(), source: 'quick' };
    const answer: ChatMsg = { id: newId(), role: 'assistant', text: '', pending: true, at: Date.now(), source: 'quick' };
    const before = cur.messages;
    setMessages([...before, user, answer]);
    setSending(true);
    setDoneWhileHidden(null);
    setSuggestion(null);
    const out = await requestAnswer(h, answer.id, text, file, before, piece => {
      setMessages(list => list.map(m => (m.id === answer.id ? { ...m, text: m.text + piece } : m)));
    });
    const done: ChatMsg = 'result' in out
      ? { ...answer, pending: false, text: out.result.predicted, result: out.result, at: Date.now() }
      : { ...answer, pending: false, error: out.error, at: Date.now() };
    setMessages(list => list.map(m => (m.id === answer.id ? { ...done, text: done.text || m.text } : m)));
    setSending(false);
    // Runde im Verlauf des Modells sichern (Hosting-Seite zeigt sie mit Markierung).
    void invoke('hosting_chat_append', { id: h.id, messages: [user, done] }).catch(() => {});
    if (!visible.current) {
      setDoneWhileHidden(Date.now());
      void invoke('hosting_set_tray_badge', { on: true }).catch(() => {});
    }
  }, []);

  // Tastatur: Esc in Stufen, ⌘N, ⌘↑, ⌘C
  useEffect(() => {
    const onKeyCapture = (e: KeyboardEvent) => {
      // Stufe 1: offene Modellliste schliessen (vor dem Eingabefeld, das den Entwurf leeren wuerde).
      if (e.key === 'Escape' && live.current.pickerOpen) {
        e.preventDefault();
        e.stopPropagation();
        setPickerOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (e.key === 'Escape') {
        // Stufe 3 (Stufe 2 erledigt das Eingabefeld und stoppt das Ereignis).
        e.preventDefault();
        void invoke('hosting_hide_quickchat').catch(() => {});
      } else if (mod && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        startNew();
      } else if (mod && e.key === 'ArrowUp') {
        e.preventDefault();
        restoreLast();
      } else if (mod && e.key.toLowerCase() === 'c' && !window.getSelection()?.toString()) {
        const el = document.activeElement as HTMLTextAreaElement | null;
        const hasInputSelection = !!el && 'selectionStart' in el && el.selectionStart !== el.selectionEnd;
        const answer = [...live.current.messages].reverse().find(m => m.role === 'assistant' && !m.pending && !m.error);
        if (!hasInputSelection && answer) {
          e.preventDefault();
          void navigator.clipboard?.writeText(answer.result?.predicted ?? answer.text);
        }
      }
    };
    window.addEventListener('keydown', onKeyCapture, true);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKeyCapture, true);
      window.removeEventListener('keydown', onKey);
    };
  }, [startNew, restoreLast]);

  // Eingabe passt nicht zum Modell → passendes gehostetes Modell vorschlagen
  const onMismatch = useCallback((kind: 'image' | 'audio' | 'video', file: Attachment) => {
    const cur = live.current;
    const other = hostAccepting(cur.hosts, kind, cur.host?.id ?? null);
    if (!other) return false;
    setSuggestion({ host: other, file });
    return true;
  }, []);

  const onDraftChange = useCallback((d: { text: string; file: Attachment | null }) => {
    draft.current = d;
    const cur = live.current;
    const spec = composerSpec(cur.host);
    // Text getippt, aber das Modell nimmt nur Dateien: Textmodell anbieten.
    if (spec.text === 'none' && d.text.trim().length >= 3 && !d.file) {
      const other = hostAccepting(cur.hosts, 'text', cur.host?.id ?? null);
      setSuggestion(s => (other ? (s?.file ? s : { host: other, text: d.text }) : s));
    } else {
      setSuggestion(s => (s && !s.file ? null : s));
    }
  }, []);

  const acceptSuggestion = useCallback(() => {
    const sug = suggestion;
    if (!sug) return;
    switchHost(sug.host.id);
    // Entwurf mitnehmen; der Anhang wird nach dem Modellwechsel gesetzt.
    resetDraft(sug.text ?? (sug.file ? '' : draft.current.text), sug.file ?? null);
  }, [suggestion, switchHost, resetDraft]);

  const cycle = useCallback((dir: 1 | -1) => {
    if (suggestion) { acceptSuggestion(); return; }
    const cur = live.current;
    const next = cycleHost(cur.hosts, cur.host?.id ?? null, dir);
    if (next) switchHost(next);
  }, [suggestion, acceptSuggestion, switchHost]);

  const openMain = () => { void invoke('hosting_open_main', { view: 'hosting' }).catch(() => {}); };
  const ago = (ms: number) => { const a = agoParts(ms); return t(`hosting.quick.ago.${a.key}`, { value: a.value }); };

  const empty = messages.length === 0;
  const lastHost = lastChat ? hosts.find(h => h.id === lastChat.hostId) : null;
  const showChips = !pickerOpen && (!!suggestion || (empty && (!!lastHost || hosts.length > 1)));
  const mod = IS_MAC ? '⌘' : 'Ctrl+';

  return (
    <div ref={rootRef}
      className={`relative overflow-hidden rounded-[26px] text-white select-none ${glass ? 'bg-[rgba(16,14,26,0.46)]' : 'bg-slate-900 border border-white/15'}`}
      style={{ boxShadow: 'inset 0 0 0 0.5px rgba(255,255,255,0.28), inset 0 1px 0 0 rgba(255,255,255,0.16)' }}>
      {loaded && hosts.length === 0 ? (
        <div className="flex items-center gap-3 pl-4 pr-1.5 min-h-[52px]">
          <Server className="w-5 h-5 text-white/55 flex-shrink-0" />
          <p className="flex-1 text-[14px] text-white/80">{t('hosting.quick.noModels')}</p>
          <button onClick={openMain} className="ft-press h-9 px-4 rounded-full text-[13px] font-medium bg-white text-slate-900">{t('hosting.page.add')}</button>
        </div>
      ) : (
        <>
          <div className="p-[5px]">
            <Composer host={host} sending={sending} onSend={send} compact autoFocus focusSignal={focusSignal}
              onCycle={cycle} onMismatch={onMismatch} onDraftChange={onDraftChange} seed={seed}
              leading={<ModelButton host={host} open={pickerOpen} onToggle={() => setPickerOpen(o => !o)} />} />
          </div>

          {pickerOpen && <ModelList hosts={hosts} host={host} onPick={id => switchHost(id, { makeDefault: true })} />}

          {showChips && (
            <div data-tauri-drag-region className="flex flex-wrap items-center gap-1.5 px-2.5 pb-2.5">
              {suggestion && (
                <>
                  <Chip icon={Lightbulb} keys="Tab" onClick={acceptSuggestion}>
                    {suggestion.file
                      ? t('hosting.quick.suggestFile', { kind: t(`hosting.composer.kind.${suggestion.file.kind}`), name: suggestion.host.name })
                      : t('hosting.quick.suggestText', { name: suggestion.host.name })}
                  </Chip>
                  <button type="button" onClick={() => setSuggestion(null)} aria-label={t('hosting.quick.dismiss')}
                    className="ft-press w-6 h-6 flex items-center justify-center rounded-full text-white/50 hover:text-white hover:bg-white/10">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
              {!suggestion && empty && lastChat && lastHost && (
                <Chip icon={History} keys={`${mod}↑`} onClick={restoreLast} title={lastHost.name}>
                  {t('hosting.quick.lastChat', { ago: ago(now - lastChat.at) })}
                </Chip>
              )}
              {!suggestion && empty && hosts.length > 1 && (
                <Chip icon={ArrowLeftRight} keys="Tab" onClick={() => cycle(1)}>{t('hosting.quick.switchModel')}</Chip>
              )}
            </div>
          )}

          {!empty && !pickerOpen && (
            <div className="mx-1.5 mb-1.5 rounded-[20px] bg-white/[0.07]">
              {doneWhileHidden !== null && (
                <div className="flex items-center gap-1.5 px-4 pt-3 text-[11px] text-white/60">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-300" />
                  {t('hosting.quick.finishedWhileAway', { ago: ago(now - doneWhileHidden) })}
                </div>
              )}
              <div className="px-4 py-3 max-h-[420px] overflow-y-auto select-text">
                <MessageList messages={messages} host={host} compact plainUser />
              </div>
              <div data-tauri-drag-region className="flex items-center justify-end gap-1.5 px-2.5 pb-2.5">
                <Chip icon={Plus} keys={`${mod}N`} onClick={startNew}>{t('hosting.quick.new')}</Chip>
                <Chip icon={ArrowUpRight} onClick={openMain}>{t('hosting.quick.openApp')}</Chip>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
