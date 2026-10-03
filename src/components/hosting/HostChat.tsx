// Chat mit einem gehosteten Modell — gemeinsam fuer die Hosting-Seite und das
// Schnell-Chat-Fenster. Der Eingabebereich richtet sich nach dem Modell:
// Text, Bild (Datei, Einfuegen, Ziehen, Bildschirmfoto), Audio (Datei,
// Mikrofon) oder Video. LLM-Antworten kommen Token fuer Token.

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ClipboardEvent, type KeyboardEvent } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { ArrowUp, Loader2, Mic, Paperclip, ScanLine, Square, X, FileAudio, FileVideo, File as FileIcon, AlertTriangle } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { toWav16kMono } from '../studio/wavEncode';
import HostResultView from './HostResultView';
import {
  baseName, canSend, composerSpec, fileKindOf, historyFor, newId,
  type Attachment, type ChatMsg, type HostInfo, type InferResult,
} from './hostingModel';
import { guardBlur } from './quickchatGuard';

const MOD = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent) ? '⌘' : 'Ctrl+';

// ── Anfrage ──────────────────────────────────────────────────────────────

/**
 * Eine Anfrage an ein gehostetes Modell. `before` ist der bisherige Verlauf
 * (nur LLMs nutzen ihn), `onToken` bekommt beim Streaming jedes Textstueck.
 */
export async function requestAnswer(
  host: HostInfo,
  requestId: string,
  text: string,
  file: Attachment | null,
  before: ChatMsg[],
  onToken: (piece: string) => void,
): Promise<{ result: InferResult } | { error: string }> {
  const spec = composerSpec(host);
  let unlisten: (() => void) | undefined;
  if (spec.chat) {
    unlisten = await listen<{ request_id: string; text: string }>('hosting-token', e => {
      if (e.payload.request_id === requestId) onToken(e.payload.text);
    }).catch(() => undefined);
  }
  try {
    const result = await invoke<InferResult>('hosting_infer', {
      id: host.id,
      requestId,
      input: {
        text,
        file_path: file?.path ?? null,
        question: spec.text === 'optional' && text.trim() ? text : null,
        messages: spec.chat ? historyFor(before) : null,
        stream: spec.chat,
      },
    });
    return { result };
  } catch (e) {
    return { error: String(e) };
  } finally {
    unlisten?.();
  }
}

// ── Zustand ──────────────────────────────────────────────────────────────

export function useHostChat(host: HostInfo | null, persist: boolean) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [sending, setSending] = useState(false);
  const hostId = host?.id ?? null;
  const loadedFor = useRef<string | null>(null);
  const sendingRef = useRef(false);

  const load = useCallback(async (id: string, alive: () => boolean) => {
    try {
      const list = await invoke<ChatMsg[]>('hosting_chat_load', { id });
      if (alive()) setMessages(Array.isArray(list) ? list.filter(m => !m.pending) : []);
    } catch { /* leer lassen */ }
    if (alive()) loadedFor.current = id;
  }, []);

  useEffect(() => {
    setMessages([]);
    loadedFor.current = null;
    if (!hostId || !persist) return;
    let alive = true;
    void load(hostId, () => alive);
    // Der Schnell-Chat haengt seine Runden an denselben Verlauf an.
    let un: (() => void) | undefined;
    listen<{ id: string }>('hosting-chat-changed', e => {
      if (e.payload.id === hostId && !sendingRef.current) void load(hostId, () => alive);
    }).then(fn => { if (alive) un = fn; else fn(); }).catch(() => {});
    return () => { alive = false; un?.(); };
  }, [hostId, persist, load]);

  const save = useCallback((list: ChatMsg[]) => {
    if (!persist || !hostId || loadedFor.current !== hostId) return;
    void invoke('hosting_chat_save', { id: hostId, messages: list.filter(m => !m.pending) }).catch(() => {});
  }, [persist, hostId]);

  const send = useCallback(async (text: string, file: Attachment | null) => {
    if (!host) return;
    const user: ChatMsg = { id: newId(), role: 'user', text, file: file ?? undefined, at: Date.now() };
    const answer: ChatMsg = { id: newId(), role: 'assistant', text: '', pending: true, at: Date.now() };
    const before = messages;
    setMessages([...before, user, answer]);
    setSending(true);
    sendingRef.current = true;
    const out = await requestAnswer(host, answer.id, text, file, before, piece => {
      setMessages(list => list.map(m => (m.id === answer.id ? { ...m, text: m.text + piece } : m)));
    });
    setSending(false);
    sendingRef.current = false;
    setMessages(list => {
      const next = list.map(m => {
        if (m.id !== answer.id) return m;
        return 'result' in out
          ? { ...answer, pending: false, text: out.result.predicted || m.text, result: out.result }
          : { ...answer, pending: false, text: m.text, error: out.error };
      });
      save(next);
      return next;
    });
  }, [host, messages, save]);

  const clear = useCallback(() => {
    setMessages([]);
    save([]);
  }, [save]);

  return { messages, sending, send, clear };
}

// ── Nachrichten ──────────────────────────────────────────────────────────

function AttachmentPreview({ file, onRemove, small = false }: { file: Attachment; onRemove?: () => void; small?: boolean }) {
  const Icon = file.kind === 'audio' ? FileAudio : file.kind === 'video' ? FileVideo : FileIcon;
  return (
    <div className={`relative inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 ${small ? 'p-1 pr-2' : 'p-1.5 pr-2.5'} max-w-full`}>
      {file.kind === 'image' ? (
        <img src={convertFileSrc(file.path)} alt="" className={`${small ? 'w-7 h-7' : 'w-10 h-10'} rounded object-cover bg-black/30`} />
      ) : (
        <Icon className="w-4 h-4 text-gray-400 flex-shrink-0" />
      )}
      <span className="text-xs text-gray-300 truncate max-w-[180px]">{file.name}</span>
      {file.kind === 'audio' && !small && <audio src={convertFileSrc(file.path)} controls className="h-7 max-w-[180px]" />}
      {onRemove && (
        <button type="button" onClick={onRemove} className="ml-0.5 text-gray-500 hover:text-white" aria-label="remove">
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

export function MessageList({ messages, host, compact = false, plainUser = false }: {
  messages: ChatMsg[]; host: HostInfo | null; compact?: boolean;
  /** Schnell-Chat: Fragen als gedimmte Zeile statt als Blase. */
  plainUser?: boolean;
}) {
  const { t } = useLanguage();
  const endRef = useRef<HTMLDivElement>(null);
  const last = messages[messages.length - 1];
  useEffect(() => { endRef.current?.scrollIntoView?.({ block: 'end' }); }, [messages.length, last?.text]);

  return (
    <div className={`flex flex-col ${compact ? 'gap-2.5' : 'gap-4'}`}>
      {messages.map((m, i) => {
        if (m.role === 'user' && plainUser) {
          // Zeigt die Antwort das Bild selbst (Erkennung mit Boxen), waere die Vorschau doppelt.
          const next = messages[i + 1];
          const shownInAnswer = m.file?.kind === 'image' && !!next?.result?.boxes?.length && !!next.result.image_width;
          if (shownInAnswer && !m.text) return null;
          return (
            <div key={m.id} className="flex items-center gap-2 min-w-0">
              {m.file && !shownInAnswer && <AttachmentPreview file={m.file} small />}
              {m.text && <p className="text-[12px] text-white/55 truncate">{m.text}</p>}
            </div>
          );
        }
        if (m.role === 'user') {
          return (
            <div key={m.id} className="self-end max-w-[85%] flex flex-col items-end gap-1.5">
              {m.source === 'quick' && <span className="text-[10px] text-gray-500">{t('hosting.chat.fromQuick')}</span>}
              {m.file && <AttachmentPreview file={m.file} small={compact} />}
              {m.text && <div className="px-3.5 py-2 rounded-2xl rounded-br-md bg-white/10 text-sm text-white whitespace-pre-wrap break-words">{m.text}</div>}
            </div>
          );
        }
        const prev = messages[i - 1];
        return (
          <div key={m.id} className="ft-answer self-start w-full max-w-full">
            {m.pending && !m.text && (
              <div className="flex items-center gap-2 text-xs text-gray-400 py-1">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                {host?.status === 'loading' || host?.status === 'sleeping' || host?.status === 'idle'
                  ? t('hosting.chat.waking') : t('hosting.chat.thinking')}
              </div>
            )}
            {m.pending && m.text && (
              <p className="text-sm text-gray-100 whitespace-pre-wrap break-words leading-relaxed">
                {m.text}<span className="inline-block w-1.5 h-4 ml-0.5 align-text-bottom bg-white/70 animate-pulse" />
              </p>
            )}
            {m.error && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span className="whitespace-pre-wrap break-words">{m.error}</span>
              </div>
            )}
            {m.result && <HostResultView result={m.result} host={host} file={prev?.file} inputText={prev?.text ?? ''} compact={compact} />}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

// ── Eingabe ──────────────────────────────────────────────────────────────

export interface ComposerSeed {
  /** Bei jeder Aenderung wird der Entwurf ersetzt. */
  nonce: number;
  text?: string;
  file?: Attachment | null;
}

interface ComposerProps {
  host: HostInfo | null;
  sending: boolean;
  onSend: (text: string, file: Attachment | null) => void;
  /** Schnell-Chat: Glas-Kapsel statt Eingabefeld der Seite. */
  compact?: boolean;
  autoFocus?: boolean;
  /** Links neben dem Eingabefeld (Schnell-Chat: Modellwahl). */
  leading?: React.ReactNode;
  /** Wird bei jedem Oeffnen des Schnell-Chats erhoeht → Fokus ins Feld. */
  focusSignal?: number;
  /** Tab / Umschalt+Tab im Eingabefeld (Schnell-Chat: Modell wechseln). */
  onCycle?: (dir: 1 | -1) => void;
  /**
   * Eingabe passt nicht zum Modell (Bild bei Textmodell …). true = der
   * Aufrufer kuemmert sich (schlaegt ein anderes Modell vor), sonst Hinweis.
   */
  onMismatch?: (kind: Exclude<Attachment['kind'], 'file'>, file: Attachment) => boolean;
  onDraftChange?: (draft: { text: string; file: Attachment | null }) => void;
  seed?: ComposerSeed;
}

/** Datei aus einem Blob (Einfuegen, Aufnahme) in den Cache schreiben. */
async function saveBlob(bytes: Uint8Array, ext: string, name: string, kind: Attachment['kind']): Promise<Attachment> {
  const path = await invoke<string>('hosting_save_upload', { bytes: Array.from(bytes), ext });
  return { path, name, kind };
}

export function Composer({ host, sending, onSend, compact = false, autoFocus = false, leading, focusSignal, onCycle, onMismatch, onDraftChange, seed }: ComposerProps) {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  const spec = composerSpec(host);
  const [text, setText] = useState('');
  const [file, setFile] = useState<Attachment | null>(null);
  const [busy, setBusy] = useState<null | 'capture' | 'record' | 'mic-wait'>(null);
  const [hint, setHint] = useState<string | null>(null);
  /** Die Aufnahme scheiterte an der fehlenden macOS-Freigabe → Knopf zu den Systemeinstellungen. */
  const [needsScreenPermission, setNeedsScreenPermission] = useState(false);
  const [dragging, setDragging] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const ready = canSend(spec, text, file) && !!host && !sending;

  // Modellwechsel: Anhang passt evtl. nicht mehr
  useEffect(() => { setFile(f => (f && spec.file && f.kind === spec.file ? f : null)); setHint(null); }, [host?.id, spec.file]);

  // Entwurf von aussen setzen (neuer Chat, Wechsel zum vorgeschlagenen Modell).
  // Steht nach dem Modellwechsel-Effekt, damit ein mitgegebener Anhang bleibt.
  const seedNonce = seed?.nonce;
  useEffect(() => {
    if (seedNonce === undefined || !seed) return;
    if (seed.text !== undefined) setText(seed.text);
    if (seed.file !== undefined) setFile(seed.file);
    setHint(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seedNonce]);

  useEffect(() => { onDraftChange?.({ text, file }); }, [text, file]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (autoFocus || focusSignal) areaRef.current?.focus(); }, [autoFocus, focusSignal]);

  // Hoehe waechst mit dem Text
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, compact ? 160 : 220)}px`;
  }, [text, compact]);

  const accept = useCallback((att: Attachment) => {
    if (att.kind !== 'file' && att.kind === spec.file) {
      setHint(null);
      setFile(att);
      return;
    }
    if (att.kind !== 'file' && onMismatch?.(att.kind, att)) { setHint(null); return; }
    setHint(spec.file
      ? t('hosting.composer.wrongFile', { kind: t(`hosting.composer.kind.${spec.file}`) })
      : t('hosting.composer.noFileNeeded'));
  }, [spec.file, onMismatch, t]);

  const acceptPath = useCallback((path: string) => accept({ path, name: baseName(path), kind: fileKindOf(path) }), [accept]);

  // Dateien ins Fenster ziehen (echte Pfade ueber Tauri)
  useEffect(() => {
    let un: (() => void) | undefined;
    let disposed = false;
    getCurrentWebview().onDragDropEvent(ev => {
      const p = ev.payload;
      if (p.type === 'over' || p.type === 'enter') setDragging(true);
      else if (p.type === 'leave') setDragging(false);
      else if (p.type === 'drop') { setDragging(false); if (p.paths?.[0]) acceptPath(p.paths[0]); }
    }).then(fn => { if (disposed) fn(); else un = fn; }).catch(() => {});
    return () => { disposed = true; un?.(); };
  }, [acceptPath]);

  const submit = () => {
    if (!ready) return;
    onSend(text.trim(), file);
    setText('');
    setFile(null);
    setHint(null);
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    } else if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 's' && spec.file === 'image') {
      e.preventDefault();
      if (busy === null) void capture();
    } else if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'o' && spec.file) {
      e.preventDefault();
      void pick();
    } else if (e.key === 'Tab' && onCycle && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      onCycle(e.shiftKey ? -1 : 1);
    } else if (e.key === 'Escape' && (text || file || hint)) {
      // Erste Stufe: Entwurf leeren. Erst das naechste Esc schliesst das Fenster.
      e.preventDefault();
      e.stopPropagation();
      setText('');
      setFile(null);
      setHint(null);
    }
  };

  const onPaste = async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const item = Array.from(e.clipboardData.files ?? [])[0];
    if (!item) return;
    const kind = item.type.startsWith('image/') ? 'image' : item.type.startsWith('audio/') ? 'audio' : item.type.startsWith('video/') ? 'video' : 'file';
    if (kind === 'file') return;
    e.preventDefault();
    const ext = (item.name.split('.').pop() || item.type.split('/')[1] || 'png').toLowerCase();
    try {
      const bytes = new Uint8Array(await item.arrayBuffer());
      accept(await saveBlob(bytes, ext, item.name || `${t('hosting.composer.pasted')}.${ext}`, kind));
    } catch (err) { setHint(String(err)); }
  };

  const pick = async () => {
    const release = guardBlur();
    try {
      const sel = await openDialog({ multiple: false, filters: spec.extensions.length ? [{ name: t(`hosting.composer.kind.${spec.file ?? 'image'}`), extensions: spec.extensions }] : undefined });
      if (typeof sel === 'string') acceptPath(sel);
    } catch (err) { setHint(String(err)); }
    finally {
      release();
      void getCurrentWindow().setFocus().catch(() => {});
      areaRef.current?.focus();
    }
  };

  const capture = async () => {
    setBusy('capture');
    setNeedsScreenPermission(false);
    const release = guardBlur();
    try {
      const path = await invoke<string | null>('hosting_capture_screenshot');
      if (path) acceptPath(path);
    } catch (err) {
      if (String(err).includes('SCREEN_PERMISSION')) {
        setNeedsScreenPermission(true);
        setHint(t('hosting.composer.screenPermission'));
      } else setHint(String(err));
    }
    finally { release(); setBusy(null); areaRef.current?.focus(); }
  };

  const toggleRecord = async () => {
    if (recorder.current) { recorder.current.stop(); return; }
    setBusy('mic-wait');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = ev => { if (ev.data.size > 0) chunks.push(ev.data); };
      rec.onstop = async () => {
        stream.getTracks().forEach(tr => tr.stop());
        recorder.current = null;
        setBusy(null);
        const blob = new Blob(chunks, { type: rec.mimeType || 'audio/mp4' });
        try {
          const wav = await toWav16kMono(await blob.arrayBuffer());
          accept(await saveBlob(wav, 'wav', `${t('hosting.composer.recording')}.wav`, 'audio'));
        } catch (err) { setHint(String(err)); }
      };
      rec.start();
      recorder.current = rec;
      setBusy('record');
    } catch (err) {
      setBusy(null);
      setHint(t('hosting.composer.micError', { detail: String(err) }));
    }
  };

  // Schnell-Chat: runde Glasknoepfe; Seite: dezente Quadrate.
  const iconBtn = compact
    ? 'ft-press w-8 h-8 flex items-center justify-center rounded-full text-white/70 hover:text-white hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0'
    : 'ft-press p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed';
  const sendBtn = compact
    ? `ft-press w-8 h-8 flex items-center justify-center rounded-full flex-shrink-0 ${ready ? 'bg-white text-slate-900' : 'bg-white/15 text-white/45'}`
    : `ft-press p-1.5 rounded-lg ${ready ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white` : 'bg-white/5 text-gray-600'}`;
  const placeholder = host
    ? t(`hosting.composer.placeholder.${spec.placeholder}`, { name: host.name })
    : t('hosting.composer.noModel');

  return (
    <div className={`relative ${dragging ? (compact ? 'ring-2 ring-white/40 rounded-[21px]' : 'ring-2 ring-white/30 rounded-2xl') : ''}`}>
      {file && (
        <div className={compact ? 'px-2 pt-1.5' : 'mb-2'}>
          <AttachmentPreview file={file} onRemove={() => setFile(null)} small={compact} />
        </div>
      )}
      <div className={`flex gap-1.5 ${compact ? 'items-center min-h-[42px] pl-1.5 pr-1.5' : 'items-end rounded-2xl border border-white/15 bg-black/20 px-2.5 py-2 focus-within:border-white/30'}`}>
        {leading}
        <textarea
          ref={areaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={onPaste}
          rows={1}
          disabled={!host}
          placeholder={placeholder}
          className={`flex-1 resize-none bg-transparent outline-none text-white ${compact ? 'text-[15px] py-2 placeholder-white/45' : 'text-sm py-1.5 placeholder-gray-500'} leading-snug min-w-0`}
        />
        {spec.file === 'image' && (
          <button type="button" className={iconBtn} onClick={capture} disabled={!host || busy !== null} title={`${t('hosting.composer.screenshot')} (${MOD}⇧S)`} aria-label={t('hosting.composer.screenshot')}>
            {busy === 'capture' ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <ScanLine className="w-[18px] h-[18px]" />}
          </button>
        )}
        {spec.file === 'audio' && (
          <button type="button" className={`${iconBtn} ${busy === 'record' ? '!text-red-400 bg-red-500/10' : ''}`} onClick={toggleRecord} disabled={!host || busy === 'capture' || busy === 'mic-wait'} title={t(busy === 'record' ? 'hosting.composer.stopRecording' : 'hosting.composer.record')} aria-label={t('hosting.composer.record')}>
            {busy === 'record' ? <Square className="w-[18px] h-[18px]" /> : busy === 'mic-wait' ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <Mic className="w-[18px] h-[18px]" />}
          </button>
        )}
        {spec.file && (
          <button type="button" className={iconBtn} onClick={pick} disabled={!host} title={`${t('hosting.composer.attach')} (${MOD}O)`} aria-label={t('hosting.composer.attach')}>
            <Paperclip className="w-[18px] h-[18px]" />
          </button>
        )}
        <button type="button" onClick={submit} disabled={!ready} aria-label={t('hosting.composer.send')} className={sendBtn}>
          {sending ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <ArrowUp className="w-[18px] h-[18px]" strokeWidth={compact ? 2.5 : 2} />}
        </button>
      </div>
      {hint && (
        <p className={`text-xs text-amber-300 select-text ${compact ? 'px-3 pb-1.5' : 'mt-1.5'}`}>
          {hint}
          {needsScreenPermission && (
            <button type="button" onClick={() => { void invoke('hosting_open_screen_permission').catch(() => {}); }}
              className="ml-2 underline underline-offset-2 text-white hover:text-amber-100">
              {t('hosting.composer.openSettings')}
            </button>
          )}
        </p>
      )}
    </div>
  );
}
