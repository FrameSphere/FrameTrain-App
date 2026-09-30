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

// ── Zustand ──────────────────────────────────────────────────────────────

export function useHostChat(host: HostInfo | null, persist: boolean) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [sending, setSending] = useState(false);
  const hostId = host?.id ?? null;
  const loadedFor = useRef<string | null>(null);

  useEffect(() => {
    setMessages([]);
    loadedFor.current = null;
    if (!hostId || !persist) return;
    let alive = true;
    invoke<ChatMsg[]>('hosting_chat_load', { id: hostId })
      .then(list => { if (alive) { setMessages(Array.isArray(list) ? list.filter(m => !m.pending) : []); loadedFor.current = hostId; } })
      .catch(() => { loadedFor.current = hostId; });
    return () => { alive = false; };
  }, [hostId, persist]);

  const save = useCallback((list: ChatMsg[]) => {
    if (!persist || !hostId || loadedFor.current !== hostId) return;
    void invoke('hosting_chat_save', { id: hostId, messages: list.filter(m => !m.pending) }).catch(() => {});
  }, [persist, hostId]);

  const send = useCallback(async (text: string, file: Attachment | null) => {
    if (!host) return;
    const spec = composerSpec(host);
    const user: ChatMsg = { id: newId(), role: 'user', text, file: file ?? undefined, at: Date.now() };
    const answer: ChatMsg = { id: newId(), role: 'assistant', text: '', pending: true, at: Date.now() };
    const before = messages;
    setMessages([...before, user, answer]);
    setSending(true);

    const requestId = answer.id;
    let unlisten: (() => void) | undefined;
    if (spec.chat) {
      unlisten = await listen<{ request_id: string; text: string }>('hosting-token', e => {
        if (e.payload.request_id !== requestId) return;
        setMessages(list => list.map(m => (m.id === requestId ? { ...m, text: m.text + e.payload.text } : m)));
      }).catch(() => undefined);
    }
    let finished: ChatMsg;
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
      finished = { ...answer, pending: false, text: result.predicted, result };
    } catch (e) {
      finished = { ...answer, pending: false, error: String(e) };
    } finally {
      unlisten?.();
      setSending(false);
    }
    setMessages(list => {
      const next = list.map(m => (m.id === requestId ? { ...finished, text: finished.text || m.text } : m));
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

export function MessageList({ messages, host, compact = false }: { messages: ChatMsg[]; host: HostInfo | null; compact?: boolean }) {
  const { t } = useLanguage();
  const endRef = useRef<HTMLDivElement>(null);
  const last = messages[messages.length - 1];
  useEffect(() => { endRef.current?.scrollIntoView?.({ block: 'end' }); }, [messages.length, last?.text]);

  return (
    <div className={`flex flex-col ${compact ? 'gap-2.5' : 'gap-4'}`}>
      {messages.map((m, i) => {
        if (m.role === 'user') {
          return (
            <div key={m.id} className="self-end max-w-[85%] flex flex-col items-end gap-1.5">
              {m.file && <AttachmentPreview file={m.file} small={compact} />}
              {m.text && <div className="px-3.5 py-2 rounded-2xl rounded-br-md bg-white/10 text-sm text-white whitespace-pre-wrap break-words">{m.text}</div>}
            </div>
          );
        }
        const prev = messages[i - 1];
        return (
          <div key={m.id} className="self-start w-full max-w-full">
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
            {m.result && <HostResultView result={m.result} host={host} file={prev?.file} inputText={prev?.text ?? ''} />}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

// ── Eingabe ──────────────────────────────────────────────────────────────

interface ComposerProps {
  host: HostInfo | null;
  sending: boolean;
  onSend: (text: string, file: Attachment | null) => void;
  compact?: boolean;
  autoFocus?: boolean;
  /** Links neben dem Eingabefeld (Schnell-Chat: Modellwahl). */
  leading?: React.ReactNode;
  /** Wird bei jedem Oeffnen des Schnell-Chats erhoeht → Fokus ins Feld. */
  focusSignal?: number;
}

/** Datei aus einem Blob (Einfuegen, Aufnahme) in den Cache schreiben. */
async function saveBlob(bytes: Uint8Array, ext: string, name: string, kind: Attachment['kind']): Promise<Attachment> {
  const path = await invoke<string>('hosting_save_upload', { bytes: Array.from(bytes), ext });
  return { path, name, kind };
}

export function Composer({ host, sending, onSend, compact = false, autoFocus = false, leading, focusSignal }: ComposerProps) {
  const { t } = useLanguage();
  const { currentTheme } = useTheme();
  const spec = composerSpec(host);
  const [text, setText] = useState('');
  const [file, setFile] = useState<Attachment | null>(null);
  const [busy, setBusy] = useState<null | 'capture' | 'record' | 'mic-wait'>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const ready = canSend(spec, text, file) && !!host && !sending;

  // Modellwechsel: Anhang passt evtl. nicht mehr
  useEffect(() => { setFile(f => (f && spec.file && f.kind === spec.file ? f : null)); setHint(null); }, [host?.id, spec.file]);

  useEffect(() => { if (autoFocus || focusSignal) areaRef.current?.focus(); }, [autoFocus, focusSignal]);

  // Hoehe waechst mit dem Text
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, compact ? 160 : 220)}px`;
  }, [text, compact]);

  const accept = useCallback((path: string) => {
    const kind = fileKindOf(path);
    if (!spec.file) { setHint(t('hosting.composer.noFileNeeded')); return; }
    if (kind !== spec.file) { setHint(t('hosting.composer.wrongFile', { kind: t(`hosting.composer.kind.${spec.file}`) })); return; }
    setHint(null);
    setFile({ path, name: baseName(path), kind });
  }, [spec.file, t]);

  // Dateien ins Fenster ziehen (echte Pfade ueber Tauri)
  useEffect(() => {
    let un: (() => void) | undefined;
    let disposed = false;
    getCurrentWebview().onDragDropEvent(ev => {
      const p = ev.payload;
      if (p.type === 'over' || p.type === 'enter') setDragging(true);
      else if (p.type === 'leave') setDragging(false);
      else if (p.type === 'drop') { setDragging(false); if (p.paths?.[0]) accept(p.paths[0]); }
    }).then(fn => { if (disposed) fn(); else un = fn; }).catch(() => {});
    return () => { disposed = true; un?.(); };
  }, [accept]);

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
    }
  };

  const onPaste = async (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const item = Array.from(e.clipboardData.files ?? [])[0];
    if (!item) return;
    const kind = item.type.startsWith('image/') ? 'image' : item.type.startsWith('audio/') ? 'audio' : item.type.startsWith('video/') ? 'video' : 'file';
    if (!spec.file || kind !== spec.file) return;
    e.preventDefault();
    const ext = (item.name.split('.').pop() || item.type.split('/')[1] || 'png').toLowerCase();
    try {
      const bytes = new Uint8Array(await item.arrayBuffer());
      setFile(await saveBlob(bytes, ext, item.name || `${t('hosting.composer.pasted')}.${ext}`, kind));
    } catch (err) { setHint(String(err)); }
  };

  const pick = async () => {
    const release = guardBlur();
    try {
      const sel = await openDialog({ multiple: false, filters: spec.extensions.length ? [{ name: t(`hosting.composer.kind.${spec.file ?? 'image'}`), extensions: spec.extensions }] : undefined });
      if (typeof sel === 'string') accept(sel);
    } catch (err) { setHint(String(err)); }
    finally {
      release();
      void getCurrentWindow().setFocus().catch(() => {});
      areaRef.current?.focus();
    }
  };

  const capture = async () => {
    setBusy('capture');
    const release = guardBlur();
    try {
      const path = await invoke<string | null>('hosting_capture_screenshot');
      if (path) accept(path);
    } catch (err) { setHint(String(err)); }
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
          setFile(await saveBlob(wav, 'wav', `${t('hosting.composer.recording')}.wav`, 'audio'));
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

  const iconBtn = 'p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed';
  const placeholder = host
    ? t(`hosting.composer.placeholder.${spec.placeholder}`, { name: host.name })
    : t('hosting.composer.noModel');

  return (
    <div className={`relative ${dragging ? 'ring-2 ring-white/30 rounded-2xl' : ''}`}>
      {file && (
        <div className={compact ? 'px-3 pt-2' : 'mb-2'}>
          <AttachmentPreview file={file} onRemove={() => setFile(null)} small={compact} />
        </div>
      )}
      <div className={`flex items-end gap-1.5 ${compact ? 'px-2.5 py-2' : 'rounded-2xl border border-white/15 bg-black/20 px-2.5 py-2 focus-within:border-white/30'}`}>
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
          className={`flex-1 resize-none bg-transparent outline-none text-white placeholder-gray-500 ${compact ? 'text-[15px] py-1.5' : 'text-sm py-1.5'} leading-relaxed min-w-0`}
        />
        {spec.file === 'image' && (
          <button type="button" className={iconBtn} onClick={capture} disabled={!host || busy !== null} title={t('hosting.composer.screenshot')} aria-label={t('hosting.composer.screenshot')}>
            {busy === 'capture' ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <ScanLine className="w-[18px] h-[18px]" />}
          </button>
        )}
        {spec.file === 'audio' && (
          <button type="button" className={`${iconBtn} ${busy === 'record' ? '!text-red-400 bg-red-500/10' : ''}`} onClick={toggleRecord} disabled={!host || busy === 'capture' || busy === 'mic-wait'} title={t(busy === 'record' ? 'hosting.composer.stopRecording' : 'hosting.composer.record')} aria-label={t('hosting.composer.record')}>
            {busy === 'record' ? <Square className="w-[18px] h-[18px]" /> : busy === 'mic-wait' ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <Mic className="w-[18px] h-[18px]" />}
          </button>
        )}
        {spec.file && (
          <button type="button" className={iconBtn} onClick={pick} disabled={!host} title={t('hosting.composer.attach')} aria-label={t('hosting.composer.attach')}>
            <Paperclip className="w-[18px] h-[18px]" />
          </button>
        )}
        <button
          type="button"
          onClick={submit}
          disabled={!ready}
          aria-label={t('hosting.composer.send')}
          className={`p-1.5 rounded-lg transition-all ${ready ? `bg-gradient-to-r ${currentTheme.colors.gradient} text-white` : 'bg-white/5 text-gray-600'}`}
        >
          {sending ? <Loader2 className="w-[18px] h-[18px] animate-spin" /> : <ArrowUp className="w-[18px] h-[18px]" />}
        </button>
      </div>
      {hint && <p className={`text-xs text-amber-300 ${compact ? 'px-3 pb-2' : 'mt-1.5'}`}>{hint}</p>}
    </div>
  );
}
