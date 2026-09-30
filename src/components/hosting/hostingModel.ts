// Hosting: Typen und reine Logik (ohne React), damit sie testbar bleibt.

export type HostStatus = 'idle' | 'loading' | 'ready' | 'error' | 'sleeping';

export interface HostInfo {
  id: string;
  model_id: string;
  name: string;
  api_name: string;
  status: HostStatus;
  error: string | null;
  modality: string | null;
  input_kind: string | null;
  classes: string[];
  task: string | null;
  is_default: boolean;
  autoload: boolean;
  busy: boolean;
  size_gb: number | null;
  loaded_at: number | null;
  last_used: number | null;
  requests: number;
}

export interface HostingSettings {
  hosted: { version_id: string; model_id: string; name: string; autoload: boolean }[];
  version?: number;
  /** Hauptschalter Schnell-Zugriff (Kuerzel, Doppeltipp, Tray) */
  quick_enabled: boolean;
  default_id: string | null;
  shortcut: string;
  double_tap: 'off' | 'control' | 'alt' | 'shift' | 'meta';
  idle_minutes: number;
  api_enabled: boolean;
  api_port: number;
  api_token: string;
  tray_enabled: boolean;
}

export interface DesktopStatus {
  shortcut_error: string | null;
  double_tap_supported: boolean;
  api: { running: boolean; url: string | null; error: string | null };
  any_loaded: boolean;
  platform: string;
}

export interface InferResult {
  predicted: string;
  confidence?: number | null;
  top_predictions?: { label?: string; class?: string; score?: number; confidence?: number }[] | null;
  inference_ms: number;
  boxes?: Record<string, unknown>[] | null;
  image_width?: number | null;
  image_height?: number | null;
  extra?: Record<string, unknown> | null;
}

export type FileKind = 'image' | 'audio' | 'video' | 'file';

export interface Attachment {
  path: string;
  name: string;
  kind: FileKind;
}

export interface ChatMsg {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  file?: Attachment;
  result?: InferResult;
  error?: string;
  pending?: boolean;
  at: number;
}

// ── Eingabe je Modell ────────────────────────────────────────────────────

export interface ComposerSpec {
  /** Welche Datei das Modell braucht (null = keine). */
  file: Exclude<FileKind, 'file'> | null;
  /** Ist Text Pflicht, optional (Frage zum Bild) oder bedeutungslos? */
  text: 'required' | 'optional' | 'none';
  /** Schluessel des Platzhalters in hosting.composer.placeholder.* */
  placeholder: string;
  /** Datei-Filter fuer den Oeffnen-Dialog */
  extensions: string[];
  /** Gespraech mit Verlauf (LLM) statt Einzelanfragen */
  chat: boolean;
}

export const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tif', 'tiff'];
export const AUDIO_EXT = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'webm'];
export const VIDEO_EXT = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'];

export function composerSpec(host: Pick<HostInfo, 'modality' | 'input_kind'> | null | undefined): ComposerSpec {
  const modality = host?.modality ?? '';
  const kind = host?.input_kind ?? '';
  if (modality === 'vlm') return { file: 'image', text: 'optional', placeholder: 'vlm', extensions: IMAGE_EXT, chat: false };
  if (kind === 'image') return { file: 'image', text: 'none', placeholder: 'image', extensions: IMAGE_EXT, chat: false };
  if (kind === 'audio') return { file: 'audio', text: 'none', placeholder: modality === 'asr' ? 'asr' : 'audio', extensions: AUDIO_EXT, chat: false };
  if (kind === 'video') return { file: 'video', text: 'none', placeholder: 'video', extensions: VIDEO_EXT, chat: false };
  if (kind === 'tensor') return { file: null, text: 'required', placeholder: 'tensor', extensions: [], chat: false };
  const placeholder = ['causal_lm', 'seq2seq', 'token', 'embedding', 'text_to_image'].includes(modality) ? modality : 'text';
  return { file: null, text: 'required', placeholder, extensions: [], chat: modality === 'causal_lm' };
}

export function canSend(spec: ComposerSpec, text: string, file: Attachment | null): boolean {
  if (spec.file && !file) return false;
  if (spec.text === 'required' && !text.trim()) return false;
  return true;
}

export function fileKindOf(path: string): FileKind {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  if (IMAGE_EXT.includes(ext)) return 'image';
  if (VIDEO_EXT.includes(ext) && ext !== 'webm') return 'video';
  if (AUDIO_EXT.includes(ext)) return 'audio';
  if (ext === 'webm') return 'video';
  return 'file';
}

export function baseName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

/**
 * Verlauf fuer das LLM: fruehere Runden ohne Fehler und ohne die gerade
 * gestellte Frage, hoechstens `max` Nachrichten.
 */
export function historyFor(messages: ChatMsg[], max = 20): { role: string; content: string }[] {
  return messages
    .filter(m => !m.pending && !m.error && m.text.trim())
    .map(m => ({ role: m.role, content: m.role === 'assistant' ? (m.result?.predicted ?? m.text) : m.text }))
    .slice(-max);
}

// ── Anzeige ──────────────────────────────────────────────────────────────

/** Aufgabe eines gehosteten Modells als i18n-Schluessel unter hosting.task.* */
export function taskKey(host: Pick<HostInfo, 'modality' | 'task' | 'input_kind'>): string {
  const m = host.modality ?? '';
  if (m === 'detect') return host.task && host.task !== 'detect' ? `yolo_${host.task}` : 'detect';
  if (m === 'image' || m === 'audio' || m === 'video' || m === 'text') return `${m}_cls`;
  return m || 'unknown';
}

export function topPredictions(r: InferResult): { label: string; score: number }[] {
  return (r.top_predictions ?? [])
    .map(p => ({ label: String(p.label ?? p.class ?? '?'), score: Number(p.score ?? p.confidence ?? 0) }))
    .filter(p => Number.isFinite(p.score));
}

export function formatMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** Token je Sekunde fuer LLM-Antworten (aus completion_tokens). */
export function tokensPerSecond(r: InferResult): number | null {
  const n = Number(r.extra?.completion_tokens);
  if (!n || !r.inference_ms) return null;
  return Math.round((n / (r.inference_ms / 1000)) * 10) / 10;
}

// ── Tastenkuerzel ─────────────────────────────────────────────────────────

const MAC_SYMBOL: Record<string, string> = {
  control: '⌃', ctrl: '⌃', alt: '⌥', option: '⌥', shift: '⇧',
  super: '⌘', cmd: '⌘', command: '⌘', meta: '⌘', commandorcontrol: '⌘', cmdorctrl: '⌘',
};
const PC_NAME: Record<string, string> = {
  control: 'Ctrl', ctrl: 'Ctrl', alt: 'Alt', option: 'Alt', shift: 'Shift',
  super: 'Win', cmd: 'Win', command: 'Win', meta: 'Win', commandorcontrol: 'Ctrl', cmdorctrl: 'Ctrl',
};

/** "Control+Alt+Super+K" → "⌃⌥⌘K" (macOS) bzw. "Ctrl+Alt+Win+K". */
export function shortcutLabel(accel: string, mac: boolean): string {
  if (!accel.trim()) return '';
  const parts = accel.split('+').map(p => p.trim()).filter(Boolean);
  const key = (k: string) => (k.toLowerCase() === 'space' ? (mac ? '␣' : 'Space') : k.length === 1 ? k.toUpperCase() : k);
  if (mac) {
    // macOS-Reihenfolge: ⌃ ⌥ ⇧ ⌘
    const order = ['⌃', '⌥', '⇧', '⌘'];
    const mods = parts.slice(0, -1).map(p => MAC_SYMBOL[p.toLowerCase()] ?? p).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return mods.join('') + key(parts[parts.length - 1]);
  }
  return parts.map((p, i) => (i < parts.length - 1 ? PC_NAME[p.toLowerCase()] ?? p : key(p))).join('+');
}

/**
 * Tastendruck → Accelerator fuer tauri-plugin-global-shortcut. Mindestens eine
 * Sondertaste ausser Umschalt (sonst stoert es beim normalen Tippen).
 * null = (noch) kein gueltiges Kuerzel.
 */
export function acceleratorFromEvent(e: Pick<KeyboardEvent, 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey' | 'code'>): string | null {
  const mods: string[] = [];
  if (e.ctrlKey) mods.push('Control');
  if (e.altKey) mods.push('Alt');
  if (e.shiftKey) mods.push('Shift');
  if (e.metaKey) mods.push('Super');
  let key: string | null = null;
  const c = e.code;
  if (/^Key[A-Z]$/.test(c)) key = c.slice(3);
  else if (/^Digit[0-9]$/.test(c)) key = c.slice(5);
  else if (/^F([1-9]|1[0-2])$/.test(c)) key = c;
  else if (c === 'Space') key = 'Space';
  else if (['Period', 'Comma', 'Slash', 'Semicolon', 'Quote', 'BracketLeft', 'BracketRight', 'Backslash', 'Minus', 'Equal', 'Backquote'].includes(c)) key = c;
  if (!key || mods.length === 0 || (mods.length === 1 && mods[0] === 'Shift')) return null;
  return [...mods, key].join('+');
}

/** Gerade gehaltene Sondertasten fuer die Anzeige beim Aufnehmen ("⌘⇧…"). */
export function heldModifiers(e: Pick<KeyboardEvent, 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey'>, mac: boolean): string {
  const parts = [e.ctrlKey && 'Control', e.altKey && 'Alt', e.shiftKey && 'Shift', e.metaKey && 'Super'].filter(Boolean) as string[];
  if (!parts.length) return '';
  return shortcutLabel([...parts, '…'].join('+'), mac);
}

/** Bekannte Belegungen, vor denen die Einstellungen warnen (i18n-Schluessel unter hosting.conflict.*). */
export function knownConflict(accel: string, mac: boolean): string | null {
  const n = accel.toLowerCase().split('+').sort().join('+');
  const table: [string, string, boolean | null][] = [
    ['alt+space', 'assistants', null],
    ['space+super', 'spotlight', true],
    ['alt+space+super', 'finder', true],
    ['control+space', 'inputSources', true],
    ['alt+control+space', 'inputSources', true],
    ['shift+space+super', 'inputLanguage', false],
  ];
  const hit = table.find(([k, , plat]) => k === n && (plat === null || plat === mac));
  return hit ? hit[1] : null;
}

// ── API-Beispiele ────────────────────────────────────────────────────────

export function apiSnippets(url: string, token: string, host: Pick<HostInfo, 'api_name' | 'modality' | 'input_kind'>): { curl: string; python: string } {
  const spec = composerSpec(host);
  if (spec.chat || host.modality === 'seq2seq') {
    return {
      curl: `curl ${url}/chat/completions \\
  -H "Authorization: Bearer ${token}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${host.api_name}", "messages": [{"role": "user", "content": "Hallo!"}], "stream": false}'`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${url}", api_key="${token}")
antwort = client.chat.completions.create(
    model="${host.api_name}",
    messages=[{"role": "user", "content": "Hallo!"}],
)
print(antwort.choices[0].message.content)`,
    };
  }
  if (spec.file) {
    const ext = spec.file === 'image' ? 'png' : spec.file === 'audio' ? 'wav' : 'mp4';
    return {
      curl: `curl ${url}/infer \\
  -H "Authorization: Bearer ${token}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${host.api_name}", "file_path": "/pfad/zur/datei.${ext}"${spec.text === 'optional' ? ', "question": "Was ist zu sehen?"' : ''}}'`,
      python: `import base64, requests

with open("datei.${ext}", "rb") as f:
    daten = base64.b64encode(f.read()).decode()

r = requests.post("${url}/infer",
    headers={"Authorization": "Bearer ${token}"},
    json={"model": "${host.api_name}", "file_base64": daten, "file_ext": "${ext}"})
print(r.json()["result"])`,
    };
  }
  return {
    curl: `curl ${url}/infer \\
  -H "Authorization: Bearer ${token}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${host.api_name}", "text": "Beispieltext"}'`,
    python: `import requests

r = requests.post("${url}/infer",
    headers={"Authorization": "Bearer ${token}"},
    json={"model": "${host.api_name}", "text": "Beispieltext"})
print(r.json()["result"]["predicted"])`,
  };
}

export function newId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
