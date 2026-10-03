// Hosting (1.5.0): Eingabe je Modell, Kuerzel, API-Beispiele, Seite und Schnell-Chat.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Wie auf dem Mac: Kuerzel erscheinen als Symbole (IS_MAC wird beim Import gelesen).
vi.hoisted(() => { Object.defineProperty(navigator, 'platform', { value: 'MacIntel', configurable: true }); });

const invokeMock = vi.fn();
const listeners: Record<string, ((e: { payload: unknown }) => void)[]> = {};
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...a: unknown[]) => invokeMock(...a),
  convertFileSrc: (p: string) => `asset://${p}`,
}));
vi.mock('@tauri-apps/api/event', () => ({
  listen: (name: string, cb: (e: { payload: unknown }) => void) => {
    (listeners[name] ??= []).push(cb);
    return Promise.resolve(() => { listeners[name] = (listeners[name] ?? []).filter(f => f !== cb); });
  },
}));
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({ onDragDropEvent: () => Promise.resolve(() => {}) }),
}));
vi.mock('@tauri-apps/api/window', () => ({
  getCurrentWindow: () => ({ label: 'main', onFocusChanged: () => Promise.resolve(() => {}), setFocus: () => Promise.resolve() }),
}));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: vi.fn() }));
vi.mock('../../contexts/ThemeContext', () => ({
  useTheme: () => ({ currentTheme: { id: 'purple', colors: { gradient: 'from-purple-600 to-pink-600', primary: '#a855f7' } } }),
}));

import {
  acceleratorFromEvent, apiSnippets, heldModifiers, canSend, composerSpec, fileKindOf, historyFor, knownConflict,
  shortcutLabel, taskKey, tokensPerSecond, sessionAction, cycleHost, hostAccepting, agoParts, type ChatMsg, type HostInfo,
} from '../hosting/hostingModel';
import HostingPanel from '../hosting/HostingPanel';
import QuickChatApp from '../hosting/QuickChatApp';
import HostResultView from '../hosting/HostResultView';
import { LanguageProvider } from '../../contexts/LanguageContext';

const host = (over: Partial<HostInfo> = {}): HostInfo => ({
  id: 'v1', model_id: 'm1', name: 'SmolLM2 · v4', api_name: 'smollm2-v4', status: 'ready', error: null,
  modality: 'causal_lm', input_kind: 'text', classes: [], task: null, is_default: true, autoload: false,
  busy: false, size_gb: 0.54, loaded_at: 1, last_used: 1, requests: 0, ...over,
});

const SETTINGS = {
  hosted: [], version: 2, quick_enabled: true, default_id: 'v1', shortcut: 'Super+Shift+Space', double_tap: 'control', idle_minutes: 30,
  api_enabled: false, api_port: 47860, api_token: 'ft-geheim', tray_enabled: true,
};

function wrap(ui: React.ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

describe('Eingabe je Modell', () => {
  it('Text-, Bild-, Audio- und VLM-Modelle verlangen das Richtige', () => {
    expect(composerSpec(host())).toMatchObject({ file: null, text: 'required', chat: true });
    expect(composerSpec(host({ modality: 'detect', input_kind: 'image' }))).toMatchObject({ file: 'image', text: 'none' });
    expect(composerSpec(host({ modality: 'vlm', input_kind: 'image' }))).toMatchObject({ file: 'image', text: 'optional' });
    expect(composerSpec(host({ modality: 'asr', input_kind: 'audio' }))).toMatchObject({ file: 'audio', placeholder: 'asr' });
    expect(composerSpec(host({ modality: 'video', input_kind: 'video' }))).toMatchObject({ file: 'video' });
    expect(composerSpec(host({ modality: 'canvas', input_kind: 'tensor' }))).toMatchObject({ placeholder: 'tensor' });
    expect(composerSpec(host({ modality: 'text_to_image', input_kind: 'text' })).chat).toBe(false);
  });

  it('Senden erst, wenn die Pflichtteile da sind', () => {
    const img = composerSpec(host({ modality: 'image', input_kind: 'image' }));
    expect(canSend(img, 'egal', null)).toBe(false);
    expect(canSend(img, '', { path: '/a.png', name: 'a.png', kind: 'image' })).toBe(true);
    expect(canSend(composerSpec(host()), '   ', null)).toBe(false);
  });

  it('Dateiart an der Endung', () => {
    expect(fileKindOf('/x/Bild.PNG')).toBe('image');
    expect(fileKindOf('C:\\a\\b.wav')).toBe('audio');
    expect(fileKindOf('/v.mov')).toBe('video');
    expect(fileKindOf('/x.txt')).toBe('file');
  });

  it('Verlauf fuer das LLM ohne Fehler und laufende Antworten', () => {
    const msgs: ChatMsg[] = [
      { id: '1', role: 'user', text: 'A', at: 0 },
      { id: '2', role: 'assistant', text: 'B', at: 0, result: { predicted: 'B', inference_ms: 1 } },
      { id: '3', role: 'user', text: 'kaputt', at: 0 },
      { id: '4', role: 'assistant', text: '', error: 'x', at: 0 },
      { id: '5', role: 'assistant', text: 'halb', pending: true, at: 0 },
    ];
    expect(historyFor(msgs)).toEqual([
      { role: 'user', content: 'A' }, { role: 'assistant', content: 'B' }, { role: 'user', content: 'kaputt' },
    ]);
  });

  it('Aufgaben-Schluessel inkl. YOLO-Varianten', () => {
    expect(taskKey(host({ modality: 'detect', task: 'pose' }))).toBe('yolo_pose');
    expect(taskKey(host({ modality: 'detect', task: 'detect' }))).toBe('detect');
    expect(taskKey(host({ modality: 'image' }))).toBe('image_cls');
  });

  it('Token pro Sekunde', () => {
    expect(tokensPerSecond({ predicted: '', inference_ms: 2000, extra: { completion_tokens: 50 } })).toBe(25);
    expect(tokensPerSecond({ predicted: '', inference_ms: 2000 })).toBeNull();
  });
});

describe('Tastenkuerzel', () => {
  it('Standard kollidiert mit nichts Bekanntem', () => {
    expect(knownConflict('Super+Shift+Space', true)).toBeNull();
    expect(knownConflict('Control+Shift+Space', false)).toBeNull();
    expect(knownConflict('Alt+Space', true)).toBe('assistants');
    expect(knownConflict('Super+Space', true)).toBe('spotlight');
  });

  it('Anzeige als Symbole bzw. Namen', () => {
    expect(shortcutLabel('Control+Alt+Super+K', true)).toBe('⌃⌥⌘K');
    expect(shortcutLabel('Super+Shift+Alt+Control+K', true)).toBe('⌃⌥⇧⌘K');
    expect(shortcutLabel('Control+Alt+Shift+K', false)).toBe('Ctrl+Alt+Shift+K');
    expect(shortcutLabel('Super+Shift+Space', true)).toBe('⇧⌘␣');
    expect(shortcutLabel('', false)).toBe('');
  });

  it('Aufnahme: eine Sondertaste reicht, nur Umschalt nicht', () => {
    const ev = (o: Partial<KeyboardEvent>) => ({ ctrlKey: false, altKey: false, shiftKey: false, metaKey: false, code: 'KeyK', ...o });
    expect(acceleratorFromEvent(ev({}))).toBeNull();
    expect(acceleratorFromEvent(ev({ shiftKey: true }))).toBeNull();
    expect(acceleratorFromEvent(ev({ metaKey: true, code: 'KeyJ' }))).toBe('Super+J');
    expect(acceleratorFromEvent(ev({ metaKey: true, shiftKey: true, code: 'Space' }))).toBe('Shift+Super+Space');
    expect(acceleratorFromEvent(ev({ ctrlKey: true, shiftKey: true, code: 'ShiftLeft' }))).toBeNull();
    expect(heldModifiers({ ctrlKey: false, altKey: false, shiftKey: true, metaKey: true }, true)).toBe('⇧⌘…');
  });
});

describe('API-Beispiele', () => {
  it('LLM nutzt das OpenAI-Format, Bildmodelle /infer mit Datei', () => {
    const llm = apiSnippets('http://127.0.0.1:47860/v1', 'TOKEN', host());
    expect(llm.curl).toContain('/chat/completions');
    expect(llm.python).toContain('OpenAI(base_url="http://127.0.0.1:47860/v1"');
    const yolo = apiSnippets('http://127.0.0.1:47860/v1', 'TOKEN', host({ modality: 'detect', input_kind: 'image', api_name: 'teile' }));
    expect(yolo.curl).toContain('/infer');
    expect(yolo.python).toContain('file_base64');
  });
});

describe('Hosting-Seite', { timeout: 15000 }, () => {
  beforeEach(() => {
    invokeMock.mockReset();
    for (const k of Object.keys(listeners)) delete listeners[k];
  });

  it('ohne Modelle: Hinweis und Knopf zum Hosten', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_desktop_status') return Promise.resolve({ shortcut_error: null, double_tap_supported: true, api: { running: false, url: null, error: null }, any_loaded: false, platform: 'macos' });
      return Promise.resolve(null);
    });
    wrap(<HostingPanel />);
    expect(await screen.findByText('Noch kein Modell gehostet')).toBeInTheDocument();
    expect(screen.getAllByText('Modell hosten').length).toBeGreaterThan(0);
  });

  it('LLM-Chat: sendet Verlauf mit Streaming und zeigt die Token live', async () => {
    let resolveInfer: (v: unknown) => void = () => {};
    invokeMock.mockImplementation((cmd: string, args: Record<string, unknown>) => {
      if (cmd === 'hosting_list') return Promise.resolve([host()]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_chat_load') return Promise.resolve([
        { id: 'a', role: 'user', text: 'Vorher', at: 0 },
        { id: 'b', role: 'assistant', text: 'Antwort', at: 0, result: { predicted: 'Antwort', inference_ms: 5 } },
      ]);
      if (cmd === 'hosting_infer') {
        return new Promise(r => { resolveInfer = r; });
      }
      return Promise.resolve(null);
    });
    wrap(<HostingPanel />);
    expect(await screen.findByText('Vorher')).toBeInTheDocument();
    const box = screen.getByPlaceholderText('Nachricht an SmolLM2 · v4 …');
    fireEvent.change(box, { target: { value: 'Neue Frage' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    await waitFor(() => expect(invokeMock.mock.calls.some(c => c[0] === 'hosting_infer')).toBe(true));
    const args = invokeMock.mock.calls.find(c => c[0] === 'hosting_infer')![1] as { input: Record<string, unknown>; requestId: string };
    expect(args.input.stream).toBe(true);
    expect(args.input.text).toBe('Neue Frage');
    expect(args.input.messages).toEqual([{ role: 'user', content: 'Vorher' }, { role: 'assistant', content: 'Antwort' }]);

    await waitFor(() => expect(listeners['hosting-token']?.length).toBeGreaterThan(0));
    act(() => { listeners['hosting-token'].forEach(f => f({ payload: { request_id: args.requestId, text: 'Hal' } })); });
    act(() => { listeners['hosting-token'].forEach(f => f({ payload: { request_id: args.requestId, text: 'lo' } })); });
    expect(await screen.findByText('Hallo')).toBeInTheDocument();

    await act(async () => { resolveInfer({ predicted: 'Hallo!', inference_ms: 1000, extra: { completion_tokens: 20 } }); });
    expect(await screen.findByText('Hallo!')).toBeInTheDocument();
    expect(screen.getByText(/20 Token\/s/)).toBeInTheDocument();
    await waitFor(() => expect(invokeMock.mock.calls.some(c => c[0] === 'hosting_chat_save')).toBe(true));
  });

  it('Bildmodell: Senden gesperrt ohne Bild, Screenshot-Knopf da', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host({ modality: 'detect', input_kind: 'image', name: 'Teile', task: 'detect' })]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_chat_load') return Promise.resolve([]);
      return Promise.resolve(null);
    });
    wrap(<HostingPanel />);
    expect(await screen.findByLabelText('Bildschirmfoto aufnehmen')).toBeInTheDocument();
    expect(screen.getByLabelText('Senden')).toBeDisabled();
  });
});

describe('Schnell-Zugriff auf der Hosting-Seite', { timeout: 15000 }, () => {
  beforeEach(() => invokeMock.mockReset());

  it('zeigt nur den Stand und fuehrt in die Einstellungen', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host()]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_chat_load') return Promise.resolve([]);
      return Promise.resolve(null);
    });
    const nav = vi.fn();
    window.addEventListener('ft_navigate', (e: Event) => nav((e as CustomEvent).detail));
    wrap(<HostingPanel />);
    expect((await screen.findAllByText('2× ⌃ oder ⇧⌘␣')).length).toBeGreaterThan(0);
    expect(screen.queryByText('Tastenkürzel')).toBeNull();
    fireEvent.click(screen.getByText('Verwalten'));
    expect(nav).toHaveBeenCalledWith('settings');
    const { takeRequestedSettingsTab } = await import('../../ui/navigationEvents');
    expect(takeRequestedSettingsTab()).toBe('quick');
  });

  it('YOLO zeigt schon vor dem Laden die Bild-Eingabe (gemerkte Aufgabe)', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host({ status: 'idle', modality: 'detect', input_kind: 'image', task: 'detect', name: 'YOLO11 v4' })]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_chat_load') return Promise.resolve([]);
      return Promise.resolve(null);
    });
    wrap(<HostingPanel />);
    expect(await screen.findByPlaceholderText('Bild einfügen, ziehen oder aufnehmen')).toBeInTheDocument();
  });
});

describe('Einstellungen Schnell-Zugriff', { timeout: 15000 }, () => {
  beforeEach(() => invokeMock.mockReset());

  it('Hauptschalter aus blendet die Wege aus, Kuerzel-Aufnahme setzt das alte aus', async () => {
    let saved: Record<string, unknown> | null = null;
    invokeMock.mockImplementation((cmd: string, args: Record<string, unknown>) => {
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_list') return Promise.resolve([]);
      if (cmd === 'hosting_save_settings') { saved = args.settings as Record<string, unknown>; return Promise.resolve(saved); }
      return Promise.resolve(null);
    });
    const { default: QuickAccessSettings } = await import('../hosting/QuickAccessSettings');
    wrap(<QuickAccessSettings />);
    const rec = await screen.findByText('⇧⌘␣');
    fireEvent.click(rec);
    expect(invokeMock).toHaveBeenCalledWith('hosting_pause_shortcut', { paused: true });
    fireEvent.keyDown(window, { code: 'KeyJ', key: 'j', metaKey: true, shiftKey: true });
    await waitFor(() => expect(saved?.shortcut).toBe('Shift+Super+J'));
    expect(invokeMock).toHaveBeenCalledWith('hosting_pause_shortcut', { paused: false });

    fireEvent.click(screen.getAllByRole('switch')[0]);
    await waitFor(() => expect(saved?.quick_enabled).toBe(false));
    expect(screen.queryByText('Zweimal tippen')).toBeNull();
  });
});

describe('Ergebnisanzeige', () => {
  it('Klassifikation mit Balken', () => {
    wrap(<HostResultView host={host({ modality: 'image', input_kind: 'image' })} result={{
      predicted: 'katze', confidence: 0.91, inference_ms: 12,
      top_predictions: [{ label: 'katze', score: 0.91 }, { label: 'hund', score: 0.09 }],
    }} />);
    expect(screen.getAllByText('katze').length).toBeGreaterThan(0);
    expect(screen.getByText('hund')).toBeInTheDocument();
    expect(screen.getByText('9.0 %')).toBeInTheDocument();
  });

  it('Text-zu-Bild zeigt das erzeugte Bild', () => {
    wrap(<HostResultView host={host({ modality: 'text_to_image' })} result={{
      predicted: '/out/a.png', inference_ms: 3000, extra: { image_path: '/out/a.png' },
    }} />);
    expect(screen.getByRole('img')).toHaveAttribute('src', 'asset:///out/a.png');
  });
});

describe('Schnell-Chat', () => {
  beforeEach(() => invokeMock.mockReset());

  it('ohne gehostete Modelle: Weg in die App', async () => {
    invokeMock.mockImplementation((cmd: string) => Promise.resolve(cmd === 'hosting_list' ? [] : null));
    wrap(<QuickChatApp />);
    fireEvent.click(await screen.findByText('Modell hosten'));
    expect(invokeMock).toHaveBeenCalledWith('hosting_open_main', { view: 'hosting' });
  });

  it('Esc blendet aus, Modellwechsel macht es zum Standard', async () => {
    invokeMock.mockImplementation((cmd: string) => Promise.resolve(cmd === 'hosting_list'
      ? [host(), host({ id: 'v2', name: 'Teile', is_default: false, modality: 'detect', input_kind: 'image' })]
      : null));
    wrap(<QuickChatApp />);
    fireEvent.click(await screen.findByText('SmolLM2 · v4'));
    fireEvent.click(await screen.findByText('Teile'));
    expect(invokeMock).toHaveBeenCalledWith('hosting_update_model', { id: 'v2', makeDefault: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(invokeMock).toHaveBeenCalledWith('hosting_hide_quickchat');
  });
});

describe('Schnell-Chat: Sitzungsregeln', () => {
  const base = { policy: 'smart' as const, pending: false, unread: false, hiddenForMs: 120_000, hasMessages: true };

  it('neu nach laengerer Pause, weiter nach kurzer', () => {
    expect(sessionAction(base)).toBe('new');
    expect(sessionAction({ ...base, hiddenForMs: 5_000 })).toBe('keep');
    expect(sessionAction({ ...base, hiddenForMs: null })).toBe('new');
    expect(sessionAction({ ...base, hasMessages: false })).toBe('keep');
  });

  it('laufende oder ungelesene Antwort geht nie verloren', () => {
    expect(sessionAction({ ...base, pending: true })).toBe('keep');
    expect(sessionAction({ ...base, unread: true })).toBe('keep');
    expect(sessionAction({ ...base, policy: 'always', unread: true })).toBe('keep');
  });

  it('Einstellung immer / nie', () => {
    expect(sessionAction({ ...base, policy: 'always', hiddenForMs: 1_000 })).toBe('new');
    expect(sessionAction({ ...base, policy: 'never' })).toBe('keep');
  });

  it('Tab laeuft im Kreis durch die Modelle', () => {
    const hs = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
    expect(cycleHost(hs, 'a', 1)).toBe('b');
    expect(cycleHost(hs, 'c', 1)).toBe('a');
    expect(cycleHost(hs, 'a', -1)).toBe('c');
    expect(cycleHost([], 'a', 1)).toBeNull();
  });

  it('passendes Modell fuer eine Eingabe, laufende zuerst', () => {
    const hs = [
      host({ id: 'llm' }),
      host({ id: 'y1', modality: 'detect', input_kind: 'image', status: 'idle' }),
      host({ id: 'y2', modality: 'image', input_kind: 'image', status: 'ready' }),
      host({ id: 'asr', modality: 'asr', input_kind: 'audio' }),
    ];
    expect(hostAccepting(hs, 'image', 'llm')?.id).toBe('y2');
    expect(hostAccepting(hs, 'audio', 'llm')?.id).toBe('asr');
    expect(hostAccepting(hs, 'text', 'y1')?.id).toBe('llm');
    expect(hostAccepting(hs, 'video', 'llm')).toBeNull();
    expect(hostAccepting(hs, 'text', 'llm')).toBeNull();
  });

  it('Zeitangaben', () => {
    expect(agoParts(3_000)).toEqual({ key: 'now', value: 0 });
    expect(agoParts(40_000)).toEqual({ key: 'sec', value: 40 });
    expect(agoParts(12 * 60_000)).toEqual({ key: 'min', value: 12 });
  });
});

describe('Schnell-Chat: Verhalten im Fenster', { timeout: 15000 }, () => {
  const fire = (name: string) => act(() => { (listeners[name] ?? []).forEach(f => f({ payload: {} })); });

  beforeEach(() => {
    invokeMock.mockReset();
    for (const k of Object.keys(listeners)) delete listeners[k];
  });

  const setup = (hostsList: HostInfo[], answer: unknown = { predicted: 'Antwort A', inference_ms: 100 }) => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve(hostsList);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_infer') return Promise.resolve(answer);
      return Promise.resolve(null);
    });
    wrap(<QuickChatApp />);
  };

  it('Antwort landet im Verlauf des Modells; erneutes Oeffnen nach Pause beginnt neu, ⌘↑ holt zurueck', async () => {
    setup([host()]);
    const box = await screen.findByPlaceholderText('Nachricht an SmolLM2 · v4 …');
    await waitFor(() => expect(listeners['quickchat-shown']?.length).toBeGreaterThan(0));
    await fire('quickchat-shown');
    fireEvent.change(box, { target: { value: 'Frage 1' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    expect(await screen.findByText('Antwort A')).toBeInTheDocument();
    const append = invokeMock.mock.calls.find(c => c[0] === 'hosting_chat_append')![1] as { id: string; messages: ChatMsg[] };
    expect(append.id).toBe('v1');
    expect(append.messages.map(m => m.source)).toEqual(['quick', 'quick']);

    // Zu, mehr als 30 s spaeter wieder auf → leerer Chat mit Chip "Letzter Chat"
    await fire('quickchat-hidden');
    const spy = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 5 * 60_000);
    await fire('quickchat-shown');
    await waitFor(() => expect(screen.queryByText('Antwort A')).toBeNull());
    expect(await screen.findByText(/Letzter Chat/)).toBeInTheDocument();
    spy.mockRestore();

    fireEvent.keyDown(window, { key: 'ArrowUp', metaKey: true });
    expect(await screen.findByText('Antwort A')).toBeInTheDocument();
  });

  it('kurze Unterbrechung: derselbe Chat bleibt', async () => {
    setup([host()]);
    const box = await screen.findByPlaceholderText('Nachricht an SmolLM2 · v4 …');
    await waitFor(() => expect(listeners['quickchat-shown']?.length).toBeGreaterThan(0));
    await fire('quickchat-shown');
    fireEvent.change(box, { target: { value: 'Frage' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    expect(await screen.findByText('Antwort A')).toBeInTheDocument();
    await fire('quickchat-hidden');
    await fire('quickchat-shown');
    expect(screen.getByText('Antwort A')).toBeInTheDocument();
  });

  it('Antwort wird bei geschlossenem Fenster fertig: Punkt am Tray, naechstes Oeffnen zeigt sie', async () => {
    let finish: (v: unknown) => void = () => {};
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host()]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_infer') return new Promise(r => { finish = r; });
      return Promise.resolve(null);
    });
    wrap(<QuickChatApp />);
    const box = await screen.findByPlaceholderText('Nachricht an SmolLM2 · v4 …');
    await waitFor(() => expect(listeners['quickchat-shown']?.length).toBeGreaterThan(0));
    await fire('quickchat-shown');
    fireEvent.change(box, { target: { value: 'Lange Frage' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    await waitFor(() => expect(invokeMock.mock.calls.some(c => c[0] === 'hosting_infer')).toBe(true));
    await fire('quickchat-hidden');
    await act(async () => { finish({ predicted: 'Spaete Antwort', inference_ms: 9000 }); });
    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('hosting_set_tray_badge', { on: true }));

    const spy = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 10 * 60_000);
    await fire('quickchat-shown');
    expect(await screen.findByText('Spaete Antwort')).toBeInTheDocument();
    expect(screen.getByText(/Antwort von vorhin/)).toBeInTheDocument();
    expect(invokeMock).toHaveBeenCalledWith('hosting_set_tray_badge', { on: false });
    spy.mockRestore();
  });

  it('Bildschirmfoto (Knopf und ⌘⇧S) haengt das Bild an; Abbruch laesst alles, wie es war', async () => {
    let shot: string | null = '/cache/screenshot_1.png';
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host({ name: 'Teile', modality: 'detect', input_kind: 'image' })]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_capture_screenshot') return Promise.resolve(shot);
      return Promise.resolve(null);
    });
    wrap(<QuickChatApp />);
    const box = await screen.findByPlaceholderText('Bild einfügen, ziehen oder aufnehmen');
    expect(screen.getByLabelText('Senden')).toBeDisabled();
    fireEvent.keyDown(box, { key: 's', metaKey: true, shiftKey: true });
    expect(await screen.findByText('screenshot_1.png')).toBeInTheDocument();
    expect(screen.getByLabelText('Senden')).not.toBeDisabled();

    fireEvent.click(screen.getByLabelText('remove'));
    shot = null;
    fireEvent.click(screen.getByLabelText('Bildschirmfoto aufnehmen'));
    await waitFor(() => expect(invokeMock.mock.calls.filter(c => c[0] === 'hosting_capture_screenshot').length).toBe(2));
    expect(screen.queryByText('screenshot_1.png')).toBeNull();
    expect(screen.getByLabelText('Senden')).toBeDisabled();
  });

  it('Bildschirmfoto ohne macOS-Freigabe: klare Meldung mit Weg in die Einstellungen', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'hosting_list') return Promise.resolve([host({ name: 'Teile', modality: 'detect', input_kind: 'image' })]);
      if (cmd === 'hosting_get_settings') return Promise.resolve(SETTINGS);
      if (cmd === 'hosting_capture_screenshot') return Promise.reject('SCREEN_PERMISSION');
      return Promise.resolve(null);
    });
    wrap(<QuickChatApp />);
    fireEvent.click(await screen.findByLabelText('Bildschirmfoto aufnehmen'));
    expect(await screen.findByText(/erlaubt FrameTrain die Bildschirmaufnahme noch nicht/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('Einstellungen öffnen'));
    expect(invokeMock).toHaveBeenCalledWith('hosting_open_screen_permission');
  });

  it('Esc in Stufen: Liste zu, Entwurf leeren, dann Fenster zu', async () => {
    setup([host(), host({ id: 'v2', name: 'Teile', is_default: false, modality: 'detect', input_kind: 'image' })]);
    const box = await screen.findByPlaceholderText('Nachricht an SmolLM2 · v4 …') as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: 'Entwurf' } });
    fireEvent.click(screen.getByText('SmolLM2 · v4'));
    expect(await screen.findByText('Objekterkennung (YOLO)')).toBeInTheDocument();

    fireEvent.keyDown(box, { key: 'Escape' });
    expect(screen.queryByText('Objekterkennung (YOLO)')).toBeNull();
    expect(box.value).toBe('Entwurf');
    expect(invokeMock).not.toHaveBeenCalledWith('hosting_hide_quickchat');

    fireEvent.keyDown(box, { key: 'Escape' });
    expect(box.value).toBe('');
    expect(invokeMock).not.toHaveBeenCalledWith('hosting_hide_quickchat');

    fireEvent.keyDown(box, { key: 'Escape' });
    expect(invokeMock).toHaveBeenCalledWith('hosting_hide_quickchat');
  });

  it('Tab wechselt das Modell und damit die Eingabe', async () => {
    setup([host(), host({ id: 'v2', name: 'Teile', is_default: false, modality: 'detect', input_kind: 'image' })]);
    const box = await screen.findByPlaceholderText('Nachricht an SmolLM2 · v4 …');
    fireEvent.keyDown(box, { key: 'Tab' });
    expect(await screen.findByPlaceholderText('Bild einfügen, ziehen oder aufnehmen')).toBeInTheDocument();
    expect(screen.getByLabelText('Bildschirmfoto aufnehmen')).toBeInTheDocument();
  });

  it('Text bei einem Bildmodell: Textmodell wird vorgeschlagen, Tab uebernimmt samt Entwurf', async () => {
    setup([host({ id: 'y', name: 'Teile', modality: 'detect', input_kind: 'image' }), host({ id: 'llm', name: 'Smol', is_default: false })]);
    const box = await screen.findByPlaceholderText('Bild einfügen, ziehen oder aufnehmen');
    fireEvent.change(box, { target: { value: 'Was ist ein Adapter?' } });
    expect(await screen.findByText('Text? Mit Smol fragen')).toBeInTheDocument();
    fireEvent.keyDown(box, { key: 'Tab' });
    const next = await screen.findByPlaceholderText('Nachricht an Smol …') as HTMLTextAreaElement;
    await waitFor(() => expect(next.value).toBe('Was ist ein Adapter?'));
  });
});
