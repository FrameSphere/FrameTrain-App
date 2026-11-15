// Sprachausgabe (text_to_speech): Erkennung, Dataset-Kompatibilitaet,
// Test-Oberflaeche mit Audio-Player, Labor-Samples und Hosting-Anzeige.
//
// Regeln:
//  * SpeechT5 als Text-to-Speech ist trainierbar; speecht5_asr / speecht5_vc
//    (gleicher model_type, andere Aufgabe) gehoeren nicht hierher.
//  * VITS / MMS-TTS wird erkannt (Test, Labor, Hosting), ist aber nicht
//    trainierbar — das Manifest fuehrt es getrennt als inference_only.
//  * Das Ergebnis ist eine WAV-Datei: gezeigt wird ein Player, nicht der Pfad.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const { mockInvoke, mockListen } = vi.hoisted(() => ({ mockInvoke: vi.fn(), mockListen: vi.fn() }));

vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
  convertFileSrc: (p: string) => `asset://localhost/${encodeURIComponent(p)}`,
}));
vi.mock('@tauri-apps/api/event', () => ({ listen: mockListen }));

import { detectPlugin, detectPluginForModel, getPluginById, hiddenTrainingFieldsForTaskType, PLUGINS } from '../registry';
import {
  detectTextToSpeech, isInferenceOnlyTts, TTS_INFERENCE_ONLY_MODEL_TYPES, TTS_TRAINABLE_MODEL_TYPES,
} from '../text-to-speech/detect';
import { detectSpeechRecognition } from '../speech-recognition/detect';
import { detectAudioClassification } from '../audio-classification/detect';
import { checkDatasetCompat } from '../datasetCompat';
import tts from '../text-to-speech';
import TextToSpeechTestPlugin from '../text-to-speech/TestPlugin';
import type { DatasetAnalysis } from '../datasetCompatHelpers';
import type { DatasetInfo } from '../types';
import { buildTaskSamples, labTaskFor, type DatasetFileInfo } from '../../components/labTaskSamples';
import { correctionKindFor } from '../../components/labCorrection';
import { composerSpec, taskKey } from '../../components/hosting/hostingModel';
import HostResultView from '../../components/hosting/HostResultView';
import { GeneratedAudio } from '../../components/LabTaskViews';
import { passendeAufgaben } from '../../components/studio/studioModels';
import { LanguageProvider } from '../../contexts/LanguageContext';
import de from '../../locales/de.json';
import en from '../../locales/en.json';

const pluginOf = (id: string, cfg?: { model_type?: string; architectures?: string[] }) => {
  const r = detectPlugin(id, cfg);
  return r.supported ? r.plugin.id : null;
};

describe('Erkennung ueber config.json', () => {
  it.each([
    [{ model_type: 'speecht5', architectures: ['SpeechT5ForTextToSpeech'] }],
    [{ model_type: 'speecht5' }],
    [{ model_type: 'vits', architectures: ['VitsModel'] }],
    [{ model_type: 'vits' }],
  ])('%o -> text-to-speech', (cfg) => {
    expect(pluginOf('/models/local_abc', cfg)).toBe('text-to-speech');
  });

  it('SpeechT5 als Spracherkenner oder Stimmwandler ist keine Sprachausgabe', () => {
    expect(detectTextToSpeech('x', { model_type: 'speecht5', architectures: ['SpeechT5ForSpeechToText'] })).toBe(false);
    expect(detectTextToSpeech('x', { model_type: 'speecht5', architectures: ['SpeechT5ForSpeechToSpeech'] })).toBe(false);
  });

  it('andere Modelle bleiben, wo sie waren', () => {
    expect(detectTextToSpeech('openai/whisper-tiny', { model_type: 'whisper' })).toBe(false);
    expect(detectTextToSpeech('bert-base-uncased', { model_type: 'bert' })).toBe(false);
    expect(pluginOf('x', { model_type: 'whisper', architectures: ['WhisperForConditionalGeneration'] })).toBe('speech-recognition');
    expect(pluginOf('x', { model_type: 'wav2vec2', architectures: ['Wav2Vec2ForSequenceClassification'] })).toBe('audio-classification');
    expect(pluginOf('t5-small', { model_type: 't5' })).toBe('seq2seq');
  });

  it('Spracherkennung und Audio-Klassifikation lehnen TTS-Modelle ab', () => {
    for (const cfg of [{ model_type: 'speecht5', architectures: ['SpeechT5ForTextToSpeech'] }, { model_type: 'vits' }]) {
      expect(detectSpeechRecognition('x', cfg)).toBe(false);
      expect(detectAudioClassification('x', cfg)).toBe(false);
    }
  });
});

describe('Erkennung am Namen (ohne config.json)', () => {
  it.each([
    'microsoft/speecht5_tts', 'facebook/mms-tts-deu', 'facebook/mms-tts-eng', 'kakao-enterprise/vits-ljs',
    'someone/speecht5_finetuned_voxpopuli_nl',
  ])('%s -> text-to-speech', (id) => expect(pluginOf(id)).toBe('text-to-speech'));

  it.each([
    'microsoft/speecht5_asr', 'microsoft/speecht5_vc', 'suno/bark', 'facebook/musicgen-small',
    // Andere TTS-Architekturen kann das Plugin weder trainieren noch abspielen.
    'parler-tts/parler-tts-mini-v1', 'coqui/XTTS-v2', 'irgendwer/tts-modell',
  ])('%s ist keine Sprachausgabe fuer dieses Plugin', (id) => expect(pluginOf(id)).not.toBe('text-to-speech'));

  it('die anderen SpeechT5-Varianten bekommen eine Begruendung', () => {
    // speecht5_asr greift schon vorher die Spracherkennung am Namensteil "asr" ab
    // (unveraendert) — die Begruendung hier gilt fuer die Stimmwandlung.
    const r = detectPlugin('microsoft/speecht5_vc');
    expect(r.supported).toBe(false);
    expect((r as { reason: string }).reason).toMatch(/Text-to-Speech-Variante/);
  });

  it('nur VITS / MMS-TTS ist "nur Inferenz"', () => {
    expect(isInferenceOnlyTts('facebook/mms-tts-deu')).toBe(true);
    expect(isInferenceOnlyTts('x', { model_type: 'vits' })).toBe(true);
    expect(isInferenceOnlyTts('microsoft/speecht5_tts')).toBe(false);
    expect(isInferenceOnlyTts('x', { model_type: 'speecht5' })).toBe(false);
  });
});

describe('Plugin-Eintrag', () => {
  it('steht vor Spracherkennung und Audio-Klassifikation', () => {
    const ids = PLUGINS.map(p => p.id);
    expect(ids.indexOf('text-to-speech')).toBeGreaterThanOrEqual(0);
    expect(ids.indexOf('text-to-speech')).toBeLessThan(ids.indexOf('speech-recognition'));
    expect(ids.indexOf('text-to-speech')).toBeLessThan(ids.indexOf('audio-classification'));
  });

  it('ist manuell waehlbar und hat den richtigen task_type', () => {
    expect(getPluginById('text-to-speech')?.taskType).toBe('text_to_speech');
    const r = detectPluginForModel({ id: 'm', name: 'irgendwas', plugin_override: 'text-to-speech' });
    expect(r.supported && r.plugin.id).toBe('text-to-speech');
  });

  it('lokal importiertes Modell: der model_type entscheidet', () => {
    const r = detectPluginForModel({ id: 'm', name: 'stimme', source_path: '/app/models/local_abc123', model_type: 'speecht5' });
    expect(r.supported && r.plugin.id).toBe('text-to-speech');
  });

  it('blendet Text-Felder aus und bringt Sprecher/Sprache als Parameter mit', () => {
    expect(tts.hiddenTrainingFields).toContain('max_seq_length');
    expect(hiddenTrainingFieldsForTaskType('text_to_speech')).toContain('use_lora');
    expect(tts.defaultPluginConfig).toMatchObject({ speaker_embedding: 'auto', language: '', max_seconds: 15 });
    // SpeechT5 verlernt bei hohen Lernraten die Aussprache.
    expect(tts.defaultTrainingConfig?.learning_rate).toBeLessThanOrEqual(2e-5);
  });

  it('Architekturlisten passen zum Train-Manifest', () => {
    const manifest = JSON.parse(readFileSync(
      join(process.cwd(), 'src-tauri', 'python', 'train_engine', 'plugins', 'text_to_speech', 'manifest.json'), 'utf-8'));
    expect(manifest.task_type).toBe(tts.taskType);
    expect([...TTS_TRAINABLE_MODEL_TYPES].sort()).toEqual([...manifest.supported_architectures].sort());
    expect([...TTS_INFERENCE_ONLY_MODEL_TYPES].sort()).toEqual([...manifest.inference_only_architectures].sort());
    const testManifest = JSON.parse(readFileSync(
      join(process.cwd(), 'src-tauri', 'python', 'test_engine', 'plugins', 'text_to_speech', 'manifest.json'), 'utf-8'));
    expect(testManifest.task_type).toBe(tts.taskType);
  });
});

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 90, pairing_status: null, warnings: [], file_count: 10, dir_count: 0, extensions, schema_hint: null,
});

describe('Dataset-Kompatibilitaet', () => {
  it('Audio + Transkript (Werkstatt-Export) ist perfekt', () => {
    expect(checkDatasetCompat('text-to-speech', [], analysis('audio_transcript', ['.wav', '.txt']), tts).overallLevel).toBe('perfect');
  });

  it('Common Voice und Parquet sind geeignet', () => {
    expect(checkDatasetCompat('text-to-speech', [], analysis('common_voice', ['.mp3', '.tsv']), tts).overallLevel).toBe('ok');
    expect(checkDatasetCompat('text-to-speech', [], analysis('flat_file', ['.parquet']), tts).overallLevel).toBe('ok');
  });

  it('ohne Audiodateien nicht geeignet', () => {
    expect(checkDatasetCompat('text-to-speech', [], analysis('flat_file', ['.csv']), tts).overallLevel).toBe('bad');
  });

  it('Klassenordner mit Audio taugt nicht (kein Transkript)', () => {
    expect(checkDatasetCompat('text-to-speech', [], analysis('folder_class', ['.wav']), tts).overallLevel).toBe('bad');
  });

  it('die Werkstatt bietet Transkript-Projekte auch der Sprachausgabe an', () => {
    expect(passendeAufgaben({ modality: 'audio', task: 'transcript' } as never)).toEqual(['speech_recognition', 'text_to_speech']);
    expect(passendeAufgaben({ modality: 'audio', task: 'classify' } as never)).toEqual(['audio_classification']);
  });
});

// ── Test-Oberflaeche ────────────────────────────────────────────────────────

type Cb = (e: { payload: unknown }) => void;
let listeners: Record<string, Cb[]> = {};
const emit = (event: string, payload: unknown) => listeners[event]?.forEach(cb => cb({ payload }));

const DS: DatasetInfo = { id: 'ds', name: 'Stimme', model_id: 'm', status: 'split', file_count: 3, size_bytes: 10 };
const props = { modelPath: '/m', versionId: 'v1', modelId: 'm', modelName: 'M', versionName: 'V', datasets: [DS] };

describe('Test-Oberflaeche', () => {
  beforeEach(() => {
    listeners = {};
    mockInvoke.mockReset();
    mockListen.mockReset();
    mockListen.mockImplementation((event: string, cb: Cb) => {
      (listeners[event] ??= []).push(cb);
      return Promise.resolve(() => { listeners[event] = listeners[event].filter(f => f !== cb); });
    });
  });

  it('Einzeltest: Text geht als single_input, das Ergebnis ist ein Audio-Player', async () => {
    mockInvoke.mockResolvedValue('single_1');
    const { container } = render(<TextToSpeechTestPlugin {...props} />);
    fireEvent.change(screen.getAllByRole('textbox')[0], { target: { value: 'Guten Morgen' } });
    fireEvent.click(screen.getAllByRole('button')[0]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('test_single_input', expect.objectContaining({
      singleInput: 'Guten Morgen', singleInputType: 'text', taskType: 'text_to_speech',
    })));
    await waitFor(() => expect(listeners['test-single-complete']?.length).toBeGreaterThan(0));
    expect(container.querySelector('audio')).toBeNull();
    act(() => emit('test-single-complete', {
      test_id: 'single_1',
      data: { predicted_output: '/out/guten_morgen.wav', output_kind: 'audio', inference_time: 2 },
    }));
    await waitFor(() => expect(container.querySelector('audio')).not.toBeNull());
    const audio = container.querySelector('audio')!;
    expect(audio.getAttribute('src')).toContain(encodeURIComponent('/out/guten_morgen.wav'));
    expect(audio.hasAttribute('controls')).toBe(true);
    // Der Pfad steht zusaetzlich darunter, aber keine erfundene Konfidenz.
    expect(screen.getByText('/out/guten_morgen.wav')).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/Konfidenz|Confidence/);
  });

  it('Datensatz-Lauf mit ASR: WER/CER und Hoerproben', async () => {
    mockInvoke.mockResolvedValue({ id: 'job_1' });
    const { container } = render(<TextToSpeechTestPlugin {...props} />);
    const buttons = screen.getAllByRole('button');
    fireEvent.click(buttons[buttons.length - 1]);
    await waitFor(() => expect(mockInvoke).toHaveBeenCalledWith('start_test', expect.objectContaining({
      datasetId: 'ds', taskType: 'text_to_speech',
    })));
    await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-complete', { data: {
      total_samples: 4, accuracy: 0.75, correct_predictions: 3,
      metrics: { wer: 0.037, cer: 0.031, real_time_factor: 1.4 },
      audios: ['/out/audio/0001.wav', '/out/audio/0002.wav'],
    } }));
    await waitFor(() => expect(container.querySelectorAll('audio').length).toBe(2));
    expect(screen.getByText('3.7 %')).toBeInTheDocument();
    expect(screen.getByText('3.1 %')).toBeInTheDocument();
    expect(screen.getByText('1.400')).toBeInTheDocument();
  });

  it('Datensatz-Lauf ohne ASR-Modell: Hoerproben, aber keine Trefferquote', async () => {
    mockInvoke.mockResolvedValue({ id: 'job_2' });
    const { container } = render(<TextToSpeechTestPlugin {...props} />);
    const buttons = screen.getAllByRole('button');
    fireEvent.click(buttons[buttons.length - 1]);
    await waitFor(() => expect(listeners['test-complete']?.length).toBeGreaterThan(0));
    act(() => emit('test-complete', { data: {
      total_samples: 2, accuracy: null, correct_predictions: null,
      metrics: { real_time_factor: 2.5 }, audios: ['/out/audio/0001.wav'],
    } }));
    await waitFor(() => expect(container.querySelectorAll('audio').length).toBe(1));
    expect(container.textContent).not.toMatch(/WER|CER/);
    expect(container.textContent).not.toMatch(/Treffer|Hits/);
  });
});

// ── Labor und Hosting ───────────────────────────────────────────────────────

const f = (path: string, split = 'test'): DatasetFileInfo => ({ name: path.split('/').pop()!, path, is_dir: false, split });
const reader = (files: Record<string, string>) => async (p: string) => {
  if (!(p in files)) throw new Error(`fehlt: ${p}`);
  return files[p];
};

describe('Labor', () => {
  it('Aufgabe aus Plugin oder Server-Modalitaet', () => {
    expect(labTaskFor('text_to_speech', null)).toBe('tts');
    expect(labTaskFor(null, 'tts')).toBe('tts');
    expect(labTaskFor('speech_recognition', null)).toBe('asr');
  });

  it('Samples: das Transkript ist die Eingabe, die Aufnahme die Referenz', async () => {
    const files = [f('/d/test/a.wav'), f('/d/test/a.txt'), f('/d/test/b.wav'), f('/d/test/b.txt'), f('/d/test/ohne.wav')];
    const s = await buildTaskSamples('tts', files, reader({ '/d/test/a.txt': ' Guten Morgen \n', '/d/test/b.txt': 'Guten Abend' }));
    expect(s?.map(x => [x.text, x.refAudio, x.filePath, x.fileKind])).toEqual([
      ['Guten Morgen', '/d/test/a.wav', undefined, undefined],
      ['Guten Abend', '/d/test/b.wav', undefined, undefined],
    ]);
  });

  it('Samples aus metadata.csv', async () => {
    const files = [f('/d/a.wav'), f('/d/metadata.csv')];
    const s = await buildTaskSamples('tts', files, reader({ '/d/metadata.csv': 'file_name,transcription\na.wav,hallo welt\n' }));
    expect(s?.map(x => x.text)).toEqual(['hallo welt']);
  });

  it('ohne Transkripte uebernimmt der allgemeine Lader', async () => {
    expect(await buildTaskSamples('tts', [f('/d/a.wav')], reader({}))).toBeNull();
  });

  it('Korrektur im Labor ist freier Text', () => {
    expect(correctionKindFor('tts')).toBe('text');
  });

  it('zeigt erzeugtes Audio und die Aufnahme aus dem Dataset', () => {
    const { container } = render(
      <LanguageProvider><GeneratedAudio path="/m/generated/x.wav" refAudio="/d/test/a.wav" /></LanguageProvider>);
    const players = [...container.querySelectorAll('audio')].map(a => a.getAttribute('src'));
    expect(players).toEqual([
      `asset://localhost/${encodeURIComponent('/m/generated/x.wav')}`,
      `asset://localhost/${encodeURIComponent('/d/test/a.wav')}`,
    ]);
    const single = render(<LanguageProvider><GeneratedAudio path="/m/generated/y.wav" /></LanguageProvider>);
    expect(single.container.querySelectorAll('audio').length).toBe(1);
  });
});

describe('Hosting', () => {
  it('Eingabe ist Text, kein Chat, eigener Platzhalter', () => {
    const spec = composerSpec({ modality: 'tts', input_kind: 'text' });
    expect(spec).toMatchObject({ file: null, text: 'required', placeholder: 'tts', chat: false });
    expect(taskKey({ modality: 'tts', task: null, input_kind: 'text' })).toBe('tts');
  });

  it('zeigt das erzeugte Audio als Player', () => {
    const host = {
      id: 'v1', model_id: 'm1', name: 'Stimme', api_name: 'stimme', status: 'ready', error: null,
      modality: 'tts', input_kind: 'text', classes: [], task: null, is_default: true, autoload: false,
      busy: false, size_gb: 0.6, loaded_at: 1, last_used: 1, requests: 0,
    } as never;
    const { container } = render(
      <LanguageProvider>
        <HostResultView host={host} inputText="Guten Morgen" result={{
          predicted: '/m/generated/a.wav', inference_ms: 3000, extra: { audio_path: '/m/generated/a.wav', output_kind: 'audio' },
        } as never} />
      </LanguageProvider>);
    const audio = container.querySelector('audio');
    expect(audio?.getAttribute('src')).toContain(encodeURIComponent('/m/generated/a.wav'));
    expect(audio?.hasAttribute('controls')).toBe(true);
  });
});

// ── Texte ───────────────────────────────────────────────────────────────────

describe('Texte in Deutsch und Englisch', () => {
  const get = (obj: unknown, path: string) => path.split('.').reduce<unknown>(
    (o, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), obj);
  const KEYS = [
    'home.pipeline.task.text_to_speech', 'plugins.text-to-speech.description',
    'testPlugins.generic.ttsLabel', 'testPlugins.generic.ttsPlaceholder', 'testPlugins.generic.ttsResult',
    'testPlugins.generic.ttsGenerated', 'testPlugins.generic.metricLabels.real_time_factor',
    'datasetCompat.msg.missing.tts', 'hosting.task.tts', 'hosting.composer.placeholder.tts',
    'laboratoryPanel.taskViews.generatedAudio', 'laboratoryPanel.taskViews.referenceAudio',
  ];
  it.each(KEYS)('%s', (key) => {
    const [d, e] = [get(de, key), get(en, key)];
    expect(typeof d === 'string' && d.length > 0, `de: ${key}`).toBe(true);
    expect(typeof e === 'string' && e.length > 0, `en: ${key}`).toBe(true);
    expect(d).not.toBe(e);
    // Keine Emojis in der UI.
    expect(`${d}${e}`).not.toMatch(/\p{Extended_Pictographic}/u);
  });
});
