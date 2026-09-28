// Spracherkennung (speech_recognition): Erkennung und Dataset-Kompatibilitaet.
//
// Vorher landete openai/whisper-* in der Audio-Klassifikation (whisper stand in
// deren Liste) und bekam dort einen Klassifikationskopf statt Transkripte.
// Regel seitdem: Whisper & *ForCTC -> ASR; nur *ForAudioClassification bzw.
// Klassifikator-Namen -> Audio-Klassifikation.

import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel, getPluginById, PLUGINS } from '../registry';
import { detectSpeechRecognition } from '../speech-recognition/detect';
import { detectAudioClassification } from '../audio-classification/detect';
import { checkDatasetCompat } from '../datasetCompat';
import asr from '../speech-recognition';
import type { DatasetAnalysis } from '../datasetCompatHelpers';

const pluginOf = (id: string, cfg?: { model_type?: string; architectures?: string[] }) => {
  const r = detectPlugin(id, cfg);
  return r.supported ? r.plugin.id : null;
};

describe('detectSpeechRecognition ueber config.json', () => {
  it.each([
    [{ model_type: 'whisper', architectures: ['WhisperForConditionalGeneration'] }],
    [{ model_type: 'wav2vec2', architectures: ['Wav2Vec2ForCTC'] }],
    [{ model_type: 'hubert', architectures: ['HubertForCTC'] }],
    [{ model_type: 'data2vec-audio', architectures: ['Data2VecAudioForCTC'] }],
    [{ model_type: 'speech_to_text', architectures: ['Speech2TextForConditionalGeneration'] }],
    [{ model_type: 'moonshine' }],
    [{ model_type: 'whisper' }],
  ])('%o -> ASR', (cfg) => {
    expect(pluginOf('/models/local_abc', cfg)).toBe('speech-recognition');
  });

  it('Klassifikatoren bleiben Audio-Klassifikation, auch Whisper als Encoder', () => {
    expect(pluginOf('x', { model_type: 'wav2vec2', architectures: ['Wav2Vec2ForSequenceClassification'] })).toBe('audio-classification');
    expect(pluginOf('x', { model_type: 'whisper', architectures: ['WhisperForAudioClassification'] })).toBe('audio-classification');
    expect(pluginOf('x', { model_type: 'audio-spectrogram-transformer' })).toBe('audio-classification');
  });

  it('wav2vec2 ohne Kopf entscheidet der Name', () => {
    expect(detectSpeechRecognition('facebook/wav2vec2-base', { model_type: 'wav2vec2' })).toBe(false);
    expect(detectSpeechRecognition('facebook/wav2vec2-base-960h', { model_type: 'wav2vec2' })).toBe(true);
    expect(detectSpeechRecognition('mein-asr-modell')).toBe(true);
  });

  it('Audio-Klassifikation lehnt Whisper und CTC ab', () => {
    expect(detectAudioClassification('openai/whisper-tiny')).toBe(false);
    expect(detectAudioClassification('x', { model_type: 'whisper' })).toBe(false);
    expect(detectAudioClassification('x', { model_type: 'wav2vec2', architectures: ['Wav2Vec2ForCTC'] })).toBe(false);
  });

  it('Textmodelle und Bildmodelle sind keine Spracherkennung', () => {
    expect(detectSpeechRecognition('bert-base-uncased', { model_type: 'bert' })).toBe(false);
    expect(detectSpeechRecognition('google/vit-base-patch16-224')).toBe(false);
    expect(detectSpeechRecognition('t5-small')).toBe(false);
  });
});

describe('Plugin-Eintrag', () => {
  it('steht vor der Audio-Klassifikation in der Registry', () => {
    const ids = PLUGINS.map(p => p.id);
    expect(ids.indexOf('speech-recognition')).toBeLessThan(ids.indexOf('audio-classification'));
  });

  it('ist manuell waehlbar (Import-Dialog) und hat den richtigen task_type', () => {
    expect(getPluginById('speech-recognition')?.taskType).toBe('speech_recognition');
    const r = detectPluginForModel({ id: 'm', name: 'irgendwas', plugin_override: 'speech-recognition' });
    expect(r.supported && r.plugin.id).toBe('speech-recognition');
  });

  it('blendet Text-Felder aus und bringt Sprache/Aufgabe als Parameter mit', () => {
    expect(asr.hiddenTrainingFields).toContain('max_seq_length');
    expect(asr.defaultPluginConfig).toMatchObject({ language: '', task: 'transcribe' });
  });
});

describe('Textklassifikation: Multi-Label/Regression als Plugin-Parameter', () => {
  it.each(['hf-encoder', 'xlm-roberta'])('%s bringt multi_label, problem_type und threshold mit', (id) => {
    // "auto" haelt den bisherigen Single-Label-Weg, solange die Daten nichts anderes zeigen.
    expect(getPluginById(id)?.defaultPluginConfig).toMatchObject({ multi_label: 'auto', problem_type: 'auto', threshold: 0.5 });
  });
});

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 90, pairing_status: null, warnings: [], file_count: 10, dir_count: 0, extensions, schema_hint: null,
});

describe('Dataset-Kompatibilitaet', () => {
  it('Audio + Transkript (Werkstatt-Export) ist perfekt', () => {
    expect(checkDatasetCompat('speech-recognition', [], analysis('audio_transcript', ['.wav', '.txt']), asr).overallLevel).toBe('perfect');
  });

  it('Common Voice und Parquet sind geeignet', () => {
    expect(checkDatasetCompat('speech-recognition', [], analysis('common_voice', ['.mp3', '.tsv']), asr).overallLevel).toBe('ok');
    expect(checkDatasetCompat('speech-recognition', [], analysis('flat_file', ['.parquet']), asr).overallLevel).toBe('ok');
  });

  it('ohne Audiodateien nicht geeignet', () => {
    expect(checkDatasetCompat('speech-recognition', [], analysis('flat_file', ['.csv']), asr).overallLevel).toBe('bad');
  });

  it('Klassenordner mit Audio kann ASR nicht lesen (kein Transkript)', () => {
    expect(checkDatasetCompat('speech-recognition', [], analysis('folder_class', ['.wav']), asr).overallLevel).toBe('bad');
  });
});
