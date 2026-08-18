// Breitentest ueber Modalitaeten – E2E-Befunde vom 18.08.2026.
//
// Die App darf nur Modelle als trainierbar melden, die ein Plugin auch
// wirklich trainieren kann. Falsche Zusagen fallen sonst erst beim
// Trainingsstart auf – nach einem Download von hunderten Megabyte.

import { describe, it, expect } from 'vitest';
import { detectPlugin } from '../registry';

const pluginOf = (id: string) => {
  const r = detectPlugin(id);
  return r.supported ? r.plugin.id : null;
};

describe('Text-Encoder werden erkannt', () => {
  it.each([
    ['bert-base-uncased', 'hf-encoder'],
    ['roberta-base', 'hf-encoder'],
    ['microsoft/deberta-v3-small', 'hf-encoder'],
    ['albert-base-v2', 'hf-encoder'],
    ['google/electra-small-discriminator', 'hf-encoder'],
    ['distilbert-base-uncased', 'hf-encoder'],
    ['microsoft/mpnet-base', 'hf-encoder'],
    ['xlm-roberta-base', 'xlm-roberta'],
  ])('%s -> %s', (id, plugin) => expect(pluginOf(id)).toBe(plugin));

  it('sentence-transformers/all-MiniLM-L6-v2 ist ein BERT und wird erkannt', () => {
    // Wurde frueher abgelehnt, obwohl es eines der meistgenutzten Modelle ist.
    expect(pluginOf('sentence-transformers/all-MiniLM-L6-v2')).toBe('hf-encoder');
  });
});

describe('Decoder und Seq2Seq werden abgelehnt', () => {
  it.each([
    'gpt2', 'distilbert/distilgpt2', 'meta-llama/Llama-3.2-1B',
    'Qwen/Qwen2.5-0.5B', 'mistralai/Mistral-7B-v0.1',
    't5-small', 'google/flan-t5-base', 'facebook/bart-base', 'google/mt5-small',
  ])('%s', (id) => expect(pluginOf(id)).toBeNull());
});

describe('Audio- und Sprachmodelle werden abgelehnt', () => {
  it.each([
    'openai/whisper-tiny', 'facebook/wav2vec2-base-960h',
    'microsoft/speecht5_tts', 'pyannote/segmentation-3.0',
  ])('%s', (id) => expect(pluginOf(id)).toBeNull());
});

describe('Bildmodelle: Klassifikatoren ja, alles andere nein', () => {
  it.each([
    'google/vit-base-patch16-224',
    'microsoft/resnet-50',
    'facebook/deit-base-patch16-224',
    'google/efficientnet-b0',
    'timm/mobilenetv3_small_100',
  ])('%s -> image-classification', (id) => expect(pluginOf(id)).toBe('image-classification'));

  it('DETR ist Objekterkennung, kein Klassifikator', () => {
    // Matchte frueher auf den Teilstring "resnet".
    expect(pluginOf('facebook/detr-resnet-50')).toBeNull();
  });

  it('CLIP ist multimodal, kein Klassifikator', () => {
    // Matchte frueher auf den Teilstring "vit-b".
    expect(pluginOf('openai/clip-vit-base-patch32')).toBeNull();
  });

  it('BLIP wird abgelehnt', () => {
    expect(pluginOf('Salesforce/blip-image-captioning-base')).toBeNull();
  });

  it('config.json schlaegt den Namen', () => {
    expect(detectPlugin('mein/resnet-ordner', { model_type: 'clip' }).supported).toBe(false);
  });
});

describe('YOLO', () => {
  it.each(['yolov8n', 'Ultralytics/YOLOv8'])('%s -> yolo', (id) =>
    expect(pluginOf(id)).toBe('yolo'));
});

describe('Ablehnungen nennen den Grund, auch ohne config.json', () => {
  const reasonOf = (id: string) => {
    const r = detectPlugin(id);
    return r.supported ? '' : (r as { supported: false; reason: string }).reason;
  };

  it.each([
    ['openai/whisper-tiny', /Spracherkennung/i],
    ['facebook/wav2vec2-base-960h', /Audio/i],
    ['microsoft/speecht5_tts', /Text-to-Speech|Sprachsynthese/i],
    ['t5-small', /Seq2Seq/i],
    ['facebook/detr-resnet-50', /Objekterkennung/i],
    ['openai/clip-vit-base-patch32', /multimodal/i],
    ['meta-llama/Llama-3.2-1B', /Decoder/i],
  ])('%s', (id, muster) => expect(reasonOf(id)).toMatch(muster));

  it('unbekannte Modelle bekommen weiterhin den allgemeinen Hinweis', () => {
    expect(reasonOf('irgendwas/voellig-unbekannt')).toMatch(/noch nicht unterstützt/i);
  });
});
