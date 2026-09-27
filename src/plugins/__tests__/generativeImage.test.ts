// Generative Bildmodelle: Text-to-Image-LoRA (Stable Diffusion) und VLMs.
//
// Die Erkennung muss drei Quellen koennen: den HF-Namen aus der Suche,
// model_index.json (_class_name) eines diffusers-Ordners und den model_type,
// den Rust beim Import speichert. Andere Diffusion-Architekturen (FLUX, SD3)
// und Embedding-Modelle (CLIP) duerfen nicht als trainierbar gelten.

import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel } from '../registry';
import { genericDatasetCompat } from '../genericDatasetCompat';
import textToImageLoraPlugin from '../text-to-image-lora';
import visionLanguagePlugin from '../vision-language';
import { isSequenceTask } from '../../components/TrainingPanel';

const pluginOf = (id: string, cfg?: Record<string, unknown>) => {
  const r = detectPlugin(id, cfg as never);
  return r.supported ? r.plugin.id : null;
};

describe('Text-to-Image-LoRA', () => {
  it.each([
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    'stabilityai/stable-diffusion-2-1-base',
    'stabilityai/stable-diffusion-xl-base-1.0',
    'segmind/tiny-sd',
    'nota-ai/bk-sdm-tiny',
    'hf-internal-testing/tiny-stable-diffusion-pipe',
    'stabilityai/sdxl-turbo',
  ])('%s -> text-to-image-lora', (id) => expect(pluginOf(id)).toBe('text-to-image-lora'));

  it.each([
    'stabilityai/stable-diffusion-3-medium-diffusers',
    'black-forest-labs/FLUX.1-schnell',
    'PixArt-alpha/PixArt-XL-2-512x512',
  ])('%s hat einen anderen Aufbau und wird nicht angeboten', (id) =>
    expect(pluginOf(id)).not.toBe('text-to-image-lora'));

  it('model_index.json entscheidet ueber _class_name', () => {
    expect(pluginOf('/lokal/mein-modell', { _class_name: 'StableDiffusionPipeline' })).toBe('text-to-image-lora');
    expect(pluginOf('/lokal/mein-modell', { _class_name: 'StableDiffusionXLPipeline' })).toBe('text-to-image-lora');
    expect(pluginOf('/lokal/stable-diffusion-3', { _class_name: 'StableDiffusion3Pipeline' })).toBeNull();
  });

  it('importiertes Modell: model_type aus Rust (_class_name bzw. altes "diffusion")', () => {
    const base = { id: 'hf_1', name: 'Mein SD', source_path: '/models/hf_abc123def456' };
    const a = detectPluginForModel({ ...base, model_type: 'StableDiffusionPipeline' });
    const b = detectPluginForModel({ ...base, model_type: 'diffusion' });
    expect(a.supported && a.plugin.id).toBe('text-to-image-lora');
    expect(b.supported && b.plugin.id).toBe('text-to-image-lora');
  });

  it('Bildklassifikatoren greifen nicht nach Diffusion-Modellen', () => {
    expect(pluginOf('/x', { model_type: 'StableDiffusionPipeline' })).toBe('text-to-image-lora');
  });

  it('ist keine Token-Sequenz-Aufgabe', () => {
    expect(isSequenceTask('text_to_image_lora')).toBe(false);
  });
});

describe('Vision-Language', () => {
  it.each([
    'HuggingFaceTB/SmolVLM-256M-Instruct',
    'HuggingFaceTB/SmolVLM-500M-Instruct',
    'Qwen/Qwen2-VL-2B-Instruct',
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'google/paligemma-3b-pt-224',
    'llava-hf/llava-1.5-7b-hf',
    'Salesforce/blip-image-captioning-base',
    'microsoft/Florence-2-base',
  ])('%s -> vision-language', (id) => expect(pluginOf(id)).toBe('vision-language'));

  it.each([
    ['idefics3'], ['smolvlm'], ['qwen2_vl'], ['qwen2_5_vl'], ['paligemma'], ['blip'], ['llava'],
  ])('model_type %s -> vision-language', (mt) => expect(pluginOf('/lokal/x', { model_type: mt })).toBe('vision-language'));

  it('CLIP bleibt nicht unterstuetzt und nennt den Grund', () => {
    const r = detectPlugin('openai/clip-vit-base-patch32');
    expect(r.supported).toBe(false);
    expect((r as { supported: false; reason: string }).reason).toMatch(/Embedding/);
    expect(pluginOf('/lokal/x', { model_type: 'clip' })).toBeNull();
    expect(pluginOf('Salesforce/blip-itm-base-coco')).toBeNull();
  });

  it('reine Sprachmodelle bleiben abgelehnt', () => {
    expect(pluginOf('Qwen/Qwen2.5-0.5B')).toBeNull();
    expect(pluginOf('/lokal/x', { model_type: 'qwen2' })).toBeNull();
  });
});

describe('Datensatz-Kompatibilitaet', () => {
  it('Text-to-Image braucht Bilder', () => {
    const r = genericDatasetCompat(textToImageLoraPlugin, null, ['.csv']);
    expect(r.overallLevel).toBe('bad');
    expect(r.summary).toMatch(/Bilddateien/);
    expect(genericDatasetCompat(textToImageLoraPlugin, null, ['.png', '.txt']).overallLevel).toBe('ok');
  });

  it('VLM: JSONL mit Bildern passt, reine Tabelle nicht', () => {
    const ok = genericDatasetCompat(visionLanguagePlugin,
      { type: 'flat_file', extensions: ['.jsonl', '.png'] } as never, []);
    expect(ok.overallLevel).toBe('perfect');
    expect(genericDatasetCompat(visionLanguagePlugin, null, ['.jsonl']).overallLevel).toBe('bad');
  });

  it('YOLO-Datensaetze sind kein Bild-Text-Material', () => {
    const r = genericDatasetCompat(visionLanguagePlugin,
      { type: 'yolo_bbox', extensions: ['.jpg', '.txt'] } as never, []);
    expect(r.overallLevel).toBe('bad');
  });
});
