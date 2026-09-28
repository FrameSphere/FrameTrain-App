// Architekturlisten des Frontends gegen die manifest.json der Train-Engine.
//
// Die Listen wurden bisher an zwei Stellen von Hand gepflegt: erkannte das
// Frontend eine Architektur, die das Python-Plugin nicht kannte (oder
// umgekehrt), stand ein Modell als "unterstuetzt" in der App und scheiterte
// erst beim Training — oder blieb ohne Grund gesperrt. Die Python-Seite
// (plugin.py <-> Manifest) prueft train_engine/test_plugin_consistency.py.

// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AUDIO_MODEL_TYPES } from '../audio-classification/detect';
import { VIDEO_MODEL_TYPES } from '../video-classification/detect';
import { SEQ2SEQ_MODEL_TYPES } from '../seq2seq/detect';
import { HF_IMAGE_MODEL_TYPES } from '../hf-image-classification/detect';
import { HF_ENCODER_SUPPORTED_MODEL_TYPES } from '../hf-encoder/detect';
import { YOLO_MODEL_TYPES } from '../yolo/detect';
import { detectXLMRoberta } from '../xlm-roberta/detect';
import { PLUGINS } from '../registry';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const TRAIN_PLUGINS = join(ROOT, 'src-tauri', 'python', 'train_engine', 'plugins');

interface Manifest { task_type: string; class: string; entry: string; supported_architectures?: string[] }

function manifests(): Record<string, Manifest> {
  const out: Record<string, Manifest> = {};
  for (const dir of readdirSync(TRAIN_PLUGINS)) {
    const file = join(TRAIN_PLUGINS, dir, 'manifest.json');
    if (!existsSync(file)) continue;
    const m = JSON.parse(readFileSync(file, 'utf-8')) as Manifest;
    out[m.task_type] = m;
  }
  return out;
}

const sorted = (xs: string[]) => [...new Set(xs)].sort();

describe('Frontend-Architekturlisten == Train-Manifeste', () => {
  const byTask = manifests();
  const cases: Array<[string, string, string[]]> = [
    ['audio_classification', 'AUDIO_MODEL_TYPES', AUDIO_MODEL_TYPES],
    ['video_classification', 'VIDEO_MODEL_TYPES', VIDEO_MODEL_TYPES],
    ['seq2seq', 'SEQ2SEQ_MODEL_TYPES', SEQ2SEQ_MODEL_TYPES],
    ['hf_image_classification', 'HF_IMAGE_MODEL_TYPES', HF_IMAGE_MODEL_TYPES],
    ['seq_classification', 'HF_ENCODER_SUPPORTED_MODEL_TYPES', HF_ENCODER_SUPPORTED_MODEL_TYPES],
    ['detect', 'YOLO_MODEL_TYPES', YOLO_MODEL_TYPES],
  ];
  // Bewusste Ausnahmen: das Python-Plugin kann es, die Namensliste im Frontend
  // leitet es aber woanders hin. whisper trainiert audio_classification als
  // WhisperForAudioClassification (Erkennung ueber die Architektur); am
  // model_type allein ist Whisper ein Spracherkenner (speech-recognition).
  const TRAIN_ONLY: Record<string, string[]> = { audio_classification: ['whisper'] };
  for (const [task, name, list] of cases) {
    it(`${name} passt zu plugins/*/manifest.json (task_type ${task})`, () => {
      expect(byTask[task], `kein Manifest fuer ${task}`).toBeTruthy();
      const skip = new Set(TRAIN_ONLY[task] ?? []);
      expect(sorted(list)).toEqual(sorted((byTask[task].supported_architectures ?? []).filter(a => !skip.has(a))));
    });
  }

  it('XLM-RoBERTa-Plugin erkennt nur, was seq_classification trainiert', () => {
    expect(byTask.seq_classification.supported_architectures).toContain('xlm-roberta');
    expect(detectXLMRoberta('x', { model_type: 'xlm-roberta' })).toBe(true);
  });

  it('jeder taskType eines Frontend-Plugins hat ein Train-Plugin', () => {
    for (const plugin of PLUGINS) {
      expect(byTask[plugin.taskType], `${plugin.id} -> ${plugin.taskType}`).toBeTruthy();
    }
  });

  it('jedes Manifest ist vollstaendig und die Plugin-Datei existiert', () => {
    for (const [task, m] of Object.entries(byTask)) {
      expect(m.class, task).toBeTruthy();
      const dir = readdirSync(TRAIN_PLUGINS).find(d => existsSync(join(TRAIN_PLUGINS, d, 'manifest.json'))
        && JSON.parse(readFileSync(join(TRAIN_PLUGINS, d, 'manifest.json'), 'utf-8')).task_type === task)!;
      const source = readFileSync(join(TRAIN_PLUGINS, dir, m.entry), 'utf-8');
      expect(source, `${m.class} in ${dir}/${m.entry}`).toMatch(new RegExp(`^class ${m.class}\\b`, 'm'));
    }
  });
});
