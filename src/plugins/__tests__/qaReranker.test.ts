// Frage-Antwort und Reranker: Erkennung gegen sentence-embedding und hf-encoder,
// Registrierung, Dataset-Pruefung und Texte (EN/DE).
import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel, hiddenTrainingFieldsForTaskType, PLUGINS } from '../registry';
import { detectQuestionAnswering } from '../question-answering/detect';
import { detectReranker } from '../reranker/detect';
import { detectSentenceEmbedding } from '../sentence-embedding/detect';
import { detectHFEncoder } from '../hf-encoder/detect';
import { genericDatasetCompat } from '../genericDatasetCompat';
import en from '../../locales/en.json';
import de from '../../locales/de.json';

const pluginOf = (id: string, cfg?: Record<string, unknown>) => {
  const r = detectPlugin(id, cfg as never);
  return r.supported ? r.plugin.id : null;
};

describe('Erkennung: Reranker vs. Embedding vs. Encoder', () => {
  const rerankers = [
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'cross-encoder/stsb-roberta-base',
    'BAAI/bge-reranker-base',
    'BAAI/bge-reranker-v2-m3',
    'mixedbread-ai/mxbai-rerank-xsmall-v1',
  ];
  it.each(rerankers)('%s -> reranker', id => {
    expect(detectReranker(id)).toBe(true);
    expect(detectSentenceEmbedding(id)).toBe(false);
    expect(pluginOf(id)).toBe('reranker');
  });

  it('mit model_type (lokaler Import) bleibt es beim Reranker', () => {
    expect(pluginOf('cross-encoder/ms-marco-MiniLM-L-6-v2', { model_type: 'bert' })).toBe('reranker');
    expect(pluginOf('BAAI/bge-reranker-base', { model_type: 'xlm-roberta' })).toBe('reranker');
  });

  it('Embedding-Modelle bleiben beim Embedding-Plugin', () => {
    for (const id of ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5', 'intfloat/e5-base-v2',
      'sentence-transformers/msmarco-MiniLM-L6-cos-v5', 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1']) {
      expect(detectReranker(id)).toBe(false);
      expect(detectQuestionAnswering(id)).toBe(false);
      expect(pluginOf(id)).toBe('sentence-embedding');
    }
  });

  it('Decoder-Reranker gehoert nicht zu diesem Plugin', () => {
    expect(detectReranker('Qwen/Qwen3-Reranker-0.6B')).toBe(false);
    expect(detectReranker('local-reranker', { model_type: 'qwen3' })).toBe(false);
    expect(pluginOf('Qwen/Qwen3-Reranker-0.6B', { model_type: 'qwen3' })).toBe('causal-lm');
  });

  it('Basis-Encoder ohne Hinweis bleiben bei der Sequenzklassifikation', () => {
    for (const id of ['bert-base-uncased', 'distilbert-base-uncased', 'microsoft/deberta-v3-base', 'roberta-base']) {
      expect(detectReranker(id)).toBe(false);
      expect(detectQuestionAnswering(id)).toBe(false);
      expect(['hf-encoder', 'xlm-roberta']).toContain(pluginOf(id));
    }
    // hf-encoder selbst erkennt Reranker-Namen weiterhin als Encoder — die
    // Reihenfolge in der Registry entscheidet.
    expect(detectHFEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')).toBe(true);
  });
});

describe('Erkennung: Frage-Antwort', () => {
  const qa = [
    'deepset/roberta-base-squad2',
    'distilbert-base-cased-distilled-squad',
    'deepset/xlm-roberta-large-squad2',
    'deepset/gelectra-base-germanquad',
    'timpal0l/mdeberta-v3-base-squad2',
  ];
  it.each(qa)('%s -> question-answering', id => {
    expect(detectQuestionAnswering(id)).toBe(true);
    expect(detectReranker(id)).toBe(false);
    expect(detectSentenceEmbedding(id)).toBe(false);
    expect(pluginOf(id)).toBe('question-answering');
  });

  it('Architektur ...ForQuestionAnswering genuegt', () => {
    expect(pluginOf('local/ver_abc123', { model_type: 'bert', architectures: ['BertForQuestionAnswering'] }))
      .toBe('question-answering');
  });

  it('LLMs mit "qa" im Namen bleiben Decoder', () => {
    expect(detectQuestionAnswering('someone/llama-3-qa')).toBe(false);
    expect(detectQuestionAnswering('x', { model_type: 'llama', architectures: ['LlamaForQuestionAnswering'] })).toBe(false);
  });

  it('lokal importiert: der vergebene Name entscheidet, die Hand-Zuordnung gewinnt', () => {
    expect(detectPluginForModel({ id: '1', name: 'faq-squad2-modell', source_path: '/models/local_abc123', model_type: 'bert' })
      .supported).toBe(true);
    const r = detectPluginForModel({ id: '1', name: 'faq-squad2-modell', source_path: '/models/local_abc123', model_type: 'bert' });
    expect(r.supported && r.plugin.id).toBe('question-answering');
    const m = detectPluginForModel({ id: '2', name: 'bert', source_path: '/m/x', model_type: 'bert', plugin_override: 'reranker' });
    expect(m.supported && m.plugin.id).toBe('reranker');
  });
});

describe('Registrierung und Texte', () => {
  it('beide Plugins sind registriert, vor den generischen Encodern', () => {
    const ids = PLUGINS.map(p => p.id);
    for (const id of ['question-answering', 'reranker']) {
      expect(ids).toContain(id);
      expect(ids.indexOf(id)).toBeLessThan(ids.indexOf('hf-encoder'));
      expect(ids.indexOf(id)).toBeLessThan(ids.indexOf('sentence-embedding'));
    }
    expect(PLUGINS.find(p => p.id === 'question-answering')!.taskType).toBe('question_answering');
    expect(PLUGINS.find(p => p.id === 'reranker')!.taskType).toBe('reranker');
    expect(hiddenTrainingFieldsForTaskType('reranker')).toContain('lora');
  });

  it('Namen, Testtexte und Kennzahl-Beschriftungen gibt es auf Englisch und Deutsch', () => {
    for (const loc of [en, de] as Array<Record<string, any>>) {
      for (const id of ['question-answering', 'reranker']) {
        expect(loc.plugins[id].name).toBeTruthy();
        expect(loc.plugins[id].description).toBeTruthy();
      }
      const g = loc.testPlugins.generic;
      for (const k of ['qaLabel', 'qaPlaceholder', 'qaQuestionLabel', 'qaQuestionPlaceholder', 'qaResult',
        'rerankLabel', 'rerankPlaceholder', 'rerankResult']) {
        expect(g[k], k).toBeTruthy();
      }
      for (const k of ['exact_match', 'f1', 'mrr_at_10', 'ndcg_at_10', 'spearman']) {
        expect(g.metricLabels[k], k).toBeTruthy();
      }
      expect(loc.datasetCompat.msg.missing.qa).toBeTruthy();
    }
  });

  it('keine Emojis in den neuen Texten', () => {
    const emoji = /\p{Extended_Pictographic}/u;
    for (const loc of [en, de] as Array<Record<string, any>>) {
      const txt = JSON.stringify([loc.plugins['question-answering'], loc.plugins.reranker,
        Object.entries(loc.testPlugins.generic).filter(([k]) => k.startsWith('qa') || k.startsWith('rerank'))]);
      expect(emoji.test(txt)).toBe(false);
    }
  });
});

describe('Dataset-Pruefung', () => {
  const plugin = (taskType: string) => ({ name: 'X', taskType, supportedDatasetTypes: ['flat_file' as const] });
  it('Bilder passen nicht zu Frage-Antwort oder Reranker', () => {
    expect(genericDatasetCompat(plugin('question_answering') as never, null, ['.jpg', '.png']).overallLevel).toBe('bad');
    expect(genericDatasetCompat(plugin('reranker') as never, null, ['.jpg']).overallLevel).toBe('bad');
  });
  it('JSON/JSONL/CSV passen', () => {
    expect(genericDatasetCompat(plugin('question_answering') as never, null, ['.json']).overallLevel).not.toBe('bad');
    expect(genericDatasetCompat(plugin('reranker') as never, null, ['.jsonl']).overallLevel).not.toBe('bad');
  });
});
