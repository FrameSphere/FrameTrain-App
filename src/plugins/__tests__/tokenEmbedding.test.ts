// Token-Klassifikation (NER/POS) und Sentence Embeddings.
//
// Beide Aufgaben laufen auf denselben Architekturen wie die
// Sequenzklassifikation (BERT, RoBERTa ...). Die Erkennung darf deshalb nur
// bei eindeutigem Signal greifen — ein Basis-BERT bleibt bei hf-encoder, sonst
// trainierte jeder Textklassifikator ploetzlich NER.

import { describe, it, expect } from 'vitest';
import { detectPlugin, detectPluginForModel, getPluginById, PLUGINS } from '../registry';
import { detectTokenClassification } from '../token-classification/detect';
import { detectSentenceEmbedding } from '../sentence-embedding/detect';
import { checkDatasetCompat } from '../datasetCompat';
import type { DatasetAnalysis } from '../datasetCompatHelpers';
import tokenPlugin from '../token-classification';
import embeddingPlugin from '../sentence-embedding';
import { metricLabel } from '../GenericTestPanel';

const pluginOf = (id: string, config?: Parameters<typeof detectPlugin>[1]) => {
  const r = detectPlugin(id, config);
  return r.supported ? r.plugin.id : null;
};

describe('Token-Klassifikation – Erkennung', () => {
  it.each([
    ['dslim/bert-base-NER', 'token-classification'],
    ['dbmdz/bert-large-cased-finetuned-conll03-english', 'token-classification'],
    ['Davlan/xlm-roberta-base-ner-hrl', 'token-classification'],
    ['vblagoje/bert-english-uncased-finetuned-pos', 'token-classification'],
    ['QCRI/bert-base-multilingual-cased-pos-english', 'token-classification'],
  ])('%s -> %s', (id, plugin) => expect(pluginOf(id)).toBe(plugin));

  it('erkennt ...ForTokenClassification in der config.json – auch ohne Namenshinweis', () => {
    expect(pluginOf('/models/mein-modell', {
      model_type: 'bert', architectures: ['BertForTokenClassification'],
    })).toBe('token-classification');
    // Vorher gewann xlm-roberta, obwohl die Architektur eindeutig NER ist.
    expect(pluginOf('/models/x', {
      model_type: 'xlm-roberta', architectures: ['XLMRobertaForTokenClassification'],
    })).toBe('token-classification');
  });

  it('Basis-BERT ohne Hinweis bleibt bei der Sequenzklassifikation', () => {
    expect(pluginOf('bert-base-uncased')).toBe('hf-encoder');
    expect(pluginOf('/models/bert', { model_type: 'bert' })).toBe('hf-encoder');
    expect(pluginOf('/models/x', { model_type: 'bert', architectures: ['BertForSequenceClassification'] })).toBe('hf-encoder');
    expect(pluginOf('xlm-roberta-base')).toBe('xlm-roberta');
  });

  it('Wortgrenzen: "ner" steckt nicht in "planner", "pos" nicht in "purpose"', () => {
    expect(detectTokenClassification('acme/bert-planner')).toBe(false);
    expect(detectTokenClassification('acme/purpose-bert')).toBe(false);
  });

  it('Decoder mit "ner" im Namen sind kein NER-Encoder', () => {
    expect(detectTokenClassification('acme/llama-ner')).toBe(false);
    expect(detectTokenClassification('/m/x', { model_type: 'llama' })).toBe(false);
  });
});

describe('Sentence Embeddings – Erkennung', () => {
  it.each([
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'BAAI/bge-small-en-v1.5',
    'BAAI/bge-m3',
    'intfloat/e5-small-v2',
    'intfloat/multilingual-e5-base',
    'thenlper/gte-small',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
  ])('%s -> sentence-embedding', id => expect(pluginOf(id)).toBe('sentence-embedding'));

  it('steht vor hf-encoder, auch wenn model_type bekannt ist', () => {
    expect(pluginOf('BAAI/bge-base-en-v1.5', { model_type: 'bert' })).toBe('sentence-embedding');
  });

  it('Cross-Encoder und Reranker sind Sequenzklassifikation', () => {
    expect(detectSentenceEmbedding('cross-encoder/stsb-roberta-base')).toBe(false);
    expect(detectSentenceEmbedding('BAAI/bge-reranker-base')).toBe(false);
    expect(pluginOf('cross-encoder/ms-marco-MiniLM-L-6-v2')).toBe('hf-encoder');
  });

  it('Decoder-Embeddings (gte-Qwen2) und fremde Architekturen nicht', () => {
    expect(detectSentenceEmbedding('Alibaba-NLP/gte-Qwen2-7B-instruct')).toBe(false);
    expect(detectSentenceEmbedding('/m/x', { model_type: 'nomic_bert' })).toBe(false);
  });

  it('mpnet-base ohne v2 bleibt ein Encoder fuer Klassifikation', () => {
    expect(pluginOf('microsoft/mpnet-base')).toBe('hf-encoder');
  });
});

describe('Import: Plugin von Hand waehlbar', () => {
  const IMPORT_DIR = '/Users/x/Library/Application Support/com.frametrain.desktop/models/local_cfc998f409d4';

  it('beide Plugins stehen in der Auswahl des Import-Dialogs', () => {
    // UnknownModelDialog bietet PLUGINS ohne canvas an.
    const choices = PLUGINS.filter(p => p.id !== 'canvas').map(p => p.id);
    expect(choices).toContain('token-classification');
    expect(choices).toContain('sentence-embedding');
    expect(getPluginById('token-classification')?.taskType).toBe('token_classification');
    expect(getPluginById('sentence-embedding')?.taskType).toBe('sentence_embedding');
  });

  it('plugin_override macht aus einem Basis-BERT ein NER-Modell', () => {
    const r = detectPluginForModel({
      id: 'local_1', name: 'bert-base', source_path: IMPORT_DIR, model_type: 'bert',
      plugin_override: 'token-classification',
    });
    expect(r.supported && r.plugin.id).toBe('token-classification');
  });

  it('der vergebene Name schlaegt die Auffang-Erkennung am model_type', () => {
    // Pfad sagt nichts, model_type "bert" allein ergaebe hf-encoder.
    const ner = detectPluginForModel({ id: 'l2', name: 'kunden-ner', source_path: IMPORT_DIR, model_type: 'bert' });
    expect(ner.supported && ner.plugin.id).toBe('token-classification');
    const emb = detectPluginForModel({ id: 'l3', name: 'faq-embeddings', source_path: IMPORT_DIR, model_type: 'bert' });
    expect(emb.supported && emb.plugin.id).toBe('sentence-embedding');
    // Ohne Hinweis im Namen bleibt alles wie bisher.
    const plain = detectPluginForModel({ id: 'l4', name: 'mein-bert', source_path: IMPORT_DIR, model_type: 'bert' });
    expect(plain.supported && plain.plugin.id).toBe('hf-encoder');
  });
});

const analysis = (detected_type: DatasetAnalysis['detected_type'], extensions: string[]): DatasetAnalysis => ({
  detected_type, confidence: 80, pairing_status: null, warnings: [], file_count: 10, dir_count: 0, extensions, schema_hint: null,
});

describe('Dataset-Kompatibilitaet', () => {
  it('NER liest JSONL und CoNLL, aber keine Bilder', () => {
    expect(checkDatasetCompat('token-classification', [], analysis('pre_split', ['.jsonl']), tokenPlugin).overallLevel).toBe('perfect');
    expect(checkDatasetCompat('token-classification', [], analysis('unknown', ['.conll']), tokenPlugin).overallLevel).toBe('perfect');
    expect(checkDatasetCompat('token-classification', [], analysis('folder_class', ['.jpg']), tokenPlugin).overallLevel).toBe('bad');
  });

  it('Embeddings brauchen Tabellen mit Satzpaaren', () => {
    expect(checkDatasetCompat('sentence-embedding', [], analysis('flat_file', ['.csv']), embeddingPlugin).overallLevel).toBe('perfect');
    expect(checkDatasetCompat('sentence-embedding', [], analysis('flat_file', ['.wav']), embeddingPlugin).overallLevel).toBe('bad');
  });
});

describe('Trainingsformular', () => {
  it('blendet LoRA aus, behaelt aber die Sequenzlaenge', () => {
    for (const p of [tokenPlugin, embeddingPlugin]) {
      expect(p.hiddenTrainingFields).toContain('lora');
      expect(p.hiddenTrainingFields).not.toContain('max_seq_length');
    }
  });
});

describe('metricLabel', () => {
  it('macht Metrik-Schluessel lesbar', () => {
    expect(metricLabel('recall_at_1')).toBe('Recall@1');
    expect(metricLabel('mrr_at_10')).toBe('Mrr@10');
    expect(metricLabel('entity_f1')).toBe('Entity f1');
  });
});
