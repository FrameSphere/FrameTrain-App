// Labor-Samples je Aufgabe. Beim Live-Test von 1.4.0 zeigte das Labor bei
// Spracherkennung den Ordnernamen als Soll, erkannte keine Videos, lud bei
// Text-zu-Bild Bilder statt Prompts und zerlegte CoNLL-Dateien in Zeilen.
import { describe, it, expect } from 'vitest';
import {
  buildTaskSamples, entitiesFromTags, folderLabel, labTaskFor, parseConll, wordErrorRate,
  type DatasetFileInfo,
} from '../labTaskSamples';

const f = (path: string, split = 'test'): DatasetFileInfo => ({ name: path.split('/').pop()!, path, is_dir: false, split });
const reader = (files: Record<string, string>) => async (p: string) => {
  if (!(p in files)) throw new Error(`fehlt: ${p}`);
  return files[p];
};

describe('labTaskFor', () => {
  it('nimmt vor dem Laden das Plugin, danach die Server-Modalitaet', () => {
    expect(labTaskFor('speech_recognition', null)).toBe('asr');
    expect(labTaskFor(null, 'video')).toBe('video');
    expect(labTaskFor('text_to_image_lora', null)).toBe('text_to_image');
    expect(labTaskFor('seq_classification', 'text')).toBe('other');
  });
});

describe('Spracherkennung', () => {
  it('Soll ist das Transkript aus metadata.csv, nicht der Ordnername', async () => {
    const files = [f('/d/test/a.wav'), f('/d/test/b.wav'), f('/d/test/metadata.csv')];
    const read = reader({ '/d/test/metadata.csv': 'file_name,transcription\na.wav,hallo welt\nb.wav,"guten tag, alle"\n' });
    const s = await buildTaskSamples('asr', files, read);
    expect(s?.map(x => [x.fileKind, x.label])).toEqual([['audio', 'hallo welt'], ['audio', 'guten tag, alle']]);
  });

  it('liest gleichnamige .txt (Datensatz-Werkstatt) und LibriSpeech *.trans.txt', async () => {
    const files = [f('/d/x/1-2-0001.flac'), f('/d/x/1-2.trans.txt'), f('/d/y/aufnahme.wav'), f('/d/y/aufnahme.txt')];
    const read = reader({ '/d/x/1-2.trans.txt': '1-2-0001 HELLO WORLD\n', '/d/y/aufnahme.txt': ' servus \n' });
    const s = await buildTaskSamples('asr', files, read);
    expect(s?.map(x => x.label)).toEqual(['HELLO WORLD', 'servus']);
  });
});

describe('Video', () => {
  it('Videos werden Samples mit Ordnername als Klasse', async () => {
    const s = await buildTaskSamples('video', [f('/d/test/jonglieren/c1.mp4'), f('/d/test/notes.md')], reader({}));
    expect(s).toEqual([expect.objectContaining({ fileKind: 'video', label: 'jonglieren', filePath: '/d/test/jonglieren/c1.mp4' })]);
  });
});

describe('Text-zu-Bild', () => {
  it('die Caption ist die Eingabe, das Bild die Referenz', async () => {
    const files = [f('/d/train/a.png'), f('/d/train/metadata.jsonl')];
    const read = reader({ '/d/train/metadata.jsonl': '{"file_name": "a.png", "text": "ein roter Hund im Stil xyz"}\n' });
    const s = await buildTaskSamples('text_to_image', files, read);
    expect(s).toEqual([expect.objectContaining({ text: 'ein roter Hund im Stil xyz', refImage: '/d/train/a.png' })]);
    expect(s?.[0].fileKind).toBeUndefined();
  });

  it('ohne Tabelle: Bild + gleichnamige .txt', async () => {
    const s = await buildTaskSamples('text_to_image', [f('/d/a.jpg'), f('/d/a.txt')], reader({ '/d/a.txt': 'sks Katze' }));
    expect(s?.[0]).toMatchObject({ text: 'sks Katze', refImage: '/d/a.jpg' });
  });
});

describe('Vision-Language', () => {
  it('Bild, Frage und Soll-Antwort aus einer Tabelle', async () => {
    const files = [f('/d/img/k.jpg'), f('/d/qa.jsonl')];
    const read = reader({ '/d/qa.jsonl': '{"image": "img/k.jpg", "question": "Welche Farbe?", "answer": "rot"}\n' });
    const s = await buildTaskSamples('vlm', files, read);
    expect(s?.[0]).toMatchObject({ fileKind: 'image', filePath: '/d/img/k.jpg', question: 'Welche Farbe?', label: 'rot' });
  });

  it('Chat-Format mit Bild im Inhalt', async () => {
    const row = { images: ['k.jpg'], messages: [
      { role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'Was ist das?' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Ein Hund' }] },
    ] };
    const s = await buildTaskSamples('vlm', [f('/d/k.jpg'), f('/d/train.jsonl')], reader({ '/d/train.jsonl': JSON.stringify(row) }));
    expect(s?.[0]).toMatchObject({ filePath: '/d/k.jpg', question: 'Was ist das?', label: 'Ein Hund' });
  });

  it('Frage und Antwort kommen nie aus derselben Spalte', async () => {
    const read = reader({ '/d/m.csv': 'file_name,prompt,text\nk.jpg,Beschreibe,Ein Hund\n' });
    const s = await buildTaskSamples('vlm', [f('/d/k.jpg'), f('/d/m.csv')], read);
    expect(s?.[0]).toMatchObject({ question: 'Beschreibe', label: 'Ein Hund' });
  });
});

describe('NER', () => {
  it('CoNLL: ein Satz je Block, Soll-Entitaeten mit Positionen', async () => {
    const conll = '-DOCSTART- O\n\nEU B-ORG\nlehnt O\ndeutsche B-MISC\nIdee O\nab O\n\nPeter B-PER\nBlackburn I-PER\n';
    const s = await buildTaskSamples('token', [f('/d/test.conll')], reader({ '/d/test.conll': conll }));
    expect(s).toHaveLength(2);
    expect(s?.[0]).toMatchObject({ text: 'EU lehnt deutsche Idee ab', label: 'EU [ORG], deutsche [MISC]' });
    expect(s?.[1].entities).toEqual([{ text: 'Peter Blackburn', label: 'PER', start: 0, end: 15 }]);
  });

  it('JSONL mit tokens/ner_tags', async () => {
    const row = { tokens: ['Berlin', 'ist', 'gross'], ner_tags: ['B-LOC', 'O', 'O'] };
    const s = await buildTaskSamples('token', [f('/d/t.jsonl')], reader({ '/d/t.jsonl': JSON.stringify(row) }));
    expect(s?.[0]).toMatchObject({ text: 'Berlin ist gross', label: 'Berlin [LOC]' });
  });

  it('BIOES und Tags ohne Praefix', () => {
    expect(entitiesFromTags(['New', 'York', 'City'], ['B-LOC', 'I-LOC', 'E-LOC']).entities)
      .toEqual([{ text: 'New York City', label: 'LOC', start: 0, end: 13 }]);
    expect(entitiesFromTags(['Anna', 'lacht'], ['PER', 'O']).entities)
      .toEqual([{ text: 'Anna', label: 'PER', start: 0, end: 4 }]);
  });

  it('parseConll nimmt die letzte Spalte als Tag', () => {
    expect(parseConll('EU NNP B-NP B-ORG\n')).toEqual([{ tokens: ['EU'], tags: ['B-ORG'] }]);
  });
});

describe('Embeddings', () => {
  it('Paare werden "A ||| B", der Score das Soll', async () => {
    const read = reader({ '/d/sts.csv': 'sentence1,sentence2,score\nEin Hund rennt,Ein Tier laeuft,4.2\n' });
    const s = await buildTaskSamples('embedding', [f('/d/sts.csv')], read);
    expect(s?.[0]).toMatchObject({ text: 'Ein Hund rennt ||| Ein Tier laeuft', label: '4.2' });
  });

  it('ohne Paar-Spalten zustaendig ist der allgemeine Lader', async () => {
    const s = await buildTaskSamples('embedding', [f('/d/t.csv')], reader({ '/d/t.csv': 'text,label\na,b\n' }));
    expect(s).toBeNull();
  });
});

describe('wordErrorRate', () => {
  it('wie jiwer: Ersetzung, Loeschung, Einfuegung; Gross/klein und Satzzeichen egal', () => {
    expect(wordErrorRate('Hallo Welt.', 'hallo welt')).toBe(0);
    expect(wordErrorRate('eins zwei drei vier', 'eins zwo drei')).toBe(0.5);
    expect(wordErrorRate('a', 'a b c')).toBe(2);
  });
});

// Live-Test 1.4.2: Bei einem YOLO-Dataset stand "erwartet: images" unter jedem Bild.
describe('folderLabel', () => {
  it('Struktur-Ordner sind keine Klasse, Klassen-Ordner schon', () => {
    expect(folderLabel('/d/test/images/a.jpg')).toBeUndefined();
    expect(folderLabel('/d/train/a.jpg')).toBeUndefined();
    expect(folderLabel('/data/ds_abc123/a.jpg')).toBeUndefined();
    expect(folderLabel('/d/test/katze/a.jpg')).toBe('katze');
  });
});
