// Labor-Samples passend zur Aufgabe des Modells.
//
// Der allgemeine Lader kennt zwei Faelle: Mediendateien (Label = Ordnername)
// und Textzeilen. Fuer die neueren Aufgaben reicht das nicht — beim Live-Test
// fielen auf:
//   - Spracherkennung: Soll war der Ordnername statt des Transkripts.
//   - Video: Videodateien wurden gar nicht als Samples erkannt.
//   - Text-zu-Bild: geladen wurden die Bilder; das Modell braucht die Captions.
//   - Vision-Language: Frage und Soll-Antwort aus dem Dataset fehlten.
//   - NER: eine CoNLL-Datei zerfiel in Einzelzeilen ("EU B-ORG").
//   - Embeddings: Paare kamen als einzelne Saetze an.
// Die Spaltennamen folgen den Python-Ladern (ft_data/asr.py, image_text.py,
// tokens.py, embedding.py), damit das Labor dieselben Datasets versteht wie
// das Training.

import { parseDelimitedRows } from './csvRows';

export type LabTask = 'asr' | 'video' | 'vlm' | 'text_to_image' | 'token' | 'embedding' | 'other';
export type MediaKind = 'image' | 'audio' | 'video';

export interface DatasetFileInfo { name: string; path: string; is_dir: boolean; split: string }

export interface ExpectedEntity { text: string; label: string; start: number; end: number }

/** Ein Sample, bevor LaboratoryPanel ID und Index vergibt. */
export interface TaskSampleDraft {
  text: string;
  label?: string;
  rawData: unknown;
  filePath?: string;
  fileKind?: MediaKind;
  /** VLM: Frage zum Bild aus dem Dataset. */
  question?: string;
  /** Text-zu-Bild: das Bild, zu dem die Caption gehoert. */
  refImage?: string;
  /** NER: Soll-Entitaeten mit Zeichenpositionen in `text`. */
  entities?: ExpectedEntity[];
}

/** Obergrenze: das Labor ist zum Durchklicken, nicht fuer 50 000 Dateien. */
export const MAX_TASK_SAMPLES = 500;

const IMAGE_EXT = /\.(jpe?g|png|bmp|webp|gif|tiff?)$/i;
const AUDIO_EXT = /\.(wav|mp3|flac|ogg|m4a|aac)$/i;
const VIDEO_EXT = /\.(mp4|mov|m4v|webm|mkv|avi)$/i;
const TABLE_EXT = /\.(jsonl|json|csv|tsv)$/i;
const CONLL_EXT = /\.(conll|conllu|iob|bio|txt)$/i;

export function mediaKindOf(name: string): MediaKind | null {
  if (IMAGE_EXT.test(name)) return 'image';
  if (AUDIO_EXT.test(name)) return 'audio';
  if (VIDEO_EXT.test(name)) return 'video';
  return null;
}

/** Aufgabe aus Plugin (vor dem Laden) oder Server-Modalitaet (nach dem Laden). */
export function labTaskFor(taskType?: string | null, modality?: string | null): LabTask {
  const m = modality ?? '';
  const tt = taskType ?? '';
  if (m === 'asr' || tt === 'speech_recognition') return 'asr';
  if (m === 'video' || tt === 'video_classification') return 'video';
  if (m === 'vlm' || tt === 'vision_language') return 'vlm';
  if (m === 'text_to_image' || tt === 'text_to_image_lora') return 'text_to_image';
  if (m === 'token' || tt === 'token_classification') return 'token';
  if (m === 'embedding' || tt === 'sentence_embedding') return 'embedding';
  return 'other';
}

const baseName = (p: string) => p.split(/[/\\]/).pop() ?? p;
const dirName = (p: string) => { const parts = p.split(/[/\\]/); parts.pop(); return parts.join('/'); };
/** Dateiname ohne Endung, klein — Schluessel fuer "gleichnamige" Dateien. */
export const stemKey = (p: string) => baseName(p).replace(/\.[^.]+$/, '').toLowerCase();
/**
 * Ordnername als Klasse (ImageFolder-Konvention). Struktur-Ordner wie
 * "images" (YOLO) oder "test" sind keine Klasse — im Live-Test stand bei
 * einem YOLO-Dataset "erwartet: images" unter jedem Bild.
 */
const STRUCTURE_FOLDERS = new Set(['images', 'image', 'imgs', 'img', 'audio', 'audios', 'wavs', 'clips', 'videos', 'video',
  'train', 'training', 'val', 'valid', 'validation', 'test', 'testing', 'data', 'dataset', 'samples', 'files', 'media']);
export const folderLabel = (p: string): string | undefined => {
  const parts = p.split(/[/\\]/);
  const folder = parts.length >= 2 ? parts[parts.length - 2] : undefined;
  return folder && !STRUCTURE_FOLDERS.has(folder.toLowerCase()) && !folder.startsWith('ds_') ? folder : undefined;
};
const parentFolder = folderLabel;

/** Tabelle (CSV/TSV/JSONL/JSON) als Zeilen-Objekte. */
export function parseTable(content: string, fileName: string): Record<string, unknown>[] {
  const body = content.replace(/^﻿/, '');
  if (/\.(csv|tsv)$/i.test(fileName)) {
    return parseDelimitedRows(body, /\.tsv$/i.test(fileName) ? '\t' : ',')
      .filter((r): r is Record<string, string> => typeof r === 'object');
  }
  const trimmed = body.trim();
  if (/\.json$/i.test(fileName) && (trimmed.startsWith('[') || trimmed.startsWith('{'))) {
    try {
      const data = JSON.parse(trimmed);
      if (Array.isArray(data)) return data.filter(r => r && typeof r === 'object');
      for (const key of ['data', 'rows', 'items', 'annotations', 'samples']) {
        if (Array.isArray(data?.[key])) return data[key].filter((r: unknown) => r && typeof r === 'object');
      }
      return [];
    } catch { /* evtl. JSONL mit .json-Endung */ }
  }
  const rows: Record<string, unknown>[] = [];
  for (const line of body.split(/\r?\n/)) {
    const l = line.trim();
    if (!l.startsWith('{')) continue;
    try { rows.push(JSON.parse(l)); } catch { /* kaputte Zeile auslassen */ }
  }
  return rows;
}

/** Erster nicht-leerer Wert unter einem der Schluessel (Gross/klein egal). */
export function pick(row: Record<string, unknown>, keys: readonly string[]): string | undefined {
  return pickEntry(row, keys)?.[1];
}

/** Wie pick, liefert aber auch die gefundene Spalte. */
function pickEntry(row: Record<string, unknown>, keys: readonly string[]): [string, string] | undefined {
  const lower = new Map(Object.keys(row).map(k => [k.toLowerCase(), k]));
  for (const key of keys) {
    const real = lower.get(key.toLowerCase());
    if (real === undefined) continue;
    let v = row[real];
    if (Array.isArray(v)) v = v[0];
    if (v && typeof v === 'object') {
      const o = v as Record<string, unknown>;
      v = o.text ?? o.answer ?? o.path ?? o.file_name;
    }
    if (typeof v === 'number') return [real, String(v)];
    if (typeof v === 'string' && v.trim()) return [real, v.trim()];
  }
  return undefined;
}

// Spaltennamen wie in ft_data (Reihenfolge = Prioritaet).
const ASR_TEXT = ['transcription', 'transcript', 'sentence', 'text', 'normalized_text', 'raw_transcription', 'raw_text', 'caption', 'label_text'];
const MEDIA_FILE = ['file_name', 'path', 'file', 'filename', 'audio_path', 'audio_filepath', 'audio', 'image', 'image_path', 'img', 'image_file', 'video', 'video_path'];
const QUESTION = ['question', 'prompt', 'query', 'instruction', 'input'];
const ANSWER = ['answer', 'answers', 'caption', 'response', 'output', 'target', 'label', 'text'];
const CAPTION = ['text', 'caption', 'prompt', 'captions', 'description', 'answer'];
const TOKENS = ['tokens', 'words', 'token', 'sentence_tokens'];
const TAGS = ['ner_tags', 'labels', 'tags', 'ner', 'pos_tags', 'upos', 'label', 'tag'];
const ANCHOR = ['anchor', 'query', 'question', 'sentence1', 'sentence_a', 'text1', 'premise', 'sentence', 'source', 'q'];
const POSITIVE = ['positive', 'document', 'answer', 'passage', 'sentence2', 'sentence_b', 'text2', 'hypothesis', 'pos', 'target', 'context', 'doc'];
const SCORE = ['score', 'similarity', 'relatedness_score', 'sim', 'similarity_score', 'label'];

type Reader = (path: string) => Promise<string>;

/** Pfad aus einer Tabelle zu einer vorhandenen Datei aufloesen (relativ oder nur Name). */
function resolveMedia(ref: string, tablePath: string, byName: Map<string, string>): string | undefined {
  const name = baseName(ref).toLowerCase();
  if (byName.has(name)) return byName.get(name);
  if (/^([a-z]:)?[/\\]/i.test(ref)) return ref;
  return `${dirName(tablePath)}/${ref.replace(/^\.\//, '')}`;
}

async function readTables(files: DatasetFileInfo[], read: Reader): Promise<{ file: DatasetFileInfo; rows: Record<string, unknown>[] }[]> {
  const out = [];
  // metadata.* zuerst (HF-imagefolder-Konvention), dann der Rest.
  const tables = files.filter(f => TABLE_EXT.test(f.name))
    .sort((a, b) => Number(!/^metadata\./i.test(a.name)) - Number(!/^metadata\./i.test(b.name)));
  for (const file of tables) {
    try { out.push({ file, rows: parseTable(await read(file.path), file.name) }); } catch { /* nicht lesbar */ }
  }
  return out;
}

/** Gleichnamige .txt je Mediendatei (so exportiert die Datensatz-Werkstatt). */
async function sidecarTexts(media: DatasetFileInfo[], files: DatasetFileInfo[], read: Reader): Promise<Map<string, string>> {
  const txtByKey = new Map<string, string>();
  for (const f of files) {
    if (/\.txt$/i.test(f.name) && !/\.trans\.txt$/i.test(f.name)) txtByKey.set(`${dirName(f.path)}/${stemKey(f.path)}`, f.path);
  }
  const out = new Map<string, string>();
  for (const m of media.slice(0, MAX_TASK_SAMPLES)) {
    const txt = txtByKey.get(`${dirName(m.path)}/${stemKey(m.path)}`);
    if (!txt) continue;
    try {
      const text = (await read(txt)).trim();
      if (text) out.set(m.path, text);
    } catch { /* fehlt eben */ }
  }
  return out;
}

async function asrSamples(files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  const audio = files.filter(f => mediaKindOf(f.name) === 'audio');
  if (!audio.length) return null;
  const transcripts = new Map<string, string>(); // stem -> Text
  for (const { rows } of await readTables(files, read)) {
    for (const row of rows) {
      const file = pick(row, MEDIA_FILE);
      const text = pick(row, ASR_TEXT);
      if (file && text) transcripts.set(stemKey(file), text);
    }
  }
  // LibriSpeech: "<id> <Transkript>" je Zeile in *.trans.txt
  for (const f of files.filter(f => /\.trans\.txt$/i.test(f.name))) {
    try {
      for (const line of (await read(f.path)).split(/\r?\n/)) {
        const m = line.match(/^(\S+)\s+(.+)$/);
        if (m) transcripts.set(m[1].toLowerCase(), m[2].trim());
      }
    } catch { /* weiter */ }
  }
  const sidecars = await sidecarTexts(audio, files, read);
  return audio.slice(0, MAX_TASK_SAMPLES).map(f => ({
    text: f.name,
    label: sidecars.get(f.path) ?? transcripts.get(stemKey(f.path)),
    rawData: { path: f.path, name: f.name },
    filePath: f.path,
    fileKind: 'audio' as const,
  }));
}

function videoSamples(files: DatasetFileInfo[]): TaskSampleDraft[] | null {
  const videos = files.filter(f => mediaKindOf(f.name) === 'video');
  if (!videos.length) return null;
  return videos.slice(0, MAX_TASK_SAMPLES).map(f => ({
    text: f.name,
    label: parentFolder(f.path),
    rawData: { path: f.path, name: f.name },
    filePath: f.path,
    fileKind: 'video' as const,
  }));
}

/** Bild + Text aus Tabellen oder gleichnamigen .txt (VLM und Text-zu-Bild). */
async function imageTextPairs(files: DatasetFileInfo[], read: Reader, textKeys: readonly string[], withQuestion: boolean) {
  const images = files.filter(f => mediaKindOf(f.name) === 'image');
  const byName = new Map(images.map(f => [f.name.toLowerCase(), f.path]));
  const pairs: { image?: string; text?: string; question?: string; row: unknown }[] = [];
  for (const { file, rows } of await readTables(files, read)) {
    for (const row of rows) {
      const conv = chatImageRow(row);
      if (conv) {
        pairs.push({ image: conv.image ? resolveMedia(conv.image, file.path, byName) : undefined, question: conv.question, text: conv.answer, row });
        continue;
      }
      const ref = pick(row, MEDIA_FILE);
      // Eine Spalte ist entweder Frage oder Antwort, nie beides.
      const q = withQuestion ? pickEntry(row, QUESTION) : undefined;
      const question = q?.[1];
      const text = pick(row, textKeys.filter(k => k.toLowerCase() !== q?.[0].toLowerCase()));
      if (!ref && !text) continue;
      pairs.push({ image: ref ? resolveMedia(ref, file.path, byName) : undefined, text, question, row });
    }
  }
  if (!pairs.length) {
    const sidecars = await sidecarTexts(images, files, read);
    for (const img of images) {
      const text = sidecars.get(img.path);
      if (text) pairs.push({ image: img.path, text, row: { path: img.path } });
    }
  }
  return { images, pairs };
}

/** VLM im Chat-Format: {"messages": [{"role":"user","content":[{"type":"image"},{"type":"text",…}]}, …]} */
function chatImageRow(row: Record<string, unknown>): { image?: string; question?: string; answer?: string } | null {
  const turns = (row.messages ?? row.conversations ?? row.conversation) as unknown;
  if (!Array.isArray(turns)) return null;
  let image = pick(row, ['image', 'images', 'image_path', 'file_name']);
  let question: string | undefined;
  let answer: string | undefined;
  for (const t of turns) {
    if (!t || typeof t !== 'object') continue;
    const m = t as Record<string, unknown>;
    const role = String(m.role ?? m.from ?? '').toLowerCase();
    const parts = Array.isArray(m.content) ? m.content : [m.content ?? m.value];
    const texts: string[] = [];
    for (const p of parts) {
      if (typeof p === 'string') texts.push(p.replace('<image>', '').trim());
      else if (p && typeof p === 'object') {
        const o = p as Record<string, unknown>;
        if (typeof o.text === 'string') texts.push(o.text);
        const img = o.image ?? o.path ?? o.url ?? o.image_url;
        if (typeof img === 'string' && !image) image = img;
      }
    }
    const text = texts.filter(Boolean).join('\n');
    if (role === 'user' || role === 'human') question = text;
    else if (['assistant', 'gpt', 'model', 'bot'].includes(role)) answer = text;
  }
  return question || answer ? { image, question, answer } : null;
}

async function vlmSamples(files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  const { images, pairs } = await imageTextPairs(files, read, ANSWER, true);
  const withImage = pairs.filter(p => p.image);
  if (withImage.length) {
    return withImage.slice(0, MAX_TASK_SAMPLES).map(p => ({
      text: baseName(p.image!),
      label: p.text,
      question: p.question,
      rawData: p.row,
      filePath: p.image,
      fileKind: 'image' as const,
    }));
  }
  if (!images.length) return null;
  // Ein Ordner pro Antwort (Bildunterschrift = Ordnername).
  return images.slice(0, MAX_TASK_SAMPLES).map(f => ({
    text: f.name, label: parentFolder(f.path), rawData: { path: f.path }, filePath: f.path, fileKind: 'image' as const,
  }));
}

async function textToImageSamples(files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  const { pairs } = await imageTextPairs(files, read, CAPTION, false);
  const prompts = pairs.filter(p => p.text);
  if (!prompts.length) return null;
  // Prompt ist die Eingabe; das Dataset-Bild zeigt, wie es aussehen soll.
  return prompts.slice(0, MAX_TASK_SAMPLES).map(p => ({ text: p.text!, rawData: p.row, refImage: p.image }));
}

// ── NER ───────────────────────────────────────────────────────────────────

/** Entitaeten aus BIO/BIOES-Tags; `text` ist die mit Leerzeichen verbundene Tokenfolge. */
export function entitiesFromTags(tokens: string[], tags: string[]): { text: string; entities: ExpectedEntity[] } {
  const entities: ExpectedEntity[] = [];
  let pos = 0;
  const offsets = tokens.map(t => { const start = pos; pos += t.length + 1; return [start, start + t.length] as const; });
  let cur: ExpectedEntity | null = null;
  tokens.forEach((_, i) => {
    const tag = tags[i] ?? 'O';
    const m = tag.match(/^([BIESLU])-(.+)$/);
    const kind = m ? m[1] : tag === 'O' ? 'O' : 'B';
    const label = m ? m[2] : tag;
    const continues = cur && (kind === 'I' || kind === 'E' || kind === 'L') && cur.label === label;
    if (tag === 'O' || tag === '' || !continues) {
      if (cur) entities.push(cur);
      cur = null;
    }
    if (tag !== 'O' && tag !== '' && !continues) {
      cur = { text: '', label, start: offsets[i][0], end: offsets[i][1] };
    } else if (continues && cur) {
      cur.end = offsets[i][1];
    }
    if (cur && (kind === 'E' || kind === 'L' || kind === 'S' || kind === 'U')) { entities.push(cur); cur = null; }
  });
  if (cur) entities.push(cur);
  const text = tokens.join(' ');
  for (const e of entities) e.text = text.slice(e.start, e.end);
  return { text, entities };
}

/** Kurzfassung wie im Modell-Server: "EU [ORG], Deutschland [LOC]". */
export const formatEntities = (ents: { text: string; label: string }[]) =>
  ents.map(e => `${e.text} [${e.label}]`).join(', ');

/** CoNLL: ein Token je Zeile, Tag in der letzten Spalte, Leerzeile trennt Saetze. */
export function parseConll(content: string): { tokens: string[]; tags: string[] }[] {
  const sentences: { tokens: string[]; tags: string[] }[] = [];
  let tokens: string[] = [];
  let tags: string[] = [];
  const flush = () => { if (tokens.length) sentences.push({ tokens, tags }); tokens = []; tags = []; };
  for (const raw of content.replace(/^﻿/, '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) { flush(); continue; }
    if (line.startsWith('-DOCSTART-') || line.startsWith('#')) continue;
    const cols = line.split(/\s+/);
    if (cols.length < 2) continue;
    tokens.push(cols[0]);
    tags.push(cols[cols.length - 1]);
  }
  flush();
  return sentences;
}

function tokenDraft(tokens: string[], tags: string[], row: unknown): TaskSampleDraft {
  const { text, entities } = entitiesFromTags(tokens, tags);
  return { text, label: formatEntities(entities) || undefined, rawData: row, entities };
}

async function tokenSamples(files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  const out: TaskSampleDraft[] = [];
  for (const f of files) {
    if (out.length >= MAX_TASK_SAMPLES) break;
    let content: string;
    try { content = await read(f.path); } catch { continue; }
    if (TABLE_EXT.test(f.name)) {
      for (const row of parseTable(content, f.name)) {
        const lower = new Map(Object.keys(row).map(k => [k.toLowerCase(), k]));
        const tk = TOKENS.map(k => lower.get(k)).find(k => k && Array.isArray(row[k]));
        const tg = TAGS.map(k => lower.get(k)).find(k => k && Array.isArray(row[k]));
        if (!tk || !tg) continue;
        // Zahlen-Tags ohne Namensliste bleiben Zahlen — besser als gar nichts.
        out.push(tokenDraft((row[tk] as unknown[]).map(String), (row[tg] as unknown[]).map(String), row));
      }
    } else if (CONLL_EXT.test(f.name)) {
      for (const s of parseConll(content)) out.push(tokenDraft(s.tokens, s.tags, s));
    }
  }
  return out.length ? out.slice(0, MAX_TASK_SAMPLES) : null;
}

// ── Embeddings ────────────────────────────────────────────────────────────

/** Trenner fuer "Satz A ||| Satz B" — so vergleicht der Modell-Server zwei Saetze. */
export const PAIR_SEPARATOR = ' ||| ';

async function embeddingSamples(files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  const out: TaskSampleDraft[] = [];
  for (const { rows } of await readTables(files, read)) {
    for (const row of rows) {
      const a = pick(row, ANCHOR);
      const p = pick(row, POSITIVE);
      if (!a || !p || a === p) continue;
      const score = pick(row, SCORE);
      out.push({ text: `${a}${PAIR_SEPARATOR}${p}`, label: score, rawData: row });
      if (out.length >= MAX_TASK_SAMPLES) return out;
    }
  }
  return out.length ? out : null;
}

/**
 * Samples fuer Aufgaben mit eigener Datenform. null = der allgemeine Lader
 * ist zustaendig (Klassifikation, LLM, Seq2Seq …) oder nichts Passendes da.
 */
export async function buildTaskSamples(task: LabTask, files: DatasetFileInfo[], read: Reader): Promise<TaskSampleDraft[] | null> {
  switch (task) {
    case 'asr': return asrSamples(files, read);
    case 'video': return videoSamples(files);
    case 'vlm': return vlmSamples(files, read);
    case 'text_to_image': return textToImageSamples(files, read);
    case 'token': return tokenSamples(files, read);
    case 'embedding': return embeddingSamples(files, read);
    default: return null;
  }
}

// ── Auswertung ────────────────────────────────────────────────────────────

/** Wortfehlerrate wie jiwer: (Ersetzungen + Loeschungen + Einfuegungen) / Soll-Woerter. */
export function wordErrorRate(reference: string, hypothesis: string): number {
  const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, ' ').split(/\s+/).filter(Boolean);
  const ref = norm(reference);
  const hyp = norm(hypothesis);
  if (!ref.length) return hyp.length ? 1 : 0;
  let prev = Array.from({ length: hyp.length + 1 }, (_, j) => j);
  for (let i = 1; i <= ref.length; i++) {
    const cur = [i];
    for (let j = 1; j <= hyp.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[hyp.length] / ref.length;
}
