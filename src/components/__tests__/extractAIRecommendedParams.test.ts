import { describe, it, expect } from 'vitest';
import { extractAIRecommendedParams } from '../AnalysisPanel';

describe('extractAIRecommendedParams', () => {
  it('liest einen geschlossenen json-Block', () => {
    const text = [
      '## Empfohlene Parameter',
      '```json',
      '{ "epochs": 5, "batch_size": 12, "learning_rate": 0.000025 }',
      '```',
    ].join('\n');
    expect(extractAIRecommendedParams(text)).toEqual({
      epochs: 5, batch_size: 12, learning_rate: 0.000025,
    });
  });

  // Der Fall aus dem Bug-Report: englische Antwort ohne umschlossenen Block.
  it('liest Inline-Paare, wenn kein Code-Block da ist', () => {
    const text =
      'Try `epochs=5`, `learning_rate=0.000025`, `batch_size=12` and `lora_r=16`.';
    expect(extractAIRecommendedParams(text)).toEqual({
      epochs: 5, learning_rate: 0.000025, batch_size: 12, lora_r: 16,
    });
  });

  it('kommt mit einem nicht geschlossenen Code-Block klar', () => {
    const text = 'Empfehlung:\n```json\n{ "epochs": 3, "optimizer": "adamw" }';
    expect(extractAIRecommendedParams(text)).toEqual({
      epochs: 3, optimizer: 'adamw',
    });
  });

  it('erkennt key: value mit Markdown-Auszeichnung', () => {
    const text = '- **epochs**: 8\n- **weight_decay**: 0.01\n- **fp16**: true';
    expect(extractAIRecommendedParams(text)).toEqual({
      epochs: 8, weight_decay: 0.01, fp16: true,
    });
  });

  it('ignoriert unbekannte Schlüssel und Platzhalter', () => {
    const text = '```json\n{ "epochs": 4, "nonsense_key": 9, "optimizer": "..." }\n```';
    expect(extractAIRecommendedParams(text)).toEqual({ epochs: 4 });
  });

  it('liefert null, wenn nichts Brauchbares drinsteht', () => {
    expect(extractAIRecommendedParams('Das Training lief gut, keine Änderung nötig.')).toBeNull();
  });

  it('greift nicht auf beliebigen Fließtext zu', () => {
    expect(extractAIRecommendedParams('Der Verlauf: loss = 0.42, Tendenz fallend.')).toBeNull();
  });

  // Berichte nennen erst den kritisierten Ist-Zustand, dann die Empfehlung.
  // Der Button darf nie die Werte übernehmen, von denen die KI abrät.
  it('bevorzugt den Empfehlungs-Abschnitt vor dem Ist-Zustand', () => {
    const text = [
      '## Gesamtbewertung',
      'Dein Training lief mit epochs=8, batch_size=16 und learning_rate=0.00005.',
      'Das war zu aggressiv.',
      '',
      '## Empfohlene Parameter für das nächste Training',
      'Reduziere auf epochs=3 und learning_rate=0.00001.',
    ].join('\n');
    expect(extractAIRecommendedParams(text)).toEqual({
      epochs: 3, learning_rate: 0.00001,
    });
  });

  it('nimmt ohne Abschnitts-Überschrift den letzten Treffer', () => {
    const text = 'Bisher epochs=8, künftig besser epochs=3.';
    expect(extractAIRecommendedParams(text)).toEqual({ epochs: 3 });
  });

  it('verwirft Werte mit falschem Typ', () => {
    // "fp16:" gefolgt von einem Artikel ist Fließtext, kein Parameter.
    expect(extractAIRecommendedParams('Bei fp16: die Hardware unterstützt das nicht.')).toBeNull();
    expect(extractAIRecommendedParams('Setze epochs: viele')).toBeNull();
    expect(extractAIRecommendedParams('```json\n{ "optimizer": "irgendwas" }\n```')).toBeNull();
  });

  it('nimmt bei zwei json-Blöcken den letzten', () => {
    const text = [
      'Aktuell:', '```json', '{ "epochs": 8 }', '```',
      'Besser:',  '```json', '{ "epochs": 2 }', '```',
    ].join('\n');
    expect(extractAIRecommendedParams(text)).toEqual({ epochs: 2 });
  });

  // Der reale Fall: die Antwort wurde von max_tokens mitten im JSON gekappt.
  // Frueher rettete nur der Inline-Fallback die Zahlen — "optimizer": "sgd",
  // also die wichtigste Empfehlung ueberhaupt, fiel hinten runter.
  it('liest einen mitten im JSON abgeschnittenen Block', () => {
    const text = [
      '## Empfohlene Parameter',
      '```json',
      '{"epochs":100,"batch_size":16,"learning_rate":0.001,"weight_decay":0.0005,'
        + '"warmup_ratio":0.03,"optimizer":"sgd","scheduler":"cosine","fp16":true,'
        + '"dropout":0.0,"label_smoothing":0.0,"eval_strategy":"epoch","save_steps":200,"logging_steps":20,',
    ].join('\n');
    const params = extractAIRecommendedParams(text);
    expect(params).toMatchObject({
      epochs: 100,
      batch_size: 16,
      learning_rate: 0.001,
      optimizer: 'sgd',
      scheduler: 'cosine',
      fp16: true,
      eval_strategy: 'epoch',
      save_steps: 200,
    });
  });

  // Diese Felder existieren in der Trainings-Config, waren aber nicht
  // uebernehmbar, weil Extraktion und Anwendung zwei getrennte Listen hatten.
  it('erkennt Regularisierungs- und Ablauf-Parameter', () => {
    const text = '```json\n{"dropout":0.1,"label_smoothing":0.05,"max_steps":500,"eval_steps":50,"bf16":true,"gradient_checkpointing":true,"seed":42}\n```';
    expect(extractAIRecommendedParams(text)).toEqual({
      dropout: 0.1, label_smoothing: 0.05, max_steps: 500,
      eval_steps: 50, bf16: true, gradient_checkpointing: true, seed: 42,
    });
  });
});
