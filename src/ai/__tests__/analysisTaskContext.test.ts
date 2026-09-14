// Regression aus dem App-Durchgang vom 14.09.2026: Die KI-Analyse eines
// Canvas-Modells empfahl Dropout und Warmup, die das Netz gar nicht hat.
import { describe, it, expect } from 'vitest';
import { canvasGraphSummary, canvasSettingsLocation, sanitizeChatJson, unavailableAnalysisFields } from '../analysisTaskContext';

const graph = {
  nodes: [
    { id: 'l', type: 'loss', params: { type: 'cross_entropy' } },
    { id: 'c', type: 'csv_loader', params: { targetCol: 'label', separator: ',' } },
    { id: 'd', type: 'dense', params: { inputSize: 4, outputSize: 2, bias: true } },
    { id: 'o', type: 'output_node', params: { numClasses: 2 } },
  ],
  execution_order: ['c', 'd', 'o', 'l'],
};

describe('analysisTaskContext', () => {
  it('beschreibt den Graph in Ausfuehrungsreihenfolge und nennt fehlende Regularisierung', () => {
    const s = canvasGraphSummary(graph)!;
    expect(s).toContain('csv_loader(targetCol=label) -> dense(inputSize=4, outputSize=2) -> output_node(numClasses=2) -> loss(type=cross_entropy)');
    expect(s).toContain("Nicht im Graph vorhanden: dropout = Node 'Dropout' (Parameter p)");
    expect(canvasGraphSummary({ nodes: [] })).toBeNull();
  });

  it('Warmup: kein eigener Node, Feld im LR Scheduler, Steps pro Epoche fuer die Rechnung', () => {
    // 1.2.72: Der Chat empfahl "vor dem Scheduler einen Warmup-Node einfuegen"
    // und rechnete "25 Epochen = 7 Warmup-Schritte".
    const s = canvasGraphSummary({ ...graph, training: { scheduler: 'cosine', warmupSteps: 0 } }, 70 / 10)!;
    expect(s).toContain("scheduler = Node 'LR Scheduler' (Parameter Schedule, Warmup Steps, Min LR)");
    expect(s).toContain('Scheduler im Training: cosine, warmupSteps=0 (kein Warmup). Einen eigenen Warmup-Node gibt es nicht.');
    expect(s).toContain('Optimizer-Steps pro Epoche: 7.');
    expect(canvasGraphSummary(graph)).not.toContain('Steps pro Epoche');
  });

  it('Canvas: nur Trainingswerte aus dem Synapse Builder sind empfehlbar', () => {
    const fields = ['epochs', 'learning_rate', 'dropout', 'warmup_ratio', 'max_seq_length', 'use_lora', 'gradient_accumulation_steps'];
    const u = unavailableAnalysisFields('canvas', fields);
    expect([...fields].filter(f => !u.has(f))).toEqual(['epochs', 'learning_rate', 'gradient_accumulation_steps']);
  });

  it('Textmodelle behalten alle Felder', () => {
    expect(unavailableAnalysisFields('seq_classification', ['dropout', 'warmup_ratio']).size).toBe(0);
  });

  it('Chat: JSON-Felder, die es nicht gibt oder die bei Canvas nicht wirken, fallen raus', () => {
    const settable = ['epochs', 'learning_rate', 'dropout', 'warmup_steps'];
    const u = unavailableAnalysisFields('canvas', settable);
    const reply = 'Empfehlung\n\n```json\n{"dropout_rate": 0.4, "warmup_steps": 4}\n```\n\nFazit';
    expect(sanitizeChatJson(reply, settable, u)).toBe('Empfehlung\n\nFazit');
    const mixed = '```json\n{"learning_rate": 0.0003, "dropout": 0.2}\n```';
    expect(sanitizeChatJson(mixed, settable, u)).toBe('```json\n{\n  "learning_rate": 0.0003\n}\n```');
    expect(sanitizeChatJson('```json\n{kaputt\n```', settable, u)).toBe('```json\n{kaputt\n```');
  });

  it('nennt die echten Orte im Synapse Builder (1.2.75: "Nodes Epochen und Lernrate", "oben rechts im Trainings-Header")', () => {
    const de = canvasSettingsLocation('de');
    expect(de).toContain('linken Seitenleiste');
    expect(de).toContain('Tab "Konfiguration": Epochen, Batch-Größe, Lernrate');
    expect(de).toContain('Node "LR Scheduler": Schedule, Warmup Steps, Min LR');
    expect(de).toContain('keine Nodes "Epochen" oder "Lernrate"');
    expect(canvasSettingsLocation('en')).toContain('tab "Config"');
  });
});
