// Befunde aus dem Screenshot-Durchgang 1.5.2 (EN-Oberflaeche).

import { describe, it, expect } from 'vitest';
import { lossTrend } from '../homeInsights';
import { apiSnippets, TOKEN_PLACEHOLDER, type HostInfo } from '../hosting/hostingModel';
import { localizeScript } from '../scriptLocale';
import { stepLogs } from '../AnalysisPanel';
import { brokenPackageFromError } from '../usePackageInstall';
import { engineErrorText } from '../../plugins/engineError';
import { templateName } from '../../utils/templateText';
import { shownPrediction } from '../labGroundTruth';
import { abschnittText } from '../studio/MediaWorkbench';
import { generateCausalLmTrainScript } from '../llmDevScript';
import { translate } from '../../contexts/LanguageContext';
import { PLUGINS } from '../../plugins/registry';
import { pluginDescription, pluginName } from '../../plugins/pluginText';

const tEn = (k: string, p?: string | Record<string, string | number>) => translate('en', k, p);
const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60_000).toISOString();

describe('Loss-Trend', () => {
  it('vergleicht nur Laeufe desselben Modells wie der juengste', () => {
    const points = lossTrend([
      { id: 'a', model_id: 'yolo', model_name: 'YOLO', status: 'completed', created_at: iso(30), completed_at: iso(20), progress: { train_loss: 2.5 } },
      { id: 'b', model_id: 'bert', model_name: 'BERT', status: 'completed', created_at: iso(20), completed_at: iso(10), progress: { train_loss: 0.4 } },
      { id: 'c', model_id: 'bert', model_name: 'BERT', status: 'completed', created_at: iso(60), completed_at: iso(50), progress: { train_loss: 0.6 } },
    ]);
    expect(points.map(p => p.id)).toEqual(['c', 'b']);
  });
});

describe('API-Beispiele', () => {
  const host = { api_name: 'bert', modality: 'text_classification', input_kind: 'text' } as Pick<HostInfo, 'api_name' | 'modality' | 'input_kind'>;
  it('Python liest den verborgenen Token aus der Umgebung statt "$FRAMETRAIN_TOKEN"', () => {
    const { python, curl } = apiSnippets('http://127.0.0.1:1/v1', TOKEN_PLACEHOLDER, host, 'en');
    expect(python).toContain(`os.environ['FRAMETRAIN_TOKEN']`);
    expect(python).not.toContain('Bearer $FRAMETRAIN_TOKEN');
    expect(curl).toContain('Bearer $FRAMETRAIN_TOKEN');
    expect(python + curl).not.toContain('Beispieltext');
  });
  it('sichtbarer Token steht als normaler String im Python-Header', () => {
    const { python } = apiSnippets('http://127.0.0.1:1/v1', 'abc', host, 'de');
    expect(python).toContain('"Bearer abc"');
    expect(python).toContain('Beispieltext');
  });
});

describe('Dev-Vorlagen', () => {
  const llm = generateCausalLmTrainScript({ modelPath: '/m', datasetPath: '/d', outputPath: '/o' });
  it('LLM-Vorlage nutzt CausalLM und LoRA statt Sequenzklassifikation', () => {
    expect(llm).toContain('AutoModelForCausalLM');
    expect(llm).toContain('get_peft_model');
    expect(llm).not.toContain('AutoModelForSequenceClassification');
  });
  it('englische Fassung ohne deutsche Kommentare und Meldungen', () => {
    const en = localizeScript(llm, 'en');
    expect(en).toContain('# This template runs as is.');
    expect(en).toContain('message="Loading model ..."');
    expect(en).not.toMatch(/Modell gespeichert|wird geladen|Speichern|Dieses Template/);
  });
  it('deutsche Fassung mit Umlauten in Kommentaren, Code unberuehrt', () => {
    const de = localizeScript('# Dieses Template laeuft so wie es ist.\nlaeuft_var = 1', 'de');
    expect(de).toBe('# Dieses Template läuft so wie es ist.\nlaeuft_var = 1');
  });
});

describe('Analyse', () => {
  it('Abschlusszeilen mit gleichem Step fallen aus den Logs', () => {
    const logs = stepLogs([
      { step: 10, val_loss: null, learning_rate: 1e-5 },
      { step: 20, val_loss: null, learning_rate: 1e-6 },
      { step: 20, val_loss: 0.3, learning_rate: 3e-5 },
      { step: 20, val_loss: 0.3, learning_rate: 3e-5 },
    ]);
    expect(logs.map(l => l.learning_rate)).toEqual([1e-5, 1e-6]);
    expect(logs[1].val_loss).toBe(0.3);
  });
});

describe('Python-Pakete', () => {
  it('erkennt scipy hinter einem dlopen-Fehler', () => {
    const b = brokenPackageFromError("ImportError: dlopen(/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/scipy/sparse/_sparsetools.cpython-311-darwin.so, 0x0002): zero-fill section");
    expect(b?.package).toBe('scipy');
  });
  it('ein fehlendes Modul ist nicht kaputt', () => {
    expect(brokenPackageFromError("No module named 'peft'")).toBeNull();
  });
  it('Engine-Fehler: Titel uebersetzt, erste Detailzeile dabei', () => {
    expect(engineErrorText({ error: 'Fehlendes Python-Paket', details: "Traceback (most recent call last):\nNo module named 'sklearn'" }, tEn, '?'))
      .toBe("Missing Python package: No module named 'sklearn'");
  });
});

describe('Uebersetzungen', () => {
  it('jedes Plugin hat englischen Namen und Beschreibung', () => {
    for (const p of PLUGINS) {
      expect(pluginName(p, tEn), p.id).not.toMatch(/Suche|Erkennung|Klassifikation/);
      expect(pluginDescription(p, tEn), p.id).not.toMatch(/ und | mit | oder |Bringt|Feinjustiert|Klassifiziert|Trainiert/);
    }
  });
  it('KI-Vorlagen zeigen den Vorsatz in der aktuellen Sprache', () => {
    expect(templateName({ name: 'KI-Empfehlung · distilbert v6', source: 'ai' }, tEn)).toBe('AI Recommendation · distilbert v6');
    expect(templateName({ name: 'KI-Empfehlung · eigener Name', source: 'user' }, tEn)).toBe('KI-Empfehlung · eigener Name');
  });
  it('"Keine Objekte" erscheint englisch', () => {
    expect(shownPrediction('Keine Objekte', tEn)).toBe('No objects');
    expect(shownPrediction('2x Person', tEn)).toBe('2x Person');
  });
  it('Segmentzeiten mit Punkt im Englischen', () => {
    expect(abschnittText(0, 4, 'en')).toBe('0.0–4.0 s');
    expect(abschnittText(0, 4)).toBe('0,0–4,0 s');
  });
});
