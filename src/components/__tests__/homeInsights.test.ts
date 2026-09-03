// Die Startseite behauptet Dinge ueber den Projektstand ("Training fehl-
// geschlagen", "bestes Modell", "Loss faellt"). Falsche Behauptungen sind
// schlimmer als gar keine — deshalb steht die Ableitung hier unter Test.

import { describe, it, expect } from 'vitest';
import {
  buildInsights, lossTrend, trendImprovementPct, sparklinePoints, topResults,
  buildBriefingFacts, factsHash, byNewest, splitBriefing,
  type InsightInput, type TrainingLike, type TestLike,
} from '../homeInsights';

const iso = (minAgo: number) => new Date(Date.now() - minAgo * 60000).toISOString();

function training(over: Partial<TrainingLike> = {}): TrainingLike {
  return {
    id: 'j1', model_id: 'm1', model_name: 'xlm-roberta', dataset_name: 'reviews',
    status: 'completed', created_at: iso(120), completed_at: iso(60),
    progress: { train_loss: 0.3 },
    ...over,
  };
}

function test_(over: Partial<TestLike> = {}): TestLike {
  return {
    id: 't1', model_id: 'm1', model_name: 'xlm-roberta', version_name: 'v1',
    dataset_name: 'reviews-test', status: 'completed', created_at: iso(50),
    completed_at: iso(45), results: { accuracy: 0.9 },
    ...over,
  };
}

function input(over: Partial<InsightInput> = {}): InsightInput {
  return {
    trainings: [], tests: [], models: [], datasets: [],
    hasActiveTraining: false, sleepPrevented: true,
    ...over,
  };
}

describe('buildInsights', () => {
  it('meldet nichts, solange alles unauffaellig ist', () => {
    expect(buildInsights(input({
      trainings: [training()],
      tests: [test_()],
      models: [{ id: 'm1', name: 'xlm-roberta' }],
      datasets: [{ id: 'd1', name: 'reviews', training_count: 3 }],
    }))).toEqual([]);
  });

  it('warnt, wenn waehrend eines Laufs der Ruhezustand nicht unterdrueckt ist', () => {
    const res = buildInsights(input({ hasActiveTraining: true, sleepPrevented: false }));
    expect(res.map(i => i.kind)).toContain('sleepRisk');
    expect(res[0].severity).toBe('warn');
  });

  it('warnt nicht, wenn kein Training laeuft', () => {
    const res = buildInsights(input({ hasActiveTraining: false, sleepPrevented: false }));
    expect(res.map(i => i.kind)).not.toContain('sleepRisk');
  });

  it('meldet ein fehlgeschlagenes Training, solange es keinen neuen Versuch gab', () => {
    const res = buildInsights(input({
      trainings: [training({ id: 'a', status: 'failed', completed_at: iso(10) })],
    }));
    expect(res.map(i => i.kind)).toContain('trainingFailed');
    expect(res[0].params.model).toBe('xlm-roberta');
  });

  it('schweigt, sobald danach ein Lauf desselben Modells erfolgreich war', () => {
    const res = buildInsights(input({
      trainings: [
        training({ id: 'a', status: 'failed', completed_at: iso(120) }),
        training({ id: 'b', status: 'completed', completed_at: iso(10) }),
      ],
    }));
    expect(res.map(i => i.kind)).not.toContain('trainingFailed');
  });

  it('trennt die Modelle — ein fremder Erfolg raeumt die Warnung nicht ab', () => {
    const res = buildInsights(input({
      trainings: [
        training({ id: 'a', model_id: 'm1', model_name: 'A', status: 'failed', completed_at: iso(120) }),
        training({ id: 'b', model_id: 'm2', model_name: 'B', status: 'completed', completed_at: iso(10) }),
      ],
    }));
    const failed = res.filter(i => i.kind === 'trainingFailed');
    expect(failed).toHaveLength(1);
    expect(failed[0].params.model).toBe('A');
  });

  it('erkennt Overfitting-Verdacht am Verhaeltnis von val_loss zu train_loss', () => {
    const res = buildInsights(input({
      trainings: [training({ progress: { train_loss: 0.2, val_loss: 0.9 } })],
    }));
    expect(res.map(i => i.kind)).toContain('overfitting');
  });

  it('sieht ein normales val_loss nicht als Overfitting', () => {
    const res = buildInsights(input({
      trainings: [training({ progress: { train_loss: 0.2, val_loss: 0.25 } })],
    }));
    expect(res.map(i => i.kind)).not.toContain('overfitting');
  });

  it('weist auf ein schwaches Testergebnis hin', () => {
    const res = buildInsights(input({ tests: [test_({ results: { accuracy: 0.42 } })] }));
    const weak = res.find(i => i.kind === 'weakAccuracy');
    expect(weak?.params.value).toBe('42.0');
  });

  it('zaehlt Modelle ohne abgeschlossenen Test', () => {
    const res = buildInsights(input({
      models: [{ id: 'm1', name: 'A' }, { id: 'm2', name: 'B' }],
      tests: [test_({ model_id: 'm1' })],
    }));
    const untested = res.find(i => i.kind === 'untestedModels');
    expect(untested?.params).toEqual({ count: '1', first: 'B' });
  });

  it('zaehlt einen abgebrochenen Test nicht als getestet', () => {
    const res = buildInsights(input({
      models: [{ id: 'm1', name: 'A' }],
      tests: [test_({ model_id: 'm1', status: 'failed' })],
    }));
    expect(res.find(i => i.kind === 'untestedModels')?.params.count).toBe('1');
  });

  it('zaehlt nie benutzte Datasets', () => {
    const res = buildInsights(input({
      datasets: [{ id: 'd1', name: 'alt', training_count: 2 }, { id: 'd2', name: 'neu', training_count: 0 }],
    }));
    expect(res.find(i => i.kind === 'unusedDatasets')?.params).toEqual({ count: '1', first: 'neu' });
  });

  it('stellt Warnungen vor Infos und deckelt die Menge', () => {
    const res = buildInsights(input({
      hasActiveTraining: true, sleepPrevented: false,
      trainings: [
        training({ id: 'a', model_id: 'm1', model_name: 'A', status: 'failed' }),
        training({ id: 'b', model_id: 'm2', model_name: 'B', status: 'failed' }),
        training({ id: 'c', model_id: 'm3', model_name: 'C', status: 'stopped' }),
        training({ id: 'd', model_id: 'm4', model_name: 'D', status: 'failed' }),
        training({ id: 'e', model_id: 'm5', model_name: 'E', status: 'failed' }),
      ],
      datasets: [{ id: 'd1', name: 'neu', training_count: 0 }],
    }));
    expect(res).toHaveLength(5);
    expect(res.every(i => i.severity === 'warn')).toBe(true);
  });
});

describe('lossTrend', () => {
  it('liefert die Punkte in Leserichtung — aeltester zuerst', () => {
    const points = lossTrend([
      training({ id: 'neu', completed_at: iso(10), progress: { train_loss: 0.2 } }),
      training({ id: 'alt', completed_at: iso(500), progress: { train_loss: 0.8 } }),
    ]);
    expect(points.map(p => p.id)).toEqual(['alt', 'neu']);
  });

  it('laesst Laeufe ohne verwertbaren Loss weg', () => {
    const points = lossTrend([
      training({ id: 'a', progress: { train_loss: 0 } }),
      training({ id: 'b', progress: undefined }),
      training({ id: 'c', progress: { train_loss: 0.5 } }),
      training({ id: 'd', status: 'failed', progress: { train_loss: 2.0 } }),
    ]);
    expect(points.map(p => p.id)).toEqual(['c']);
  });

  it('rechnet die Verbesserung ueber den Verlauf', () => {
    const points = lossTrend([
      training({ id: 'alt', completed_at: iso(500), progress: { train_loss: 0.8 } }),
      training({ id: 'neu', completed_at: iso(10), progress: { train_loss: 0.2 } }),
    ]);
    expect(trendImprovementPct(points)).toBeCloseTo(75);
  });

  it('gibt ohne Vergleichspunkt null zurueck', () => {
    expect(trendImprovementPct(lossTrend([training()]))).toBeNull();
  });
});

describe('sparklinePoints', () => {
  it('legt eine flache Reihe auf die Mittellinie statt auf den Rand', () => {
    const points = lossTrend([
      training({ id: 'a', completed_at: iso(200), progress: { train_loss: 0.5 } }),
      training({ id: 'b', completed_at: iso(100), progress: { train_loss: 0.5 } }),
    ]);
    const coords = sparklinePoints(points, 100, 40).split(' ').map(p => Number(p.split(',')[1]));
    expect(coords.every(y => y === 20)).toBe(true);
  });

  it('laesst einen fallenden Loss auch im Bild fallen', () => {
    // SVG zaehlt y von oben: hoher Loss gehoert nach oben (kleines y),
    // niedriger nach unten. Andersherum widerspricht die Kurve der Prozent-
    // angabe daneben ("-73.9%" bei steigender Linie).
    const points = lossTrend([
      training({ id: 'a', completed_at: iso(200), progress: { train_loss: 1.0 } }),
      training({ id: 'b', completed_at: iso(100), progress: { train_loss: 0.0001 } }),
    ]);
    const ys = sparklinePoints(points, 100, 40, 2).split(' ').map(p => Number(p.split(',')[1]));
    expect(ys[0]).toBeCloseTo(2);
    expect(ys[1]).toBeCloseTo(38);
  });

  it('laesst einen steigenden Loss auch im Bild steigen', () => {
    const points = lossTrend([
      training({ id: 'a', completed_at: iso(200), progress: { train_loss: 0.2 } }),
      training({ id: 'b', completed_at: iso(100), progress: { train_loss: 0.9 } }),
    ]);
    const ys = sparklinePoints(points, 100, 40, 2).split(' ').map(p => Number(p.split(',')[1]));
    expect(ys[0]).toBeGreaterThan(ys[1]);
  });

  it('kommt mit einer leeren Reihe klar', () => {
    expect(sparklinePoints([], 100, 40)).toBe('');
  });
});

describe('topResults', () => {
  it('sortiert absteigend nach Accuracy', () => {
    const res = topResults([
      test_({ id: 'a', version_name: 'v1', results: { accuracy: 0.7 } }),
      test_({ id: 'b', version_name: 'v2', results: { accuracy: 0.95 } }),
      test_({ id: 'c', version_name: 'v3', results: { accuracy: 0.8 } }),
    ]);
    expect(res.map(r => r.version_name)).toEqual(['v2', 'v3', 'v1']);
  });

  it('nimmt pro Modell-Version nur den besten Lauf', () => {
    const res = topResults([
      test_({ id: 'a', version_name: 'v1', results: { accuracy: 0.7 } }),
      test_({ id: 'b', version_name: 'v1', results: { accuracy: 0.91 } }),
    ]);
    expect(res).toHaveLength(1);
    expect(res[0].accuracy).toBeCloseTo(0.91);
  });

  it('ignoriert Tests ohne Accuracy oder ohne Abschluss', () => {
    expect(topResults([
      test_({ id: 'a', results: { accuracy: null } }),
      test_({ id: 'b', status: 'running', results: { accuracy: 0.99 } }),
    ])).toEqual([]);
  });
});

describe('Briefing-Fakten', () => {
  it('enthaelt die Kennzahlen und die juengsten Laeufe', () => {
    const facts = buildBriefingFacts(input({
      trainings: [training({ model_name: 'xlm-roberta', progress: { train_loss: 0.31 } })],
      tests: [test_()],
      models: [{ id: 'm1', name: 'xlm-roberta' }],
      datasets: [{ id: 'd1', name: 'reviews', training_count: 1 }],
    }));
    expect(facts).toContain('Modelle: 1');
    expect(facts).toContain('xlm-roberta auf reviews: completed train_loss=0.3100');
    expect(facts).toContain('accuracy=90.0%');
  });

  it('ergibt fuer denselben Stand denselben Hash — sonst kostet jeder Blick Tokens', () => {
    const a = input({ trainings: [training()], models: [{ id: 'm1', name: 'A' }] });
    const b = input({ trainings: [training()], models: [{ id: 'm1', name: 'A' }] });
    expect(factsHash(buildBriefingFacts(a))).toBe(factsHash(buildBriefingFacts(b)));
  });

  it('aendert den Hash, sobald ein neuer Lauf dazukommt', () => {
    const before = factsHash(buildBriefingFacts(input({ trainings: [training({ id: 'a' })] })));
    const after = factsHash(buildBriefingFacts(input({
      trainings: [training({ id: 'a' }), training({ id: 'b', completed_at: iso(1) })],
    })));
    expect(before).not.toBe(after);
  });
});

describe('byNewest', () => {
  it('faellt auf created_at zurueck, wenn completed_at fehlt', () => {
    const res = byNewest([
      { id: 'alt', created_at: iso(500), completed_at: null },
      { id: 'neu', created_at: iso(5), completed_at: null },
    ]);
    expect(res.map(r => r.id)).toEqual(['neu', 'alt']);
  });
});

describe('splitBriefing', () => {
  it('trennt die letzte fette Empfehlungszeile ab', () => {
    const { body, nextStep } = splitBriefing(
      'Lage ist gut.\n\n- Punkt eins\n\n**Naechster Schritt:** Eine Epoche weniger.',
    );
    expect(nextStep).toBe('**Naechster Schritt:** Eine Epoche weniger.');
    expect(body).toBe('Lage ist gut.\n\n- Punkt eins');
  });

  it('funktioniert auch mit englischem Label', () => {
    const { nextStep } = splitBriefing('Alles ok.\n\n**Next step:** Run the test.');
    expect(nextStep).toBe('**Next step:** Run the test.');
  });

  it('laesst den Text unangetastet, wenn das Modell die Struktur ignoriert', () => {
    const text = 'Nur ein Fliesstext ohne Empfehlungszeile.';
    expect(splitBriefing(text)).toEqual({ body: text, nextStep: '' });
  });

  it('trennt nichts ab, wenn die Empfehlung der ganze Text ist', () => {
    const text = '**Naechster Schritt:** Leg ein Modell an.';
    expect(splitBriefing(text)).toEqual({ body: text, nextStep: '' });
  });

  it('verlangt einen Doppelpunkt im Label — eine fette Schlusszeile allein reicht nicht', () => {
    const text = 'Lage ist gut.\n\n**Sehr ordentlich**';
    expect(splitBriefing(text).nextStep).toBe('');
  });

  it('ignoriert Leerzeilen am Ende', () => {
    const { nextStep } = splitBriefing('Lage.\n\n**Naechster Schritt:** Los.\n\n\n');
    expect(nextStep).toBe('**Naechster Schritt:** Los.');
  });
});
