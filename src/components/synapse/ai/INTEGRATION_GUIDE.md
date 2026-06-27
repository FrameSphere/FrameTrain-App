/**
 * INTEGRATION GUIDE: Synapse AI Coach in SynapseBuilder
 * 
 * Schritt 1: Import hinzufügen
 * Schritt 2: State für Training-Ergebnis
 * Schritt 3: Panel im Render hinzufügen
 * Schritt 4: Training-Ergebnis erfassen
 */

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 1: Im SynapseBuilder.tsx - Imports hinzufügen
// ─────────────────────────────────────────────────────────────────────────────

import { SynapseAICoachPanel } from "./ai/SynapseAICoachPanel";
import { TrainingResult } from "./ai/SynapseAICoach";

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 2: Im Component State - hinzufügen nach anderen States
// ─────────────────────────────────────────────────────────────────────────────

// Suche diese Zeile (ungefähr Zeile 100-150):
// const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("idle");

// Füge danach hinzu:
const [lastTrainingResult, setLastTrainingResult] = useState<TrainingResult | null>(null);
const [showAICoach, setShowAICoach] = useState(false);

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 3: Training-Result erfassen
// ─────────────────────────────────────────────────────────────────────────────

// Suche die Stelle wo "Training complete" geloggt wird (Zeile ~337):
// logger.info(`[✓ Training complete]...`);

// Füge davor ein:
const parseTrainingLogForMetrics = (logs: string[]): TrainingResult => {
  const hasError = logs.some(log => log.includes('[ERROR]'));
  const finalLosMatch = logs.find(log => log.includes('Best loss:'));
  const finalLoss = finalLosMatch 
    ? parseFloat(finalLosMatch.match(/\d+\.\d+/)?.[0] || "0")
    : 0;

  let errorMessage = "";
  const errorLog = logs.find(log => log.includes('[ERROR]'));
  if (errorLog) {
    errorMessage = errorLog.substring(errorLog.indexOf('[ERROR]') + 7);
  }

  return {
    success: !hasError,
    jobId: trainingJobId || "unknown",
    duration: Date.now() - (trainingStartTime || Date.now()),
    epochs: trainingConfig?.epochs || 0,
    finalLoss,
    error: errorMessage,
    errorType: errorMessage.includes('shape') ? 'shape' : 'unknown',
    timestamp: Date.now(),
  };
};

// In der Training Event Handler (suche nach "Training Status Updates"):
const handleTrainingComplete = (success: boolean, finalLoss: number) => {
  const result: TrainingResult = {
    success,
    jobId: trainingJobId || "unknown",
    duration: Date.now() - (trainingStartTime || Date.now()),
    epochs: trainingConfig?.epochs || 0,
    finalLoss,
    timestamp: Date.now(),
  };
  
  setLastTrainingResult(result);
  
  // Wenn Fehler, zeige AI Coach Panel
  if (!success) {
    setShowAICoach(true);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 4: Error-Handler erweitern
// ─────────────────────────────────────────────────────────────────────────────

// Suche diese Event-Handler (Zeile ~355):
// if (!ev.payload?.success && trainingStatus !== "error" && trainingStatus !== "done") {

// Füge hinzu:
if (!ev.payload?.success && trainingStatus !== "error" && trainingStatus !== "done") {
  const result: TrainingResult = {
    success: false,
    jobId: trainingJobId || "unknown",
    duration: Date.now() - (trainingStartTime || Date.now()),
    epochs: trainingConfig?.epochs || 0,
    error: ev.payload?.error || "Unknown error",
    errorType: 'runtime',
    errorDetails: ev.payload?.details,
    timestamp: Date.now(),
  };
  
  setLastTrainingResult(result);
  setShowAICoach(true);
  setTrainingStatus("error");
  // ... rest of error handling
}

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 5: Im Render - Panel hinzufügen
// ─────────────────────────────────────────────────────────────────────────────

// Suche die return/render Funktion und füge am Ende (vor der schließenden Klammer) hinzu:

{showAICoach && lastTrainingResult && (
  <SynapseAICoachPanel
    trainingResult={lastTrainingResult}
    nodes={nodes}
    edges={edges}
    layerConfig={currentModelConfig?.layers || []}
    onApplyFix={(fix) => {
      console.log("🔧 Applying fix:", fix);
      // TODO: Implementiere Auto-Fix für verschiedene Fix-Typen
      // Beispiel: remove_node, adjust_params, insert_bridge, etc.
      
      if (fix.action === "remove_node") {
        setNodes(nodes.filter(n => n.id !== fix.targetNodeId));
        setEdges(edges.filter(e => 
          e.source !== fix.targetNodeId && e.target !== fix.targetNodeId
        ));
      } else if (fix.action === "adjust_params") {
        setNodes(nodes.map(n => 
          n.id === fix.targetNodeId
            ? { ...n, data: { ...n.data, params: { ...n.data?.params, ...fix.params } } }
            : n
        ));
      }
      
      // Zeige Toast/Notification
      logger.info(`✅ Fix angewendet: ${fix.title}`);
    }}
    onClose={() => setShowAICoach(false)}
  />
)}

// ─────────────────────────────────────────────────────────────────────────────
// SCHRITT 6 (OPTIONAL): Canvas-Config als currentModelConfig speichern
// ─────────────────────────────────────────────────────────────────────────────

// Füge einen State hinzu für die aktuelle Model-Config:
const [currentModelConfig, setCurrentModelConfig] = useState<ModelGraphConfig | null>(null);

// In handleStartTraining:
const canvasConfig = generateModelConfigFromGraph(nodes, edges, NODE_DEFINITIONS, modelName);
if (canvasConfig) {
  setCurrentModelConfig(canvasConfig); // Speichere für AI Coach
}

// ─────────────────────────────────────────────────────────────────────────────
// RESULTAT
// ─────────────────────────────────────────────────────────────────────────────

/*
Wenn Training fehlgeschlagen:
1. AI Coach Panel öffnet sich automatisch
2. Zeigt:
   - Training-Status & Fehlertyp
   - Identifizierte Probleme (z.B. Shape-Mismatches)
   - Warnungen
3. User kann fragen:
   - "Was sind die Probleme?"
   - "Fixe Shape-Fehler"
   - "War das Training erfolgreich?"
4. AI Coach schlägt Fixes vor (z.B. Layer entfernen, Bridge einfügen)
5. Mit "Anwenden" Button werden Fixes automatisch auf Graph angewendet
6. User kann dann erneut trainieren
*/
