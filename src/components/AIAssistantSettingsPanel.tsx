import React, { useState } from 'react';
import {
  Brain,
  Sparkles,
  Zap,
  Monitor,
  CheckCircle,
  AlertCircle,
  Download,
  Info,
  Leaf,
  Scale,
  Star,
  Flame,
  AlertTriangle,
} from 'lucide-react';
import { useAISettings, type AIProvider, type TokenBudget, TOKEN_BUDGET_CONFIG } from '../contexts/AISettingsContext';
import { useLanguage } from '../contexts/LanguageContext';
import { PROVIDER_META } from '../ai/providerMeta';

/**
 * Wiederverwendbare KI-Assistent-Konfigurationsoberfläche.
 * Identisch zum "KI-Assistent"-Tab in Settings.tsx — wird sowohl dort
 * als auch im First-Launch-Setup verwendet, damit Nutzer den KI-Anbieter
 * direkt beim ersten Start konfigurieren können.
 */
export default function AIAssistantSettingsPanel() {
  const { settings: aiSettings, updateSettings: updateAISettings } = useAISettings();
  const { t } = useLanguage();
  const [showApiKeyField, setShowApiKeyField] = useState(false);

  const providerIcon = (provider: AIProvider) => {
    const common = 'w-7 h-7';
    switch (provider) {
      case 'anthropic':
        return <Brain className={`${common} text-purple-300`} />;
      case 'openai':
        return <Sparkles className={`${common} text-emerald-300`} />;
      case 'groq':
        return <Zap className={`${common} text-amber-300`} />;
      case 'ollama':
        return <Monitor className={`${common} text-blue-300`} />;
      default:
        return <Sparkles className={`${common} text-gray-300`} />;
    }
  };

  const meta = PROVIDER_META[aiSettings.provider];

  return (
    <div className="space-y-6">
      {/* Enable/Disable Toggle */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-white">{t('settings.ai.title')}</h3>
            <p className="text-sm text-gray-400 mt-1">{t('settings.ai.subtitle')}</p>
          </div>
          <button
            onClick={() => updateAISettings({ enabled: !aiSettings.enabled })}
            className={`w-11 h-6 rounded-full transition-all ${
              aiSettings.enabled ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'bg-white/10'
            }`}
          >
            <div
              className={`w-5 h-5 rounded-full bg-white shadow-lg transform transition-transform ${
                aiSettings.enabled ? 'translate-x-5' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>
        {aiSettings.enabled && (
          <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-sm text-green-300 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 flex-shrink-0" />
            {t('settings.ai.active')}
          </div>
        )}
        {!aiSettings.enabled && (
          <div className="p-3 bg-gray-500/10 border border-gray-500/20 rounded-lg text-sm text-gray-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {t('settings.ai.inactive')}
          </div>
        )}
      </div>

      {aiSettings.enabled && (
        <>
          {/* Provider Selection */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">{t('settings.ai.providerTitle')}</h3>
            <div className="grid grid-cols-2 gap-3">
              {(Object.entries(PROVIDER_META) as [AIProvider, typeof PROVIDER_META[AIProvider]][]).map(([key, m]) => (
                <button
                  key={key}
                  onClick={() => updateAISettings({ provider: key, selectedModel: m.models[0] })}
                  className={`flex items-center gap-3 p-4 rounded-xl border text-left transition-all ${
                    aiSettings.provider === key
                      ? 'bg-purple-500/20 border-purple-500/50 text-white'
                      : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <span className="flex-shrink-0">{providerIcon(key)}</span>
                  <div className="min-w-0">
                    <div className="text-sm font-semibold">{m.label}</div>
                    <div className="text-xs opacity-60 mt-0.5 flex items-center gap-1.5">
                      {m.needsKey ? (
                        <>{t('settings.ai.keyNeeded')}</>
                      ) : (
                        <>
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                          {t('settings.ai.noKeyNeeded')}
                        </>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Provider-specific Configuration */}
          {meta.needsKey ? (
            <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">{t('settings.ai.apiKeyTitle')}</h3>
                <a
                  href={meta.keyLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-purple-400 hover:text-purple-300 transition-colors flex items-center gap-1"
                >
                  {t('settings.ai.getKey')}
                </a>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">{meta.keyHint}</label>
                <div className="flex items-center gap-2">
                  <input
                    type={showApiKeyField ? 'text' : 'password'}
                    value={aiSettings.apiKey}
                    onChange={(e) => updateAISettings({ apiKey: e.target.value })}
                    placeholder={meta.keyPlaceholder}
                    className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                  <button
                    onClick={() => setShowApiKeyField(!showApiKeyField)}
                    className="px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-gray-400 hover:text-white transition-all text-sm"
                  >
                    {showApiKeyField ? t('settings.ai.hide') : t('settings.ai.show')}
                  </button>
                </div>
              </div>

              <p className="text-xs text-gray-500">{t('settings.ai.keyLocalOnly')}</p>

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-semibold text-white mb-2">{t('settings.ai.model')}</label>
                <div className="flex flex-wrap gap-2">
                  {meta.models.map(m => (
                    <button
                      key={m}
                      onClick={() => updateAISettings({ selectedModel: m })}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                        aiSettings.selectedModel === m
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                {/* Freitextfeld: Anbieter mustern Modelle aus, ohne dass ein
                    App-Update zur Stelle ist. Ohne dieses Feld war jede
                    KI-Funktion blockiert, sobald alle Vorschläge veraltet waren. */}
                <div className="mt-3">
                  <label className="block text-xs text-gray-400 mb-1.5">
                    {t('settings.ai.customModelLabel')}
                  </label>
                  <input
                    type="text"
                    value={aiSettings.selectedModel ?? ''}
                    onChange={(e) => updateAISettings({ selectedModel: e.target.value })}
                    placeholder={meta.defaultModel}
                    spellCheck={false}
                    className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded-lg text-white text-xs font-mono placeholder:text-gray-600 focus:outline-none focus:border-purple-500/50 transition-all"
                  />
                  <p className="text-xs text-gray-500 mt-1.5">
                    {t('settings.ai.customModelHint')}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            /* Ollama Configuration */
            <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">{t('settings.ai.ollamaTitle')}</h3>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">{t('settings.ai.ollamaModelLabel')}</label>
                <input
                  type="text"
                  value={aiSettings.ollamaModel}
                  onChange={(e) => updateAISettings({ ollamaModel: e.target.value, selectedModel: e.target.value })}
                  placeholder="llama3.2"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-white mb-2">{t('settings.ai.ollamaPopular')}</label>
                <div className="flex flex-wrap gap-2">
                  {PROVIDER_META.ollama.models.map(m => (
                    <button
                      key={m}
                      onClick={() => updateAISettings({ ollamaModel: m, selectedModel: m })}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                        aiSettings.ollamaModel === m
                          ? 'bg-green-500/20 border-green-500/50 text-green-300'
                          : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-4 bg-white/[0.03] rounded-lg border border-white/10 text-xs text-gray-400 space-y-2">
                <div className="font-semibold text-gray-300 flex items-center gap-2">
                  <Download className="w-4 h-4 text-blue-300" />
                  {t('settings.ai.ollamaInstallTitle')}
                </div>
                <div>{t('settings.ai.ollamaInstallStep1')} <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:underline">ollama.com</a>{t('settings.ai.ollamaInstallStep1Suffix')}</div>
                <div>{t('settings.ai.ollamaInstallStep2')} <code className="bg-black/30 px-1 py-0.5 rounded font-mono text-xs">{t('settings.ai.ollamaInstallCommand')}</code></div>
                <div>{t('settings.ai.ollamaInstallStep3')}</div>
              </div>
            </div>
          )}

          {/* Token Budget */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10 space-y-4">
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">{t('settings.ai.tokenBudget.title')}</h3>
              <p className="text-sm text-gray-400">{t('settings.ai.tokenBudget.subtitle')}</p>
            </div>

            {/* Budget Stufen */}
            <div className="grid grid-cols-2 gap-3">
              {(Object.entries(TOKEN_BUDGET_CONFIG) as [TokenBudget, typeof TOKEN_BUDGET_CONFIG[TokenBudget]][]).map(([key, cfg]) => {
                const active = (aiSettings.tokenBudget ?? 'balanced') === key;
                const colors: Record<TokenBudget, string> = {
                  minimal:  'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
                  balanced: 'border-blue-500/50 bg-blue-500/10 text-blue-300',
                  quality:  'border-purple-500/50 bg-purple-500/10 text-purple-300',
                  max:      'border-amber-500/50 bg-amber-500/10 text-amber-300',
                };
                const iconComponents: Record<TokenBudget, React.ReactNode> = {
                  minimal:  <Leaf      className="w-4 h-4 text-emerald-400" />,
                  balanced: <Scale     className="w-4 h-4 text-blue-400" />,
                  quality:  <Star      className="w-4 h-4 text-purple-400" />,
                  max:      <Flame     className="w-4 h-4 text-amber-400" />,
                };
                return (
                  <button
                    key={key}
                    onClick={() => updateAISettings({ tokenBudget: key })}
                    className={`flex flex-col gap-2 p-4 rounded-xl border-2 text-left transition-all ${
                      active ? colors[key] : 'border-white/10 bg-white/[0.03] text-gray-400 hover:bg-white/5 hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{iconComponents[key]}</span>
                      {active && <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-white/20">{t('settings.ai.tokenBudget.active')}</span>}
                    </div>
                    <div>
                      <div className="font-semibold text-sm">{cfg.label}</div>
                      <div className="text-xs opacity-70 mt-0.5 leading-relaxed">{t(`settings.ai.tokenBudget.${key}Desc`)}</div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Live-Vorschau */}
            {(() => {
              const budget = TOKEN_BUDGET_CONFIG[aiSettings.tokenBudget ?? 'balanced'];
              const provMeta = PROVIDER_META[aiSettings.provider];

              // Kosten-Schätzung pro Nachricht (Input + Output)
              // Grobe Durchschnittswerte ($ per 1M tokens)
              const pricing: Record<string, { input: number; output: number }> = {
                'claude-opus-4-5':         { input: 15,   output: 75 },
                'claude-sonnet-4-5':       { input: 3,    output: 15 },
                'claude-haiku-4-5':        { input: 0.8,  output: 4  },
                'gpt-4o':                  { input: 2.5,  output: 10 },
                'gpt-4o-mini':             { input: 0.15, output: 0.6 },
                'llama-3.3-70b-versatile': { input: 0,    output: 0  }, // Groq Free
                'llama-3.1-8b-instant':    { input: 0,    output: 0  }, // Groq Free
                'ollama':                  { input: 0,    output: 0  },
              };
              const model = aiSettings.provider === 'ollama'
                ? 'ollama'
                : (aiSettings.selectedModel || provMeta.models[0]);
              const p = pricing[model] ?? { input: 1, output: 5 };

              // Geschätzte Input-Tokens pro Nachricht: System (~300) + History-Budget + User-Message (~80)
              const estimatedInput  = 300 + budget.historyTokenBudget + 80;
              const estimatedOutput = budget.maxTokens * 0.7; // ~70% Ausnutzung
              const costPerMsg = ((estimatedInput * p.input) + (estimatedOutput * p.output)) / 1_000_000;

              // Groq Rate-Limit-Indikator (6000 TPM Free Tier)
              const isGroq = aiSettings.provider === 'groq';
              const groqTpm = estimatedInput + estimatedOutput;
              const groqMsgsPerMinute = Math.floor(6000 / groqTpm);

              return (
                <div className="rounded-xl border border-white/10 bg-black/20 p-4 space-y-3">
                  <p className="text-xs font-semibold text-gray-300 uppercase tracking-wider">{t('settings.ai.tokenBudget.previewTitle')}</p>

                  {/* Token-Balken */}
                  <div className="space-y-2">
                    {([
                      [t('settings.ai.tokenBudget.previewCoach'),   budget.maxTokens,        1500],
                      [t('settings.ai.tokenBudget.previewSynapse'), budget.synapseMaxTokens, 8000],
                      [t('settings.ai.tokenBudget.previewHistory'), budget.historyTokenBudget, 4000],
                    ] as [string, number, number][]).map(([label, val, max]) => (
                      <div key={label}>
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>{label}</span>
                          <span className="font-mono text-gray-300">{val.toLocaleString()} tokens</span>
                        </div>
                        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              (aiSettings.tokenBudget ?? 'balanced') === 'minimal'  ? 'bg-emerald-400' :
                              (aiSettings.tokenBudget ?? 'balanced') === 'balanced' ? 'bg-blue-400' :
                              (aiSettings.tokenBudget ?? 'balanced') === 'quality'  ? 'bg-purple-400' :
                                                                                      'bg-amber-400'
                            }`}
                            style={{ width: `${Math.min((val / max) * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Kosten + Rate-Limit */}
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <div className="bg-white/5 rounded-lg p-3">
                      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewCostLabel')}</p>
                      <p className="text-sm font-semibold text-white">
                        {costPerMsg < 0.0001
                          ? t('settings.ai.tokenBudget.previewCostFree')
                          : `~${costPerMsg.toFixed(4)}`}
                      </p>
                      <p className="text-[10px] text-gray-500 mt-0.5">{t('settings.ai.tokenBudget.previewCostPerMsg')}</p>
                    </div>
                    {isGroq ? (
                      <div className={`rounded-lg p-3 ${
                        groqMsgsPerMinute >= 3 ? 'bg-emerald-500/10' : groqMsgsPerMinute >= 1 ? 'bg-amber-500/10' : 'bg-red-500/10'
                      }`}>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewGroqLimit')}</p>
                        <p className={`text-sm font-semibold ${
                          groqMsgsPerMinute >= 3 ? 'text-emerald-300' : groqMsgsPerMinute >= 1 ? 'text-amber-300' : 'text-red-300'
                        }`}>
                          ~{groqMsgsPerMinute} {t('settings.ai.tokenBudget.previewGroqMsgs')}
                        </p>
                        <p className="text-[10px] text-gray-500 mt-0.5">{t('settings.ai.tokenBudget.previewGroqPerMin')}</p>
                      </div>
                    ) : (
                      <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{t('settings.ai.tokenBudget.previewQuality')}</p>
                        <p className="text-sm font-semibold text-white">
                          {(aiSettings.tokenBudget ?? 'balanced') === 'minimal'  ? t('settings.ai.tokenBudget.qualityConcise') :
                           (aiSettings.tokenBudget ?? 'balanced') === 'balanced' ? t('settings.ai.tokenBudget.qualityBalanced') :
                           (aiSettings.tokenBudget ?? 'balanced') === 'quality'  ? t('settings.ai.tokenBudget.qualityDetailed') :
                                                                                   t('settings.ai.tokenBudget.qualityMax')}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Groq-Warning */}
                  {isGroq && (aiSettings.tokenBudget === 'quality' || aiSettings.tokenBudget === 'max') && (
                    <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-amber-300">{t('settings.ai.tokenBudget.groqWarning')}</p>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>

          {/* Info Box */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-sm text-blue-300 flex items-start gap-3">
            <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold mb-1">{t('settings.ai.globalNote')}</div>
              <p className="text-xs text-blue-200">{t('settings.ai.globalNoteDesc')}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
