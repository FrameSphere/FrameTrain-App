import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function CausalLMTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="causal_lm"
      inputKind="text"
      singleLabel={t('testPlugins.generic.causalLmLabel')}
      singlePlaceholder={t('testPlugins.generic.causalLmPlaceholder')}
      resultLabel={t('testPlugins.generic.causalLmResult')}
      // Freier Text hat keine Klassen – eine Konfidenz gaebe es nicht ehrlich.
      showConfidence={false}
    />
  );
}
