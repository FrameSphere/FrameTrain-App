import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function TokenClassificationTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="token_classification"
      inputKind="text"
      singleLabel={t('testPlugins.generic.tokenLabel')}
      singlePlaceholder={t('testPlugins.generic.tokenPlaceholder')}
      resultLabel={t('testPlugins.generic.tokenResult')}
      // Konfidenz = unsicherste Entitaet; darunter jede Entitaet mit ihrem Score.
      showConfidence
      showTopList
    />
  );
}
