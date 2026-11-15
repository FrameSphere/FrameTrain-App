import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function TabularTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="tabular"
      inputKind="text"
      singleLabel={t('testPlugins.tabular.singleLabel')}
      singlePlaceholder={t('testPlugins.tabular.singlePlaceholder')}
      resultLabel={t('testPlugins.tabular.result')}
      showConfidence
    />
  );
}
