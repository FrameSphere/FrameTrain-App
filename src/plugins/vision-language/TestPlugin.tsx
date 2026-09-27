import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function VisionLanguageTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="vision_language"
      inputKind="file"
      singleLabel={t('testPlugins.generic.vlmImageLabel')}
      singlePlaceholder={t('testPlugins.generic.vlmImagePlaceholder')}
      resultLabel={t('testPlugins.generic.vlmResult')}
      // Freier Text — keine ehrliche Konfidenz.
      showConfidence={false}
      secondaryInput={{
        label: t('testPlugins.generic.vlmQuestionLabel'),
        placeholder: t('testPlugins.generic.vlmQuestionPlaceholder'),
        configKey: 'question',
      }}
    />
  );
}
