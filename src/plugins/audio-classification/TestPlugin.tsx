import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function AudioTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="audio_classification"
      inputKind="file"
      singleLabel={t('testPlugins.generic.audioLabel')}
      singlePlaceholder={t('testPlugins.generic.audioPlaceholder')}
      resultLabel={t('testPlugins.generic.detectedClass')}
    />
  );
}
