import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function VideoTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="video_classification"
      inputKind="file"
      singleLabel={t('testPlugins.generic.videoLabel')}
      singlePlaceholder={t('testPlugins.generic.videoPlaceholder')}
      resultLabel={t('testPlugins.generic.detectedClass')}
    />
  );
}
