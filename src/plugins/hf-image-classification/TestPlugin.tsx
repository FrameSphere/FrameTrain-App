import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function HFImageTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="hf_image_classification"
      inputKind="file"
      singleLabel={t('testPlugins.generic.imageLabel')}
      singlePlaceholder={t('testPlugins.generic.imagePlaceholder')}
      resultLabel={t('testPlugins.generic.detectedClass')}
    />
  );
}
