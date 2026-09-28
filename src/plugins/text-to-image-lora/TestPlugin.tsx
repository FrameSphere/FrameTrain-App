import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function TextToImageTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="text_to_image_lora"
      inputKind="text"
      singleLabel={t('testPlugins.generic.t2iLabel')}
      singlePlaceholder={t('testPlugins.generic.t2iPlaceholder')}
      resultLabel={t('testPlugins.generic.t2iResult')}
      // Ein erzeugtes Bild hat keine Klassen-Wahrscheinlichkeit.
      showConfidence={false}
      resultKind="image"
    />
  );
}
