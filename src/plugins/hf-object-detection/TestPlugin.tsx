import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

// Einzelbild mit Boxen im Bild, Dataset-Lauf mit mAP50/mAP50-95 — die Test-
// Engine liefert denselben Ergebnisaufbau wie beim YOLO-Test.
export default function HFObjectDetectionTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="hf_object_detection"
      inputKind="file"
      singleLabel={t('testPlugins.generic.imageLabel')}
      singlePlaceholder={t('testPlugins.generic.imagePlaceholder')}
      resultLabel={t('testPlugins.generic.detectedObjects')}
      resultKind="boxes"
      showTopList={false}
    />
  );
}
