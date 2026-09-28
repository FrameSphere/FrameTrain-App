import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function SpeechRecognitionTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="speech_recognition"
      inputKind="file"
      singleLabel={t('testPlugins.generic.audioLabel')}
      singlePlaceholder={t('testPlugins.generic.audioPlaceholder')}
      resultLabel={t('testPlugins.generic.transcript')}
      // Ein Transkript hat keine ehrliche Konfidenz — wie bei Seq2Seq.
      showConfidence={false}
    />
  );
}
