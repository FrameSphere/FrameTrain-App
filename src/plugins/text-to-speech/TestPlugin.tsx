import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function TextToSpeechTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="text_to_speech"
      inputKind="text"
      singleLabel={t('testPlugins.generic.ttsLabel')}
      singlePlaceholder={t('testPlugins.generic.ttsPlaceholder')}
      resultLabel={t('testPlugins.generic.ttsResult')}
      // Erzeugtes Audio hat keine Klassen-Wahrscheinlichkeit.
      showConfidence={false}
      // Das Ergebnis ist der Pfad einer WAV-Datei und wird als Player gezeigt.
      resultKind="audio"
    />
  );
}
