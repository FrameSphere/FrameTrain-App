import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function QuestionAnsweringTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="question_answering"
      inputKind="text"
      singleLabel={t('testPlugins.generic.qaLabel')}
      singlePlaceholder={t('testPlugins.generic.qaPlaceholder')}
      resultLabel={t('testPlugins.generic.qaResult')}
      // Die Engine liefert die Antwort und den Kontext mit »markierter« Stelle;
      // darunter die naechstbesten Kandidaten mit ihrem Rohscore.
      showConfidence
      showTopList
      scoreFormat="raw"
      secondaryInput={{
        label: t('testPlugins.generic.qaQuestionLabel'),
        placeholder: t('testPlugins.generic.qaQuestionPlaceholder'),
        configKey: 'question',
      }}
    />
  );
}
