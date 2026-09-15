import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function Seq2SeqTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="seq2seq"
      inputKind="text"
      singleLabel={t('testPlugins.generic.seq2seqLabel')}
      singlePlaceholder={t('testPlugins.generic.seq2seqPlaceholder')}
      resultLabel={t('testPlugins.generic.seq2seqResult')}
      // Freier Text hat keine Klassen – eine Konfidenz gaebe es nicht ehrlich.
      showConfidence={false}
    />
  );
}
