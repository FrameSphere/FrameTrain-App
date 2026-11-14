import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function RerankerTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="reranker"
      inputKind="text"
      singleLabel={t('testPlugins.generic.rerankLabel')}
      singlePlaceholder={t('testPlugins.generic.rerankPlaceholder')}
      resultLabel={t('testPlugins.generic.rerankResult')}
      // Relevanz 0..1 je Passage, keine Klassen-Konfidenz.
      showConfidence={false}
      showTopList
      scoreFormat="raw"
      singleUsesDataset
    />
  );
}
