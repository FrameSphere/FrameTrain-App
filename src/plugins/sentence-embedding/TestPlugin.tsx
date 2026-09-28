import type { TestPluginProps } from '../types';
import { useLanguage } from '../../contexts/LanguageContext';
import GenericTestPanel from '../GenericTestPanel';

export default function SentenceEmbeddingTestPlugin(props: TestPluginProps) {
  const { t } = useLanguage();
  return (
    <GenericTestPanel
      {...props}
      taskType="sentence_embedding"
      inputKind="text"
      singleLabel={t('testPlugins.generic.embeddingLabel')}
      singlePlaceholder={t('testPlugins.generic.embeddingPlaceholder')}
      resultLabel={t('testPlugins.generic.embeddingResult')}
      // Ein Vektor ist keine Entscheidung mit Wahrscheinlichkeit — gezeigt
      // werden die aehnlichsten Texte mit ihrer Kosinus-Aehnlichkeit.
      showConfidence={false}
      showTopList
      scoreFormat="raw"
      singleUsesDataset
    />
  );
}
