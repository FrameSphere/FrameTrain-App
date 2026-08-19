import type { TestPluginProps } from '../types';
import GenericTestPanel from '../GenericTestPanel';

export default function Seq2SeqTestPlugin(props: TestPluginProps) {
  return (
    <GenericTestPanel
      {...props}
      taskType="seq2seq"
      inputKind="text"
      singleLabel="Eingabetext"
      singlePlaceholder="Text eingeben, der umgeformt werden soll…"
      resultLabel="Erzeugter Text"
      // Freier Text hat keine Klassen – eine Konfidenz gaebe es nicht ehrlich.
      showConfidence={false}
    />
  );
}
