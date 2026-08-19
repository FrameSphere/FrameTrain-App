import type { TestPluginProps } from '../types';
import GenericTestPanel from '../GenericTestPanel';

export default function HFImageTestPlugin(props: TestPluginProps) {
  return (
    <GenericTestPanel
      {...props}
      taskType="hf_image_classification"
      inputKind="file"
      singleLabel="Einzelnes Bild"
      singlePlaceholder="Vollständiger Pfad zu einer Bilddatei, z.B. /Users/du/bilder/katze.png"
      resultLabel="Erkannte Klasse"
    />
  );
}
