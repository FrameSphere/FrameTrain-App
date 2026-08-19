import type { TestPluginProps } from '../types';
import GenericTestPanel from '../GenericTestPanel';

export default function AudioTestPlugin(props: TestPluginProps) {
  return (
    <GenericTestPanel
      {...props}
      taskType="audio_classification"
      inputKind="file"
      singleLabel="Einzelne Audiodatei"
      singlePlaceholder="Vollständiger Pfad zu einer Audiodatei, z.B. /Users/du/toene/probe.wav"
      resultLabel="Erkannte Klasse"
    />
  );
}
