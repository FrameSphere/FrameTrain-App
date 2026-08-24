// Bildet den DatasetTypeIconKey aus datasetCompatHelpers auf ein lucide-Icon
// ab — die UI zeigt Icons statt Emojis (Grundsatz: keine Emojis in der UI).

import {
  FileText, Target, Image, FolderTree, Folder,
  Mic, Volume2, Split, Layers, HelpCircle,
  type LucideIcon,
} from 'lucide-react';
import type { DatasetTypeIconKey } from '../plugins/datasetCompatHelpers';

const ICON_MAP: Record<DatasetTypeIconKey, LucideIcon> = {
  'file-text':   FileText,
  target:        Target,
  image:         Image,
  'folder-tree': FolderTree,
  folder:        Folder,
  mic:           Mic,
  volume:        Volume2,
  split:         Split,
  layers:        Layers,
  help:          HelpCircle,
};

export default function DatasetTypeIcon({
  icon,
  className = 'w-4 h-4',
}: {
  icon: DatasetTypeIconKey;
  className?: string;
}) {
  const Icon = ICON_MAP[icon] ?? HelpCircle;
  return <Icon className={className} />;
}
