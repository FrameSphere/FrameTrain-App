// Die Seiten, die dem Coach einen Live-Zustand melden — einmal zentral, damit
// Skill- und Tool-Tests nicht auseinanderlaufen.
import type { PageId } from '../../coachContext';

export const PAGE_KNOWLEDGE_PAGES: PageId[] = [
  'home', 'models', 'training', 'training-dev', 'dataset',
  'analysis', 'tests', 'tests-dev', 'laboratory', 'versions', 'synapse', 'settings',
];
