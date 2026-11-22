/** Voreinstellungen fuer die Aufteilung beim Export: [train, val] — der Rest ist test. */
export const SPLITS: { key: string; train: number; val: number }[] = [
  { key: 'none', train: 0, val: 0 },
  { key: '8020', train: 0.8, val: 0.2 },
  { key: '701515', train: 0.7, val: 0.15 },
];
