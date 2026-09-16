// Gegenstueck zu den Typen in src-tauri/src/studio_manager.rs.

import type { StudioBox } from './studioBoxes';

export interface StudioProject {
  id:            string;
  name:          string;
  modality:      string;
  task:          string;
  target_format: string;
  classes:       string[];
  created_at:    string;
  updated_at:    string;
  sample_count?:    number;
  confirmed_count?: number;
}

export type SampleStatus = 'new' | 'suggested' | 'confirmed' | 'skipped';

export interface StudioSample {
  id:       string;
  media:    string;
  mime:     string;
  status:   SampleStatus;
  ann:      { boxes: StudioBox[] };
  src:      { kind: string; origin?: string | null; license?: string | null; at: string };
  meta:     { w: number; h: number; group?: string | null };
  abs_path: string;
}

export interface SamplePage { total: number; items: StudioSample[]; }

export interface ImportReport {
  added:         number;
  duplicates:    number;
  unreadable:    number;
  with_labels:   number;
  classes_added: string[];
}

export interface StudioStats {
  total:           number;
  new:             number;
  suggested:       number;
  confirmed:       number;
  skipped:         number;
  boxes_total:     number;
  per_class:       number[];
  empty_confirmed: number;
}
