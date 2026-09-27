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
  /** Bei Text steht der Inhalt hier statt in einer Datei. */
  content?: string | null;
  status:   SampleStatus;
  ann:      { boxes: StudioBox[]; label?: string | null; target?: string | null;
              /** Sicherheit des Modells beim Vorschlag (0..1). */
              confidence?: number | null };
  src:      { kind: string; origin?: string | null; license?: string | null; at: string;
              /** Seite, auf der eine Datei aus dem Netz gefunden wurde. */
              page?: string | null };
  meta:     { w: number; h: number; group?: string | null;
              /** Videoabschnitt in Sekunden. */
              start?: number | null; end?: number | null };
  abs_path: string;
  doubt?:   Doubt | null;
}

export interface SamplePage { total: number; items: StudioSample[]; }

export interface ImportReport {
  added:          number;
  duplicates:     number;
  unreadable:     number;
  with_labels:    number;
  classes_added:  string[];
  unknown_ids:    number[];
  labels_ignored: number;
}

export interface Doubt { missing: string[]; extra: string[]; at: string; }

export interface StudioStats {
  total:           number;
  new:             number;
  suggested:       number;
  confirmed:       number;
  skipped:         number;
  boxes_total:     number;
  per_class:       number[];
  empty_confirmed: number;
  doubts:          number;
}

/** Bericht eines Exports — Gegenstueck zu quality::ExportReport. */
export interface ExportReport {
  total:                  number;
  per_class:              [string, number][];
  per_split:              [string, number][];
  groups:                 number;
  group_leaks:            string[];
  near_duplicates:        number;
  near_duplicates_across: number;
  near_checked:           boolean;
  licenses:               [string, number][];
  without_license:        number;
  sources:                [string, number][];
  warnings:               string[];
  /** Dieselben Hinweise als Code mit Werten — hier uebersetzt. */
  hints?:                 { code: string; params: Record<string, string | number> }[];
}

export interface ExportResult {
  dataset: { id: string; name: string };
  report:  ExportReport;
  path:    string;
}

/** Filter der Sample-Liste. "uncertain": Vorschlaege, unsicherste zuerst. */
export type SampleFilter = 'all' | 'open' | 'confirmed' | 'doubt' | 'uncertain';
