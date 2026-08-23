// DatasetFileManager.tsx – Datei-Browser für einzelne Datasets
// Portiert aus desktop-app2

import { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  FileText, Trash2, Upload, Search, X, Eye,
  File, Loader2, RefreshCw, Tag, ArrowRight,
  FolderOpen, Edit3, Plus, Check, Save,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { useLanguage } from '../contexts/LanguageContext';
import type { DatasetType } from '../plugins/datasetCompatHelpers';

// ── Types ──────────────────────────────────────────────────────────────────

interface FileInfo {
  name: string;
  path: string;
  size: number;
  is_dir: boolean;
  split: 'train' | 'val' | 'test' | 'unsplit';
}

interface DatasetFileManagerProps {
  datasetId:    string;
  datasetName:  string;
  datasetType?: DatasetType;
  onClose:      () => void;
}

const PAIRED_TYPES: DatasetType[] = ['yolo_bbox', 'pascal_voc', 'audio_transcript', 'coco_json'];
// ── Helpers ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const SPLIT_COLORS: Record<string, string> = {
  train:       '#3b82f6',
  val:         '#a855f7',
  test:        '#10b981',
  unsplit:     '#6b7280',
  info:        '#8b5cf6',
  images:      '#f97316',
  labels:      '#eab308',
  annotations: '#eab308',
  imgs:        '#f97316',
  clips:       '#06b6d4',
};

const SPLIT_LABELS: Record<string, string> = {
  train: 'train', val: 'val', test: 'test', unsplit: 'unsplit', info: 'info',
  images: 'images/', labels: 'labels/', annotations: 'annotations/', imgs: 'imgs/', clips: 'clips/',
};

// ── Component ──────────────────────────────────────────────────────────────

export default function DatasetFileManager({ datasetId, datasetName, datasetType, onClose }: DatasetFileManagerProps) {
  const { currentTheme } = useTheme();
  const { success, error } = useNotification();
  const { t } = useLanguage();

  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [currentSplit, setCurrentSplit] = useState<string>('all');
  const [viewingFile, setViewingFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState('');
  const [loadingContent, setLoadingContent] = useState(false);
  const [activeTab, setActiveTab] = useState<'files' | 'yaml'>('files');

  // Parquet-Preview State
  interface ParquetColumn { name: string; dtype: string; }
  interface ParquetPreview {
    columns: ParquetColumn[];
    rows: Record<string, unknown>[];
    total_rows: number;
    total_cols: number;
    shown_rows: number;
  }
  const [parquetPreview, setParquetPreview] = useState<ParquetPreview | null>(null);
  const [parquetError, setParquetError] = useState<string | null>(null);

  // YAML Editor State
  interface YamlData { exists: boolean; train_path: string; val_path: string; nc: number; names: string[]; yaml_path?: string; }
  const [yamlData, setYamlData] = useState<YamlData | null>(null);
  const [yamlLoading, setYamlLoading] = useState(false);
  const [yamlSaving, setYamlSaving] = useState(false);
  const [editTrainPath, setEditTrainPath] = useState('');
  const [editValPath, setEditValPath] = useState('');
  const [editNames, setEditNames] = useState<string[]>([]);
  const [newClassName, setNewClassName] = useState('');
  const [yamlSaved, setYamlSaved] = useState(false);

  useEffect(() => { loadFiles(); }, [datasetId]);

  const loadFiles = async () => {
    setLoading(true);
    try {
      const result = await invoke<FileInfo[]>('get_dataset_files', { datasetId });
      setFiles(result);
    } catch (err: unknown) {
      error(t('datasetFileManager.notifications.loadError'), String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadYaml = async () => {
    if (yamlData !== null) return; // bereits geladen
    setYamlLoading(true);
    try {
      const data = await invoke<{ exists: boolean; train_path: string; val_path: string; nc: number; names: string[]; yaml_path?: string; }>(
        'get_dataset_yaml', { datasetId }
      );
      setYamlData(data);
      setEditTrainPath(data.train_path ?? 'images/train');
      setEditValPath(data.val_path ?? 'images/val');
      setEditNames(data.names ?? []);
    } catch (err) {
      error(t('datasetFileManager.yaml.yamlError'), String(err));
    } finally {
      setYamlLoading(false);
    }
  };

  // YAML-Tab aktiviert → laden
  useEffect(() => {
    if (activeTab === 'yaml' && datasetType === 'yolo_bbox') loadYaml();
  }, [activeTab]);

  // Stats
  const stats = useMemo(() => {
    const s = { train: 0, val: 0, test: 0, unsplit: 0 };
    files.forEach(f => { s[f.split]++; });
    return s;
  }, [files]);

  // Filtered files
  const filteredFiles = useMemo(() =>
    files
      .filter(f => {
        const matchSearch = f.name.toLowerCase().includes(searchTerm.toLowerCase());
        const matchSplit  = currentSplit === 'all' || f.split === currentSplit;
        return matchSearch && matchSplit;
      })
      .sort((a, b) => a.name.localeCompare(b.name)),
    [files, searchTerm, currentSplit]
  );

  // ── Actions ──

  const viewFile = async (filePath: string) => {
    setLoadingContent(true);
    setViewingFile(filePath);
    setParquetPreview(null);
    setParquetError(null);
    const isParquet = filePath.toLowerCase().endsWith('.parquet');
    try {
      if (isParquet) {
        const preview = await invoke<ParquetPreview>('preview_parquet_file', { filePath, maxRows: 50 });
        setParquetPreview(preview);
      } else {
        const content = await invoke<string>('read_dataset_file', { filePath });
        setFileContent(content);
      }
    } catch (err: unknown) {
      if (isParquet) {
        setParquetError(String(err));
      } else {
        error(t('datasetFileManager.notifications.readError'), String(err));
        setViewingFile(null);
      }
    } finally {
      setLoadingContent(false);
    }
  };

  const moveFiles = async (targetSplit: 'train' | 'val' | 'test') => {
    if (selectedFiles.size === 0) return;
    try {
      await invoke('move_dataset_files', {
        datasetId, filePaths: Array.from(selectedFiles), targetSplit,
      });
      success(
        t('datasetFileManager.notifications.moveSuccess').replace('{count}', String(selectedFiles.size)).replace('{split}', targetSplit),
        '',
      );
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: unknown) {
      error(t('datasetFileManager.notifications.moveError'), String(err));
    }
  };

  const deleteFiles = async () => {
    if (selectedFiles.size === 0) return;
    try {
      await invoke('delete_dataset_files', {
        datasetId, filePaths: Array.from(selectedFiles),
      });
      success(t('datasetFileManager.notifications.deleteSuccess').replace('{count}', String(selectedFiles.size)), '');
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: unknown) {
      error(t('datasetFileManager.notifications.deleteError'), String(err));
    }
  };

  const addFiles = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({ multiple: true, title: t('datasetFileManager.toolbar.addDialogTitle') });
      if (selected) {
        const paths = Array.isArray(selected) ? selected : [selected];
        const result = await invoke<{ added: number }>('add_files_to_dataset', { datasetId, filePaths: paths });
        success(t('datasetFileManager.notifications.addSuccess').replace('{count}', String(result.added ?? paths.length)), '');
        loadFiles();
      }
    } catch (err: unknown) {
      error(t('datasetFileManager.notifications.addError'), String(err));
    }
  };

  const saveYaml = async () => {
    setYamlSaving(true);
    try {
      await invoke('save_dataset_yaml', {
        datasetId,
        trainPath: editTrainPath,
        valPath: editValPath,
        names: editNames,
      });
      setYamlData(prev => prev ? { ...prev, train_path: editTrainPath, val_path: editValPath, names: editNames, nc: editNames.length } : prev);
      setYamlSaved(true);
      setTimeout(() => setYamlSaved(false), 2000);
      success(t('datasetFileManager.yaml.saveSuccess'), t('datasetFileManager.yaml.saveSuccessDetail'));
    } catch (err) {
      error(t('datasetFileManager.yaml.saveError'), String(err));
    } finally {
      setYamlSaving(false);
    }
  };

  const toggleFile = (path: string) => {
    const s = new Set(selectedFiles);
    s.has(path) ? s.delete(path) : s.add(path);
    setSelectedFiles(s);
  };

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        className="relative w-full max-w-5xl h-[88vh] rounded-2xl shadow-2xl flex flex-col border border-white/10 bg-[rgb(13,20,38)]"
      >
        {/* Header */}
        <div
          className="px-6 py-5 border-b border-white/10 flex justify-between items-start flex-shrink-0"
          style={{ background: `linear-gradient(to right, ${currentTheme.colors.primary}12, transparent)` }}
        >
          <div>
            <h2 className="text-xl font-bold text-white">{datasetName}</h2>
            {datasetType && PAIRED_TYPES.includes(datasetType) && (
              <div className="mt-2 flex items-start gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs">
                <span className="flex-shrink-0 mt-0.5">⚠️</span>
                <span>{t(`datasetFileManager.pairedHints.${datasetType}`)}</span>
              </div>
            )}
            <div className="flex items-center gap-4 mt-1.5 text-xs text-gray-400">
              {/* "Dateien gesamt" war falsch: gelistet werden nur die Dateien in den
                  bekannten Split- und Medienordnern, nicht jede Datei im Datensatz
                  (die Karte zaehlt rekursiv alles, z. B. auch mitkopierte Rohdaten). */}
              <span>{t('datasetFileManager.header.totalFiles').replace('{count}', String(files.length))}</span>
              <span className="text-blue-400">{t('datasetFileManager.header.statTrain').replace('{count}', String(stats.train))}</span>
              <span className="text-purple-400">{t('datasetFileManager.header.statVal').replace('{count}', String(stats.val))}</span>
              <span className="text-green-400">{t('datasetFileManager.header.statTest').replace('{count}', String(stats.test))}</span>
              {stats.unsplit > 0 && <span className="text-gray-500">{t('datasetFileManager.header.statUnsplit').replace('{count}', String(stats.unsplit))}</span>}
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs: Dateien / dataset.yaml */}
        {datasetType === 'yolo_bbox' && (
          <div className="flex border-b border-white/10 flex-shrink-0">
            {([{ id: 'files' as const, label: t('datasetFileManager.tabs.files'), icon: <FolderOpen className="w-4 h-4" /> }, { id: 'yaml' as const, label: t('datasetFileManager.tabs.yaml'), icon: <Edit3 className="w-4 h-4" /> }]).map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all border-b-2 ${
                  activeTab === tab.id ? 'text-white' : 'text-gray-500 hover:text-gray-300 border-transparent'
                }`}
                style={activeTab === tab.id ? { borderColor: currentTheme.colors.primary, color: currentTheme.colors.primary } : {}}
              >
                {tab.icon}{tab.label}
              </button>
            ))}
          </div>
        )}

        {/* Toolbar – nur im Files-Tab */}
        {activeTab === 'files' && (
        <div className="px-4 py-3 border-b border-white/10 flex gap-3 items-center flex-wrap bg-white/[0.015] flex-shrink-0">
          {/* Search */}
          <div className="flex-1 min-w-[180px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
              type="text"
              placeholder={t('datasetFileManager.toolbar.searchPlaceholder')}
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder:text-gray-600 outline-none focus:border-white/20 transition-all"
            />
          </div>

          {/* Split filter */}
          <div className="flex gap-1.5 flex-wrap">
            {['all', 'train', 'val', 'test', 'images', 'labels', 'annotations'].filter(split => {
              if (split === 'all') return true;
              // Nur anzeigen wenn Dateien in diesem Split/Ordner vorhanden
              return files.some(f => f.split === split);
            }).map(split => (
              <button
                key={split}
                onClick={() => setCurrentSplit(split)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                style={{
                  backgroundColor: currentSplit === split ? (SPLIT_COLORS[split] ?? currentTheme.colors.primary) + '33' : 'transparent',
                  color: currentSplit === split ? '#fff' : '#9ca3af',
                  border: `1px solid ${
                    currentSplit === split
                      ? (SPLIT_COLORS[split] ?? currentTheme.colors.primary)
                      : 'rgba(255,255,255,0.08)'
                  }`,
                }}
              >
                {split === 'all' ? t('datasetFileManager.toolbar.filterAll') : (SPLIT_LABELS[split] ?? split)}
                {split !== 'all' && (
                  <span className="ml-1.5 opacity-60 text-xs">
                    {files.filter(f => f.split === split).length}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Add & Refresh */}
          <button
            onClick={addFiles}
            className="px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 hover:opacity-90 transition-all text-white"
            style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})` }}
          >
            <Upload className="w-4 h-4" /> {t('datasetFileManager.toolbar.addButton')}
          </button>
          <button onClick={loadFiles} disabled={loading} className="p-1.5 rounded-lg hover:bg-white/10 transition-all text-gray-500 hover:text-white">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        )} {/* end activeTab === 'files' Toolbar */}

        {/* YAML-Editor Tab */}
        {activeTab === 'yaml' && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {yamlLoading ? (
              <div className="flex items-center justify-center h-40">
                <Loader2 className="w-7 h-7 animate-spin" style={{ color: currentTheme.colors.primary }} />
              </div>
            ) : (
              <>
                <p className="text-xs text-gray-500">{t('datasetFileManager.yaml.description')}</p>

                {/* Pfade */}
                <section className="space-y-3">
                  <h3 className="text-sm font-semibold text-white">{t('datasetFileManager.yaml.sectionPaths')}</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="block text-xs text-gray-400">
                        <code className="text-orange-400">{t('datasetFileManager.yaml.trainPathLabel')}</code>
                      </label>
                      <input
                        type="text"
                        value={editTrainPath}
                        onChange={e => setEditTrainPath(e.target.value)}
                        placeholder={t('datasetFileManager.yaml.trainPathPlaceholder')}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm font-mono focus:outline-none focus:border-white/30 transition-all"
                      />
                      <p className="text-[10px] text-gray-600">z.B. <code>{t('datasetFileManager.yaml.trainPathPlaceholder')}</code> oder <code>train/images</code></p>
                    </div>
                    <div className="space-y-1.5">
                      <label className="block text-xs text-gray-400">
                        <code className="text-orange-400">{t('datasetFileManager.yaml.valPathLabel')}</code>
                      </label>
                      <input
                        type="text"
                        value={editValPath}
                        onChange={e => setEditValPath(e.target.value)}
                        placeholder={t('datasetFileManager.yaml.valPathPlaceholder')}
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm font-mono focus:outline-none focus:border-white/30 transition-all"
                      />
                      <p className="text-[10px] text-gray-600">z.B. <code>{t('datasetFileManager.yaml.valPathPlaceholder')}</code> oder <code>val/images</code></p>
                    </div>
                  </div>
                </section>

                {/* Klassen */}
                <section className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">
                      {t('datasetFileManager.yaml.sectionClasses')} <span className="text-gray-500 font-normal ml-1">{t('datasetFileManager.yaml.ncLabel').replace('{count}', String(editNames.length))}</span>
                    </h3>
                  </div>

                  {/* Klassen-Liste */}
                  <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                    {editNames.map((name, idx) => (
                      <div key={idx} className="flex items-center gap-2 group">
                        <span className="text-gray-600 text-xs font-mono w-6 text-right flex-shrink-0">{idx}</span>
                        <input
                          type="text"
                          value={name}
                          onChange={e => {
                            const next = [...editNames];
                            next[idx] = e.target.value;
                            setEditNames(next);
                          }}
                          className="flex-1 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-white/30 transition-all"
                        />
                        <button
                          onClick={() => setEditNames(editNames.filter((_, i) => i !== idx))}
                          className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
                          title={t('datasetFileManager.yaml.removeClassTooltip')}
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                    {editNames.length === 0 && (
                      <p className="text-gray-600 text-xs text-center py-4">{t('datasetFileManager.yaml.noClasses')}</p>
                    )}
                  </div>

                  {/* Neue Klasse */}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={newClassName}
                      onChange={e => setNewClassName(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && newClassName.trim()) {
                          setEditNames([...editNames, newClassName.trim()]);
                          setNewClassName('');
                        }
                      }}
                      placeholder={t('datasetFileManager.yaml.newClassPlaceholder')}
                      className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all"
                    />
                    <button
                      onClick={() => { if (newClassName.trim()) { setEditNames([...editNames, newClassName.trim()]); setNewClassName(''); } }}
                      disabled={!newClassName.trim()}
                      className="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-white text-sm transition-all disabled:opacity-40 flex items-center gap-1.5"
                    >
                      <Plus className="w-4 h-4" /> {t('datasetFileManager.yaml.addClassButton')}
                    </button>
                  </div>
                </section>

                {/* Vorschau */}
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-white">{t('datasetFileManager.yaml.sectionPreview')}</h3>
                  <pre className="text-xs font-mono bg-black/40 border border-white/10 rounded-xl p-4 text-gray-300 leading-relaxed overflow-x-auto">{
`# FrameTrain – dataset.yaml (Ultralytics-kompatibel)
path: <dataset-root>
train: ${editTrainPath || 'images/train'}  # relativer Pfad zum Trainings-Bilder-Ordner
val:   ${editValPath || 'images/val'}      # relativer Pfad zum Validierungs-Bilder-Ordner

nc: ${editNames.length}
names:
${editNames.length > 0 ? editNames.map(n => `  - '${n}'`).join('\n') : `  # ${t('datasetFileManager.yaml.previewNoClasses')}`}`.trim()
                  }</pre>
                </section>

                {/* Save */}
                <button
                  onClick={saveYaml}
                  disabled={yamlSaving}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50"
                >
                  {yamlSaving
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('datasetFileManager.yaml.savingButton')}</>
                    : yamlSaved
                    ? <><Check className="w-4 h-4" /> {t('datasetFileManager.yaml.savedButton')}</>
                    : <><Save className="w-4 h-4" /> {t('datasetFileManager.yaml.saveButton')}</>
                  }
                </button>
              </>
            )}
          </div>
        )}

        {/* Selection Actions – nur im Files-Tab */}
        {activeTab === 'files' && selectedFiles.size > 0 && (
          <div
            className="px-5 py-2.5 border-b flex gap-3 items-center text-sm flex-shrink-0"
            style={{ background: `linear-gradient(to right, ${currentTheme.colors.primary}18, transparent)`, borderColor: currentTheme.colors.primary + '25' }}
          >
            <span className="text-white font-medium">{t('datasetFileManager.selection.selectedCount').replace('{count}', String(selectedFiles.size))}</span>
            <div className="flex gap-2 ml-auto">
              {(['train', 'val', 'test'] as const).map(s => (
                  <button key={s} onClick={() => moveFiles(s)}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1 transition-all"
                  style={{ backgroundColor: SPLIT_COLORS[s] + '20', color: SPLIT_COLORS[s], border: `1px solid ${SPLIT_COLORS[s]}40` }}
                >
                  <ArrowRight className="w-3 h-3" /> {s === 'train' ? t('datasetFileManager.selection.moveToTrain') : s === 'val' ? t('datasetFileManager.selection.moveToVal') : t('datasetFileManager.selection.moveToTest')}
                </button>
              ))}
              <button
                onClick={deleteFiles}
                className="px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1 bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-all"
              >
                <Trash2 className="w-3 h-3" /> {t('datasetFileManager.selection.deleteButton')}
              </button>
              <button onClick={() => setSelectedFiles(new Set())} className="px-2.5 py-1 rounded-lg text-xs text-gray-400 hover:text-white bg-white/5 border border-white/10 transition-all">
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* File List – nur im Files-Tab */}
        {activeTab === 'files' && (
          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <Loader2 className="w-7 h-7 animate-spin" style={{ color: currentTheme.colors.primary }} />
              </div>
            ) : filteredFiles.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-600">
                <FileText className="w-10 h-10" />
                <p className="text-sm">{t('datasetFileManager.table.noFiles')}</p>
              </div>
            ) : (
              <table className="w-full">
                <thead className="sticky top-0 bg-[rgb(13,20,38)] z-10">
                  <tr className="border-b border-white/10">
                    <th className="text-left p-2.5 w-8">
                      <input
                        type="checkbox"
                        className="w-4 h-4 cursor-pointer accent-violet-500"
                        checked={selectedFiles.size === filteredFiles.length && filteredFiles.length > 0}
                        onChange={e => {
                          if (e.target.checked) setSelectedFiles(new Set(filteredFiles.map(f => f.path)));
                          else setSelectedFiles(new Set());
                        }}
                      />
                    </th>
                    <th className="text-left p-2.5 text-xs text-gray-500 font-medium">{t('datasetFileManager.table.colName')}</th>
                    <th className="text-left p-2.5 text-xs text-gray-500 font-medium">{t('datasetFileManager.table.colFolder')}</th>
                    <th className="text-left p-2.5 text-xs text-gray-500 font-medium">{t('datasetFileManager.table.colSize')}</th>
                    <th className="p-2.5 w-10" />
                  </tr>
                </thead>
                <tbody>
                  {filteredFiles.map(file => (
                    <tr key={file.path} className="border-b border-white/[0.05] hover:bg-white/[0.03] transition-colors group">
                      <td className="p-2.5 w-8">
                        <input
                          type="checkbox"
                          checked={selectedFiles.has(file.path)}
                          onChange={() => toggleFile(file.path)}
                          className="w-4 h-4 cursor-pointer accent-violet-500"
                        />
                      </td>
                      <td className="p-2.5">
                        <div className="flex items-center gap-2">
                          <File className="w-4 h-4 text-gray-600 flex-shrink-0" />
                          <span className="text-white text-sm truncate max-w-xs" title={file.name}>{file.name}</span>
                        </div>
                      </td>
                      <td className="p-2.5">
                        <span
                          className="px-2 py-0.5 rounded text-xs font-medium"
                          style={{
                            backgroundColor: (SPLIT_COLORS[file.split] ?? '#6b7280') + '33',
                            color: SPLIT_COLORS[file.split] ?? '#9ca3af',
                          }}
                        >
                          {SPLIT_LABELS[file.split] ?? file.split}
                        </span>
                      </td>
                      <td className="p-2.5 text-xs text-gray-500 tabular-nums">{formatBytes(file.size)}</td>
                      <td className="p-2.5">
                        <button
                          onClick={() => viewFile(file.path)}
                          className="p-1.5 rounded-lg hover:bg-white/10 transition-all text-gray-600 hover:text-white opacity-0 group-hover:opacity-100"
                          title={t('datasetFileManager.table.viewTooltip')}
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {/* File Viewer Overlay */}
        {viewingFile && (
          <div className="absolute inset-0 z-20 bg-black/90 backdrop-blur-sm flex items-center justify-center p-8 rounded-2xl">
            <div
              className={`w-full h-full rounded-2xl flex flex-col border bg-[rgb(13,20,38)] ${parquetPreview ? 'max-w-6xl' : 'max-w-4xl'}`}
              style={{ borderColor: currentTheme.colors.primary + '40' }}
            >
              <div
                className="px-5 py-4 border-b border-white/10 flex justify-between items-center flex-shrink-0"
                style={{ background: `linear-gradient(to right, ${currentTheme.colors.primary}10, transparent)` }}
              >
                <h3 className="text-white text-sm font-medium truncate flex-1">
                  <Tag className="inline w-4 h-4 mr-2 opacity-50" />
                  {viewingFile.replace(/\\/g, '/').split('/').pop()}
                </h3>
                <button onClick={() => setViewingFile(null)} className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 overflow-auto p-5 bg-black/20">
                {loadingContent ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2 className="w-7 h-7 animate-spin" style={{ color: currentTheme.colors.primary }} />
                  </div>
                ) : parquetError ? (
                  <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-8">
                    <FileText className="w-10 h-10 text-red-400/60" />
                    <p className="text-red-300 text-sm font-medium">{t('datasetFileManager.parquet.previewErrorTitle')}</p>
                    <p className="text-gray-500 text-xs max-w-md">{parquetError}</p>
                    <p className="text-gray-600 text-xs">{t('datasetFileManager.parquet.previewErrorHint')}</p>
                  </div>
                ) : parquetPreview ? (
                  <div className="space-y-3">
                    {/* Meta-Info Bar */}
                    <div className="flex items-center gap-4 text-xs text-gray-400 pb-3 border-b border-white/10">
                      <span className="flex items-center gap-1.5">
                        <span className="text-violet-400 font-semibold">{parquetPreview.total_rows.toLocaleString()}</span>
                        {t('datasetFileManager.parquet.totalRows')}
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="text-violet-400 font-semibold">{parquetPreview.total_cols}</span>
                        {t('datasetFileManager.parquet.totalCols')}
                      </span>
                      <span className="text-gray-600">
                        {t('datasetFileManager.parquet.showingRows').replace('{count}', String(parquetPreview.shown_rows))}
                      </span>
                    </div>

                    {/* Spalten-Schema */}
                    <div className="flex flex-wrap gap-1.5 pb-1">
                      {parquetPreview.columns.map(col => (
                        <span key={col.name} className="px-2 py-1 rounded-lg bg-violet-500/10 border border-violet-500/20 text-violet-300 text-[11px] font-mono">
                          {col.name} <span className="text-violet-400/50">·{col.dtype}</span>
                        </span>
                      ))}
                    </div>

                    {/* Tabelle */}
                    <div className="overflow-auto rounded-xl border border-white/10">
                      <table className="w-full text-xs">
                        <thead className="sticky top-0 bg-[rgb(13,20,38)] z-10">
                          <tr className="border-b border-white/10">
                            <th className="text-left p-2 text-gray-600 font-medium w-10">#</th>
                            {parquetPreview.columns.map(col => (
                              <th key={col.name} className="text-left p-2 text-gray-400 font-medium whitespace-nowrap">{col.name}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {parquetPreview.rows.map((row, i) => (
                            <tr key={i} className="border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors">
                              <td className="p-2 text-gray-600 tabular-nums">{i}</td>
                              {parquetPreview.columns.map(col => {
                                const val = row[col.name];
                                const display = val === null || val === undefined ? '—'
                                  : typeof val === 'object' ? JSON.stringify(val)
                                  : String(val);
                                return (
                                  <td key={col.name} className="p-2 text-gray-300 max-w-xs truncate" title={display}>
                                    {display.length > 120 ? display.slice(0, 120) + '…' : display}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {parquetPreview.total_rows > parquetPreview.shown_rows && (
                      <p className="text-gray-600 text-[11px] text-center pt-1">
                        {t('datasetFileManager.parquet.moreRowsNote').replace('{count}', String(parquetPreview.total_rows - parquetPreview.shown_rows))}
                      </p>
                    )}
                  </div>
                ) : (
                  <pre className="text-sm font-mono whitespace-pre-wrap break-words text-gray-300 leading-relaxed">
                    {fileContent}
                  </pre>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
