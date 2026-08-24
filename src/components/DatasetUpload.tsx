// DatasetUpload.tsx – Dataset-Manager v2
// Neu: DatasetType-Erkennung, Analyse-Vorschau, Typ-Badge in Cards, Pairing-Status

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { open } from '@tauri-apps/plugin-dialog';
import {
  Upload, FolderOpen, Download, Trash2, Search,
  HardDrive, Cloud, CheckCircle, Loader2, Database,
  Calendar, ExternalLink, X, RefreshCw, ChevronDown,
  Scissors, Layers, FileText, Filter, AlertTriangle, AlertCircle,
  Zap, Heart, Info, Target, Folder, Mic, FolderTree,
} from 'lucide-react';
import { useContextMenuActions } from '../ui/contextMenuRegistry';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';
import { usePageContext } from '../contexts/PageContext';
import { onCoachCommand, consumePendingCoachCommand, type CoachCommand } from '../ai/coachToolEvents';
import { useLanguage, type Language } from '../contexts/LanguageContext';
import DatasetFileManager from './DatasetFileManager';
import { DATASET_TYPE_LABELS } from '../plugins/datasetCompatHelpers';
import DatasetTypeIcon from './DatasetTypeIcon';
import type { DatasetType, PairingStatus, DatasetAnalysis } from '../plugins/datasetCompatHelpers';
import { detectPlugin } from '../plugins/registry';
import { dateLocale } from '../utils/dateLocale';

// ── Types ──────────────────────────────────────────────────────────────────

interface ModelInfo { id: string; name: string; source: string; }

interface SplitInfo {
  train_count: number; val_count: number; test_count: number;
  train_ratio: number; val_ratio: number; test_ratio: number;
}

interface DatasetInfo {
  id:              string;
  name:            string;
  model_id:        string;
  source:          'local' | 'huggingface';
  source_path:     string | null;
  size_bytes:      number;
  file_count:      number;
  created_at:      string;
  status:          'unused' | 'split';
  split_info:      SplitInfo | null;
  training_count:  number;
  last_used_at:    string | null;
  extensions?:     string[];
  // v2
  dataset_type?:   DatasetType;
  pairing_status?: PairingStatus | null;
  warnings?:       string[];
}

interface HuggingFaceDataset {
  id: string; author?: string;
  downloads?: number; likes?: number; tags?: string[];
}

interface FilterOptions { tasks: string[]; languages: string[]; sizes: string[]; }

type ImportMode = 'local' | 'huggingface';

// ── Helpers ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatDate(ds: string, language: Language): string {
  return new Date(ds).toLocaleDateString(dateLocale(language), {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDownloads(n: number | undefined): string {
  if (!n) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

// ── AnalysisPreview (im Import-Modal nach Ordnerauswahl) ───────────────────

interface AnalysisPreviewProps {
  analysis: DatasetAnalysis;
  /** Falls gesetzt: Plugin-Kompatibilitätsprüfung anzeigen */
  modelId?: string | null;
}

function AnalysisPreview({ analysis, modelId }: AnalysisPreviewProps) {
  const { t } = useLanguage();
  const typeMeta = DATASET_TYPE_LABELS[analysis.detected_type] ?? DATASET_TYPE_LABELS['unknown'];

  // Plugin-Kompatibilität prüfen
  let pluginCompat: { ok: boolean; label: string; preferred: boolean } | null = null;
  if (modelId) {
    const result = detectPlugin(modelId);
    if (result.supported) {
      const { plugin } = result;
      const supported = plugin.supportedDatasetTypes;
      const preferred = plugin.preferredDatasetType;
      if (supported && supported.length > 0) {
        const isSupported = supported.includes(analysis.detected_type);
        const isPreferred = preferred === analysis.detected_type;
        pluginCompat = {
          ok: isSupported,
          preferred: isPreferred,
          label: isPreferred
            ? t('datasetUpload.analysisPreview.preferredType').replace('{name}', plugin.name)
            : isSupported
            ? t('datasetUpload.analysisPreview.compatible').replace('{name}', plugin.name)
            : t('datasetUpload.analysisPreview.notRecommended')
                .replace('{name}', plugin.name)
                .replace('{types}', supported.map(t => DATASET_TYPE_LABELS[t]?.label ?? t).join(', ')),
        };
      }
    }
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <DatasetTypeIcon icon={typeMeta.icon} className={`w-6 h-6 flex-shrink-0 ${typeMeta.color}`} />
          <div>
            <p className={`text-sm font-semibold ${typeMeta.color}`}>{typeMeta.label}</p>
            <p className="text-gray-500 text-xs">{t('datasetUpload.analysisPreview.confidenceLabel').replace('{confidence}', String(analysis.confidence)).replace('{count}', String(analysis.file_count))}</p>
          </div>
        </div>
        <div title={t('datasetUpload.analysisPreview.autoDetectedTooltip')}>
          <Zap className="w-4 h-4 text-gray-600 flex-shrink-0 mt-0.5" />
        </div>
      </div>

      {/* Plugin-Kompatibilität */}
      {pluginCompat && (
        <div className={`flex items-start gap-2 text-xs px-3 py-2 rounded-lg border ${
          pluginCompat.preferred
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
            : pluginCompat.ok
            ? 'bg-blue-500/10 border-blue-500/20 text-blue-300'
            : 'bg-red-500/10 border-red-500/20 text-red-300'
        }`}>
          {pluginCompat.preferred
            ? <CheckCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            : pluginCompat.ok
            ? <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            : <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          }
          <span>{pluginCompat.label}</span>
        </div>
      )}

      {/* Pairing-Status */}
      {analysis.pairing_status && (
        <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${
          analysis.pairing_status.is_paired
            ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
            : 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
        }`}>
          {analysis.pairing_status.is_paired ? (
            <><CheckCircle className="w-3.5 h-3.5" /> {t('datasetUpload.analysisPreview.pairedCount').replace('{count}', String(analysis.pairing_status.paired_count))}</>
          ) : (
            <><AlertTriangle className="w-3.5 h-3.5" /> {t('datasetUpload.analysisPreview.orphanCount').replace('{count}', String(analysis.pairing_status.orphan_primaries.length))}</>
          )}
        </div>
      )}

      {/* Warnungen */}
      {analysis.warnings.length > 0 && (
        <div className="space-y-1">
          {analysis.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-amber-400/80">
              <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Extensions */}
      {analysis.extensions.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {analysis.extensions.map(ext => (
            <span key={ext} className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-gray-400 text-xs font-mono">{ext}</span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── DatasetStructureGuide ────────────────────────────────────────────────

const STRUCTURE_TYPES = [
  {
    id: 'yolo',
    icon: <Target className="w-4 h-4" />,
    labelKey: 'yolo',
    color: 'text-orange-400',
    hintKey: 'yolo',
    hintColor: 'text-amber-400',
    tree: [
      { text: 'mein-dataset/',   indent: 0, bold: true,  marker: '▸' },
      { text: 'images/',         indent: 1, bold: false, marker: '├─' },
      { text: 'foto1.jpg',       indent: 2, bold: false, marker: '│  ├─' },
      { text: 'foto2.jpg',       indent: 2, bold: false, marker: '│  └─' },
      { text: 'labels/',         indent: 1, bold: false, marker: '└─' },
      { text: 'foto1.txt',       indent: 2, bold: false, marker: '   ├─' },
      { text: 'foto2.txt',       indent: 2, bold: false, marker: '   └─' },
    ],
  },
  {
    id: 'flatfile',
    icon: <FileText className="w-4 h-4" />,
    labelKey: 'flatfile',
    color: 'text-violet-400',
    hintKey: 'flatfile',
    hintColor: 'text-gray-400',
    tree: [
      { text: 'mein-dataset/',   indent: 0, bold: true,  marker: '▸' },
      { text: 'train.jsonl',     indent: 1, bold: false, marker: '├─' },
      { text: 'val.jsonl',       indent: 1, bold: false, marker: '└─' },
    ],
  },
  {
    id: 'folderclass',
    icon: <Folder className="w-4 h-4" />,
    labelKey: 'folderclass',
    color: 'text-blue-400',
    hintKey: 'folderclass',
    hintColor: 'text-gray-400',
    tree: [
      { text: 'mein-dataset/',   indent: 0, bold: true,  marker: '▸' },
      { text: 'katze/',          indent: 1, bold: false, marker: '├─' },
      { text: 'bild1.jpg',       indent: 2, bold: false, marker: '│  └─' },
      { text: 'hund/',           indent: 1, bold: false, marker: '└─' },
      { text: 'bild2.jpg',       indent: 2, bold: false, marker: '   └─' },
    ],
  },
  {
    id: 'audio',
    icon: <Mic className="w-4 h-4" />,
    labelKey: 'audio',
    color: 'text-cyan-400',
    hintKey: 'audio',
    hintColor: 'text-gray-400',
    tree: [
      { text: 'mein-dataset/',   indent: 0, bold: true,  marker: '▸' },
      { text: 'aufnahme1.wav',   indent: 1, bold: false, marker: '├─' },
      { text: 'aufnahme1.txt',   indent: 1, bold: false, marker: '├─' },
      { text: 'aufnahme2.mp3',   indent: 1, bold: false, marker: '├─' },
      { text: 'aufnahme2.txt',   indent: 1, bold: false, marker: '└─' },
    ],
  },
  {
    id: 'pascal',
    icon: <FolderTree className="w-4 h-4" />,
    labelKey: 'pascal',
    color: 'text-yellow-400',
    hintKey: 'pascal',
    hintColor: 'text-gray-400',
    tree: [
      { text: 'mein-dataset/',   indent: 0, bold: true,  marker: '▸' },
      { text: 'images/',         indent: 1, bold: false, marker: '├─' },
      { text: 'bild1.jpg',       indent: 2, bold: false, marker: '│  └─' },
      { text: 'annotations/',    indent: 1, bold: false, marker: '└─' },
      { text: 'bild1.xml',       indent: 2, bold: false, marker: '   └─' },
    ],
  },
];

function DatasetStructureGuide() {
  const { t } = useLanguage();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState('yolo');
  const current = STRUCTURE_TYPES.find(t => t.id === active)!;

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-all"
      >
        <span className="flex items-center gap-2 text-sm text-gray-400">
          <FolderOpen className="w-4 h-4" />
          {t('datasetUpload.structureGuide.toggleLabel')}
        </span>
        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="border-t border-white/10">
          {/* Typ-Tabs */}
          <div className="flex overflow-x-auto gap-1 p-2 border-b border-white/10 scrollbar-hide">
            {STRUCTURE_TYPES.map(type => (
              <button
                key={type.id}
                onClick={() => setActive(type.id)}
                className={`flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  active === type.id
                    ? 'bg-white/10 text-white'
                    : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                <span>{type.icon}</span>
                <span>{t(`datasetUpload.structureGuide.types.${type.id}.label`)}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="p-4 space-y-3">
            {/* Ordner-Baum */}
            <div className="rounded-lg bg-black/30 border border-white/5 p-3 font-mono text-xs space-y-0.5">
              {current.tree.map((row, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <span className="text-gray-600 select-none">{row.marker}</span>
                  <span className={row.bold ? `font-semibold ${current.color}` : 'text-gray-300'}>
                    {row.text}
                  </span>
                </div>
              ))}
            </div>

            {/* Erwartete Spalten. Der Guide zeigte bisher nur die
                Ordnerstruktur — welche Spalten eine CSV/JSONL haben muss,
                stand nirgends, und ohne 'text'/'label' scheitert das
                Training mit einem unverständlichen Fehler. */}
            {current.id === 'flatfile' && (
              <div className="rounded-lg bg-white/5 border border-white/10 p-3 space-y-2">
                <p className="text-xs font-semibold text-white">
                  {t('datasetUpload.structureGuide.schema.title')}
                </p>
                <div className="font-mono text-xs text-gray-300 bg-black/30 rounded-md p-2 overflow-x-auto">
                  text,label<br />
                  Der Film war grossartig,1<br />
                  Totale Zeitverschwendung,0
                </div>
                <p className="text-xs text-gray-400 leading-relaxed">
                  {t('datasetUpload.structureGuide.schema.columns')}
                </p>
                <p className="text-xs text-gray-500 leading-relaxed">
                  {t('datasetUpload.structureGuide.schema.alternatives')}
                </p>
              </div>
            )}

            {/* Hinweis */}
            <div className={`flex items-start gap-2 text-xs ${current.hintColor}`}>
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span>{t(`datasetUpload.structureGuide.types.${current.id}.hint`)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Delete Dialog ──────────────────────────────────────────────────────────

function DeleteDialog({ name, onConfirm, onCancel }: { name: string; onConfirm: () => void; onCancel: () => void }) {
  const { t } = useLanguage();
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}>
      <div className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-sm mx-4 overflow-hidden">
        <div className="h-1 bg-gradient-to-r from-red-500 to-orange-500" />
        <div className="p-6 space-y-5">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center flex-shrink-0">
              <Trash2 className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <h2 className="text-white font-semibold text-lg">{t('datasetUpload.deleteDialog.title')}</h2>
              <p className="text-gray-400 text-sm mt-1.5 leading-relaxed">
                {t('datasetUpload.deleteDialog.description').replace('{name}', name)}
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={onCancel} className="flex-1 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white text-sm font-medium transition-all">{t('datasetUpload.deleteDialog.cancelButton')}</button>
            <button onClick={onConfirm} className="flex-1 py-2.5 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-xl text-red-300 text-sm font-medium transition-all">{t('datasetUpload.deleteDialog.confirmButton')}</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function DatasetUpload() {
  const { currentTheme } = useTheme();
  const { success, error, warning, info } = useNotification();
  const { setCurrentPageContent } = usePageContext();
  const { t } = useLanguage();

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const [deleteTarget, setDeleteTarget] = useState<DatasetInfo | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  const [importMode, setImportMode] = useState<ImportMode>('local');

  // Local import
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [analysisResult, setAnalysisResult] = useState<DatasetAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // HuggingFace
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<HuggingFaceDataset[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [selectedHfDataset, setSelectedHfDataset] = useState<HuggingFaceDataset | null>(null);
  const [hfDatasetName, setHfDatasetName] = useState('');
  const [downloading, setDownloading] = useState(false);
  /** Bleibt im Dialog stehen, auch wenn der Fehler-Toast längst weg ist. */
  const [hfDownloadError, setHfDownloadError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<{
    status: string; currentFile: string; currentFileIndex: number;
    totalFiles: number; downloadedBytes: number; totalBytes: number;
    progressPercent: number; speedMbs: number; elapsedSecs: number;
    etaSecs: number; message: string;
  } | null>(null);

  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterTask, setFilterTask] = useState('');
  const [filterLanguage, setFilterLanguage] = useState('');
  const [filterSize, setFilterSize] = useState('');

  const [showSplitModal, setShowSplitModal] = useState(false);
  const [datasetToSplit, setDatasetToSplit] = useState<DatasetInfo | null>(null);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [splitting, setSplitting] = useState(false);

  const [showHalveModal, setShowHalveModal] = useState(false);
  const [datasetToHalve, setDatasetToHalve] = useState<DatasetInfo | null>(null);
  const [halving, setHalving] = useState(false);
  // Halbieren: train/val/test-Struktur an beide Hälften vererben (Standard: an)
  const [inheritSplits, setInheritSplits] = useState(true);

  const [fileManagerDataset, setFileManagerDataset] = useState<DatasetInfo | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Init ──

  useEffect(() => { initLoad(); }, []);
  useEffect(() => { if (selectedModelId) loadDatasets(); }, [selectedModelId]);

  useEffect(() => {
    const selModel = models.find(m => m.id === selectedModelId);
    const contextLines: string[] = [
      '=== FrameTrain Dataset-Manager ===',
      '',
      '--- SEITENZWECK ---',
      'Lade, verwalte und splitte Trainings-Datensätze für Machine Learning Models.',
      'Wähle Modell → Lade Dataset → Prüfe Kompatibilität → Teile in Train/Val/Test → Trainiere.',
      '',
      '--- MODELL ---',
    ];

    if (!selModel) {
      contextLines.push('❌ Kein Modell → Wähle aus Dropdown');
    } else {
      contextLines.push(`✓ Modell: ${selModel.name}`);
    }

    contextLines.push('');
    contextLines.push('--- DATENSÄTZE ---');
    contextLines.push(`Gesamt: ${datasets.length} Dataset${datasets.length !== 1 ? 's' : ''}`);

    if (datasets.length > 0) {
      contextLines.push('');
      datasets.forEach(d => {
        const typeMeta = d.dataset_type ? DATASET_TYPE_LABELS[d.dataset_type] : null;
        contextLines.push(`• ${d.name}`);
        if (typeMeta) contextLines.push(`  Type: ${typeMeta.label}`);
        contextLines.push(`  Status: ${d.status === 'split' ? '✓ Split' : '⚠️ Nicht aufgeteilt'}`);
        contextLines.push(`  Size: ${d.file_count} Dateien · ${formatBytes(d.size_bytes)}`);
      });
    } else {
      contextLines.push('(Keine Datensätze → Lade eine Datei oben)');
    }

    contextLines.push('');
    contextLines.push('--- UI LAYOUT ---');
    contextLines.push('**OBEN (Header):**');
    contextLines.push('  • [Modell Dropdown] (linke Seite)');
    contextLines.push('');
    contextLines.push('**OBEN RECHTS (Upload Area):**');
    contextLines.push('  • 📤 Drag & Drop Zone (große Box: "Dateien hier ziehen")');
    contextLines.push('  • [Datei durchsuchen Button]');
    contextLines.push('  • Unterstützte Formate: CSV, JSON, JSONL, Parquet, Excel');
    contextLines.push('');
    contextLines.push('**MITTE (Dataset Liste):**');
    contextLines.push('  • Tabelle mit Spalten: Name, Type, Status, Size, Dateien');
    contextLines.push('  • Rechts von jedem Dataset: [Actions Icon ⋮]');
    contextLines.push('    - Ansicht Details');
    contextLines.push('    - Dataset aufteilen (Split)');
    contextLines.push('    - Löschen');
    contextLines.push('');
    contextLines.push('**MODAL (wenn Dataset aufgeteilt wird):**');
    contextLines.push('  • Train/Val/Test Ratio Slider (z.B. 70/15/15)');
    contextLines.push('  • Seed Feld (für Reproduzierbarkeit)');
    contextLines.push('  • [Split Button] (unten rechts)');
    contextLines.push('');
    contextLines.push('--- VERFÜGBARE AKTIONEN ---');
    contextLines.push('1. **Dataset hochladen:**');
    contextLines.push('   → Drag & Drop Datei in Upload Area ODER [Datei durchsuchen]');
    contextLines.push('   → Wähle Modell oben (wird automatisch validiert)');
    contextLines.push('');
    contextLines.push('2. **Dataset aufteilen (Split):**');
    contextLines.push('   → Klick [Actions ⋮] neben Dataset');
    contextLines.push('   → Wähle "Dataset aufteilen"');
    contextLines.push('   → Passe Ratio an (normalerweise 70/15/15)');
    contextLines.push('   → Klick [Split Button]');
    contextLines.push('');
    contextLines.push('3. **Dataset Details:**');
    contextLines.push('   → Klick Dataset Name');
    contextLines.push('   → Sehe Datei-Vorschau, Spalten, Größe');
    contextLines.push('');
    contextLines.push('4. **Mit Training verwenden:**');
    contextLines.push('   → Gehe zu Training Panel');
    contextLines.push('   → Wähle [Dataset Dropdown]');
    contextLines.push('   → Nur aufgeteilte Datensätze erscheinen');

    setCurrentPageContent(contextLines.join('\n'), 'dataset');
  }, [models, selectedModelId, datasets, setCurrentPageContent]);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    const setup = async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        unlisten = await listen<{
          status: string; current_file: string; current_file_index: number;
          total_files: number; downloaded_bytes: number; total_bytes: number;
          progress_percent: number; speed_mbs: number; elapsed_secs: number;
          eta_secs: number; message: string;
        }>('dataset-download-progress', (event) => {
          const p = event.payload;
          setDownloadProgress({
            status: p.status, currentFile: p.current_file,
            currentFileIndex: p.current_file_index, totalFiles: p.total_files,
            downloadedBytes: p.downloaded_bytes, totalBytes: p.total_bytes,
            progressPercent: p.progress_percent, speedMbs: p.speed_mbs,
            elapsedSecs: p.elapsed_secs, etaSecs: p.eta_secs, message: p.message,
          });
          if (p.status === 'complete' || p.status === 'error') setDownloading(false);
        });
      } catch { /* ignore */ }
    };
    setup();
    return () => { if (unlisten) unlisten(); };
  }, []);

  useEffect(() => {
    if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);
    if (hfQuery.trim().length < 2) { setHfResults([]); setHfSearching(false); return; }
    setHfSearching(true);
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const res = await invoke<HuggingFaceDataset[]>('search_huggingface_datasets', {
          query: hfQuery.trim(), limit: 15,
          filterTask: filterTask || null, filterLanguage: filterLanguage || null,
          filterSize: filterSize || null,
        });
        setHfResults(res);
      } catch { /* ignore */ } finally { setHfSearching(false); }
    }, 300);
    return () => { if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current); };
  }, [hfQuery, filterTask, filterLanguage, filterSize]);

  // ── Load ──

  const initLoad = async () => {
    try {
      const list = await invoke<ModelInfo[]>('list_models');
      setModels(list);
      if (list.length > 0) setSelectedModelId(list[0].id);
      try {
        const opts = await invoke<FilterOptions>('get_dataset_filter_options');
        setFilterOptions(opts);
      } catch { /* optional */ }
    } catch (err: unknown) {
      error(t('datasetUpload.notifications.loadModelsError'), String(err));
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    if (!selectedModelId) return;
    try {
      const list = await invoke<DatasetInfo[]>('list_datasets_for_model', { modelId: selectedModelId });
      setDatasets(list);
    } catch (err: unknown) {
      error(t('datasetUpload.notifications.loadDatasetsError'), String(err));
    }
  };

  // ── Rechtsklick-Menü: Dataset-Aktionen ────────────────────────────────────
  useContextMenuActions(() => [
    {
      id: 'ds-import', group: t('sidebar.nav.datasets'),
      label: t('datasetUpload.emptyState.noDatasets.addButton'), icon: Upload,
      disabled: !selectedModelId,
      onSelect: () => setShowImportModal(true),
    },
    {
      id: 'ds-refresh', group: t('sidebar.nav.datasets'),
      label: t('common.refresh'), icon: RefreshCw,
      disabled: !selectedModelId,
      onSelect: () => { void loadDatasets(); },
    },
  ]);

  // ── Local Import ──

  const validateAndSetPath = async (path: string) => {
    setSelectedPath(path);
    setDatasetName(path.split(/[/\\]/).pop() ?? 'Dataset');
    setAnalysisResult(null);
    setAnalysisLoading(true);
    try {
      const analysis = await invoke<DatasetAnalysis>('analyze_dataset_path', { path });
      setAnalysisResult(analysis);
    } catch {
      // Analyse-Fehler ist nicht kritisch
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleBrowseFolder = async () => {
    try {
      const sel = await open({ directory: true, multiple: false, title: t('datasetUpload.importModal.local.browseFolderTitle') });
      if (sel && typeof sel === 'string') await validateAndSetPath(sel);
    } catch (err: unknown) { error(t('common.error'), String(err)); }
  };

  const handleLocalImport = async () => {
    if (!selectedPath || !datasetName.trim() || !selectedModelId) {
      warning(t('datasetUpload.importModal.local.missingFieldsTitle'), t('datasetUpload.importModal.local.missingFieldsDetail'));
      return;
    }
    setImporting(true);
    try {
      const ds = await invoke<DatasetInfo>('import_local_dataset', {
        sourcePath: selectedPath, datasetName: datasetName.trim(), modelId: selectedModelId,
      });
      success(t('datasetUpload.importModal.local.importSuccess'), t('datasetUpload.importModal.local.importSuccessDetail').replace('{name}', ds.name));
      closeModal();
      await loadDatasets();
    } catch (err: unknown) {
      error(t('datasetUpload.importModal.local.importError'), String(err));
    } finally {
      setImporting(false);
    }
  };

  // DOM-Handler: unter Tauri v2 fängt das Webview das OS-Drop ab, bevor das
  // HTML5-drop-Event feuert → echte Pfade kommen über onDragDropEvent (useEffect
  // unten). Diese Handler unterdrücken nur das Browser-Default-Verhalten.
  const handleDragOver  = useCallback((e: React.DragEvent) => { e.preventDefault(); }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); }, []);
  const handleDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); }, []);

  // Tauri v2 Datei-Drop: liefert echte Dateisystempfade. Nur aktiv, solange der
  // lokale Import-Dialog offen ist, damit Drops woanders nichts auslösen.
  useEffect(() => {
    if (!showImportModal || importMode !== 'local') return;
    let active = true;
    let unlisten: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        if (p.type === 'over') {
          setIsDragging(true);
        } else if (p.type === 'drop') {
          setIsDragging(false);
          const first = p.paths?.[0];
          if (first) void validateAndSetPath(first);
        } else {
          setIsDragging(false);
        }
      })
      .then((fn) => { if (active) unlisten = fn; else fn(); })
      .catch(() => { /* Drag-Drop nicht verfügbar — Ordner-Auswahl bleibt */ });
    return () => { active = false; setIsDragging(false); unlisten?.(); };
    // validateAndSetPath ist stabil genug (nutzt nur setState + invoke)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showImportModal, importMode]);

  // ── HuggingFace ──

  const handleHfSelect = (ds: HuggingFaceDataset) => {
    setSelectedHfDataset(ds);
    setHfDatasetName(ds.id.split('/').pop() ?? ds.id);
  };

  const handleHfDownload = async () => {
    setHfDownloadError(null);
    if (!selectedHfDataset || !hfDatasetName.trim() || !selectedModelId) {
      warning(t('datasetUpload.importModal.hf.missingFieldsTitle'), t('datasetUpload.importModal.hf.missingFieldsDetail')); return;
    }
    setDownloading(true); setDownloadProgress(null);
    try {
      const ds = await invoke<DatasetInfo>('download_huggingface_dataset', {
        repoId: selectedHfDataset.id, datasetName: hfDatasetName.trim(), modelId: selectedModelId,
      });
      success(t('datasetUpload.importModal.hf.downloadSuccess'), t('datasetUpload.importModal.hf.downloadSuccessDetail').replace('{name}', ds.name));
      setSelectedHfDataset(null); setHfDatasetName(''); setHfQuery('');
      setHfResults([]); setShowImportModal(false); setDownloadProgress(null);
      await loadDatasets();
    } catch (err: unknown) {
      // Zusaetzlich zum Toast persistent im Dialog anzeigen. Toasts blenden
      // nach wenigen Sekunden aus — wer in der Zeit wegsieht, erlebt den
      // Download als reaktionslos und weiss nicht, dass er fehlgeschlagen ist.
      setHfDownloadError(String(err));
      error(t('datasetUpload.importModal.hf.downloadError'), String(err));
    } finally {
      setDownloading(false);
    }
  };

  const handleCancelDownload = () => {
    setDownloading(false); setDownloadProgress(null);
    info(t('datasetUpload.importModal.hf.cancelSuccess'), t('datasetUpload.importModal.hf.cancelDetail'));
  };

  // ── Split ──

  const openSplitModal = (ds: DatasetInfo) => {
    setDatasetToSplit(ds);
    setTrainRatio(0.8); setValRatio(0.1); setTestRatio(0.1);
    setShowSplitModal(true);
  };

  // AI-Coach: [[split:name]] → passenden Split-Dialog öffnen (User bestätigt)
  const coachSplitRef = useRef<(cmd: CoachCommand) => void>(() => {});
  coachSplitRef.current = (cmd: CoachCommand) => {
    if (cmd.kind !== 'splitDataset') return;
    const wanted = cmd.name?.trim().toLowerCase();
    const target = wanted
      ? datasets.find(d => d.name.toLowerCase().includes(wanted))
      : datasets.find(d => d.status !== 'split') ?? datasets[0];
    if (target) openSplitModal(target);
  };
  useEffect(() => {
    const handle = (cmd: CoachCommand) => coachSplitRef.current(cmd);
    const pendingCmd = consumePendingCoachCommand(c => c.kind === 'splitDataset');
    if (pendingCmd) handle(pendingCmd);
    return onCoachCommand(handle);
  }, []);

  const handleSplit = async () => {
    if (!datasetToSplit || !selectedModelId) return;
    if (Math.abs(trainRatio + valRatio + testRatio - 1) > 0.01) {
      warning(t('datasetUpload.splitModal.invalidRatioTitle'), t('datasetUpload.splitModal.invalidRatioDetail')); return;
    }
    setSplitting(true);
    try {
      await invoke('split_dataset', {
        datasetId: datasetToSplit.id, modelId: selectedModelId, trainRatio, valRatio, testRatio,
      });
      success(t('datasetUpload.splitModal.successTitle'), t('datasetUpload.splitModal.successDetail').replace('{name}', datasetToSplit.name));
      setShowSplitModal(false); setDatasetToSplit(null);
      await loadDatasets();
    } catch (err: unknown) {
      error(t('datasetUpload.splitModal.errorTitle'), String(err));
    } finally {
      setSplitting(false);
    }
  };

  // ── Halve ──

  const handleHalve = async () => {
    if (!datasetToHalve || !selectedModelId) return;
    setHalving(true);
    try {
      const result = await invoke<{ dataset_a: DatasetInfo; dataset_b: DatasetInfo }>(
        'split_dataset_in_half', { datasetId: datasetToHalve.id, modelId: selectedModelId, preserveSplits: inheritSplits }
      );
      success(t('datasetUpload.halveModal.successTitle'), t('datasetUpload.halveModal.successDetail').replace('{a}', result.dataset_a.name).replace('{b}', result.dataset_b.name));
      setShowHalveModal(false); setDatasetToHalve(null);
      await loadDatasets();
    } catch (err: unknown) {
      error(t('datasetUpload.halveModal.errorTitle'), String(err));
    } finally {
      setHalving(false);
    }
  };

  // ── Delete ──

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    try {
      await invoke('delete_dataset', { datasetId: deleteTarget.id, modelId: deleteTarget.model_id });
      success(t('datasetUpload.notifications.deleteSuccess'), t('datasetUpload.notifications.deleteSuccessDetail').replace('{name}', deleteTarget.name));
      await loadDatasets();
    } catch (err: unknown) {
      error(t('datasetUpload.notifications.deleteError'), String(err));
    } finally {
      setDeleteTarget(null);
    }
  };

  const closeModal = () => {
    setShowImportModal(false);
    setSelectedPath(null); setDatasetName(''); setAnalysisResult(null);
    setSelectedHfDataset(null); setHfDatasetName('');
    setHfQuery(''); setHfResults([]); setDownloadProgress(null);
    setFilterTask(''); setFilterLanguage(''); setFilterSize('');
  };

  const selectedModel = models.find(m => m.id === selectedModelId);

  if (loading) return (
    <div className="flex items-center justify-center py-24">
      <Loader2 className="w-8 h-8 text-gray-500 animate-spin" />
    </div>
  );

  if (models.length === 0) return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">{t('datasetUpload.title')}</h1><p className="text-gray-400 mt-1">{t('datasetUpload.subtitleAlt')}</p></div>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-16 text-center space-y-4">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
          <Layers className="w-8 h-8 text-gray-500" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-lg">{t('datasetUpload.emptyState.noModel.title')}</h3>
          <p className="text-gray-400 text-sm mt-1">{t('datasetUpload.emptyState.noModel.description')}</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{t('datasetUpload.title')}</h1>
          <p className="text-gray-400 mt-1">{t('datasetUpload.subtitle')}</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={loadDatasets} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all" title={t('datasetUpload.header.refreshTooltip')}>
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowImportModal(true)}
            disabled={!selectedModelId}
            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-40`}
          >
            <Upload className="w-4 h-4" /> {t('datasetUpload.header.addButton')}
          </button>
        </div>
      </div>

      {/* Model Selector */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-2">
        <label className="block text-sm font-medium text-gray-300">{t('datasetUpload.modelSelector.label')}</label>
        <div className="relative">
          <select
            value={selectedModelId ?? ''}
            onChange={e => setSelectedModelId(e.target.value)}
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm appearance-none cursor-pointer focus:outline-none focus:border-white/20 transition-all"
          >
            {models.map(m => (
              <option key={m.id} value={m.id} className="bg-slate-900">
                {m.name} ({m.source === 'huggingface' ? t('datasetUpload.modelSelector.sourceHF') : t('datasetUpload.modelSelector.sourceLocal')})
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Dataset Grid */}
      {datasets.length === 0 ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-16 text-center space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
            <Database className="w-8 h-8 text-gray-500" />
          </div>
          <div>
            <h3 className="text-white font-semibold text-lg">{t('datasetUpload.emptyState.noDatasets.title')}</h3>
            <p className="text-gray-400 text-sm mt-1">{t('datasetUpload.emptyState.noDatasets.description').replace('{model}', selectedModel?.name ?? '')}</p>
          </div>
          <button
            onClick={() => setShowImportModal(true)}
            className={`inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r ${currentTheme.colors.gradient} rounded-xl text-white text-sm font-medium hover:opacity-90 transition-all`}
          >
            <Upload className="w-4 h-4" /> {t('datasetUpload.emptyState.noDatasets.addButton')}
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map(ds => (
            <DatasetCard
              key={ds.id} dataset={ds} gradientClass={currentTheme.colors.gradient}
              onDelete={() => setDeleteTarget(ds)}
              onSplit={() => openSplitModal(ds)}
              onHalve={() => { setDatasetToHalve(ds); setInheritSplits(true); setShowHalveModal(true); }}
              onFiles={() => setFileManagerDataset(ds)}
            />
          ))}
        </div>
      )}

      {/* ── Modals ── */}

      {deleteTarget && (
        <DeleteDialog name={deleteTarget.name} onConfirm={handleDeleteConfirm} onCancel={() => setDeleteTarget(null)} />
      )}

      {fileManagerDataset && (
        <DatasetFileManager
          datasetId={fileManagerDataset.id}
          datasetName={fileManagerDataset.name}
          datasetType={fileManagerDataset.dataset_type}
          onClose={() => { setFileManagerDataset(null); loadDatasets(); }}
        />
      )}

      {/* Split Modal */}
      {showSplitModal && datasetToSplit && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">{t('datasetUpload.splitModal.title')}</h2>
                <p className="text-sm text-gray-400 mt-0.5">{datasetToSplit.name}</p>
              </div>
              <button onClick={() => { setShowSplitModal(false); setDatasetToSplit(null); }} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              {/* Erklärung was der Split macht */}
              {datasetToSplit.dataset_type && ['coco_json', 'common_voice'].includes(datasetToSplit.dataset_type) ? (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                  <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-red-300 space-y-1">
                    <p className="font-medium">{t('datasetUpload.splitModal.notSplittableTitle')}</p>
                    <p>{datasetToSplit.dataset_type === 'coco_json'
                      ? t('datasetUpload.splitModal.notSplittableCoco')
                      : t('datasetUpload.splitModal.notSplittableCommonVoice')}</p>
                  </div>
                </div>
              ) : datasetToSplit.dataset_type && ['flat_file', 'multi_shard'].includes(datasetToSplit.dataset_type) ? (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
                  <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-blue-300 space-y-1">
                    <p className="font-medium">{t('datasetUpload.splitModal.rowSplitTitle')}</p>
                    <p>{t('datasetUpload.splitModal.rowSplitDesc')}</p>
                  </div>
                </div>
              ) : datasetToSplit.dataset_type && ['yolo_bbox', 'pascal_voc', 'audio_transcript'].includes(datasetToSplit.dataset_type) ? (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-amber-300 space-y-1">
                    <p className="font-medium">{t('datasetUpload.splitModal.pairedSplitTitle')}</p>
                    <p>{t('datasetUpload.splitModal.pairedWarning').replace('{type}', DATASET_TYPE_LABELS[datasetToSplit.dataset_type]?.label ?? '')}</p>
                  </div>
                </div>
              ) : !datasetToSplit.dataset_type || datasetToSplit.dataset_type === 'unknown' ? (
                <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-amber-300 space-y-1">
                    <p className="font-medium">{t('datasetUpload.splitModal.unknownTypeTitle')}</p>
                    <p>{t('datasetUpload.splitModal.unknownTypeDesc')}</p>
                  </div>
                </div>
              ) : null}

              {/* Allgemeine Warnung: Datei-Sicherheit */}
              <div className="flex items-start gap-3 p-3 rounded-xl bg-red-500/8 border border-red-500/15">
                <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300">{t('datasetUpload.splitModal.dataIntegrityWarning')}</p>
              </div>

              <p className="text-gray-400 text-sm">{t(
                !datasetToSplit.dataset_type || datasetToSplit.dataset_type === 'unknown'
                  ? 'datasetUpload.splitModal.fileCountTotal'
                  : 'datasetUpload.splitModal.fileCount'
              ).replace('{count}', String(datasetToSplit.file_count))}</p>

              {([
                { label: t('datasetUpload.splitModal.labelTrain'), color: '#3b82f6', ratio: trainRatio, set: (v: number) => {
                  const r = 1 - v; const vp = valRatio / (valRatio + testRatio) || 0.5;
                  setTrainRatio(v); setValRatio(r * vp); setTestRatio(r * (1 - vp));
                }},
                { label: t('datasetUpload.splitModal.labelVal'), color: '#a855f7', ratio: valRatio, set: (v: number) => {
                  const r = 1 - v; const tp = trainRatio / (trainRatio + testRatio) || 0.5;
                  setValRatio(v); setTrainRatio(r * tp); setTestRatio(r * (1 - tp));
                }},
                { label: t('datasetUpload.splitModal.labelTest'), color: '#10b981', ratio: testRatio, set: (v: number) => {
                  const r = 1 - v; const tp = trainRatio / (trainRatio + valRatio) || 0.5;
                  setTestRatio(v); setTrainRatio(r * tp); setValRatio(r * (1 - tp));
                }},
              ] as const).map(({ label, color, ratio, set }) => (
                <div key={label} className="space-y-1.5">
                  <div className="flex justify-between text-sm">
                    <span style={{ color }}>{label}</span>
                    <span className="text-white">{Math.round(ratio * 100)}%</span>
                  </div>
                  <input type="range" min="0" max="100" value={Math.round(ratio * 100)}
                    onChange={e => set(parseInt(e.target.value) / 100)}
                    className="w-full" style={{ accentColor: color }} />
                </div>
              ))}

              {(() => {
                // Bei zeilenbasiertem Split (Flat File/Parquet) beziehen sich die Anteile auf
                // Zeilen, nicht auf Dateien — Datei-Zahlen wären hier irreführend (z.B. "1/0/0").
                // Bei unbekanntem Typ ist ebenfalls nicht vorhersagbar, wie viele Dateien
                // tatsächlich zugeteilt werden: file_count zählt rekursiv alles mit, gesplittet
                // werden aber nur die Datendateien im Hauptordner (vorher versprach der Dialog
                // z.B. "3005/376/376" und teilte dann 3 Dateien zu).
                const isRowSplit = !datasetToSplit.dataset_type
                  || ['flat_file', 'multi_shard', 'unknown'].includes(datasetToSplit.dataset_type);
                return (
                  <div className="grid grid-cols-3 gap-2 text-center text-sm">
                    {[
                      { label: t('datasetUpload.splitModal.labelTrain'), color: 'blue',   ratio: trainRatio },
                      { label: t('datasetUpload.splitModal.labelVal'),   color: 'purple', ratio: valRatio },
                      { label: t('datasetUpload.splitModal.labelTest'),  color: 'green',  ratio: testRatio },
                    ].map(({ label, color, ratio }) => (
                      <div key={label} className={`p-3 rounded-xl bg-${color}-500/10`}>
                        <div className={`text-${color}-400 font-bold text-lg`}>
                          {isRowSplit ? `${Math.round(ratio * 100)}%` : Math.round(datasetToSplit.file_count * ratio)}
                        </div>
                        <div className="text-gray-500 text-xs">{label}</div>
                      </div>
                    ))}
                  </div>
                );
              })()}

              <button onClick={handleSplit}
                disabled={splitting || (!!datasetToSplit.dataset_type && ['coco_json', 'common_voice'].includes(datasetToSplit.dataset_type))}
                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed`}>
                {splitting ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('datasetUpload.splitModal.splittingButton')}</> : <><Scissors className="w-4 h-4" /> {t('datasetUpload.splitModal.splitButton')}</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Halve Modal */}
      {showHalveModal && datasetToHalve && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-md">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
              <div>
                <h2 className="text-xl font-bold text-white">{t('datasetUpload.halveModal.title')}</h2>
                <p className="text-sm text-gray-400 mt-0.5">{datasetToHalve.name}</p>
              </div>
              <button onClick={() => { setShowHalveModal(false); setDatasetToHalve(null); }} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-5">
              <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10">
                <p className="text-amber-300 text-sm font-medium inline-flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  {t('datasetUpload.halveModal.whyTitle')}
                </p>
                <p className="text-gray-300 text-sm mt-1">{t('datasetUpload.halveModal.whyDescription')}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[{ label: t('datasetUpload.halveModal.halfLabel').replace('{n}', '1'), count: Math.ceil(datasetToHalve.file_count / 2) }, { label: t('datasetUpload.halveModal.halfLabel').replace('{n}', '2'), count: Math.floor(datasetToHalve.file_count / 2) }].map(({ label, count }) => (
                  <div key={label} className="p-3 rounded-xl bg-white/5 text-center">
                    <div className="text-white font-bold text-lg">{count}</div>
                    <div className="text-gray-500 text-xs mt-0.5">{t('datasetUpload.halveModal.filesLabel').replace('{label}', label)}</div>
                  </div>
                ))}
              </div>
              {datasetToHalve.status === 'split' && (
                <label className="flex items-start gap-3 p-3 rounded-xl bg-white/5 border border-white/10 cursor-pointer hover:bg-white/[0.07] transition-all">
                  <input
                    type="checkbox"
                    checked={inheritSplits}
                    onChange={e => setInheritSplits(e.target.checked)}
                    className="mt-0.5 accent-emerald-500 cursor-pointer"
                  />
                  <span>
                    <span className="text-sm text-white">{t('datasetUpload.halveModal.inheritSplitsLabel')}</span>
                    <span className="block text-xs text-gray-500 mt-0.5">{t('datasetUpload.halveModal.inheritSplitsHint')}</span>
                  </span>
                </label>
              )}
              <p className="text-gray-500 text-xs">{t('datasetUpload.halveModal.originalNote')}</p>
              <button onClick={handleHalve} disabled={halving}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50">
                {halving ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('datasetUpload.halveModal.halvingButton')}</> : <>{t('datasetUpload.halveModal.halveButton')}</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">

            <div className="px-6 py-5 border-b border-white/10 flex items-start justify-between flex-shrink-0">
              <div>
                <h2 className="text-xl font-bold text-white">{t('datasetUpload.importModal.title')}</h2>
                <p className="text-sm text-gray-400 mt-0.5">{t('datasetUpload.importModal.forModel').replace('{model}', selectedModel?.name ?? '')}</p>
              </div>
              <button onClick={closeModal} className="p-2 rounded-xl hover:bg-white/5 text-gray-400 hover:text-white transition-all">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex border-b border-white/10 flex-shrink-0">
                {([
                { mode: 'local' as ImportMode, icon: <HardDrive className="w-4 h-4" />, label: t('datasetUpload.importModal.tabLocal') },
                { mode: 'huggingface' as ImportMode, icon: <Cloud className="w-4 h-4" />, label: t('datasetUpload.importModal.tabHuggingFace') },
              ]).map(({ mode, icon, label }) => (
                <button key={mode} onClick={() => setImportMode(mode)}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-3.5 text-sm font-medium transition-all border-b-2 ${
                    importMode === mode ? 'text-white' : 'text-gray-400 hover:text-white border-transparent'
                  }`}
                  style={importMode === mode ? { borderColor: currentTheme.colors.primary, color: currentTheme.colors.primary } : {}}>
                  {icon}{label}
                </button>
              ))}
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              {importMode === 'local' ? (
                <div className="space-y-5">
                  {/* Struktur-Guide */}
              <DatasetStructureGuide />

              {/* Phase 7: Plugin-Empfehlung basierend auf ausgewähltem Modell */}
              {selectedModelId && (() => {
                const result = detectPlugin(selectedModelId);
                if (!result.supported) return null;
                const { plugin } = result;
                const preferred = plugin.preferredDatasetType;
                const supported = plugin.supportedDatasetTypes;
                if (!preferred && (!supported || supported.length === 0)) return null;
                const preferredMeta = preferred ? (DATASET_TYPE_LABELS[preferred] ?? null) : null;
                return (
                  <div className="flex items-start gap-2.5 px-3 py-2.5 rounded-xl bg-blue-500/8 border border-blue-500/20">
                    <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="text-xs text-blue-300 space-y-0.5">
                      <p className="font-medium">{plugin.name}</p>
                      {preferredMeta && (
                      <p className="flex items-center gap-1">{t('datasetUpload.importModal.local.pluginPreferredType')} <span className={`inline-flex items-center gap-1 ${preferredMeta.color}`}><DatasetTypeIcon icon={preferredMeta.icon} className="w-3.5 h-3.5" /> {preferredMeta.label}</span></p>
                      )}
                      {supported && supported.length > 1 && (
                        <p className="text-blue-400/60">{t('datasetUpload.importModal.local.pluginAlsoCompatible')} {supported
                          .filter(t => t !== preferred)
                          .map(t => DATASET_TYPE_LABELS[t]?.label ?? t)
                          .join(', ')}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })()}

              <div className="flex flex-wrap gap-2 text-xs">
                    {t('datasetUpload.importModal.local.supportedFormats').split(' ').map(f => (
                      <span key={f} className="px-2 py-1 rounded-lg bg-violet-500/10 border border-violet-500/20 text-violet-300 font-mono">{f}</span>
                    ))}
                  </div>

                  <div
                    onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
                      isDragging ? 'border-violet-500 bg-violet-500/10' :
                      selectedPath ? 'border-emerald-500/50 bg-emerald-500/5' :
                      'border-white/15 hover:border-white/30'
                    }`}
                  >
                    {selectedPath ? (
                      <div className="space-y-3">
                        <CheckCircle className="w-10 h-10 text-emerald-400 mx-auto" />
                        <div>
                          <p className="text-white font-medium">{t('datasetUpload.importModal.local.folderSelected')}</p>
                          <p className="text-gray-400 text-sm mt-0.5 break-all">{selectedPath}</p>
                        </div>
                        <button onClick={() => { setSelectedPath(null); setDatasetName(''); setAnalysisResult(null); }}
                          className="text-sm text-gray-400 hover:text-white underline transition-colors">
                          {t('datasetUpload.importModal.local.changeFolderLink')}
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white/5 border border-white/10">
                          <Upload className="w-7 h-7 text-gray-400" />
                        </div>
                        <div>
                          <p className="text-white font-medium">{isDragging ? t('datasetUpload.importModal.local.dropzoneDragging') : t('datasetUpload.importModal.local.dropzoneIdle')}</p>
                          <p className="text-gray-500 text-sm mt-1">{t('datasetUpload.importModal.local.dropzoneSubtitle')}</p>
                        </div>
                        <button onClick={handleBrowseFolder} className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/15 rounded-xl text-white text-sm transition-all">
                          <FolderOpen className="w-4 h-4" /> {t('datasetUpload.importModal.local.browseButton')}
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Analyse-Ergebnis */}
                  {analysisLoading && (
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>{t('datasetUpload.importModal.local.analysisLoading')}</span>
                    </div>
                  )}
                  {!analysisLoading && analysisResult && (
                    <AnalysisPreview analysis={analysisResult} modelId={selectedModelId} />
                  )}

                  {selectedPath && (
                    <>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-gray-300">{t('datasetUpload.importModal.local.nameLabel')}</label>
                        <input type="text" value={datasetName} onChange={e => setDatasetName(e.target.value)}
                          placeholder={t('datasetUpload.importModal.local.namePlaceholder')}
                          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all" />
                      </div>
                      <button onClick={handleLocalImport} disabled={importing || !datasetName.trim()}
                        className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r ${currentTheme.colors.gradient} text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50`}>
                        {importing ? <><Loader2 className="w-4 h-4 animate-spin" /> {t('datasetUpload.importModal.local.importingButton')}</> : <><Upload className="w-4 h-4" /> {t('datasetUpload.importModal.local.importButton')}</>}
                      </button>
                    </>
                  )}
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="space-y-3">
                    <button onClick={() => setShowFilters(!showFilters)}
                      className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-all">
                      <Filter className="w-4 h-4" />
                      {showFilters ? t('datasetUpload.importModal.hf.filterToggleHide') : t('datasetUpload.importModal.hf.filterToggleShow')}
                      <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                    </button>
                    {showFilters && filterOptions && (
                      <div className="grid grid-cols-3 gap-3">
                        {([
                          { label: t('datasetUpload.importModal.hf.filterTask'), val: filterTask, set: setFilterTask, opts: filterOptions.tasks },
                          { label: t('datasetUpload.importModal.hf.filterLanguage'), val: filterLanguage, set: setFilterLanguage, opts: filterOptions.languages.map(l => l.toUpperCase()) },
                          { label: t('datasetUpload.importModal.hf.filterSize'), val: filterSize, set: setFilterSize, opts: filterOptions.sizes },
                        ] as const).map(({ label, val, set, opts }) => (
                          <div key={label} className="space-y-1">
                            <label className="block text-xs text-gray-500">{label}</label>
                            <select value={val} onChange={e => (set as (v: string) => void)(e.target.value)}
                              className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none appearance-none">
                              <option value="" className="bg-slate-900">{t('datasetUpload.importModal.hf.filterAll')}</option>
                              {opts.map(o => <option key={o} value={o} className="bg-slate-900">{o}</option>)}
                            </select>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="space-y-1.5">
                    <label className="block text-sm font-medium text-gray-300">{t('datasetUpload.importModal.hf.searchLabel')}</label>
                    <div className="relative">
                      <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input type="text" value={hfQuery} onChange={e => setHfQuery(e.target.value)}
                        placeholder={t('datasetUpload.importModal.hf.searchPlaceholder')}
                        className="w-full pl-10 pr-10 py-3 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all" />
                      {hfSearching && <Loader2 className="absolute right-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />}
                    </div>
                    <p className="text-gray-600 text-xs">{t('datasetUpload.importModal.hf.searchHint')}</p>
                  </div>

                  {hfResults.length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-gray-500 text-xs">{t('datasetUpload.importModal.hf.resultsCount').replace('{count}', String(hfResults.length))}</p>
                      <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                        {hfResults.map(ds => (
                          <button key={ds.id} onClick={() => handleHfSelect(ds)}
                            className={`w-full flex items-center justify-between p-3 rounded-xl border text-left transition-all ${
                              selectedHfDataset?.id === ds.id ? 'bg-violet-500/10 border-violet-500/40' : 'bg-white/5 border-white/10 hover:bg-white/10'
                            }`}>
                            <div className="min-w-0">
                              <p className="text-white text-sm font-medium truncate">{ds.id}</p>
                              <div className="flex items-center gap-2 mt-0.5 text-xs text-gray-500">
                                <span>↓ {formatDownloads(ds.downloads)}</span>
                                {ds.likes ? (
                                  <span className="inline-flex items-center gap-1">
                                    <Heart className="w-3.5 h-3.5" />
                                    {formatDownloads(ds.likes)}
                                  </span>
                                ) : null}
                              </div>
                            </div>
                            {selectedHfDataset?.id === ds.id && <CheckCircle className="w-4 h-4 text-violet-400 flex-shrink-0 ml-2" />}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedHfDataset && (
                    <div className="space-y-4 p-4 rounded-2xl border border-white/10 bg-white/5">
                      <div className="flex items-center gap-3">
                        <Cloud className="w-5 h-5 text-gray-400" />
                        <div>
                          <p className="text-white font-medium text-sm">{selectedHfDataset.id}</p>
                          <a href={`https://huggingface.co/datasets/${selectedHfDataset.id}`}
                            target="_blank" rel="noopener noreferrer"
                            className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors">
                            {t('datasetUpload.importModal.hf.viewOnHF')} <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-gray-300">{t('datasetUpload.importModal.hf.localNameLabel')}</label>
                        <input type="text" value={hfDatasetName} onChange={e => setHfDatasetName(e.target.value)}
                          placeholder={t('datasetUpload.importModal.hf.localNamePlaceholder')}
                          className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white text-sm placeholder:text-gray-600 focus:outline-none focus:border-white/30 transition-all" />
                      </div>

                      {hfDownloadError && !downloading && (
                        <div className="flex items-start gap-2.5 p-3 rounded-xl bg-red-500/10 border border-red-500/30">
                          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                          <div className="min-w-0 space-y-1">
                            <p className="text-red-300 text-xs font-medium">{t('datasetUpload.importModal.hf.downloadError')}</p>
                            <p className="text-red-200/80 text-xs break-words">{hfDownloadError}</p>
                          </div>
                        </div>
                      )}

                      {downloading && downloadProgress ? (
                        <div className="space-y-3 p-3 rounded-xl bg-black/30 border border-white/5">
                          {(downloadProgress.status === 'connecting' || downloadProgress.status === 'preparing' || (downloadProgress.status === 'downloading' && downloadProgress.totalFiles === 0)) ? (
                            <div className="space-y-3">
                              <div className="flex items-center gap-3">
                                <Loader2 className="w-5 h-5 text-white animate-spin flex-shrink-0" />
                                <div className="min-w-0">
                                  <p className="text-white text-sm font-medium truncate">{downloadProgress.message || t('datasetUpload.importModal.hf.connectingLabel')}</p>
                                  <p className="text-gray-500 text-xs mt-0.5">{t('datasetUpload.importModal.hf.elapsedLabel')}: {formatTime(downloadProgress.elapsedSecs)}</p>
                                </div>
                              </div>
                              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                <div className={`h-full bg-gradient-to-r ${currentTheme.colors.gradient} rounded-full animate-pulse`} style={{ width: '60%' }} />
                              </div>
                              <p className="text-gray-600 text-xs text-center">{t('datasetUpload.importModal.hf.waitingNote')}</p>
                            </div>
                          ) : (
                            <div className="space-y-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                                  <span className="text-white text-sm font-medium">{downloadProgress.progressPercent}%</span>
                                </div>
                                <span className="text-xs text-gray-400">{formatBytes(downloadProgress.downloadedBytes)} / {formatBytes(downloadProgress.totalBytes)}</span>
                              </div>
                              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                <div className={`h-full bg-gradient-to-r ${currentTheme.colors.gradient} transition-all duration-300`} style={{ width: `${downloadProgress.progressPercent}%` }} />
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-white text-xs truncate">{downloadProgress.currentFile || downloadProgress.message}</p>
                                <p className="text-gray-500 text-xs">{downloadProgress.currentFileIndex}/{downloadProgress.totalFiles}</p>
                              </div>
                              <div className="grid grid-cols-3 gap-2">
                                <div className="text-center"><p className="text-gray-500 text-xs">{t('datasetUpload.importModal.hf.speedLabel')}</p><p className="text-white text-sm font-medium">{downloadProgress.speedMbs.toFixed(1)} MB/s</p></div>
                                <div className="text-center"><p className="text-gray-500 text-xs">{t('datasetUpload.importModal.hf.elapsedLabel')}</p><p className="text-white text-sm font-medium">{formatTime(downloadProgress.elapsedSecs)}</p></div>
                                <div className="text-center"><p className="text-gray-500 text-xs">{t('datasetUpload.importModal.hf.remainingLabel')}</p><p className="text-white text-sm font-medium">{downloadProgress.etaSecs > 0 ? formatTime(downloadProgress.etaSecs) : '—'}</p></div>
                              </div>
                            </div>
                          )}
                          <button onClick={handleCancelDownload}
                            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-white/5 hover:bg-red-500/10 border border-white/10 hover:border-red-500/30 text-gray-400 hover:text-red-400 text-sm transition-all">
                            <X className="w-4 h-4" /> {t('datasetUpload.importModal.hf.cancelDownload')}
                          </button>
                        </div>
                      ) : (
                        <>
                          <button onClick={handleHfDownload} disabled={!hfDatasetName.trim()}
                            className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-white text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                              !hfDatasetName.trim() ? 'bg-white/5' : `bg-gradient-to-r ${currentTheme.colors.gradient} hover:opacity-90`
                            }`}>
                            <Download className="w-4 h-4" /> {t('datasetUpload.importModal.hf.downloadButton')}
                          </button>
                          <p className="text-xs text-gray-600 text-center">{t('datasetUpload.importModal.hf.downloadNote')}</p>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── DatasetCard ────────────────────────────────────────────────────────────

interface DatasetCardProps {
  dataset:       DatasetInfo;
  gradientClass: string;
  onDelete:      () => void;
  onSplit:       () => void;
  onHalve:       () => void;
  onFiles:       () => void;
}

function DatasetCard({ dataset, gradientClass, onDelete, onSplit, onHalve, onFiles }: DatasetCardProps) {
  const { t, language } = useLanguage();
  const hasWarnings = (dataset.warnings?.length ?? 0) > 0;
  const typeMeta = dataset.dataset_type ? DATASET_TYPE_LABELS[dataset.dataset_type] : null;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 hover:bg-white/[0.07] transition-all group flex flex-col gap-4">
      {/* Top row */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className={`p-2 rounded-xl bg-gradient-to-r ${gradientClass} flex-shrink-0`}>
            {dataset.source === 'huggingface' ? <Cloud className="w-4 h-4 text-white" /> : <HardDrive className="w-4 h-4 text-white" />}
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-white truncate" title={dataset.name}>{dataset.name}</h3>
            <div className="flex items-center gap-1.5 mt-1 flex-wrap">
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                dataset.status === 'split' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
              }`}>
                <span className="inline-flex items-center gap-1.5">
                  {dataset.status === 'split'
                    ? <CheckCircle className="w-3.5 h-3.5" />
                    : <AlertTriangle className="w-3.5 h-3.5" />
                  }
                  <span>{dataset.status === 'split' ? t('datasetUpload.statusSplit') : t('datasetUpload.statusUnused')}</span>
                </span>
              </span>
              {dataset.training_count > 0 && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400">
                  {t('datasetUpload.card.usedCount').replace('{count}', String(dataset.training_count))}
                </span>
              )}
            </div>
          </div>
        </div>
        <button onClick={onDelete} className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Dataset-Typ Badge */}
      {typeMeta && dataset.dataset_type !== 'unknown' && (
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/5 border border-white/10">
          <DatasetTypeIcon icon={typeMeta.icon} className={`w-4 h-4 flex-shrink-0 ${typeMeta.color}`} />
          <span className={`text-xs font-medium ${typeMeta.color}`}>{typeMeta.label}</span>
          {dataset.pairing_status && (
            <span className={`ml-auto text-xs ${dataset.pairing_status.is_paired ? 'text-emerald-400/70' : 'text-amber-400/70'}`}>
              <span className="inline-flex items-center gap-1.5">
                {dataset.pairing_status.is_paired ? (
                  <CheckCircle className="w-3.5 h-3.5" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5" />
                )}
                <span>
                  {dataset.pairing_status.is_paired
                    ? t('datasetUpload.card.pairedCount').replace('{count}', String(dataset.pairing_status.paired_count))
                    : t('datasetUpload.card.orphanCount').replace('{count}', String(dataset.pairing_status.orphan_primaries.length))}
                </span>
              </span>
            </span>
          )}
        </div>
      )}

      {/* Warnungen */}
      {hasWarnings && (
        <div className="space-y-1">
          {dataset.warnings!.slice(0, 2).map((w, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-amber-400/70">
              <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
              <span className="truncate">{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Meta */}
      <div className="space-y-1.5 text-sm">
        <div className="flex items-center justify-between text-gray-400">
          <span className="flex items-center gap-1.5"><Database className="w-3.5 h-3.5" />{t('datasetUpload.card.filesLabel').replace('{count}', String(dataset.file_count))}</span>
          <span>{formatBytes(dataset.size_bytes)}</span>
        </div>
        <div className="flex items-center gap-1.5 text-gray-500 text-xs">
          <Calendar className="w-3 h-3" />{formatDate(dataset.created_at, language)}
        </div>
        {dataset.last_used_at && (
          <div className="flex items-center gap-1.5 text-gray-500 text-xs">
            <CheckCircle className="w-3 h-3 text-cyan-500/60" /> {t('datasetUpload.card.lastUsed')} {formatDate(dataset.last_used_at, language)}
          </div>
        )}
        {/* Extension chips */}
        {dataset.extensions && dataset.extensions.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-0.5">
            {dataset.extensions.slice(0, 4).map(ext => {
              const EXT_LABELS: Record<string, string> = {
                '.parquet': 'Parquet (HF)',
                '.arrow':   'Arrow',
                '.jsonl':   'JSONL',
                '.json':    'JSON',
                '.csv':     'CSV',
                '.tsv':     'TSV',
                '.txt':     'TXT',
                '.jpg':     'JPEG',
                '.jpeg':    'JPEG',
                '.png':     'PNG',
                '.wav':     'WAV',
                '.mp3':     'MP3',
                '.xml':     'XML (VOC)',
                '.yaml':    'YAML',
                '.yml':     'YAML',
              };
              const EXT_TOOLTIPS: Record<string, string> = {
                '.parquet': 'Apache Parquet — bin\u00e4res Spaltenformat, standard beim HuggingFace-Download',
                '.arrow':   'Apache Arrow — bin\u00e4res Spaltenformat',
                '.jsonl':   'JSON Lines — eine JSON-Zeile pro Beispiel',
                '.xml':     'Pascal VOC XML-Annotationen',
                '.yaml':    'YOLO dataset.yaml Konfiguration',
              };
              const label   = EXT_LABELS[ext] ?? ext;
              const tooltip = EXT_TOOLTIPS[ext];
              return (
                <span
                  key={ext}
                  className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-gray-500 text-[10px] font-mono cursor-default"
                  title={tooltip}
                >{label}</span>
              );
            })}
            {dataset.extensions.length > 4 && (
              <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-gray-600 text-[10px]">+{dataset.extensions.length - 4}</span>
            )}
          </div>
        )}
      </div>

      {/* Split preview */}
      {(() => {
        // split_info aus DB verwenden wenn vorhanden,
        // sonst bei PreSplit/HF-Downloads die echten Ordner-Counts anzeigen
        const info = dataset.split_info;
        if (!info) return null;
        return (
          <div className="grid grid-cols-3 gap-2 pt-1">
            {[
              { label: t('datasetUpload.card.splitTrain'), color: 'blue',   count: info.train_count },
              { label: t('datasetUpload.card.splitVal'),   color: 'purple', count: info.val_count },
              { label: t('datasetUpload.card.splitTest'),  color: 'green',  count: info.test_count },
            ].map(({ label, color, count }) => (
              <div key={label} className={`p-2 rounded-xl bg-${color}-500/10 text-center`}>
                <div className={`text-${color}-400 font-semibold text-sm`}>{count}</div>
                <div className="text-gray-600 text-[11px]">{label}</div>
              </div>
            ))}
          </div>
        );
      })()}

      {/* Actions */}
      <div className="flex gap-2 pt-1">
        <button onClick={onFiles} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs transition-all">
          <FileText className="w-3.5 h-3.5" /> {t('datasetUpload.card.filesButton')}
        </button>
        {dataset.status === 'unused' && (
          <button onClick={onSplit} className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white text-xs transition-all">
            <Scissors className="w-3.5 h-3.5" /> {t('datasetUpload.card.splitButton')}
          </button>
        )}
        {/* Halbieren auch für gesplittete Datasets: die train/val/test-Struktur
            wird dabei standardmäßig an beide Hälften vererbt (Checkbox im Modal) —
            kein Data-Leakage, keine Vermischung der Splits. */}
        {(dataset.status === 'unused' || dataset.status === 'split') && (
          <button onClick={onHalve}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl bg-white/5 hover:bg-amber-500/10 border border-white/10 hover:border-amber-500/20 text-gray-400 hover:text-amber-400 text-xs transition-all"
            title={t('datasetUpload.card.halveTooltip')}>
            {t('datasetUpload.card.halveButton')}
          </button>
        )}
      </div>
    </div>
  );
}
