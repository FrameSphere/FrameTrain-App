// DatasetFileManager.tsx – Datei-Browser für einzelne Datasets
// Portiert aus desktop-app2

import { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  FileText, Trash2, Upload, Search, X, Eye,
  File, Loader2, RefreshCw, Tag, ArrowRight,
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

// ── Types ──────────────────────────────────────────────────────────────────

interface FileInfo {
  name: string;
  path: string;
  size: number;
  is_dir: boolean;
  split: 'train' | 'val' | 'test' | 'unsplit';
}

interface DatasetFileManagerProps {
  datasetId: string;
  datasetName: string;
  onClose: () => void;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const SPLIT_COLORS: Record<string, string> = {
  train:   '#3b82f6',
  val:     '#a855f7',
  test:    '#10b981',
  unsplit: '#6b7280',
  info:    '#8b5cf6',
};

// ── Component ──────────────────────────────────────────────────────────────

export default function DatasetFileManager({ datasetId, datasetName, onClose }: DatasetFileManagerProps) {
  const { currentTheme } = useTheme();
  const { success, error } = useNotification();

  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [currentSplit, setCurrentSplit] = useState<'train' | 'val' | 'test' | 'all'>('all');
  const [viewingFile, setViewingFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState('');
  const [loadingContent, setLoadingContent] = useState(false);

  useEffect(() => { loadFiles(); }, [datasetId]);

  const loadFiles = async () => {
    setLoading(true);
    try {
      const result = await invoke<FileInfo[]>('get_dataset_files', { datasetId });
      setFiles(result);
    } catch (err: unknown) {
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

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
    try {
      const content = await invoke<string>('read_dataset_file', { filePath });
      setFileContent(content);
    } catch (err: unknown) {
      error('Fehler beim Lesen', String(err));
      setViewingFile(null);
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
      success('Verschoben', `${selectedFiles.size} Datei(en) → ${targetSplit}`);
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: unknown) {
      error('Fehler beim Verschieben', String(err));
    }
  };

  const deleteFiles = async () => {
    if (selectedFiles.size === 0) return;
    try {
      await invoke('delete_dataset_files', {
        datasetId, filePaths: Array.from(selectedFiles),
      });
      success('Gelöscht', `${selectedFiles.size} Datei(en) gelöscht`);
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: unknown) {
      error('Fehler beim Löschen', String(err));
    }
  };

  const addFiles = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({ multiple: true, title: 'Dateien hinzufügen' });
      if (selected) {
        const paths = Array.isArray(selected) ? selected : [selected];
        const count = await invoke<number>('add_files_to_dataset', { datasetId, filePaths: paths });
        success('Hinzugefügt', `${count} Datei(en) hinzugefügt`);
        loadFiles();
      }
    } catch (err: unknown) {
      error('Fehler', String(err));
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
            <div className="flex items-center gap-4 mt-1.5 text-xs text-gray-400">
              <span>{files.length} Dateien gesamt</span>
              <span className="text-blue-400">Train: {stats.train}</span>
              <span className="text-purple-400">Val: {stats.val}</span>
              <span className="text-green-400">Test: {stats.test}</span>
              {stats.unsplit > 0 && <span className="text-gray-500">Unsplit: {stats.unsplit}</span>}
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/10 text-gray-400 hover:text-white transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Toolbar */}
        <div className="px-4 py-3 border-b border-white/10 flex gap-3 items-center flex-wrap bg-white/[0.015] flex-shrink-0">
          {/* Search */}
          <div className="flex-1 min-w-[180px] relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Dateien durchsuchen…"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder:text-gray-600 outline-none focus:border-white/20 transition-all"
            />
          </div>

          {/* Split filter */}
          <div className="flex gap-1.5">
            {(['all', 'train', 'val', 'test'] as const).map(split => (
              <button
                key={split}
                onClick={() => setCurrentSplit(split)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                style={{
                  backgroundColor: currentSplit === split ? currentTheme.colors.primary + '33' : 'transparent',
                  color: currentSplit === split ? '#fff' : '#9ca3af',
                  border: `1px solid ${currentSplit === split ? currentTheme.colors.primary : 'rgba(255,255,255,0.08)'}`,
                }}
              >
                {split === 'all' ? 'Alle' : split.charAt(0).toUpperCase() + split.slice(1)}
                {split !== 'all' && (
                  <span className="ml-1.5 opacity-60 text-xs">{stats[split]}</span>
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
            <Upload className="w-4 h-4" /> Hinzufügen
          </button>
          <button onClick={loadFiles} disabled={loading} className="p-1.5 rounded-lg hover:bg-white/10 transition-all text-gray-500 hover:text-white">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Selection Actions */}
        {selectedFiles.size > 0 && (
          <div
            className="px-5 py-2.5 border-b flex gap-3 items-center text-sm flex-shrink-0"
            style={{ background: `linear-gradient(to right, ${currentTheme.colors.primary}18, transparent)`, borderColor: currentTheme.colors.primary + '25' }}
          >
            <span className="text-white font-medium">{selectedFiles.size} ausgewählt</span>
            <div className="flex gap-2 ml-auto">
              {(['train', 'val', 'test'] as const).map(s => (
                <button key={s} onClick={() => moveFiles(s)}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1 transition-all"
                  style={{ backgroundColor: SPLIT_COLORS[s] + '20', color: SPLIT_COLORS[s], border: `1px solid ${SPLIT_COLORS[s]}40` }}
                >
                  <ArrowRight className="w-3 h-3" /> → {s}
                </button>
              ))}
              <button
                onClick={deleteFiles}
                className="px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1 bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-all"
              >
                <Trash2 className="w-3 h-3" /> Löschen
              </button>
              <button onClick={() => setSelectedFiles(new Set())} className="px-2.5 py-1 rounded-lg text-xs text-gray-400 hover:text-white bg-white/5 border border-white/10 transition-all">
                ✕
              </button>
            </div>
          </div>
        )}

        {/* File List */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-7 h-7 animate-spin" style={{ color: currentTheme.colors.primary }} />
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-600">
              <FileText className="w-10 h-10" />
              <p className="text-sm">Keine Dateien gefunden</p>
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
                  <th className="text-left p-2.5 text-xs text-gray-500 font-medium">Name</th>
                  <th className="text-left p-2.5 text-xs text-gray-500 font-medium">Split</th>
                  <th className="text-left p-2.5 text-xs text-gray-500 font-medium">Größe</th>
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
                        style={{ backgroundColor: SPLIT_COLORS[file.split] + '33', color: SPLIT_COLORS[file.split] }}
                      >
                        {file.split}
                      </span>
                    </td>
                    <td className="p-2.5 text-xs text-gray-500 tabular-nums">{formatBytes(file.size)}</td>
                    <td className="p-2.5">
                      <button
                        onClick={() => viewFile(file.path)}
                        className="p-1.5 rounded-lg hover:bg-white/10 transition-all text-gray-600 hover:text-white opacity-0 group-hover:opacity-100"
                        title="Ansehen"
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

        {/* File Viewer Overlay – absolute relativ zur Karte (relative oben gesetzt) */}
        {viewingFile && (
          <div className="absolute inset-0 z-20 bg-black/90 backdrop-blur-sm flex items-center justify-center p-8 rounded-2xl">
            <div
              className="w-full max-w-4xl h-full rounded-2xl flex flex-col border bg-[rgb(13,20,38)]"
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
