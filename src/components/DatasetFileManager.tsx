import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { 
  FileText, 
  Folder, 
  FolderOpen,
  ArrowLeft,
  ArrowRight,
  Trash2,
  Download,
  Upload,
  Search,
  X,
  Eye,
  ChevronRight,
  File,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useNotification } from '../contexts/NotificationContext';

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

export default function DatasetFileManager({ datasetId, datasetName, onClose }: DatasetFileManagerProps) {
  const { currentTheme } = useTheme();
  const { success, error } = useNotification();
  
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [currentSplit, setCurrentSplit] = useState<'train' | 'val' | 'test' | 'all'>('all');
  const [viewingFile, setViewingFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [loadingContent, setLoadingContent] = useState(false);

  useEffect(() => {
    loadFiles();
  }, [datasetId]);

  const loadFiles = async () => {
    setLoading(true);
    try {
      const result = await invoke<FileInfo[]>('get_dataset_files', { datasetId });
      setFiles(result);
    } catch (err: any) {
      error('Fehler beim Laden', String(err));
    } finally {
      setLoading(false);
    }
  };

  const viewFile = async (filePath: string) => {
    setLoadingContent(true);
    setViewingFile(filePath);
    try {
      const content = await invoke<string>('read_dataset_file', { filePath });
      setFileContent(content);
    } catch (err: any) {
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
        datasetId,
        filePaths: Array.from(selectedFiles),
        targetSplit
      });
      success('Verschoben', `${selectedFiles.size} Datei(en) nach ${targetSplit} verschoben`);
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: any) {
      error('Fehler beim Verschieben', String(err));
    }
  };

  const deleteFiles = async () => {
    if (selectedFiles.size === 0) return;
    if (!confirm(`${selectedFiles.size} Datei(en) wirklich löschen?`)) return;
    
    try {
      await invoke('delete_dataset_files', {
        datasetId,
        filePaths: Array.from(selectedFiles)
      });
      success('Gelöscht', `${selectedFiles.size} Datei(en) gelöscht`);
      setSelectedFiles(new Set());
      loadFiles();
    } catch (err: any) {
      error('Fehler beim Löschen', String(err));
    }
  };

  const addFiles = async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        multiple: true,
        title: 'Dateien zum Dataset hinzufügen'
      });
      
      if (selected && Array.isArray(selected)) {
        const result = await invoke<number>('add_files_to_dataset', { 
          datasetId, 
          filePaths: selected 
        });
        success('Dateien hinzugefügt', `${result} Datei(en) wurden erfolgreich hinzugefügt`);
        loadFiles();
      } else if (selected && typeof selected === 'string') {
        const result = await invoke<number>('add_files_to_dataset', { 
          datasetId, 
          filePaths: [selected] 
        });
        success('Datei hinzugefügt', 'Datei wurde erfolgreich hinzugefügt');
        loadFiles();
      }
    } catch (err: any) {
      error('Fehler beim Hinzufügen', String(err));
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const filteredFiles = files.filter(file => {
    const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSplit = currentSplit === 'all' || file.split === currentSplit;
    return matchesSearch && matchesSplit;
  });

  const toggleSelectFile = (path: string) => {
    const newSelected = new Set(selectedFiles);
    if (newSelected.has(path)) {
      newSelected.delete(path);
    } else {
      newSelected.add(path);
    }
    setSelectedFiles(newSelected);
  };

  const getSplitStats = () => {
    const stats = { train: 0, val: 0, test: 0, unsplit: 0 };
    files.forEach(file => {
      stats[file.split]++;
    });
    return stats;
  };

  const stats = getSplitStats();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div 
        className="w-full max-w-6xl h-[90vh] rounded-2xl shadow-2xl flex flex-col border"
        style={{
          backgroundColor: 'rgb(15, 23, 42)',
          borderColor: 'rgba(255, 255, 255, 0.1)'
        }}
      >
        {/* Header */}
        <div 
          className="p-6 border-b flex justify-between items-center"
          style={{
            borderColor: 'rgba(255, 255, 255, 0.1)',
            background: `linear-gradient(to right, ${currentTheme.colors.primary}10, transparent)`
          }}
        >
          <div>
            <h2 className="text-2xl font-bold text-white">{datasetName}</h2>
            <div className="flex items-center gap-3 mt-2 text-sm">
              <span className="text-gray-400">{files.length} Dateien total</span>
              <span className="text-blue-400">Train: {stats.train}</span>
              <span className="text-purple-400">Val: {stats.val}</span>
              <span className="text-green-400">Test: {stats.test}</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
          >
            <X size={24} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="p-4 border-b border-white/10 flex gap-4 items-center flex-wrap bg-white/[0.02]">
          {/* Search */}
          <div className="flex-1 min-w-[200px] relative">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Dateien durchsuchen..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 outline-none focus:ring-2 transition-all"
              style={{ '--tw-ring-color': currentTheme.colors.primary } as React.CSSProperties}
            />
          </div>

          {/* Filter */}
          <div className="flex gap-2">
            {(['all', 'train', 'val', 'test'] as const).map(split => (
              <button
                key={split}
                onClick={() => setCurrentSplit(split)}
                className="px-4 py-2 rounded-lg font-medium transition-all"
                style={{
                  backgroundColor: currentSplit === split ? currentTheme.colors.primary : 'transparent',
                  color: currentSplit === split ? '#fff' : '#9ca3af',
                  border: `1px solid ${currentSplit === split ? currentTheme.colors.primary : 'rgba(255,255,255,0.1)'}`,
                  boxShadow: currentSplit === split ? `0 0 20px ${currentTheme.colors.primary}40` : 'none'
                }}
              >
                {split === 'all' ? 'Alle' : split.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Actions */}
          <button
            onClick={addFiles}
            className="px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 hover:opacity-90"
            style={{
              background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`,
              color: '#fff',
              boxShadow: `0 4px 20px ${currentTheme.colors.primary}40`
            }}
          >
            <Upload size={18} />
            Hinzufügen
          </button>

          <button
            onClick={loadFiles}
            className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
            disabled={loading}
          >
            <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* Selected Actions */}
        {selectedFiles.size > 0 && (
          <div 
            className="p-4 border-b border-white/10 flex gap-4 items-center"
            style={{ 
              background: `linear-gradient(to right, ${currentTheme.colors.primary}20, ${currentTheme.colors.primary}10)`,
              borderColor: currentTheme.colors.primary + '30'
            }}
          >
            <span className="font-medium text-white">
              {selectedFiles.size} Datei(en) ausgewählt
            </span>
            <div className="flex gap-2 ml-auto">
              <button
                onClick={() => moveFiles('train')}
                className="px-3 py-1.5 rounded-lg bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 transition flex items-center gap-2 text-blue-400 hover:text-blue-300"
              >
                <ArrowRight size={16} />
                → Train
              </button>
              <button
                onClick={() => moveFiles('val')}
                className="px-3 py-1.5 rounded-lg bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition flex items-center gap-2 text-purple-400 hover:text-purple-300"
              >
                <ArrowRight size={16} />
                → Val
              </button>
              <button
                onClick={() => moveFiles('test')}
                className="px-3 py-1.5 rounded-lg bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 transition flex items-center gap-2 text-green-400 hover:text-green-300"
              >
                <ArrowRight size={16} />
                → Test
              </button>
              <button
                onClick={deleteFiles}
                className="px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 transition flex items-center gap-2 text-red-400 hover:text-red-300"
              >
                <Trash2 size={16} />
                Löschen
              </button>
              <button
                onClick={() => setSelectedFiles(new Set())}
                className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition text-gray-400 hover:text-white"
              >
                Abbrechen
              </button>
            </div>
          </div>
        )}

        {/* File List */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 size={32} className="animate-spin" style={{ color: currentTheme.colors.primary }} />
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full">
              <FileText size={48} className="text-gray-600" />
              <p className="mt-4 text-gray-400">Keine Dateien gefunden</p>
            </div>
          ) : (
            <div className="p-4">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left p-2 w-8"></th>
                    <th className="text-left p-2 text-gray-400 font-medium">Name</th>
                    <th className="text-left p-2 text-gray-400 font-medium">Split</th>
                    <th className="text-left p-2 text-gray-400 font-medium">Größe</th>
                    <th className="text-left p-2 text-gray-400 font-medium">Aktionen</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredFiles.map((file) => (
                    <tr 
                      key={file.path}
                      className="border-b border-white/10 hover:bg-white/5 transition"
                    >
                      <td className="p-2">
                        <input
                          type="checkbox"
                          checked={selectedFiles.has(file.path)}
                          onChange={() => toggleSelectFile(file.path)}
                          className="w-4 h-4 cursor-pointer"
                        />
                      </td>
                      <td className="p-2 flex items-center gap-2">
                        {file.is_dir ? <Folder size={16} className="text-gray-400" /> : <File size={16} className="text-gray-400" />}
                        <span className="truncate text-white">{file.name}</span>
                      </td>
                      <td className="p-2">
                        <span 
                          className="px-2 py-1 rounded text-xs font-medium"
                          style={{
                            backgroundColor: 
                              file.split === 'train' ? '#10b981' :
                              file.split === 'val' ? '#f59e0b' :
                              file.split === 'test' ? '#3b82f6' : '#6b7280',
                            color: '#fff'
                          }}
                        >
                          {file.split}
                        </span>
                      </td>
                      <td className="p-2 text-sm text-gray-400">
                        {formatBytes(file.size)}
                      </td>
                      <td className="p-2">
                        <button
                          onClick={() => viewFile(file.path)}
                          className="p-1.5 rounded hover:bg-white/10 transition text-gray-400 hover:text-white"
                          title="Ansehen"
                        >
                          <Eye size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* File Viewer Modal */}
        {viewingFile && (
          <div className="absolute inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center p-8">
            <div 
              className="w-full max-w-4xl h-full rounded-xl flex flex-col border"
              style={{
                backgroundColor: 'rgb(15, 23, 42)',
                borderColor: currentTheme.colors.primary + '40'
              }}
            >
              <div 
                className="p-4 border-b flex justify-between items-center"
                style={{
                  borderColor: 'rgba(255, 255, 255, 0.1)',
                  background: `linear-gradient(to right, ${currentTheme.colors.primary}10, transparent)`
                }}
              >
                <h3 className="font-medium truncate flex-1 text-white">
                  {viewingFile.split('/').pop()}
                </h3>
                <button
                  onClick={() => setViewingFile(null)}
                  className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
                >
                  <X size={20} />
                </button>
              </div>
              <div className="flex-1 overflow-auto p-4 bg-black/20">
                {loadingContent ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2 size={32} className="animate-spin" style={{ color: currentTheme.colors.primary }} />
                  </div>
                ) : (
                  <pre className="text-sm font-mono whitespace-pre-wrap break-words text-gray-300">
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
