import React, { useState, useEffect } from 'react';
import { invoke } from "@tauri-apps/api/core";
import { listen } from '@tauri-apps/api/event';
import { Check, Download, Package, Clock, HardDrive, Loader2 } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface PluginInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  built_in: boolean;
  required_packages: string[];
  estimated_size_mb: number;
  install_time_minutes: number;
  is_selected?: boolean;
  is_installed?: boolean;
}

interface InstallProgress {
  plugin_id: string;
  status: string;
  message: string;
  progress?: number;
}

const FirstLaunchSetup: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const { currentTheme } = useTheme();
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  const [selectedPlugins, setSelectedPlugins] = useState<Set<string>>(new Set(['text']));
  const [installing, setInstalling] = useState(false);
  const [installProgress, setInstallProgress] = useState<Map<string, InstallProgress>>(new Map());
  
  // Computed values
  const totalSize = Array.from(selectedPlugins)
    .map(id => plugins.find(p => p.id === id))
    .filter(p => p && !p.built_in)
    .reduce((sum, p) => sum + (p?.estimated_size_mb || 0), 0);
  
  const totalTime = Array.from(selectedPlugins)
    .map(id => plugins.find(p => p.id === id))
    .filter(p => p && !p.built_in)
    .reduce((sum, p) => sum + (p?.install_time_minutes || 0), 0);
  
  useEffect(() => {
    loadPlugins();
    setupListeners();
  }, []);
  
  const loadPlugins = async () => {
    try {
      const pluginList = await invoke<PluginInfo[]>('get_available_plugins');
      setPlugins(pluginList);
      
      // Pre-select plugins
      const preSelected = new Set(['text']);
      pluginList.forEach(p => {
        if (p.is_selected || p.built_in) {
          preSelected.add(p.id);
        }
      });
      setSelectedPlugins(preSelected);
    } catch (error) {
      console.error('Failed to load plugins:', error);
    }
  };
  
  const setupListeners = async () => {
    await listen<InstallProgress>('plugin-install-progress', (event) => {
      const progress = event.payload;
      setInstallProgress(prev => new Map(prev).set(progress.plugin_id, progress));
    });
    
    await listen('plugin-install-complete', () => {
      setInstalling(false);
      onComplete();
    });
  };
  
  const togglePlugin = (pluginId: string) => {
    if (pluginId === 'text') return;
    
    setSelectedPlugins(prev => {
      const next = new Set(prev);
      if (next.has(pluginId)) {
        next.delete(pluginId);
      } else {
        next.add(pluginId);
      }
      return next;
    });
  };
  
  const startInstallation = async () => {
    setInstalling(true);
    setInstallProgress(new Map());
    
    try {
      await invoke('install_plugins', {
        pluginIds: Array.from(selectedPlugins)
      });
    } catch (error) {
      console.error('Installation failed:', error);
      setInstalling(false);
    }
  };
  
  const skipSetup = async () => {
    try {
      await invoke('install_plugins', { pluginIds: ['text'] });
      onComplete();
    } catch (error) {
      console.error('Skip setup failed:', error);
    }
  };
  
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'complete': return 'text-green-400';
      case 'failed': return 'text-red-400';
      default: return 'text-blue-400';
    }
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      text: 'from-purple-500 to-pink-500',
      vision: 'from-blue-500 to-cyan-500',
      audio: 'from-green-500 to-teal-500',
      tabular: 'from-orange-500 to-red-500',
      graph: 'from-indigo-500 to-purple-500',
      multimodal: 'from-pink-500 to-rose-500',
      rl: 'from-yellow-500 to-orange-500'
    };
    return colors[category] || 'from-gray-500 to-gray-600';
  };
  
  return (
    <div className={`h-screen flex items-center justify-center bg-gradient-to-br ${currentTheme.colors.background} p-6`}>
      <div className="w-full max-w-5xl h-[90vh] bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-2xl flex flex-col overflow-hidden">
        
        {/* Header */}
        <div className="flex-shrink-0 p-8 border-b border-white/10">
          <div className="flex items-center gap-4 mb-4">
            <div className={`p-3 bg-gradient-to-br ${currentTheme.colors.gradient} rounded-xl`}>
              <Package className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Welcome to FrameTrain</h1>
              <p className="text-gray-300">Choose which AI capabilities to install</p>
            </div>
          </div>
        </div>
        
        {!installing ? (
          <>
            {/* Plugin List - Scrollable */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="grid gap-3">
                {plugins.map(plugin => {
                  const isSelected = selectedPlugins.has(plugin.id);
                  const isDisabled = plugin.built_in;
                  
                  return (
                    <div
                      key={plugin.id}
                      onClick={() => !isDisabled && togglePlugin(plugin.id)}
                      className={`
                        group relative p-4 rounded-xl border transition-all cursor-pointer
                        ${isSelected 
                          ? 'bg-white/20 border-white/40 shadow-lg' 
                          : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                        }
                        ${isDisabled ? 'opacity-60 cursor-default' : ''}
                      `}
                    >
                      <div className="flex items-start gap-4">
                        {/* Checkbox */}
                        <div className="flex-shrink-0 mt-1">
                          <div className={`
                            w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all
                            ${isSelected 
                              ? 'bg-gradient-to-br from-blue-500 to-purple-500 border-transparent' 
                              : 'border-white/30 group-hover:border-white/50'
                            }
                          `}>
                            {isSelected && <Check className="w-4 h-4 text-white" strokeWidth={3} />}
                          </div>
                        </div>
                        
                        {/* Category Badge */}
                        <div className="flex-shrink-0">
                          <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getCategoryColor(plugin.category)} flex items-center justify-center`}>
                            <span className="text-2xl">{plugin.icon}</span>
                          </div>
                        </div>
                        
                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="text-lg font-semibold text-white">{plugin.name}</h3>
                            {plugin.built_in && (
                              <span className="px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded-full border border-green-400/30">
                                Built-in
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-gray-300 mb-2">{plugin.description}</p>
                          
                          {!plugin.built_in && (
                            <div className="flex items-center gap-3 text-xs text-gray-400">
                              <span className="flex items-center gap-1">
                                <HardDrive className="w-3 h-3" />
                                {plugin.estimated_size_mb >= 1024 
                                  ? `${(plugin.estimated_size_mb / 1024).toFixed(1)} GB`
                                  : `${plugin.estimated_size_mb} MB`
                                }
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                ~{plugin.install_time_minutes} min
                              </span>
                              <span className="flex items-center gap-1">
                                <Package className="w-3 h-3" />
                                {plugin.required_packages.length} packages
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Footer with Summary and Actions */}
            <div className="flex-shrink-0 p-6 border-t border-white/10 bg-black/20">
              <div className="flex items-center justify-between mb-4">
                <div className="flex gap-8">
                  <div>
                    <div className="text-2xl font-bold text-white">{selectedPlugins.size}</div>
                    <div className="text-xs text-gray-400">Plugins Selected</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">
                      {totalSize >= 1024 
                        ? `${(totalSize / 1024).toFixed(1)} GB`
                        : `${totalSize} MB`
                      }
                    </div>
                    <div className="text-xs text-gray-400">Download Size</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-white">~{totalTime} min</div>
                    <div className="text-xs text-gray-400">Install Time</div>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <button
                    onClick={skipSetup}
                    className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all border border-white/20"
                  >
                    Skip for Now
                  </button>
                  <button
                    onClick={startInstallation}
                    disabled={selectedPlugins.size === 0}
                    className={`
                      px-6 py-3 rounded-xl font-semibold transition-all flex items-center gap-2
                      ${selectedPlugins.size === 0 
                        ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                        : `bg-gradient-to-r ${currentTheme.colors.gradient} text-white hover:shadow-lg hover:shadow-blue-500/50`
                      }
                    `}
                  >
                    <Download className="w-5 h-5" />
                    Install Selected ({selectedPlugins.size})
                  </button>
                </div>
              </div>
              
              <p className="text-xs text-gray-400 text-center">
                You can install additional plugins later in Settings
              </p>
            </div>
          </>
        ) : (
          /* Installation Progress */
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            <div className="w-full max-w-2xl">
              <div className="text-center mb-8">
                <Loader2 className="w-16 h-16 text-blue-400 animate-spin mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-2">Installing Plugins</h2>
                <p className="text-gray-400">This may take a few minutes...</p>
              </div>
              
              <div className="space-y-3">
                {Array.from(selectedPlugins).map(pluginId => {
                  const plugin = plugins.find(p => p.id === pluginId);
                  const progress = installProgress.get(pluginId);
                  
                  if (!plugin || plugin.built_in) return null;
                  
                  return (
                    <div key={pluginId} className="bg-white/5 rounded-xl p-4 border border-white/10">
                      <div className="flex items-center gap-3 mb-2">
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${getCategoryColor(plugin.category)} flex items-center justify-center flex-shrink-0`}>
                          <span className="text-xl">{plugin.icon}</span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-white">{plugin.name}</span>
                            {progress && (
                              <span className={`text-sm ${getStatusColor(progress.status)}`}>
                                {progress.status === 'complete' && <Check className="w-4 h-4 inline" />}
                                {progress.status === 'failed' && '✗ Failed'}
                                {!['complete', 'failed'].includes(progress.status) && 'Installing...'}
                              </span>
                            )}
                          </div>
                          {progress?.message && (
                            <p className="text-xs text-gray-400 mt-1">{progress.message}</p>
                          )}
                        </div>
                      </div>
                      
                      {progress && progress.progress !== undefined && (
                        <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-full transition-all duration-300"
                            style={{ width: `${progress.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              
              <p className="text-sm text-gray-500 text-center mt-8">
                Please don't close the application during installation
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FirstLaunchSetup;
