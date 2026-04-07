"""
FrameTrain Plugin Manager
=========================
Central plugin management system that handles:
- Plugin discovery and registration
- Dependency installation
- Just-in-time plugin loading
- Communication with Rust backend

This module is used by both train_engine and test_engine.
"""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class PluginInfo:
    """Information about a plugin"""
    id: str
    name: str
    description: str
    category: str
    icon: str
    built_in: bool
    train_plugin: Optional[str]
    test_plugin: Optional[str]
    required_packages: List[str]
    optional_packages: List[str]
    estimated_size_mb: int
    install_time_minutes: int
    github_path: Optional[str]
    priority: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginInfo':
        """Create from dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PluginRegistry:
    """Registry of all available plugins"""
    
    def __init__(self, registry_file: Optional[Path] = None):
        if registry_file is None:
            registry_file = Path(__file__).parent / "plugin_registry.json"
        
        self.registry_file = registry_file
        self.plugins: Dict[str, PluginInfo] = {}
        self.categories: Dict[str, Dict[str, str]] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load plugin registry from JSON"""
        if not self.registry_file.exists():
            print(f"[PluginRegistry] Warning: Registry file not found: {self.registry_file}")
            return
        
        with open(self.registry_file, 'r') as f:
            data = json.load(f)
        
        # Load plugins
        for plugin_data in data.get('plugins', []):
            plugin = PluginInfo.from_dict(plugin_data)
            self.plugins[plugin.id] = plugin
        
        # Load categories
        self.categories = data.get('categories', {})
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin info by ID"""
        return self.plugins.get(plugin_id)
    
    def get_all_plugins(self) -> List[PluginInfo]:
        """Get all plugins sorted by priority"""
        return sorted(self.plugins.values(), key=lambda p: p.priority)
    
    def get_plugins_by_category(self, category: str) -> List[PluginInfo]:
        """Get all plugins in a category"""
        return [p for p in self.plugins.values() if p.category == category]
    
    def get_installable_plugins(self) -> List[PluginInfo]:
        """Get plugins that need installation (not built-in)"""
        return [p for p in self.plugins.values() if not p.built_in]


class PluginState:
    """Track plugin installation state"""
    
    def __init__(self, state_file: Optional[Path] = None):
        if state_file is None:
            state_file = Path.home() / ".frametrain" / "plugin_state.json"
        
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.installed_plugins: Dict[str, Dict[str, Any]] = {}
        self.dependency_cache: Dict[str, bool] = {}
        
        self._load_state()
    
    def _load_state(self):
        """Load state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.installed_plugins = data.get('installed_plugins', {})
                    self.dependency_cache = data.get('dependency_cache', {})
            except Exception as e:
                print(f"[PluginState] Warning: Could not load state: {e}")
    
    def _save_state(self):
        """Save state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'installed_plugins': self.installed_plugins,
                    'dependency_cache': self.dependency_cache
                }, f, indent=2)
        except Exception as e:
            print(f"[PluginState] Warning: Could not save state: {e}")
    
    def is_plugin_installed(self, plugin_id: str) -> bool:
        """Check if plugin is installed"""
        return plugin_id in self.installed_plugins
    
    def mark_plugin_installed(self, plugin_id: str, metadata: Dict[str, Any] = None):
        """Mark plugin as installed"""
        self.installed_plugins[plugin_id] = {
            'installed_at': time.time(),
            'metadata': metadata or {}
        }
        self._save_state()
    
    def get_installed_plugins(self) -> List[str]:
        """Get list of installed plugin IDs"""
        return list(self.installed_plugins.keys())
    
    def is_dependency_installed(self, package: str) -> bool:
        """Check if dependency is installed (cached)"""
        return self.dependency_cache.get(package, False)
    
    def mark_dependency_installed(self, package: str):
        """Mark dependency as installed"""
        self.dependency_cache[package] = True
        self._save_state()


class PluginManager:
    """
    Main plugin manager that coordinates:
    - Plugin discovery
    - Dependency checking and installation
    - Just-in-time loading
    """
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.state = PluginState()
        self.settings_file = Path.home() / ".frametrain" / "settings.json"
        self._ensure_settings()
    
    def _ensure_settings(self):
        """Ensure settings file exists"""
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.settings_file.exists():
            default_settings = {
                'auto_install_dependencies': False,
                'first_launch_completed': False,
                'selected_plugins': []
            }
            with open(self.settings_file, 'w') as f:
                json.dump(default_settings, f, indent=2)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        with open(self.settings_file, 'r') as f:
            return json.load(f)
    
    def update_settings(self, updates: Dict[str, Any]):
        """Update settings"""
        settings = self.get_settings()
        settings.update(updates)
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
    
    def is_first_launch(self) -> bool:
        """Check if this is the first launch"""
        settings = self.get_settings()
        return not settings.get('first_launch_completed', False)
    
    def mark_first_launch_complete(self):
        """Mark first launch as completed"""
        self.update_settings({'first_launch_completed': True})
    
    def get_plugins_for_first_launch(self) -> List[Dict[str, Any]]:
        """Get plugin info for first launch selection"""
        plugins = self.registry.get_installable_plugins()
        
        # Add text as pre-selected (built-in)
        result = []
        
        # Add built-in text plugin
        text_plugin = self.registry.get_plugin('text')
        if text_plugin:
            result.append({
                **text_plugin.to_dict(),
                'is_selected': True,
                'is_installed': True
            })
        
        # Add installable plugins
        for plugin in plugins:
            result.append({
                **plugin.to_dict(),
                'is_selected': plugin.id in ['vision'],  # Pre-select popular ones
                'is_installed': self.state.is_plugin_installed(plugin.id)
            })
        
        return result
    
    def install_selected_plugins(self, plugin_ids: List[str], 
                                progress_callback=None) -> Dict[str, bool]:
        """
        Install selected plugins and their dependencies
        
        Args:
            plugin_ids: List of plugin IDs to install
            progress_callback: Function to call with progress updates
                              callback(plugin_id, status, message)
        
        Returns:
            Dictionary mapping plugin_id to success status
        """
        results = {}
        
        for plugin_id in plugin_ids:
            plugin = self.registry.get_plugin(plugin_id)
            
            if not plugin:
                results[plugin_id] = False
                continue
            
            if plugin.built_in:
                # Built-in plugins don't need installation
                results[plugin_id] = True
                self.state.mark_plugin_installed(plugin_id)
                continue
            
            if progress_callback:
                progress_callback(plugin_id, 'installing_dependencies', 
                                f'Installing dependencies for {plugin.name}...')
            
            # Install dependencies using existing dependency manager
            success = self._install_plugin_dependencies(plugin, progress_callback)
            
            if success:
                self.state.mark_plugin_installed(plugin_id, {
                    'name': plugin.name,
                    'category': plugin.category
                })
                
                if progress_callback:
                    progress_callback(plugin_id, 'complete', 
                                    f'{plugin.name} installed successfully')
            else:
                if progress_callback:
                    progress_callback(plugin_id, 'failed', 
                                    f'Failed to install {plugin.name}')
            
            results[plugin_id] = success
        
        # Save selected plugins to settings
        self.update_settings({'selected_plugins': plugin_ids})
        
        return results
    
    def _install_plugin_dependencies(self, plugin: PluginInfo, 
                                    progress_callback=None) -> bool:
        """Install dependencies for a plugin"""
        # Import dependency manager from train_engine/plugins
        dep_manager_path = Path(__file__).parent / "train_engine" / "plugins" / "plugin_dependency_manager.py"
        
        if not dep_manager_path.exists():
            print(f"[PluginManager] Dependency manager not found at: {dep_manager_path}")
            return False
        
        try:
            # Import DependencyManager
            spec = importlib.util.spec_from_file_location("plugin_dependency_manager", dep_manager_path)
            dep_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dep_module)
            
            DependencyManager = dep_module.DependencyManager
            dep_manager = DependencyManager()
            
            # Check what needs to be installed
            missing_packages = []
            for pkg in plugin.required_packages:
                if not dep_manager.is_package_installed(pkg):
                    missing_packages.append(pkg)
            
            if not missing_packages:
                print(f"[PluginManager] All dependencies for {plugin.name} already installed")
                return True
            
            print(f"[PluginManager] Installing packages for {plugin.name}: {missing_packages}")
            
            # Install each package
            for i, pkg in enumerate(missing_packages):
                if progress_callback:
                    progress = int((i / len(missing_packages)) * 100)
                    progress_callback(plugin.id, 'installing_package', 
                                    f'Installing {pkg}... ({progress}%)')
                
                success = dep_manager.install_package(pkg, verbose=True)
                if success:
                    self.state.mark_dependency_installed(pkg)
                else:
                    print(f"[PluginManager] Failed to install {pkg}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"[PluginManager] Error installing dependencies: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ensure_plugin_available(self, plugin_id: str, modality: str = None) -> bool:
        """
        Ensure a plugin is available for use.
        If not installed, trigger installation process.
        
        This is called by train_engine/test_engine when they detect
        a needed plugin is missing.
        
        Args:
            plugin_id: ID of the plugin needed
            modality: Optional modality hint (vision, audio, etc.)
        
        Returns:
            True if plugin is available, False otherwise
        """
        # Check if already installed
        if self.state.is_plugin_installed(plugin_id):
            return True
        
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            print(f"[PluginManager] Unknown plugin: {plugin_id}")
            return False
        
        print(f"\n[PluginManager] Plugin '{plugin.name}' is required but not installed")
        
        # Check settings for auto-install preference
        settings = self.get_settings()
        auto_install = settings.get('auto_install_dependencies', False)
        
        if not auto_install:
            # Send message to Rust backend requesting user approval
            self._request_plugin_install_approval(plugin)
            
            # Wait for approval file
            approved = self._wait_for_install_approval(plugin_id, timeout=120)
            
            if not approved:
                print(f"[PluginManager] User declined to install {plugin.name}")
                return False
        
        # Install the plugin
        print(f"[PluginManager] Installing {plugin.name}...")
        
        result = self.install_selected_plugins([plugin_id])
        
        return result.get(plugin_id, False)
    
    def _request_plugin_install_approval(self, plugin: PluginInfo):
        """Request user approval for plugin installation via Rust backend"""
        # Send JSON message that Rust will intercept
        message = {
            'type': 'plugin_install_request',
            'plugin': plugin.to_dict(),
            'timestamp': time.time()
        }
        print(f"__PLUGIN_INSTALL_REQUEST__{json.dumps(message)}__END__", flush=True)
    
    def _wait_for_install_approval(self, plugin_id: str, timeout: int) -> bool:
        """Wait for user approval of plugin installation"""
        approval_file = Path.home() / ".frametrain" / f"approval_{plugin_id}.json"
        
        start = time.time()
        while time.time() - start < timeout:
            if approval_file.exists():
                try:
                    with open(approval_file, 'r') as f:
                        response = json.load(f)
                    approval_file.unlink()  # Clean up
                    
                    # Check if user also enabled auto-install for future
                    if response.get('remember', False):
                        self.update_settings({'auto_install_dependencies': True})
                    
                    return response.get('approved', False)
                except Exception as e:
                    print(f"[PluginManager] Error reading approval: {e}")
                    return False
            
            time.sleep(0.5)
        
        print(f"[PluginManager] Timeout waiting for approval")
        return False


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    """Test the plugin manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FrameTrain Plugin Manager")
    parser.add_argument('--list', action='store_true', help='List all plugins')
    parser.add_argument('--install', nargs='+', help='Install plugins by ID')
    parser.add_argument('--check', nargs='+', help='Check if plugins are installed')
    parser.add_argument('--first-launch', action='store_true', help='Show first launch info')
    
    args = parser.parse_args()
    
    manager = PluginManager()
    
    if args.list:
        print("\n=== Available Plugins ===\n")
        for plugin in manager.registry.get_all_plugins():
            status = "✓ Installed" if manager.state.is_plugin_installed(plugin.id) else "○ Not installed"
            print(f"{plugin.icon} {plugin.name} ({plugin.id})")
            print(f"   {plugin.description}")
            print(f"   Category: {plugin.category} | Size: ~{plugin.estimated_size_mb} MB | {status}")
            print()
    
    elif args.first_launch:
        # Output JSON for Rust to parse
        plugins = manager.get_plugins_for_first_launch()
        print(json.dumps(plugins))
    
    elif args.check:
        print("\n=== Plugin Status ===\n")
        for plugin_id in args.check:
            installed = manager.state.is_plugin_installed(plugin_id)
            plugin = manager.registry.get_plugin(plugin_id)
            if plugin:
                status = "✓ Installed" if installed else "○ Not installed"
                print(f"{plugin.name}: {status}")
            else:
                print(f"{plugin_id}: Unknown plugin")
    
    elif args.install:
        print(f"\n=== Installing Plugins: {', '.join(args.install)} ===\n")
        
        def progress(plugin_id, status, message):
            print(f"[{plugin_id}] {status}: {message}")
        
        results = manager.install_selected_plugins(args.install, progress_callback=progress)
        
        print("\n=== Installation Results ===")
        for plugin_id, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{plugin_id}: {status}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
