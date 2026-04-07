"""
Plugin Dependency Manager
=========================
Manages automatic installation of plugin dependencies.

Features:
- Parse plugin manifests
- Check installed packages
- Auto-install with user consent
- Cache installed dependencies
"""

import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PluginManifest:
    """Plugin metadata and dependencies"""
    plugin_name: str
    description: str
    modality: str
    required_packages: List[str]
    optional_packages: List[str]
    min_python_version: str = "3.8"
    
    @classmethod
    def from_docstring(cls, plugin_path: Path) -> Optional['PluginManifest']:
        """Extract manifest from plugin docstring"""
        try:
            with open(plugin_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find MANIFEST section in docstring
            if 'MANIFEST:' not in content:
                return None
            
            # Extract manifest JSON
            start = content.find('MANIFEST:')
            end = content.find('"""', start)
            manifest_str = content[start+9:end].strip()
            
            # Parse as JSON
            manifest_data = json.loads(manifest_str)
            
            return cls(
                plugin_name=manifest_data.get('name', plugin_path.stem),
                description=manifest_data.get('description', ''),
                modality=manifest_data.get('modality', 'unknown'),
                required_packages=manifest_data.get('required', []),
                optional_packages=manifest_data.get('optional', []),
                min_python_version=manifest_data.get('python', '3.8')
            )
        except Exception as e:
            print(f"[DepManager] Warning: Could not parse manifest from {plugin_path}: {e}")
            return None


class DependencyManager:
    """Manages plugin dependencies"""
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or Path.home() / ".frametrain" / "dep_cache.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.installed_cache: Set[str] = self._load_cache()
    
    def _load_cache(self) -> Set[str]:
        """Load cache of installed packages"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('installed', []))
            except:
                pass
        return set()
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'installed': list(self.installed_cache)}, f)
        except Exception as e:
            print(f"[DepManager] Warning: Could not save cache: {e}")
    
    def is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed (real check, not just cache)"""
        # DON'T trust cache alone - always verify
        try:
            # Handle package name mapping (e.g., opencv-python -> cv2)
            import_name = self._get_import_name(package_name)
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                # Found - update cache
                self.installed_cache.add(package_name)
                self._save_cache()
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
        
        # Not found - remove from cache if present (stale)
        if package_name in self.installed_cache:
            print(f"[DepManager] Warning: {package_name} in cache but not actually installed")
            self.installed_cache.discard(package_name)
            self._save_cache()
        
        return False
    
    def _get_import_name(self, package_name: str) -> str:
        """Map pip package name to import name"""
        mapping = {
            'opencv-python': 'cv2',
            'opencv-python-headless': 'cv2',
            'pillow': 'PIL',
            'scikit-learn': 'sklearn',
            'scikit-image': 'skimage',
            'python-dateutil': 'dateutil',
            'pyyaml': 'yaml',
            'torch-geometric': 'torch_geometric',
        }
        return mapping.get(package_name, package_name.replace('-', '_'))
    
    def check_dependencies(self, manifest: PluginManifest) -> Tuple[List[str], List[str]]:
        """
        Check which dependencies are missing
        Returns: (missing_required, missing_optional)
        """
        missing_required = [
            pkg for pkg in manifest.required_packages 
            if not self.is_package_installed(pkg)
        ]
        
        missing_optional = [
            pkg for pkg in manifest.optional_packages 
            if not self.is_package_installed(pkg)
        ]
        
        return missing_required, missing_optional
    
    def install_package(self, package_name: str, verbose: bool = True) -> bool:
        """Install a single package using pip"""
        try:
            if verbose:
                print(f"[DepManager] Installing {package_name}...")
            
            # Use same Python interpreter as current process
            python_exe = sys.executable
            
            # Install with upgrade (but don't force-reinstall to avoid breaking existing packages)
            cmd = [python_exe, '-m', 'pip', 'install', '--upgrade', package_name]
            
            if verbose:
                print(f"[DepManager] Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large packages
            )
            
            if result.returncode == 0:
                if verbose:
                    print(f"[DepManager] ✓ Successfully installed {package_name}")
                    if result.stdout:
                        print(f"[DepManager] Output: {result.stdout[:200]}...")  # First 200 chars
                self.installed_cache.add(package_name)
                self._save_cache()
                return True
            else:
                if verbose:
                    print(f"[DepManager] ✗ Failed to install {package_name}")
                    print(f"[DepManager] STDERR: {result.stderr}")
                    print(f"[DepManager] STDOUT: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"[DepManager] ✗ Timeout installing {package_name}")
            return False
        except Exception as e:
            if verbose:
                print(f"[DepManager] ✗ Error installing {package_name}: {e}")
            return False
    
    def install_dependencies(self, manifest: PluginManifest, 
                           auto_install: bool = False,
                           install_optional: bool = False) -> bool:
        """
        Install missing dependencies for a plugin
        
        Args:
            manifest: Plugin manifest
            auto_install: If False, ask for user confirmation
            install_optional: Whether to install optional packages
            
        Returns:
            True if all required dependencies installed successfully
        """
        missing_required, missing_optional = self.check_dependencies(manifest)
        
        if not missing_required and (not install_optional or not missing_optional):
            print(f"[DepManager] ✓ All dependencies for {manifest.plugin_name} are installed")
            return True
        
        # Show what needs to be installed
        if missing_required:
            print(f"\n[DepManager] Plugin '{manifest.plugin_name}' requires:")
            for pkg in missing_required:
                print(f"  - {pkg}")
        
        if missing_optional and install_optional:
            print(f"\n[DepManager] Optional packages:")
            for pkg in missing_optional:
                print(f"  - {pkg} (optional)")
        
        # Ask for consent if not auto
        if not auto_install:
            response = input("\n[DepManager] Install these packages? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("[DepManager] Installation cancelled by user")
                return False
        
        # Install required packages
        print(f"\n[DepManager] Installing required packages...")
        all_success = True
        
        for pkg in missing_required:
            if not self.install_package(pkg):
                all_success = False
        
        # Install optional packages if requested
        if install_optional and missing_optional:
            print(f"\n[DepManager] Installing optional packages...")
            for pkg in missing_optional:
                self.install_package(pkg)  # Don't fail on optional
        
        if all_success:
            print(f"\n[DepManager] ✓ Plugin {manifest.plugin_name} ready to use!")
        else:
            print(f"\n[DepManager] ✗ Some required packages failed to install")
        
        return all_success


class PluginLoader:
    """Enhanced plugin loader with dependency management"""
    
    def __init__(self, plugin_dir: Path, dep_manager: Optional[DependencyManager] = None):
        self.plugin_dir = Path(plugin_dir)
        self.dep_manager = dep_manager or DependencyManager()
        self.loaded_plugins: Dict[str, any] = {}
    
    def load_plugin(self, plugin_file: str, auto_install: bool = False) -> Optional[any]:
        """
        Load a plugin with automatic dependency management
        
        Args:
            plugin_file: Name of plugin file (e.g., 'plugin_vision.py')
            auto_install: Automatically install dependencies without asking
            
        Returns:
            Loaded module or None if failed
        """
        plugin_path = self.plugin_dir / plugin_file
        
        if not plugin_path.exists():
            print(f"[PluginLoader] Plugin not found: {plugin_path}")
            return None
        
        # Parse manifest
        manifest = PluginManifest.from_docstring(plugin_path)
        
        if manifest is None:
            print(f"[PluginLoader] No manifest found in {plugin_file}, loading without dep check...")
        else:
            print(f"\n[PluginLoader] Loading plugin: {manifest.plugin_name}")
            print(f"[PluginLoader] Description: {manifest.description}")
            print(f"[PluginLoader] Modality: {manifest.modality}")
            
            # Check and install dependencies
            if not self.dep_manager.install_dependencies(manifest, auto_install=auto_install):
                print(f"[PluginLoader] ✗ Failed to install dependencies for {manifest.plugin_name}")
                return None
        
        # Load the plugin module
        try:
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_plugins[plugin_file] = module
            print(f"[PluginLoader] ✓ Successfully loaded {plugin_file}")
            return module
            
        except Exception as e:
            print(f"[PluginLoader] ✗ Error loading {plugin_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_all_plugins(self, auto_install: bool = False) -> Dict[str, any]:
        """Load all plugins in plugin directory"""
        plugin_files = list(self.plugin_dir.glob("plugin_*.py"))
        
        print(f"\n[PluginLoader] Found {len(plugin_files)} plugins")
        
        for plugin_file in plugin_files:
            self.load_plugin(plugin_file.name, auto_install=auto_install)
        
        return self.loaded_plugins


# ============ CLI for Testing ============

def main():
    """Test the dependency manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Plugin Dependency Manager")
    parser.add_argument("--plugin-dir", type=str, default=".", help="Plugin directory")
    parser.add_argument("--plugin", type=str, help="Specific plugin to load")
    parser.add_argument("--auto-install", action="store_true", help="Auto-install without asking")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies, don't install")
    
    args = parser.parse_args()
    
    plugin_dir = Path(args.plugin_dir)
    dep_manager = DependencyManager()
    
    if args.check_only:
        # Just check what's installed
        if args.plugin:
            plugin_path = plugin_dir / args.plugin
            manifest = PluginManifest.from_docstring(plugin_path)
            if manifest:
                missing_req, missing_opt = dep_manager.check_dependencies(manifest)
                print(f"\nPlugin: {manifest.plugin_name}")
                print(f"Missing required: {missing_req if missing_req else 'None'}")
                print(f"Missing optional: {missing_opt if missing_opt else 'None'}")
        else:
            print("Please specify --plugin for check-only mode")
    else:
        # Load plugin(s)
        loader = PluginLoader(plugin_dir, dep_manager)
        
        if args.plugin:
            loader.load_plugin(args.plugin, auto_install=args.auto_install)
        else:
            loader.load_all_plugins(auto_install=args.auto_install)


if __name__ == "__main__":
    main()
