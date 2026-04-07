import { useEffect, useState } from 'react';
import { getVersion } from '@tauri-apps/api/app';

interface UpdateInfo {
  available: boolean;
  latestVersion: string;
  currentVersion: string;
  releaseUrl: string;
  releaseName: string;
}

export function UpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const [showDialog, setShowDialog] = useState(false);

  useEffect(() => {
    checkForUpdates();
  }, []);

  async function checkForUpdates() {
    try {
      setChecking(true);
      
      // Get current version from Tauri
      const currentVersion = await getVersion();
      
      // Fetch latest release from GitHub
      const response = await fetch(
        'https://api.github.com/repos/KarolP-tech/FrameTrain/releases/latest',
        {
          headers: {
            'Accept': 'application/vnd.github.v3+json'
          }
        }
      );
      
      if (!response.ok) {
        console.error('Failed to check for updates:', response.status);
        return;
      }
      
      const release = await response.json();
      const latestVersion = release.tag_name.replace('v', ''); // Remove 'v' prefix
      
      // Simple version comparison
      const isUpdateAvailable = compareVersions(latestVersion, currentVersion) > 0;
      
      if (isUpdateAvailable) {
        setUpdateInfo({
          available: true,
          latestVersion,
          currentVersion,
          releaseUrl: release.html_url,
          releaseName: release.name || `v${latestVersion}`
        });
        setShowDialog(true);
      } else {
        console.log('App is up to date:', currentVersion);
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
    } finally {
      setChecking(false);
    }
  }

  // Simple semantic version comparison (e.g., "1.0.28" > "1.0.27")
  function compareVersions(v1: string, v2: string): number {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);
    
    for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
      const p1 = parts1[i] || 0;
      const p2 = parts2[i] || 0;
      if (p1 > p2) return 1;
      if (p1 < p2) return -1;
    }
    return 0;
  }

  function openDownloadPage() {
    if (updateInfo?.releaseUrl) {
      // Open in default browser using window.open
      window.open(updateInfo.releaseUrl, '_blank');
    }
  }

  if (!showDialog || !updateInfo) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Update Available! 🎉
        </h2>
        
        <div className="mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-2">
            A new version of FrameTrain is available!
          </p>
          
          <div className="bg-gray-100 dark:bg-gray-700 rounded p-4 my-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Current Version:</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">
                v{updateInfo.currentVersion}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Latest Version:</span>
              <span className="font-mono font-semibold text-green-600 dark:text-green-400">
                v{updateInfo.latestVersion}
              </span>
            </div>
          </div>
          
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            Click "View Release" to download the latest installer from GitHub.
          </p>
          
          <div className="text-xs text-gray-500 dark:text-gray-500 bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
            💡 Download the installer for your platform:
            <br />• macOS: .dmg file
            <br />• Windows: .exe or .msi file
            <br />• Linux: .AppImage or .deb file
          </div>
        </div>
        
        <div className="flex gap-3">
          <button
            onClick={openDownloadPage}
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition"
          >
            View Release
          </button>
          
          <button
            onClick={() => setShowDialog(false)}
            className="flex-1 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 font-semibold py-2 px-4 rounded transition"
          >
            Later
          </button>
        </div>
        
        {checking && (
          <div className="mt-3 text-center text-sm text-gray-500">
            Checking for updates...
          </div>
        )}
      </div>
    </div>
  );
}
