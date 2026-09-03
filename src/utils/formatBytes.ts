/**
 * Bytes in eine lesbare Groesse. Bis 1 GB ohne Nachkommastelle — "812 MB" liest
 * sich schneller als "812.4 MB"; darueber eine Stelle, weil "12.4 GB" und
 * "12 GB" spuerbar unterschiedliche Mengen sind.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / Math.pow(1024, i);
  const digits = i >= 3 ? 1 : 0;
  return `${value.toFixed(digits)} ${units[i]}`;
}
