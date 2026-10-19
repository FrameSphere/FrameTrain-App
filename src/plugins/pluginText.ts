// Plugin-Namen und -Beschreibungen stehen deutsch im Plugin selbst; die
// englische Fassung liegt unter plugins.<id> in en.json. Fehlt dort ein
// Eintrag, bleibt der Text aus dem Plugin stehen.

type Translate = (key: string, paramsOrFallback?: string | Record<string, string | number>) => string;

interface PluginLike { id: string; name: string; description: string }

export function pluginName(p: PluginLike, t: Translate): string {
  return t(`plugins.${p.id}.name`, p.name);
}

export function pluginDescription(p: PluginLike, t: Translate): string {
  return t(`plugins.${p.id}.description`, p.description);
}
