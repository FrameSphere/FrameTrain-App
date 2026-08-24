// Plugin-eigene Trainingsparameter (z. B. imgsz/augment/patience bei YOLO).
//
// Eigener Hook, weil die naheliegende Fassung einen subtilen Fehler hatte:
// detectPlugin() liefert bei jedem Render ein neues Objekt. Haengt das
// Zuruecksetzen an diesem Objekt, verwirft jeder Render die Eingaben — die
// Felder sprangen sofort auf die Vorgabe zurueck und das Training bekam nie
// die eingestellten Werte. Massgeblich ist deshalb allein die Plugin-ID.

import { useEffect, useMemo, useRef, useState } from 'react';

export type PluginParamValue = number | boolean | string;

/** Steht schon im allgemeinen Teil des Formulars — hier nicht doppelt anbieten. */
export const PLUGIN_PARAM_BLOCKLIST = ['task_type', 'epochs', 'batch', 'batch_size', 'device'];

export function pluginParamDefaultsFrom(
  raw: Record<string, unknown> | undefined,
): Record<string, PluginParamValue> {
  return Object.fromEntries(
    Object.entries(raw ?? {}).filter(([k, v]) =>
      !PLUGIN_PARAM_BLOCKLIST.includes(k) &&
      (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string'),
    ),
  ) as Record<string, PluginParamValue>;
}

export function usePluginParams(
  pluginId: string | null,
  defaultConfig: Record<string, unknown> | undefined,
) {
  const defaults = useMemo(
    () => pluginParamDefaultsFrom(defaultConfig),
    // Absicht: nur die Plugin-ID entscheidet, nicht die Objekt-Identitaet.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pluginId],
  );
  const [params, setParams] = useState<Record<string, PluginParamValue>>(defaults);
  const appliedFor = useRef<string | null>(pluginId);

  useEffect(() => {
    if (appliedFor.current === pluginId) return;
    appliedFor.current = pluginId;
    setParams(defaults);
  }, [pluginId, defaults]);

  return { params, setParams, defaults };
}
