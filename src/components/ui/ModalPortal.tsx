// ModalPortal — rendert Vollbild-Overlays direkt in document.body.
//
// Die Seiten laufen im scrollenden <main className="overflow-auto"> des
// Dashboards. Ein `fixed inset-0`-Overlay mit backdrop-blur, das darin steckt,
// deckt in WKWebView (macOS) nicht zuverlaessig das ganze Fenster ab: ist die
// Seite gescrollt, bleibt oben ein Streifen unabgedunkelt. Ausserhalb des
// Scroll-Containers tritt das nicht auf — deshalb haengt jedes Seiten-Modal
// ueber dieses Portal am body, genau wie die App-weiten Overlays.

import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';

export default function ModalPortal({ children }: { children: ReactNode }) {
  return createPortal(children, document.body);
}
